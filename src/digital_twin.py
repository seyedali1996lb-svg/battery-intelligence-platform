"""
Phase 3 — Digital Twin architecture (docs/history.md's "not started" item).

What this is
------------
The platform's Phase 3 goal is "a defined architecture connecting a cell's
measured history, its derived health indicators, and a physics-based
degradation model into a single continuously-updated representation."
This module is that architecture's concrete, minimal form:

    CellTwin  --holds-->  measured history (cycles as they arrive)
                  + derived health indicators (SOH, fade rate, knee, EOL)
                  + a physics projection (SEI-fade fit, same model as
                    src/pybamm_rul.py) re-fit on every update

The point is the *continuously-updated representation*, not new physics:
the SEI-fade model is the same one src/pybamm_rul.py already validates,
and the health indicators are the same ones the rest of the platform
computes. `CellTwin.update()` merges new cycles, re-derives indicators,
and re-fits the projection — so a consumer holding a twin always reads a
single self-consistent state rather than re-assembling three independent
computations and hoping they agree (the bug class this platform has hit
repeatedly with independently-computed verdicts).

What this is NOT
----------------
A live-synced digital twin in the Siemens/ABB sense (a continuously
re-parameterized physics model running against a real BMS feed). This is
the honest, bounded version the Live Monitor already labels "not a
live-synced digital twin": re-fit on each cycle batch that arrives, over
measured history the platform already holds. The same real-BMS-validation
trigger that gates the rest of the Lifecycle Intelligence layer gates a
deeper twin.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np

# ── Health indicator extractors (reused, not re-derived) ────────────────────

def _fade_rate_30cy(soh_pct: np.ndarray, capacity_ah: np.ndarray, cycles: np.ndarray) -> "float | None":
    """% capacity lost per cycle over the last 30 cycles (linear slope,
    normalized to initial capacity) — the platform's standard fade metric,
    computed here so the twin is independent of which feature columns the
    upstream DataFrame happened to carry."""
    if len(soh_pct) < 31:
        return None
    tail = slice(-30, None)
    cap0 = max(float(capacity_ah[tail][0]), 1e-9)
    slope = float(np.polyfit(cycles[tail], capacity_ah[tail], 1)[0])
    return slope / cap0 * 100.0


class CellTwin:
    """One cell's continuously-updated {history + indicators + projection}.

    Parameters
    ----------
    cell_id : str
    data_mode : str
        One of the platform's data-source keys (\"severson\", \"nasa\",
        \"synthetic\", \"uploaded\") — selects the physics parameter set.
    eol_threshold : float
        SOH % at which the cell counts as end-of-life (default 80).
    anchor_spm : bool
        Whether the first update runs the PyBaMM SPM single cycle to anchor
        nominal capacity (slow, ~seconds). False skips it and the projection
        is a pure SEI-fade fit — honest, just without the physics anchor
        (tests and offline batch use this).
    """

    def __init__(
        self,
        cell_id: str,
        data_mode: str,
        eol_threshold: float = 80.0,
        anchor_spm: bool = True,
    ) -> None:
        self.cell_id = cell_id
        self.data_mode = data_mode
        self.eol_threshold = eol_threshold
        self.anchor_spm = anchor_spm

        # Measured history: cycle_number -> row, merged in arrival order.
        self._cycles: dict[int, dict[str, float]] = {}
        # Cached physics anchor (nominal capacity from SPM); None until anchored.
        self._spm_capacity_ah: "float | None" = None
        self._param_set: "str | None" = None
        self._updated_at: "datetime.datetime | None" = None
        self._last_error: "str | None" = None

    # ── state ────────────────────────────────────────────────────────────────

    def update(self, df) -> "dict[str, Any]":
        """Merge new per-cycle rows into the twin and re-derive everything.

        ``df`` is a per-cycle DataFrame with at least ``cycle_number`` and
        ``capacity_ah`` (``soh_pct`` used when present, else derived from
        capacity / first-capacity). Idempotent per cycle number: feeding the
        same cycles twice is a no-op merge, not a double-count.

        Returns the same dict as :meth:`snapshot` (convenience for callers
        that want the state right after an update).
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            raise ValueError("update() needs a non-empty per-cycle DataFrame")

        for _, row in df.iterrows():
            cyc = int(row["cycle_number"])
            record: dict[str, float] = {}
            for k, v in row.items():
                if k == "cycle_number":
                    continue
                if isinstance(v, bool):
                    continue  # flags (is_eol etc.) aren't twin state
                if isinstance(v, (int, float)) and not np.isnan(float(v)):
                    record[k] = float(v)
                # strings / None / NaN silently skipped — the twin only
                # ingests numeric per-cycle measurements.
            self._cycles[cyc] = record

        self._recompute()
        return self.snapshot()

    def _recompute(self) -> None:
        self._updated_at = datetime.datetime.now(datetime.timezone.utc)
        try:
            cyc = sorted(self._cycles)
            # Only cycles that actually carry capacity participate in the
            # fit — a NaN/garbage cycle must not poison the history.
            cyc = [c for c in cyc if "capacity_ah" in self._cycles[c]]
            if not cyc:
                return
            rows = [self._cycles[c] for c in cyc]
            cycles = np.array(cyc, dtype=float)
            cap = np.array([r["capacity_ah"] for r in rows], dtype=float)
            first_cap = max(float(cap[0]), 1e-9)
            if "soh_pct" in rows[0]:
                soh = np.array([r.get("soh_pct", np.nan) for r in rows], dtype=float)
                if np.isnan(soh).any():
                    soh = cap / first_cap * 100.0
            else:
                soh = cap / first_cap * 100.0

            # Health indicators
            self._last_soh_pct = float(soh[-1])
            self._last_cycle = int(cycles[-1])
            self._fade_rate_30cy = _fade_rate_30cy(soh, cap, cycles)
            self._is_eol = self._last_soh_pct < self.eol_threshold
            from batlab.features.knee_detection import detect_knee
            import pandas as _pd
            self._knee = detect_knee(_pd.Series(soh), _pd.Series(cycles))

            # Physics projection — SEI-fade fit on measured history
            from pybamm_rul import _fit_sei_fade, _PARAM_MAP

            self._param_set = _PARAM_MAP.get(self.data_mode, "Marquis2019")
            if self.anchor_spm and self._spm_capacity_ah is None:
                # One-time SPM anchor (slow; happens on the first update only).
                try:
                    from pybamm_rul import _run_spm_single_cycle
                    self._spm_capacity_ah = _run_spm_single_cycle(self._param_set)
                except Exception as exc:  # noqa: BLE001 — anchor is optional
                    self._last_error = f"SPM anchor failed: {exc}"
                    self._spm_capacity_ah = None

            if len(cyc) >= 5:
                beta, beta_sigma = _fit_sei_fade(cycles, soh)
                self._beta = float(beta)
                self._beta_sigma = float(beta_sigma)
                self._projection = self._project(beta, beta_sigma, cycles, soh)
            else:
                self._beta = self._beta_sigma = None
                self._projection = None
        except Exception as exc:  # noqa: BLE001 — the twin must never crash a caller
            self._last_error = str(exc)

    def _project(
        self, beta: float, beta_sigma: float, cycles: np.ndarray, soh: np.ndarray
    ) -> "dict[str, Any]":
        """Project SOH forward with the sqrt-fade model (same as pybamm_rul),
        returning a bounded dict (projections sampled every 10 cycles)."""
        eol = self.eol_threshold
        n0 = cycles[0] if cycles[0] > 0 else 1.0
        cur = int(cycles[-1])
        future = np.arange(cur + 1, cur + 501)
        n_shift = future - n0 + 1.0
        beta_lo = max(1e-9, beta + 2 * beta_sigma)   # pessimistic
        beta_hi = max(1e-9, beta - 2 * beta_sigma)   # optimistic
        central = np.clip(1.0 - beta * np.sqrt(n_shift), 0.0, 1.0) * 100.0
        lo = np.clip(1.0 - beta_lo * np.sqrt(n_shift), 0.0, 1.0) * 100.0
        hi = np.clip(1.0 - beta_hi * np.sqrt(n_shift), 0.0, 1.0) * 100.0
        below = np.where(central < eol)[0]
        rul = int(future[below[0]]) - cur if len(below) else None

        sample = slice(None, None, 10)
        return {
            "rul_cycles_to_eol": rul,
            "eol_threshold": eol,
            "proj_cycles": future[sample].tolist(),
            "proj_soh_pct": np.round(central[sample], 2).tolist(),
            "proj_soh_lo_pct": np.round(lo[sample], 2).tolist(),
            "proj_soh_hi_pct": np.round(hi[sample], 2).tolist(),
        }

    # ── read ─────────────────────────────────────────────────────────────────

    def snapshot(self) -> "dict[str, Any]":
        """The twin's full current state — history summary, indicators,
        projection, and honest labels. Safe to JSON-serialize."""
        n_cycles = len(self._cycles)
        history_summary: "dict[str, Any]" = {"n_cycles": n_cycles}
        if n_cycles:
            first = self._cycles[sorted(self._cycles)[0]]
            last = self._cycles[sorted(self._cycles)[-1]]
            history_summary.update({
                "first_cycle": int(sorted(self._cycles)[0]),
                "last_cycle": int(sorted(self._cycles)[-1]),
                "first_soh_pct": round(float(first.get("soh_pct", 100.0)), 2),
                "last_soh_pct": round(float(last.get("soh_pct", 100.0)), 2),
            })

        indicators = {
            "soh_pct": getattr(self, "_last_soh_pct", None),
            "fade_rate_30cy": getattr(self, "_fade_rate_30cy", None),
            "is_eol": getattr(self, "_is_eol", None),
            "knee": {
                "detected": self._knee.get("detected", False),
                "cycle": self._knee.get("cycle"),
                "soh_at_knee": self._knee.get("soh_at_knee"),
                "phase": self._knee.get("phase"),
            } if getattr(self, "_knee", None) else None,
        }

        return {
            "cell_id": self.cell_id,
            "data_mode": self.data_mode,
            "updated_at": self._updated_at.isoformat() if self._updated_at else None,
            "history": history_summary,
            "indicators": indicators,
            "projection": {
                "physics_model": "SEI sqrt-fade (same as src/pybamm_rul.py)",
                "param_set": self._param_set,
                "spm_capacity_ah": self._spm_capacity_ah,
                "beta": getattr(self, "_beta", None),
                "beta_sigma": getattr(self, "_beta_sigma", None),
                **self._projection,
            } if getattr(self, "_projection", None) else None,
            "labels": [
                "projection, not prediction — physics-based forward extrapolation of measured fade",
                "continuously re-fit on each new cycle batch",
                "not a live-synced digital twin — no real BMS feed (see Live Monitor's Physics Twin Check)",
            ],
            "last_error": self._last_error,
        }


def twin_from_cell(
    cell_id: str,
    df,
    data_mode: str,
    eol_threshold: float = 80.0,
    anchor_spm: bool = True,
) -> CellTwin:
    """Build a fully-updated twin from a cell's complete history in one call
    (the offline/batch entry point; the streaming path calls update() per
    cycle batch instead)."""
    twin = CellTwin(cell_id, data_mode, eol_threshold=eol_threshold, anchor_spm=anchor_spm)
    twin.update(df)
    return twin
