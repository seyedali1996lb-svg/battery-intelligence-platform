"""
Robustness metrics on degraded input — what happens to the model when the
data is worse than a laboratory's.

Why this exists
---------------
Every number the platform publishes is computed on clean laboratory
cycling records: one row per cycle, complete capacity/resistance
measurements, no counter gaps. A real fleet is not like that. BMS
firmware resets lose cycle counts, telemetry drops cycles, and
electrical transients corrupt individual readings. "How accurate is the
model?" is therefore incomplete without "and for how long does it stay
accurate as the data degrades?"

`run_robustness_lco()` answers it mechanically: it applies each
degradation mode at increasing severity to a COPY of the raw cycles,
rebuilds features from the degraded data (so the degradation is real,
not simulated at feature level), and re-runs the identical leave-cell-out
harness. Each scenario reports the metric DELTA against the clean
baseline, and the level at which the model's SOH R² degrades past a
pre-declared floor is surfaced as the failure threshold.

Degradation modes (each models a real failure mode):

  - missing_cycles: drop a random fraction of cycle rows. Models general
    telemetry loss — the record keeps its cycle numbers but has holes.
  - bms_reset: remove a contiguous block of cycles. Models a BMS
    firmware reset / counter gap — a whole stretch of life vanishes
    between two surviving cycles.
  - current_spike: multiply resistance_ohm at a random set of cycles by
    a large factor. Models electrical transients / sensor artefacts
    surviving into the record.

Honest scope: degradations are applied with a fixed seed (reproducible),
the clean baseline is recomputed in the same run (never imported from an
older registry row), and the RUL metrics keep the v12 observed-EOL rules
— a degraded record can lose its EOL rows entirely, which shows up as
reduced label coverage, itself a robustness finding.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from batlab.validation.lco import run_lco, unwrap_cell_data

# SOH R² floor for the failure threshold: below this the per-chemistry
# model is no longer meaningfully better than a fleet-mean predictor on
# this fleet (matches the accuracy story's honesty bar).
SOH_R2_FAILURE_FLOOR = 0.5

# Scenario grid: (mode, severity, label). Severities are chosen so the
# mildest is a realistic bad day and the harshest is data abuse.
SCENARIOS = [
    ("missing_cycles", 0.05, "5% cycles lost"),
    ("missing_cycles", 0.15, "15% cycles lost"),
    ("missing_cycles", 0.30, "30% cycles lost"),
    ("bms_reset", 1, "1 counter gap"),
    ("bms_reset", 3, "3 counter gaps"),
    ("bms_reset", 7, "7 counter gaps"),
    ("current_spike", 3, "3 spike cycles"),
    ("current_spike", 10, "10 spike cycles"),
    ("current_spike", 30, "30 spike cycles"),
]

SPIKE_MULTIPLIER = 1.5  # transient resistance artefact
RESET_BLOCK_FRACTION = 0.05  # each counter gap eats ~5% of the record


def degrade_cycles(df: pd.DataFrame, mode: str, severity, seed: int = 42) -> pd.DataFrame:
    """Return a degraded COPY of one cell's raw cycle DataFrame.

    Deterministic for a given seed. Never mutates the input. Modes and
    severities are documented on SCENARIOS; an unknown mode raises — a
    silently-passed-through clean frame would fake a robustness result.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    # Stable per-scenario seed offset: builtins.hash is randomized per
    # process (PYTHONHASHSEED), so it would make "deterministic" draws
    # unreproducible across runs. CRC32 of the mode string is stable.
    scenario_offset = zlib.crc32(f"{mode}:{severity}".encode()) % 1000
    if mode == "missing_cycles":
        n_drop = int(round(n * severity))
        if n_drop > 0:
            drop_idx = rng.choice(n, size=n_drop, replace=False)
            out = out.drop(out.index[drop_idx]).sort_values("cycle_number").reset_index(drop=True)
        return out
    if mode == "bms_reset":
        # Remove `severity` contiguous blocks of ~RESET_BLOCK_FRACTION each,
        # spaced through the record (not the head — a cell with no early
        # data is a different failure mode).
        block = max(2, int(n * RESET_BLOCK_FRACTION))
        drop_idx = set()
        for _ in range(int(severity)):
            if n - len(drop_idx) <= block * 2:
                break
            start = int(rng.integers(block, n - block))
            drop_idx.update(range(start, min(start + block, n)))
        if drop_idx:
            out = out.drop(index=out.index[list(drop_idx)]).sort_values("cycle_number").reset_index(drop=True)
        return out
    if mode == "current_spike":
        if "resistance_ohm" in out.columns:
            n_spike = min(int(severity), n)
            spike_idx = rng.choice(n, size=n_spike, replace=False)
            out.iloc[spike_idx, out.columns.get_loc("resistance_ohm")] = (
                out["resistance_ohm"].iloc[spike_idx] * SPIKE_MULTIPLIER
            )
        return out
    raise ValueError(f"unknown degradation mode: {mode}")


def run_robustness_lco(cell_data: dict, seed: int = 42, scenarios=None) -> dict:
    """Benchmark the production GBRT's accuracy under degraded input.

    Runs the identical leave-cell-out harness on the clean data (the
    baseline) and on each degraded copy, reporting metric deltas. The
    same seed drives every fold's GBRT and every degradation draw, so
    scenario deltas isolate the DATA effect, not refit noise.

    Returns:
        {
          "baseline":  {soh_r2, rul_r2, rul_label_coverage},
          "scenarios": [{mode, severity, severity_label, soh_r2,
                         soh_r2_delta, rul_r2, rul_r2_delta,
                         rul_label_coverage, n_cells_evaluated}],
          "failure_threshold": {mode: first severity label where
                         soh_r2 < SOH_R2_FAILURE_FLOOR, else None},
        }
    """
    cell_data = unwrap_cell_data(cell_data)
    base = run_lco(cell_data, seed=seed)
    baseline = {
        "soh_r2": base.get("soh_r2"),
        "rul_r2": base.get("rul_r2"),
        "rul_label_coverage": base.get("rul_label_coverage"),
    }

    rows = []
    thresholds: dict = {}
    for mode, severity, label in (scenarios or SCENARIOS):
        degraded = {
            cid: degrade_cycles(df, mode, severity, seed=seed + zlib.crc32(f"{mode}:{severity}".encode()) % 1000)
            for cid, df in (cell_data or {}).items()
        }
        # A degradation can empty a cell entirely (all rows dropped) —
        # run_lco needs >= 2 cells; skip only scenarios that destroy the
        # fleet, and record the survivors.
        degraded = {cid: df for cid, df in degraded.items() if len(df) >= 4}
        if len(degraded) < 2:
            rows.append({
                "mode": mode, "severity": severity, "severity_label": label,
                "soh_r2": None, "soh_r2_delta": None,
                "rul_r2": None, "rul_r2_delta": None,
                "rul_label_coverage": None,
                "n_cells_evaluated": len(degraded),
            })
            continue
        res = run_lco(degraded, seed=seed)
        soh_r2 = res.get("soh_r2")
        rul_r2 = res.get("rul_r2")
        soh_delta = (
            float(soh_r2) - float(baseline["soh_r2"])
            if (soh_r2 is not None and baseline["soh_r2"] is not None
                and not (isinstance(soh_r2, float) and np.isnan(soh_r2)))
            else None
        )
        rul_delta = (
            float(rul_r2) - float(baseline["rul_r2"])
            if (rul_r2 is not None and baseline["rul_r2"] is not None
                and not (isinstance(rul_r2, float) and np.isnan(rul_r2))
                and not (isinstance(baseline["rul_r2"], float) and np.isnan(baseline["rul_r2"])))
            else None
        )
        rows.append({
            "mode": mode, "severity": severity, "severity_label": label,
            "soh_r2": soh_r2, "soh_r2_delta": soh_delta,
            "rul_r2": rul_r2, "rul_r2_delta": rul_delta,
            "rul_label_coverage": res.get("rul_label_coverage"),
            "n_cells_evaluated": len(degraded),
        })
        # First severity at which SOH R² breaks the floor (per mode).
        if (
            thresholds.get(mode) is None
            and soh_r2 is not None
            and not (isinstance(soh_r2, float) and np.isnan(soh_r2))
            and float(soh_r2) < SOH_R2_FAILURE_FLOOR
        ):
            thresholds[mode] = label

    for mode, _sev, _label in (scenarios or SCENARIOS):
        thresholds.setdefault(mode, None)

    return {
        "baseline": baseline,
        "scenarios": rows,
        "failure_threshold": thresholds,
        "soh_r2_failure_floor": SOH_R2_FAILURE_FLOOR,
    }
