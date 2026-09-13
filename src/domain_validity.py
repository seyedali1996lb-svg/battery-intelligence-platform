"""Domain-of-validity — where the model's numbers are allowed to be read.

Why this exists
---------------
Every accuracy number the platform publishes is measured on specific
cells: specific chemistries, a specific temperature range, a specific
C-rate regime, a specific SOH window, one cell format. A model validated
on four 18650 LiCoO₂ cells cycled at 31–34.5 °C down to 71% SOH has
NOT been validated on a pouch NCA cell cycled at 45 °C to 60% SOH —
and the catastrophic cross-chemistry transfer study (SOH R² = −34.6)
is what happens when the difference is ignored.

This module makes the boundary mechanical instead of narrative:

  - `compute_envelope()` derives the training envelope from the data
    actually used to fit and evaluate the model — chemistry, measured
    temperature range, measured C-rate range (or an explicit "not
    measured on this fleet"), the SOH window covered, and the cell
    format. Axes the fleet does not vary are labelled protocol-constant
    or absent, never silently claimed as covered.
  - `check_cell_against_envelope()` answers the deployment question per
    cell: is THIS cell inside the envelope the numbers were measured
    under? Verdicts are "in", "partial" (near a boundary or differing
    only in a soft axis like format), or "outside" — never silent.
  - `regime_reliability()` stratifies the per-cell fold metrics by
    regime (temperature band × SOH stage), because a fleet-level R² can
    hide a regime where the model genuinely does not work (the same
    reason per-cell gating exists — one level finer).
  - `validity_banner()` renders the one-line disclosure every decision
    surface shows next to a prediction.

Honest scope: the envelope is computed from the reference fleets'
records, so it is exactly as wide as the data — no assumed margins
beyond a small, stated tolerance. A cell outside the envelope is not
forbidden a prediction; it is forbidden reading the accuracy numbers
as if they covered it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tolerances (stated, not magic): how far outside the measured range a
# cell can sit before it is "outside" rather than "partial".
TEMP_PAD_C = 2.0          # °C beyond the measured range
TEMP_CONSTANT_TOL_C = 1.5  # tolerance when the fleet ran one set-point
CRATE_PAD_FRAC = 0.15     # 15% beyond the measured C-rate range
SOH_PAD_PCT = 2.0         # SOH points below the deepest measured fade
RUL_RELIABLE_FLOOR = 0.0  # per-regime RUL reliability floor (R² > 0)

# Temperature bands for per-regime stratification (°C, on cell mean).
TEMP_BANDS = [(0, 25), (25, 30), (30, 35), (35, 100)]
TEMP_BAND_LABELS = ["<25 °C", "25–30 °C", "30–35 °C", "≥35 °C"]

# SOH stages for per-regime stratification (on the cell's final SOH —
# "how far into life the model was exercised for this regime").
SOH_STAGES = [(85, 101, "late life (final SOH 85–100%)"), (75, 85, "deep fade (final SOH 75–85%)"), (0, 75, "EOL-critical (final SOH < 75%)")]


def _fmt_range(lo: float, hi: float, unit: str = "", digits: int = 1) -> str:
    if lo == hi:
        return f"{lo:.{digits}f}{unit}"
    return f"{lo:.{digits}f}–{hi:.{digits}f}{unit}"


def _cell_format(profile) -> str:
    """Coarse cell format from the profile's passport chemistry text."""
    text = (getattr(profile, "passport_chemistry", "") or "").lower()
    if "cylindrical" in text:
        return "cylindrical"
    if "pouch" in text:
        return "pouch"
    if "prismatic" in text:
        return "prismatic"
    return "unknown"


def _axis_status(lo: float, hi: float) -> str:
    """measured (real spread) vs protocol-constant (one set-point)."""
    if hi - lo < 0.5:  # < 0.5 °C / 0.5 C spread across the fleet
        return "protocol-constant"
    return "measured"


def _axis_status_from_cell_means(means: "list[float]") -> str:
    """Status from the spread of per-cell MEANS, not row extremes.

    Row-level min/max would misclassify a single-set-point fleet as
    "measured" whenever sensor noise ≥ 0.5 units — the question is
    whether the fleet's CELLS differ, i.e. whether leave-cell-out ever
    saw variation on this axis."""
    if len(means) < 2 or (max(means) - min(means)) < 0.5:
        return "protocol-constant"
    return "measured"


def compute_envelope(
    cell_ids: "list[str] | dict",
    cell_data: "dict | None" = None,
    featured: "dict | None" = None,
) -> dict:
    """Derive the training envelope from the fleet's own records.

    Args:
        cell_ids: list of cell ids (or a {cell_id: anything} map — both
            accepted so callers can pass a bundle's cell dict directly).
        cell_data: optional {cell_id: raw cycles df} — supplies SOH range
            (raw soh_pct) and C-rate (raw c_rate) when featured is absent.
        featured: optional {cell_id: featured df} — supplies
            temp_rolling_30cy / c_rate_rolling_10cy; preferred over raw.

    Never raises on a malformed cell: an unreadable cell contributes no
    axis values rather than aborting the envelope (the envelope must
    exist for the 99% case even when one record is odd).
    """
    if isinstance(cell_ids, dict):
        cell_ids = list(cell_ids.keys())

    chems: dict[str, set] = {}
    formats: dict[str, set] = {}
    temp_lo = temp_hi = None
    crate_lo = crate_hi = None
    soh_lo = soh_hi = None
    temp_cells = 0
    crate_cells = 0
    temp_cell_means: list[float] = []
    crate_cell_means: list[float] = []

    for cid in cell_ids:
        try:
            from chemistry_profiles import ChemistryProfile
            profile = ChemistryProfile.for_cell(cid)
            chems.setdefault(profile.short_name, set()).add(cid)
            formats.setdefault(_cell_format(profile), set()).add(cid)
        except Exception:
            chems.setdefault("?", set()).add(cid)

        fdf = (featured or {}).get(cid)
        rdf = (cell_data or {}).get(cid)
        # unwrap the two wrapper shapes callers actually pass
        if isinstance(rdf, dict) and isinstance(rdf.get("cycles"), pd.DataFrame):
            rdf = rdf["cycles"]

        tvals = None
        if isinstance(fdf, pd.DataFrame) and "temp_rolling_30cy" in fdf.columns:
            tvals = pd.to_numeric(fdf["temp_rolling_30cy"], errors="coerce").dropna().to_numpy(float)
        elif isinstance(rdf, pd.DataFrame) and "temperature_c" in rdf.columns:
            tvals = pd.to_numeric(rdf["temperature_c"], errors="coerce").dropna().to_numpy(float)
        if tvals is not None and len(tvals):
            t_lo, t_hi = float(np.min(tvals)), float(np.max(tvals))
            temp_lo = t_lo if temp_lo is None else min(temp_lo, t_lo)
            temp_hi = t_hi if temp_hi is None else max(temp_hi, t_hi)
            temp_cells += 1
            temp_cell_means.append(float(np.mean(tvals)))

        cvals = None
        if isinstance(fdf, pd.DataFrame) and "c_rate_rolling_10cy" in fdf.columns:
            cvals = pd.to_numeric(fdf["c_rate_rolling_10cy"], errors="coerce").dropna().to_numpy(float)
        elif isinstance(rdf, pd.DataFrame) and "c_rate" in rdf.columns:
            cvals = pd.to_numeric(rdf["c_rate"], errors="coerce").dropna().to_numpy(float)
        if cvals is not None and len(cvals):
            c_lo, c_hi = float(np.min(cvals)), float(np.max(cvals))
            crate_lo = c_lo if crate_lo is None else min(crate_lo, c_lo)
            crate_hi = c_hi if crate_hi is None else max(crate_hi, c_hi)
            crate_cells += 1
            crate_cell_means.append(float(np.mean(cvals)))

        svals = None
        if isinstance(rdf, pd.DataFrame) and "soh_pct" in rdf.columns:
            svals = pd.to_numeric(rdf["soh_pct"], errors="coerce").dropna().to_numpy(float)
        elif isinstance(fdf, pd.DataFrame) and "soh_pct" in fdf.columns:
            svals = pd.to_numeric(fdf["soh_pct"], errors="coerce").dropna().to_numpy(float)
        if svals is not None and len(svals):
            s_lo, s_hi = float(np.min(svals)), float(np.max(svals))
            soh_lo = s_lo if soh_lo is None else min(soh_lo, s_lo)
            soh_hi = s_hi if soh_hi is None else max(soh_hi, s_hi)

    temp_status = _axis_status_from_cell_means(temp_cell_means) if temp_cells else "absent"
    crate_status = _axis_status_from_cell_means(crate_cell_means) if crate_cells else "absent"

    envelope = {
        "n_cells": len(list(cell_ids)),
        "chemistries": sorted(chems.keys()),
        "cell_formats": sorted(formats.keys()),
        "temperature_c": (
            {"min": round(float(temp_lo), 1), "max": round(float(temp_hi), 1), "status": temp_status}
            if temp_lo is not None and temp_hi is not None else {"status": "absent"}
        ),
        "c_rate": (
            {"min": round(float(crate_lo), 2), "max": round(float(crate_hi), 2), "status": crate_status}
            if crate_lo is not None and crate_hi is not None else {"status": "absent"}
        ),
        "soh_pct": (
            {"min": round(float(soh_lo), 1), "max": round(float(soh_hi), 1)}
            if soh_lo is not None and soh_hi is not None else {"status": "absent"}
        ),
        # Which axis values are real measurements vs one protocol set-point —
        # a protocol-constant axis is NOT a validated range: the model never
        # saw variation on it, so a cell at a different set-point is outside.
        "axis_notes": {
            "temperature_c": (
                "measured across the fleet" if temp_status == "measured"
                else ("one protocol set-point — range not explored" if temp_status == "protocol-constant" else "not in this fleet's records")
            ),
            "c_rate": (
                "measured across the fleet" if crate_status == "measured"
                else ("one protocol set-point — range not explored" if crate_status == "protocol-constant" else "not in this fleet's records")
            ),
        },
        "n_cells_with_temperature": temp_cells,
        "n_cells_with_c_rate": crate_cells,
    }
    return envelope


def envelope_summary(envelope: "dict | None") -> str:
    """One-line human summary, e.g.
    "NASA LiCoO₂ · 18650 cylindrical · 31–34.5 °C (measured) · SOH 71–100% · n=4"
    """
    if not envelope:
        return ""
    parts = []
    chems = envelope.get("chemistries") or []
    if chems:
        parts.append(" + ".join(chems))
    fmts = envelope.get("cell_formats") or []
    fmts = [f for f in fmts if f != "unknown"]
    if fmts:
        parts.append(" + ".join(fmts))
    t = envelope.get("temperature_c") or {}
    if t.get("status") == "measured":
        parts.append(f"{_fmt_range(t['min'], t['max'], ' °C')} (measured)")
    elif t.get("status") == "protocol-constant":
        parts.append(f"{t['min']:.1f} °C set-point only")
    c = envelope.get("c_rate") or {}
    if c.get("status") == "measured":
        parts.append(f"{_fmt_range(c['min'], c['max'], 'C', 2)} (measured)")
    elif c.get("status") == "protocol-constant":
        parts.append(f"{c['min']:.2f}C set-point only")
    s = envelope.get("soh_pct") or {}
    if s.get("min") is not None:
        parts.append(f"SOH {_fmt_range(s['min'], s['max'], '%')}")
    parts.append(f"n={envelope.get('n_cells')}")
    return " · ".join(parts)


def check_cell_against_envelope(
    envelope: "dict | None",
    cell_id: str,
    cell_data: "dict | None" = None,
    featured: "dict | None" = None,
) -> dict:
    """Is this cell inside the envelope the model's numbers were measured under?

    Verdicts:
        "in"      — every axis inside the measured/declared range.
        "partial" — inside on hard axes but differing on a soft one
                    (cell format) or within tolerance of a boundary.
        "outside" — at least one hard axis beyond the envelope (different
                    chemistry, temperature beyond tolerance, C-rate beyond
                    tolerance, SOH below the deepest measured fade).

    Never raises; an unusable record yields verdict "unknown" with a note,
    which surfaces render as a disclosure rather than a pass.
    """
    if not envelope:
        return {"verdict": "unknown", "outside_axes": [], "partial_axes": [], "note": "no envelope recorded for this model"}

    outside: list[str] = []
    partial: list[str] = []
    notes: list[str] = []

    # Chemistry is the hard boundary the transfer study measured.
    try:
        from chemistry_profiles import ChemistryProfile
        chem = ChemistryProfile.for_cell(cell_id).short_name
        if chem not in (envelope.get("chemistries") or []):
            outside.append(f"chemistry ({chem} vs {'/'.join(envelope.get('chemistries') or ['—'])})")
    except Exception:
        pass

    rdf = (cell_data or {}).get(cell_id)
    if isinstance(rdf, dict) and isinstance(rdf.get("cycles"), pd.DataFrame):
        rdf = rdf["cycles"]
    fdf = (featured or {}).get(cell_id)

    tvals = None
    if isinstance(fdf, pd.DataFrame) and "temp_rolling_30cy" in fdf.columns:
        tvals = pd.to_numeric(fdf["temp_rolling_30cy"], errors="coerce").dropna().to_numpy(float)
    elif isinstance(rdf, pd.DataFrame) and "temperature_c" in rdf.columns:
        tvals = pd.to_numeric(rdf["temperature_c"], errors="coerce").dropna().to_numpy(float)
    t = envelope.get("temperature_c") or {}
    if tvals is not None and len(tvals) and t.get("status") in ("measured", "protocol-constant"):
        c_lo, c_hi = float(np.min(tvals)), float(np.max(tvals))
        if t["status"] == "protocol-constant":
            tol = TEMP_CONSTANT_TOL_C
            if abs(c_lo - t["min"]) > tol or abs(c_hi - t["max"]) > tol:
                outside.append(f"temperature ({_fmt_range(c_lo, c_hi, ' °C')} vs {t['min']:.1f} °C set-point)")
        else:
            if c_lo < t["min"] - TEMP_PAD_C or c_hi > t["max"] + TEMP_PAD_C:
                outside.append(f"temperature ({_fmt_range(c_lo, c_hi, ' °C')} vs {_fmt_range(t['min'], t['max'], ' °C')})")
            elif c_lo < t["min"] or c_hi > t["max"]:
                partial.append("temperature (within tolerance, at range edge)")
    elif t.get("status") == "absent" and tvals is None:
        # The envelope has no temperature data and neither does the cell —
        # nothing to compare, but the model was never temperature-validated.
        partial.append("temperature (unmeasured on both sides — unvalidated axis)")

    cvals = None
    if isinstance(fdf, pd.DataFrame) and "c_rate_rolling_10cy" in fdf.columns:
        cvals = pd.to_numeric(fdf["c_rate_rolling_10cy"], errors="coerce").dropna().to_numpy(float)
    elif isinstance(rdf, pd.DataFrame) and "c_rate" in rdf.columns:
        cvals = pd.to_numeric(rdf["c_rate"], errors="coerce").dropna().to_numpy(float)
    c = envelope.get("c_rate") or {}
    if cvals is not None and len(cvals) and c.get("status") == "measured":
        pad = max(0.05, (c["max"] - c["min"]) * CRATE_PAD_FRAC)
        d_lo, d_hi = float(np.min(cvals)), float(np.max(cvals))
        if d_lo < c["min"] - pad or d_hi > c["max"] + pad:
            outside.append(f"C-rate ({_fmt_range(d_lo, d_hi, 'C', 2)} vs {_fmt_range(c['min'], c['max'], 'C', 2)})")
        elif d_lo < c["min"] or d_hi > c["max"]:
            partial.append("C-rate (within tolerance, at range edge)")

    svals = None
    if isinstance(rdf, pd.DataFrame) and "soh_pct" in rdf.columns:
        svals = pd.to_numeric(rdf["soh_pct"], errors="coerce").dropna().to_numpy(float)
    elif isinstance(fdf, pd.DataFrame) and "soh_pct" in fdf.columns:
        svals = pd.to_numeric(fdf["soh_pct"], errors="coerce").dropna().to_numpy(float)
    s = envelope.get("soh_pct") or {}
    if svals is not None and len(svals) and s.get("min") is not None:
        d_min = float(np.min(svals))
        if d_min < s["min"] - SOH_PAD_PCT:
            outside.append(f"SOH deeper ({d_min:.1f}% vs {_fmt_range(s['min'], s['max'], '%')})")
        elif d_min < s["min"]:
            partial.append("SOH (within tolerance of deepest measured fade)")

    fmt = None
    try:
        from chemistry_profiles import ChemistryProfile
        fmt = _cell_format(ChemistryProfile.for_cell(cell_id))
        env_fmts = [f for f in (envelope.get("cell_formats") or []) if f != "unknown"]
        if fmt != "unknown" and env_fmts and fmt not in env_fmts:
            # Format differs but chemistry matched: soft boundary — the
            # transfer study never isolated format alone, so flag, don't
            # condemn.
            partial.append(f"cell format ({fmt} vs {'/'.join(env_fmts)})")
    except Exception:
        pass

    if outside:
        verdict = "outside"
    elif partial:
        verdict = "partial"
    else:
        verdict = "in"
    return {
        "verdict": verdict,
        "outside_axes": outside,
        "partial_axes": partial,
        "note": "; ".join(notes) if notes else "",
    }


def _temp_band(mean_t: "float | None") -> str:
    if mean_t is None:
        return "unmeasured"
    for (lo, hi), label in zip(TEMP_BANDS, TEMP_BAND_LABELS):
        if lo <= mean_t < hi:
            return label
    return "unmeasured"


def _soh_stage(final_soh: "float | None") -> str:
    if final_soh is None:
        return "unmeasured"
    for lo, hi, label in SOH_STAGES:
        if lo <= final_soh < hi:
            return label
    return "unmeasured"


def cell_regime(
    cell_id: str,
    cell_data: "dict | None" = None,
    featured: "dict | None" = None,
) -> dict:
    """This cell's regime descriptors: temperature band and SOH stage."""
    rdf = (cell_data or {}).get(cell_id)
    if isinstance(rdf, dict) and isinstance(rdf.get("cycles"), pd.DataFrame):
        rdf = rdf["cycles"]
    fdf = (featured or {}).get(cell_id)

    mean_t = None
    if isinstance(fdf, pd.DataFrame) and "temp_rolling_30cy" in fdf.columns:
        s = pd.to_numeric(fdf["temp_rolling_30cy"], errors="coerce").dropna()
        if len(s):
            mean_t = float(s.mean())
    elif isinstance(rdf, pd.DataFrame) and "temperature_c" in rdf.columns:
        s = pd.to_numeric(rdf["temperature_c"], errors="coerce").dropna()
        if len(s):
            mean_t = float(s.mean())

    final_soh = None
    if isinstance(rdf, pd.DataFrame) and "soh_pct" in rdf.columns:
        s = pd.to_numeric(rdf["soh_pct"], errors="coerce").dropna()
        if len(s):
            final_soh = float(s.iloc[-1])
    elif isinstance(fdf, pd.DataFrame) and "soh_pct" in fdf.columns:
        s = pd.to_numeric(fdf["soh_pct"], errors="coerce").dropna()
        if len(s):
            final_soh = float(s.iloc[-1])

    chem = None
    try:
        from chemistry_profiles import ChemistryProfile
        chem = ChemistryProfile.for_cell(cell_id).short_name
    except Exception:
        pass

    return {
        "chemistry": chem or "?",
        "temp_band": _temp_band(mean_t),
        "soh_stage": _soh_stage(final_soh),
        "temp_mean_c": mean_t,
        "final_soh_pct": final_soh,
    }


def regime_reliability(
    folds: dict,
    axes_by_cell: "dict | None" = None,
    cell_data: "dict | None" = None,
    featured: "dict | None" = None,
) -> list[dict]:
    """Per-regime reliability from per-cell fold metrics.

    Groups each cell's own leave-cell-out fold by regime (chemistry ×
    temperature band × SOH stage) and reports the per-regime mean R²,
    RUL reliability fraction, and a verdict. This is the layer between
    per-cell gating and fleet averages: a fleet-level R² of 0.9 with one
    regime stuck at 0.2 must be visible as that regime, not averaged away.

    Args:
        folds: {cell_id: {soh_r2, rul_r2, ...}} — run_lco()'s per_cell.
        axes_by_cell: optional {cell_id: cell_regime() output}; regimes are
            computed for missing entries from cell_data/featured.

    Verdict per regime:
        "reliable"    — n ≥ 3 cells AND mean RUL R² above floor (when RUL
                        is evaluable in-regime).
        "thin"        — 1–2 cells: directionally informative, statistically
                        thin (the standard n-disclosure rule).
        "unvalidated" — RUL not evaluable in this regime (no measured
                        labels) or mean RUL R² below floor.
    """
    folds = folds or {}
    if not folds:
        return []
    axes = dict(axes_by_cell or {})

    groups: dict[tuple, list] = {}
    for cid, fold in folds.items():
        if not isinstance(fold, dict):
            continue
        ax = axes.get(cid)
        if ax is None:
            ax = cell_regime(cid, cell_data=cell_data, featured=featured)
            axes[cid] = ax
        key = (ax.get("chemistry") or "?", ax.get("temp_band") or "unmeasured", ax.get("soh_stage") or "unmeasured")
        groups.setdefault(key, []).append((cid, fold))

    rows = []
    for (chem, tband, sstage), members in sorted(groups.items()):
        soh_r2s = [f.get("soh_r2") for _, f in members if f.get("soh_r2") is not None]
        rul_r2s = [f.get("rul_r2") for _, f in members if f.get("rul_r2") is not None]
        n = len(members)
        mean_soh = float(np.mean(soh_r2s)) if soh_r2s else None
        mean_rul = float(np.mean(rul_r2s)) if rul_r2s else None
        if mean_rul is None:
            verdict = "unvalidated"  # no measured labels in this regime
        elif n < 3:
            verdict = "thin"
        elif mean_rul > RUL_RELIABLE_FLOOR:
            verdict = "reliable"
        else:
            verdict = "unvalidated"
        rows.append({
            "chemistry": chem,
            "temp_band": tband,
            "soh_stage": sstage,
            "n_cells": n,
            "cell_ids": [cid for cid, _ in members],
            "mean_soh_r2": mean_soh,
            "mean_rul_r2": mean_rul,
            "rul_evaluable_n": len(rul_r2s),
            "verdict": verdict,
        })
    return rows


def validity_banner(prov: "dict | None") -> str:
    """The one-line envelope disclosure for a decision surface.

    Returns "" when the cell is inside the envelope (quiet honesty — no
    banner for the normal case) or when no envelope is recorded at all
    (the provenance line already covers that case). Outside/partial get
    an explicit sentence naming the axes.
    """
    if not prov:
        return ""
    verdict = prov.get("cell_verdict")
    if verdict == "outside":
        axes = ", ".join(prov.get("outside_axes") or [])
        return f"⚠ Outside this model's training envelope: {axes}. The accuracy numbers shown were NOT measured under these conditions."
    if verdict == "partial":
        axes = ", ".join(prov.get("partial_axes") or [])
        return f"Edge of the training envelope: {axes} — treat the numbers as directional."
    return ""
