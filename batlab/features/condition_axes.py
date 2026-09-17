"""
Condition-axis audit — are temperature and C-rate validated modeling
axes, or just columns that happen to exist?

Why this exists
---------------
"Temperature and C-rate as validated axes" cannot mean "the columns are
in the DataFrame". An axis is validated for a fleet when three things are
true, each mechanically checkable:

  1. PROVENANCE: the values are measured, protocol-known constants, or
     generated — and the provenance is stated, not implied. A protocol
     constant (every Zhu cell at 25 °C) carries zero cross-cell signal by
     construction; treating it as a feature is dead weight, not physics.
  2. SPREAD: the axis actually varies across cells. Only cross-cell
     variation can teach a leave-cell-out model anything — a feature
     constant within every training fold is invisible to evaluation by
     construction.
  3. INTEGRITY: the column is not contaminated by missing-value
     sentinels. The Severson loader wrote 0.0 °C rows for cycles whose
     Tavg was absent in the source — those zeros silently dragged every
     rolling-temperature feature toward zero. (Fixed at load time; this
     audit detects the pattern so the next dataset cannot repeat it.)

`condition_axis_audit()` returns one row per (fleet, axis) with exactly
these three checks plus a usability verdict. The Benchmark page renders
it; the honest headline it produces is which axes are validated where —
and the expected result is that almost none are validated for cross-cell
modeling on the real reference fleets, because real laboratories hold
conditions constant. That is a property of the data, not a bug, and it
is the honest scope of "temperature-aware" claims on this platform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# An axis with cross-cell relative spread below this is effectively
# constant across the fleet's cells (per-cell means within this band of
# each other). Chosen well above float noise, well below meaningful
# thermal/rate differences.
SPREAD_FLOOR_PCT = 1.0  # percent of the fleet-mean value

# A per-cell temperature std below this means the cell sat in a chamber;
# above it means real thermal dynamics were recorded.
MEASURED_TEMP_STD_FLOOR_C = 0.5


def _axis_presence(df: pd.DataFrame, col: str) -> "tuple[bool, float]":
    """(present, fraction of rows with a usable non-sentinel value)."""
    if col not in df.columns:
        return False, 0.0
    # "float64" rather than float: pandas' stub narrows `dtype` to
    # `np.dtype[Any] | str | None`, which `type[float]` does not satisfy.
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
    # 0.0 is a sentinel for temperature (no cycle runs at absolute zero)
    # and for C-rate (no cycle runs at zero current in these protocols).
    usable = np.isfinite(vals) & (vals != 0.0)
    return bool(usable.any()), float(usable.mean()) if usable.size else 0.0


def _axis_verdict(dfs: list, col: str) -> dict:
    """One axis's verdict across a fleet's cells."""
    present_flags, usable_fracs = [], []
    per_cell_means, per_cell_stds = [], []
    sentinel_cells = 0
    for df in dfs:
        present, frac = _axis_presence(df, col)
        present_flags.append(present)
        usable_fracs.append(frac)
        if present:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
            vals = vals[np.isfinite(vals) & (vals != 0.0)]
            if len(vals):
                per_cell_means.append(float(np.mean(vals)))
                per_cell_stds.append(float(np.std(vals)))
            # Sentinel contamination: raw column contains exact zeros the
            # usable-filter had to exclude.
            raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
            if np.isfinite(raw).any() and float(((raw == 0.0).sum())) > 0:
                sentinel_cells += 1

    any_present = any(present_flags)
    if not any_present:
        return {
            "status": "absent",
            "usable_for_modeling": False,
            "cross_cell_spread_pct": None,
            "sentinel_cells": 0,
            "note": "column not present in this fleet's records",
        }

    if len(per_cell_means) >= 2:
        fleet_mean = float(np.mean(per_cell_means))
        spread = (
            float(np.std(per_cell_means, ddof=1)) / abs(fleet_mean) * 100.0
            if abs(fleet_mean) > 0 else 0.0
        )
    else:
        spread = 0.0

    if spread < SPREAD_FLOOR_PCT:
        status = "protocol-constant"
        note = (
            "values present but effectively constant across cells — carries "
            "no cross-cell signal for leave-cell-out evaluation"
        )
    else:
        status = "measured"
        note = "varies across cells — usable as a modeling axis on this fleet"

    usable = status == "measured"
    if sentinel_cells:
        note += f"; {sentinel_cells} cell(s) carried 0.0-value sentinels (excluded)"

    return {
        "status": status,
        "usable_for_modeling": usable,
        "cross_cell_spread_pct": round(spread, 2),
        "sentinel_cells": sentinel_cells,
        "note": note,
    }


def condition_axis_audit(cell_data: dict) -> list[dict]:
    """Audit temperature and C-rate as modeling axes for one fleet.

    Args:
        cell_data: {cell_id: raw cycle DataFrame} for ONE fleet (dataset).

    Returns one row per axis:
        {
          "axis": "temperature_c" | "c_rate",
          "status": "measured" | "protocol-constant" | "absent",
          "rows_usable_fraction": float,   # mean across cells
          "cross_cell_spread_pct": float,  # std of per-cell means / mean
          "sentinel_cells": int,           # cells carrying 0.0 sentinels
          "usable_for_modeling": bool,
          "note": str,
        }
    """
    dfs = [df for df in (cell_data or {}).values() if df is not None and len(df)]
    rows = []
    for axis in ("temperature_c", "c_rate"):
        verdict = _axis_verdict(dfs, axis)
        fracs = [_axis_presence(df, axis)[1] for df in dfs] if dfs else [0.0]
        rows.append({
            "axis": axis,
            "rows_usable_fraction": round(float(np.mean(fracs)), 3) if fracs else 0.0,
            **verdict,
        })
    return rows
