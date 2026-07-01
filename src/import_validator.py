"""
Piece 2 — Upload validation for user-uploaded battery cycle data.

validate_upload(df) takes a raw pandas DataFrame (as parsed from a CSV upload)
and returns a structured result dict. It never raises — all issues are
collected into errors (blocking) or warnings (non-blocking).

Validation order matters: later checks assume earlier structural checks passed,
so the function short-circuits after structural errors.
"""

import pandas as pd
import numpy as np

REQUIRED_COLUMNS  = ["cell_id", "cycle_number", "capacity_ah", "resistance_ohm"]
OPTIONAL_COLUMNS  = ["temperature_c", "test_date", "notes"]

CAPACITY_MIN, CAPACITY_MAX     = 0.01, 500.0   # Ah
RESISTANCE_MIN, RESISTANCE_MAX = 0.001, 10.0   # Ω
RESISTANCE_MOHM_THRESHOLD      = 100.0          # values above this look like mΩ


def validate_upload(df: pd.DataFrame) -> dict:
    """
    Validate a user-uploaded cycle DataFrame against the import schema.

    Returns:
        {
          "valid": bool,
          "errors":   [str, ...],   # blocking — must fix before analysis
          "warnings": [str, ...],   # non-blocking — user must acknowledge
          "summary": {
              "n_cells":          int,
              "cycles_per_cell":  {cell_id: int, ...},
              "has_temperature":  bool,
              "has_dates":        bool,
              "missing_optional": [str, ...],
          }
        }
    """
    errors   = []
    warnings = []

    # ── 1. Required columns present ──────────────────────────────────────────
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(
                f"Missing required column: '{col}'. "
                f"Add this column to your CSV before uploading."
            )

    # If any required column is absent we cannot safely run further checks.
    if errors:
        return _result(errors, warnings, df, short_circuit=True)

    # ── 2. Data types ─────────────────────────────────────────────────────────
    # cycle_number must be integer-castable
    bad_cycle_rows = _non_castable_rows(df, "cycle_number", int)
    if bad_cycle_rows:
        errors.append(
            f"'cycle_number' has non-integer values in "
            f"{len(bad_cycle_rows)} row(s): rows {_fmt_rows(bad_cycle_rows)}. "
            f"cycle_number must be a whole number (e.g. 1, 2, 3)."
        )

    # capacity_ah must be float-castable
    bad_cap_rows = _non_castable_rows(df, "capacity_ah", float)
    if bad_cap_rows:
        errors.append(
            f"'capacity_ah' has non-numeric values in "
            f"{len(bad_cap_rows)} row(s): rows {_fmt_rows(bad_cap_rows)}. "
            f"capacity_ah must be a number (e.g. 2.031)."
        )

    # resistance_ohm must be float-castable
    bad_res_rows = _non_castable_rows(df, "resistance_ohm", float)
    if bad_res_rows:
        errors.append(
            f"'resistance_ohm' has non-numeric values in "
            f"{len(bad_res_rows)} row(s): rows {_fmt_rows(bad_res_rows)}. "
            f"resistance_ohm must be a number (e.g. 0.052)."
        )

    if errors:
        return _result(errors, warnings, df, short_circuit=True)

    # Cast to correct types now that we know they are safe
    df = df.copy()
    df["cycle_number"]   = pd.to_numeric(df["cycle_number"],   errors="coerce").astype("Int64")
    df["capacity_ah"]    = pd.to_numeric(df["capacity_ah"],    errors="coerce").astype(float)
    df["resistance_ohm"] = pd.to_numeric(df["resistance_ohm"], errors="coerce").astype(float)

    # ── 3. Null values in required columns ────────────────────────────────────
    for col in REQUIRED_COLUMNS:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            errors.append(
                f"'{col}' has {n_null} missing value(s). "
                f"All required columns must be filled for every row."
            )

    if errors:
        return _result(errors, warnings, df, short_circuit=True)

    # ── 4. Minimum cell count ─────────────────────────────────────────────────
    cell_ids = df["cell_id"].astype(str).str.strip()
    df["cell_id"] = cell_ids
    unique_cells = cell_ids.unique().tolist()
    n_cells = len(unique_cells)

    if n_cells < 2:
        errors.append(
            f"Only {n_cells} unique cell_id found: {unique_cells}. "
            f"Leave-cell-out validation requires at least 2 cells — the model "
            f"is tested by training on all cells except one; with only 1 cell "
            f"this test cannot be run, so there is no way to know whether the "
            f"model's predictions would generalise to an unseen cell."
        )

    if errors:
        return _result(errors, warnings, df, short_circuit=True)

    # ── 5. Duplicate cycle numbers within a cell ──────────────────────────────
    dup_cells = []
    for cid in unique_cells:
        cell_df = df[df["cell_id"] == cid]
        if cell_df["cycle_number"].duplicated().any():
            dup_cells.append(cid)
    if dup_cells:
        errors.append(
            f"Duplicate cycle_number values found within cell(s): "
            f"{', '.join(dup_cells)}. "
            f"Each row within a cell must have a unique cycle_number."
        )

    # ── 6. Capacity plausible range ───────────────────────────────────────────
    cap_series = df["capacity_ah"].dropna()
    out_of_range_cap = cap_series[
        (cap_series < CAPACITY_MIN) | (cap_series > CAPACITY_MAX)
    ]
    if not out_of_range_cap.empty:
        mn, mx = out_of_range_cap.min(), out_of_range_cap.max()
        errors.append(
            f"'capacity_ah' has {len(out_of_range_cap)} value(s) outside the "
            f"plausible range {CAPACITY_MIN}–{CAPACITY_MAX} Ah "
            f"(found: min={mn:.4g}, max={mx:.4g}). "
            f"Common cause: data is in mAh rather than Ah — divide all "
            f"capacity values by 1000 to convert."
        )

    # ── 7. Resistance plausible range ─────────────────────────────────────────
    res_series = df["resistance_ohm"].dropna()
    out_of_range_res = res_series[
        (res_series < RESISTANCE_MIN) | (res_series > RESISTANCE_MAX)
    ]
    if not out_of_range_res.empty:
        mn, mx = out_of_range_res.min(), out_of_range_res.max()
        errors.append(
            f"'resistance_ohm' has {len(out_of_range_res)} value(s) outside "
            f"the plausible range {RESISTANCE_MIN}–{RESISTANCE_MAX} Ω "
            f"(found: min={mn:.4g}, max={mx:.4g}). "
            f"Common cause: data is in mΩ rather than Ω — divide all "
            f"resistance values by 1000 to convert."
        )

    # ── Warnings ──────────────────────────────────────────────────────────────

    # W1. Fewer than 3 cells
    if n_cells == 2:
        warnings.append(
            f"Only 2 cells uploaded — this is the minimum for leave-cell-out "
            f"validation. With 2 cells each training fold contains only 1 cell, "
            f"which gives the model very little signal to learn from. "
            f"3 or more cells are recommended for reliable per-cell reliability "
            f"scores. Results will appear but treat RUL predictions as "
            f"directional only."
        )

    # W2. Cells with fewer than 100 cycles
    for cid in unique_cells:
        n_cy = int((df["cell_id"] == cid).sum())
        if n_cy < 100:
            warnings.append(
                f"'{cid}' has {n_cy} cycle(s) — fewer than the 100-cycle "
                f"recommendation. Reliability flags may show 'Calibrating' "
                f"for this cell because fade-rate and resistance-trend features "
                f"need sufficient history to stabilise."
            )

    # W3. Missing temperature column
    if "temperature_c" not in df.columns:
        warnings.append(
            f"Column 'temperature_c' not found. The platform will substitute "
            f"25°C for all cycles and label every temperature-dependent output "
            f"'Assumed 25°C — not measured' so the assumption is visible, "
            f"not hidden."
        )

    # W4. Resistance values that look like mΩ
    if not res_series.empty and (res_series > RESISTANCE_MOHM_THRESHOLD).all():
        warnings.append(
            f"All 'resistance_ohm' values are above {RESISTANCE_MOHM_THRESHOLD} Ω "
            f"(min found: {res_series.min():.1f}). This strongly suggests the "
            f"data is in milliohms (mΩ) rather than ohms (Ω). "
            f"Divide all resistance values by 1000 before uploading if so."
        )

    return _result(errors, warnings, df)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _non_castable_rows(df: pd.DataFrame, col: str, dtype: type) -> list[int]:
    """Return 1-indexed row numbers where col cannot be cast to dtype."""
    bad = []
    for i, val in enumerate(df[col], start=1):
        if pd.isna(val):
            continue   # nulls caught separately
        try:
            dtype(val)
        except (ValueError, TypeError):
            bad.append(i)
    return bad


def _fmt_rows(rows: list[int]) -> str:
    if len(rows) <= 5:
        return ", ".join(str(r) for r in rows)
    return ", ".join(str(r) for r in rows[:5]) + f" … ({len(rows)} total)"


def _result(
    errors: list,
    warnings: list,
    df: pd.DataFrame,
    short_circuit: bool = False,
) -> dict:
    """Build the structured result dict."""
    summary = {"n_cells": 0, "cycles_per_cell": {}, "has_temperature": False,
               "has_dates": False, "missing_optional": []}

    if not short_circuit and "cell_id" in df.columns:
        cell_ids = df["cell_id"].astype(str).str.strip()
        summary["n_cells"] = int(cell_ids.nunique())
        summary["cycles_per_cell"] = {
            cid: int((cell_ids == cid).sum())
            for cid in cell_ids.unique()
        }
        summary["has_temperature"] = (
            "temperature_c" in df.columns
            and df["temperature_c"].notna().any()
        )
        summary["has_dates"] = (
            "test_date" in df.columns
            and df["test_date"].notna().any()
        )
        summary["missing_optional"] = [
            c for c in OPTIONAL_COLUMNS if c not in df.columns
        ]

    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
        "summary":  summary,
    }
