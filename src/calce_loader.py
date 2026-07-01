"""
CALCE (Center for Advanced Life Cycle Engineering) battery dataset loader.

Download from: https://web.calce.umd.edu/batteries/data.htm
(CALCE Battery Research Group).
Place CSV/XLSX files in data/calce/

Column names vary across CALCE sub-datasets. Common variants include:
    Cycle_Index, Discharge_Capacity(Ah), Internal_Resistance(Ohm),
    Temperature (°C), etc.
This loader uses flexible case-insensitive matching to normalise them.
"""

import os
import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_col_name(col: str) -> str:
    """Lowercase, strip special chars to underscores for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


_CYCLE_CANDIDATES = [
    "cycle_index", "cycle_number", "cycle", "cyc", "cycle_no",
]
_CAPACITY_CANDIDATES = [
    "discharge_capacity_ah", "discharge_capacity", "capacity_ah",
    "cap_ah", "qd", "q_d", "discharge_cap", "capacity",
]
_RESISTANCE_CANDIDATES = [
    "internal_resistance_ohm", "internal_resistance", "resistance_ohm",
    "ir", "resistance", "dc_internal_resistance",
]
_TEMPERATURE_CANDIDATES = [
    "temperature_c", "temperature", "temp_c", "temp", "avg_temperature",
    "tavg", "average_temperature",
]


def _find_column(norm_cols: dict[str, str], candidates: list[str]) -> str | None:
    """Return the original column name that best matches any candidate."""
    for c in candidates:
        if c in norm_cols:
            return norm_cols[c]
    # Substring fallback: any normalised col that *contains* any candidate keyword
    keywords = [c.split("_")[0] for c in candidates]
    for norm, orig in norm_cols.items():
        for kw in keywords:
            if kw in norm:
                return orig
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_calce_file(path: str, cell_id: str | None = None) -> pd.DataFrame:
    """
    Load a single CALCE CSV or XLSX file and normalise column names.

    Parameters
    ----------
    path : str
        Absolute or relative path to the file.
    cell_id : str, optional
        Ignored (reserved for future use). Cell identity is typically the
        filename stem when loading a directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cycle_number, capacity_ah,
        resistance_ohm (may be NaN), temperature_c (may be NaN).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Build normalised → original name map
    norm_cols = {_normalise_col_name(c): c for c in df.columns}

    cycle_col = _find_column(norm_cols, _CYCLE_CANDIDATES)
    cap_col   = _find_column(norm_cols, _CAPACITY_CANDIDATES)
    res_col   = _find_column(norm_cols, _RESISTANCE_CANDIDATES)
    temp_col  = _find_column(norm_cols, _TEMPERATURE_CANDIDATES)

    if cycle_col is None or cap_col is None:
        raise ValueError(
            f"Could not identify required columns (cycle, capacity) in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "cycle_number":   pd.to_numeric(df[cycle_col], errors="coerce"),
        "capacity_ah":    pd.to_numeric(df[cap_col],   errors="coerce"),
        "resistance_ohm": pd.to_numeric(df[res_col],   errors="coerce") if res_col  else np.nan,
        "temperature_c":  pd.to_numeric(df[temp_col],  errors="coerce") if temp_col else np.nan,
    })

    out = out.dropna(subset=["cycle_number", "capacity_ah"])
    out["cycle_number"] = out["cycle_number"].astype(int)
    out = out.sort_values("cycle_number").reset_index(drop=True)

    return out


def load_calce_directory(directory: str) -> dict[str, pd.DataFrame]:
    """
    Load all CSV and XLSX files in `directory`.

    Parameters
    ----------
    directory : str
        Path to a directory containing CALCE data files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of filename stem -> normalised DataFrame.
    """
    cells: dict[str, pd.DataFrame] = {}

    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".csv", ".xlsx", ".xls"):
            continue
        stem = os.path.splitext(fname)[0]
        fpath = os.path.join(directory, fname)
        try:
            df = load_calce_file(fpath, cell_id=stem)
            cells[stem] = df
            print(f"  [calce] Loaded {len(df)} cycles from {fname}")
        except Exception as exc:
            print(f"  [calce] Warning: could not load {fname}: {exc}")

    return cells
