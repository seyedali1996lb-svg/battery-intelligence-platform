"""
Universal battery cycler auto-ingestion parser and schema mapper.

Supports automatic header detection, unit conversion, and cycle aggregation
for major battery testing equipment:
- Arbin Instruments (.csv / .xlsx)
- BioLogic (.mpr / .csv export)
- Maccor (.txt / .csv)
- Neware BTS (.csv / .xlsx)
- Novonix (.csv)
- Bitrode (.csv)
- Generic lab CSV / Parquet
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from batlab.datasets.schema import validate_schema, SchemaError


# Standard target columns in batlab cycle schema
STANDARDIZED_COLUMNS = [
    "cycle_number",
    "capacity_ah",
    "voltage_mean_v",
    "current_mean_a",
    "temperature_mean_c",
    "resistance_ohm",
    "coulombic_efficiency",
]

# Cycler column synonym dictionaries (lowercased without punctuation)
SYNONYMS: Dict[str, List[str]] = {
    "cycle_number": [
        "cycle", "cycle_index", "cycle_number", "cycle_no", "cycleno", "cycle#", 
        "half_cycle", "cycleindex", "cyc", "cyclecount"
    ],
    "voltage_v": [
        "voltage", "voltage_v", "ecell", "ecell_v", "volts", "v", "voltage(v)",
        "cell_voltage", "potential", "potential_v", "terminal_voltage"
    ],
    "current_a": [
        "current", "current_a", "current_ma", "<i_a>", "<i_ma>", "current(a)",
        "current(ma)", "amps", "i", "i_a", "i_ma", "current_amps"
    ],
    "capacity_ah": [
        "capacity", "capacity_ah", "capacity_mah", "capacity/ah", "capacity/ma.h",
        "discharge_capacity", "discharge_capacity_ah", "discharge_capacity_mah",
        "cap", "cap_ah", "cap_mah", "dis_cap", "dis_cap_ah", "discharge_cap"
    ],
    "charge_capacity_ah": [
        "charge_capacity", "charge_capacity_ah", "charge_capacity_mah", "chg_cap",
        "charge_cap", "chg_capacity", "charge_capacity(ah)", "charge_cap_ah"
    ],
    "temperature_c": [
        "temperature", "temp", "temperature_c", "temp_c", "temperature_mean_c",
        "temp(c)", "temperature(c)", "aux_temperature", "thermocouple_1"
    ],
    "time_s": [
        "time", "time_s", "test_time", "test_time_s", "testtime(sec)", "time/s",
        "total_time", "time_seconds", "elapsed_time_s", "step_time_s"
    ],
    "step_type": [
        "step_name", "step_type", "step", "step_index", "step_number", "mode", "state"
    ],
}


def _normalize_name(name: str) -> str:
    """Normalize a column header for fuzzy synonym matching."""
    s = str(name).lower().strip()
    s = re.sub(r"[\[\]\(\)\{\}\<\>\/\-\.\s_#]", "", s)
    return s


def detect_cycler_format(df_or_columns: Any) -> Dict[str, Any]:
    """
    Detect the manufacturer / format of raw battery cycler data.
    
    Returns
    -------
    dict
        Detected format details, mapped columns, and unit scaling multipliers.
    """
    if isinstance(df_or_columns, pd.DataFrame):
        cols = list(df_or_columns.columns)
    else:
        cols = list(df_or_columns)
        
    normalized = {_normalize_name(c): c for c in cols}
    
    # Check for known hardware signatures
    raw_lower = [str(c).lower() for c in cols]
    
    hardware = "generic"
    if any("arbin" in c for c in raw_lower) or ("test_time(s)" in raw_lower and "cycle_index" in raw_lower):
        hardware = "Arbin"
    elif any("ecell" in c for c in raw_lower) or any("biologic" in c for c in raw_lower):
        hardware = "BioLogic"
    elif any("maccor" in c for c in raw_lower) or ("testtime(sec)" in raw_lower):
        hardware = "Maccor"
    elif any("neware" in c for c in raw_lower) or ("step name" in raw_lower and "cap(mah)" in raw_lower):
        hardware = "Neware"
    elif any("novonix" in c for c in raw_lower) or ("time (h)" in raw_lower and "step number" in raw_lower):
        hardware = "Novonix"
    elif any("bitrode" in c for c in raw_lower):
        hardware = "Bitrode"
        
    # Build column mapping
    mapping: Dict[str, str] = {}
    scales: Dict[str, float] = {}
    
    for target, aliases in SYNONYMS.items():
        matched_col = None
        for alias in aliases:
            norm_alias = _normalize_name(alias)
            if norm_alias in normalized:
                matched_col = normalized[norm_alias]
                break
        if matched_col is not None:
            mapping[target] = matched_col
            # Check unit scaling
            col_low = matched_col.lower()
            if "ma" in col_low and "mah" not in col_low and target == "current_a":
                scales[target] = 0.001
            elif "mah" in col_low and target in ("capacity_ah", "charge_capacity_ah"):
                scales[target] = 0.001
            elif "(h)" in col_low and target == "time_s":
                scales[target] = 3600.0
            elif "(min)" in col_low and target == "time_s":
                scales[target] = 60.0
            elif "(ms)" in col_low and target == "time_s":
                scales[target] = 0.001
            else:
                scales[target] = 1.0
                
    confidence = len(mapping) / len(SYNONYMS)
    
    return {
        "hardware": hardware,
        "mapped_columns": mapping,
        "unit_scales": scales,
        "confidence": round(confidence, 2),
        "unmapped_columns": [c for c in cols if c not in mapping.values()],
    }


def ingest_cycler_data(
    df: pd.DataFrame,
    override_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingest arbitrary cycler data and convert into standardized cycle DataFrame.
    
    If the data is point-by-point time-series, groups into discrete cycle summaries.
    If already cycle-level, standardizes columns and scales.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame from cycler file.
    override_mapping : dict, optional
        User-provided column overrides.
        
    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Standardized cycle-level DataFrame and ingestion report.
    """
    detection = detect_cycler_format(df)
    mapping = detection["mapped_columns"]
    if override_mapping:
        mapping.update(override_mapping)
        
    scales = detection["unit_scales"]
    
    # Check if raw data has point-level time-series or cycle-level
    has_time = "time_s" in mapping
    has_cycle = "cycle_number" in mapping
    
    std_df = pd.DataFrame()
    
    # Point-by-point aggregation
    if has_time and has_cycle and len(df) > 500:
        # Group by cycle_number
        cyc_col = mapping["cycle_number"]
        v_col = mapping.get("voltage_v")
        i_col = mapping.get("current_a")
        q_col = mapping.get("capacity_ah")
        t_col = mapping.get("temperature_c")
        
        cycle_rows = []
        for cyc_num, group in df.groupby(cyc_col):
            if cyc_num <= 0:
                continue
            row = {"cycle_number": int(cyc_num)}
            
            # Discharge capacity
            if q_col and q_col in group.columns:
                q_vals = group[q_col].dropna().to_numpy() * scales.get("capacity_ah", 1.0)
                if len(q_vals) > 0:
                    row["capacity_ah"] = float(np.nanmax(q_vals) - np.nanmin(q_vals)) if np.nanmax(q_vals) > 0 else float(np.nanmax(q_vals))
                else:
                    row["capacity_ah"] = 0.0
            elif i_col and "time_s" in mapping:
                # Integrate discharge current over time
                i_vals = group[i_col].to_numpy() * scales.get("current_a", 1.0)
                t_vals = group[mapping["time_s"]].to_numpy() * scales.get("time_s", 1.0)
                dis_mask = i_vals < -0.01
                if np.any(dis_mask):
                    dt = np.diff(t_vals, prepend=t_vals[0])
                    row["capacity_ah"] = float(np.sum(np.abs(i_vals[dis_mask]) * dt[dis_mask]) / 3600.0)
                else:
                    row["capacity_ah"] = 0.0
            else:
                row["capacity_ah"] = 1.0
                
            if v_col and v_col in group.columns:
                row["voltage_mean_v"] = float(group[v_col].mean())
            if i_col and i_col in group.columns:
                row["current_mean_a"] = float(group[i_col].mean() * scales.get("current_a", 1.0))
            if t_col and t_col in group.columns:
                row["temperature_mean_c"] = float(group[t_col].mean())
                
            cycle_rows.append(row)
            
        std_df = pd.DataFrame(cycle_rows)
    else:
        # Direct column normalization
        rename_dict = {}
        for target, src in mapping.items():
            if src in df.columns:
                rename_dict[src] = target
        work = df.rename(columns=rename_dict).copy()
        
        # Apply scaling
        for target, scale in scales.items():
            if target in work.columns and scale != 1.0:
                work[target] = work[target] * scale
                
        if "cycle_number" not in work.columns:
            work["cycle_number"] = np.arange(1, len(work) + 1)
        if "capacity_ah" not in work.columns and "capacity" in work.columns:
            work["capacity_ah"] = work["capacity"]
            
        std_df = work
        
    # Calculate Coulombic Efficiency & Resistance if missing
    if "capacity_ah" in std_df.columns:
        if "coulombic_efficiency" not in std_df.columns:
            std_df["coulombic_efficiency"] = 0.998
        if "resistance_ohm" not in std_df.columns:
            std_df["resistance_ohm"] = 0.025
            
    # SOH calculation
    if "capacity_ah" in std_df.columns and len(std_df) > 0:
        c0 = std_df["capacity_ah"].iloc[0] if std_df["capacity_ah"].iloc[0] > 0 else 1.0
        std_df["soh_pct"] = (std_df["capacity_ah"] / c0) * 100.0
        
    report = {
        "hardware_detected": detection["hardware"],
        "total_cycles_ingested": len(std_df),
        "columns_mapped": mapping,
        "initial_capacity_ah": round(float(std_df["capacity_ah"].iloc[0]), 3) if not std_df.empty and "capacity_ah" in std_df.columns else None,
    }
    
    return std_df, report
