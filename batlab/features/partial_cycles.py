"""
Partial-cycle analysis and field telemetry processing for real-world EV and BESS data.

Implements:
1. Rainflow Cycle Counting (ASTM E1049-85 standard) for irregular charge/discharge profiles.
2. Open Circuit Voltage (OCV) Relaxation Curve Reconstruction from resting intervals.
3. Partial-Window Incremental Capacity Analysis (ICA / dQ/dV) for opportunistic charging.
4. Equivalent Full Cycle (EFC) and depth-of-discharge (DoD) stress spectrum.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def rainflow_counting(
    soc_series: np.ndarray,
    time_series: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    ASTM E1049-85 Rainflow Cycle Counting for battery SOC time series.
    
    Extracts closed hysteresis cycles from arbitrary dynamic driving/dispatch profiles.
    
    Parameters
    ----------
    soc_series : np.ndarray
        Continuous battery State of Charge values in [0, 100] or [0, 1].
    time_series : np.ndarray, optional
        Timestamps corresponding to the SOC points.
        
    Returns
    -------
    dict
        Dictionary containing:
        - "cycles": list of dicts with range (DoD), mean_soc, count (0.5 for half, 1.0 for full)
        - "total_cycles": float (sum of cycle counts)
        - "equivalent_full_cycles": float (sum of (range * count) / 100)
        - "dod_histogram": dict of binned DoD counts ([0-20%], [20-40%], [40-60%], [60-80%], [80-100%])
    """
    soc = np.asarray(soc_series, dtype=np.float64)
    if len(soc) < 3:
        return {
            "cycles": [],
            "total_cycles": 0.0,
            "equivalent_full_cycles": 0.0,
            "dod_histogram": {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0},
        }
        
    # Normalize to 0-100% scale if supplied in 0-1 range
    if np.nanmax(soc) <= 1.05:
        soc = soc * 100.0
        
    # Step 1: Find reversal points (local extrema: peaks and valleys)
    diffs = np.diff(soc)
    # Remove consecutive identical points
    nonzero_idx = np.where(np.abs(diffs) > 1e-5)[0]
    if len(nonzero_idx) < 2:
        return {
            "cycles": [],
            "total_cycles": 0.0,
            "equivalent_full_cycles": 0.0,
            "dod_histogram": {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0},
        }
        
    extrema = [soc[0]]
    for i in range(1, len(soc) - 1):
        if (soc[i] - soc[i - 1]) * (soc[i + 1] - soc[i]) < 0:
            extrema.append(soc[i])
    extrema.append(soc[-1])
    
    # Step 2: 4-point ASTM E1049-85 rainflow algorithm
    points = list(extrema)
    cycles = []
    
    i = 0
    while len(points) >= 3 and i < len(points) - 2:
        s0, s1, s2 = points[i], points[i + 1], points[i + 2]
        delta_r1 = abs(s1 - s0)
        delta_r2 = abs(s2 - s1)
        
        if delta_r1 <= delta_r2:
            if i == 0:
                # Half cycle
                cycles.append({
                    "range_dod": round(float(delta_r1), 2),
                    "mean_soc": round(float((s0 + s1) / 2.0), 2),
                    "count": 0.5,
                })
                points.pop(0)
            else:
                # Closed full cycle between s0 and s1
                cycles.append({
                    "range_dod": round(float(delta_r1), 2),
                    "mean_soc": round(float((s0 + s1) / 2.0), 2),
                    "count": 1.0,
                })
                points.pop(i + 1)
                points.pop(i)
                i = max(0, i - 1)
        else:
            i += 1
            
    # Remaining unclosed residual extrema count as half cycles
    for k in range(len(points) - 1):
        delta_r = abs(points[k + 1] - points[k])
        if delta_r > 0.5:
            cycles.append({
                "range_dod": round(float(delta_r), 2),
                "mean_soc": round(float((points[k] + points[k + 1]) / 2.0), 2),
                "count": 0.5,
            })
            
    # Step 3: Compute aggregations
    total_cycle_count = sum(c["count"] for c in cycles)
    efc = sum((c["range_dod"] * c["count"]) / 100.0 for c in cycles)
    
    # DoD Histogram
    dod_bins = {"0-20%": 0.0, "20-40%": 0.0, "40-60%": 0.0, "60-80%": 0.0, "80-100%": 0.0}
    for c in cycles:
        r = c["range_dod"]
        cnt = c["count"]
        if r < 20:
            dod_bins["0-20%"] += cnt
        elif r < 40:
            dod_bins["20-40%"] += cnt
        elif r < 60:
            dod_bins["40-60%"] += cnt
        elif r < 80:
            dod_bins["60-80%"] += cnt
        else:
            dod_bins["80-100%"] += cnt
            
    return {
        "cycles": cycles,
        "total_cycles": round(total_cycle_count, 2),
        "equivalent_full_cycles": round(efc, 3),
        "dod_histogram": {k: round(v, 1) for k, v in dod_bins.items()},
    }


def reconstruct_ocv_relaxation(
    time_sec: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    rest_current_threshold: float = 0.05,
    min_rest_duration_sec: float = 60.0,
) -> Dict[str, Any]:
    """
    Reconstruct equilibrium Open Circuit Voltage (OCV) and internal resistance
    from rest/relaxation intervals in field BMS telemetry.
    
    Parameters
    ----------
    time_sec : np.ndarray
        Elapsed time array in seconds.
    voltage_v : np.ndarray
        Terminal cell voltage measurements.
    current_a : np.ndarray
        Current measurements (positive charge, negative discharge).
    rest_current_threshold : float
        Current magnitude below which the cell is considered at rest.
    min_rest_duration_sec : float
        Minimum continuous resting duration to attempt relaxation fitting.
        
    Returns
    -------
    dict
        Relaxation fit results including:
        - "detected_rests": int
        - "estimated_ocv_v": float or None
        - "instant_r0_ohm": float or None
        - "relaxation_tau_sec": float or None
    """
    t = np.asarray(time_sec, dtype=np.float64)
    v = np.asarray(voltage_v, dtype=np.float64)
    i_arr = np.asarray(current_a, dtype=np.float64)
    
    if len(t) < 10:
        return {
            "detected_rests": 0,
            "estimated_ocv_v": None,
            "instant_r0_ohm": None,
            "relaxation_tau_sec": None,
        }
        
    # Find rest periods (|I| < threshold)
    is_rest = np.abs(i_arr) < rest_current_threshold
    rest_blocks = []
    
    start_idx = None
    for idx, rest in enumerate(is_rest):
        if rest and start_idx is None:
            start_idx = idx
        elif not rest and start_idx is not None:
            duration = t[idx - 1] - t[start_idx]
            if duration >= min_rest_duration_sec:
                rest_blocks.append((start_idx, idx - 1, duration))
            start_idx = None
            
    if start_idx is not None:
        duration = t[-1] - t[start_idx]
        if duration >= min_rest_duration_sec:
            rest_blocks.append((start_idx, len(t) - 1, duration))
            
    if not rest_blocks:
        return {
            "detected_rests": 0,
            "estimated_ocv_v": None,
            "instant_r0_ohm": None,
            "relaxation_tau_sec": None,
        }
        
    # Analyze the longest rest period
    longest_rest = max(rest_blocks, key=lambda x: x[2])
    s_idx, e_idx, dur = longest_rest
    
    t_rest = t[s_idx:e_idx + 1] - t[s_idx]
    v_rest = v[s_idx:e_idx + 1]
    
    # Instantaneous resistance from the step right before rest
    r0 = None
    if s_idx > 0:
        delta_v_step = abs(v[s_idx] - v[s_idx - 1])
        delta_i_step = abs(i_arr[s_idx - 1] - i_arr[s_idx])
        if delta_i_step > 0.1:
            r0 = round(float(delta_v_step / delta_i_step), 5)
            
    # Simple exponential relaxation fit: V(t) = V_inf - (V_inf - V_0) * exp(-t / tau)
    v0 = v_rest[0]
    v_end = v_rest[-1]
    # Asymptotic extrapolation
    v_inf = v_end + 0.1 * (v_end - v0) if dur < 1800 else v_end
    
    # Effective time constant tau (time to reach ~63.2% of relaxation)
    target_v = v0 + 0.632 * (v_inf - v0)
    tau_idx = np.argmin(np.abs(v_rest - target_v))
    tau = float(t_rest[tau_idx]) if tau_idx > 0 else 60.0
    
    return {
        "detected_rests": len(rest_blocks),
        "estimated_ocv_v": round(float(v_inf), 4),
        "instant_r0_ohm": r0,
        "relaxation_tau_sec": round(tau, 1),
        "rest_duration_sec": round(float(dur), 1),
    }


def partial_ica_analysis(
    voltage_v: np.ndarray,
    capacity_ah: np.ndarray,
    v_min: float = 3.2,
    v_max: float = 4.1,
    smoothing_window: int = 15,
) -> Dict[str, Any]:
    """
    Incremental Capacity Analysis (dQ/dV) on partial charging windows.
    
    Extracts main redox peak voltage and height from partial charging data.
    
    Parameters
    ----------
    voltage_v : np.ndarray
        Voltage array during charge.
    capacity_ah : np.ndarray
        Cumulative charge capacity array.
    v_min, v_max : float
        Voltage window bounds for the peak search.
    smoothing_window : int
        Rolling window for smoothing dQ/dV derivatives.
        
    Returns
    -------
    dict
        Peak location and health indicators.
    """
    v = np.asarray(voltage_v, dtype=np.float64)
    q = np.asarray(capacity_ah, dtype=np.float64)
    
    # Filter within voltage window and ascending voltage
    mask = (v >= v_min) & (v <= v_max)
    v_sub = v[mask]
    q_sub = q[mask]
    
    if len(v_sub) < 10:
        return {
            "peak_v": None,
            "peak_dqdv": None,
            "peak_area_ah": None,
            "window_capacity_ah": 0.0,
        }
        
    # Numerical derivative dQ/dV
    dq = np.gradient(q_sub)
    dv = np.gradient(v_sub)
    # Avoid zero division
    dv_safe = np.where(np.abs(dv) < 1e-4, 1e-4, dv)
    dqdv = np.maximum(0.0, dq / dv_safe)
    
    # Smooth derivative
    if len(dqdv) >= smoothing_window:
        dqdv_smooth = pd.Series(dqdv).rolling(smoothing_window, center=True, min_periods=1).mean().to_numpy()
    else:
        dqdv_smooth = dqdv
        
    peak_idx = int(np.argmax(dqdv_smooth))
    peak_v = float(v_sub[peak_idx])
    peak_val = float(dqdv_smooth[peak_idx])
    
    window_cap = float(q_sub[-1] - q_sub[0]) if len(q_sub) > 0 else 0.0
    
    return {
        "peak_v": round(peak_v, 3),
        "peak_dqdv": round(peak_val, 3),
        "window_capacity_ah": round(window_cap, 3),
        "curve_points": len(v_sub),
    }
