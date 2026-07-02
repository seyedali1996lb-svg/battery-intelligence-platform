"""
Differential capacity (dQ/dV) feature extraction.

Simulates a voltage-capacity discharge curve using a LiCoO2 OCV model and
extracts physically-meaningful peak features that track electrode degradation.
"""

import numpy as np
import pandas as pd


def simulate_vq_curve(
    capacity_ah: float,
    resistance_ohm: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a discharge V(Q) curve for a single cycle.

    Uses a LiCoO2 open-circuit voltage (OCV) model and a constant 2A discharge
    current to produce a realistic voltage profile.

    Parameters
    ----------
    capacity_ah : float
        Discharge capacity for this cycle (Ah).
    resistance_ohm : float
        Internal resistance (Ohm).
    n_points : int
        Number of points along the discharge curve.

    Returns
    -------
    Q_array : np.ndarray  shape (n_points,)
        Cumulative discharge (Ah), from 0 to capacity_ah.
    V_array : np.ndarray  shape (n_points,)
        Terminal voltage (V) at each Q point.
    """
    Q_array = np.linspace(0, capacity_ah, n_points)
    soc_array = 1.0 - Q_array / max(capacity_ah, 1e-9)

    # LiCoO2 OCV model
    soc = soc_array
    ocv = (
        3.7
        + 0.7 * soc
        - 0.5 * soc ** 2
        + 0.3 * soc ** 3
        - 0.1 * (1 - soc) ** 3
        + 0.08 * np.exp(-20 * soc)
        - 0.05 * np.exp(-20 * (1 - soc))
    )

    I_discharge = 2.0  # Amperes
    V_array = ocv - I_discharge * resistance_ohm

    return Q_array, V_array


def extract_dqdv_features(capacity_ah: float, resistance_ohm: float) -> dict:
    """
    Compute dQ/dV peak features for a single cycle.

    Parameters
    ----------
    capacity_ah : float
    resistance_ohm : float

    Returns
    -------
    dict with keys:
        dqdv_peak_value  — amplitude of the main dQ/dV peak
        dqdv_peak_soc    — SOC position of the main peak
        dqdv_area        — trapz integral of dQ/dV over Q (total area)
        dqdv_fwhm        — full-width at half-maximum of the peak (Ah)
    """
    Q, V = simulate_vq_curve(capacity_ah, resistance_ohm)

    # dV/dQ via numerical gradient
    dv_dq = np.gradient(V, Q)

    # Avoid division by zero / near-zero
    dv_dq_clipped = np.where(np.abs(dv_dq) < 1e-4, np.sign(dv_dq + 1e-12) * 1e-4, dv_dq)
    dq_dv = 1.0 / dv_dq_clipped

    # Peak value and its position
    peak_idx = int(np.argmax(dq_dv))
    dqdv_peak_value = float(dq_dv[peak_idx])

    soc_array = 1.0 - Q / max(capacity_ah, 1e-9)
    dqdv_peak_soc = float(soc_array[peak_idx])

    # Area under dQ/dV curve (trapz over Q)
    # np.trapezoid is NumPy 2.0+; np.trapz removed in NumPy 2.0
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    dqdv_area = float(_trapz(dq_dv, Q))

    # FWHM: find indices where dQ/dV > peak / 2
    half_max = dqdv_peak_value / 2.0
    above = np.where(dq_dv > half_max)[0]
    if len(above) >= 2:
        dqdv_fwhm = float(Q[above[-1]] - Q[above[0]])
    else:
        dqdv_fwhm = 0.0

    return {
        "dqdv_peak_value": dqdv_peak_value,
        "dqdv_peak_soc":   dqdv_peak_soc,
        "dqdv_area":       dqdv_area,
        "dqdv_fwhm":       dqdv_fwhm,
    }


def add_dqdv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dQ/dV feature extraction to every row of a cycle DataFrame.

    Expects columns: capacity_ah, resistance_ohm.
    Adds columns: dqdv_peak_value, dqdv_peak_soc, dqdv_area, dqdv_fwhm.

    Parameters
    ----------
    df : pd.DataFrame
        Cycle-level DataFrame (one row per cycle).

    Returns
    -------
    pd.DataFrame
        Modified copy with four new dQ/dV columns appended.
    """
    df = df.copy()

    results = df.apply(
        lambda row: extract_dqdv_features(
            row["capacity_ah"], row["resistance_ohm"]
        ),
        axis=1,
        result_type="expand",
    )

    for col in ["dqdv_peak_value", "dqdv_peak_soc", "dqdv_area", "dqdv_fwhm"]:
        df[col] = results[col]

    return df
