"""
Differential capacity (dQ/dV) feature extraction.

Simulates a voltage-capacity discharge curve using a LiCoO2 OCV model and
extracts physically-meaningful peak features that track electrode degradation.

Reference: dQ/dV (differential capacity) peak position, amplitude, and
width shifts are an established non-destructive signature of electrode-level
degradation mechanisms (loss of active material / loss of lithium
inventory) — see Dubarry, M., Liaw, B.Y. "Identify capacity fading
mechanism in a commercial LiFePO4 cell." Journal of Power Sources, 194(1),
541-549 (2009). This module simulates the underlying V(Q) curve from a
parametric OCV model rather than measuring it directly (no raw
voltage/current time series is available in the cycle-summary datasets
this library loads), so treat its features as a physically-motivated
proxy, not a direct measurement — the same caveat the app UI already
surfaces for LFP cells, whose flat OCV plateau makes this LiCoO2-shaped
model inapplicable.
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
    dict with keys (all prefixed "dqdv_sim_" — see module docstring for why
    "sim": the V(Q) curve these are computed from is simulated from a
    parametric OCV model, not measured, so the flag needs to survive into
    any raw column-name listing, not just this docstring):
        dqdv_sim_peak_value  — amplitude of the main dQ/dV peak
        dqdv_sim_peak_soc    — SOC position of the main peak
        dqdv_sim_area        — trapz integral of dQ/dV over Q (total area)
        dqdv_sim_fwhm        — full-width at half-maximum of the peak (Ah)
    """
    Q, V = simulate_vq_curve(capacity_ah, resistance_ohm)

    # dV/dQ via numerical gradient
    dv_dq = np.gradient(V, Q)

    # Avoid division by zero / near-zero
    dv_dq_clipped = np.where(np.abs(dv_dq) < 1e-4, np.sign(dv_dq + 1e-12) * 1e-4, dv_dq)
    dq_dv = 1.0 / dv_dq_clipped

    # Peak value and its position
    peak_idx = int(np.argmax(dq_dv))
    dqdv_sim_peak_value = float(dq_dv[peak_idx])

    soc_array = 1.0 - Q / max(capacity_ah, 1e-9)
    dqdv_sim_peak_soc = float(soc_array[peak_idx])

    # Area under dQ/dV curve (trapz over Q)
    # np.trapezoid is NumPy 2.0+; np.trapz removed in NumPy 2.0
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    dqdv_sim_area = float(_trapz(dq_dv, Q))

    # FWHM: find indices where dQ/dV > peak / 2
    half_max = dqdv_sim_peak_value / 2.0
    above = np.where(dq_dv > half_max)[0]
    if len(above) >= 2:
        dqdv_sim_fwhm = float(Q[above[-1]] - Q[above[0]])
    else:
        dqdv_sim_fwhm = 0.0

    return {
        "dqdv_sim_peak_value": dqdv_sim_peak_value,
        "dqdv_sim_peak_soc":   dqdv_sim_peak_soc,
        "dqdv_sim_area":       dqdv_sim_area,
        "dqdv_sim_fwhm":       dqdv_sim_fwhm,
    }


def add_dqdv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized dQ/dV feature extraction across all cycle rows.

    dV/dQ = d(OCV)/dt · (1/cap) where t = Q/cap is normalised charge.
    The OCV derivative is identical for every row (IR offset cancels in
    the gradient), so we compute it once on a (N,) array and broadcast
    across all rows — replacing a Python-level loop with pure NumPy ops.
    """
    df = df.copy()

    caps = df["capacity_ah"].to_numpy(dtype=float)
    n, N = len(caps), 200

    t   = np.linspace(0, 1, N)          # normalised Q  (0 → 1)
    soc = 1.0 - t                        # SOC decreases as Q increases

    # LiCoO2 OCV model — same for all rows
    ocv = (
        3.7
        + 0.7 * soc - 0.5 * soc**2 + 0.3 * soc**3
        - 0.1 * (1 - soc)**3
        + 0.08 * np.exp(-20 * soc)
        - 0.05 * np.exp(-20 * (1 - soc))
    )

    # d(OCV)/dt  — uniform spacing dt = 1/(N-1), compute once
    docv_dt = np.gradient(ocv, t)
    docv_dt = np.where(np.abs(docv_dt) < 1e-4,
                       np.sign(docv_dt + 1e-12) * 1e-4, docv_dt)

    # dQ/dV[i, j] = caps[i] / docv_dt[j]   shape (n, N)
    dqdv = caps[:, np.newaxis] / docv_dt[np.newaxis, :]

    # Q arrays for each row   shape (n, N)
    Q = caps[:, np.newaxis] * t[np.newaxis, :]

    # ── Peak value + SOC position ──
    peak_idx            = np.argmax(dqdv, axis=1)
    dqdv_sim_peak_value = dqdv[np.arange(n), peak_idx]
    dqdv_sim_peak_soc   = soc[peak_idx]

    # ── Area: trapz over Q (uniform spacing per row) ──
    dq            = caps / (N - 1)
    dqdv_sim_area = np.sum(
        (dqdv[:, 1:] + dqdv[:, :-1]) * 0.5 * dq[:, np.newaxis], axis=1
    )

    # ── FWHM ──
    half_max  = dqdv_sim_peak_value / 2.0
    above     = dqdv > half_max[:, np.newaxis]           # (n, N) bool
    has_any   = above.any(axis=1)
    first_idx = np.argmax(above, axis=1)
    last_idx  = N - 1 - np.argmax(above[:, ::-1], axis=1)
    dqdv_sim_fwhm = np.where(
        has_any,
        Q[np.arange(n), last_idx] - Q[np.arange(n), first_idx],
        0.0,
    )

    # "dqdv_sim_" prefix (not just "dqdv_") -- these are simulated from a
    # parametric OCV model, not measured from real V(t) telemetry (no raw
    # voltage/current time series exists in the cycle-summary datasets this
    # library loads). The prefix needs to survive into FEATURE_COLUMNS and
    # any SHAP-style importance table, not just live in this module's
    # docstring where a reader skimming a column list would never see it.
    df["dqdv_sim_peak_value"] = dqdv_sim_peak_value
    df["dqdv_sim_peak_soc"]   = dqdv_sim_peak_soc
    df["dqdv_sim_area"]       = dqdv_sim_area
    df["dqdv_sim_fwhm"]       = dqdv_sim_fwhm

    return df
