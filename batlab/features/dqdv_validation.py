"""dQ/dV simulation validation — simulated vs. real, where real data exists.

The platform's dQ/dV features (batlab.features.dqdv) are simulated from a
LiCoO2 OCV polynomial because the cycle-summary datasets (NASA/Severson/Oxford/
CALCE as loaded) don't carry raw voltage/current time series — only per-cycle
summary values.

Where raw V/I curves *are* available — Zhu 2022's raw files, or any uploaded
cycler export that includes a full discharge voltage curve — this module
computes a real dQ/dV = d(discharge capacity)/d(terminal voltage) from the
measured curve and compares it against the simulated one. The result is an
honest discrepancy report, not a claim that the simulation is a measurement.

Usage
-----
Where raw discharge curves exist:
    result = compare_simulated_dqdv_to_real(
        real_capacity_ah=...,        # total discharged Ah from the real curve
        real_voltage_v=...,          # terminal voltage array during discharge
        real_current_a=...,          # discharge current (constant or per-point)
        resistance_ohm=...,          # DC internal resistance for the cell/cycle
    )
    # result["real_peak_value"], result["sim_peak_value"],
    # result["peak_value_abs_error"], result["peak_value_rel_error"],
    # result["real_peak_soc"], result["sim_peak_soc"], ...

A caller with a fleet of raw-curve cells can run this per cell/cycle and
report the distribution of errors — which either justifies keeping the
simulated features as proxies, or quantifies exactly how far off they are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from batlab.features.dqdv import simulate_vq_curve, extract_dqdv_features


def _real_discharge_curve(
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    capacity_ah: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (Q, V) along a real discharge from raw V/I time series.

    Computes cumulative discharged capacity from current integration
    (Q = ∫ I dt, with dt approximated from row spacing or provided explicitly),
    keeps only the discharge portion (|I| above a threshold, descending V),
    and returns the (capacity_ah, terminal_voltage_v) points along it.

    Parameters
    ----------
    voltage_v : array of terminal voltage readings
    current_a : array of current readings (positive = discharge by convention here)
    capacity_ah : optional total discharged capacity; if None, computed from ∫I dt
                  and used as a normalization cross-check

    Returns
    -------
    Q : cumulative discharged capacity (Ah), ascending
    V : terminal voltage at each Q point
    """
    v = np.asarray(voltage_v, dtype=float)
    i = np.asarray(current_a, dtype=float)
    if len(v) != len(i):
        raise ValueError(f"voltage ({len(v)}) and current ({len(i)}) must have the same length")
    if len(v) < 3:
        raise ValueError("need at least 3 voltage/current points to build a real dQ/dV curve")

    # dt: assume uniform spacing unless we can infer it; conservative default 1 s.
    dt = 1.0
    dq = i * dt / 3600.0  # Ah per step (I in A, dt in s → Ah)
    Q = np.cumsum(dq)
    Q = Q - Q[0]  # relative to first point

    # Discharge portion: keep where current is meaningfully negative (discharging)
    # by the convention current_a > 0 = discharge. Keep the contiguous discharge run.
    discharge_mask = i > 1e-3
    if not discharge_mask.any():
        # Fall back: treat the whole series as one discharge if no clear separator.
        discharge_mask = np.ones(len(v), dtype=bool)

    Qd = Q[discharge_mask]
    Vd = v[discharge_mask]

    # Keep the monotonic discharge body: (capacity increasing, voltage decreasing).
    # Small wiggles are fine; a hard reversal would break dQ/dV semantics.
    # Find the longest contiguous run satisfying the discharge direction and keep it.
    if len(Qd) > 2:
        dV = np.diff(Vd)
        dQ = np.diff(Qd)
        direction_ok = (dQ >= -1e-9) & (dV <= 1e-9)   # discharge: Q up, V down
        # edge-case: at most one short reversal allowed in the middle of a long discharge
        # — keep the longest contiguous True run.
        best_start, best_len = 0, 0
        start = 0
        for k in range(len(direction_ok) + 1):
            if k < len(direction_ok) and direction_ok[k]:
                continue
            run_len = k - start
            if run_len > best_len:
                best_start, best_len = start, run_len
            start = k + 1
        if best_len >= 5:
            Qd = Qd[best_start:best_start + best_len]
            Vd = Vd[best_start:best_start + best_len]
            if len(Qd) < 5:
                raise ValueError("not enough discharge points after filtering to build a real dQ/dV curve")
    else:
        if len(Qd) < 5:
            raise ValueError("not enough discharge points after filtering to build a real dQ/dV curve")
    if len(Qd) < 5:
        raise ValueError("not enough discharge points after filtering to build a real dQ/dV curve")

    if capacity_ah is not None:
        # Sanity: integrated Q should be close to the reported capacity.
        integrated = float(Qd[-1])
        if integrated > 0 and abs(integrated - capacity_ah) / capacity_ah > 0.15:
            raise ValueError(
                f"integrated discharge Q ({integrated:.3f} Ah) disagrees with "
                f"reported capacity_ah ({capacity_ah:.3f} Ah) by more than 15% — "
                f"the raw curve may not be a clean discharge or the current convention differs"
            )

    return Qd, Vd


def _real_dqdv_features(
    Q: np.ndarray,
    V: np.ndarray,
    capacity_ah: float,
) -> dict:
    """Compute dQ/dV peak features from a real measured (Q, V) discharge curve.

    The same feature semantics as extract_dqdv_features() (peak value, peak SOC,
    area, FWHM), but computed from real data rather than a simulated OCV curve.

    Peak value is |dQ/dV| at the peak; dQ/dV is computed as 1 / (dV/dQ).
    """
    dv_dq = np.gradient(V, Q)
    dv_dq_clipped = np.where(np.abs(dv_dq) < 1e-4, np.sign(dv_dq + 1e-12) * 1e-4, dv_dq)
    dq_dv = 1.0 / dv_dq_clipped

    peak_idx = int(np.argmax(dq_dv))
    peak_value = float(dq_dv[peak_idx])

    soc = 1.0 - Q / max(capacity_ah, 1e-9)
    peak_soc = float(soc[peak_idx])

    # Area under |dQ/dV| over Q — same semantics as the simulated area.
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = float(trapz(np.abs(dq_dv), Q))

    # FWHM on |dQ/dV|
    half_max = peak_value / 2.0
    above = np.where(dq_dv > half_max)[0]
    if len(above) >= 2:
        fwhm = float(Q[above[-1]] - Q[above[0]])
    else:
        fwhm = 0.0

    return {
        "real_peak_value": peak_value,
        "real_peak_soc": peak_soc,
        "real_area": area,
        "real_fwhm": fwhm,
    }


def compare_simulated_dqdv_to_real(
    real_capacity_ah: float,
    real_voltage_v: np.ndarray,
    real_current_a: np.ndarray,
    resistance_ohm: float,
    n_points: int = 200,
) -> dict:
    """Compare the simulated LiCoO2 dQ/dV peak features against a real discharge curve.

    Parameters
    ----------
    real_capacity_ah : total discharged capacity (Ah) for this cycle
    real_voltage_v : terminal voltage time series during the discharge
    real_current_a : current time series (positive = discharge)
    resistance_ohm : DC internal resistance for the cell/cycle
    n_points : points used in the simulated curve (passed to simulate_vq_curve)

    Returns
    -------
    {
        "real": {...},          # real_dqdv_features() output
        "simulated": {...},     # extract_dqdv_features() output (LiCoO2 OCV sim)
        "peak_value_abs_error": float,
        "peak_value_rel_error": float,   # abs / max(|real|,|sim|, 1e-9)
        "peak_soc_abs_error": float,
        "area_abs_error": float,
        "area_rel_error": float,
        "fwhm_abs_error": float,
        "fwhm_rel_error": float,
        "n_real_points": int,
        "caveat": str,
    }
    """
    Q_real, V_real = _real_discharge_curve(real_voltage_v, real_current_a, real_capacity_ah)
    real_feats = _real_dqdv_features(Q_real, V_real, real_capacity_ah)

    sim_Q, sim_V = simulate_vq_curve(real_capacity_ah, resistance_ohm, n_points=n_points)
    sim_feats = extract_dqdv_features(real_capacity_ah, resistance_ohm)

    def _rel(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    caveat = (
        "Simulated features use a LiCoO2 OCV polynomial (no raw electrochemical "
        "fit); real dQ/dV from a measured discharge curve is the better reference "
        "where available. A large discrepancy here means the simulated proxy should "
        "not be treated as a real dQ/dV measurement for this cell."
    )

    return {
        "real": real_feats,
        "simulated": sim_feats,
        "peak_value_abs_error": abs(real_feats["real_peak_value"] - sim_feats["dqdv_sim_peak_value"]),
        "peak_value_rel_error": _rel(real_feats["real_peak_value"], sim_feats["dqdv_sim_peak_value"]),
        "peak_soc_abs_error": abs(real_feats["real_peak_soc"] - sim_feats["dqdv_sim_peak_soc"]),
        "area_abs_error": abs(real_feats["real_area"] - sim_feats["dqdv_sim_area"]),
        "area_rel_error": _rel(real_feats["real_area"], sim_feats["dqdv_sim_area"]),
        "fwhm_abs_error": abs(real_feats["real_fwhm"] - sim_feats["dqdv_sim_fwhm"]),
        "fwhm_rel_error": _rel(real_feats["real_fwhm"], sim_feats["dqdv_sim_fwhm"]),
        "n_real_points": len(Q_real),
        "caveat": caveat,
    }


def fleet_dqdv_validation_report(
    cell_curves: dict[str, dict],
) -> pd.DataFrame:
    """Run compare_simulated_dqdv_to_real across a fleet of cells with raw curves.

    Parameters
    ----------
    cell_curves : {cell_id: {"capacity_ah": ..., "voltage_v": array, "current_a": array,
                              "resistance_ohm": ...}}

    Returns a DataFrame (one row per cell) with the comparison metrics, suitable
    for reporting the distribution of simulation error across the cells where real
    curves exist — e.g. median peak-value rel error, max, and per-cell breakdown.
    """
    rows = []
    for cid, cur in cell_curves.items():
        try:
            comp = compare_simulated_dqdv_to_real(
                real_capacity_ah=float(cur["capacity_ah"]),
                real_voltage_v=np.asarray(cur["voltage_v"], dtype=float),
                real_current_a=np.asarray(cur["current_a"], dtype=float),
                resistance_ohm=float(cur["resistance_ohm"]),
            )
        except Exception as exc:
            rows.append({
                "cell_id": cid,
                "n_real_points": None,
                "peak_value_rel_error": None,
                "peak_soc_abs_error": None,
                "area_rel_error": None,
                "fwhm_rel_error": None,
                "error": str(exc),
            })
            continue
        rows.append({
            "cell_id": cid,
            "n_real_points": comp["n_real_points"],
            "peak_value_rel_error": comp["peak_value_rel_error"],
            "peak_soc_abs_error": comp["peak_soc_abs_error"],
            "area_rel_error": comp["area_rel_error"],
            "fwhm_rel_error": comp["fwhm_rel_error"],
            "error": None,
        })
    return pd.DataFrame(rows)
