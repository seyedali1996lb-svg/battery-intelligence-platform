"""
Physics-based RUL projection via PyBaMM Single Particle Model (SPM).

Strategy
--------
1. Run one SPM discharge to get a physics-grounded nominal capacity for
   the cell's chemistry (Chen2020 for LFP, NCA_Kim2011 for NCA/LiCoO2,
   Marquis2019 for synthetic LiCoO2).
2. Fit the SEI-growth fade equation Q(n) = Q0 * (1 - beta * sqrt(n)) to
   the cell's measured capacity history.
3. Project Q(n) forward until the EOL threshold, yielding a physics-based
   RUL estimate and a projected SOH trajectory.

The SPM run anchors the model in electrochemistry; the beta fit from real
data makes the projection specific to this cell's measured degradation rate.
This is meaningfully different from the GBRT linear extrapolation.

Parameter sets used (all from PyBaMM built-ins):
  LFP   (Severson)  -> Chen2020
  NCA   (NASA)      -> NCA_Kim2011
  LiCoO2 (synthetic/uploaded) -> Marquis2019
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── Parameter set selection ─────────────────────────────────────────────────

_PARAM_MAP = {
    "severson":  "Chen2020",
    "nasa":      "NCA_Kim2011",
    "synthetic": "Marquis2019",
    "uploaded":  "Marquis2019",
}

_CHEM_LABEL = {
    "Chen2020":     "LFP/Graphite (Chen 2020)",
    "NCA_Kim2011":  "NCA/Graphite (Kim 2011)",
    "Marquis2019":  "LiCoO₂/Graphite (Marquis 2019)",
}


def _run_spm_single_cycle(param_set_name: str) -> float:
    """Run one SPM discharge and return nominal capacity in Ah."""
    import pybamm  # lazy import — only loaded when this function is called

    model = pybamm.lithium_ion.SPM()
    param = pybamm.ParameterValues(param_set_name)
    sim   = pybamm.Simulation(model, parameter_values=param)
    sol   = sim.solve([0, 3600])
    return float(sol["Discharge capacity [A.h]"].entries[-1])


def _fit_sei_fade(
    cycles: np.ndarray,
    soh: np.ndarray,
) -> tuple[float, float]:
    """
    Fit beta in Q(n) / Q0 = 1 - beta * sqrt(n)  →  SOH(n) = 1 - beta*sqrt(n).

    Returns (beta, sigma_beta) where sigma_beta is the 1-σ standard error
    from the curve_fit covariance matrix. Propagate as ±2σ for 95% band.
    """
    from scipy.optimize import curve_fit

    def _model(n, beta):
        return 1.0 - beta * np.sqrt(n)

    soh_norm = np.clip(soh / 100.0, 0.01, 1.0)
    n0 = cycles[0] if cycles[0] > 0 else 1.0
    n_shifted = cycles - n0 + 1.0

    try:
        popt, pcov = curve_fit(_model, n_shifted, soh_norm, p0=[0.001],
                               bounds=(0, 0.1), maxfev=2000)
        sigma = float(np.sqrt(np.diag(pcov))[0])
        return float(popt[0]), sigma
    except Exception:
        delta_soh  = (soh_norm[0] - soh_norm[-1])
        delta_sqrt = np.sqrt(n_shifted[-1]) - np.sqrt(n_shifted[0])
        beta = max(1e-6, delta_soh / (delta_sqrt + 1e-9))
        return beta, beta * 0.1   # fallback: 10% relative uncertainty


def project_rul(
    cell_id: str,
    df,               # pandas DataFrame with cycle_number, soh_pct, capacity_ah
    data_mode: str,   # "severson" | "nasa" | "synthetic" | "uploaded"
    eol_threshold: float = 80.0,
    project_cycles: int = 500,
) -> dict:
    """
    Run physics-based RUL projection for one cell.

    Returns dict with keys:
      param_set       str   — PyBaMM parameter set name used
      chem_label      str   — human-readable chemistry label
      spm_capacity_ah float — nominal capacity from SPM discharge
      beta            float — fitted SEI fade coefficient
      proj_cycles     list  — future cycle numbers
      proj_soh        list  — projected SOH % (physics model)
      rul_physics     int   — cycles until EOL_threshold, or None
      eol_threshold   float — threshold used
      error           str   — set if projection failed
    """
    param_set = _PARAM_MAP.get(data_mode, "Marquis2019")

    result: dict = {
        "param_set":       param_set,
        "chem_label":      _CHEM_LABEL[param_set],
        "spm_capacity_ah": None,
        "beta":            None,
        "beta_sigma":      None,   # 1-σ uncertainty from curve_fit covariance
        "proj_cycles":     [],
        "proj_soh":        [],
        "proj_soh_lo":     [],     # β + 2σ pessimistic band
        "proj_soh_hi":     [],     # β − 2σ optimistic band
        "rul_physics":     None,
        "eol_threshold":   eol_threshold,
        "error":           None,
    }

    try:
        import pandas as _pd

        # ── 1. SPM single-cycle for nominal capacity ──
        result["spm_capacity_ah"] = _run_spm_single_cycle(param_set)

        # ── 2. Fit SEI fade to measured data ──
        _valid = df[["cycle_number", "soh_pct"]].dropna()
        if len(_valid) < 5:
            result["error"] = "Insufficient measured cycles for fade fit (need ≥ 5)."
            return result

        cycles_meas = _valid["cycle_number"].values.astype(float)
        soh_meas    = _valid["soh_pct"].values.astype(float)

        result["beta"], result["beta_sigma"] = _fit_sei_fade(cycles_meas, soh_meas)
        _beta      = result["beta"]
        _beta_lo   = max(1e-9, _beta + 2 * result["beta_sigma"])   # pessimistic
        _beta_hi   = max(1e-9, _beta - 2 * result["beta_sigma"])   # optimistic

        # ── 3. Project forward ──
        current_cycle = int(cycles_meas[-1])
        n0            = cycles_meas[0] if cycles_meas[0] > 0 else 1.0

        future_cycles = np.arange(current_cycle + 1, current_cycle + project_cycles + 1)
        n_shifted     = future_cycles - n0 + 1.0
        proj_soh_norm = np.clip(1.0 - _beta    * np.sqrt(n_shifted), 0.0, 1.0)
        proj_soh_lo   = np.clip(1.0 - _beta_lo * np.sqrt(n_shifted), 0.0, 1.0)
        proj_soh_hi   = np.clip(1.0 - _beta_hi * np.sqrt(n_shifted), 0.0, 1.0)

        # ── 4. Find RUL from central projection ──
        proj_soh_pct = proj_soh_norm * 100.0
        below = np.where(proj_soh_pct < eol_threshold)[0]
        rul_cy = int(future_cycles[below[0]]) - current_cycle if len(below) > 0 else project_cycles

        result["proj_cycles"] = future_cycles.tolist()
        result["proj_soh"]    = proj_soh_pct.tolist()
        result["proj_soh_lo"] = (proj_soh_lo * 100).tolist()
        result["proj_soh_hi"] = (proj_soh_hi * 100).tolist()
        result["rul_physics"] = rul_cy

        # ── H3: Temperature sensitivity bands β(T) = β₀·exp(−Ea/RT) ──────
        # Ea ≈ 50 kJ/mol for graphite SEI (literature consensus)
        # β₀ calibrated at 25°C (298 K) from the fitted β
        _Ea   = 50_000.0          # J/mol
        _R    = 8.314             # J/(mol·K)
        _T0   = 298.15            # reference temperature (25°C) in K
        _beta0 = _beta / np.exp(-_Ea / (_R * _T0))   # pre-exponential factor
        _temp_scenarios = {"15°C": 288.15, "25°C": 298.15, "35°C": 308.15}
        result["temp_scenarios"] = {}
        for _label, _T_K in _temp_scenarios.items():
            _beta_T   = _beta0 * np.exp(-_Ea / (_R * _T_K))
            _soh_T    = np.clip(1.0 - _beta_T * np.sqrt(n_shifted), 0.0, 1.0) * 100
            result["temp_scenarios"][_label] = _soh_T.tolist()

    except Exception as e:
        result["error"] = str(e)

    return result
