"""
Physics-informed capacity fade projection.

Uses an Arrhenius-weighted power-law model to project SOH forward:
    SOH(n) = 100 - A * n^beta * exp(-Ea / (R * T_K))

where:
    n    = cycle number
    A    = pre-exponential degradation coefficient (fitted)
    beta = power-law exponent (typically 0.5–0.8, SEI-limited regime)
    Ea   = activation energy (eV, converted to J internally)
    T_K  = temperature in Kelvin
    R    = gas constant (8.314 J/mol/K)

Fitting: scipy.optimize.curve_fit on observed (cycle, soh_pct) pairs.
Projection: forward-integrate to EOL threshold under three temperature scenarios.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass


R_GAS = 8.314        # J / (mol · K)
EV_TO_J = 96485.0   # eV → J conversion (1 eV = 1 e × 1 V, use F for per-mol)
# Actually use kB for per-atom: Ea in eV, kB = 8.617e-5 eV/K
KB = 8.617e-5        # eV / K


@dataclass
class FadeProjection:
    scenario: str
    temp_c: float
    cycles: np.ndarray      # projected cycle numbers
    soh_pct: np.ndarray     # projected SOH %
    eol_cycle: int | None   # cycle at which SOH crosses eol_threshold
    eol_threshold: float


@dataclass
class FitResult:
    A: float
    beta: float
    Ea_eV: float
    r2: float
    projections: list[FadeProjection]


def _soh_model(n: np.ndarray, A: float, beta: float, Ea_eV: float, T_K: float) -> np.ndarray:
    """Arrhenius + power-law SOH model."""
    return 100.0 - A * (n ** beta) * np.exp(-Ea_eV / (KB * T_K))


def fit_and_project(
    cycles: np.ndarray,
    soh_pct: np.ndarray,
    current_temp_c: float = 25.0,
    eol_threshold: float = 80.0,
    n_project: int = 500,
) -> FitResult | None:
    """
    Fit the Arrhenius power-law model to observed SOH data and project forward.

    Returns None if fitting fails (insufficient data or convergence error).
    """
    if len(cycles) < 10:
        return None

    T_K = current_temp_c + 273.15
    soh_arr = np.asarray(soh_pct, dtype=float)
    cyc_arr = np.asarray(cycles,  dtype=float)

    # Remove NaNs
    mask = np.isfinite(soh_arr) & np.isfinite(cyc_arr) & (cyc_arr > 0)
    if mask.sum() < 10:
        return None

    cyc_fit  = cyc_arr[mask]
    soh_fit  = soh_arr[mask]

    # Partial model: fix T_K, fit A, beta, Ea
    def _model(n, A, beta, Ea_eV):
        return _soh_model(n, A, beta, Ea_eV, T_K)

    try:
        p0     = [0.05, 0.55, 0.40]
        bounds = ([1e-6, 0.3, 0.1], [10.0, 1.0, 1.5])
        popt, _ = curve_fit(_model, cyc_fit, soh_fit, p0=p0, bounds=bounds,
                            maxfev=5000, method="trf")
        A, beta, Ea_eV = popt
    except Exception:
        return None

    soh_pred = _model(cyc_fit, A, beta, Ea_eV)
    ss_res   = np.sum((soh_fit - soh_pred) ** 2)
    ss_tot   = np.sum((soh_fit - soh_fit.mean()) ** 2)
    r2       = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Project three scenarios
    n_start  = int(cyc_arr[-1]) + 1
    n_end    = n_start + n_project
    n_proj   = np.arange(n_start, n_end)

    scenarios = [
        ("Conservative (cool, low C-rate)", current_temp_c - 5),
        ("Nominal (current conditions)",     current_temp_c),
        ("Aggressive (warm, high C-rate)",   current_temp_c + 10),
    ]

    projections = []
    for label, temp in scenarios:
        T_sc  = temp + 273.15
        soh_p = _soh_model(n_proj, A, beta, Ea_eV, T_sc)
        soh_p = np.clip(soh_p, 0, 100)

        # Find EOL crossing
        below = np.where(soh_p <= eol_threshold)[0]
        eol_cy = int(n_proj[below[0]]) if len(below) > 0 else None

        projections.append(FadeProjection(
            scenario=label,
            temp_c=temp,
            cycles=n_proj,
            soh_pct=soh_p,
            eol_cycle=eol_cy,
            eol_threshold=eol_threshold,
        ))

    return FitResult(A=A, beta=beta, Ea_eV=Ea_eV, r2=r2, projections=projections)
