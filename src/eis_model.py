"""
Electrochemical Impedance Spectroscopy (EIS) Model for Li-ion Batteries
========================================================================

This module implements a Modified Randles circuit model for simulating and
analysing EIS data from lithium-ion cells. EIS probes the cell at multiple
frequencies to separate contributions from different physical processes:

  - Ohmic resistance (R_ohm): electrolyte + contact resistance, ~μs timescale
  - SEI film resistance (R_SEI || CPE_SEI): solid-electrolyte interphase,
    millisecond timescale
  - Charge-transfer resistance (R_ct || CPE_dl): electrode kinetics,
    milliseconds to seconds
  - Warburg diffusion (Z_w): solid-state Li diffusion in active material,
    seconds to minutes (low-frequency tail)

Circuit topology (Modified Randles):
    Z(ω) = R_ohm + (R_SEI ∥ CPE_SEI) + (R_ct ∥ CPE_dl) + Z_Warburg(ω)

CPE (Constant Phase Element) replaces ideal capacitors to capture
surface heterogeneity and distributed time constants. A CPE with α=1
is a pure capacitor; α<1 introduces a phase angle shift that flattens
the semicircle in the Nyquist plot — this is universally observed in
real battery electrodes.

References:
  - Plett, "Battery Management Systems Vol. 1", Artech House, 2015
  - Barsoukov & Macdonald, "Impedance Spectroscopy", Wiley, 2018
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frequency grid — 60 logarithmically spaced points from 100 kHz to 10 mHz
# This range captures all four impedance contributions:
#   100 kHz–1 kHz : R_ohm + SEI arc
#   1 kHz–1 Hz   : charge-transfer arc
#   1 Hz–10 mHz  : Warburg diffusion tail
# ---------------------------------------------------------------------------
FREQ_HZ: np.ndarray = np.logspace(5, -2, 60)   # Hz
OMEGA: np.ndarray = 2.0 * np.pi * FREQ_HZ       # rad/s


def decompose_resistance(
    resistance_ohm: np.ndarray,
    cycle_number: np.ndarray,
    cumulative_days: np.ndarray,
    stress_factor: float = 1.0,
) -> dict:
    """
    Decompose total DC resistance into physical sub-components.

    Physical basis
    --------------
    The total resistance of an aged Li-ion cell can be partitioned into
    three additive terms:

    1. R_ohm  — bulk electrolyte + current collector + contact resistance.
                 Dominated by ionic conductivity of the liquid electrolyte.
                 Essentially time-invariant under normal ageing; ~35 % of
                 the initial (fresh-cell) resistance.

    2. R_SEI  — resistance of the solid-electrolyte interphase grown on the
                 anode surface. SEI forms continuously via solvent reduction;
                 its thickness scales with sqrt(time) — a classic diffusion-
                 limited solid-state growth law (parabolic rate law). Calendar
                 ageing dominates this term.

    3. R_ct   — charge-transfer resistance at the electrode/electrolyte
                 interface. Reflects the activation energy barrier for Li-ion
                 insertion/extraction. Active material cracking and isolation
                 during cycling increases R_ct; empirically it follows a
                 power-law in cycle count with exponent ~0.6.

    The Warburg coefficient (sigma_w) is not a resistance but governs
    diffusion impedance magnitude; it scales with total resistance growth
    because both reflect material degradation.

    Conservation check: R_ohm + R_SEI + R_ct ≈ resistance_ohm (by design).

    Parameters
    ----------
    resistance_ohm   : array-like, total measured DC resistance [Ω]
    cycle_number     : array-like, cumulative full-equivalent cycles
    cumulative_days  : array-like, calendar age in days
    stress_factor    : float, multiplicative accelerated-stress modifier (default 1.0)

    Returns
    -------
    dict with keys 'R_ohm', 'R_SEI', 'R_ct', 'sigma_w' — each a np.ndarray
    of the same length as the input arrays.
    """
    resistance_ohm = np.asarray(resistance_ohm, dtype=float)
    cycle_number = np.asarray(cycle_number, dtype=float)
    cumulative_days = np.asarray(cumulative_days, dtype=float)

    if resistance_ohm.size == 0:
        return {"R_ohm": np.array([]), "R_SEI": np.array([]),
                "R_ct": np.array([]), "sigma_w": np.array([])}

    # --- R_ohm: constant bulk term -------------------------------------------
    # Take 35 % of the INITIAL (minimum) resistance; clip to prevent negatives.
    r_initial = np.min(resistance_ohm)
    R_ohm = np.full_like(resistance_ohm, 0.35 * r_initial)

    # --- Calendar-driven SEI growth -------------------------------------------
    # R_SEI ∝ sqrt(cumulative_days) from the parabolic SEI growth law.
    # Normalised so that the sum R_ohm + R_SEI + R_ct = resistance_ohm.
    # sqrt(days) kernel, guard against day=0 edge case.
    safe_days = np.maximum(cumulative_days, 0.0)
    sei_kernel = np.sqrt(safe_days) * stress_factor

    # --- Cycle-driven charge-transfer growth -----------------------------------
    # R_ct ∝ N^0.6; empirical power law from capacity-fade literature.
    safe_cycles = np.maximum(cycle_number, 0.0)
    ct_kernel = (safe_cycles ** 0.6) * stress_factor

    # Scale kernels so the two degradation terms sum to the residual resistance
    # after subtracting R_ohm:  residual = resistance_ohm - R_ohm
    residual = np.maximum(resistance_ohm - R_ohm, 0.0)

    kernel_sum = sei_kernel + ct_kernel

    # If both kernels are zero (fresh cell, day 0, cycle 0) split residual 50/50
    with np.errstate(invalid="ignore", divide="ignore"):
        sei_fraction = np.where(kernel_sum > 0, sei_kernel / kernel_sum, 0.5)
        ct_fraction = np.where(kernel_sum > 0, ct_kernel / kernel_sum, 0.5)

    R_SEI = sei_fraction * residual
    R_ct = ct_fraction * residual

    # --- Warburg coefficient --------------------------------------------------
    # sigma_w has units Ω/sqrt(rad/s) and captures finite Warburg diffusion.
    # Physically it grows as solid-state diffusivity in graphite decreases with
    # cracking and SEI occlusion — tracks overall resistance growth linearly.
    r_fresh = np.maximum(r_initial, 1e-9)   # guard divide-by-zero
    # Baseline sigma_w for a fresh cell: empirical Li-ion value ~0.015 Ω√s
    SIGMA_W_INITIAL = 0.015  # Ω / sqrt(rad/s)
    sigma_w = SIGMA_W_INITIAL * (resistance_ohm / r_fresh) * stress_factor

    return {
        "R_ohm": R_ohm,
        "R_SEI": R_SEI,
        "R_ct": R_ct,
        "sigma_w": sigma_w,
    }


def simulate_nyquist(
    R_ohm: float | np.ndarray,
    R_SEI: float | np.ndarray,
    R_ct: float | np.ndarray,
    sigma_w: float | np.ndarray,
    alpha_SEI: float = 0.82,
    alpha_dl: float = 0.88,
    Q_SEI: float = 8e-6,
    Q_dl: float = 3e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Nyquist impedance spectrum for a Modified Randles circuit.

    Circuit elements
    ----------------
    CPE (Constant Phase Element):
        Z_CPE(ω) = 1 / (Q × (j·ω)^α)
        α=1 → pure capacitor; α→0 → pure resistor.
        Q has units F·s^(α-1).

    Parallel R ∥ CPE:
        Z_parallel = R · Z_CPE / (R + Z_CPE)
        This produces a depressed (non-ideal) semicircle in the Nyquist plot.

    Semi-infinite Warburg (short Warburg approximation):
        Z_W(ω) = σ_w · (1 - j) / sqrt(ω)
        Appears as a 45° line in the Nyquist plot at low frequencies.

    Nyquist convention:
        Traditionally plotted as Re(Z) on x-axis, -Im(Z) on y-axis so that
        capacitive loops appear in the upper half-plane.

    Parameters
    ----------
    R_ohm   : Ohmic (bulk) resistance [Ω]; scalar or array (one cell per row)
    R_SEI   : SEI film resistance [Ω]
    R_ct    : Charge-transfer resistance [Ω]
    sigma_w : Warburg coefficient [Ω/sqrt(rad/s)]
    alpha_SEI : CPE exponent for SEI (0 < α ≤ 1, default 0.82)
    alpha_dl  : CPE exponent for double-layer (default 0.88)
    Q_SEI   : CPE pre-factor for SEI [F·s^(α-1)], default 8 µF·s^(α-1)
    Q_dl    : CPE pre-factor for double-layer, default 30 µF·s^(α-1)

    Returns
    -------
    z_real : np.ndarray, shape (n_freq,), Re(Z) in Ω
    z_imag : np.ndarray, shape (n_freq,), -Im(Z) in Ω  (positive = capacitive)
    """
    # Promote scalars to arrays for vectorised computation
    R_ohm = float(np.ravel(R_ohm)[0]) if np.ndim(R_ohm) == 0 else float(R_ohm)
    R_SEI = float(np.ravel(R_SEI)[0]) if np.ndim(R_SEI) == 0 else float(R_SEI)
    R_ct = float(np.ravel(R_ct)[0]) if np.ndim(R_ct) == 0 else float(R_ct)
    sigma_w = float(np.ravel(sigma_w)[0]) if np.ndim(sigma_w) == 0 else float(sigma_w)

    jw = 1j * OMEGA  # complex angular frequency vector, shape (60,)

    # --- CPE for SEI film ----------------------------------------------------
    # Z_CPE_SEI: high-frequency semicircle (visible above ~1 kHz)
    z_cpe_sei = 1.0 / (Q_SEI * (jw ** alpha_SEI))
    denom_sei = R_SEI + z_cpe_sei
    with np.errstate(invalid="ignore", divide="ignore"):
        z_sei = np.where(np.abs(denom_sei) > 0,
                         R_SEI * z_cpe_sei / denom_sei,
                         0.0 + 0.0j)

    # --- CPE for double layer / charge transfer ------------------------------
    # Z_CPE_dl: mid-frequency semicircle (1 kHz to ~1 Hz)
    z_cpe_dl = 1.0 / (Q_dl * (jw ** alpha_dl))
    denom_dl = R_ct + z_cpe_dl
    with np.errstate(invalid="ignore", divide="ignore"):
        z_dl = np.where(np.abs(denom_dl) > 0,
                        R_ct * z_cpe_dl / denom_dl,
                        0.0 + 0.0j)

    # --- Warburg diffusion element -------------------------------------------
    # Semi-infinite Warburg: 45° line at low frequencies
    # Guard against omega=0 (should not occur with logspace but defensive)
    safe_omega = np.maximum(OMEGA, 1e-12)
    z_warburg = sigma_w * (1.0 - 1.0j) / np.sqrt(safe_omega)

    # --- Total impedance ------------------------------------------------------
    Z_total = R_ohm + z_sei + z_dl + z_warburg

    z_real = np.real(Z_total)
    z_imag = -np.imag(Z_total)   # sign flip: positive = capacitive (convention)

    return z_real, z_imag


def ea_from_eis(
    r_sei_series: pd.Series,
    temperature_c_series: pd.Series,
) -> dict:
    """
    Estimate SEI activation energy (Ea) from Arrhenius analysis of R_SEI(T).

    Physical basis
    --------------
    The SEI ionic conductivity (and hence R_SEI) follows an Arrhenius law:

        R_SEI ∝ exp(Ea / (k_B · T))

    Taking the natural log:

        ln(R_SEI) = (Ea / k_B) · (1/T) + const
                  = (Ea / R_gas) · (1000/T) · (1/1000) + const

    A linear regression of ln(R_SEI) vs 1000/T(K) gives a slope from which
    Ea can be extracted. Typical Li-ion SEI Ea values: 0.3–0.7 eV.

    Parameters
    ----------
    r_sei_series         : pd.Series of R_SEI values [Ω]
    temperature_c_series : pd.Series of corresponding temperatures [°C]

    Returns
    -------
    dict with keys:
        ea_ev       : Ea in electron-volts
        ea_j_mol    : Ea in J/mol (= ea_ev × F, where F ≈ 96485 C/mol ≈ eV→J/mol)
        r_squared   : coefficient of determination of the linear fit
        intercept   : Arrhenius pre-exponential term (intercept of ln(R) vs 1000/T)
    """
    R_SEI = np.asarray(r_sei_series, dtype=float)
    T_c = np.asarray(temperature_c_series, dtype=float)

    # Guard: need at least 2 valid points
    valid = (R_SEI > 0) & np.isfinite(R_SEI) & np.isfinite(T_c)
    if valid.sum() < 2:
        return {"ea_ev": float("nan"), "ea_j_mol": float("nan"),
                "r_squared": float("nan"), "intercept": float("nan")}

    T_k = T_c[valid] + 273.15              # Celsius → Kelvin
    inv_T = 1000.0 / np.maximum(T_k, 1.0)  # 1000/T  [K⁻¹ × 10³]
    ln_R = np.log(R_SEI[valid])

    # --- Linear regression via normal equations (pure numpy) -----------------
    # Model: ln(R) = m · (1000/T) + b
    n = len(inv_T)
    x_mean = np.mean(inv_T)
    y_mean = np.mean(ln_R)

    ss_xx = np.sum((inv_T - x_mean) ** 2)
    ss_xy = np.sum((inv_T - x_mean) * (ln_R - y_mean))

    if ss_xx < 1e-20:
        # All temperatures identical — cannot resolve Ea
        return {"ea_ev": float("nan"), "ea_j_mol": float("nan"),
                "r_squared": float("nan"), "intercept": float("nan")}

    slope = ss_xy / ss_xx          # slope = Ea / (k_B × 1000)  [K × 10⁻³]
    intercept = y_mean - slope * x_mean

    # --- Convert slope → Ea --------------------------------------------------
    # slope units: dimensionless / (1000 K⁻¹)  →  slope has units of K
    # Ea = slope × k_B × 1000
    k_B_eV = 8.617333e-5    # Boltzmann constant [eV/K]
    k_B_J  = 1.380649e-23   # Boltzmann constant [J/K]
    N_A    = 6.02214076e23  # Avogadro number

    ea_ev    = slope * k_B_eV * 1000.0         # eV
    ea_j_mol = slope * k_B_J  * 1000.0 * N_A  # J/mol

    # --- R² ------------------------------------------------------------------
    y_pred = slope * inv_T + intercept
    ss_res = np.sum((ln_R - y_pred) ** 2)
    ss_tot = np.sum((ln_R - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "ea_ev": float(ea_ev),
        "ea_j_mol": float(ea_j_mol),
        "r_squared": float(r_squared),
        "intercept": float(intercept),
    }
