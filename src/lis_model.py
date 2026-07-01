"""
Lithium-Sulfur (Li-S) Battery Model
=====================================

This module implements electrochemical models specific to lithium-sulfur cells,
with direct relevance to research chemistries such as TALISSMAN and SUBLIME
(EU Horizon projects targeting next-generation Li-S batteries for aviation and
automotive applications).

Why Li-S differs from Li-ion
-----------------------------
Li-S cells exploit the high theoretical capacity of sulfur cathodes
(~1675 mAh/g), but practical cells achieve ~1.0 Ah in the formats modelled
here due to:

  1. Polysulfide shuttle: dissolved polysulfide intermediates (Li2Sn, n=4–8)
     migrate to the anode and are chemically reduced, causing irreversible
     capacity loss and low coulombic efficiency (CE). This is the dominant
     early-life degradation mode.

  2. Li anode consumption: metallic Li is consumed by side reactions with
     polysulfides and electrolyte, progressively increasing internal
     resistance. This becomes the dominant late-life failure.

  3. Li2S passivation: insulating Li2S precipitates on cathode surfaces,
     reducing active surface area.

Li-S voltage profile (dual plateau)
--------------------------------------
The discharge curve shows two characteristic plateaus:
  - Upper plateau (~2.35 V): S8 dissolves and reduces to long-chain
    polysulfides (S8 → Li2S4). Liquid-phase reaction, fast kinetics.
  - Lower plateau (~2.1 V): Polysulfides precipitate as short-chain Li2S2
    and Li2S. Solid-phase reaction, slower kinetics, capacity-limiting.

Resistance baseline
-------------------
Li-S electrolytes (ether-based, e.g. DME/DOL) have higher ionic conductivity
than carbonate Li-ion electrolytes, giving lower initial ohmic resistance
(~0.08 Ω vs ~0.12 Ω for Li-ion). However, Li2S passivation and Li consumption
cause faster resistance growth over cycling.

References:
  - Manthiram et al., "Lithium-Sulfur Batteries: Progress and Prospects",
    Advanced Materials, 2015
  - Wild et al., "Lithium Sulfur Batteries, a Mechanistic Review",
    Energy & Environmental Science, 2015
  - TALISSMAN project: https://talissman-project.eu
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_Q0_AH = 1.0          # nominal capacity of the cell [Ah]
_R0_OHM = 0.08        # initial ohmic resistance [Ω] (ether electrolyte)
_CE0 = 0.965          # initial coulombic efficiency (polysulfide shuttle loss)
_CE_EOL = 0.92        # CE at end of life
_K_SHUTTLE = 3.5e-4   # shuttle capacity-fade rate constant
_K_RESISTANCE = 2.0e-4  # resistance growth rate constant per cycle


def generate_lis_cell_data(
    n_cycles: int,
    temp_mean: float = 25.0,
    c_rate: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic cycling dataset for a Li-S cell.

    The simulation captures the two dominant ageing mechanisms of Li-S cells:

    1. Polysulfide shuttle (capacity fade):
       capacity_ah = Q0 × (1 - k_shuttle × N^1.2) × CE_running
       The exponent 1.2 is steeper than Li-ion (0.6–0.8) because sulfur
       cathode morphological change accelerates at higher cycle counts.

    2. Lithium anode consumption (resistance growth):
       resistance_ohm = R0 × (1 + k_resistance × N^1.4)
       Faster power law than Li-ion because metallic Li is more reactive
       than graphite and is consumed both chemically and electrochemically.

    Coulombic efficiency
    --------------------
    CE degrades from ~96.5% to ~92% over the cell's life:
       CE(N) = CE0 - (CE0 - CE_EOL) × (N / n_cycles)^0.7

    The sub-linear exponent reflects early rapid shuttle establishment
    followed by a slower quasi-steady-state regime.

    Parameters
    ----------
    n_cycles   : total number of charge-discharge cycles to simulate
    temp_mean  : mean operating temperature [°C] (default 25 °C)
    c_rate     : C-rate for cycling (default 0.5C); affects temperature variance
    seed       : random seed for reproducibility (default None)

    Returns
    -------
    pd.DataFrame with columns:
        cycle_number, capacity_ah, resistance_ohm, coulombic_efficiency,
        temperature_c, days_between_cycles, cumulative_days, chemistry
    """
    if n_cycles <= 0:
        return pd.DataFrame(columns=[
            "cycle_number", "capacity_ah", "resistance_ohm",
            "coulombic_efficiency", "temperature_c",
            "days_between_cycles", "cumulative_days", "chemistry",
        ])

    rng = np.random.default_rng(seed)
    N = np.arange(1, n_cycles + 1, dtype=float)

    # --- Coulombic efficiency degradation ------------------------------------
    # CE falls from CE0 to CE_EOL over the cycle life
    ce_decay = (_CE0 - _CE_EOL) * (N / n_cycles) ** 0.7
    ce_running = _CE0 - ce_decay
    # Add cycle-to-cycle scatter (polysulfide shuttle is stochastic)
    ce_noise = rng.normal(0.0, 0.001, size=n_cycles)
    ce = np.clip(ce_running + ce_noise, _CE_EOL - 0.01, _CE0 + 0.005)

    # --- Capacity fade via shuttle mechanism ---------------------------------
    # shuttle_factor captures progressive cathode degradation
    shuttle_factor = 1.0 - _K_SHUTTLE * (N ** 1.2)
    shuttle_factor = np.maximum(shuttle_factor, 0.0)   # cannot go negative
    capacity_ah = _Q0_AH * shuttle_factor * ce
    # Ensure monotonically non-increasing with small noise
    cap_noise = rng.normal(0.0, 0.003, size=n_cycles)
    capacity_ah = np.maximum(capacity_ah + cap_noise, 0.0)

    # --- Resistance growth via Li anode consumption --------------------------
    resistance_ohm = _R0_OHM * (1.0 + _K_RESISTANCE * (N ** 1.4))
    res_noise = rng.normal(0.0, 0.001, size=n_cycles)
    resistance_ohm = np.maximum(resistance_ohm + res_noise, _R0_OHM)

    # --- Temperature: mean + C-rate dependent variance + daily cycles --------
    # Higher C-rate → more Joule heating → larger spread
    temp_std = 2.0 + c_rate * 3.0
    temperature_c = rng.normal(temp_mean, temp_std, size=n_cycles)

    # --- Calendar time -------------------------------------------------------
    # Assume roughly 2–3 cycles per day at 0.5C with rest periods
    cycles_per_day = max(0.5, 2.0 / c_rate)  # scales with C-rate
    days_between = rng.exponential(1.0 / cycles_per_day, size=n_cycles)
    days_between = np.maximum(days_between, 0.1)
    cumulative_days = np.cumsum(days_between)

    return pd.DataFrame({
        "cycle_number": N.astype(int),
        "capacity_ah": capacity_ah,
        "resistance_ohm": resistance_ohm,
        "coulombic_efficiency": ce,
        "temperature_c": temperature_c,
        "days_between_cycles": days_between,
        "cumulative_days": cumulative_days,
        "chemistry": "Li-S",
    })


def simulate_lis_vq_curve(
    capacity_ah: float,
    resistance_ohm: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Li-S discharge voltage-capacity (V-Q) curve.

    The characteristic dual-plateau shape arises from two sequential
    electrochemical reactions during discharge:

    Upper plateau (~2.35 V) — first 25% of discharge
    --------------------------------------------------
    S8 dissolves into the electrolyte and is electrochemically reduced
    to long-chain polysulfides:
        S8 + 2Li⁺ + 2e⁻ → Li2S8 → Li2S6 → Li2S4
    This is a liquid-phase reaction. The OCV is high (~2.4 V) but
    utilisation is limited to ~25% of total capacity by the solubility
    plateau of polysulfides.

    Lower plateau (~2.1 V) — remaining 75% of discharge
    ----------------------------------------------------
    Short-chain polysulfides precipitate as insulating solid products:
        Li2S4 + 4Li⁺ + 4e⁻ → 2Li2S2 → 4Li2S
    Solid-state kinetics are slower, giving a lower, flatter plateau.
    This dominates the usable capacity.

    Terminal voltage model:
        V(q) = OCV(q) - I × R
    where I = c_rate × capacity_ah (0.5C for a 1Ah cell → I = 0.5A).

    Parameters
    ----------
    capacity_ah    : actual (aged) cell capacity [Ah]
    resistance_ohm : internal resistance [Ω]
    n_points       : number of points in the curve (default 200)

    Returns
    -------
    q_ah    : np.ndarray, discharged capacity [Ah], shape (n_points,)
    voltage : np.ndarray, terminal voltage [V], shape (n_points,)
    """
    if capacity_ah <= 0 or not np.isfinite(capacity_ah):
        return np.array([]), np.array([])

    if n_points < 2:
        n_points = 2

    # Discharge current at 0.5C
    I = 0.5 * capacity_ah  # [A]

    # Capacity axis: 0 → capacity_ah
    q_ah = np.linspace(0.0, capacity_ah, n_points)
    q_norm = q_ah / capacity_ah   # normalised 0→1

    # --- Open circuit voltage profile ----------------------------------------
    # Upper plateau: first 25% of discharge (q_norm ≤ 0.25)
    # Lower plateau: remaining 75% (q_norm > 0.25)
    # Transition region: smooth sigmoid to avoid discontinuity

    V_upper = 2.35    # V, upper plateau OCV
    V_lower = 2.10    # V, lower plateau OCV
    V_cutoff = 1.70   # V, end-of-discharge cutoff

    # Sigmoid transition centred at q_norm = 0.25, width ~0.04
    k_transition = 80.0
    transition = 1.0 / (1.0 + np.exp(-k_transition * (q_norm - 0.25)))

    # Blend upper and lower plateau OCV
    ocv = V_upper * (1.0 - transition) + V_lower * transition

    # Add a gentle slope within each plateau (OCV is not perfectly flat)
    # Upper plateau: slight drop of 30 mV over 25% capacity
    slope_upper = -0.12 * q_norm * (1.0 - transition)
    # Lower plateau: gentle slope + acceleration toward cutoff
    q_lower = np.maximum(q_norm - 0.25, 0.0) / 0.75   # normalised within lower plateau
    slope_lower = -0.08 * q_lower - 0.12 * (q_lower ** 3)
    slope_lower *= transition

    ocv = ocv + slope_upper + slope_lower

    # --- Resistive voltage drop ----------------------------------------------
    v_drop = I * resistance_ohm

    # --- Terminal voltage + realistic noise ----------------------------------
    # Li-S discharge is noisier than Li-ion due to polysulfide dynamics
    rng_noise = np.random.default_rng(int(capacity_ah * 1e4) % (2**32))
    noise_amplitude = 0.003   # 3 mV standard deviation
    noise = rng_noise.normal(0.0, noise_amplitude, size=n_points)
    # Smooth the noise with a simple rolling average (5-point)
    kernel = np.ones(5) / 5.0
    smooth_noise = np.convolve(noise, kernel, mode="same")

    voltage = ocv - v_drop + smooth_noise

    # Clamp to physical limits
    voltage = np.clip(voltage, V_cutoff, 2.50)

    return q_ah, voltage


def lis_degradation_mechanism(df: pd.DataFrame) -> dict:
    """
    Classify the dominant degradation mechanism of a Li-S cell from cycling data.

    Li-S cells typically exhibit two limiting failure modes:

    1. Shuttle-dominated (early/mid life)
       - Polysulfide shuttle is active; CE is chronically low (<97%) but
         relatively stable.
       - Capacity falls faster than resistance rises.
       - Common in cells with inadequate electrolyte additives or insufficient
         separator coatings.

    2. Anode-dominated (mid/late life)
       - Metallic Li is consumed by side reactions; resistance rises rapidly.
       - CE drops noticeably with cycles as Li inventory depletes.
       - Rate: >0.1 percentage-point drop per 100 cycles.
       - Common in cells with electrolyte starvation or thin Li foils.

    3. Mixed
       - Both mechanisms contribute at comparable rates; intermediate CE and
         resistance trends.

    Classification thresholds
    --------------------------
    Shuttle-dominated:
        mean(CE) < 0.97 AND  |d(CE)/dN| < 0.001/100 cycles

    Anode-dominated:
        |d(CE)/dN| > 0.001/100 cycles AND mean(dR/dN) > threshold

    Mixed: neither clearly dominant.

    Parameters
    ----------
    df : pd.DataFrame with at least columns 'cycle_number',
         'coulombic_efficiency', 'resistance_ohm'

    Returns
    -------
    dict with keys:
        mechanism    : str — 'Shuttle-dominated', 'Anode-dominated', or 'Mixed'
        confidence   : float in [0, 1]
        explanation  : str — human-readable rationale
    """
    required = {"cycle_number", "coulombic_efficiency", "resistance_ohm"}
    missing = required - set(df.columns)
    if missing or len(df) < 3:
        return {
            "mechanism": "Unknown",
            "confidence": 0.0,
            "explanation": "Insufficient data for classification.",
        }

    ce = np.asarray(df["coulombic_efficiency"].dropna(), dtype=float)
    res = np.asarray(df["resistance_ohm"].dropna(), dtype=float)
    cycles = np.asarray(df["cycle_number"].dropna(), dtype=float)

    if len(ce) < 3 or len(res) < 3:
        return {
            "mechanism": "Unknown",
            "confidence": 0.0,
            "explanation": "Too few valid data points.",
        }

    mean_ce = float(np.mean(ce))

    # --- CE trend: linear regression slope -----------------------------------
    n = len(cycles)
    c_mean = np.mean(cycles)
    ce_mean = np.mean(ce)
    ss_xx = np.sum((cycles - c_mean) ** 2)
    ss_xy = np.sum((cycles - c_mean) * (ce - ce_mean))
    ce_slope_per_cycle = ss_xy / ss_xx if ss_xx > 0 else 0.0
    # Convert to per-100-cycles for interpretability
    ce_drop_per_100 = -ce_slope_per_cycle * 100.0   # positive = dropping

    # --- Resistance growth rate ----------------------------------------------
    res_mean = np.mean(res)
    ss_res_xy = np.sum((cycles - c_mean) * (res - res_mean))
    res_slope = ss_res_xy / ss_xx if ss_xx > 0 else 0.0  # Ω/cycle

    # --- Classification logic -------------------------------------------------
    # Threshold: CE drop > 0.1 pp / 100 cycles flags anode mode
    CE_DROP_THRESHOLD = 0.001          # fraction / 100 cycles  (= 0.1 pp)
    # Resistance growth threshold: > 1e-4 Ω/cycle flags anode mode
    RES_GROWTH_THRESHOLD = 1.0e-4      # Ω/cycle

    shuttle_signal = mean_ce < 0.97
    anode_ce_signal = ce_drop_per_100 > CE_DROP_THRESHOLD
    anode_res_signal = res_slope > RES_GROWTH_THRESHOLD

    if anode_ce_signal and anode_res_signal:
        mechanism = "Anode-dominated"
        confidence = min(0.95, 0.6 + (ce_drop_per_100 / CE_DROP_THRESHOLD - 1) * 0.1
                         + (res_slope / RES_GROWTH_THRESHOLD - 1) * 0.1)
        explanation = (
            f"CE is dropping at {ce_drop_per_100:.3f} pp/100 cycles "
            f"(threshold {CE_DROP_THRESHOLD*100:.2f} pp/100 cycles) and resistance "
            f"is growing at {res_slope*1e4:.2f}×10⁻⁴ Ω/cycle — consistent with "
            f"progressive lithium anode consumption."
        )
    elif shuttle_signal and not anode_ce_signal:
        mechanism = "Shuttle-dominated"
        confidence = min(0.90, 0.5 + (0.97 - mean_ce) * 20)
        explanation = (
            f"Mean CE = {mean_ce:.3%} is below 97 % but stable "
            f"(trend {ce_drop_per_100:.3f} pp/100 cycles). "
            f"Polysulfide shuttle is active but lithium anode is not critically depleted."
        )
    else:
        mechanism = "Mixed"
        confidence = 0.5
        explanation = (
            f"CE = {mean_ce:.3%}, drop rate = {ce_drop_per_100:.3f} pp/100 cycles, "
            f"resistance growth = {res_slope*1e4:.2f}×10⁻⁴ Ω/cycle. "
            f"Both shuttle and anode degradation are contributing."
        )

    return {
        "mechanism": mechanism,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "explanation": explanation,
    }
