"""
Feature engineering for battery cycle-degradation data.

Takes a cycle-level DataFrame (batlab.datasets.schema kind="cycle") and
produces a feature matrix (X) and targets (y_soh, y_rul) for scikit-learn.

References
----------
These are background/context citations, establishing that the general
*category* of signal each feature block tracks (fade-rate windows,
dQ/dV peak shifts, Coulombic Efficiency trends, Arrhenius/rate-dependent
aging stress) is an established degradation diagnostic in the literature —
not a claim that this code implements a specific numbered equation from any
cited paper. None of build_features()'s formulas are transcribed line-for-
line from a source; the cited papers motivate *why* a feature is worth
computing, not *how* this code computes it. Only the stress_index block's
functional form (sub-linear C-rate exponent, Arrhenius temperature term) is
inspired by a specific physical model (Doyle-Fuller-Newman porous-electrode
theory) rather than being a general literature-motivated proxy like the
rest — see its own inline derivation note for exactly what's a physical
approximation there.

Fade-rate windows, resistance trend, and general capacity-fade diagnostics:
    Birkl, C.R., Roberts, M.R., McTurk, E., Bruce, P.G., Howey, D.A.
    "Degradation diagnostics for lithium ion cells." Journal of Power
    Sources, 341, 373-386 (2017).
dQ/dV differential-capacity peak features (delegated to batlab.features.dqdv):
    Dubarry, M., Liaw, B.Y. "Identify capacity fading mechanism in a
    commercial LiFePO4 cell." Journal of Power Sources, 194(1), 541-549 (2009).
Coulombic Efficiency as a degradation-tracking signal:
    Smith, A.J., Burns, J.C., Dahn, J.R. "A High Precision Study of the
    Coulombic Efficiency of Li-Ion Batteries." Electrochemical and
    Solid-State Letters, 13(12), A177 (2010).
Stress index (Arrhenius(T) x C-rate^0.7 composite aging driver): see the
    inline derivation note above the `stress_index` block below — sub-linear
    C-rate exponent follows porous-electrode current-distribution behavior
    (Doyle-Fuller-Newman model); Arrhenius term uses Ea/R for a typical
    LiCoO2 SEI growth activation energy.
Knee detection is a separate function — see batlab.features.knee_detection
    for its own citation.

With multi-cell training, features like fade_rate and resistance_normalized
become meaningful predictors of cross-cell variation — two cells at cycle 500
with different operating histories will have different SOH, and the model
must use these signals to distinguish them.
"""

import numpy as np
import pandas as pd

from batlab.features.dqdv import add_dqdv_features

# physics_calibration lives in src/ (the app-specific integration layer),
# not batlab — batlab is a standalone, independently-citable research
# library (see pyproject.toml: its own minimal dependency list, no PyBaMM,
# no dependency on the app's sys.path setup) and every existing import
# direction in this codebase is src/app -> batlab, never the reverse.
# build_features() below imports physics_calibration lazily, inside the
# function body, guarded by try/except, and only when a cell_id is actually
# supplied -- so `pip install batlab` on its own (no src/ on sys.path, no
# PyBaMM installed) still works exactly as it does today; only this app's
# own pipeline (which has both) actually exercises that import path.
PHYSICS_FEATURE_COLUMNS = [
    "physics_beta_sei",
    "physics_beta_lam",
    "physics_k_r",
    "physics_fit_r2",
    "physics_spm_capacity_ah",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Single source of truth for "which version of build_features()/
# get_model_matrix() produced this feature DataFrame". Bump whenever either
# function changes in a way that changes the resulting feature values.
#
# Two other modules need to know when this changes and previously both kept
# their own separately-maintained copy of the string, in sync only by a
# non-blocking CI warning (a human had to remember to bump both):
#   - src/bundle_cache.py combines this with its own MODEL_VERSION (for
#     batlab/models/gbrt.py changes) into the app's disk-cache signature.
#   - batlab/validation/manifest.py re-exports this directly as its
#     FEATURE_VERSION, since a benchmark manifest's reproducibility depends
#     only on feature values, not on model training code.
# Both now import it from here instead, so it is structurally impossible
# for them to disagree with each other or with this module.
FEATURE_VERSION = "v11-usage-profile"

def build_features(
    df: pd.DataFrame, eol_threshold_pct: float = 80.0, cell_id: "str | None" = None,
) -> pd.DataFrame:
    df = df.copy().sort_values("cycle_number").reset_index(drop=True)

    # ── Capacity fade — ensure present regardless of source ──
    if "capacity_fade_ah" not in df.columns:
        initial_cap = float(df["capacity_ah"].iloc[0])
        df["capacity_fade_ah"] = (initial_cap - df["capacity_ah"]).clip(lower=0)

    if "soh_rolling_avg" not in df.columns:
        df["soh_rolling_avg"] = df["soh_pct"].rolling(10, min_periods=1).mean()

    if "is_eol" not in df.columns:
        df["is_eol"] = df["soh_pct"] < eol_threshold_pct

    if "capacity_fade_rate" not in df.columns:
        df["capacity_fade_rate"] = df["capacity_ah"].diff().abs().rolling(5, min_periods=1).mean()

    if "cumulative_days" not in df.columns:
        # No real timestamp data — approximate 1 day per cycle as a neutral fallback
        df["cumulative_days"] = df["cycle_number"].astype(float)

    # ── Fade rate at multiple windows ──
    for window in [10, 30, 50]:
        df[f"fade_rate_{window}cy"] = (
            df["capacity_ah"].diff().abs()
            .rolling(window, min_periods=1).mean()
        )

    # ── Fade acceleration (second derivative of capacity) ──
    df["fade_acceleration"] = (
        df["capacity_ah"].diff().diff()
        .rolling(10, min_periods=1).mean()
    )

    # ── SOH velocity: % health lost per cycle (50-cycle window) ──
    df["soh_velocity_50cy"] = df["soh_pct"].diff().rolling(50, min_periods=2).mean()

    # ── Resistance signals ──
    if "resistance_ohm" in df.columns:
        df["resistance_trend_30cy"] = (
            df["resistance_ohm"].diff().rolling(30, min_periods=1).mean()
        )
        # Use first non-zero value as reference; avoids div-by-zero when cycle 1
        # has a missing IR measurement stored as 0 (e.g. Severson LFP cells).
        _r_pos = df["resistance_ohm"][df["resistance_ohm"] > 0]
        initial_r = float(_r_pos.iloc[0]) if len(_r_pos) else 1.0
        df["resistance_normalized"] = df["resistance_ohm"] / initial_r
    else:
        df["resistance_trend_30cy"] = np.nan
        df["resistance_normalized"] = np.nan

    # ── Temperature: rolling mean captures operating regime, not cycle noise ──
    if "temperature_c" in df.columns:
        df["temp_rolling_30cy"] = df["temperature_c"].rolling(30, min_periods=1).mean()
    else:
        df["temp_rolling_30cy"] = np.nan

    # ── Cycle-level anomaly flags ──
    # A cycle is anomalous when its value deviates more than 2.5 std deviations
    # from a 30-cycle rolling baseline. This catches sudden resistance spikes,
    # capacity dips, and thermal events that are precursors to failure.
    for col, out in [("capacity_ah", "capacity_anomaly"), ("resistance_ohm", "resistance_anomaly")]:
        if col in df.columns:
            roll_mean = df[col].rolling(30, min_periods=5, center=True).mean()
            roll_std  = df[col].rolling(30, min_periods=5, center=True).std()
            df[out]   = (df[col] - roll_mean).abs() > (2.5 * roll_std.clip(lower=1e-9))
        else:
            df[out] = False

    # ── State of Power (SoP) — relative peak-power capability ──
    # P_peak ∝ 1/R (constant-voltage approximation).  We express it as a
    # percentage of the cell's initial power capability so it's dimensionless
    # and comparable across resistance scales (NASA vs synthetic).
    if "resistance_ohm" in df.columns:
        _r_pos2   = df["resistance_ohm"][df["resistance_ohm"] > 0]
        initial_r = float(_r_pos2.iloc[0]) if len(_r_pos2) else 1.0
        df["sop_pct"] = (initial_r / df["resistance_ohm"].clip(lower=1e-6)) * 100.0
    else:
        df["sop_pct"] = np.nan

    # ── Power-fade threshold flag ──
    # 70% of initial peak-power capability is a conventional power-fade EOL
    # floor for pulse-power duty (UPS, frequency regulation) — structurally
    # mirrors is_eol's 80% capacity floor above, not the same literature
    # source. NaN sop_pct (no resistance data) compares False, not NaN.
    df["is_power_limited"] = df["sop_pct"] < 70.0

    # ── RUL target ──
    eol_capacity = df["capacity_ah"].iloc[0] * (eol_threshold_pct / 100.0)
    eol_cycles = df.loc[df["capacity_ah"] <= eol_capacity, "cycle_number"]

    if len(eol_cycles) > 0:
        eol_at = eol_cycles.iloc[0]
        df["rul"] = (eol_at - df["cycle_number"]).clip(lower=0)
    else:
        fade_rate = df["fade_rate_50cy"].clip(lower=1e-6)
        df["rul"] = ((df["capacity_ah"] - eol_capacity) / fade_rate).clip(lower=0)

    # ── Coulombic Efficiency features ──
    if "coulombic_efficiency" in df.columns:
        df["ce_rolling_30cy"] = df["coulombic_efficiency"].rolling(30, min_periods=1).mean()
        df["ce_drop_rate"]    = df["coulombic_efficiency"].diff().rolling(10, min_periods=1).mean()
    else:
        df["ce_rolling_30cy"] = np.nan
        df["ce_drop_rate"]    = np.nan

    # ── Energy / capacity throughput ──
    # Cumulative Ah and kWh delivered by the cell over its lifetime.
    # kWh uses 3.7 V LiCoO2 nominal voltage.
    df["cumulative_ah"]  = df["capacity_ah"].cumsum()
    df["cumulative_kwh"] = (df["capacity_ah"] * 3.7).cumsum() / 1000.0

    # ── Equivalent cycles (stress-normalized) ──
    # Σ stress_weight gives the number of baseline cycles (25°C, 1C, 100% DoD)
    # that would produce equivalent degradation — the metric CATL and BYD use
    # for warranty throughput accounting.
    if "cycle_stress_weight" in df.columns:
        df["equivalent_cycles"] = df["cycle_stress_weight"].cumsum()
    else:
        df["equivalent_cycles"] = np.nan

    # ── EIS-derived component trends ──
    for _col, _out in [("r_sei", "r_sei_trend_30cy"), ("r_ct", "r_ct_trend_30cy")]:
        if _col in df.columns:
            df[_out] = df[_col].diff().rolling(30, min_periods=1).mean()
        else:
            df[_out] = np.nan

    # ── C-rate features ──
    # c_rate is written by the synthetic generator for all Cell1-8.
    # NASA cells get it injected by nasa loader (protocol-known: 1.0C discharge).
    # Severson/uploaded cells get NaN — gracefully dropped from FEATURE_COLUMNS
    # by get_model_matrix() since it filters to notna() columns only.
    if "c_rate" in df.columns and df["c_rate"].notna().any():
        # Smoothed C-rate: 10-cycle rolling mean removes per-cycle noise.
        # The window is shorter than temperature (30cy) because C-rate changes
        # between charge and discharge steps within a cycle; the per-cycle value
        # already integrates this, so less smoothing is needed.
        df["c_rate_rolling_10cy"] = (
            df["c_rate"].rolling(10, min_periods=1).mean()
        )
        # Stress index: combines C-rate and temperature into a single aging driver.
        #   C-rate term: (C / 1C)^0.7  — sub-linear because current distribution
        #     in a porous electrode is not uniform (Doyle-Fuller-Newman model).
        #   Temperature term: Arrhenius with Ea/R ≈ 6920 K (typical LiCoO2 SEI).
        #   Reference: 25°C = 298.15 K.
        # The result is dimensionless, normalized to 1.0 at (1C, 25°C).
        _c     = df["c_rate_rolling_10cy"].clip(lower=0.01)
        _T_col = "temp_rolling_30cy" if "temp_rolling_30cy" in df.columns else None
        if _T_col and df[_T_col].notna().any():
            _T_K = (df[_T_col].fillna(25.0) + 273.15).clip(lower=233.15)
            _arrhenius = np.exp(6920.0 * (1.0 / 298.15 - 1.0 / _T_K))
        else:
            _arrhenius = 1.0  # isothermal fallback
        df["stress_index"] = (_c / 1.0) ** 0.7 * _arrhenius
        # DoD proxy: ratio of current capacity to initial capacity.
        # When DoD < 100% the cell cycles over a fraction of its full range —
        # visible as a capacity plateau higher than the EOL threshold.
        _initial_cap = float(df["capacity_ah"].iloc[0])
        if _initial_cap > 0:
            df["dod_proxy"] = (df["capacity_ah"] / _initial_cap).clip(0.0, 1.0)
        else:
            df["dod_proxy"] = np.nan
    else:
        df["c_rate_rolling_10cy"] = np.nan
        df["stress_index"]        = np.nan
        df["dod_proxy"]           = np.nan

    # ── Usage-profile classification (duty-cycle regime) ──
    # c_rate_rolling_10cy/dod_proxy above are raw, instantaneous values --
    # neither captures *regime*. A cell alternating between 0.2C and 2C
    # every few cycles looks identical, feature-by-feature, at any single
    # snapshot to one held steady at a 1.1C average; only the ROLLING
    # VARIABILITY of C-rate/DoD tells an EV-like duty cycle (fast, variable
    # rate; partial, variable depth-of-discharge) apart from a
    # stationary-storage-like one (slow, steady rate; consistent near-full
    # depth-of-discharge) -- a real, previously-invisible signal since
    # cells with the same instantaneous stress_index can still accumulate
    # damage differently depending on how regular that stress is.
    # usage_profile_code is the numeric (tree-friendly, ordinal) form fed
    # to the model; usage_profile is the display-only string label.
    if df["c_rate_rolling_10cy"].notna().any():
        _c_mean = df["c_rate_rolling_10cy"].rolling(50, min_periods=5).mean()
        _c_std  = df["c_rate_rolling_10cy"].rolling(50, min_periods=5).std()
        _d_std  = df["dod_proxy"].rolling(50, min_periods=5).std()
        _is_ev = (_c_mean > 0.8) & ((_c_std > 0.15) | (_d_std > 0.05))
        _is_stationary = (_c_mean <= 0.5) & (_c_std <= 0.1) & (_d_std <= 0.03)
        df["usage_profile_code"] = np.select([_is_ev, _is_stationary], [2.0, 0.0], default=1.0)
        df.loc[_c_mean.isna(), "usage_profile_code"] = np.nan
        df["usage_profile"] = df["usage_profile_code"].map(
            {0.0: "Stationary-like", 1.0: "Mixed", 2.0: "EV-like"}
        )
    else:
        df["usage_profile_code"] = np.nan
        df["usage_profile"] = None

    # ── dQ/dV features ──
    df = add_dqdv_features(df)

    # ── Physics calibration features (SEI/LAM decomposition) ──
    # Lazy, guarded import — see the module-level note above
    # PHYSICS_FEATURE_COLUMNS for why this isn't a top-level import.
    try:
        from physics_calibration import calibrated_feature_series
        physics_df = calibrated_feature_series(df, cell_id)
    except Exception:
        # src/ not on sys.path, PyBaMM/scipy import failure, or any other
        # environment gap -- physics features are simply absent (NaN),
        # same graceful-degradation contract as every other optional
        # feature block above (c_rate, temperature, resistance, ...).
        physics_df = pd.DataFrame(
            {col: np.full(len(df), np.nan) for col in PHYSICS_FEATURE_COLUMNS},
            index=df.index,
        )
    for col in PHYSICS_FEATURE_COLUMNS:
        df[col] = physics_df[col]

    return df


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Age
    "cycle_number",
    # Fade rate signals
    "fade_rate_10cy",
    "fade_rate_30cy",
    "fade_rate_50cy",
    # Fade dynamics
    "fade_acceleration",
    "soh_velocity_50cy",
    # Resistance
    "resistance_ohm",
    "resistance_normalized",
    "resistance_trend_30cy",
    # Temperature regime — 30-cycle rolling mean separates operating
    # conditions from cycle-to-cycle noise. Only meaningful across cells
    # with different temperature profiles (multi-cell training).
    "temp_rolling_30cy",
    # dQ/dV differential capacity features -- "sim" in the column name
    # itself (not just this comment) because these are simulated from a
    # parametric OCV model, not measured. See batlab.features.dqdv.
    "dqdv_sim_peak_value",
    "dqdv_sim_peak_soc",
    "dqdv_sim_area",
    "dqdv_sim_fwhm",
    # Coulombic Efficiency — tracks SEI lithium consumption
    "ce_rolling_30cy",
    "ce_drop_rate",
    # C-rate and composite stress features.
    # c_rate_rolling_10cy: smoothed charge/discharge rate. Fast charging (>1C)
    #   accelerates SEI growth and lithium plating, compressing RUL non-linearly.
    # stress_index: Arrhenius(T) × C-rate^0.7 — the composite aging driver used
    #   by cell manufacturers for warranty throughput accounting. Dimensionless,
    #   normalized to 1.0 at (1C, 25°C). Absent for cells without C-rate data.
    # dod_proxy: actual capacity / initial capacity — cells cycled at partial DoD
    #   degrade slower; this captures that cross-cell variation for the model.
    "c_rate_rolling_10cy",
    "stress_index",
    "dod_proxy",
    # usage_profile_code: 0=Stationary-like, 1=Mixed, 2=EV-like -- ordinal
    # duty-cycle regime classification from the rolling variability of
    # c_rate/dod_proxy above (see the usage-profile block for the full
    # derivation). Absent (NaN, dropped by get_model_matrix) wherever
    # c_rate_rolling_10cy itself is absent -- same cells that don't get
    # stress_index/dod_proxy.
    "usage_profile_code",
    # Physics calibration (src/physics_calibration.py) — per-cell scipy fit
    # of a two-term degradation model against this cell's OWN measured
    # history, refit causally every 25 cycles (no future leakage). Only
    # populated for NASA/Severson cells; NaN (dropped by get_model_matrix)
    # for everything else.
    #   physics_beta_sei: SEI/LLI-driven sqrt(n) capacity-loss rate.
    #   physics_beta_lam: active-material-loss linear-in-n rate.
    #   physics_k_r: SEI resistance growth rate (independent corroborating
    #     signal, fit from resistance_ohm rather than capacity).
    #   physics_fit_r2: goodness of the two-term fit -- lets the model learn
    #     to discount these features when the fit itself is untrustworthy.
    # physics_spm_capacity_ah is deliberately NOT included here: it is cached
    # per PyBaMM parameter set (see physics_calibration._nominal_capacity_ah),
    # so it is constant across every cell of a given chemistry within any
    # single per-source model (this app never trains one combined model
    # across chemistries -- see app/main.py's load_everything() docstring) --
    # a GBRT feature with zero within-model variance carries no signal. It
    # stays in the featured DataFrame for display/diagnostic use only.
    "physics_beta_sei",
    "physics_beta_lam",
    "physics_k_r",
    "physics_fit_r2",
]

TARGET_SOH = "soh_pct"
TARGET_RUL = "rul"


def get_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Only use columns that exist AND are not entirely NaN/inf (e.g. NASA cells
    # lack temperature_c, coulombic_efficiency — those columns are all-NaN)
    available = [
        c for c in FEATURE_COLUMNS
        if c in df.columns
        and df[c].notna().any()
        and not np.isinf(df[c].replace([np.nan], 0)).all()
    ]
    cols = available + [TARGET_SOH, TARGET_RUL]
    matrix = df[cols].copy()
    # Replace remaining inf with NaN then drop incomplete rows
    matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    matrix = matrix.dropna(subset=available)
    return matrix[available], matrix[TARGET_SOH], matrix[TARGET_RUL]


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[available].describe().round(4)
