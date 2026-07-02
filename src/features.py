"""
Feature engineering — Phase 1.

Takes an enriched cycles DataFrame and produces a feature matrix (X) and
targets (y_soh, y_rul) for scikit-learn.

With multi-cell training, features like fade_rate and resistance_normalized
become meaningful predictors of cross-cell variation — two cells at cycle 500
with different operating histories will have different SOH, and the model
must use these signals to distinguish them.
"""

import numpy as np
import pandas as pd
from dqdv import add_dqdv_features


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, eol_threshold_pct: float = 80.0) -> pd.DataFrame:
    df = df.copy().sort_values("cycle_number").reset_index(drop=True)

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
        initial_r = df["resistance_ohm"].iloc[0]
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
        initial_r   = df["resistance_ohm"].iloc[0]
        df["sop_pct"] = (initial_r / df["resistance_ohm"].clip(lower=1e-6)) * 100.0
    else:
        df["sop_pct"] = np.nan

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

    # ── dQ/dV features ──
    df = add_dqdv_features(df)

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
    # dQ/dV differential capacity features
    "dqdv_peak_value",
    "dqdv_peak_soc",
    "dqdv_area",
    "dqdv_fwhm",
    # Coulombic Efficiency — tracks SEI lithium consumption
    "ce_rolling_30cy",
    "ce_drop_rate",
]

TARGET_SOH = "soh_pct"
TARGET_RUL = "rul"


def get_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    import numpy as np
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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import build_battery, get_cell_df, CELL_STRESS_PROFILES

    battery = build_battery("Oxford_B1", list(CELL_STRESS_PROFILES.keys()))

    for cell_id in list(CELL_STRESS_PROFILES.keys())[:3]:
        df = get_cell_df(battery, cell_id)
        feat = build_features(df)
        X, y_soh, y_rul = get_model_matrix(feat)
        print(f"{cell_id}: {X.shape[0]} rows | SOH {y_soh.min():.1f}–{y_soh.max():.1f}% | "
              f"RUL {y_rul.min():.0f}–{y_rul.max():.0f} cy")
