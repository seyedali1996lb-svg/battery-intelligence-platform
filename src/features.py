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

    # ── RUL target ──
    eol_capacity = df["capacity_ah"].iloc[0] * (eol_threshold_pct / 100.0)
    eol_cycles = df.loc[df["capacity_ah"] <= eol_capacity, "cycle_number"]

    if len(eol_cycles) > 0:
        eol_at = eol_cycles.iloc[0]
        df["rul"] = (eol_at - df["cycle_number"]).clip(lower=0)
    else:
        fade_rate = df["fade_rate_50cy"].clip(lower=1e-6)
        df["rul"] = ((df["capacity_ah"] - eol_capacity) / fade_rate).clip(lower=0)

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
]

TARGET_SOH = "soh_pct"
TARGET_RUL = "rul"


def get_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    matrix = df[available + [TARGET_SOH, TARGET_RUL]].dropna()
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
