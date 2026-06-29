"""
Feature engineering — Phase 1.

Takes the enriched cycles DataFrame from data_loader.py and produces a
feature matrix (X) and target vector (y) ready for scikit-learn.

Features answer the question: "what signals predict remaining useful life?"
  - How fast is capacity fading RIGHT NOW (recent trend)?
  - Is the fade accelerating?
  - How has resistance changed?
  - Where are we in the expected life cycle?

Targets:
  - soh_pct  : State of Health (regression — predicts a continuous % value)
  - rul       : Remaining Useful Life in cycles (regression)
  - is_eol   : Has the cell already crossed 80% SOH? (classification)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, eol_threshold_pct: float = 80.0) -> pd.DataFrame:
    """
    Compute all model features from the cycles DataFrame.

    Args:
        df:                 Cycles DataFrame from data_loader.enrich_cycles().
        eol_threshold_pct: SOH % below which the cell is considered end-of-life.

    Returns:
        A new DataFrame with one row per cycle containing:
          - All original columns
          - Engineered feature columns (prefixed with no special marker —
            they're just columns like everything else)
          - Target columns: soh_pct (already present), rul, is_eol (already present)
    """
    df = df.copy().sort_values("cycle_number").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Feature 1: Fade rate over the last N cycles                         #
    # "How fast is capacity dropping right now?"                          #
    # ------------------------------------------------------------------ #
    # rolling().mean() over the *diff* gives us average Ah lost per cycle
    # over a sliding window. Larger window = smoother but less reactive.
    for window in [10, 30, 50]:
        df[f"fade_rate_{window}cy"] = (
            df["capacity_ah"]
            .diff()           # change from previous cycle (negative = fading)
            .abs()            # we want magnitude, not sign
            .rolling(window, min_periods=1)
            .mean()
        )

    # ------------------------------------------------------------------ #
    # Feature 2: Fade acceleration                                         #
    # "Is the fade speeding up or levelling off?"                          #
    # ------------------------------------------------------------------ #
    # Second derivative of capacity: positive = fade is accelerating.
    df["fade_acceleration"] = df["capacity_ah"].diff().diff().rolling(10, min_periods=1).mean()

    # ------------------------------------------------------------------ #
    # Feature 3: Cumulative capacity fade                                  #
    # "How much total capacity has been lost so far?"                      #
    # ------------------------------------------------------------------ #
    # Already computed in enrich_cycles as capacity_fade_ah — keep it.

    # ------------------------------------------------------------------ #
    # Feature 4: Resistance trend                                          #
    # "Is internal resistance growing faster than expected?"               #
    # ------------------------------------------------------------------ #
    if "resistance_ohm" in df.columns:
        df["resistance_trend_30cy"] = (
            df["resistance_ohm"].diff().rolling(30, min_periods=1).mean()
        )
        # Normalised resistance: current / initial (1.0 = fresh, >1 = aged)
        initial_resistance = df["resistance_ohm"].iloc[0]
        df["resistance_normalized"] = df["resistance_ohm"] / initial_resistance
    else:
        df["resistance_trend_30cy"] = np.nan
        df["resistance_normalized"] = np.nan

    # ------------------------------------------------------------------ #
    # Feature 5: Cycle number (normalised)                                 #
    # "How old is this cell in relative terms?"                            #
    # Raw cycle number leaks absolute age; normalised version is more      #
    # generalisable across cells with different expected lifespans.        #
    # ------------------------------------------------------------------ #
    max_cycle = df["cycle_number"].max()
    df["cycle_normalized"] = df["cycle_number"] / max_cycle

    # ------------------------------------------------------------------ #
    # Feature 6: SOH velocity (slope of SOH over last 50 cycles)          #
    # "What is the current rate of health decline in % per cycle?"         #
    # ------------------------------------------------------------------ #
    df["soh_velocity_50cy"] = (
        df["soh_pct"].diff().rolling(50, min_periods=2).mean()
    )

    # ------------------------------------------------------------------ #
    # Target: Remaining Useful Life (RUL) in cycles                        #
    # "How many cycles until this cell hits 80% SOH?"                      #
    #                                                                       #
    # If the cell has already hit EOL, RUL = 0.                            #
    # If it hasn't hit EOL yet, we estimate by projecting the current      #
    # fade rate forward. This gives us a target to train against even      #
    # when the cell never actually reached EOL in the dataset.             #
    # ------------------------------------------------------------------ #
    eol_capacity = df["capacity_ah"].iloc[0] * (eol_threshold_pct / 100.0)
    eol_cycles = df.loc[df["capacity_ah"] <= eol_capacity, "cycle_number"]

    if len(eol_cycles) > 0:
        # We observed EOL: RUL = cycles remaining until that point.
        eol_at_cycle = eol_cycles.iloc[0]
        df["rul"] = (eol_at_cycle - df["cycle_number"]).clip(lower=0)
    else:
        # EOL not yet observed: project using the 50-cycle average fade rate.
        # Project from each cycle: how many more cycles until capacity hits eol_capacity?
        # RUL_i ≈ (capacity_i - eol_capacity) / current_fade_rate
        current_fade_rate = df["fade_rate_50cy"].clip(lower=1e-6)  # avoid div-by-zero
        df["rul"] = ((df["capacity_ah"] - eol_capacity) / current_fade_rate).clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# Feature matrix builder (what goes into the model)
# ---------------------------------------------------------------------------

# These are the columns we'll pass to scikit-learn.
# Keeping this list explicit means you can see exactly what the model "knows".
FEATURE_COLUMNS = [
    # Age signal — single cycle count; cycle_normalized removed (redundant: it's
    # just cycle_number / max, so the model would split importance between two
    # versions of the same signal, making the Insights page misleading).
    "cycle_number",
    # Fade rate signals — HOW FAST is capacity dropping right now?
    "fade_rate_10cy",
    "fade_rate_30cy",
    "fade_rate_50cy",
    # Fade acceleration — is the fade speeding up?
    "fade_acceleration",
    # SOH trend — rate of health decline in % per cycle
    "soh_velocity_50cy",
    # Resistance signals — independent degradation indicator
    "resistance_ohm",
    "resistance_normalized",
    "resistance_trend_30cy",
    # NOTE: capacity_ah and capacity_fade_ah are intentionally excluded.
    # SOH is defined as capacity_ah / initial_capacity, so including
    # capacity_ah would let the model trivially "predict" SOH by just
    # reading its own target. The model must learn from trend signals instead.
]

TARGET_SOH = "soh_pct"
TARGET_RUL = "rul"


def get_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract the feature matrix X and targets y_soh, y_rul from the
    fully-featured DataFrame.

    Drops rows where any feature is NaN — these occur in the first few
    cycles before rolling windows have enough data to compute.

    Returns:
        X      : DataFrame of shape (n_samples, n_features)
        y_soh  : Series of SOH % values (regression target)
        y_rul  : Series of RUL cycle counts (regression target)
    """
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    matrix = df[available_features + [TARGET_SOH, TARGET_RUL]].dropna()

    X = matrix[available_features]
    y_soh = matrix[TARGET_SOH]
    y_rul = matrix[TARGET_RUL]

    return X, y_soh, y_rul


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick stats table on each engineered feature — useful for sanity checks."""
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[available].describe().round(4)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from data_loader import build_battery, get_cell_df

    print("=== Phase 1 — Feature Engineering ===\n")

    battery = build_battery(battery_id="Oxford_B1", cell_ids=["Cell1"])
    raw_df = get_cell_df(battery, "Cell1")

    featured_df = build_features(raw_df)
    X, y_soh, y_rul = get_model_matrix(featured_df)

    print(f"Feature matrix shape: {X.shape}  (rows=cycles, cols=features)")
    print(f"SOH target range:     {y_soh.min():.1f}% — {y_soh.max():.1f}%")
    print(f"RUL target range:     {y_rul.min():.0f} — {y_rul.max():.0f} cycles")
    print("\nFeature summary:")
    print(feature_summary(featured_df).to_string())
