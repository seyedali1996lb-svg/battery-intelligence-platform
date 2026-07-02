"""
SOH / RUL prediction model — Phase 1.

Uses a Gradient Boosting Regressor from scikit-learn.
Gradient boosting was chosen over random forest because it produces slightly
better accuracy on small-to-medium tabular datasets like this one, and its
feature importances are equally interpretable.

Two models are trained:
  - soh_model : predicts State of Health % (continuous, 80–100%)
  - rul_model : predicts Remaining Useful Life in cycles (continuous, 0–N)

Explainability:
  Both models expose feature_importances_ — how much each feature
  contributed to reducing prediction error. We surface these in the
  dashboard's Insights page.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# These hyperparameters were chosen to balance accuracy vs. training speed
# for a ~1000-cycle dataset. n_estimators=200 is enough trees to converge;
# max_depth=4 prevents overfitting on small data; learning_rate=0.05 is
# conservative (slower but more stable than the default 0.1).
GBRT_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)

# Fewer trees for quantile models — they're trained twice (Q10 + Q90) and
# quantile loss converges faster than squared loss.
GBRT_QUANTILE_PARAMS = dict(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(
    X: pd.DataFrame,
    y_soh: pd.Series,
    y_rul: pd.Series,
    test_size: float = 0.2,
) -> dict:
    """
    Train SOH and RUL models and return a model bundle.

    Why split train/test chronologically (shuffle=False)?
    Battery data is a time series — cycle 900 "knows" things that cycle 100
    doesn't. If we shuffled randomly before splitting, the test set would
    contain cycles the model already implicitly saw during training, making
    evaluation misleadingly optimistic. Chronological split is honest.

    Args:
        X:         Feature matrix from features.get_model_matrix()
        y_soh:     SOH target Series
        y_rul:     RUL target Series
        test_size: Fraction of cycles held out for evaluation (default 20%)

    Returns:
        A dict containing trained models, scalers, feature names, and metrics.
    """
    # Drop NaN/inf rows (rolling warm-up and division artefacts)
    import numpy as np
    valid = X.notna().all(axis=1) & ~np.isinf(X).any(axis=1)
    X, y_soh, y_rul = X[valid], y_soh[valid], y_rul[valid]

    if len(X) == 0:
        raise ValueError("No valid (non-NaN) training rows after feature warm-up period.")

    # Chronological split — no shuffling.
    # Guard: if dataset is too small for a test split, train on all data.
    if len(X) < max(4, int(1 / test_size) + 1):
        X_train, X_test = X, X
        y_soh_train, y_soh_test = y_soh, y_soh
        y_rul_train, y_rul_test = y_rul, y_rul
    else:
        X_train, X_test, y_soh_train, y_soh_test, y_rul_train, y_rul_test = (
            train_test_split(X, y_soh, y_rul, test_size=test_size, shuffle=False)
        )

    # Scale features: gradient boosting is tree-based so it doesn't strictly
    # need scaling, but it makes feature importances more comparable across
    # features with very different units (Ah vs. cycle count vs. Ω).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- SOH model ---
    print("Training SOH model...")
    soh_model = GradientBoostingRegressor(**GBRT_PARAMS)
    soh_model.fit(X_train_scaled, y_soh_train)

    soh_pred_test = soh_model.predict(X_test_scaled)
    soh_mae = mean_absolute_error(y_soh_test, soh_pred_test)
    soh_r2 = r2_score(y_soh_test, soh_pred_test)
    print(f"  SOH  MAE: {soh_mae:.3f}%  |  R2: {soh_r2:.4f}")

    # --- RUL model (point estimate) ---
    print("Training RUL model...")
    rul_model = GradientBoostingRegressor(**GBRT_PARAMS)
    rul_model.fit(X_train_scaled, y_rul_train)

    rul_pred_test = rul_model.predict(X_test_scaled)
    rul_mae = mean_absolute_error(y_rul_test, rul_pred_test)
    rul_r2 = r2_score(y_rul_test, rul_pred_test)
    print(f"  RUL  MAE: {rul_mae:.1f} cycles  |  R2: {rul_r2:.4f}")

    # --- RUL quantile models: 80% prediction interval (Q10 / Q90) ---
    # Quantile loss trains the model to predict the α-th percentile rather
    # than the mean. Q10+Q90 form an 80% interval — wide enough to be honest
    # about uncertainty without being uselessly vague. These use fewer trees
    # because quantile loss converges faster than squared error.
    print("Training RUL interval models (Q10/Q90)...")
    rul_q10_model = GradientBoostingRegressor(
        loss="quantile", alpha=0.10, **GBRT_QUANTILE_PARAMS
    )
    rul_q90_model = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, **GBRT_QUANTILE_PARAMS
    )
    rul_q10_model.fit(X_train_scaled, y_rul_train)
    rul_q90_model.fit(X_train_scaled, y_rul_train)
    rul_q10_test = rul_q10_model.predict(X_test_scaled)
    rul_q90_test = rul_q90_model.predict(X_test_scaled)
    # Coverage: fraction of true values inside the interval
    interval_coverage = float(np.mean(
        (y_rul_test.values >= rul_q10_test) & (y_rul_test.values <= rul_q90_test)
    ))
    print(f"  RUL interval coverage (80% target): {interval_coverage*100:.1f}%")

    feature_names = list(X.columns)

    return {
        "soh_model":     soh_model,
        "rul_model":     rul_model,
        "rul_q10_model": rul_q10_model,
        "rul_q90_model": rul_q90_model,
        "scaler":        scaler,
        "feature_names": feature_names,
        "metrics": {
            "soh_mae": soh_mae,
            "soh_r2":  soh_r2,
            "rul_mae": rul_mae,
            "rul_r2":  rul_r2,
            "rul_interval_coverage": interval_coverage,
        },
        "test_data": {
            "X_test":    X_test,
            "y_soh_test": y_soh_test,
            "y_rul_test": y_rul_test,
            "soh_pred":  soh_pred_test,
            "rul_pred":  rul_pred_test,
            "rul_q10":   rul_q10_test,
            "rul_q90":   rul_q90_test,
        },
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model_bundle: dict, X: pd.DataFrame) -> dict:
    """
    Run both models on new data and return predictions with a confidence tag.

    The "Calibrating" tag logic:
    - When we have fewer than 50 cycles of data, the rolling-window features
      (fade_rate_50cy, soh_velocity_50cy) haven't had enough history to
      stabilise. Predictions made from noisy early-cycle features are
      genuinely unreliable, so we surface "Calibrating" instead of a number.

    Args:
        model_bundle: Dict returned by train_models().
        X:            Feature matrix (one or more rows).

    Returns:
        Dict with soh_pred, rul_pred, and confidence_tag per row.
    """
    scaler = model_bundle["scaler"]
    X_scaled = scaler.transform(X)

    soh_pred = model_bundle["soh_model"].predict(X_scaled)
    rul_pred = model_bundle["rul_model"].predict(X_scaled)

    # Quantile interval (Q10/Q90) — only if quantile models are present
    # (older bundles loaded from disk may not have them; degrade gracefully).
    if "rul_q10_model" in model_bundle and "rul_q90_model" in model_bundle:
        rul_q10 = np.clip(model_bundle["rul_q10_model"].predict(X_scaled), 0, None)
        rul_q90 = np.clip(model_bundle["rul_q90_model"].predict(X_scaled), 0, None)
    else:
        rul_q10 = np.clip(rul_pred, 0, None)
        rul_q90 = np.clip(rul_pred, 0, None)

    cycle_col = X["cycle_number"] if "cycle_number" in X.columns else None
    if cycle_col is not None:
        confidence_tags = [
            "Calibrating" if c < 50 else "Model" for c in cycle_col
        ]
    else:
        confidence_tags = ["Model"] * len(soh_pred)

    return {
        "soh_pred":       soh_pred,
        "rul_pred":       np.clip(rul_pred, 0, None),
        "rul_q10":        rul_q10,
        "rul_q90":        rul_q90,
        "confidence_tag": confidence_tags,
    }


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def feature_importance_df(model_bundle: dict, model: str = "soh") -> pd.DataFrame:
    """
    Return a DataFrame of feature importances sorted descending.

    Feature importance in gradient boosting = how much each feature reduced
    the loss function (prediction error) across all trees. Higher = more
    influential. These numbers are the backbone of the Insights page.

    Args:
        model_bundle: Dict returned by train_models().
        model:        "soh" or "rul"

    Returns:
        DataFrame with columns: feature, importance, importance_pct
    """
    key = "soh_model" if model == "soh" else "rul_model"
    importances = model_bundle[key].feature_importances_
    feature_names = model_bundle["feature_names"]

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(1)
    return df


def top_drivers(model_bundle: dict, model: str = "soh", top_n: int = 5) -> list[dict]:
    """
    Return the top N most important features as a list of plain dicts.
    Used by the dashboard to build the "Why this prediction?" breakdown.
    """
    df = feature_importance_df(model_bundle, model=model)
    return df.head(top_n).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from data_loader import build_battery, get_cell_df
    from features import build_features, get_model_matrix

    print("=== Phase 1 — Model Training ===\n")

    battery = build_battery(battery_id="Oxford_B1", cell_ids=["Cell1"])
    df = get_cell_df(battery, "Cell1")
    featured_df = build_features(df)
    X, y_soh, y_rul = get_model_matrix(featured_df)

    model_bundle = train_models(X, y_soh, y_rul)

    print("\n--- Feature Importance (SOH model) ---")
    print(feature_importance_df(model_bundle, model="soh").to_string(index=False))

    print("\n--- Top 5 Drivers ---")
    for d in top_drivers(model_bundle):
        print(f"  {d['feature']:35s} {d['importance_pct']:.1f}%")

    print("\n--- Sample Prediction (last 3 cycles) ---")
    X_last = X.tail(3)
    preds = predict(model_bundle, X_last)
    for i, (soh, rul, tag) in enumerate(
        zip(preds["soh_pred"], preds["rul_pred"], preds["confidence_tag"])
    ):
        print(f"  Cycle {X_last['cycle_number'].iloc[i]:4.0f}: "
              f"SOH={soh:.1f}%  RUL={rul:.0f} cycles  [{tag}]")
