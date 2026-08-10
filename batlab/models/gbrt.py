"""
SOH / RUL prediction model.

Uses a Gradient Boosting Regressor from scikit-learn.
Gradient boosting was chosen over random forest because it produces slightly
better accuracy on small-to-medium tabular datasets like this one, and its
feature importances are equally interpretable.

Two point-estimate models are trained:
  - soh_model : predicts State of Health % (continuous, 80–100%)
  - rul_model : predicts Remaining Useful Life in cycles (continuous, 0–N)
Plus two quantile models (Q10/Q90) forming an 80% RUL prediction interval.

Explainability:
  Both models expose feature_importances_ — how much each feature
  contributed to reducing prediction error.
"""

import copy

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
        X:         Feature matrix from batlab.features.get_model_matrix()
        y_soh:     SOH target Series
        y_rul:     RUL target Series
        test_size: Fraction of cycles held out for evaluation (default 20%)

    Returns:
        A dict containing trained models, scalers, feature names, and metrics.
    """
    if len(X) == 0:
        raise ValueError("Training dataset is empty — all rows were dropped during feature filtering.")

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
    soh_model = GradientBoostingRegressor(**GBRT_PARAMS)
    soh_model.fit(X_train_scaled, y_soh_train)

    soh_pred_test = soh_model.predict(X_test_scaled)
    soh_mae = mean_absolute_error(y_soh_test, soh_pred_test)
    soh_r2 = r2_score(y_soh_test, soh_pred_test)

    # --- RUL model (point estimate) ---
    rul_model = GradientBoostingRegressor(**GBRT_PARAMS)
    rul_model.fit(X_train_scaled, y_rul_train)

    rul_pred_test = rul_model.predict(X_test_scaled)
    rul_mae = mean_absolute_error(y_rul_test, rul_pred_test)
    rul_r2 = r2_score(y_rul_test, rul_pred_test)

    # --- RUL quantile models: 80% prediction interval (Q10 / Q90) ---
    # Quantile loss trains the model to predict the α-th percentile rather
    # than the mean. Q10+Q90 form an 80% interval — wide enough to be honest
    # about uncertainty without being uselessly vague. These use fewer trees
    # because quantile loss converges faster than squared error.
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
# Incremental (warm-start) updating
# ---------------------------------------------------------------------------

# How many new boosting estimators a single warm-start update adds. Fewer
# trees than a full retrain (200/150) because this is meant to be called
# repeatedly on each new batch of telemetry, not once — the ensemble grows
# a little each time rather than being rebuilt.
WARM_START_N_NEW_ESTIMATORS = 20

# sklearn's GradientBoostingRegressor grows unboundedly under repeated
# warm_start calls (each call only ever adds trees, never prunes). Past
# this many total estimators, warm_start_update() refuses and the caller
# should fall back to a full train_models() refit instead, which also
# resets tree count back to GBRT_PARAMS' baseline.
WARM_START_MAX_ESTIMATORS = 500


def warm_start_update(
    model_bundle: dict,
    X: pd.DataFrame,
    y_soh: pd.Series,
    y_rul: pd.Series,
    n_new_estimators: int = WARM_START_N_NEW_ESTIMATORS,
    test_size: float = 0.2,
) -> dict:
    """
    Incrementally extend an already-fitted model bundle with a few more
    boosting estimators trained on new data, instead of a full
    train_models() refit from scratch — the "warm-start incremental
    refit" answer to why this exists (see README/pending-work: sklearn's
    GBRT has no true partial_fit, only warm_start, which keeps existing
    trees and only fits the newly-added ones).

    Why the scaler is reused, never refit: the existing trees' splits were
    learned in the old scaler's feature space. Refitting the scaler here
    would silently shift every feature's scale out from under trees that
    already exist, corrupting them without raising an error — so the new
    trees are fit in the SAME space as the ones already grown.

    This is not equivalent to a full retrain on the combined dataset — it
    can drift from what train_models() would produce on the same
    cumulative data, especially if the new batch's distribution has
    shifted from what the original fit saw. Compare the result against
    warm_start_vs_full_retrain_drift() before trusting it in place of an
    occasional full refit; callers should also fall back to a full
    train_models() call once WARM_START_MAX_ESTIMATORS is reached (see
    that constant) rather than growing the ensemble forever.

    Raises ValueError if the update would exceed WARM_START_MAX_ESTIMATORS
    or if X's columns don't match the existing bundle's feature_names —
    silently training on a shifted feature set is exactly the kind of
    invisible failure this platform's honest-validation philosophy exists
    to prevent.
    """
    if list(X.columns) != list(model_bundle["feature_names"]):
        raise ValueError(
            "warm_start_update(): X's columns don't match the existing bundle's "
            "feature_names — a full train_models() refit is required when the "
            "feature set has changed, not an incremental update."
        )

    current_n = model_bundle["soh_model"].n_estimators
    if current_n + n_new_estimators > WARM_START_MAX_ESTIMATORS:
        raise ValueError(
            f"warm_start_update(): {current_n} + {n_new_estimators} estimators "
            f"would exceed WARM_START_MAX_ESTIMATORS ({WARM_START_MAX_ESTIMATORS}) — "
            "call train_models() for a full refit instead, which resets tree count."
        )

    new_bundle = copy.deepcopy(model_bundle)
    scaler = new_bundle["scaler"]
    X_scaled = scaler.transform(X)

    # Chronological holdout on the NEW data only, same shuffle=False honesty
    # rule as train_models() — this new batch is itself a time series.
    if len(X) < max(4, int(1 / test_size) + 1):
        train_idx = test_idx = slice(None)
    else:
        n_test = max(1, int(len(X) * test_size))
        train_idx = slice(0, len(X) - n_test)
        test_idx = slice(len(X) - n_test, None)

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_soh_train, y_soh_test = y_soh.iloc[train_idx], y_soh.iloc[test_idx]
    y_rul_train, y_rul_test = y_rul.iloc[train_idx], y_rul.iloc[test_idx]

    for key, y_train in (
        ("soh_model", y_soh_train), ("rul_model", y_rul_train),
        ("rul_q10_model", y_rul_train), ("rul_q90_model", y_rul_train),
    ):
        model = new_bundle[key]
        model.set_params(warm_start=True, n_estimators=model.n_estimators + n_new_estimators)
        model.fit(X_train, y_train)

    soh_pred_test = new_bundle["soh_model"].predict(X_test)
    rul_pred_test = new_bundle["rul_model"].predict(X_test)
    rul_q10_test = np.clip(new_bundle["rul_q10_model"].predict(X_test), 0, None)
    rul_q90_test = np.clip(new_bundle["rul_q90_model"].predict(X_test), 0, None)

    new_bundle["metrics"] = {
        "soh_mae": mean_absolute_error(y_soh_test, soh_pred_test),
        "soh_r2":  r2_score(y_soh_test, soh_pred_test),
        "rul_mae": mean_absolute_error(y_rul_test, rul_pred_test),
        "rul_r2":  r2_score(y_rul_test, rul_pred_test),
        "rul_interval_coverage": float(np.mean(
            (y_rul_test.values >= rul_q10_test) & (y_rul_test.values <= rul_q90_test)
        )),
    }
    new_bundle["test_data"] = {
        "X_test": X.iloc[test_idx], "y_soh_test": y_soh_test, "y_rul_test": y_rul_test,
        "soh_pred": soh_pred_test, "rul_pred": rul_pred_test,
        "rul_q10": rul_q10_test, "rul_q90": rul_q90_test,
    }
    new_bundle["n_estimators_total"] = new_bundle["soh_model"].n_estimators
    new_bundle["n_estimators_added"] = n_new_estimators
    return new_bundle


def warm_start_vs_full_retrain_drift(incremental_bundle: dict, full_bundle: dict) -> dict:
    """
    Compare a warm_start_update() result against a fresh train_models()
    full refit on the same cumulative data, so a caller can tell whether
    the faster incremental path is trustworthy enough to serve or whether
    this update should fall back to a full refit. Mirrors
    src/experiment_registry.py's replay_run() hyperparams_match/
    hyperparams_diff pattern — additive, never silently swaps one for the
    other, just makes the divergence visible.

    Thresholds are deliberately conservative (2 percentage points of SOH
    MAE, 15% relative RUL MAE) — a silently-drifted RUL number is exactly
    the kind of quiet failure this platform's honest-validation philosophy
    exists to catch, so acceptable=False errs toward "fall back to a full
    refit" rather than toward trusting the cheaper path.
    """
    inc, full = incremental_bundle["metrics"], full_bundle["metrics"]
    soh_mae_diff = abs(inc["soh_mae"] - full["soh_mae"])
    rul_mae_diff = abs(inc["rul_mae"] - full["rul_mae"])
    rul_mae_relative_diff = rul_mae_diff / max(full["rul_mae"], 1e-6)
    acceptable = soh_mae_diff <= 2.0 and rul_mae_relative_diff <= 0.15
    return {
        "acceptable": acceptable,
        "soh_mae_diff": soh_mae_diff,
        "rul_mae_diff": rul_mae_diff,
        "rul_mae_relative_diff": rul_mae_relative_diff,
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
    influential.

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
