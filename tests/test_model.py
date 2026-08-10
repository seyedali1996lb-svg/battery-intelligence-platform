"""Unit tests for batlab.models.gbrt — train_models() / predict() / feature importance."""

import pytest
from conftest import make_cycles_df
from batlab.features.engineering import build_features, get_model_matrix
from batlab.models.gbrt import (
    train_models, predict, feature_importance_df, top_drivers,
    warm_start_update, warm_start_vs_full_retrain_drift, WARM_START_MAX_ESTIMATORS,
)


@pytest.fixture(scope="module")
def bundle():
    df = build_features(make_cycles_df(n_cycles=300))
    X, y_soh, y_rul = get_model_matrix(df)
    return train_models(X, y_soh, y_rul), X


def test_train_models_raises_on_empty_data():
    import pandas as pd
    with pytest.raises(ValueError):
        train_models(pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float))


def test_train_models_returns_expected_bundle_shape(bundle):
    bndl, _ = bundle
    for key in ["soh_model", "rul_model", "rul_q10_model", "rul_q90_model",
                "scaler", "feature_names", "metrics", "test_data"]:
        assert key in bndl
    for metric in ["soh_mae", "soh_r2", "rul_mae", "rul_r2", "rul_interval_coverage"]:
        assert metric in bndl["metrics"]


def test_predict_returns_calibrating_tag_for_early_cycles(bundle):
    bndl, X = bundle
    preds = predict(bndl, X)
    assert len(preds["soh_pred"]) == len(X)
    assert len(preds["confidence_tag"]) == len(X)
    early_mask = X["cycle_number"] < 50
    if early_mask.any():
        early_tags = [t for t, is_early in zip(preds["confidence_tag"], early_mask) if is_early]
        assert all(t == "Calibrating" for t in early_tags)


def test_predict_rul_never_negative(bundle):
    bndl, X = bundle
    preds = predict(bndl, X)
    assert (preds["rul_pred"] >= 0).all()
    assert (preds["rul_q10"] >= 0).all()
    assert (preds["rul_q90"] >= 0).all()


def test_feature_importance_sums_to_100pct(bundle):
    bndl, _ = bundle
    df = feature_importance_df(bndl, model="soh")
    assert abs(df["importance_pct"].sum() - 100.0) < 0.5
    assert list(df.columns) == ["feature", "importance", "importance_pct"]


def test_top_drivers_respects_top_n(bundle):
    bndl, _ = bundle
    drivers = top_drivers(bndl, model="soh", top_n=3)
    assert len(drivers) == 3
    assert drivers[0]["importance_pct"] >= drivers[-1]["importance_pct"]


# ---------------------------------------------------------------------------
# warm_start_update() / warm_start_vs_full_retrain_drift() — incremental
# refit path, distinct from train_models()'s always-from-scratch fit.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def new_batch():
    """A second, later batch of the same synthetic cell's data — stands in
    for 'new telemetry arrived' without needing two different cells (which
    would also change feature distributions for reasons unrelated to what's
    being tested here)."""
    df = build_features(make_cycles_df(n_cycles=300, fade_per_cycle=0.0007))
    return get_model_matrix(df)


def test_warm_start_update_adds_estimators_without_losing_existing_ones(bundle, new_batch):
    bndl, _ = bundle
    X2, y_soh2, y_rul2 = new_batch
    base_n = bndl["soh_model"].n_estimators
    updated = warm_start_update(bndl, X2, y_soh2, y_rul2, n_new_estimators=20)
    assert updated["soh_model"].n_estimators == base_n + 20
    assert updated["n_estimators_added"] == 20
    # Original bundle must be untouched -- warm_start_update() must not
    # mutate its input in place (deepcopy contract).
    assert bndl["soh_model"].n_estimators == base_n


def test_warm_start_update_returns_valid_metrics_shape(bundle, new_batch):
    bndl, _ = bundle
    X2, y_soh2, y_rul2 = new_batch
    updated = warm_start_update(bndl, X2, y_soh2, y_rul2)
    for metric in ["soh_mae", "soh_r2", "rul_mae", "rul_r2", "rul_interval_coverage"]:
        assert metric in updated["metrics"]


def test_warm_start_update_rejects_mismatched_feature_columns(bundle):
    bndl, _ = bundle
    import pandas as pd
    bad_X = pd.DataFrame({"not_a_real_feature": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="feature_names"):
        warm_start_update(bndl, bad_X, pd.Series([90.0, 89.0, 88.0]), pd.Series([100, 90, 80]))


def test_warm_start_update_refuses_past_max_estimators(bundle, new_batch):
    bndl, _ = bundle
    X2, y_soh2, y_rul2 = new_batch
    with pytest.raises(ValueError, match="WARM_START_MAX_ESTIMATORS"):
        warm_start_update(
            bndl, X2, y_soh2, y_rul2,
            n_new_estimators=WARM_START_MAX_ESTIMATORS,  # base (200) + this always exceeds the cap
        )


def test_drift_check_accepts_incremental_bundle_close_to_full_retrain(bundle, new_batch):
    bndl, _ = bundle
    X2, y_soh2, y_rul2 = new_batch
    incremental = warm_start_update(bndl, X2, y_soh2, y_rul2)
    full = train_models(X2, y_soh2, y_rul2)
    drift = warm_start_vs_full_retrain_drift(incremental, full)
    assert "acceptable" in drift
    assert isinstance(drift["acceptable"], bool)
    assert drift["soh_mae_diff"] >= 0
    assert drift["rul_mae_relative_diff"] >= 0


def test_drift_check_rejects_large_divergence():
    """Hand-built bundles with deliberately far-apart metrics should be
    flagged unacceptable -- confirms the check actually discriminates,
    not just always returning True."""
    incremental = {"metrics": {"soh_mae": 10.0, "rul_mae": 500.0}}
    full = {"metrics": {"soh_mae": 1.0, "rul_mae": 50.0}}
    drift = warm_start_vs_full_retrain_drift(incremental, full)
    assert drift["acceptable"] is False
