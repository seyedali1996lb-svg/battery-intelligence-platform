"""Unit tests for batlab.models.gbrt — train_models() / predict() / feature importance."""

import pytest
from conftest import make_cycles_df
from batlab.features.engineering import build_features, get_model_matrix
from batlab.models.gbrt import train_models, predict, feature_importance_df, top_drivers


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
