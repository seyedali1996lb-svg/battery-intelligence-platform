"""Unit tests for batlab.models.attribution — occlusion-based local attribution."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from batlab.models.attribution import mean_attribution, occlusion_attribution, top_attributions


def _make_bundle(seed: int = 0):
    """A tiny GBRT where `signal` and `signal2` drive y and `noise` is
    irrelevant — the minimal setup that proves attribution ranks drivers."""
    rng = np.random.default_rng(seed)
    n = 400
    X = pd.DataFrame({
        "signal": rng.uniform(0, 1, n),
        "signal2": rng.uniform(0, 1, n),
        "noise": rng.normal(size=n),
    })
    y = 3.0 * X["signal"] - 2.0 * X["signal2"] + rng.normal(0, 0.05, n)
    model = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=seed)
    scaler = StandardScaler()
    model.fit(scaler.fit_transform(X), y)
    return {"rul_model": model, "scaler": scaler, "feature_names": list(X.columns)}, X


def test_occlusion_attribution_ranks_signal_over_noise():
    bundle, X = _make_bundle()
    attr = occlusion_attribution(bundle, X.head(8), X_ref=X, n_samples=20, seed=1)
    assert attr.shape == (8, 3)
    assert list(attr.columns) == ["signal", "signal2", "noise"]
    assert (attr >= 0).all().all()

    mean = mean_attribution(attr)
    assert mean["signal"] > mean["noise"] * 3   # a real driver moves the prediction far more
    assert mean["signal2"] > mean["noise"] * 2


def test_top_attributions_shape_and_order():
    bundle, X = _make_bundle()
    attr = occlusion_attribution(bundle, X.head(3), X_ref=X, n_samples=10, seed=2)
    tops = top_attributions(attr, top_n=2)
    assert len(tops) == 3
    for row in tops:
        assert len(row) == 2
        assert row[0]["attribution"] >= row[1]["attribution"]
        assert {"feature", "attribution"} == set(row[0].keys())
        assert set(f["feature"] for f in row) <= {"signal", "signal2", "noise"}


def test_occlusion_attribution_deterministic_for_fixed_seed():
    bundle, X = _make_bundle()
    a1 = occlusion_attribution(bundle, X.head(4), X_ref=X, n_samples=15, seed=99)
    a2 = occlusion_attribution(bundle, X.head(4), X_ref=X, n_samples=15, seed=99)
    pd.testing.assert_frame_equal(a1, a2)


def test_occlusion_attribution_guards():
    bundle, X = _make_bundle()
    with pytest.raises(ValueError, match="model"):
        occlusion_attribution(bundle, X, model="nope")
    with pytest.raises(ValueError, match="missing"):
        occlusion_attribution(bundle, X.drop(columns=["noise"]))
