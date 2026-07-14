"""Unit tests for batlab.features.dqdv — dQ/dV curve simulation and feature extraction."""

import numpy as np
import pandas as pd
from batlab.features.dqdv import simulate_vq_curve, extract_dqdv_features, add_dqdv_features


def test_simulate_vq_curve_shapes_and_bounds():
    Q, V = simulate_vq_curve(capacity_ah=2.0, resistance_ohm=0.05, n_points=200)
    assert Q.shape == (200,)
    assert V.shape == (200,)
    assert Q[0] == 0.0
    assert abs(Q[-1] - 2.0) < 1e-9
    assert np.isfinite(V).all()


def test_extract_dqdv_features_keys_and_types():
    feats = extract_dqdv_features(capacity_ah=2.0, resistance_ohm=0.05)
    for key in ["dqdv_sim_peak_value", "dqdv_sim_peak_soc", "dqdv_sim_area", "dqdv_sim_fwhm"]:
        assert key in feats
        assert isinstance(feats[key], float)
    assert 0.0 <= feats["dqdv_sim_peak_soc"] <= 1.0
    assert feats["dqdv_sim_fwhm"] >= 0.0


def test_add_dqdv_features_matches_per_row_extraction():
    """The vectorized add_dqdv_features() must agree with calling
    extract_dqdv_features() row-by-row (this vectorization was a real perf
    fix — see README debugging story — so a correctness regression here
    would be easy to introduce silently)."""
    df = pd.DataFrame({
        "capacity_ah": [2.0, 1.9, 1.8],
        "resistance_ohm": [0.05, 0.052, 0.055],
    })
    vectorized = add_dqdv_features(df)

    for i in range(len(df)):
        single = extract_dqdv_features(df["capacity_ah"].iloc[i], df["resistance_ohm"].iloc[i])
        assert abs(vectorized["dqdv_sim_peak_value"].iloc[i] - single["dqdv_sim_peak_value"]) < 1e-6
        assert abs(vectorized["dqdv_sim_area"].iloc[i] - single["dqdv_sim_area"]) < 1e-6
        assert abs(vectorized["dqdv_sim_fwhm"].iloc[i] - single["dqdv_sim_fwhm"]) < 1e-6


def test_add_dqdv_features_adds_all_columns():
    df = pd.DataFrame({"capacity_ah": [2.0, 1.9], "resistance_ohm": [0.05, 0.06]})
    out = add_dqdv_features(df)
    for col in ["dqdv_sim_peak_value", "dqdv_sim_peak_soc", "dqdv_sim_area", "dqdv_sim_fwhm"]:
        assert col in out.columns
        assert out[col].notna().all()
