"""Unit tests for batlab.features.engineering — build_features() / get_model_matrix()."""

import numpy as np
from conftest import make_cycles_df
from batlab.features.engineering import build_features, get_model_matrix, FEATURE_COLUMNS


def test_build_features_adds_expected_columns():
    df = build_features(make_cycles_df())
    for col in ["fade_rate_10cy", "fade_rate_30cy", "fade_rate_50cy",
                "fade_acceleration", "soh_velocity_50cy", "resistance_normalized",
                "resistance_trend_30cy", "temp_rolling_30cy", "rul",
                "dqdv_sim_peak_value", "dqdv_sim_area", "dqdv_sim_fwhm", "cumulative_ah"]:
        assert col in df.columns, f"missing column: {col}"


def test_rul_decreases_toward_eol():
    df = build_features(make_cycles_df(n_cycles=400, fade_per_cycle=0.001))
    # A cell that's monotonically fading should show RUL trending down as
    # cycle_number increases (allow some rolling-window noise near the start).
    early_rul = df["rul"].iloc[100]
    late_rul  = df["rul"].iloc[300]
    assert late_rul < early_rul


def test_resistance_normalized_handles_zero_first_cycle():
    """Regression test: Severson cells store resistance_ohm=0.0 on cycle 1 as a
    missing-data marker. Using it as the reference would divide by zero and
    produce all-inf resistance_normalized (see README debugging story)."""
    df = build_features(make_cycles_df(first_resistance_is_zero=True))
    assert np.isfinite(df["resistance_normalized"]).all()
    assert not (df["resistance_normalized"] == 0).all()


def test_missing_optional_columns_degrade_gracefully():
    """Severson/uploaded cells arrive without coulombic_efficiency, c_rate,
    r_sei/r_ct — build_features() must not crash, just produce NaN columns."""
    df = make_cycles_df().drop(columns=["temperature_c"])
    out = build_features(df)
    assert out["temp_rolling_30cy"].isna().all()
    assert out["ce_rolling_30cy"].isna().all()
    assert out["stress_index"].isna().all()


def test_is_power_limited_flags_when_resistance_has_grown_enough():
    """sop_pct (peak-power capability, ∝ 1/R) drops below the 70% power-fade
    floor once resistance has roughly grown ~43%+ over its initial value —
    is_power_limited should flip True at that point, mirroring is_eol's
    80%-capacity-floor pattern but for power instead of capacity."""
    df = build_features(make_cycles_df(n_cycles=400, fade_per_cycle=0.001, resistance_rise_per_cycle=0.0002))
    assert df["is_power_limited"].dtype == bool
    assert not df["is_power_limited"].iloc[0]
    assert df["is_power_limited"].iloc[-1]


def test_is_power_limited_false_without_resistance_data():
    df = make_cycles_df().drop(columns=["resistance_ohm"])
    out = build_features(df)
    assert out["sop_pct"].isna().all()
    assert not out["is_power_limited"].any()


def test_usage_profile_ev_like_when_fast_and_variable():
    """Block-alternating C-rate (long stretches of 0.3C then 2.1C) survives
    the 10-cycle rolling-mean smoothing that a continuous sine wave at this
    amplitude/period would get averaged away by -- closer to a real EV duty
    cycle (a day of driving vs. a day parked) than a smooth sinusoid anyway."""
    df_raw = make_cycles_df(n_cycles=200)
    block = np.concatenate([np.full(15, 0.3), np.full(15, 2.1)])
    c_rate = np.tile(block, 200 // len(block) + 1)[:200]
    df_raw = df_raw.assign(c_rate=c_rate)
    out = build_features(df_raw)
    assert out["usage_profile"].iloc[-1] == "EV-like"
    assert out["usage_profile_code"].iloc[-1] == 2.0
    assert "usage_profile_code" in FEATURE_COLUMNS


def test_usage_profile_stationary_like_when_slow_and_steady():
    df_raw = make_cycles_df(n_cycles=200).assign(c_rate=np.full(200, 0.3))
    out = build_features(df_raw)
    assert out["usage_profile"].iloc[-1] == "Stationary-like"
    assert out["usage_profile_code"].iloc[-1] == 0.0


def test_usage_profile_absent_without_c_rate_data():
    out = build_features(make_cycles_df(n_cycles=100))
    assert out["usage_profile_code"].isna().all()
    assert out["usage_profile"].isna().all()


def test_get_model_matrix_drops_allnan_and_incomplete_rows():
    df = build_features(make_cycles_df())
    X, y_soh, y_rul = get_model_matrix(df)
    assert len(X) == len(y_soh) == len(y_rul)
    assert len(X) > 0
    # Columns that were entirely NaN (no temperature_c/c_rate in this fixture's
    # case they ARE present) should still only include available, non-all-NaN cols
    assert set(X.columns).issubset(set(FEATURE_COLUMNS))
    assert not X.isna().any().any()
