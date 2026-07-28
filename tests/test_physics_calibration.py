"""Unit tests for src/physics_calibration.py.

PyBaMM is not installed in this dev environment (same as every other test
in this suite that touches src/pybamm_rul.py) — every test here either
exercises the pure-scipy fit logic directly (no PyBaMM involved at all) or
monkeypatches _nominal_capacity_ah / the underlying SPM-discharge helper to
verify the caching contract without a real PyBaMM install. calibrate_cell()
degrading to spm_capacity_ah=None when PyBaMM is genuinely unavailable is
itself asserted, not worked around.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from conftest import make_cycles_df
import physics_calibration as pc

NASA_ID = "B0005"       # matches chemistry_profiles._NASA_CELL_IDS
SEVERSON_ID = "S-cell1"  # matches LFPSeversonProfile's "S-" prefix rule
SYNTH_ID = "Cell1"       # matches LiCoO2SyntheticProfile's "Cell" prefix rule
UPLOADED_ID = "MyUploadedCell"


@pytest.fixture(autouse=True)
def _clear_cache():
    pc.reset_nominal_capacity_cache()
    yield
    pc.reset_nominal_capacity_cache()


# ---------------------------------------------------------------------------
# Eligibility gating
# ---------------------------------------------------------------------------

def test_eligibility_nasa_and_severson_only():
    assert pc._eligible_for_calibration(NASA_ID) is True
    assert pc._eligible_for_calibration(SEVERSON_ID) is True
    assert pc._eligible_for_calibration(SYNTH_ID) is False
    assert pc._eligible_for_calibration(UPLOADED_ID) is False


def test_param_set_resolution():
    assert pc._param_set_for_cell(NASA_ID) == "NCA_Kim2011"
    assert pc._param_set_for_cell(SEVERSON_ID) == "Chen2020"
    assert pc._param_set_for_cell(SYNTH_ID) is None


# ---------------------------------------------------------------------------
# Pure-scipy fits — no PyBaMM needed
# ---------------------------------------------------------------------------

def test_fit_two_term_fade_recovers_sei_dominant_signal():
    n = np.arange(1, 501, dtype=float)
    true_beta_sei = 0.004
    soh = (1.0 - true_beta_sei * np.sqrt(n)) * 100.0
    result = pc.fit_two_term_fade(n, soh)
    assert result["r2"] > 0.9
    assert result["beta_sei"] == pytest.approx(true_beta_sei, rel=0.25)
    assert result["n_cycles_used"] == 500


def test_fit_two_term_fade_recovers_lam_dominant_signal():
    n = np.arange(1, 501, dtype=float)
    true_beta_lam = 0.00015
    soh = (1.0 - true_beta_lam * n) * 100.0
    result = pc.fit_two_term_fade(n, soh)
    assert result["r2"] > 0.9
    # LAM-only signal: fitted linear contribution should dominate the fitted sqrt contribution
    contrib_lam = result["beta_lam"] * n[-1]
    contrib_sei = result["beta_sei"] * np.sqrt(n[-1])
    assert contrib_lam > contrib_sei


def test_fit_two_term_fade_handles_flat_data_without_crashing():
    n = np.arange(1, 30, dtype=float)
    soh = np.full(len(n), 100.0)
    result = pc.fit_two_term_fade(n, soh)
    assert result["beta_sei"] >= 0
    assert result["beta_lam"] >= 0


def test_fit_resistance_growth_recovers_signal():
    n = np.arange(1, 401, dtype=float)
    true_k_r = 0.02
    r = 0.05 * (1.0 + true_k_r * np.sqrt(n))
    result = pc.fit_resistance_growth(n, r)
    assert result is not None
    assert result["k_r"] == pytest.approx(true_k_r, rel=0.25)


def test_fit_resistance_growth_returns_none_when_insufficient_data():
    n = np.arange(1, 5, dtype=float)
    r = np.array([0.05, 0.051, 0.052, 0.053])
    assert pc.fit_resistance_growth(n, r) is None


def test_fit_resistance_growth_ignores_zero_filled_readings():
    n = np.arange(1, 51, dtype=float)
    r = 0.05 * (1.0 + 0.01 * np.sqrt(n))
    r[0] = 0.0  # Severson-style missing-first-cycle marker
    result = pc.fit_resistance_growth(n, r)
    assert result is not None
    assert result["n_cycles_used"] == 49


# ---------------------------------------------------------------------------
# Dominant-mode classification
# ---------------------------------------------------------------------------

def test_dominant_mode_lli_when_sei_dominates():
    key, label = pc.dominant_mode(beta_sei=0.005, beta_lam=0.00001, at_cycle=500, fit_r2=0.9)
    assert key == "lli"
    assert "LLI" in label


def test_dominant_mode_lam_when_lam_dominates():
    key, label = pc.dominant_mode(beta_sei=0.0001, beta_lam=0.001, at_cycle=500, fit_r2=0.9)
    assert key == "lam"
    assert "LAM" in label


def test_dominant_mode_mixed_when_comparable():
    key, _ = pc.dominant_mode(beta_sei=0.002, beta_lam=0.00006, at_cycle=500, fit_r2=0.9)
    assert key == "mixed"


def test_dominant_mode_insufficient_data_below_r2_floor():
    key, label = pc.dominant_mode(beta_sei=0.005, beta_lam=0.00001, at_cycle=500, fit_r2=0.1)
    assert key == "insufficient_data"
    assert label == "Insufficient data"


# ---------------------------------------------------------------------------
# calibrate_cell — full-history single-shot calibration
# ---------------------------------------------------------------------------

def test_calibrate_cell_ineligible_cell_reports_reason_not_crash():
    df = make_cycles_df(n_cycles=200)
    result = pc.calibrate_cell(SYNTH_ID, df)
    assert result["eligible"] is False
    assert result["error"] is not None
    assert result["beta_sei"] is None


def test_calibrate_cell_too_few_cycles():
    df = make_cycles_df(n_cycles=5)
    result = pc.calibrate_cell(NASA_ID, df)
    assert result["eligible"] is True
    assert result["error"] is not None
    assert "Insufficient" in result["error"]


def test_calibrate_cell_eligible_populates_all_fields():
    """spm_capacity_ah is a real PyBaMM SPM discharge result when PyBaMM is
    installed (float) or None if it's unavailable/fails — either is a valid
    degradation, but every scipy-derived field must always populate."""
    df = make_cycles_df(n_cycles=300, fade_per_cycle=0.001)
    result = pc.calibrate_cell(NASA_ID, df)
    assert result["eligible"] is True
    assert result["error"] is None
    assert result["spm_capacity_ah"] is None or result["spm_capacity_ah"] > 0
    assert result["beta_sei"] is not None
    assert result["fit_r2"] is not None
    assert result["k_r"] is not None  # make_cycles_df includes resistance_ohm
    assert result["dominant_mode_key"] in ("lli", "lam", "mixed", "insufficient_data")


def test_calibrate_cell_spm_unavailable_degrades_gracefully(monkeypatch):
    """When PyBaMM genuinely can't produce a result (import failure, solver
    error, anything), spm_capacity_ah must be None and every other field
    must still populate — never let a PyBaMM failure block the scipy fits."""
    import pybamm_rul

    def _broken_spm(param_set):
        raise RuntimeError("simulated PyBaMM failure")

    monkeypatch.setattr(pybamm_rul, "_run_spm_single_cycle", _broken_spm)
    df = make_cycles_df(n_cycles=300, fade_per_cycle=0.001)
    result = pc.calibrate_cell(NASA_ID, df)
    assert result["eligible"] is True
    assert result["error"] is None
    assert result["spm_capacity_ah"] is None
    assert result["beta_sei"] is not None


def test_calibrate_cell_no_resistance_column_leaves_k_r_none():
    df = make_cycles_df(n_cycles=200).drop(columns=["resistance_ohm"])
    result = pc.calibrate_cell(SEVERSON_ID, df)
    assert result["eligible"] is True
    assert result["k_r"] is None


# ---------------------------------------------------------------------------
# _nominal_capacity_ah caching — per param_set, not per cell
# ---------------------------------------------------------------------------

def test_nominal_capacity_cached_per_param_set(monkeypatch):
    import pybamm_rul
    calls = []

    def _fake_spm(param_set):
        calls.append(param_set)
        return 2.0

    monkeypatch.setattr(pybamm_rul, "_run_spm_single_cycle", _fake_spm)

    assert pc._nominal_capacity_ah("NCA_Kim2011") == 2.0
    assert pc._nominal_capacity_ah("NCA_Kim2011") == 2.0
    assert pc._nominal_capacity_ah("Chen2020") == 2.0
    assert calls == ["NCA_Kim2011", "Chen2020"]  # one call per distinct param_set, not per invocation


def test_calibrated_feature_series_reuses_cache_across_cells(monkeypatch):
    import pybamm_rul
    calls = []

    def _fake_spm(param_set):
        calls.append(param_set)
        return 2.0

    monkeypatch.setattr(pybamm_rul, "_run_spm_single_cycle", _fake_spm)

    df_a = make_cycles_df(n_cycles=100)
    df_b = make_cycles_df(n_cycles=120)
    pc.calibrated_feature_series(df_a, "B0005")
    pc.calibrated_feature_series(df_b, "B0006")
    assert calls == ["NCA_Kim2011"]  # second NASA cell reused the cached discharge


# ---------------------------------------------------------------------------
# calibrated_feature_series — causal, no future leakage
# ---------------------------------------------------------------------------

def test_feature_series_all_nan_for_ineligible_cell():
    df = make_cycles_df(n_cycles=200)
    out = pc.calibrated_feature_series(df, SYNTH_ID)
    assert list(out.columns) == pc.PHYSICS_FEATURE_COLUMNS
    assert out.isna().all().all()


def test_feature_series_all_nan_when_cell_id_none():
    df = make_cycles_df(n_cycles=200)
    out = pc.calibrated_feature_series(df, None)
    assert out.isna().all().all()


def test_feature_series_nan_before_min_cycles_then_populated():
    df = make_cycles_df(n_cycles=200)
    out = pc.calibrated_feature_series(df, NASA_ID)
    assert out["physics_beta_sei"].iloc[: pc.MIN_CYCLES_FOR_CALIBRATION - 1].isna().all()
    assert out["physics_beta_sei"].iloc[-1] == out["physics_beta_sei"].iloc[-1]  # not NaN


def test_feature_series_is_causal_no_future_leakage():
    """Two cells identical up through cycle 100, diverging sharply after —
    physics features at row 100 must be identical between them, since a
    causal (expanding-window, no-future-peek) fit cannot see the divergence
    that only happens afterward."""
    shared = make_cycles_df(n_cycles=100, fade_per_cycle=0.0008)
    tail_mild = make_cycles_df(n_cycles=200, fade_per_cycle=0.0008)
    tail_aggr = make_cycles_df(n_cycles=200, fade_per_cycle=0.005)

    df_mild = pd.concat([shared, tail_mild.iloc[100:]], ignore_index=True)
    df_aggr = pd.concat([shared, tail_aggr.iloc[100:]], ignore_index=True)
    df_mild["soh_pct"] = (df_mild["capacity_ah"] / df_mild["capacity_ah"].iloc[0]) * 100.0
    df_aggr["soh_pct"] = (df_aggr["capacity_ah"] / df_aggr["capacity_ah"].iloc[0]) * 100.0

    out_mild = pc.calibrated_feature_series(df_mild, NASA_ID)
    out_aggr = pc.calibrated_feature_series(df_aggr, NASA_ID)

    row_idx = 99  # last row of the shared history — 0-indexed cycle 100
    for col in pc.PHYSICS_FEATURE_COLUMNS:
        a, b = out_mild[col].iloc[row_idx], out_aggr[col].iloc[row_idx]
        if a == a:  # not NaN
            assert a == pytest.approx(b, rel=1e-9), f"{col} leaked future data at row {row_idx}"


# ---------------------------------------------------------------------------
# physics_ml_agreement
# ---------------------------------------------------------------------------

def test_physics_ml_agreement_ineligible_cell():
    df = make_cycles_df(n_cycles=200)
    result = pc.physics_ml_agreement(SYNTH_ID, df)
    assert result["agree"] is None
    assert "NASA and Severson" in result["note"] or "not this cell" in result["note"] or result["physics"]["error"]


def test_physics_ml_agreement_returns_structured_result():
    df = make_cycles_df(n_cycles=300, fade_per_cycle=0.0015, resistance_rise_per_cycle=0.0003)
    result = pc.physics_ml_agreement(NASA_ID, df)
    assert "physics" in result and "ml" in result
    assert result["agree"] in (True, False, None)
    assert isinstance(result["note"], str) and len(result["note"]) > 0
