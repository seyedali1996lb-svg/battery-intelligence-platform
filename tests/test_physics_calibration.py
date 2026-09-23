"""Unit tests for batlab.features.physics_calibration.

These target the LIBRARY module (it moved out of the demo app's src/ in
battery-lab 0.2.0 — src/physics_calibration.py is now only a re-export shim
that registers the app's mechanism classifier).

Eligibility reads the frame's own schema attrs (df.attrs["source"] /
["chemistry"], both REQUIRED_ATTRS of the batlab schema) against
ANCHOR_PARAM_SETS — not a cell-id prefix rule — so a hand-built fixture must
say what it is. That is what makes the allow-list unable to widen by
accident: a synthetic LiCoO₂ cell is *not* eligible even though its chemistry
matches a NASA cell's, which is asserted below.

PyBaMM is not installed in this dev environment, so every test either
exercises the pure-scipy fit logic directly (no PyBaMM involved at all) or
monkeypatches pc._spm_nominal_capacity_ah to verify the caching contract
without a real PyBaMM install. calibrate_cell() degrading to
spm_capacity_ah=None when PyBaMM is genuinely unavailable is itself asserted,
not worked around.
"""

import pathlib
import sys

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
import numpy as np
import pandas as pd
import pytest

from conftest import make_cycles_df
from batlab.features import physics_calibration as pc

NASA_ID = "B0005"
SEVERSON_ID = "S-cell1"
SYNTH_ID = "Cell1"
UPLOADED_ID = "MyUploadedCell"


def _attributed(df: pd.DataFrame, source: str, chemistry: str) -> pd.DataFrame:
    """A fixture frame that declares its provenance, as a real loader's does."""
    df.attrs["source"] = source
    df.attrs["chemistry"] = chemistry
    return df


def _nasa_df(**kwargs) -> pd.DataFrame:
    return _attributed(make_cycles_df(**kwargs), "nasa", "LiCoO2")


def _severson_df(**kwargs) -> pd.DataFrame:
    return _attributed(make_cycles_df(**kwargs), "severson2019", "LFP")


def _synthetic_df(**kwargs) -> pd.DataFrame:
    # Deliberately the SAME chemistry as the NASA cells: eligibility must key on
    # the (source, chemistry) pair, or the synthetic fleet would inherit
    # physics features it was never validated with.
    return _attributed(make_cycles_df(**kwargs), "synthetic", "LiCoO2")


@pytest.fixture(autouse=True)
def _clear_cache():
    pc.reset_nominal_capacity_cache()
    yield
    pc.reset_nominal_capacity_cache()


# ---------------------------------------------------------------------------
# Eligibility gating
# ---------------------------------------------------------------------------

def test_eligibility_is_the_anchor_allow_list():
    assert pc._eligible_for_calibration(_nasa_df(n_cycles=200)) is True
    assert pc._eligible_for_calibration(_severson_df(n_cycles=200)) is True
    # Same chemistry as NASA, different source -> NOT eligible.
    assert pc._eligible_for_calibration(_synthetic_df(n_cycles=200)) is False
    # Unknown source, eligible chemistry -> NOT eligible.
    assert pc._eligible_for_calibration(_attributed(make_cycles_df(n_cycles=60), "uploaded", "LFP")) is False
    # Eligible source, unaccounted chemistry -> NOT eligible.
    assert pc._eligible_for_calibration(_attributed(make_cycles_df(n_cycles=60), "nasa", "NCA")) is False
    # A frame that declares nothing is not calibrated (never guessed from a cell_id).
    assert pc._eligible_for_calibration(make_cycles_df(n_cycles=60)) is False
    assert pc._eligible_for_calibration(make_cycles_df(n_cycles=60), NASA_ID) is False


def test_param_set_resolution_is_by_anchor_not_by_cell_id():
    assert pc._param_set_for_cell(_nasa_df(n_cycles=200)) == "NCA_Kim2011"
    assert pc._param_set_for_cell(_severson_df(n_cycles=200)) == "Chen2020"
    assert pc._param_set_for_cell(_synthetic_df(n_cycles=200)) is None


def test_a_registered_anchor_extends_eligibility_without_widening_it():
    """The escape hatch for a caller's own fleet, scoped to what it vouches for."""
    df = _attributed(make_cycles_df(n_cycles=200), "open_ldr", "NMC")
    assert pc._eligible_for_calibration(df) is False

    pc.register_anchor_param_set(
        lambda source, chemistry, cell_id: "Marquis2019" if (source, chemistry) == ("open_ldr", "NMC") else None
    )
    try:
        assert pc._eligible_for_calibration(df) is True
        assert pc._param_set_for_cell(df) == "Marquis2019"
        # Still nothing else: the resolver is consulted, not obeyed blindly.
        assert pc._eligible_for_calibration(_synthetic_df(n_cycles=200)) is False
    finally:
        pc.register_anchor_param_set(None)


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
    df = _synthetic_df(n_cycles=200)
    result = pc.calibrate_cell(SYNTH_ID, df)
    assert result["eligible"] is False
    assert result["error"] is not None
    assert result["beta_sei"] is None
    # The reason names what the frame actually declared, so the disclosure is
    # actionable ("which chemistry do I have to register?") rather than blank.
    assert "synthetic" in result["error"] and "LiCoO2" in result["error"]


def test_calibrate_cell_too_few_cycles():
    df = _nasa_df(n_cycles=5)
    result = pc.calibrate_cell(NASA_ID, df)
    assert result["eligible"] is True
    assert result["error"] is not None
    assert "Insufficient" in result["error"]


def test_calibrate_cell_eligible_populates_all_fields():
    """spm_capacity_ah is a real PyBaMM SPM discharge result when PyBaMM is
    installed (float) or None if it's unavailable/fails — either is a valid
    degradation, but every scipy-derived field must always populate."""
    df = _nasa_df(n_cycles=300, fade_per_cycle=0.001)
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
    def _broken_spm(param_set_name):
        raise RuntimeError("simulated PyBaMM failure")

    monkeypatch.setattr(pc, "_spm_nominal_capacity_ah", _broken_spm)
    df = _nasa_df(n_cycles=300, fade_per_cycle=0.001)
    result = pc.calibrate_cell(NASA_ID, df)
    assert result["eligible"] is True
    assert result["error"] is None
    assert result["spm_capacity_ah"] is None
    assert result["beta_sei"] is not None


def test_calibrate_cell_no_resistance_column_leaves_k_r_none():
    df = _severson_df(n_cycles=200).drop(columns=["resistance_ohm"])
    result = pc.calibrate_cell(SEVERSON_ID, df)
    assert result["eligible"] is True
    assert result["k_r"] is None


# ---------------------------------------------------------------------------
# _nominal_capacity_ah caching — per param_set, not per cell
# ---------------------------------------------------------------------------

def test_nominal_capacity_cached_per_param_set(monkeypatch):
    calls = []

    def _fake_spm(param_set_name):
        calls.append(param_set_name)
        return 2.0

    monkeypatch.setattr(pc, "_spm_nominal_capacity_ah", _fake_spm)

    assert pc._nominal_capacity_ah("NCA_Kim2011") == 2.0
    assert pc._nominal_capacity_ah("NCA_Kim2011") == 2.0
    assert pc._nominal_capacity_ah("Chen2020") == 2.0
    assert calls == ["NCA_Kim2011", "Chen2020"]  # one call per distinct param_set, not per invocation


def test_calibrated_feature_series_reuses_cache_across_cells(monkeypatch):
    calls = []

    def _fake_spm(param_set_name):
        calls.append(param_set_name)
        return 2.0

    monkeypatch.setattr(pc, "_spm_nominal_capacity_ah", _fake_spm)

    df_a = _nasa_df(n_cycles=100)
    df_b = _nasa_df(n_cycles=120)
    pc.calibrated_feature_series(df_a, "B0005")
    pc.calibrated_feature_series(df_b, "B0006")
    assert calls == ["NCA_Kim2011"]  # second NASA cell reused the cached discharge


# ---------------------------------------------------------------------------
# calibrated_feature_series — causal, no future leakage
# ---------------------------------------------------------------------------

def test_feature_series_all_nan_for_ineligible_cell():
    df = _synthetic_df(n_cycles=200)
    out = pc.calibrated_feature_series(df, SYNTH_ID)
    assert list(out.columns) == pc.PHYSICS_FEATURE_COLUMNS
    assert out.isna().all().all()


def test_feature_series_all_nan_when_cell_id_none():
    df = _nasa_df(n_cycles=200)
    out = pc.calibrated_feature_series(df, None)
    assert out.isna().all().all()


def test_feature_series_nan_before_min_cycles_then_populated():
    df = _nasa_df(n_cycles=200)
    out = pc.calibrated_feature_series(df, NASA_ID)
    assert out["physics_beta_sei"].iloc[: pc.MIN_CYCLES_FOR_CALIBRATION - 1].isna().all()
    assert out["physics_beta_sei"].iloc[-1] == out["physics_beta_sei"].iloc[-1]  # not NaN


def test_feature_series_is_causal_no_future_leakage():
    """Two cells identical up through cycle 100, diverging sharply after —
    physics features at row 100 must be identical between them, since a
    causal (expanding-window, no-future-peek) fit cannot see the divergence
    that only happens afterward."""
    shared = _nasa_df(n_cycles=100, fade_per_cycle=0.0008)
    tail_mild = _nasa_df(n_cycles=200, fade_per_cycle=0.0008)
    tail_aggr = _nasa_df(n_cycles=200, fade_per_cycle=0.005)

    # concat does not carry attrs reliably -> re-declare after joining.
    df_mild = _attributed(pd.concat([shared, tail_mild.iloc[100:]], ignore_index=True), "nasa", "LiCoO2")
    df_aggr = _attributed(pd.concat([shared, tail_aggr.iloc[100:]], ignore_index=True), "nasa", "LiCoO2")
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

def test_physics_ml_agreement_says_so_when_no_classifier_is_registered():
    """The library never reaches into the demo app for a classifier.

    With nothing registered it reports no comparison — rather than silently
    degrading to "agrees" or importing whatever `recommendations` it finds on
    sys.path, which is the defect this module was moved out of the app to fix.
    """
    pc.register_mechanism_classifier(None)
    df = _nasa_df(n_cycles=300, fade_per_cycle=0.0015)
    result = pc.physics_ml_agreement(NASA_ID, df)
    assert result["ml"] is None and result["agree"] is None
    assert "register_mechanism_classifier" in result["note"]
    assert result["physics"]["eligible"] is True  # the physics half still ran


def test_physics_ml_agreement_ineligible_cell():
    pc.register_mechanism_classifier(lambda _df: {"verdict": "Mixed LLI + LAM"})
    try:
        result = pc.physics_ml_agreement(SYNTH_ID, _synthetic_df(n_cycles=200))
    finally:
        pc.register_mechanism_classifier(None)
    assert result["agree"] is None
    assert result["physics"]["error"]  # the ineligibility reason becomes the note


def test_physics_ml_agreement_returns_structured_result():
    df = _nasa_df(n_cycles=300, fade_per_cycle=0.0015, resistance_rise_per_cycle=0.0003)
    # The classifier's own result shape (recommendations.diagnose_mechanism's).
    pc.register_mechanism_classifier(
        lambda _df: {"verdict": "LLI — Loss of Lithium Inventory", "confidence_label": "high"}
    )
    try:
        result = pc.physics_ml_agreement(NASA_ID, df)
    finally:
        pc.register_mechanism_classifier(None)
    assert "physics" in result and "ml" in result
    assert result["agree"] in (True, False, None)
    assert isinstance(result["note"], str) and len(result["note"]) > 0


def test_the_application_shim_reexports_and_registers_the_apps_classifier():
    """src/physics_calibration.py is a shim now, not a second implementation.

    Every app surface still does `from physics_calibration import ...`; that must
    resolve to the library's function objects (one implementation) and the shim's
    import must be what registers the app's classifier.
    """
    import importlib
    import pathlib
    import sys as _sys

    _root = pathlib.Path(__file__).resolve().parent.parent
    if str(_root / "src") not in _sys.path:
        _sys.path.insert(0, str(_root / "src"))
    import physics_calibration as shim  # noqa: E402

    assert shim.calibrate_cell is pc.calibrate_cell
    assert shim.calibrated_feature_series is pc.calibrated_feature_series
    assert set(shim.__all__) == set(pc.__all__)
    # Registration is an import SIDE EFFECT, so it can only be observed while
    # the module body actually runs — and a plain `import` re-runs nothing once
    # anything has imported this module first (this file's own fit test does,
    # and `cell_scene` reads `R2_FLOOR` from it), while the agreement test above
    # has quite correctly unregistered its stub in a `finally`. Re-executing
    # the body — exactly what a first import runs — pins the contract whatever
    # order the suite happens to collect files in, instead of passing only when
    # this file happens to be the first to touch the shim.
    importlib.reload(shim)
    assert pc.get_mechanism_classifier() is not None
    # A shim, not a fork: if a copy of the fit logic ever reappears here, this
    # module has grown a second implementation of a library feature again.
    shim_source = pathlib.Path(shim.__file__).read_text(encoding="utf-8")
    for name in ("def calibrate_cell", "def calibrated_feature_series", "def fit_two_term_fade"):
        assert name not in shim_source


# ---------------------------------------------------------------------------
# physics_gbrt_divergence_report — held-out-cell validation
# ---------------------------------------------------------------------------

def test_divergence_report_skips_ineligible_cells():
    cell_data = {
        SYNTH_ID: _synthetic_df(n_cycles=200),
        "Cell2": _synthetic_df(n_cycles=200, fade_per_cycle=0.001),
    }
    report = pc.physics_gbrt_divergence_report(cell_data)
    assert report == []  # neither cell is NASA/Severson


def test_divergence_report_structure_for_eligible_cells():
    cell_data = {
        "B0005": _nasa_df(n_cycles=200, fade_per_cycle=0.0008, resistance_rise_per_cycle=0.00004),
        "B0006": _nasa_df(n_cycles=220, fade_per_cycle=0.0012, resistance_rise_per_cycle=0.00006),
        "B0007": _nasa_df(n_cycles=180, fade_per_cycle=0.0009, resistance_rise_per_cycle=0.00003),
    }
    report = pc.physics_gbrt_divergence_report(cell_data)
    assert len(report) == 3
    reported_ids = {r["cell_id"] for r in report}
    assert reported_ids == {"B0005", "B0006", "B0007"}
    for r in report:
        assert r["gbrt_soh_mae"] >= 0
        assert r["physics_soh_mae"] >= 0
        assert r["closer_model"] in ("physics", "gbrt", "comparable")
        assert "not measuring the same thing" in r["note"]


def test_divergence_report_ignores_ineligible_cells_when_mixed_with_eligible():
    cell_data = {
        "B0005": _nasa_df(n_cycles=200, fade_per_cycle=0.001, resistance_rise_per_cycle=0.00005),
        "B0006": _nasa_df(n_cycles=200, fade_per_cycle=0.0015, resistance_rise_per_cycle=0.0001),
        SYNTH_ID: _synthetic_df(n_cycles=200, fade_per_cycle=0.002),
    }
    report = pc.physics_gbrt_divergence_report(cell_data)
    reported_ids = {r["cell_id"] for r in report}
    assert reported_ids == {"B0005", "B0006"}
    assert SYNTH_ID not in reported_ids
