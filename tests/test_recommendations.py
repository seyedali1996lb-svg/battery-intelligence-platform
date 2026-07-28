"""Unit tests for src/recommendations.py's diagnose_mechanism()."""

import numpy as np
import pandas as pd
from recommendations import diagnose_mechanism, mechanism_corroboration_note, physics_ml_agreement_note


def _make_cell_df(n=100, ce_slope=0.0, ce_base=0.999, nonlinearity=0.0,
                   fade_total=10.0, resistance_slope=0.0, columns=None):
    """Synthetic featured-DataFrame with the 3 columns diagnose_mechanism()
    reads: cycle_number, coulombic_efficiency, soh_pct, resistance_normalized.
    `columns` restricts which of those are present (defaults to all)."""
    cycles = np.arange(1, n + 1, dtype=float)
    cy_norm = (cycles - cycles.min()) / (cycles.max() - cycles.min())

    ce = ce_base + ce_slope * cycles
    # soh: linear term for total fade, quadratic term for nonlinearity
    soh = 100.0 - fade_total * cy_norm + nonlinearity * (cy_norm ** 2) * 10
    resistance = 1.0 + resistance_slope * cycles / 1000.0

    data = {
        "cycle_number": cycles,
        "coulombic_efficiency": ce,
        "soh_pct": soh,
        "resistance_normalized": resistance,
    }
    df = pd.DataFrame(data)
    if columns is not None:
        keep = ["cycle_number"] + list(columns)
        df = df[keep]
    return df


def test_declining_ce_and_linear_fade_yields_lli_verdict():
    df = _make_cell_df(ce_slope=-0.0005, nonlinearity=0.0, resistance_slope=0.005)
    result = diagnose_mechanism(df)
    assert "LLI" in result["verdict"]
    assert result["lli_score"] > result["lam_score"]


def test_accelerating_fade_and_fast_resistance_rise_yields_lam_verdict():
    df = _make_cell_df(ce_slope=0.0, nonlinearity=-5.0, resistance_slope=0.05)
    result = diagnose_mechanism(df)
    assert "LAM" in result["verdict"]
    assert result["lam_score"] > result["lli_score"]


def test_no_signals_available_yields_insufficient_data():
    df = pd.DataFrame({"cycle_number": np.arange(1, 5, dtype=float)})
    result = diagnose_mechanism(df)
    assert result["verdict"] == "Insufficient data"
    assert result["confidence_label"] == "No data"
    assert result["confidence_notes"] == []


def test_confidence_label_reflects_number_of_available_signals():
    df = _make_cell_df(columns=["coulombic_efficiency"])
    result = diagnose_mechanism(df)
    assert result["confidence_label"] == "Low"
    assert result["confidence_notes"] == ["CE trend"]


def test_all_three_signals_present_yields_high_confidence():
    df = _make_cell_df(ce_slope=-0.0003, nonlinearity=-2.0, resistance_slope=0.03)
    result = diagnose_mechanism(df)
    assert result["confidence_label"] == "High"
    assert set(result["confidence_notes"]) == {"CE trend", "fade shape", "resistance"}


def test_returned_dict_has_all_expected_keys():
    df = _make_cell_df()
    result = diagnose_mechanism(df)
    for key in ("verdict", "verdict_color", "verdict_icon", "verdict_body",
                "confidence_label", "confidence_color", "confidence_notes",
                "lli_score", "lam_score", "signals"):
        assert key in result


# ---------------------------------------------------------------------------
# mechanism_corroboration_note() -- Decision Support review finding:
# classify() never sees the mechanism classifier's verdict, so a
# lower-urgency action ("continue"/"inspect") could be recommended for a
# cell whose mechanism verdict shows an accelerating, higher-risk pattern
# (LAM) with no arbitration between the two surfaces. This is the additive
# caution-note fix the user chose over deeper (and riskier) changes to
# classify() itself.
# ---------------------------------------------------------------------------

def _mech(verdict, confidence_label):
    return {"verdict": verdict, "confidence_label": confidence_label}


def test_no_note_when_mechanism_agrees_with_continue():
    note = mechanism_corroboration_note("continue", _mech("LLI — Loss of Lithium Inventory", "High"))
    assert note is None


def test_caution_note_when_lam_detected_with_continue_action():
    note = mechanism_corroboration_note("continue", _mech("LAM — Loss of Active Material", "High"))
    assert note is not None
    assert "LAM" in note
    assert "elevated caution" in note


def test_caution_note_when_mixed_mechanism_with_inspect_action():
    note = mechanism_corroboration_note("inspect", _mech("Mixed LLI + LAM", "Medium"))
    assert note is not None


def test_no_note_when_action_already_urgent():
    """Second-life/recycle already reflect an urgent posture -- no need to
    caution the user that degradation is accelerating when the
    recommendation already treats the cell as needing action."""
    assert mechanism_corroboration_note("second_life", _mech("LAM — Loss of Active Material", "High")) is None
    assert mechanism_corroboration_note("recycle", _mech("LAM — Loss of Active Material", "High")) is None


def test_no_note_when_mechanism_confidence_too_low():
    """A Low/No-data mechanism verdict isn't reliable enough to second-guess
    the recommendation over."""
    assert mechanism_corroboration_note("continue", _mech("LAM — Loss of Active Material", "Low")) is None
    assert mechanism_corroboration_note("continue", _mech("LAM — Loss of Active Material", "No data")) is None


# ---------------------------------------------------------------------------
# physics_ml_agreement_note() -- same caution-note contract as
# mechanism_corroboration_note() above, wrapping
# src/physics_calibration.py's physics_ml_agreement() diagnostic (Phase 6
# physics-informed intelligence).
# ---------------------------------------------------------------------------

def test_no_note_when_physics_and_ml_agree():
    assert physics_ml_agreement_note({"agree": True}) is None


def test_no_note_when_comparison_undecidable():
    """agree=None means insufficient data on one or both sides -- not a
    disagreement worth surfacing."""
    assert physics_ml_agreement_note({"agree": None}) is None


def test_caution_note_when_physics_and_ml_disagree():
    agreement = {
        "agree": False,
        "physics": {"dominant_mode_label": "LAM — Loss of Active Material (physics: linear fade channel)"},
        "ml": {"verdict": "LLI — Loss of Lithium Inventory", "confidence_label": "High"},
    }
    note = physics_ml_agreement_note(agreement)
    assert note is not None
    assert "LAM" in note and "LLI" in note
    assert "elevated caution" in note
