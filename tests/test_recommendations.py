"""Unit tests for src/recommendations.py's diagnose_mechanism()."""

import numpy as np
import pandas as pd
from recommendations import diagnose_mechanism


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
