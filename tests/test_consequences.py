"""Unit tests for src/consequences.py's eol_r_code_recommendation()."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from consequences import eol_r_code_recommendation, application_fit


def test_high_soh_yields_r0_reuse():
    result = eol_r_code_recommendation(soh=95.0, fade_30_mah_cy=0.0)
    assert result["r_code"].startswith("R0")
    assert result["best_app"] is None


def test_second_life_band_with_good_fit_yields_r3():
    """SOH in the second-life window with a slow, healthy fade rate should
    find at least one application_fit() app scored fit/marginal -- R3."""
    result = eol_r_code_recommendation(soh=78.0, fade_30_mah_cy=0.05)
    assert result["r_code"].startswith("R3")
    assert result["best_app"] is not None


def test_second_life_band_with_no_fit_yields_r4_not_r3():
    """A cell whose SOH is below every application's grace-adjusted floor
    scores "not_fit" everywhere in application_fit() -- confirms
    eol_r_code_recommendation() is genuinely fit-driven (reads
    application_fit()'s real output) rather than re-deriving its own SOH
    cutoff independently, which is the actual bug this fix closes: two
    parallel derivations of the same taxonomy that could silently
    disagree. The assertion on `fit` itself proves this isn't a vacuous
    test -- application_fit() really does return not_fit for every app at
    this SOH."""
    fit = application_fit(65.0, 0.0, fleet_fade_median=None)
    assert all(v["fit"] == "not_fit" for v in fit.values()), \
        "test setup assumption broken -- expected every app to score not_fit at SOH=65"
    result = eol_r_code_recommendation(soh=65.0, fade_30_mah_cy=0.0)
    assert result["r_code"].startswith("R4")


def test_low_soh_with_no_fit_yields_r5():
    result = eol_r_code_recommendation(soh=40.0, fade_30_mah_cy=0.0)
    assert result["r_code"].startswith("R5")


def test_boundary_soh_90_is_r0_not_r3():
    """>= 90 is the R0 cutoff, matching the Passport's own R0 field text
    ("SOH >= 90% required")."""
    result = eol_r_code_recommendation(soh=90.0, fade_30_mah_cy=0.0)
    assert result["r_code"].startswith("R0")
