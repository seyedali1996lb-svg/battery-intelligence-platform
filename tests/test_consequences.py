"""Unit tests for src/consequences.py's eol_r_code_recommendation() and
best_fit_application()."""

import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from consequences import eol_r_code_recommendation, application_fit, best_fit_application


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


# ---------------------------------------------------------------------------
# best_fit_application() -- the single ranking helper that replaced 5
# independently re-typed copies of the same {"fit": 2, "marginal": 1,
# "not_fit": 0} + max() selection across consequences.py, compliance.py,
# _fleet_diagnostics.py, and recommendations.py (x2).
# ---------------------------------------------------------------------------

def test_best_fit_application_picks_fit_over_marginal_and_not_fit():
    fit_scores = {
        "a": {"fit": "not_fit"},
        "b": {"fit": "marginal"},
        "c": {"fit": "fit"},
    }
    key, best = best_fit_application(fit_scores)
    assert key == "c"
    assert best["fit"] == "fit"


def test_best_fit_application_picks_marginal_over_not_fit():
    fit_scores = {
        "a": {"fit": "not_fit"},
        "b": {"fit": "marginal"},
    }
    key, best = best_fit_application(fit_scores)
    assert key == "b"


def test_best_fit_application_tie_keeps_first_insertion_order():
    """Ties should resolve to the first-inserted key -- matches every prior
    caller's behaviour before this was unified (Python's max() and a
    reverse-sorted list's [0] both pick the first-encountered maximal item
    when the input order is the same)."""
    fit_scores = {
        "first":  {"fit": "fit"},
        "second": {"fit": "fit"},
    }
    key, _ = best_fit_application(fit_scores)
    assert key == "first"


def test_best_fit_application_matches_real_application_fit_output():
    """Sanity check against application_fit()'s real dict shape, not just a
    hand-rolled stub."""
    fit = application_fit(78.0, 0.05, fleet_fade_median=None)
    key, best = best_fit_application(fit)
    assert key in fit
    assert best is fit[key]


# ---------------------------------------------------------------------------
# application_fit()'s optional sop_pct (State-of-Power) parameter — additive
# only: absent, every app's power_status stays "ok" and behaves exactly as
# before. Present, it only affects apps flagged requires_power (ups_backup).
# ---------------------------------------------------------------------------

def test_sop_pct_omitted_leaves_power_status_ok_for_every_app():
    fit = application_fit(80.0, 0.05, fleet_fade_median=None)
    assert all(v["power_status"] == "ok" for v in fit.values())


def test_healthy_soh_but_low_sop_fails_ups_backup_only():
    """A cell at ups_backup-band SOH with badly degraded peak-power capability
    (resistance grown a lot) should fail specifically the power check for
    the one application that actually needs pulse power -- other apps that
    don't require power shouldn't be affected by a low sop_pct at all."""
    fit = application_fit(80.0, 0.0, fleet_fade_median=None, sop_pct=50.0)
    assert fit["ups_backup"]["power_status"] == "fail"
    assert fit["ups_backup"]["fit"] == "not_fit"
    assert any("power" in r.lower() for r in fit["ups_backup"]["reasons"])
    # residential_ess doesn't require_power -- unaffected by low sop_pct
    assert fit["residential_ess"]["power_status"] == "ok"


def test_marginal_sop_flags_marginal_not_fail():
    fit = application_fit(80.0, 0.0, fleet_fade_median=None, sop_pct=70.0)
    assert fit["ups_backup"]["power_status"] == "marginal"
    assert fit["ups_backup"]["fit"] != "not_fit"


def test_eol_r_code_recommendation_accepts_optional_sop_pct():
    """sop_pct forwards through to application_fit() without breaking the
    existing SOH/fade-only call shape."""
    result_no_sop = eol_r_code_recommendation(soh=78.0, fade_30_mah_cy=0.05)
    result_with_sop = eol_r_code_recommendation(soh=78.0, fade_30_mah_cy=0.05, sop_pct=95.0)
    assert result_no_sop["r_code"] == result_with_sop["r_code"]
