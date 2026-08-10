"""Unit tests for src/marketplace_matching.py's buyer-scoring/ranking."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from marketplace_matching import score_buyer_match, rank_buyers_for_cell


def _buyer(**overrides):
    b = {"id": "b1", "name": "Test Buyer", "application_type": "ups_backup",
         "min_soh_pct": 75.0, "price_per_kwh_usd": 40.0}
    b.update(overrides)
    return b


def test_eligible_when_soh_and_application_fit_both_pass():
    result = score_buyer_match(80.0, 0.0, None, _buyer(min_soh_pct=75.0))
    assert result["eligible"] is True
    assert result["application_fit"] in ("fit", "marginal")
    assert result["meets_buyer_soh_floor"] is True


def test_ineligible_when_below_buyer_soh_floor_even_if_application_fits():
    result = score_buyer_match(76.0, 0.0, None, _buyer(min_soh_pct=85.0))
    assert result["meets_buyer_soh_floor"] is False
    assert result["eligible"] is False
    assert any("below this buyer's stated" in r for r in result["reasons"])


def test_unknown_application_type_is_ineligible_with_clear_reason():
    result = score_buyer_match(80.0, 0.0, None, _buyer(application_type="not_a_real_app"))
    assert result["eligible"] is False
    assert result["application_fit"] is None
    assert "not_a_real_app" in result["reasons"][0]


def test_low_soh_fails_application_fit_even_with_generous_buyer_floor():
    result = score_buyer_match(20.0, 0.0, None, _buyer(min_soh_pct=0.0))
    assert result["eligible"] is False
    assert result["application_fit"] == "not_fit"


def test_rank_buyers_puts_eligible_before_ineligible():
    buyers = [
        _buyer(id="strict", min_soh_pct=95.0),   # ineligible at 80% SOH
        _buyer(id="lenient", min_soh_pct=70.0),  # eligible
    ]
    ranked = rank_buyers_for_cell(80.0, 0.0, None, buyers)
    assert ranked[0]["id"] == "lenient"
    assert ranked[0]["eligible"] is True
    assert ranked[-1]["eligible"] is False


def test_rank_buyers_sorts_eligible_by_price_descending():
    buyers = [
        _buyer(id="low", min_soh_pct=70.0, price_per_kwh_usd=20.0),
        _buyer(id="high", min_soh_pct=70.0, price_per_kwh_usd=60.0),
    ]
    ranked = rank_buyers_for_cell(80.0, 0.0, None, buyers)
    assert [b["id"] for b in ranked] == ["high", "low"]


def test_rank_buyers_handles_empty_list():
    assert rank_buyers_for_cell(80.0, 0.0, None, []) == []
