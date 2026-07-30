"""Unit tests for battery_copilot.answer_business_case() — the Copilot's
"business case" chip-click question.

Fixed to read from real classify()/financial_comparison() results instead
of its own independent 90%/80% SOH thresholds and flat replacement-cost
model, which could disagree with the real Decide & Ask page for the same
cell. See its docstring for the full bug history.
"""

from battery_copilot import answer_business_case


def _ctx(cell_id="B0006", soh=78.0):
    return {"cell_id": cell_id, "soh": soh}


def _result(action="inspect", reasons=None):
    return {
        "action": action,
        "action_reasons": reasons or [f"SOH is in the inspection band."],
        "confidence": "medium",
    }


def _financials():
    return {
        "cell_kwh": 0.0072, "current_kwh": 0.0056,
        "sl_gross": 60.0, "sl_net": 45.0,
        "recycle_value": 1.5, "new_cell_cost": 150.0, "repack_cost": 15.0,
    }


def _assumptions():
    return {
        "recycling_value": 1.5, "new_cell_cost": 150.0,
        "second_life_value_per_kwh": 500.0, "repack_cost": 15.0,
        "co2_manufacture": 8.0, "material_recovery": 0.6,
    }


def test_headline_uses_real_action_label():
    text = answer_business_case(_ctx(), _result(action="continue"), _financials(), _assumptions())
    assert "Continue Operation" in text


def test_headline_matches_recycle_action():
    text = answer_business_case(_ctx(), _result(action="recycle"), _financials(), _assumptions())
    assert "Recycle" in text


def test_headline_matches_second_life_action():
    text = answer_business_case(_ctx(), _result(action="second_life"), _financials(), _assumptions())
    assert "Route to Second-Life" in text


def test_includes_real_classify_reasons_not_hardcoded_prose():
    reasons = ["Fade rate is 2.3x baseline — above the 2x acceleration threshold."]
    text = answer_business_case(_ctx(), _result(reasons=reasons), _financials(), _assumptions())
    assert reasons[0] in text


def test_includes_real_financial_comparison_numbers():
    text = answer_business_case(_ctx(), _result(), _financials(), _assumptions())
    assert "$150" in text     # new_cell_cost
    assert "$45.00" in text   # sl_net
    assert "$1.50" in text    # recycle_value


def test_financial_section_omitted_gracefully_when_none():
    """A cell whose chemistry has no CELL_NOMINAL_KWH entry (e.g. an
    unspecified user upload) must not crash or fabricate a capacity."""
    text = answer_business_case(_ctx(), _result(), None, _assumptions())
    assert "not available" in text or "isn't specified" in text or "aren't available" in text


def test_uses_real_soh_and_cell_id():
    text = answer_business_case(_ctx(cell_id="S-b1c2", soh=71.4), _result(), _financials(), _assumptions())
    assert "S-b1c2" in text
    assert "71.4% SOH" in text


def test_footer_uses_real_assumption_value_not_hardcoded_150():
    custom_assumptions = {
        "recycling_value": 1.5, "new_cell_cost": 200.0,
        "second_life_value_per_kwh": 500.0, "repack_cost": 15.0,
    }
    text = answer_business_case(_ctx(), _result(), _financials(), custom_assumptions)
    assert "$200" in text
