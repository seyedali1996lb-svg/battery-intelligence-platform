"""Unit tests for battery_copilot.answer_cell_story() — the Decide & Ask
page's single-cell synthesized narrative (Data Storytelling review finding,
the last item left open in the 2026-07-30 self-audit)."""

from battery_copilot import answer_cell_story


def _base_story(**overrides):
    story = {
        "cell_id": "S-b1c2",
        "soh": 78.5,
        "action_label": "Schedule Inspection",
        "confidence_label": "High confidence",
        "mechanism": {
            "verdict": "LLI — Loss of Lithium Inventory",
            "confidence_label": "Medium",
        },
        "fit_scores": {
            "stationary": {"name": "Stationary Storage", "fit": "fit"},
            "ups":        {"name": "UPS Backup", "fit": "not_fit"},
        },
        "financial_best_label": "Wait to EOL",
        "financial_best_npv": 120.0,
        "financial_disagrees": False,
        "npv_max_label": "Wait to EOL",
        "npv_max_value": 120.0,
    }
    story.update(overrides)
    return story


def test_opening_sentence_states_cell_soh_and_action():
    text = answer_cell_story(_base_story())
    assert "S-b1c2" in text
    assert "78.5% SOH" in text
    assert "Schedule Inspection" in text
    assert "High confidence" in text


def test_mechanism_sentence_included_when_available():
    text = answer_cell_story(_base_story())
    assert "LLI — Loss of Lithium Inventory" in text
    assert "Medium confidence" in text


def test_mechanism_sentence_omitted_when_insufficient_data():
    story = _base_story(mechanism={"verdict": "Insufficient data", "confidence_label": "No data"})
    text = answer_cell_story(story)
    assert "Insufficient data" not in text


def test_mechanism_sentence_omitted_when_none():
    text = answer_cell_story(_base_story(mechanism=None))
    assert "Degradation is driven" not in text


def test_fit_sentence_names_the_fit_app():
    text = answer_cell_story(_base_story())
    assert "Stationary Storage" in text
    assert "UPS Backup" not in text  # not_fit app shouldn't be named as a good fit


def test_fit_sentence_falls_back_to_marginal_when_no_fit():
    story = _base_story(fit_scores={
        "stationary": {"name": "Stationary Storage", "fit": "marginal"},
        "ups":        {"name": "UPS Backup", "fit": "not_fit"},
    })
    text = answer_cell_story(story)
    assert "marginal fit" in text
    assert "Stationary Storage" in text


def test_fit_sentence_when_nothing_fits():
    story = _base_story(fit_scores={
        "stationary": {"name": "Stationary Storage", "fit": "not_fit"},
        "ups":        {"name": "UPS Backup", "fit": "not_fit"},
    })
    text = answer_cell_story(story)
    assert "does not fit any tracked second-life application" in text


def test_fit_sentence_omitted_when_empty():
    text = answer_cell_story(_base_story(fit_scores={}))
    assert "second-life" not in text


def test_financial_agreement_sentence():
    text = answer_cell_story(_base_story())
    assert "recommendations agree" in text


def test_financial_disagreement_sentence_names_both_options():
    story = _base_story(
        financial_best_label="Wait to EOL", financial_best_npv=50.0,
        financial_disagrees=True,
        npv_max_label="Repurpose (2nd life)", npv_max_value=200.0,
    )
    text = answer_cell_story(story)
    assert "Repurpose (2nd life)" in text
    assert "$200" in text
    assert "Compare alternatives" in text


def test_never_raises_on_minimal_story():
    """Every optional field missing -- should degrade gracefully, not crash."""
    minimal = {
        "cell_id": "X1", "soh": 95.0,
        "action_label": "Continue Operation", "confidence_label": "High",
    }
    text = answer_cell_story(minimal)
    assert "X1" in text
    assert "95.0% SOH" in text
