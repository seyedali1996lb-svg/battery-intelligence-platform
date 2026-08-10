"""Unit tests for src/bankability_report.py's build_bankability_report()."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from bankability_report import build_bankability_report


def _base_kwargs(**overrides):
    kwargs = dict(
        cell_id="B0005", source="nasa", chemistry="LiCoO2", soh=80.0,
        fade_30_mah_cy=0.001, fleet_fade_median=0.0009, cycle_count=100,
        n_lco_cells=4, lco_soh_r2=0.8, rul_reliable=True,
        rul_q10=150.0, rul_pred=200.0, rul_q90=250.0,
        mechanism={"verdict": "LLI", "confidence": "Medium"},
    )
    kwargs.update(overrides)
    return kwargs


def test_returns_expected_top_level_shape():
    report = build_bankability_report(**_base_kwargs())
    for key in ("cell_id", "identity", "condition", "second_life", "financial", "provenance", "disclaimer"):
        assert key in report
    assert report["cell_id"] == "B0005"


def test_disclaimer_states_not_investment_advice():
    report = build_bankability_report(**_base_kwargs())
    assert "NOT INVESTMENT ADVICE" in report["disclaimer"]
    assert "NOT" in report["disclaimer"] and "GUARANTEE" in report["disclaimer"]


def test_reliable_rul_shown_as_available():
    report = build_bankability_report(**_base_kwargs())
    rul_field = next(f for f in report["condition"] if "Remaining Useful Life" in f["label"])
    assert rul_field["state"] == "available"
    assert "200" in rul_field["value"]


def test_unreliable_rul_shown_as_unavailable_not_fabricated():
    report = build_bankability_report(**_base_kwargs(rul_reliable=False, rul_pred=None, rul_q10=None, rul_q90=None))
    rul_field = next(f for f in report["condition"] if "Remaining Useful Life" in f["label"])
    assert rul_field["state"] == "unavailable"
    assert "not reliable" in rul_field["value"].lower()


def test_second_life_section_reuses_application_fit_not_reinvented():
    report = build_bankability_report(**_base_kwargs())
    labels = [f["label"] for f in report["second_life"]]
    assert "Best-fit second-life application" in labels
    assert "End-of-life recommendation (IEC 62902)" in labels


def test_financial_fields_computed_internally_from_assumptions():
    """financial_comparison() is now computed inside build_bankability_report()
    itself (reused, not passed in) -- confirms the 3 expected fields are
    present with real dollar values, not a caller-supplied stub."""
    report = build_bankability_report(**_base_kwargs())
    labels = [f["label"] for f in report["financial"]]
    assert any("Second-life reuse value" in l for l in labels)
    assert any("recycle value" in l.lower() for l in labels)
    for f in report["financial"]:
        assert f["value"].startswith("$")


def test_assumptions_override_changes_financial_figures():
    """Passing an assumptions override (same convention as
    src/spine_export.py's build_second_life_export()) must actually change
    the computed financial figures, confirming the override path is real."""
    default_report = build_bankability_report(**_base_kwargs())
    overridden_report = build_bankability_report(**_base_kwargs(assumptions={"new_cell_cost": 999.0}))
    default_val = next(f["value"] for f in default_report["financial"] if "replacement cost" in f["label"].lower())
    overridden_val = next(f["value"] for f in overridden_report["financial"] if "replacement cost" in f["label"].lower())
    assert default_val != overridden_val
    assert "999.00" in overridden_val


def test_provenance_reflects_missing_run_id_honestly():
    report = build_bankability_report(**_base_kwargs(experiment_run_id=None, git_commit=None))
    run_field = next(f for f in report["provenance"] if "run ID" in f["label"])
    assert run_field["state"] == "unavailable"
    assert run_field["value"] == "Not available"


def test_provenance_reflects_present_run_id():
    report = build_bankability_report(**_base_kwargs(experiment_run_id="run-123", git_commit="abc1234"))
    run_field = next(f for f in report["provenance"] if "run ID" in f["label"])
    assert run_field["state"] == "available"
    assert run_field["value"] == "run-123"


def test_sop_pct_included_when_given():
    report = build_bankability_report(**_base_kwargs(sop_pct=90.0))
    assert any("Power" in f["label"] for f in report["condition"])
