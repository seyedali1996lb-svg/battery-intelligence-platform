"""Unit tests for src/us_ira_export.py's field-structure demonstration export."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from us_ira_export import build_ira_30d_entry, to_ira_30d_document, PREFIX


def test_entry_has_correct_type_and_id():
    entry = build_ira_30d_entry("B0005", "LiCoO2", 85.0, recycled_in_north_america_pathway_available=False)
    assert entry["type"] == "us_ira_section_30d_traceability"
    assert entry["id"] == "B0005"


def test_regulatory_thresholds_are_available_not_unavailable():
    """The 70% thresholds are regulatory constants this platform can cite
    regardless of per-cell data -- must be 'available', not 'unavailable'."""
    entry = build_ira_30d_entry("B0005", "LiCoO2", 85.0, recycled_in_north_america_pathway_available=False)
    attrs = entry["attributes"]
    assert attrs[f"{PREFIX}critical_minerals_percentage_required_2026"]["state"] == "available"
    assert attrs[f"{PREFIX}critical_minerals_percentage_required_2026"]["value"] == 70
    assert attrs[f"{PREFIX}battery_components_percentage_required_2026"]["value"] == 70


def test_supply_chain_dependent_fields_are_unavailable():
    """No fabricated percentages -- every field requiring real supply-chain
    data must be explicitly unavailable with value None."""
    entry = build_ira_30d_entry("B0005", "LiCoO2", 85.0, recycled_in_north_america_pathway_available=False)
    attrs = entry["attributes"]
    for key in ("critical_minerals_percentage_actual", "battery_components_percentage_actual", "feoc_compliant"):
        assert attrs[f"{PREFIX}{key}"]["state"] == "unavailable"
        assert attrs[f"{PREFIX}{key}"]["value"] is None


def test_recycled_in_north_america_pathway_reflects_input():
    entry_true = build_ira_30d_entry("B0005", "LiCoO2", 85.0, recycled_in_north_america_pathway_available=True)
    entry_false = build_ira_30d_entry("B0005", "LiCoO2", 85.0, recycled_in_north_america_pathway_available=False)
    key = f"{PREFIX}recycled_in_north_america_pathway_available"
    assert entry_true["attributes"][key]["value"] is True
    assert entry_false["attributes"][key]["value"] is False
    assert entry_true["attributes"][key]["state"] == "estimated"


def test_document_has_non_compliance_disclaimer():
    entry = build_ira_30d_entry("B0005", "LiCoO2", 85.0, recycled_in_north_america_pathway_available=False)
    doc = to_ira_30d_document(entry, "B0005")
    assert "NOT a compliance claim" in doc["meta"]["disclaimer"]
    assert doc["data"] is entry
    assert doc["meta"]["cell_id"] == "B0005"
