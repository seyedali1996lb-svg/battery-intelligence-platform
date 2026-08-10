"""Unit tests for src/optimade_export.py — build_battery_cell_entry() and
to_optimade_document()."""

import json

from optimade_export import build_battery_cell_entry, to_optimade_document, ENTRY_TYPE, PREFIX


def _sample_mechanism():
    return {"verdict": "LLI-dominant", "verdict_body": "...", "confidence": "high"}


def test_entry_has_confirmed_jsonapi_resource_shape():
    entry = build_battery_cell_entry("TestCell", "nasa", "LiCoO2", soh=82.0)
    assert entry["type"] == ENTRY_TYPE
    assert entry["id"] == "TestCell"
    assert isinstance(entry["attributes"], dict)


def test_every_attribute_name_carries_the_provider_prefix():
    entry = build_battery_cell_entry(
        "TestCell", "nasa", "LiCoO2", soh=82.0, capacity_ah=2.0,
        rul_reliable=True, rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
        mechanism=_sample_mechanism(),
        condition_completeness={"score": 0.8, "caveats": ["a known gap"]},
    )
    assert all(name.startswith(PREFIX) for name in entry["attributes"])


def test_core_fields_always_present():
    entry = build_battery_cell_entry("TestCell", "severson2019", "LFP", soh=91.5)
    attrs = entry["attributes"]
    assert attrs[f"{PREFIX}chemistry"] == "LFP"
    assert attrs[f"{PREFIX}soh_pct"] == 91.5
    assert attrs[f"{PREFIX}source_dataset"] == "severson2019"


def test_rul_written_only_when_reliable_and_present():
    reliable = build_battery_cell_entry(
        "TestCell", "nasa", "LiCoO2", soh=82.0,
        rul_reliable=True, rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
    )
    attrs = reliable["attributes"]
    assert attrs[f"{PREFIX}rul_cycles_p10"] == 200.0
    assert attrs[f"{PREFIX}rul_cycles_p50"] == 300.0
    assert attrs[f"{PREFIX}rul_cycles_p90"] == 400.0

    unreliable = build_battery_cell_entry(
        "TestCell", "nasa", "LiCoO2", soh=82.0,
        rul_reliable=False, rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
    )
    assert not any(k.startswith(f"{PREFIX}rul_cycles") for k in unreliable["attributes"])


def test_mechanism_and_condition_completeness_only_written_when_provided():
    bare = build_battery_cell_entry("TestCell", "nasa", "LiCoO2", soh=82.0)
    assert f"{PREFIX}degradation_mechanism" not in bare["attributes"]
    assert f"{PREFIX}condition_completeness_score" not in bare["attributes"]

    enriched = build_battery_cell_entry(
        "TestCell", "nasa", "LiCoO2", soh=82.0,
        mechanism=_sample_mechanism(),
        condition_completeness={"score": 1.0, "caveats": []},
    )
    assert enriched["attributes"][f"{PREFIX}degradation_mechanism"] == "LLI-dominant"
    assert enriched["attributes"][f"{PREFIX}condition_completeness_score"] == 1.0
    # No caveats -> no caveats key written (honest omission, not an empty list)
    assert f"{PREFIX}condition_completeness_caveats" not in enriched["attributes"]


def test_condition_completeness_caveats_written_when_present():
    entry = build_battery_cell_entry(
        "TestCell", "severson2019", "LFP", soh=82.0,
        condition_completeness={"score": 0.8, "caveats": ["multi-step charge policy"]},
    )
    assert entry["attributes"][f"{PREFIX}condition_completeness_caveats"] == ["multi-step charge policy"]


def test_entry_is_json_serializable():
    entry = build_battery_cell_entry(
        "TestCell", "nasa", "LiCoO2", soh=82.0, capacity_ah=2.0,
        rul_reliable=True, rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
        mechanism=_sample_mechanism(),
        condition_completeness={"score": 1.0, "caveats": []},
    )
    serialized = json.dumps(entry)
    assert json.loads(serialized) == entry


def test_to_optimade_document_wraps_data_and_meta():
    entry = build_battery_cell_entry("TestCell", "nasa", "LiCoO2", soh=82.0)
    doc = to_optimade_document(entry, "TestCell")
    assert doc["data"] == entry
    assert doc["meta"]["cell_id"] == "TestCell"
    assert "disclaimer" in doc["meta"]
    json.dumps(doc)  # must round-trip as JSON too
