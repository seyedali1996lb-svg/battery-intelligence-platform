"""Unit tests for src/spine_export.py — build_second_life_export() and
to_export_document()."""

import json

from spine_export import build_second_life_export, to_export_document, ENTITY_CLASS


def _sample_mechanism():
    return {"verdict": "LLI-dominant", "verdict_body": "...", "confidence": "high"}


def test_export_has_required_top_level_keys():
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=True,
        rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
        mechanism=_sample_mechanism(),
    )
    for key in ["entity_classes", "entities", "parameter_definitions", "parameter_values", "alternatives"]:
        assert key in data


def test_entity_class_and_entity_shape():
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=False,
    )
    assert data["entity_classes"][0][0] == ENTITY_CLASS
    assert data["entity_classes"][0][1] == []  # 0-D entity class
    assert data["entities"] == [[ENTITY_CLASS, "TestCell", None]]


def test_parameter_definitions_cover_every_written_parameter():
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=True,
        rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
        mechanism=_sample_mechanism(), run_id="run123", git_commit="abc1234",
    )
    defined_names = {row[1] for row in data["parameter_definitions"]}
    written_names = {row[2] for row in data["parameter_values"]}
    assert written_names.issubset(defined_names)
    # every parameter_definitions row has the confirmed 5-element shape
    for row in data["parameter_definitions"]:
        assert len(row) == 5
        assert row[0] == ENTITY_CLASS


def test_rul_cycles_written_as_alternatives_when_reliable():
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=True,
        rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
    )
    rul_rows = {row[4]: row[3] for row in data["parameter_values"] if row[2] == "rul_cycles"}
    assert rul_rows == {"p10": 200.0, "p50": 300.0, "p90": 400.0}
    alt_names = {row[0] for row in data["alternatives"]}
    assert alt_names == {"p10", "p50", "p90"}


def test_rul_cycles_omitted_when_not_reliable():
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=False,
        rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
    )
    rul_rows = [row for row in data["parameter_values"] if row[2] == "rul_cycles"]
    assert rul_rows == []
    assert data["alternatives"] == []


def test_mechanism_and_provenance_only_written_when_provided():
    data_without = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=False,
    )
    names_without = {row[2] for row in data_without["parameter_values"]}
    assert "degradation_mechanism" not in names_without
    assert "source_experiment_run_id" not in names_without
    assert "source_git_commit" not in names_without

    data_with = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=False,
        mechanism=_sample_mechanism(), run_id="run123", git_commit="abc1234",
    )
    values_with = {row[2]: row[3] for row in data_with["parameter_values"]}
    assert values_with["degradation_mechanism"] == "LLI-dominant"
    assert values_with["source_experiment_run_id"] == "run123"
    assert values_with["source_git_commit"] == "abc1234"


def test_second_life_application_fit_lists_fitting_categories_or_none():
    # SOH 78% with a healthy (below-median) fade rate should fit at least
    # one of the three SECOND_LIFE_APPS bands.
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=78.0, fade_30_mah_cy=0.5,
        fleet_fade_median=1.0, rul_reliable=False,
    )
    fit_row = next(row for row in data["parameter_values"] if row[2] == "second_life_application_fit")
    assert fit_row[3] != ""

    # SOH far outside every band (still in primary life) should honestly say "none".
    data_none = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=99.0, fade_30_mah_cy=0.5,
        fleet_fade_median=1.0, rul_reliable=False,
    )
    fit_row_none = next(row for row in data_none["parameter_values"] if row[2] == "second_life_application_fit")
    assert fit_row_none[3] == "none"


def test_export_is_json_serializable():
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=True,
        rul_q10=200.0, rul_pred=300.0, rul_q90=400.0,
        mechanism=_sample_mechanism(), run_id="run123", git_commit="abc1234",
    )
    serialized = json.dumps(data)
    assert json.loads(serialized) == data


def test_to_export_document_wraps_data_and_metadata():
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=False,
    )
    doc = to_export_document(data, "TestCell")
    assert doc["data"] == data
    assert doc["metadata"]["cell_id"] == "TestCell"
    assert "disclaimer" in doc["metadata"]
    assert "source_condition_completeness" not in doc["metadata"]
    json.dumps(doc)  # must round-trip as JSON too


def test_to_export_document_includes_condition_completeness_when_given():
    data = build_second_life_export(
        "TestCell", "nasa", "LiCoO2", soh=82.0, fade_30_mah_cy=1.2,
        fleet_fade_median=1.0, rul_reliable=False,
    )
    completeness = {"known": {"c_rate": True}, "score": 1.0, "caveats": []}
    doc = to_export_document(data, "TestCell", condition_completeness=completeness)
    assert doc["metadata"]["source_condition_completeness"] == completeness
    json.dumps(doc)
