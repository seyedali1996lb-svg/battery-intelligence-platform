"""Unit tests for src/passport_export.py — to_json_ld() and document_id()."""

import pandas as pd
from passport import build_passport
from passport_export import to_json_ld, document_id


def _sample_passport():
    df = pd.DataFrame({
        "soh_pct": [100, 99, 98, 97], "cycle_number": [1, 2, 3, 4],
        "resistance_ohm": [0.05, 0.051, 0.052, 0.053], "fade_rate_30cy": [0.001] * 4,
        "capacity_ah": [2.0, 1.98, 1.96, 1.94], "rul_pred": [None, None, None, 600.0],
    })
    bundle = {"metrics": {"lco_soh_r2": 0.9}}
    return build_passport("TestCell", df, bundle, rul_reliable=True)


def test_json_ld_has_required_top_level_keys():
    ld = to_json_ld(_sample_passport(), "TestCell")
    for key in ["@context", "@type", "@id", "identifier", "identity",
                "stateOfHealth", "lifecycle", "carbonFootprint", "completeness"]:
        assert key in ld
    assert ld["@id"] == "urn:battery-passport:TestCell"
    assert ld["identifier"] == "TestCell"


def test_json_ld_preserves_field_counts_and_provenance():
    p = _sample_passport()
    ld = to_json_ld(p, "TestCell")
    assert len(ld["identity"]) == len(p["identity"])
    assert len(ld["stateOfHealth"]) == len(p["soh"])
    assert ld["completeness"]["total"] == p["summary"]["n_total"]
    # Every entry must carry over its available/estimated/unavailable tag
    for entry in ld["identity"]:
        assert entry["provenance"] in ("available", "estimated", "unavailable")


def test_json_ld_is_actually_json_serializable():
    import json
    ld = to_json_ld(_sample_passport(), "TestCell")
    serialized = json.dumps(ld)
    assert json.loads(serialized) == ld


def test_document_id_is_deterministic_for_same_inputs():
    p = _sample_passport()
    d1 = document_id(p, "TestCell", "2026-01-01T00:00:00")
    d2 = document_id(p, "TestCell", "2026-01-01T00:00:00")
    assert d1 == d2
    assert len(d1) == 12


def test_document_id_differs_for_different_timestamp():
    p = _sample_passport()
    d1 = document_id(p, "TestCell", "2026-01-01T00:00:00")
    d2 = document_id(p, "TestCell", "2026-01-02T00:00:00")
    assert d1 != d2


def test_json_ld_reuses_provided_doc_id():
    p = _sample_passport()
    fixed_id = document_id(p, "TestCell", "2026-01-01T00:00:00")
    ld = to_json_ld(p, "TestCell", doc_id=fixed_id)
    assert ld["documentId"] == fixed_id
