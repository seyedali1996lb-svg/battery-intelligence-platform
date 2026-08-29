"""Unit tests for src/china_recycling_export.py's field-structure demonstration export."""

import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from china_recycling_export import build_china_recycling_entry, to_china_recycling_document, PREFIX


def test_entry_has_correct_type_and_id():
    entry = build_china_recycling_entry("B0006", "LiCoO2", 58.3, "R5 — Recover (energy or material)", recommended_recycler_name="GEM Co.")
    assert entry["type"] == "china_nev_battery_recycling_traceability"
    assert entry["id"] == "B0006"


def test_regulation_reference_is_available():
    entry = build_china_recycling_entry("B0006", "LiCoO2", 58.3, "R5 — Recover (energy or material)", recommended_recycler_name=None)
    attrs = entry["attributes"]
    assert attrs[f"{PREFIX}regulation_reference"]["state"] == "available"
    assert "2026-04-01" in attrs[f"{PREFIX}regulation_reference"]["value"]


def test_platform_registration_fields_are_unavailable():
    entry = build_china_recycling_entry("B0006", "LiCoO2", 58.3, "R5 — Recover (energy or material)", recommended_recycler_name=None)
    attrs = entry["attributes"]
    for key in ("gbt_34014_2017_coding_registered", "digital_id_registered", "epr_responsible_party"):
        assert attrs[f"{PREFIX}{key}"]["state"] == "unavailable"
        assert attrs[f"{PREFIX}{key}"]["value"] is None


def test_r_code_reused_directly_not_recomputed():
    entry = build_china_recycling_entry("B0006", "LiCoO2", 58.3, "R4 — Recycle (hydrometallurgical / direct)", recommended_recycler_name="GEM Co.")
    assert entry["attributes"][f"{PREFIX}recommended_recycling_pathway"]["value"] == "R4 — Recycle (hydrometallurgical / direct)"
    assert entry["attributes"][f"{PREFIX}recommended_recycling_pathway"]["state"] == "estimated"


def test_recycler_name_none_yields_unavailable_state():
    entry = build_china_recycling_entry("B0005", "LiCoO2", 90.0, "R0 — Reuse (primary life)", recommended_recycler_name=None)
    field = entry["attributes"][f"{PREFIX}recommended_recycler"]
    assert field["value"] is None
    assert field["state"] == "unavailable"


def test_recycler_name_present_yields_estimated_state():
    entry = build_china_recycling_entry("B0006", "LiCoO2", 58.3, "R5 — Recover (energy or material)", recommended_recycler_name="GEM Co.")
    field = entry["attributes"][f"{PREFIX}recommended_recycler"]
    assert field["value"] == "GEM Co."
    assert field["state"] == "estimated"


def test_document_has_non_registration_disclaimer():
    entry = build_china_recycling_entry("B0006", "LiCoO2", 58.3, "R5 — Recover (energy or material)", recommended_recycler_name="GEM Co.")
    doc = to_china_recycling_document(entry, "B0006")
    assert "NOT a real registration" in doc["meta"]["disclaimer"]
    assert doc["data"] is entry
