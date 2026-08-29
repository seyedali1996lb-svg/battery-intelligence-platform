"""Unit tests for src/plugin_registry.py's ADAPTER_REGISTRY."""

import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from plugin_registry import ADAPTER_REGISTRY, AdapterField, AdapterMeta, register_adapter


def test_all_6_bms_adapters_registered():
    assert set(ADAPTER_REGISTRY.keys()) == {
        "victron_vrm", "orion_bms", "modbus_bms", "can_bus_bms", "ocpp", "generic_rest_bms",
    }


def test_pre_existing_5_adapters_not_rendered_via_registry():
    """The 5 hand-written Settings sections keep their own UI -- their
    registry entries are documentation-only, confirming this module
    doesn't quietly try to take over their rendering."""
    for key in ("victron_vrm", "orion_bms", "modbus_bms", "can_bus_bms", "ocpp"):
        assert ADAPTER_REGISTRY[key].rendered_via_registry is False


def test_generic_rest_bms_is_rendered_via_registry():
    assert ADAPTER_REGISTRY["generic_rest_bms"].rendered_via_registry is True


def test_generic_rest_bms_has_real_field_specs():
    meta = ADAPTER_REGISTRY["generic_rest_bms"]
    assert len(meta.fields) == 5
    field_keys = {f.key for f in meta.fields}
    assert "generic_rest_base_url" in field_keys
    assert "generic_rest_field_map" in field_keys


def test_generic_rest_bms_secret_field_is_flagged():
    meta = ADAPTER_REGISTRY["generic_rest_bms"]
    secret_field = next(f for f in meta.fields if f.key == "generic_rest_auth_header_value")
    assert secret_field.secret is True
    assert secret_field.kind == "password"


def test_generic_rest_bms_build_kwargs_maps_values_to_constructor_args():
    meta = ADAPTER_REGISTRY["generic_rest_bms"]
    values = {
        "generic_rest_base_url": "https://api.example.invalid/cells/1",
        "generic_rest_auth_header_name": "Authorization",
        "generic_rest_auth_header_value": "Bearer xyz",
        "generic_rest_cell_id": "CELL-1",
        "generic_rest_field_map": {"capacity_ah": "data.capacity_ah"},
    }
    kwargs = meta.build_kwargs(values)
    adapter = meta.adapter_class(**kwargs)
    assert adapter.base_url == "https://api.example.invalid/cells/1"
    assert adapter.cell_id == "CELL-1"
    assert adapter.field_map == {"capacity_ah": "data.capacity_ah"}
    assert adapter.is_configured()


def test_generic_rest_bms_auth_fields_are_not_required():
    meta = ADAPTER_REGISTRY["generic_rest_bms"]
    auth_name_field = next(f for f in meta.fields if f.key == "generic_rest_auth_header_name")
    auth_value_field = next(f for f in meta.fields if f.key == "generic_rest_auth_header_value")
    assert auth_name_field.required is False
    assert auth_value_field.required is False
    base_url_field = next(f for f in meta.fields if f.key == "generic_rest_base_url")
    assert base_url_field.required is True


def test_register_adapter_rejects_duplicate_key():
    dummy = AdapterMeta(key="victron_vrm", name="dup", kind="bms", description="", adapter_class=object)
    import pytest
    with pytest.raises(ValueError, match="already registered"):
        register_adapter(dummy)


def test_every_adapter_class_is_a_real_importable_class():
    """Confirms adapter_class references are real classes, not typos --
    a registry entry pointing at a nonexistent/misspelled class would
    only surface at actual-use time otherwise."""
    for meta in ADAPTER_REGISTRY.values():
        assert isinstance(meta.adapter_class, type)
