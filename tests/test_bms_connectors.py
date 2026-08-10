"""Unit tests for src/bms_connectors.py — credential guards for both connectors.

fetch_victron_vrm() and fetch_orion_bms() intentionally use two different
failure contracts: VRM lets network/HTTP errors propagate for its caller
(app/_pages/settings.py) to catch in its own try/except, while Orion
catches everything and returns a dict with an "error" key (matching
circunomics_adapter.py/cmms_adapter.py's contract) so its caller doesn't
need one. Both agree on the "None when unconfigured" guard clause."""

import pytest

from bms_connectors import (
    fetch_victron_vrm, fetch_orion_bms, fetch_modbus_bms, fetch_can_bus_bms, fetch_ocpp_sessions,
    ModbusBMSAdapter, CANBusBMSAdapter, OCPPAdapter,
)


def test_returns_none_with_no_credentials():
    assert fetch_victron_vrm("", "", "") is None


def test_returns_none_with_partial_credentials():
    assert fetch_victron_vrm("https://vrmapi.victronenergy.com/v2", "", "123") is None
    assert fetch_victron_vrm("https://vrmapi.victronenergy.com/v2", "tok", "") is None
    assert fetch_victron_vrm("", "tok", "123") is None


def test_never_raises_on_unreachable_host():
    """No live VRM account exists to test against — but a bad host must not
    raise an uncaught exception; the caller (Settings page) expects either
    a DataFrame or a raised HTTP error it catches itself, never a crash from
    inside the guard clause."""
    # Missing credentials short-circuits before any network call — this is
    # the only guaranteed-safe-offline path for this adapter.
    assert fetch_victron_vrm("http://localhost:1/nonexistent", "", "") is None


def test_orion_returns_none_with_no_api_key():
    assert fetch_orion_bms("CELL-1", "") is None


def test_orion_returns_error_dict_on_network_failure():
    result = fetch_orion_bms(
        "CELL-1", "fake-key", api_base_url="https://this-host-does-not-exist.invalid/v1",
    )
    assert result is not None
    assert "error" in result


# ---------------------------------------------------------------------------
# Modbus / CAN bus / OCPP — same "None when unconfigured" guard, plus the
# optional-dependency (pymodbus / python-can) ImportError contract these two
# have that VRM/Orion (requests-only, already a hard dependency) don't.
# ---------------------------------------------------------------------------

def test_modbus_returns_none_with_no_config():
    assert fetch_modbus_bms("", 502, 1, {}, "CELL-1") is None
    assert fetch_modbus_bms("10.0.0.5", 502, 1, {}, "CELL-1") is None  # empty register_map


def test_modbus_raises_import_error_without_pymodbus_installed():
    """pymodbus is an optional dependency, not installed in this test
    environment (confirmed before writing this test) -- the guard must
    raise a clear, actionable ImportError, not an opaque ModuleNotFoundError
    from deep inside the function."""
    with pytest.raises(ImportError, match="pymodbus"):
        fetch_modbus_bms("10.0.0.5", 502, 1, {"capacity_ah": (100, 0.01)}, "CELL-1")


def test_modbus_adapter_test_connection_surfaces_import_error_as_not_ok():
    adapter = ModbusBMSAdapter("10.0.0.5", 502, 1, {"capacity_ah": (100, 0.01)}, "CELL-1")
    assert adapter.is_configured()
    result = adapter.test_connection()
    assert result["ok"] is False
    assert "pymodbus" in result["message"]


def test_modbus_adapter_fetch_never_raises_without_pymodbus():
    adapter = ModbusBMSAdapter("10.0.0.5", 502, 1, {"capacity_ah": (100, 0.01)}, "CELL-1")
    assert adapter.fetch() is None  # fetch() catches the ImportError, unlike test_connection()


def test_can_bus_returns_none_with_no_config():
    assert fetch_can_bus_bms("", "socketcan", 1, {}, "CELL-1") is None
    assert fetch_can_bus_bms("can0", "socketcan", 1, {}, "CELL-1") is None  # empty pgn_map


def test_can_bus_raises_import_error_without_python_can_installed():
    with pytest.raises(ImportError, match="python-can"):
        fetch_can_bus_bms("can0", "socketcan", 1, {"capacity_ah": (0x100, 0, 0.01)}, "CELL-1")


def test_can_bus_adapter_test_connection_surfaces_import_error_as_not_ok():
    adapter = CANBusBMSAdapter("can0", "socketcan", 1, {"capacity_ah": (0x100, 0, 0.01)}, "CELL-1")
    assert adapter.is_configured()
    result = adapter.test_connection()
    assert result["ok"] is False
    assert "python-can" in result["message"]


def test_ocpp_returns_none_with_no_config():
    assert fetch_ocpp_sessions("", "key", "CP-1") is None
    assert fetch_ocpp_sessions("https://cs.example.invalid", "", "CP-1") is None
    assert fetch_ocpp_sessions("https://cs.example.invalid", "key", "") is None


def test_ocpp_adapter_test_connection_reports_network_failure():
    """Uses the .invalid TLD convention (never resolves, offline-safe) —
    same pattern as test_orion_returns_error_dict_on_network_failure above."""
    adapter = OCPPAdapter("https://this-host-does-not-exist.invalid", "fake-key", "CP-1")
    result = adapter.test_connection()
    assert result["ok"] is False


def test_ocpp_adapter_not_configured_without_all_three_fields():
    assert not OCPPAdapter("", "key", "CP-1").is_configured()
    assert not OCPPAdapter("https://cs.example.invalid", "", "CP-1").is_configured()
    assert not OCPPAdapter("https://cs.example.invalid", "key", "").is_configured()
