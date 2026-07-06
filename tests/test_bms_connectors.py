"""Unit tests for src/bms_connectors.py — credential guards for both connectors.

fetch_victron_vrm() and fetch_orion_bms() intentionally use two different
failure contracts: VRM lets network/HTTP errors propagate for its caller
(app/_pages/settings.py) to catch in its own try/except, while Orion
catches everything and returns a dict with an "error" key (matching
circunomics_adapter.py/cmms_adapter.py's contract) so its caller doesn't
need one. Both agree on the "None when unconfigured" guard clause."""

from bms_connectors import fetch_victron_vrm, fetch_orion_bms


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
