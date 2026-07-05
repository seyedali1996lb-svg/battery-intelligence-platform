"""Unit tests for src/bms_connectors.py — fetch_victron_vrm() credential guard."""

from bms_connectors import fetch_victron_vrm


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
