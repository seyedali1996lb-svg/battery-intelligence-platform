"""Unit tests for the BMSAdapter protocol and its two conforming wrapper
classes (src/bms_connectors.py). Confirms both existing connectors
(Victron VRM, Orion Jr2) now satisfy one structural contract, and that the
class-based interface never raises even though the underlying free
functions have two deliberately different error contracts (VRM raises,
Orion returns an {"error": ...} dict) -- see tests/test_bms_connectors.py
for the free functions' own direct tests, unchanged by this module."""

import bms_connectors
from bms_connectors import BMSAdapter, OrionBMSAdapter, VictronVRMAdapter


def test_victron_adapter_satisfies_protocol():
    adapter = VictronVRMAdapter("https://vrmapi.victronenergy.com/v2", "tok", "123")
    assert isinstance(adapter, BMSAdapter)


def test_orion_adapter_satisfies_protocol():
    adapter = OrionBMSAdapter("CELL-1", "fake-key")
    assert isinstance(adapter, BMSAdapter)


def test_victron_adapter_is_configured():
    assert VictronVRMAdapter("https://x", "tok", "123").is_configured() is True
    assert VictronVRMAdapter("https://x", "", "123").is_configured() is False
    assert VictronVRMAdapter("", "tok", "123").is_configured() is False


def test_orion_adapter_is_configured():
    assert OrionBMSAdapter("CELL-1", "fake-key").is_configured() is True
    assert OrionBMSAdapter("CELL-1", "").is_configured() is False


def test_victron_adapter_fetch_returns_none_when_unconfigured():
    assert VictronVRMAdapter("", "", "").fetch() is None


def test_orion_adapter_fetch_returns_none_when_unconfigured():
    assert OrionBMSAdapter("CELL-1", "").fetch() is None


def test_victron_adapter_fetch_delegates_to_free_function(monkeypatch):
    calls = {}

    def _fake_fetch(base_url, api_token, installation_id):
        calls["args"] = (base_url, api_token, installation_id)
        import pandas as pd
        return pd.DataFrame([{"cell_id": "VRM-123", "cycle_number": 0}])

    monkeypatch.setattr(bms_connectors, "fetch_victron_vrm", _fake_fetch)
    adapter = VictronVRMAdapter("https://vrmapi.victronenergy.com/v2", "tok", "123")
    df = adapter.fetch()
    assert calls["args"] == ("https://vrmapi.victronenergy.com/v2", "tok", "123")
    assert len(df) == 1


def test_victron_adapter_fetch_catches_raised_exception(monkeypatch):
    """fetch_victron_vrm() lets HTTP errors propagate (its own tested
    contract) -- the class wrapper must catch that and return None instead
    of raising, since that's the new uniform contract BMSAdapter.fetch()
    promises."""
    def _raiser(base_url, api_token, installation_id):
        raise ConnectionError("boom")

    monkeypatch.setattr(bms_connectors, "fetch_victron_vrm", _raiser)
    adapter = VictronVRMAdapter("https://x", "tok", "123")
    assert adapter.fetch() is None


def test_orion_adapter_fetch_normalizes_error_dict_to_none(monkeypatch):
    """fetch_orion_bms() returns {"error": ...} on failure (its own tested
    contract) -- the class wrapper normalizes that into the shared
    None/DataFrame contract rather than leaking a dict where a caller
    expects a DataFrame."""
    monkeypatch.setattr(bms_connectors, "fetch_orion_bms", lambda *a, **k: {"error": "unreachable"})
    adapter = OrionBMSAdapter("CELL-1", "fake-key")
    assert adapter.fetch() is None


def test_victron_test_connection_reports_not_configured():
    result = VictronVRMAdapter("", "", "").test_connection()
    assert result["ok"] is False
    assert result["n_records"] is None


def test_victron_test_connection_reports_failure_without_raising(monkeypatch):
    def _raiser(base_url, api_token, installation_id):
        raise ConnectionError("boom")

    monkeypatch.setattr(bms_connectors, "fetch_victron_vrm", _raiser)
    result = VictronVRMAdapter("https://x", "tok", "123").test_connection()
    assert result["ok"] is False
    assert "boom" in result["message"]


def test_victron_test_connection_reports_success(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        bms_connectors, "fetch_victron_vrm",
        lambda *a, **k: pd.DataFrame([{"cell_id": "x"}, {"cell_id": "x"}]),
    )
    result = VictronVRMAdapter("https://x", "tok", "123").test_connection()
    assert result["ok"] is True
    assert result["n_records"] == 2


def test_orion_test_connection_reports_not_configured():
    result = OrionBMSAdapter("CELL-1", "").test_connection()
    assert result["ok"] is False
    assert result["n_records"] is None


def test_orion_test_connection_reports_failure_from_error_dict(monkeypatch):
    monkeypatch.setattr(bms_connectors, "fetch_orion_bms", lambda *a, **k: {"error": "unreachable"})
    result = OrionBMSAdapter("CELL-1", "fake-key").test_connection()
    assert result["ok"] is False
    assert "unreachable" in result["message"]


def test_orion_test_connection_reports_success(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        bms_connectors, "fetch_orion_bms",
        lambda *a, **k: pd.DataFrame([{"cell_id": "CELL-1"}]),
    )
    result = OrionBMSAdapter("CELL-1", "fake-key").test_connection()
    assert result["ok"] is True
    assert result["n_records"] == 1


def test_adapter_names_are_distinct():
    assert VictronVRMAdapter("", "", "").name == "Victron VRM"
    assert OrionBMSAdapter("", "").name == "Orion Jr2"
