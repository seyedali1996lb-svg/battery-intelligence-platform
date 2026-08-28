"""
Tests for src.market_data — pluggable market price/carbon adapters.
The synthetic adapter is the fully-tested path; EIA/ENTSO-E request handling
is exercised with monkeypatched HTTP (no network).
"""

import pytest

from src import market_data
from src.market_data import (
    SyntheticMarketAdapter,
    EIAAdapter,
    ENTSOEAdapter,
    to_eur_per_kwh,
    resolve_carbon_intensity,
    get_market_adapter,
    registered_market_adapters,
    register_market_adapter,
    _parse_entsoe_prices,
)
from src.dynamic_circularity import GRID_CARBON_INTENSITY


# ---------------------------------------------------------------------------
# Synthetic adapter
# ---------------------------------------------------------------------------

def test_synthetic_prices_shape_and_units():
    syn = SyntheticMarketAdapter()
    result = syn.fetch_hourly_prices()
    assert result["adapter"] == "Synthetic"
    assert result["unit"] == "EUR/kWh"
    assert len(result["prices"]) == 48
    assert all(p > 0 for p in result["prices"])


def test_synthetic_deterministic_same_seed():
    a = SyntheticMarketAdapter(seed=42).fetch_hourly_prices()["prices"]
    b = SyntheticMarketAdapter(seed=42).fetch_hourly_prices()["prices"]
    assert a == b


def test_synthetic_spike_injection():
    syn = SyntheticMarketAdapter(spike_hour=12, spike_price_eur=0.45)
    prices = syn.fetch_hourly_prices()["prices"]
    assert prices[12] == 0.45
    assert all(p <= 0.45 for p in prices)


def test_synthetic_carbon_intensity_shape():
    syn = SyntheticMarketAdapter()
    result = syn.fetch_carbon_intensity()
    assert result["unit"] == "g CO2e/kWh"
    assert len(result["series"]) == 48
    assert all(v > 0 for v in result["series"])


# ---------------------------------------------------------------------------
# Currency normalization
# ---------------------------------------------------------------------------

def test_to_eur_per_kwh_passthrough_eur():
    result = {"unit": "EUR/kWh", "prices": [0.10, 0.12]}
    out = to_eur_per_kwh(result)
    assert out is not result  # never mutate the caller's dict
    assert out["prices"] == [0.10, 0.12]
    assert "fx_assumption" not in out


def test_to_eur_per_kwh_converts_usd():
    result = {"unit": "USD/kWh", "prices": [0.10, 0.20]}
    out = to_eur_per_kwh(result)
    assert out["unit"] == "EUR/kWh"
    assert out["prices"] == [round(p * market_data.USD_TO_EUR, 5) for p in [0.10, 0.20]]
    assert out["fx_assumption"]["usd_to_eur"] == market_data.USD_TO_EUR


def test_to_eur_per_kwh_unknown_unit_raises():
    with pytest.raises(ValueError):
        to_eur_per_kwh({"unit": "GBP/kWh", "prices": [1.0]})


# ---------------------------------------------------------------------------
# EIA adapter
# ---------------------------------------------------------------------------

def test_eia_unconfigured_returns_none():
    assert EIAAdapter("").fetch_hourly_prices() is None
    assert EIAAdapter("").fetch_carbon_intensity() is None
    assert not EIAAdapter("").is_configured()


def test_eia_fetch_error_never_raises(monkeypatch):
    def fake_get(url, params, timeout):
        raise ConnectionError("boom")
    monkeypatch.setattr(market_data.requests, "get", fake_get)
    result = EIAAdapter("key").fetch_hourly_prices()
    assert isinstance(result, dict) and "error" in result


def test_eia_parses_documented_shape(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": {"data": [
                {"period": "2026-08-01T00", "value": 40.0},  # $/MWh
                {"period": "2026-08-01T01", "value": 55.0},
            ]}}

    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: FakeResp())
    result = EIAAdapter("key", respondent="PJM").fetch_hourly_prices()
    assert result["unit"] == "USD/kWh"
    assert result["prices"] == [0.04, 0.055]
    assert result["respondent"] == "PJM"


def test_eia_carbon_explicitly_unsupported():
    result = EIAAdapter("key").fetch_carbon_intensity()
    assert "error" in result


# ---------------------------------------------------------------------------
# ENTSO-E adapter
# ---------------------------------------------------------------------------

SAMPLE_ENTSOE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
  <TimeSeries>
    <Period>
      <Point><position>1</position><price.amount>45.0</price.amount></Point>
      <Point><position>2</position><price.amount>52.5</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
"""


def test_entsoe_parse_xml():
    prices = _parse_entsoe_prices(SAMPLE_ENTSOE_XML)
    assert prices == [0.045, 0.0525]  # EUR/MWh -> EUR/kWh


def test_entsoe_parse_malformed_returns_empty():
    assert _parse_entsoe_prices(b"<not-the-document/>") == []


def test_entsoe_unconfigured_returns_none():
    assert ENTSOEAdapter("").fetch_hourly_prices() is None


def test_entsoe_fetch_error_never_raises(monkeypatch):
    def fake_get(url, params, timeout):
        raise ConnectionError("boom")
    monkeypatch.setattr(market_data.requests, "get", fake_get)
    result = ENTSOEAdapter("key").fetch_hourly_prices()
    assert isinstance(result, dict) and "error" in result


def test_entsoe_parses_response(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        content = SAMPLE_ENTSOE_XML

    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: FakeResp())
    result = ENTSOEAdapter("key", bidding_zone="DE_LU").fetch_hourly_prices()
    assert result["unit"] == "EUR/kWh"
    assert result["prices"] == [0.045, 0.0525]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lists_builtins():
    assert "synthetic" in registered_market_adapters()


def test_get_market_adapter_unknown_raises():
    with pytest.raises(KeyError):
        get_market_adapter("bogus")


def test_register_market_adapter_adds_custom():
    register_market_adapter("custom-test", SyntheticMarketAdapter())
    try:
        assert "custom-test" in registered_market_adapters()
        assert get_market_adapter("custom-test").name == "Synthetic"
    finally:
        # Registration is module-global — don't leak state into other tests.
        market_data._MARKET_ADAPTERS.pop("custom-test", None)


# ---------------------------------------------------------------------------
# Carbon resolution
# ---------------------------------------------------------------------------

def test_resolve_carbon_live_when_adapter_provides():
    result = resolve_carbon_intensity("EU_AVG", SyntheticMarketAdapter())
    assert result["source"] == "live"
    assert result["per_hour"] is not None
    assert result["g_co2_per_kwh"] > 0


def test_resolve_carbon_static_fallback():
    result = resolve_carbon_intensity("GERMANY")
    assert result["source"] == "static"
    assert result["g_co2_per_kwh"] == GRID_CARBON_INTENSITY["GERMANY"]


def test_resolve_carbon_falls_back_when_adapter_unsupported():
    result = resolve_carbon_intensity("GERMANY", EIAAdapter("key"))
    assert result["source"] == "static"
