"""
Tests for src.grid_services — arbitrage + frequency regulation + capacity
revenue stack with honest assumption provenance.
"""

import pytest

from src.market_data import SyntheticMarketAdapter
from src.grid_services import grid_services_revenue, GRID_SERVICES_ASSUMPTIONS


def _prices(hours=48):
    prices = SyntheticMarketAdapter(spike_hour=12, spike_price_eur=0.45).fetch_hourly_prices()["prices"]
    return (prices * (hours // len(prices) + 1))[:hours]


def test_revenue_breakdown_and_totals():
    result = grid_services_revenue(_prices(), battery_kwh=10.0)
    assert result["arbitrage_eur"] > 0
    assert result["frequency_regulation_eur"] > 0
    assert result["capacity_eur"] > 0
    assert result["total_eur"] == round(
        result["arbitrage_eur"] + result["frequency_regulation_eur"] + result["capacity_eur"], 2
    )
    assert result["dispatchable_power_kw"] > 0


def test_default_assumptions_used():
    result = grid_services_revenue(_prices(), battery_kwh=10.0)
    used = result["assumptions_used"]
    assert used["frequency_regulation_price_eur_per_mw_h"] == \
        GRID_SERVICES_ASSUMPTIONS["frequency_regulation_price_eur_per_mw_h"]["value"]
    assert used["capacity_payment_eur_per_mw_year"] == \
        GRID_SERVICES_ASSUMPTIONS["capacity_payment_eur_per_mw_year"]["value"]
    assert result["exclusivity_note"]
    # Every assumption ships with a provenance label.
    assert all(labels["label"] for labels in result["labels"].values())


def test_override_assumptions_flow_through():
    result = grid_services_revenue(
        _prices(), battery_kwh=10.0,
        frequency_regulation_price_eur_per_mw_h=50.0,
        capacity_payment_eur_per_mw_year=80000.0,
    )
    assert result["assumptions_used"]["frequency_regulation_price_eur_per_mw_h"] == 50.0
    assert result["assumptions_used"]["capacity_payment_eur_per_mw_year"] == 80000.0


def test_window_annualization_flag():
    short = grid_services_revenue(_prices(48), battery_kwh=10.0)
    assert short["arbitrage_annualized_from_window"] is True
    annual = grid_services_revenue(_prices(8760), battery_kwh=10.0, price_window_is_annual=True)
    assert annual["arbitrage_annualized_from_window"] is False


def test_health_aware_power_reduces_regulation_revenue():
    healthy = grid_services_revenue(_prices(), battery_kwh=10.0, soh_pct=100.0)
    limited = grid_services_revenue(_prices(), battery_kwh=10.0, soh_pct=100.0, sop_pct=50.0)
    assert limited["dispatchable_power_kw"] < healthy["dispatchable_power_kw"]
    assert limited["frequency_regulation_eur"] < healthy["frequency_regulation_eur"]


def test_empty_prices_raises():
    with pytest.raises(ValueError):
        grid_services_revenue([], 10.0)
