"""
Tests for src.managed_charging — tariff-aware EV charging plans with the
unmanaged baseline and flexibility (EFC) measurement.
"""

import pytest

from src.market_data import SyntheticMarketAdapter
from src.managed_charging import managed_charge_plan, CHARGING_EFFICIENCY


def _prices(hours=48, spike_hour=12):
    return SyntheticMarketAdapter(spike_hour=spike_hour, spike_price_eur=0.45).fetch_hourly_prices()["prices"]


def test_plan_reaches_target_when_horizon_sufficient():
    prices = _prices()
    # 60 kWh pack, 20 -> 80% = 36 kWh battery-side = ~38.3 kWh wall at
    # 11 kW -> ~4 hours; 48h horizon is plenty.
    result = managed_charge_plan(prices, battery_kwh=60.0, initial_soc_pct=20.0,
                                 target_soc_pct=80.0, max_charge_kw=11.0)
    assert result["reached_soc_pct"] == pytest.approx(80.0, abs=0.5)
    assert result["delivered_kwh"] == pytest.approx(36.0, abs=0.1)
    assert len(result["plan"]) == len(prices)


def test_managed_cost_no_more_than_unmanaged():
    result = managed_charge_plan(_prices(), 60.0, 20.0, 80.0, 11.0)
    assert result["cost_eur"] <= result["unmanaged_cost_eur"] + 1e-9
    assert result["savings_eur"] >= 0
    assert result["savings_pct"] >= 0


def test_plan_avoids_expensive_hours():
    prices = _prices(spike_hour=12)
    result = managed_charge_plan(prices, 60.0, 20.0, 80.0, 11.0)
    # The spike hour is the most expensive hour; a cost-minimizing plan
    # never charges there.
    assert result["plan"][12]["charge_kw"] == 0.0


def test_plan_deterministic():
    prices = _prices()
    a = managed_charge_plan(prices, 60.0, 20.0, 80.0, 11.0)
    b = managed_charge_plan(prices, 60.0, 20.0, 80.0, 11.0)
    assert a["plan"] == b["plan"]
    assert a["cost_eur"] == b["cost_eur"]


def test_plan_flexibility_efc_positive():
    result = managed_charge_plan(_prices(), 60.0, 20.0, 80.0, 11.0)
    # A 20 -> 80% charge is a 60pp half-cycle -> ~0.3 EFC.
    assert result["flexibility_efc"] > 0
    assert result["flexibility_efc"] == pytest.approx(0.3, abs=0.05)


def test_plan_honest_limitations():
    result = managed_charge_plan(_prices(), 60.0, 20.0, 80.0, 11.0)
    assert any("not a control signal" in lim for lim in result["limitations"])
    assert any("OCPP" in lim for lim in result["limitations"])


def test_plan_no_op_when_target_already_met():
    result = managed_charge_plan(_prices(), 60.0, 80.0, 80.0, 11.0)
    assert result["cost_eur"] == 0.0
    assert result["flexibility_efc"] == 0.0
    assert all(s["charge_kw"] == 0.0 for s in result["plan"])


def test_invalid_inputs_raise():
    prices = _prices()
    with pytest.raises(ValueError):
        managed_charge_plan([], 60.0, 20.0, 80.0, 11.0)
    with pytest.raises(ValueError):
        managed_charge_plan(prices, 0.0, 20.0, 80.0, 11.0)
    with pytest.raises(ValueError):
        managed_charge_plan(prices, 60.0, 20.0, 80.0, 0.0)
    with pytest.raises(ValueError):
        managed_charge_plan(prices, 60.0, 90.0, 80.0, 11.0)


def test_unmanaged_start_hour_shifts_baseline():
    prices = _prices()
    now = managed_charge_plan(prices, 60.0, 20.0, 80.0, 11.0, unmanaged_start_hour=0)
    delayed = managed_charge_plan(prices, 60.0, 20.0, 80.0, 11.0, unmanaged_start_hour=24)
    assert now["unmanaged_cost_eur"] != delayed["unmanaged_cost_eur"]
