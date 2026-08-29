"""
Tests for src.health_aware_dispatch — SoP-limited, RUL/SOH-aware arbitrage
dispatch, and the healthy-assumption vs health-aware comparison.
"""

import pytest

from market_data import SyntheticMarketAdapter
from health_aware_dispatch import (
    health_constrained_band,
    arbitrage_schedule,
    schedule_comparison,
    RUL_CAUTION_CYCLES,
    SOH_CAUTION_PCT,
    SOP_POWER_FACTOR_FLOOR,
)


def _prices(hours=48, spike_hour=None):
    return SyntheticMarketAdapter(spike_hour=spike_hour, spike_price_eur=0.45).fetch_hourly_prices()["prices"]


# ---------------------------------------------------------------------------
# health_constrained_band
# ---------------------------------------------------------------------------

def test_band_healthy_defaults():
    band = health_constrained_band(soh_pct=100.0)
    assert band["power_cap_factor"] == 1.0
    assert band["min_soc_pct"] == 10.0
    assert band["max_soc_pct"] == 95.0
    assert band["caution"] is False
    assert band["reasons"] == []


def test_band_sop_scales_power_cap():
    band = health_constrained_band(soh_pct=100.0, sop_pct=50.0)
    assert band["power_cap_factor"] == 0.5
    assert not band["caution"]  # SoP alone does not narrow the band


def test_band_sop_floor():
    band = health_constrained_band(soh_pct=100.0, sop_pct=5.0)
    assert band["power_cap_factor"] == SOP_POWER_FACTOR_FLOOR


def test_band_sop_none_keeps_full_power():
    band = health_constrained_band(soh_pct=100.0, sop_pct=None)
    assert band["power_cap_factor"] == 1.0


def test_band_caution_on_reliable_low_rul():
    band = health_constrained_band(soh_pct=100.0, rul_cycles=RUL_CAUTION_CYCLES - 50, rul_reliable=True)
    assert band["caution"] is True
    assert band["min_soc_pct"] == 40.0
    assert band["max_soc_pct"] == 85.0
    assert any("RUL" in r for r in band["reasons"])


def test_band_caution_on_low_soh_without_reliable_rul():
    band = health_constrained_band(soh_pct=SOH_CAUTION_PCT - 5, rul_reliable=False)
    assert band["caution"] is True
    assert any("SOH" in r for r in band["reasons"])


def test_band_healthy_soh_with_reliable_high_rul_no_caution():
    band = health_constrained_band(soh_pct=85.0, rul_cycles=500, rul_reliable=True)
    assert band["caution"] is False


# ---------------------------------------------------------------------------
# arbitrage_schedule
# ---------------------------------------------------------------------------

def test_arbitrage_schedule_shape_and_band_respected():
    prices = _prices(spike_hour=12)
    result = arbitrage_schedule(prices, battery_kwh=10.0, soh_pct=78.0)
    assert len(result["schedule"]) == len(prices)
    socs = [s["soc_pct"] for s in result["schedule"]]
    band = result["band"]
    assert min(socs) >= band["min_soc_pct"] - 1e-6
    assert max(socs) <= band["max_soc_pct"] + 1e-6
    assert result["efc"] > 0  # the two-peak window forces real cycling
    assert result["limitations"]
    assert result["thresholds_eur_per_kwh"]["charge_below"] < result["thresholds_eur_per_kwh"]["discharge_above"]


def test_arbitrage_schedule_deterministic():
    prices = _prices()
    a = arbitrage_schedule(prices, 10.0)
    b = arbitrage_schedule(prices, 10.0)
    assert a["schedule"] == b["schedule"]
    assert a["revenue_eur"] == b["revenue_eur"]


def test_arbitrage_schedule_power_cap_scaled_by_sop():
    prices = _prices()
    healthy = arbitrage_schedule(prices, 10.0, soh_pct=100.0)
    limited = arbitrage_schedule(prices, 10.0, soh_pct=100.0, sop_pct=50.0)
    max_h = max(s["charge_kw"] for s in healthy["schedule"])
    max_l = max(s["charge_kw"] for s in limited["schedule"])
    assert max_l <= max_h * 0.5 + 1e-6


def test_arbitrage_schedule_empty_prices_raises():
    with pytest.raises(ValueError):
        arbitrage_schedule([], 10.0)


def test_arbitrage_schedule_zero_battery_raises():
    with pytest.raises(ValueError):
        arbitrage_schedule([0.1, 0.2], 0.0)


# ---------------------------------------------------------------------------
# schedule_comparison
# ---------------------------------------------------------------------------

def test_comparison_healthy_leg_models_cohort_assumption():
    prices = _prices(spike_hour=12)
    result = schedule_comparison(prices, 10.0, soh_pct=78.0, sop_pct=50.0, rul_cycles=90, rul_reliable=True)
    healthy_band = result["healthy_assumption"]["band"]
    constrained_band = result["health_constrained"]["band"]
    # The healthy leg uses the full default band regardless of real health
    # (that IS the cohort's assumption); the constrained leg narrows it.
    assert healthy_band["min_soc_pct"] == 10.0
    assert healthy_band["max_soc_pct"] == 95.0
    assert constrained_band["caution"] is True
    assert constrained_band["min_soc_pct"] > healthy_band["min_soc_pct"]
    assert constrained_band["max_soc_pct"] < healthy_band["max_soc_pct"]
    assert set(result["delta"]) == {"revenue_eur_delta", "efc_delta", "mean_cycle_dod_pct_delta"}


def test_comparison_healthy_cell_deltas_zero():
    prices = _prices(spike_hour=12)
    result = schedule_comparison(prices, 10.0, soh_pct=100.0)
    # Identical inputs on both legs -> identical schedules.
    assert result["delta"]["revenue_eur_delta"] == 0.0
    assert result["delta"]["efc_delta"] == 0.0
