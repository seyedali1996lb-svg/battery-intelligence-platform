"""Unit tests for src/deployment_sizing.py.

estimate_annual_savings() is a monthly energy-balance approximation (see
module docstring for the overestimation caveat) — these tests check its
boundary behaviour, not physical realism. size_deployment()'s grid search
is tested with an injected fake pv_yield_fn so it never hits the network,
with a call-counter wrapper to lock in the "at most one PVGIS call per
unique pv_kwp step" dedup behaviour described in the module docstring."""

import pytest

from deployment_sizing import (
    SIZING_ASSUMPTIONS,
    estimate_annual_savings,
    payback_years,
    npv_eur,
    size_deployment,
)


FLAT_MONTH = [100.0] * 12


# ---------------------------------------------------------------------------
# estimate_annual_savings
# ---------------------------------------------------------------------------

def test_zero_pv_falls_back_to_pure_arbitrage():
    zero_pv = [0.0] * 12
    result = estimate_annual_savings(
        pv_monthly_kwh=zero_pv, monthly_consumption_kwh=FLAT_MONTH,
        battery_kwh=10.0, tariff_high_eur=0.30, tariff_low_eur=0.10,
    )
    assert result["annual_savings_eur"] > 0
    for m in result["monthly"]:
        assert m["direct_pv_kwh"] == 0.0
        assert m["pv_via_battery_kwh"] == 0.0


def test_zero_battery_is_pv_self_consumption_only():
    result = estimate_annual_savings(
        pv_monthly_kwh=FLAT_MONTH, monthly_consumption_kwh=FLAT_MONTH,
        battery_kwh=0.0, tariff_high_eur=0.30, tariff_low_eur=0.10,
    )
    for m in result["monthly"]:
        assert m["pv_via_battery_kwh"] == 0.0
        assert m["arbitrage_kwh"] == 0.0
    assert result["annual_savings_eur"] > 0


def test_pv_much_greater_than_load_still_bounded_by_derating():
    huge_pv = [10_000.0] * 12
    tiny_load = [10.0] * 12
    result = estimate_annual_savings(
        pv_monthly_kwh=huge_pv, monthly_consumption_kwh=tiny_load,
        battery_kwh=5.0, tariff_high_eur=0.30, tariff_low_eur=0.10,
        self_consumption_derating=0.70,
    )
    for m in result["monthly"]:
        # direct PV offset can never exceed load itself
        assert m["direct_pv_kwh"] <= 10.0 + 1e-9
        assert m["export_kwh"] > 0  # huge surplus after battery is full


def test_load_much_greater_than_pv():
    tiny_pv = [1.0] * 12
    huge_load = [1000.0] * 12
    result = estimate_annual_savings(
        pv_monthly_kwh=tiny_pv, monthly_consumption_kwh=huge_load,
        battery_kwh=5.0, tariff_high_eur=0.30, tariff_low_eur=0.10,
    )
    for m in result["monthly"]:
        assert m["grid_import_kwh"] > 0  # far from fully offset


def test_derating_boundary_zero_and_one_never_raise():
    for derating in (0.0, 1.0):
        result = estimate_annual_savings(
            pv_monthly_kwh=FLAT_MONTH, monthly_consumption_kwh=FLAT_MONTH,
            battery_kwh=5.0, tariff_high_eur=0.30, tariff_low_eur=0.10,
            self_consumption_derating=derating,
        )
        assert isinstance(result["annual_savings_eur"], float)


def test_derating_one_yields_more_direct_pv_than_derating_zero():
    hi = estimate_annual_savings(
        pv_monthly_kwh=FLAT_MONTH, monthly_consumption_kwh=FLAT_MONTH,
        battery_kwh=0.0, tariff_high_eur=0.30, tariff_low_eur=0.10,
        self_consumption_derating=1.0,
    )
    lo = estimate_annual_savings(
        pv_monthly_kwh=FLAT_MONTH, monthly_consumption_kwh=FLAT_MONTH,
        battery_kwh=0.0, tariff_high_eur=0.30, tariff_low_eur=0.10,
        self_consumption_derating=0.0,
    )
    assert hi["annual_savings_eur"] > lo["annual_savings_eur"]


# ---------------------------------------------------------------------------
# payback_years / npv_eur
# ---------------------------------------------------------------------------

def test_payback_years_none_on_zero_or_negative_savings():
    assert payback_years(1000.0, 0.0) is None
    assert payback_years(1000.0, -50.0) is None


def test_payback_years_basic():
    assert payback_years(1000.0, 250.0) == pytest.approx(4.0)


def test_npv_eur_basic_shape():
    value = npv_eur(1000.0, 200.0, discount_rate=0.08, horizon_years=15)
    assert isinstance(value, float)
    # Positive discounted stream of 200/yr for 15 years at 8% clearly exceeds 1000 upfront
    assert value > 0


def test_npv_eur_negative_when_savings_too_low():
    value = npv_eur(10_000.0, 10.0, discount_rate=0.08, horizon_years=15)
    assert value < 0


# ---------------------------------------------------------------------------
# size_deployment — fake pv_yield_fn, no network
# ---------------------------------------------------------------------------

class _CountingPvYield:
    """Fake pv_yield_fn that records every distinct call for dedup testing."""

    def __init__(self, monthly_kwh=None):
        self.monthly_kwh = monthly_kwh or [50.0] * 12
        self.calls = []

    def __call__(self, lat, lon, peakpower_kwp, tilt_deg, azimuth_deg, **kwargs):
        self.calls.append(peakpower_kwp)
        return {"annual_kwh": sum(self.monthly_kwh), "monthly_kwh": self.monthly_kwh}


def test_size_deployment_calls_pv_yield_fn_at_most_once_per_kwp_step():
    fake = _CountingPvYield()
    result = size_deployment(
        lat=45.8, lon=15.98, tilt_deg=30, azimuth_compass_deg=180,
        available_area_m2=20.0, cell_kwh_per_cell=0.05,
        monthly_consumption_kwh=FLAT_MONTH,
        tariff_high_eur=0.30, tariff_low_eur=0.10,
        max_payoff_years=15, max_investment_eur=50_000,
        n_cells_range=range(1, 11),
        pv_yield_fn=fake,
    )
    # 6 pv_kwp steps (0..max), but pv_kwp==0 is synthesized without a call,
    # so at most 5 real calls regardless of how many n_cells are explored.
    assert len(fake.calls) <= 5
    assert len(fake.calls) == len(set(round(c, 3) for c in fake.calls))
    assert result["candidates"]


def test_size_deployment_returns_feasible_winner_when_affordable():
    fake = _CountingPvYield(monthly_kwh=[200.0] * 12)
    result = size_deployment(
        lat=45.8, lon=15.98, tilt_deg=30, azimuth_compass_deg=180,
        available_area_m2=50.0, cell_kwh_per_cell=0.05,
        monthly_consumption_kwh=[150.0] * 12,
        tariff_high_eur=0.35, tariff_low_eur=0.08,
        max_payoff_years=25, max_investment_eur=200_000,
        pv_yield_fn=fake,
    )
    assert result["feasible"] is True
    assert result["winner"] is not None
    assert result["constraint_note"] is None


def test_size_deployment_near_miss_when_nothing_feasible():
    fake = _CountingPvYield(monthly_kwh=[1.0] * 12)  # negligible PV yield
    result = size_deployment(
        lat=45.8, lon=15.98, tilt_deg=30, azimuth_compass_deg=180,
        available_area_m2=5.0, cell_kwh_per_cell=0.01,
        monthly_consumption_kwh=[500.0] * 12,
        tariff_high_eur=0.10, tariff_low_eur=0.09,  # near-zero arbitrage spread
        max_payoff_years=0.01,  # impossible constraint
        max_investment_eur=1_000_000,
        pv_yield_fn=fake,
    )
    assert result["feasible"] is False
    assert result["winner"] is not None  # honest near-miss, not a blank result
    assert result["constraint_note"] is not None


def test_size_deployment_degrades_gracefully_on_pv_errors():
    def failing_pv_yield(**kwargs):
        return {"error": "simulated PVGIS outage"}

    result = size_deployment(
        lat=45.8, lon=15.98, tilt_deg=30, azimuth_compass_deg=180,
        available_area_m2=20.0, cell_kwh_per_cell=0.05,
        monthly_consumption_kwh=FLAT_MONTH,
        tariff_high_eur=0.30, tariff_low_eur=0.10,
        max_payoff_years=15, max_investment_eur=50_000,
        pv_yield_fn=failing_pv_yield,
    )
    # pv_kwp=0 step is synthesized locally (no PVGIS call needed), so a
    # battery-only result should still come back rather than an empty result.
    assert result["candidates"]
    assert result["pv_errors"]
    assert all(c["pv_kwp"] == 0.0 for c in result["candidates"])


def test_sizing_assumptions_shape_matches_convention():
    for key, entry in SIZING_ASSUMPTIONS.items():
        assert set(entry.keys()) == {"value", "slider_range", "unit", "label", "source"}
        assert entry["label"] in ("Cited estimate", "Illustrative — not sourced")
        lo, hi = entry["slider_range"]
        assert lo <= entry["value"] <= hi
