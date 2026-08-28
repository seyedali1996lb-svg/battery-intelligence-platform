"""
Integration tests for the Lifecycle Intelligence (P1) API endpoints:
market prices, health-aware dispatch, grid-services revenue, managed
charging, and fleet dispatchable-capacity offers.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app, _create_access_token
from src.market_data import SyntheticMarketAdapter


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    token = _create_access_token({"username": "engineer", "org_id": 1, "role": "engineer"})
    return {"Authorization": f"Bearer {token}"}


def _prices():
    return SyntheticMarketAdapter(spike_hour=12, spike_price_eur=0.45).fetch_hourly_prices()["prices"]


def test_market_prices_synthetic(client, auth_headers):
    res = client.get("/market/prices?adapter=synthetic&hours=48&region=GERMANY", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["prices"]["unit"] == "EUR/kWh"
    assert len(data["prices"]["prices"]) == 48
    assert data["prices"]["adapter"] == "Synthetic"
    # Carbon resolution: synthetic adapter provides a live series.
    assert data["carbon_intensity"]["source"] == "live"


def test_market_prices_unconfigured_adapter(client, auth_headers):
    res = client.get("/market/prices?adapter=eia", headers=auth_headers)
    assert res.status_code == 400
    assert "not configured" in res.json()["detail"]


def test_market_prices_unknown_adapter(client, auth_headers):
    res = client.get("/market/prices?adapter=bogus", headers=auth_headers)
    assert res.status_code == 400


def test_market_prices_requires_auth(client):
    assert client.get("/market/prices").status_code == 401


def test_dispatch_schedule(client, auth_headers):
    res = client.post("/analytics/dispatch-schedule", headers=auth_headers, json={
        "prices_eur_per_kwh": _prices(),
        "battery_kwh": 10.0,
        "soh_pct": 78.0,
        "sop_pct": 50.0,
        "rul_cycles": 90,
        "rul_reliable": True,
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["schedule"]) == 48
    assert data["band"]["caution"] is True
    assert data["band"]["power_cap_factor"] == 0.5
    assert data["limitations"]


def test_dispatch_schedule_invalid_input(client, auth_headers):
    res = client.post("/analytics/dispatch-schedule", headers=auth_headers, json={
        "prices_eur_per_kwh": [],
        "battery_kwh": 10.0,
    })
    assert res.status_code == 400


def test_dispatch_comparison(client, auth_headers):
    res = client.post("/analytics/dispatch-comparison", headers=auth_headers, json={
        "prices_eur_per_kwh": _prices(),
        "battery_kwh": 10.0,
        "soh_pct": 78.0,
        "sop_pct": 50.0,
        "rul_cycles": 90,
        "rul_reliable": True,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["healthy_assumption"]["band"]["min_soc_pct"] == 10.0
    assert data["health_constrained"]["band"]["caution"] is True
    assert set(data["delta"]) == {"revenue_eur_delta", "efc_delta", "mean_cycle_dod_pct_delta"}


def test_grid_services_revenue(client, auth_headers):
    res = client.post("/analytics/grid-services-revenue", headers=auth_headers, json={
        "prices_eur_per_kwh": _prices(),
        "battery_kwh": 10.0,
        "soh_pct": 100.0,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["arbitrage_eur"] > 0
    assert data["frequency_regulation_eur"] > 0
    assert data["capacity_eur"] > 0
    assert data["exclusivity_note"]


def test_managed_charge_plan(client, auth_headers):
    res = client.post("/analytics/managed-charge-plan", headers=auth_headers, json={
        "prices_eur_per_kwh": _prices(),
        "battery_kwh": 60.0,
        "initial_soc_pct": 20.0,
        "target_soc_pct": 80.0,
        "max_charge_kw": 11.0,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["reached_soc_pct"] == pytest.approx(80.0, abs=0.5)
    assert data["savings_eur"] >= 0
    assert data["plan"][12]["charge_kw"] == 0.0  # avoids the price spike


def test_fleet_dispatchable_capacity(client, auth_headers):
    res = client.post("/fleet/dispatchable-capacity", headers=auth_headers, json={
        "cells": [
            {"cell_id": "c1", "nominal_kwh": 10.0, "soh_pct": 95.0, "soc_pct": 30.0, "sop_pct": 90.0},
            {"cell_id": "c2", "nominal_kwh": 10.0, "soh_pct": 74.0, "soc_pct": 50.0,
             "sop_pct": 55.0, "rul_cycles": 90, "rul_reliable": True},
        ],
        "window_hours": 2,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["offer"]["service"] == "dispatchable_capacity"
    assert data["offer"]["energy_kwh"] > 0
    assert data["fleet_context"]["n_included"] == 2
    assert data["caveats"]


def test_fleet_dispatchable_capacity_empty_cells(client, auth_headers):
    res = client.post("/fleet/dispatchable-capacity", headers=auth_headers, json={"cells": []})
    assert res.status_code == 400


def test_dynamic_lca_accepts_live_intensity(client, auth_headers):
    res = client.get("/cells/B0005/dynamic-lca?region=GERMANY&grid_intensity_g_kwh=120", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["grid_intensity_g_kwh"] == 120.0
    assert data["grid_intensity_source"].startswith("live")
