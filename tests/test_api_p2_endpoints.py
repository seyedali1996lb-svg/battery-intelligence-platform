"""
Integration tests for the P2 API endpoints: health-as-a-service
(GET /cells/{id}/health), live-carbon dynamic-LCA, and the ML-based
anomaly scan (POST /analytics/ml-anomaly).
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app, _create_access_token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    token = _create_access_token({"username": "engineer", "org_id": 1, "role": "engineer"})
    return {"Authorization": f"Bearer {token}"}


def _cycles():
    return [
        {"cycle_number": i + 1, "capacity_ah": 2.0 - 0.0005 * (i + 1),
         "resistance_ohm": 0.05 + 0.00002 * (i + 1), "temperature_c": 25.0}
        for i in range(100)
    ]


# ---------------------------------------------------------------------------
# Health-as-a-service
# ---------------------------------------------------------------------------

def test_cell_health_full_record(client, auth_headers):
    res = client.get("/cells/B0005/health", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["cell_id"] == "B0005"
    assert data["soh_pct"] > 0
    assert data["status"] in ("Healthy", "Degrading", "End of Life")
    # RUL fields present; values gated on the per-cell reliability floor.
    assert "rul_pred" in data and "rul_q10" in data and "rul_q90" in data
    assert isinstance(data["rul_reliable"], bool)
    assert set(data["confidence"]) == {"soh", "rul", "sop"}
    assert "passport_fragments" in data
    fragments = data["passport_fragments"]
    assert fragments["chemistry"]
    assert fragments["r_code"].startswith("R")
    assert "best_second_life_application" in fragments
    assert "model_card" in data


def test_cell_health_requires_auth(client):
    assert client.get("/cells/B0005/health").status_code == 401


def test_cell_health_unknown_cell_404(client, auth_headers):
    assert client.get("/cells/ZZZZ/health", headers=auth_headers).status_code == 404


# ---------------------------------------------------------------------------
# Live-carbon dynamic LCA
# ---------------------------------------------------------------------------

def test_dynamic_lca_static_by_default(client, auth_headers):
    res = client.get("/cells/B0005/dynamic-lca?region=GERMANY", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["grid_intensity_source"].startswith("static")
    assert "carbon_resolution" not in data


def test_dynamic_lca_live_carbon_opt_in(client, auth_headers):
    res = client.get("/cells/B0005/dynamic-lca?region=GERMANY&use_live_carbon=true", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["grid_intensity_source"].startswith("live")
    assert data["carbon_resolution"]["source"] == "live"
    assert data["carbon_resolution"]["g_co2_per_kwh"] > 0


def test_dynamic_lca_live_carbon_explicit_value_wins(client, auth_headers):
    res = client.get(
        "/cells/B0005/dynamic-lca?region=GERMANY&use_live_carbon=true&grid_intensity_g_kwh=99",
        headers=auth_headers,
    )
    assert res.status_code == 200
    assert res.json()["grid_intensity_g_kwh"] == 99.0


def test_dynamic_lca_live_carbon_unknown_adapter_400(client, auth_headers):
    res = client.get("/cells/B0005/dynamic-lca?use_live_carbon=true&carbon_adapter=bogus", headers=auth_headers)
    assert res.status_code == 400


def test_dynamic_lca_live_carbon_unconfigured_400(client, auth_headers):
    res = client.get("/cells/B0005/dynamic-lca?use_live_carbon=true&carbon_adapter=eia", headers=auth_headers)
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# ML anomaly scan
# ---------------------------------------------------------------------------

def test_ml_anomaly_scan(client, auth_headers):
    cycles = _cycles()
    cycles[60]["capacity_ah"] -= 0.5  # cycle 61
    res = client.post("/analytics/ml-anomaly", headers=auth_headers, json={"cell_id": "B0005", "cycles": cycles})
    assert res.status_code == 200
    data = res.json()
    assert 61 in data["flagged_cycles"]
    assert data["n_warmup_unscored"] == 30
    assert data["per_cycle"][0]["anomaly_score"] is None  # JSON null, not NaN
    assert data["caveats"]


def test_ml_anomaly_scan_empty_cycles_400(client, auth_headers):
    res = client.post("/analytics/ml-anomaly", headers=auth_headers, json={"cell_id": "B0005", "cycles": []})
    assert res.status_code == 400


def test_ml_anomaly_scan_too_few_cycles_400(client, auth_headers):
    cycles = _cycles()[:10]
    res = client.post("/analytics/ml-anomaly", headers=auth_headers, json={"cell_id": "B0005", "cycles": cycles})
    assert res.status_code == 400


def test_ml_anomaly_scan_requires_auth(client):
    assert client.post("/analytics/ml-anomaly", json={"cell_id": "x", "cycles": _cycles()}).status_code == 401
