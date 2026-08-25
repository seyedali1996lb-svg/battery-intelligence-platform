"""
Integration tests for the extended FastAPI endpoints.
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



def test_actions_api(client, auth_headers):
    # List actions
    res = client.get("/actions", headers=auth_headers)
    assert res.status_code == 200
    actions = res.json()
    assert isinstance(actions, list)
    assert len(actions) > 0

    # Create action
    payload = {
        "cell_id": "B0005",
        "title": "API Test Anomaly",
        "category": "SAFETY",
        "severity": "CRITICAL",
        "description": "Critical voltage fluctuation",
        "recommended_action": "INSPECT_CELL",
        "soh_pct": 74.5,
    }
    create_res = client.post("/actions", json=payload, headers=auth_headers)
    assert create_res.status_code == 200
    act_data = create_res.json()
    assert act_data["id"].startswith("act-")

    # Triage action
    triage_res = client.post(
        f"/actions/{act_data['id']}/triage",
        json={"status": "IN_PROGRESS", "assigned_to": "Engineer Bob"},
        headers=auth_headers,
    )
    assert triage_res.status_code == 200
    assert triage_res.json()["status"] == "IN_PROGRESS"

    # Dispatch workflow
    dispatch_res = client.post(
        f"/actions/{act_data['id']}/dispatch",
        json={"target_system": "CMMS"},
        headers=auth_headers,
    )
    assert dispatch_res.status_code == 200
    assert dispatch_res.json()["status"] == "SUCCESS"


def test_partial_cycles_api(client, auth_headers):
    # Rainflow counting
    rainflow_payload = {
        "soc_series": [100.0, 80.0, 85.0, 50.0, 60.0, 20.0, 100.0],
    }
    res = client.post("/analytics/rainflow", json=rainflow_payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total_cycles"] > 0
    assert "dod_histogram" in data

    # OCV reconstruction
    ocv_payload = {
        "time_sec": list(range(0, 150)),
        "voltage_v": [3.6 if t < 30 else 3.7 + 0.1 * (1 - 2.718 ** (-(t - 30) / 20)) for t in range(0, 150)],
        "current_a": [-2.0 if t < 30 else 0.0 for t in range(0, 150)],
    }
    res_ocv = client.post("/analytics/ocv-reconstruct", json=ocv_payload, headers=auth_headers)
    assert res_ocv.status_code == 200
    assert res_ocv.json()["detected_rests"] == 1


def test_cycler_detect_api(client, auth_headers):
    cols = ["Test_Time(s)", "Cycle_Index", "Voltage(V)", "Current(A)", "Discharge_Capacity(Ah)"]
    res = client.post("/ingest/cycler-detect", json=cols, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["hardware"] == "Arbin"
    assert "voltage_v" in data["mapped_columns"]


def test_pinn_estimate_api(client, auth_headers):
    payload = {
        "cycles": list(range(1, 30)),
        "soh_pct": [100.0 - 0.5 * (c ** 0.5) for c in range(1, 30)],
        "future_cycles_count": 200,
        "temperature_c": 25.0,
        "c_rate": 1.0,
    }
    res = client.post("/physics/pinn-estimate", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "rul_estimation" in data
    assert "parameters" in data


def test_dynamic_lca_and_passport_api(client, auth_headers):
    res_lca = client.get("/cells/B0005/dynamic-lca?region=GERMANY", headers=auth_headers)
    assert res_lca.status_code == 200
    lca_data = res_lca.json()
    assert lca_data["net_lifecycle_co2_kg"] > 0

    res_vp = client.get("/cells/B0005/verifiable-passport", headers=auth_headers)
    assert res_vp.status_code == 200
    vp_data = res_vp.json()
    assert vp_data["credentialSubject"]["cellId"] == "B0005"
    assert "proof" in vp_data

    res_bids = client.get("/cells/B0005/second-life-bids", headers=auth_headers)
    assert res_bids.status_code == 200
    bids_data = res_bids.json()
    assert isinstance(bids_data, list)
    assert len(bids_data) > 0


def test_streaming_process_api(client, auth_headers):
    sample = {
        "cell_id": "B0005",
        "voltage_v": 3.82,
        "current_a": -2.0,
        "temperature_c": 26.0,
    }
    res = client.post("/telemetry/process", json=sample, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["cell_id"] == "B0005"
    assert "mahalanobis_score" in data
