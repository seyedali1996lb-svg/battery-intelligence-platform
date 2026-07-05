"""
Unit tests for src/api.py's auth layer (Phase D).

Uses FastAPI's TestClient against the real app — the bundle-loading path
(`_get_bundle()` -> `load_everything()`) uses the same on-disk model cache
as the Streamlit app (src/bundle_cache.py), so this doesn't retrain models
per test run once the cache is warm.
"""

import datetime

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

from api import app, _JWT_SECRET, _JWT_ALGORITHM

client = TestClient(app)

DEMO_USER = "engineer"
DEMO_PASSWORD = "battery"


def test_root_does_not_require_auth():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Battery Intelligence Platform API"


def test_login_with_correct_demo_credentials_succeeds():
    resp = client.post("/auth/login", json={"username": DEMO_USER, "password": DEMO_PASSWORD})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["org_name"] == "Demo Org"
    assert body["role"] == "engineer"
    assert body["access_token"]


def test_login_with_wrong_password_rejected():
    resp = client.post("/auth/login", json={"username": DEMO_USER, "password": "wrong"})
    assert resp.status_code == 401


def test_login_with_unknown_username_rejected():
    resp = client.post("/auth/login", json={"username": "nobody-such-user", "password": "x"})
    assert resp.status_code == 401


@pytest.mark.parametrize("path", [
    "/health",
    "/cells",
    "/fleet/summary",
    "/fleet/alerts",
])
def test_gated_endpoints_reject_missing_token(path):
    resp = client.get(path)
    assert resp.status_code == 401


@pytest.mark.parametrize("path", [
    "/health",
    "/cells",
    "/fleet/summary",
    "/fleet/alerts",
])
def test_gated_endpoints_reject_garbage_token(path):
    resp = client.get(path, headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_gated_endpoint_rejects_expired_token():
    expired_payload = {
        "sub": DEMO_USER, "org_id": 1, "role": "engineer",
        "exp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1),
    }
    expired_token = pyjwt.encode(expired_payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)
    resp = client.get("/cells", headers={"Authorization": f"Bearer {expired_token}"})
    assert resp.status_code == 401


def test_gated_endpoints_accept_valid_token_and_return_real_data():
    login_resp = client.post("/auth/login", json={"username": DEMO_USER, "password": DEMO_PASSWORD})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/cells", headers=headers)
    assert resp.status_code == 200
    cells = resp.json()["cells"]
    assert len(cells) > 0
    # A real cell ID, not a bundle key — regression guard for the
    # featured_dfs/bundles tuple-unpacking bug fixed alongside this test.
    assert not set(cells) <= {"nasa", "severson", "synth"}

    resp = client.get("/fleet/summary", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["total_cells"] > 0

    first_cell = cells[0]
    resp = client.get(f"/cells/{first_cell}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["cell_id"] == first_cell
