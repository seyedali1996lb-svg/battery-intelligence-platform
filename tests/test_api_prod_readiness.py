"""API-level tests for the production-readiness batch: the Phase 3 digital
twin endpoint and the per-org rate limiting guard."""

import pytest
from fastapi.testclient import TestClient

from api import app, _create_access_token
import rate_limit as rl


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    token = _create_access_token({"username": "engineer", "org_id": 1, "role": "engineer"})
    return {"Authorization": f"Bearer {token}"}


# ── Digital twin endpoint ────────────────────────────────────────────────────

def test_cell_twin_snapshot(client, auth_headers):
    res = client.get("/cells/B0005/twin", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["cell_id"] == "B0005"
    assert data["history"]["n_cycles"] > 0
    assert data["indicators"]["soh_pct"] is not None
    assert data["projection"]["beta"] is not None
    assert data["projection"]["spm_capacity_ah"] is None  # API skips the slow SPM anchor
    assert any("not a live-synced digital twin" in l for l in data["labels"])


def test_cell_twin_requires_auth(client):
    assert client.get("/cells/B0005/twin").status_code == 401


def test_cell_twin_unknown_cell_404(client, auth_headers):
    assert client.get("/cells/ZZZZ/twin", headers=auth_headers).status_code == 404


# ── Per-org rate limiting ────────────────────────────────────────────────────

def test_rate_limit_blocks_third_request_per_org(client, auth_headers):
    rl.set_rate_limit(2)
    try:
        assert client.get("/cells", headers=auth_headers).status_code == 200
        assert client.get("/cells", headers=auth_headers).status_code == 200
        third = client.get("/cells", headers=auth_headers)
        assert third.status_code == 429
        assert "Retry-After" in third.headers
    finally:
        rl.set_rate_limit(0)  # back to disabled — never leak the limit into other tests


def test_rate_limit_is_per_org(client, auth_headers):
    rl.set_rate_limit(1)
    try:
        # Org 1 exhausts its bucket on /cells…
        assert client.get("/cells", headers=auth_headers).status_code == 200
        assert client.get("/cells", headers=auth_headers).status_code == 429
        # …but a different org's token is unaffected (same endpoint).
        other = _create_access_token({"username": "engineer", "org_id": 2, "role": "engineer"})
        res = client.get("/cells", headers={"Authorization": f"Bearer {other}"})
        assert res.status_code == 200
    finally:
        rl.set_rate_limit(0)
