"""
Server-side role-based write gating (src/rbac.py).

Covers the Settings roadmap row that previously read "no server-side
write-action gating yet": before src/rbac.py, the /actions write endpoints
(/actions create, /actions/{id}/triage, /actions/{id}/dispatch) trusted any
authenticated role, so a read-only role could mutate operational tickets by
calling the API directly even though the UI hid the buttons.

Expected matrix:
  create  (action.create)    -> admin, engineer, fleet
  triage  (action.triage)    -> admin, engineer, fleet
  dispatch(action.dispatch)  -> admin, engineer            (external CMMS commit)
  compliance / unknown role  -> denied on every write

Existing /actions behavior for engineer is already covered by
test_api_v2_endpoints.py::test_actions_api; this file is the cross-role and
denial surface.
"""

import pytest
from fastapi.testclient import TestClient

import rbac
from src.api import app, _create_access_token


@pytest.fixture
def client():
    return TestClient(app)


def _headers(role: str):
    token = _create_access_token({"username": role, "org_id": 1, "role": role})
    return {"Authorization": f"Bearer {token}"}


def _payload(title="RBAC test ticket", category="DEGRADATION", severity="MEDIUM"):
    return {
        "cell_id": "B0005",
        "title": title,
        "category": category,
        "severity": severity,
        "description": "created by RBAC test",
        "recommended_action": "INSPECT_CELL",
        "soh_pct": 90.0,
        "sla_hours": 24,
    }


# ── Pure policy (no FastAPI needed) ──────────────────────────────────────────

def test_matrix_create():
    for role in ("admin", "engineer", "fleet"):
        assert rbac.can(role, rbac.ACTION_CREATE_TICKET)
    assert not rbac.can("compliance", rbac.ACTION_CREATE_TICKET)
    assert not rbac.can("executive", rbac.ACTION_CREATE_TICKET)
    assert not rbac.can("ghost", rbac.ACTION_CREATE_TICKET)


def test_matrix_triage():
    for role in ("admin", "engineer", "fleet"):
        assert rbac.can(role, rbac.ACTION_TRIAGE_TICKET)
    assert not rbac.can("compliance", rbac.ACTION_TRIAGE_TICKET)


def test_matrix_dispatch_engineer_only():
    # Dispatch commits an external work order -> engineer+admin, above fleet.
    assert rbac.can("admin", rbac.ACTION_DISPATCH_TICKET)
    assert rbac.can("engineer", rbac.ACTION_DISPATCH_TICKET)
    assert not rbac.can("fleet", rbac.ACTION_DISPATCH_TICKET)
    assert not rbac.can("compliance", rbac.ACTION_DISPATCH_TICKET)


def test_unknown_action_denied_all():
    assert not rbac.can("admin", "nope.not-an-action")
    for role in ("admin", "engineer", "fleet", "compliance"):
        assert not rbac.can(role, "nope.not-an-action")


def test_unknown_role_fails_closed():
    assert not rbac.can("superuser", rbac.ACTION_CREATE_TICKET)
    assert not rbac.can("superuser", rbac.ACTION_DISPATCH_TICKET)


# ── API-level enforcement ────────────────────────────────────────────────────

def test_create_denied_for_compliance(client):
    res = client.post("/actions", json=_payload(), headers=_headers("compliance"))
    assert res.status_code == 403
    assert "not allowed to perform" in res.json()["detail"]


def test_triage_denied_for_compliance(client):
    # compliance can read the seeded ticket list but not mutate
    res = client.post("/actions/act-101/triage",
                      json={"status": "IN_PROGRESS", "assigned_to": "compl"},
                      headers=_headers("compliance"))
    assert res.status_code == 403


def test_dispatch_denied_for_fleet(client):
    # fleet is allowed to create/triage but NOT to commit an external dispatch
    res = client.post("/actions/act-101/dispatch",
                      json={"target_system": "CMMS", "payload": {"work_type": "REPAIR"}},
                      headers=_headers("fleet"))
    assert res.status_code == 403


def test_read_stays_open_for_compliance(client):
    # Gating is write-specific: a read-only role can still list tickets.
    res = client.get("/actions", headers=_headers("compliance"))
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_fleet_can_create_and_triage_but_not_dispatch(client):
    created = client.post("/actions", json=_payload(title="fleet-op ticket"),
                          headers=_headers("fleet"))
    assert created.status_code == 200
    act_id = created.json()["id"]

    triaged = client.post(f"/actions/{act_id}/triage",
                          json={"status": "IN_PROGRESS", "assigned_to": "operator"},
                          headers=_headers("fleet"))
    assert triaged.status_code == 200
    assert triaged.json()["status"] == "IN_PROGRESS"

    dispatched = client.post(f"/actions/{act_id}/dispatch",
                             json={"target_system": "CMMS", "payload": {}},
                             headers=_headers("fleet"))
    assert dispatched.status_code == 403


def test_engineer_can_dispatch(client):
    created = client.post("/actions", json=_payload(title="eng ticket"),
                          headers=_headers("engineer"))
    act_id = created.json()["id"]
    res = client.post(f"/actions/{act_id}/dispatch",
                      json={"target_system": "WARRANTY", "payload": {}},
                      headers=_headers("engineer"))
    assert res.status_code == 200
    assert res.json()["status"] == "SUCCESS"


def test_admin_can_everything(client):
    created = client.post("/actions", json=_payload(title="admin ticket"),
                          headers=_headers("admin"))
    assert created.status_code == 200
    act_id = created.json()["id"]

    assert client.post(f"/actions/{act_id}/triage",
                       json={"status": "IN_PROGRESS"}, headers=_headers("admin")).status_code == 200
    assert client.post(f"/actions/{act_id}/dispatch",
                       json={"target_system": "CIRCULARITY", "payload": {}},
                       headers=_headers("admin")).status_code == 200


def test_unknown_role_denied(client):
    res = client.post("/actions", json=_payload(), headers=_headers("ghost"))
    assert res.status_code == 403


def test_write_still_requires_auth(client):
    assert client.post("/actions", json=_payload()).status_code == 401
    assert client.post("/actions/act-101/triage",
                       json={"status": "IN_PROGRESS"}).status_code == 401