"""
Server-side RBAC over the remaining app write surface (src/api.py endpoints +
src/db.py trust-boundary functions), enforced on the SAME src/rbac.py capability
keys the Streamlit UI reads -- so hiding a section and refusing a write can never
drift apart.

Surfaces covered beyond the Decision/CMMS tickets:
  - decision.log            -> append to the org's decision log (admin/engineer/fleet)
  - webhooks.manage         -> add/edit/remove webhook destinations (admin)
  - fleet-assets.manage     -> site/fleet/pack CRUD + cell assignment (admin)

For each: the pure rbac matrix, the db.py-level refusal (nothing persisted), and
the API boundary (403 for denied roles, 200 for granted, reads open for every
authenticated role).
"""

import pathlib
import re

import pytest
from fastapi.testclient import TestClient

import rbac
import db as db_module
from src.api import app, _create_access_token

_TEST_ENCRYPTION_KEY = "03ZJHIomd1hhT9w4FWvNxoN2wqPUnjfg3bSycZqUmgY="  # test-only, not the fallback


@pytest.fixture
def db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", _TEST_ENCRYPTION_KEY)
    monkeypatch.setattr(db_module, "_fernet", None)
    db_module.init_db()
    return db_module


@pytest.fixture
def client():
    return TestClient(app)


def _headers(role: str):
    token = _create_access_token({"username": role, "org_id": 1, "role": role})
    return {"Authorization": f"Bearer {token}"}


_OPERATOR_ROLES = (rbac.ROLE_ADMIN, rbac.ROLE_ENGINEER, rbac.ROLE_FLEET)


# ── Pure policy matrix ───────────────────────────────────────────────────────

def test_decision_log_grants_operator_identities_only():
    assert rbac.allowed_roles(rbac.DECISION_LOG) == set(_OPERATOR_ROLES)
    for role in _OPERATOR_ROLES:
        assert rbac.can(role, rbac.DECISION_LOG)
    assert not rbac.can(rbac.ROLE_COMPLIANCE, rbac.DECISION_LOG)
    assert not rbac.can(None, rbac.DECISION_LOG)
    assert not rbac.can("ghost", rbac.DECISION_LOG)


def test_webhooks_manage_is_admin_only():
    assert rbac.allowed_roles(rbac.WEBHOOKS_MANAGE) == {rbac.ROLE_ADMIN}
    assert rbac.can(rbac.ROLE_ADMIN, rbac.WEBHOOKS_MANAGE)
    for role in (rbac.ROLE_ENGINEER, rbac.ROLE_FLEET, rbac.ROLE_COMPLIANCE):
        assert not rbac.can(role, rbac.WEBHOOKS_MANAGE)


def test_fleet_assets_manage_is_admin_only():
    assert rbac.allowed_roles(rbac.FLEET_ASSETS_MANAGE) == {rbac.ROLE_ADMIN}
    assert rbac.can(rbac.ROLE_ADMIN, rbac.FLEET_ASSETS_MANAGE)
    for role in (rbac.ROLE_ENGINEER, rbac.ROLE_FLEET, rbac.ROLE_COMPLIANCE):
        assert not rbac.can(role, rbac.FLEET_ASSETS_MANAGE)


def test_personas_do_not_grant_org_write_surface():
    init = (rbac.ACTION_CREATE_TICKET,)  # personas hold no writes (see ui_unified)
    for persona in (rbac.PERSONA_ENGINEER, rbac.PERSONA_FLEET,
                    rbac.PERSONA_EXECUTIVE, rbac.PERSONA_COMPLIANCE):
        for cap in (rbac.DECISION_LOG, rbac.WEBHOOKS_MANAGE, rbac.FLEET_ASSETS_MANAGE):
            assert not rbac.can(persona, cap), (persona, cap)


# ── db.py trust-boundary enforcement (nothing persisted on refusal) ─────────

def test_save_decision_refuses_compliance_and_missing_role(db):
    entry = {"id": "dec-rogue", "cell_id": "B0005", "action": "replace",
             "timestamp": "2026-01-01T00:00", "status": "Pending"}
    for role in (None, "compliance", "ghost"):
        with pytest.raises(db_module.InsufficientRoleError):
            db_module.save_decision(1, dict(entry), caller_role=role)
    assert not any(d["id"] == "dec-rogue" for d in db_module.load_decisions(1))


def test_save_decision_allows_operator_identities(db):
    for i, role in enumerate(_OPERATOR_ROLES):
        db_module.save_decision(1, {"id": f"dec-ok-{role}", "cell_id": "B0005",
                                    "action": "Continue", "timestamp": "2026-01-01",
                                    "status": "Pending"}, caller_role=role)
    ids = [d["id"] for d in db_module.load_decisions(1)]
    assert {f"dec-ok-{r}" for r in _OPERATOR_ROLES} <= set(ids)


def test_webhook_writes_refuse_non_admin(db):
    entry = {"id": "wh-rogue", "name": "Rogue", "url": "https://x", "event_types": [],
             "created_at": "2026-01-01"}
    for role in (None, "engineer", "fleet", "compliance"):
        with pytest.raises(db_module.InsufficientRoleError):
            db_module.save_webhook_subscription(1, dict(entry), caller_role=role)
    assert db_module.get_webhook_subscriptions(1) == []

    db_module.save_webhook_subscription(1, dict(entry), caller_role="admin")
    subs = db_module.get_webhook_subscriptions(1)
    assert [s["id"] for s in subs] == ["wh-rogue"]
    for role in ("engineer", None):
        with pytest.raises(db_module.InsufficientRoleError):
            db_module.delete_webhook_subscription(1, "wh-rogue", caller_role=role)
    # admin can delete
    db_module.delete_webhook_subscription(1, "wh-rogue", caller_role="admin")
    assert db_module.get_webhook_subscriptions(1) == []


def test_fleet_asset_writes_refuse_non_admin(db):
    for role in (None, "engineer", "fleet", "compliance"):
        with pytest.raises(db_module.InsufficientRoleError):
            db_module.create_site(1, "Rogue Site", caller_role=role)
    assert not any(s["name"] == "Rogue Site" for s in db_module.list_sites(1))

    site = db_module.create_site(1, "Real Site", caller_role="admin")
    with pytest.raises(db_module.InsufficientRoleError):
        db_module.create_fleet(1, site["id"], "Rogue Fleet", caller_role="fleet")
    fleet = db_module.create_fleet(1, site["id"], "Real Fleet", caller_role="admin")
    with pytest.raises(db_module.InsufficientRoleError):
        db_module.create_pack(1, fleet["id"], "Rogue Pack", caller_role="engineer")
    pack = db_module.create_pack(1, fleet["id"], "Real Pack", caller_role="admin")
    with pytest.raises(db_module.InsufficientRoleError):
        db_module.add_cell_to_pack(1, pack["id"], "B0005", caller_role="compliance")
    assert db_module.list_pack_cells(1, pack["id"]) == []
    db_module.add_cell_to_pack(1, pack["id"], "B0005", caller_role="admin")
    assert db_module.list_pack_cells(1, pack["id"]) == ["B0005"]


# ── API boundary enforcement ─────────────────────────────────────────────────

def test_decisions_write_denied_for_compliance(client):
    assert client.get("/decisions", headers=_headers("compliance")).status_code == 200
    res = client.post("/decisions", json={"id": "dec-api-1", "cell_id": "B0005",
                                          "action": "replace", "timestamp": "2026-01-01"},
                      headers=_headers("compliance"))
    assert res.status_code == 403
    assert "not allowed to perform" in res.json()["detail"]


def test_decisions_write_allowed_for_operator(client):
    res = client.post("/decisions", json={"id": "dec-api-2", "cell_id": "B0005",
                                          "action": "Continue", "timestamp": "2026-01-01"},
                      headers=_headers("engineer"))
    assert res.status_code == 200
    names = {d["id"] for d in client.get("/decisions", headers=_headers("engineer")).json()}
    assert "dec-api-2" in names


def test_webhooks_writes_denied_for_non_admin(client):
    assert client.get("/webhooks", headers=_headers("engineer")).status_code == 200
    body = {"name": "Slack", "url": "https://hooks.slack.com/x", "event_types": ["TEST_PING"]}
    assert client.post("/webhooks", json=body, headers=_headers("engineer")).status_code == 403
    assert client.post("/webhooks", json=body, headers=_headers("compliance")).status_code == 403


def test_webhooks_create_and_delete_by_admin(client):
    created = client.post("/webhooks", json={"name": "PD", "url": "https://x", "id": "wh-api-1"},
                          headers=_headers("admin"))
    assert created.status_code == 200
    ids = [s["id"] for s in client.get("/webhooks", headers=_headers("admin")).json()]
    assert "wh-api-1" in ids
    assert client.delete("/webhooks/wh-api-1", headers=_headers("admin")).status_code == 200
    ids = [s["id"] for s in client.get("/webhooks", headers=_headers("admin")).json()]
    assert "wh-api-1" not in ids


def test_site_fleet_pack_writes_denied_for_non_admin(client):
    assert client.get("/sites", headers=_headers("engineer")).status_code == 200
    assert client.post("/sites", json={"name": "Rogue"}, headers=_headers("engineer")).status_code == 403
    assert client.post("/sites", json={"name": "Rogue"}, headers=_headers("compliance")).status_code == 403


def test_site_fleet_pack_crud_by_admin(client):
    site = client.post("/sites", json={"name": "Depot 1"}, headers=_headers("admin"))
    assert site.status_code == 200
    site_id = site.json()["id"]
    fleet = client.post(f"/sites/{site_id}/fleets", json={"name": "Fleet A"},
                        headers=_headers("admin"))
    assert fleet.status_code == 200
    fleet_id = fleet.json()["id"]
    pack = client.post(f"/fleets/{fleet_id}/packs", json={"name": "Pack 1"},
                       headers=_headers("admin"))
    assert pack.status_code == 200
    pack_id = pack.json()["id"]

    assert client.post(f"/packs/{pack_id}/cells", json={"cell_id": "B0005", "position": 0},
                       headers=_headers("admin")).status_code == 200
    cells = client.get(f"/packs/{pack_id}/cells", headers=_headers("admin")).json()
    assert cells == ["B0005"]
    assert client.delete(f"/packs/{pack_id}/cells/B0005",
                         headers=_headers("admin")).status_code == 200
    assert client.get(f"/packs/{pack_id}/cells", headers=_headers("admin")).json() == []


def test_read_stays_open_for_all_roles(client):
    for role in ("admin", "engineer", "fleet", "compliance"):
        assert client.get("/sites", headers=_headers(role)).status_code == 200
        assert client.get("/decisions", headers=_headers(role)).status_code == 200
        assert client.get("/webhooks", headers=_headers(role)).status_code == 200
    assert client.post("/decisions", json={"id": "x", "timestamp": "2026-01-01"}).status_code == 401


# ── structural drift guards ─────────────────────────────────────────────────

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SETTINGS_PY = _ROOT / "app" / "_pages" / "_settings_config.py"
_DB_PY = _ROOT / "src" / "db.py"


def test_settings_binds_webhooks_and_fleets_to_granular_caps():
    src = _SETTINGS_PY.read_text(encoding="utf-8")
    # The write surfaces that these sections expose must be gated by the same
    # capability the API boundary enforces -- not a raw admin-string check.
    assert "rbac.can(st.session_state.get(\"auth_role\"), rbac.WEBHOOKS_MANAGE)" in src
    assert "rbac.can(st.session_state.get(\"auth_role\"), rbac.FLEET_ASSETS_MANAGE)" in src


def test_db_enforcement_references_rbac_caps_not_hardcoded_role():
    src = _DB_PY.read_text(encoding="utf-8")
    # db.py is the trust boundary; its gated writes must call _require_cap
    # (which checks rbac.can), and must not hand-roll caller_role == "admin".
    assert "rbac.can(" in src or "_require_cap(" in src
    assert "caller_role == \"admin\"" not in src.replace("_require_admin", "")


def test_db_unit_helpers_delegate_to_registry(db):
    # _require_admin == CAP_SETTINGS_MANAGE equivalence with the registry.
    allowed_by_rbac = {r for r in (rbac.ROLE_ADMIN, rbac.ROLE_ENGINEER,
                                   rbac.ROLE_FLEET, rbac.ROLE_COMPLIANCE, None)
                       if rbac.can(r, rbac.CAP_SETTINGS_MANAGE)}
    assert allowed_by_rbac == {rbac.ROLE_ADMIN}
    db._require_admin(rbac.ROLE_ADMIN, "probe")  # no raise
    with pytest.raises(db_module.InsufficientRoleError):
        db._require_cap(rbac.ROLE_FLEET, rbac.CAP_SETTINGS_MANAGE, "probe")
    with pytest.raises(db_module.InsufficientRoleError):
        db._require_cap(rbac.ROLE_COMPLIANCE, rbac.WEBHOOKS_MANAGE, "probe")
    with pytest.raises(db_module.InsufficientRoleError):
        db._require_cap(rbac.ROLE_COMPLIANCE, rbac.FLEET_ASSETS_MANAGE, "probe")