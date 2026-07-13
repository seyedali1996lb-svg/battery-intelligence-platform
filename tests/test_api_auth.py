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


# ---------------------------------------------------------------------------
# Org-scoping — each org's own uploaded ("My Data") fleet, merged on top of
# the shared global reference-cell fleet, via bundle_cache.load_tenant_bundle().
# ---------------------------------------------------------------------------

def _fake_org_triple(cell_id: str, soh: float):
    """A minimal (featured_dfs, bundle, split_cycles) triple, same shape
    bundle_cache.load_tenant_bundle() returns for a real uploaded fleet."""
    import pandas as pd
    df = pd.DataFrame({
        "cycle_number": [1, 2], "soh_pct": [100.0, soh],
        "capacity_ah": [2.0, 2.0 * soh / 100], "rul_pred": [400, 380],
        "fade_rate_30cy": [0.001, 0.001], "resistance_normalized": [1.0, 1.02],
        "coulombic_efficiency": [0.999, 0.998],
    })
    bundle = {"metrics": {"rul_reliable": True, "per_cell_rul_reliable": {cell_id: True}}}
    return ({cell_id: df}, bundle, 1)


def test_get_featured_dfs_merges_org_upload_on_top_of_global(monkeypatch):
    import api as api_module
    monkeypatch.setattr(api_module, "_get_org_bundle", lambda org_id: _fake_org_triple("ORG1-CELL", 92.0) if org_id else None)

    fdfs_no_org = api_module._get_featured_dfs(None)
    fdfs_with_org = api_module._get_featured_dfs(1)

    assert "ORG1-CELL" not in fdfs_no_org
    assert "ORG1-CELL" in fdfs_with_org
    # Global reference cells are still present alongside the org's own cell.
    assert set(fdfs_no_org.keys()) <= set(fdfs_with_org.keys())


def test_get_bundles_adds_upload_key_when_org_has_uploaded_data(monkeypatch):
    import api as api_module
    monkeypatch.setattr(api_module, "_get_org_bundle", lambda org_id: _fake_org_triple("ORG1-CELL", 92.0) if org_id else None)

    bundles_no_org = api_module._get_bundles(None)
    bundles_with_org = api_module._get_bundles(1)

    assert "upload" not in bundles_no_org
    assert "upload" in bundles_with_org


def test_cross_org_isolation_uploaded_cells_never_leak(monkeypatch):
    """Two orgs' uploaded fleets must never mix — the actual acceptance
    criterion for org-scoping, mirroring tests/test_db.py's cross-org test."""
    import api as api_module

    def fake_get_org_bundle(org_id):
        if org_id == 111:
            return _fake_org_triple("ORG111-CELL", 95.0)
        if org_id == 222:
            return _fake_org_triple("ORG222-CELL", 85.0)
        return None

    monkeypatch.setattr(api_module, "_get_org_bundle", fake_get_org_bundle)

    fdfs_111 = api_module._get_featured_dfs(111)
    fdfs_222 = api_module._get_featured_dfs(222)

    assert "ORG111-CELL" in fdfs_111
    assert "ORG222-CELL" not in fdfs_111
    assert "ORG222-CELL" in fdfs_222
    assert "ORG111-CELL" not in fdfs_222


def test_cells_endpoint_returns_merged_cells_for_org_with_upload(monkeypatch):
    """End-to-end: the real /cells endpoint, called with a real demo-org
    bearer token, reflects the merge — not just the helper functions."""
    import api as api_module
    monkeypatch.setattr(api_module, "_get_org_bundle", lambda org_id: _fake_org_triple("DEMO-ORG-UPLOAD", 88.0))

    login_resp = client.post("/auth/login", json={"username": DEMO_USER, "password": DEMO_PASSWORD})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/cells", headers=headers)
    assert resp.status_code == 200
    cells = set(resp.json()["cells"])
    assert "DEMO-ORG-UPLOAD" in cells

    resp = client.get("/cells/DEMO-ORG-UPLOAD", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["cell_id"] == "DEMO-ORG-UPLOAD"
    assert resp.json()["rul_reliable"] is True


# ---------------------------------------------------------------------------
# _load_bundle_from_disk_cache() — _get_bundle()'s fallback when `main` (the
# Streamlit app) can't be imported, e.g. a bare `uvicorn src.api:app` process.
# Regression test: this fallback used to call a nonexistent load_bundle()
# (an uncaught ImportError) and, separately, would have produced a dict
# shape ({"severson": ..., "nasa": ...}) that _get_featured_dfs()/
# _get_bundles() don't actually read from (they look for "featured_dfs"/
# "bundles" keys) — so even a working fallback would have silently served
# zero cells. Both fixed; both covered here.
#
# _load_bundle_from_disk_cache() does its `load_cached_unchecked` import
# locally (inside the function, both `from src.bundle_cache import ...` and
# a bare `from bundle_cache import ...` fallback), so these tests patch the
# attribute on whichever of those two already-imported module objects is
# reachable, rather than guessing which import path the function will take.
# ---------------------------------------------------------------------------

def _patch_load_cached_unchecked(monkeypatch, fake_fn):
    import bundle_cache as bc_module
    monkeypatch.setattr(bc_module, "load_cached_unchecked", fake_fn)
    try:
        import src.bundle_cache as src_bc_module
        monkeypatch.setattr(src_bc_module, "load_cached_unchecked", fake_fn)
    except ImportError:
        pass


def test_load_bundle_from_disk_cache_shapes_dict_from_per_source_triples(monkeypatch):
    import api as api_module

    fake_cache = {
        "nasa": ({"metrics": {"soh_r2": 0.9}}, {"B0005": "nasa_df"}, {"B0005": 80}),
        "severson": ({"metrics": {"soh_r2": 0.8}}, {"S-b1c2": "severson_df"}, {"S-b1c2": 400}),
    }
    _patch_load_cached_unchecked(monkeypatch, fake_cache.get)

    result = api_module._load_bundle_from_disk_cache()

    assert set(result.keys()) == {"featured_dfs", "bundles"}
    assert result["featured_dfs"] == {"B0005": "nasa_df", "S-b1c2": "severson_df"}
    assert set(result["bundles"]) == {"nasa", "severson"}
    assert result["bundles"]["nasa"]["metrics"]["soh_r2"] == 0.9


def test_load_bundle_from_disk_cache_returns_empty_shape_when_nothing_cached(monkeypatch):
    import api as api_module

    _patch_load_cached_unchecked(monkeypatch, lambda key: None)

    result = api_module._load_bundle_from_disk_cache()

    assert result == {"featured_dfs": {}, "bundles": {}}


def test_get_bundle_falls_back_without_crashing_when_main_unimportable(monkeypatch):
    """The actual regression: _get_bundle() must not raise ImportError from
    inside its own except-ImportError branch."""
    import api as api_module
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "main":
            raise ImportError("simulated: no Streamlit app in this process")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(api_module, "_bundle_cache", None)
    monkeypatch.setattr(
        api_module, "_load_bundle_from_disk_cache",
        lambda: {"featured_dfs": {}, "bundles": {}},
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = api_module._get_bundle()  # must not raise
    assert result == {"featured_dfs": {}, "bundles": {}}
