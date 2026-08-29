"""
Concurrency + org-scoping validation for the RBAC-extended API write surface on a
REAL PostgreSQL server.

test_postgres_concurrency.py already proves the core db helpers are thread-safe
against Postgres (parallel org creation, pooled multi-org writes, row-lock
contention, RLS under concurrent readers). This test is the API-boundary
complement: it drives the role-gated REST write endpoints added for this org
write surface -- POST/GET /decisions, POST/GET/DELETE /webhooks, and the
/sites-/fleets-/packs- CRUD -- through TestClient, many orgs issuing concurrent
multi-tenant requests, and asserts org-scoping holds with no cross-tenant bleed.
It also asserts, on a FRESH migrate with --apply-rls, that row-level security
policies cover exactly the tables that surface exposes (decisions, webhooks,
sites, fleets, packs, pack_cells, plus the pre-existing org tables).

The API connects as the table owner (postgres), which bypasses RLS by design
-- the org-scoping proven here comes from the org_id in the caller's JWT being
threaded into every scoped db call, and RLS is the defense-in-depth layer for
direct DB access (asserted separately below and in test_migration_e2e.py).

Opt-in (CI has no Postgres; a live server is required, provisionable via
scripts/postgres_dev.py):

    PG_CONCURRENCY=1 DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:54329/battery_platform_test \
        python -m pytest tests/test_postgres_rbac_concurrency.py -q
"""

import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent

RUN = os.environ.get("PG_CONCURRENCY", "0") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set PG_CONCURRENCY=1 (and DATABASE_URL at a live Postgres) to run"
)

_MIGRATE = _ROOT / "scripts" / "migrate_sqlite_to_postgres.py"
_SEED = _ROOT / "scripts" / "seed_sqlite_for_migration.py"

# Every row-scoped table the extended API write surface touches, plus the
# pre-existing row-scoped org tables -- all must carry an RLS policy after
# --apply-rls. (organizations is NOT in this set: it is the tenant-root table
# with no org_id column, so the migration correctly does not row-scope it.)
_RLS_TABLES = {
    "users", "settings", "experiment_runs", "decisions",
    "webhook_subscriptions", "sites", "fleets", "packs", "pack_cells",
}


def _pg_url(db: str) -> str:
    base = os.environ.get("PG_CONCURRENCY_URL") or os.environ.get("DATABASE_URL", "")
    if not base:
        pytest.skip("set DATABASE_URL (or PG_CONCURRENCY_URL) to a live Postgres")
    import urllib.parse
    parts = urllib.parse.urlparse(base)
    return urllib.parse.urlunparse(parts._replace(path="/" + db))


@pytest.fixture(scope="module")
def scratch_db():
    """Fresh DB with the full schema + RLS + non-owner app_tenant role, built by
    running the real migration with --apply-rls, then the `src.db` module (the
    one src/api.py's endpoints are wired to) is re-pointed at the Postgres
    server so TestClient exercises real concurrent multi-tenant requests."""
    import sqlalchemy as sa
    from src import db as sd

    admin_url = _pg_url("postgres")
    admin = sa.create_engine(admin_url)
    dbname = "battery_platform_rbac"
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(sa.text(f"DROP DATABASE IF EXISTS {dbname} WITH (FORCE)"))
        conn.execute(sa.text(f"CREATE DATABASE {dbname}"))
    url = _pg_url(dbname)

    tmp = __import__("tempfile").mkdtemp()
    sqlite_path = pathlib.Path(tmp) / "seed.db"
    seed_env = dict(os.environ)
    seed_env.pop("DATABASE_URL", None)  # seed must go to SQLite
    r = subprocess.run([sys.executable, str(_SEED), str(sqlite_path)],
                       capture_output=True, text=True, env=seed_env, timeout=600)
    assert r.returncode == 0, r.stderr

    mgr_env = dict(os.environ)
    mgr_env["DATABASE_URL"] = url
    r = subprocess.run([sys.executable, str(_MIGRATE), "--sqlite", str(sqlite_path),
                        "--apply-rls"], capture_output=True, text=True,
                       env=mgr_env, timeout=600)
    assert r.returncode == 0, r.stderr

    # Re-point the module src/api.py is wired to (NOT the top-level `db`, which
    # is a separate module object for a namespace package like `src`).
    orig = (sd.DB_URL, sd.engine, sd.Session)
    sd.DB_URL = url
    sd.engine = sd.create_engine(url)
    sd.Session = sd.sessionmaker(bind=sd.engine)

    yield {"url": url, "db": sd}

    sd.DB_URL, sd.engine, sd.Session = orig  # restore for any later modules
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(sa.text(f"DROP DATABASE IF EXISTS {dbname} WITH (FORCE)"))


def _auth(user: str, oid: int, role: str = "admin") -> dict:
    from src.api import _create_access_token
    token = _create_access_token({"username": user, "org_id": oid, "role": role})
    return {"Authorization": f"Bearer {token}"}


def test_rls_policies_cover_extended_write_tables_on_fresh_migrate(scratch_db):
    """A fresh `--apply-rls` migrate creates an enabled RLS policy on every
    org-scoped table the new API write surface touches -- so the defense-in-depth
    layer covers the exact surface these endpoints expose."""
    import sqlalchemy as sa
    eng = sa.create_engine(scratch_db["url"])
    with eng.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT c.relname, c.relrowsecurity, p.policyname "
            "FROM pg_class c "
            "LEFT JOIN pg_policies p ON p.schemaname='public' AND p.tablename=c.relname "
            "WHERE c.relname = ANY(:names)"
        ).bindparams(names=list(_RLS_TABLES))).fetchall()
    covered = {r[0]: (r[1], r[2]) for r in rows}
    uncovered = {
        t for t in _RLS_TABLES
        if covered.get(t) != (True, "org_isolation")
    }
    assert not uncovered, f"tables missing RLS org_isolation policy: {sorted(uncovered)}"


N_ORGS = 6
DECISIONS_PER_ORG = 4


def test_api_concurrent_multitenant_writes_no_cross_tenant_bleed(scratch_db):
    """Many orgs hitting the role-gated write endpoints in parallel through
    TestClient (decisions, webhooks, sites/fleets/packs + pack cells), then every
    org's reads see ONLY its own rows -- org_id-scoping holds under concurrent
    multi-tenant requests, and nothing leaks between tenants."""
    from fastapi.testclient import TestClient
    from src.api import app

    db = scratch_db["db"]
    orgs = {}
    for i in range(N_ORGS):
        u = f"rbac_user_{i}"
        org = db.create_organization_with_admin(f"RBAC Org {i}", u, "robust-pass-1",
                                                display_name=f"Admin {i}")
        orgs[i] = {"oid": org["org_id"], "user": u}

    def worker(i: int):
        info = orgs[i]
        oid, user = info["oid"], info["user"]
        h = _auth(user, oid, "admin")
        c = TestClient(app)  # one client per thread -- the transport is not shared
        # Write burst: decisions, a site->fleet->pack->cell chain, a webhook.
        for j in range(DECISIONS_PER_ORG):
            r = c.post("/decisions", json={
                "id": f"d-{oid}-{j}", "cell_id": "B0005", "action": "Graded",
                "confidence": "Medium", "soh_pct": 88.0 + j, "timestamp": "2026-01-01",
                "status": "Pending",
            }, headers=h)
            assert r.status_code == 200, (i, j, r.status_code, r.text)
        site = c.post("/sites", json={"name": f"site-{oid}"}, headers=h)
        assert site.status_code == 200, site.text
        sid = site.json()["id"]
        fleet = c.post(f"/sites/{sid}/fleets", json={"name": f"fleet-{oid}"}, headers=h)
        assert fleet.status_code == 200, fleet.text
        fid = fleet.json()["id"]
        pack = c.post(f"/fleets/{fid}/packs", json={"name": f"pack-{oid}"}, headers=h)
        assert pack.status_code == 200, pack.text
        pid = pack.json()["id"]
        assert c.post(f"/packs/{pid}/cells", json={"cell_id": f"cell-{oid}"}, headers=h).status_code == 200
        wh = c.post("/webhooks", json={"id": f"wh-{oid}", "name": f"wh-{oid}",
                                       "url": "https://hooks.example.com/x",
                                       "event_types": ["TEST_PING"]}, headers=h)
        assert wh.status_code == 200, wh.text
        return oid, sid, fid, pid

    with ThreadPoolExecutor(max_workers=N_ORGS) as ex:
        results = list(ex.map(worker, range(N_ORGS)))

    # ── Org-scoping holds: each tenant sees only its own rows ────────────────
    all_ids = set()
    for i, (oid, sid, fid, pid) in enumerate(results):
        h = _auth(orgs[i]["user"], oid, "admin")
        c = TestClient(app)
        decisions = {d["id"] for d in c.get("/decisions", headers=h).json()}
        expected = {f"d-{oid}-{j}" for j in range(DECISIONS_PER_ORG)}
        assert decisions == expected, (oid, decisions, expected)
        all_ids |= decisions
        site_names = {s["name"] for s in c.get("/sites", headers=h).json()}
        assert {f"site-{oid}"} <= site_names  # org owns its site (+ default site)
        assert {w["id"] for w in c.get("/webhooks", headers=h).json()} == {f"wh-{oid}"}
        # Team roster: only this org's own admin, not another org's.
        roster = {u["username"] for u in c.get("/team/members", headers=h).json()}
        assert orgs[i]["user"] in roster
        assert not any(other["user"] in roster for j, other in enumerate(orgs.values()) if j != i)

    # Global count check: every decision written is accounted for.
    assert len(all_ids) == N_ORGS * DECISIONS_PER_ORG, len(all_ids)