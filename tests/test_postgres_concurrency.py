"""
Concurrency + soak validation for the PostgreSQL backend.

The Postgres correctness path (SQLite->Postgres migration, schema parity, and
RLS enforcement) is already validated in test_migration_e2e.py. This file
exercises the dimension that round-trip correctness cannot prove on its own:
CONCURRENT multi-user behaviour on a real server — many orgs writing through a
single pooled SQLAlchemy engine, row-lock contention on one shared row, and
row-level security holding under concurrent readers.

It deliberately drives the same db.py write helpers the live app uses
(create_organization_with_admin / save_decision / update_decision /
load_decisions) so the test proves those specific entry points are
thread-safe against Postgres — the exact degree of freedom the SQLite default
(check_same_thread=True, single-writer) never exercised.

Opt-in (CI has no Postgres; a live server is required, provisionable via
scripts/postgres_dev.py):

    # with the dev instance up (scripts/postgres_dev.py start):
    PG_CONCURRENCY=1 DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:54329/battery_platform_test \
        python -m pytest tests/test_postgres_concurrency.py -q
"""

import os
import pathlib
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent

RUN = os.environ.get("PG_CONCURRENCY", "0") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set PG_CONCURRENCY=1 (and DATABASE_URL at a live Postgres) to run"
)

_MIGRATE = _ROOT / "scripts" / "migrate_sqlite_to_postgres.py"
_SEED = _ROOT / "scripts" / "seed_sqlite_for_migration.py"


def _pg_url(db: str) -> str:
    base = os.environ.get("PG_CONCURRENCY_URL") or os.environ.get("DATABASE_URL", "")
    if not base:
        pytest.skip("set DATABASE_URL (or PG_CONCURRENCY_URL) to a live Postgres")
    import urllib.parse
    parts = urllib.parse.urlparse(base)
    return urllib.parse.urlunparse(parts._replace(path="/" + db))


@pytest.fixture(scope="module")
def scratch_db(tmp_path_factory):
    """Fresh DB with the full schema + RLS + non-owner app_tenant role,
    built by running the real migration with --apply-rls on a scratch DB."""
    import sqlalchemy as sa

    admin_url = _pg_url("postgres")
    admin = sa.create_engine(admin_url)
    dbname = "battery_platform_concurrency"
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(sa.text(f"DROP DATABASE IF EXISTS {dbname} WITH (FORCE)"))
        conn.execute(sa.text(f"CREATE DATABASE {dbname}"))
    url = _pg_url(dbname)

    tmp = tmp_path_factory.mktemp("seed")
    sqlite_path = tmp / "seed.db"
    seed_env = dict(os.environ)
    seed_env.pop("DATABASE_URL", None)
    r = subprocess.run([sys.executable, str(_SEED), str(sqlite_path)],
                       capture_output=True, text=True, env=seed_env, timeout=600)
    assert r.returncode == 0, r.stderr

    mgr_env = dict(os.environ)
    mgr_env["DATABASE_URL"] = url
    r = subprocess.run([sys.executable, str(_MIGRATE), "--sqlite", str(sqlite_path), "--apply-rls"],
                       capture_output=True, text=True, env=mgr_env, timeout=600)
    assert r.returncode == 0, r.stderr

    # Point the app's db module at this server so the test drives real helpers.
    import db as db_module
    db_module.DB_URL = url
    db_module.engine = db_module.create_engine(url)
    db_module.Session = db_module.sessionmaker(bind=db_module.engine)

    yield {"url": url, "db_module": db_module}

    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(sa.text(f"DROP DATABASE IF EXISTS {dbname} WITH (FORCE)"))


def _tenant_url(url: str) -> str:
    """Re-point the scratch URL at the non-owner app_tenant role the migration
    creates (RLS is enforced as this role; the owner bypasses RLS)."""
    parts = url.replace("postgresql+psycopg2://postgres:postgres@", "")
    return "postgresql+psycopg2://app_tenant:app_tenant@" + parts


# ── 1. Sequence / org-id contention ──────────────────────────────────────────

N_ORGS = 8


def test_parallel_org_creation_no_seq_collisions(scratch_db):
    """P orgs created on separate threads through one pooled engine get
    distinct, gap-free ids — proving the autoincrement sequences stay correct
    under concurrent nextval (the bug class 'copied ids don't advance the
    sequence' would surface here as a collision / duplicate-key error)."""
    db = scratch_db["db_module"]
    results = []

    def create(i: int):
        results.append(db.create_organization_with_admin(
            f"Soak Org {i}", f"soak_user_{i}", "soak-pass", display_name=f"user{i}",
        ))

    with ThreadPoolExecutor(max_workers=N_ORGS) as ex:
        list(ex.map(create, range(N_ORGS)))

    org_ids = {r["org_id"] for r in results}
    assert len(org_ids) == N_ORGS, f"org-id collision under concurrency: {results}"
    # Seed owns ids 1 and 2 -> concurrent self-service orgs are 3..10, in order.
    assert sorted(org_ids) == list(range(3, 3 + N_ORGS)), sorted(org_ids)


# ── 2. Pooled multi-org writes (dialect/thread-safety guards) ───────────────

PER_ORG_THREADS = 4
DECISIONS_PER_THREAD = 5


def test_pooled_concurrent_writes_across_orgs_no_lost_updates(scratch_db):
    """Many threads writing decisions to many orgs through the SAME pooled
    engine: every row lands exactly once, nothing is lost or duplicated. This
    is the case that would break if the SQLite-tuned dialect guards
    (e.g. check_same_thread assumptions) leaked into the Postgres path."""
    db = scratch_db["db_module"]
    org_ids = [db.create_organization_with_admin(
        f"Write Org {i}", f"write_user_{i}", "p", display_name=f"w{i}",
    )["org_id"] for i in range(N_ORGS)]

    def write_one(org_id: int, k: int):
        for j in range(DECISIONS_PER_THREAD):
            db.save_decision(org_id, {
                "id": f"d-{org_id}-{k}-{j}",
                "cell_id": "B0005",
                "action": "Graded",
                "confidence": 0.9,
                "soh_pct": 88.0 + j,
                "timestamp": "2026-01-01T00:00:00",
                "status": "Pending",
            }, caller_role="admin")

    jobs = [(oid, k) for oid in org_ids for k in range(PER_ORG_THREADS)]
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda t: write_one(*t), jobs))

    expected = PER_ORG_THREADS * DECISIONS_PER_THREAD
    counts = {oid: len(db.load_decisions(oid)) for oid in org_ids}
    assert all(c == expected for c in counts.values()), counts


# ── 3. Row-lock contention on one shared row ─────────────────────────────────

def test_concurrent_update_same_row_no_deadlock(scratch_db):
    """A dozen threads updating ONE decision row concurrently must not deadlock
    or error. Postgres uses row-level locks; last-writer-wins under READ
    COMMITTED, so the final status is one of the written values. This directly
    exercises the 'row-lock contention' risk unique to a real shared DB."""
    db = scratch_db["db_module"]
    org_id = db.create_organization_with_admin(
        "Lock Org", "lock_user", "p", display_name="lo")["org_id"]
    db.save_decision(org_id, {
        "id": "lock-row", "cell_id": "B0005", "action": "Graded",
        "confidence": 0.8, "soh_pct": 90.0, "timestamp": "2026-01-01T00:00:00",
        "status": "start",
    }, caller_role="admin")

    def bump(i: int):
        db.update_decision(org_id, "lock-row", status=f"status-{i}")

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(bump, range(12)))

    rows = db.load_decisions(org_id)
    final = rows[0]["status"]
    assert final in {f"status-{i}" for i in range(12)}, final


# ── 4. RLS isolation holding under concurrent readers ────────────────────────

def test_rls_isolation_under_concurrency(scratch_db):
    """With RLS enforced as the non-owner app_tenant role, concurrent readers
    each scoping to their own org must see ONLY their org's rows — the
    SET LOCAL txn-scoped pattern must not let one org's data leak into another
    even while every session is mid-transaction at once."""
    db = scratch_db["db_module"]
    import sqlalchemy as sa

    tenants = {i: db.create_organization_with_admin(
        f"Tenant Org {i}", f"tenant_user_{i}", "p", display_name=f"t{i}",
    )["org_id"] for i in range(N_ORGS)}

    per = 3
    for i, oid in tenants.items():
        for k in range(per):
            db.save_decision(oid, {
                "id": f"t-{oid}-{k}", "cell_id": "B0005", "action": "Graded",
                "confidence": 0.7, "soh_pct": 91.0 + k,
                "timestamp": "2026-01-01T00:00:00", "status": "Pending",
            }, caller_role="admin")

    engine = sa.create_engine(_tenant_url(scratch_db["url"]))

    def scope_and_count(i: int) -> tuple:
        oid = tenants[i]
        with engine.connect() as conn:
            count = conn.execute(sa.text(
                "SET app.org_id=:oid; SELECT count(*) FROM decisions"
            ).bindparams(oid=str(oid))).scalar()
            return oid, count

    with ThreadPoolExecutor(max_workers=N_ORGS) as ex:
        seen = {oid: c for oid, c in ex.map(scope_and_count, range(N_ORGS))}

    for oid, c in seen.items():
        # Only this org's rows visible (policies hold under concurrency), and
        # all of them are present.
        assert c == per, (oid, c)


# ── 5. A short soak burst ────────────────────────────────────────────────────

def test_soak_burst_mixed_workload(scratch_db):
    """A modest end-to-end burst: many orgs doing create+write+update+read in
    overlapping threads, asserting the engine keeps producing correct results
    (no exhaustion, no cross-org bleed) the whole time."""
    db = scratch_db["db_module"]
    failures: list = []
    lock = threading.Lock()

    def worker(i: int):
        try:
            org = db.create_organization_with_admin(
                f"Soak Burst {i}", f"burst_user_{i}", "p", display_name=f"b{i}")
            oid = org["org_id"]
            for k in range(6):
                db.save_decision(oid, {
                    "id": f"b-{oid}-{k}", "cell_id": "B0005", "action": "Graded",
                    "confidence": 0.6, "soh_pct": 92.0 + k,
                    "timestamp": "2026-01-01T00:00:00", "status": "Pending",
                }, caller_role="admin")
            db.update_decision(oid, f"b-{oid}-3", status="Closed")
            assert len(db.load_decisions(oid)) == 6
        except Exception as e:  # noqa: BLE001
            with lock:
                failures.append((i, repr(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not failures, failures[:5]