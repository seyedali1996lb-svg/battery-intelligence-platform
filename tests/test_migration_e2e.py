"""
End-to-end validation of scripts/migrate_sqlite_to_postgres.py against a
REAL PostgreSQL server (the production-readiness batch's Postgres path —
see docs/history.md). This is opt-in because it needs a running server:

    # with the project-local dev instance up (scripts/postgres_dev.py):
    PG_VALIDATE=1 DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:54329/battery_platform_test \\
        python -m pytest tests/test_migration_e2e.py -q

It seeds a deterministic two-org SQLite DB (scripts/seed_sqlite_for_migration.py),
migrates it into a scratch Postgres database, asserts full per-table row
parity, applies --apply-rls, and verifies the row-level-security isolation
as a non-owner role (org 1 sees only its rows; org 2 only its own; no
setting -> zero rows; autoincrement keeps working after the copy).

The sqlite-side unit checks live in tests/test_db_postgres_readiness.py;
this is the live-server complement that cannot run in CI (no Postgres
there) — hence the explicit PG_VALIDATE=1 gate, same opt-in pattern as the
RUN_PERF_TESTS guards.
"""

import os
import pathlib
import sqlite3
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent

RUN = os.environ.get("PG_VALIDATE", "0") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set PG_VALIDATE=1 (and point DATABASE_URL at a live Postgres) to run"
)

_SEED = _ROOT / "scripts" / "seed_sqlite_for_migration.py"
_MIGRATE = _ROOT / "scripts" / "migrate_sqlite_to_postgres.py"
_PG_BIN = _ROOT / ".tools" / "postgres" / "postgresql-18.4.0-x86_64-pc-windows-msvc" / "bin"

_ORG_SCOPED = {"organizations", "users", "settings", "experiment_runs", "decisions"}


def _run(args, env=None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, env=env, timeout=600, **kw)


def _pg_url(db: str) -> str:
    base = os.environ.get("PG_VALIDATE_URL") or os.environ.get("DATABASE_URL", "")
    if not base:
        pytest.skip("set DATABASE_URL (or PG_VALIDATE_URL) to a live Postgres")
    # Re-point at the scratch database name.
    import urllib.parse
    parts = urllib.parse.urlparse(base)
    return urllib.parse.urlunparse(parts._replace(path="/" + db))


@pytest.fixture(scope="module")
def scratch_db():
    import sqlalchemy as sa

    admin_url = _pg_url("postgres")
    admin = sa.create_engine(admin_url)
    dbname = "battery_platform_e2e"
    # DROP/CREATE DATABASE cannot run inside a transaction block.
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(sa.text(f"DROP DATABASE IF EXISTS {dbname}"))
        conn.execute(sa.text(f"CREATE DATABASE {dbname}"))
    yield _pg_url(dbname)
    # WITH (FORCE): terminate any pooled connections the test left open
    # (Postgres 13+), otherwise DROP fails with "database is being accessed".
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(sa.text(f"DROP DATABASE IF EXISTS {dbname} WITH (FORCE)"))


def _seed_sqlite(tmp_path: pathlib.Path) -> pathlib.Path:
    sqlite_path = tmp_path / "seed.db"
    env = dict(os.environ)
    env.pop("DATABASE_URL", None)  # seed must go to SQLite
    r = _run([sys.executable, str(_SEED), str(sqlite_path)], env=env)
    assert r.returncode == 0, r.stderr
    return sqlite_path


def _sqlite_counts(path: pathlib.Path) -> dict:
    conn = sqlite3.connect(str(path))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]
    counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    conn.close()
    return counts


def _pg_counts(url: str) -> dict:
    import sqlalchemy as sa
    eng = sa.create_engine(url)
    with eng.connect() as conn:
        tables = [r[0] for r in conn.execute(
            sa.text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        )]
        counts = {t: conn.execute(sa.text(f"SELECT count(*) FROM {t}")).scalar() for t in tables}
    return counts


def test_migration_parity_and_rls(scratch_db, tmp_path):
    sqlite_path = _seed_sqlite(tmp_path)

    # ── 1. Migrate + apply RLS ──────────────────────────────────────────────
    env = dict(os.environ)
    env["DATABASE_URL"] = scratch_db
    r = _run([sys.executable, str(_MIGRATE), "--sqlite", str(sqlite_path), "--apply-rls"], env=env)
    assert r.returncode == 0, r.stderr
    assert "RLS applied" in r.stdout

    # ── 2. Full per-table row parity ────────────────────────────────────────
    sq = _sqlite_counts(sqlite_path)
    pg = _pg_counts(scratch_db)
    assert set(sq) <= set(pg), f"tables missing in postgres: {set(sq) - set(pg)}"
    mismatches = {t: (sq[t], pg[t]) for t in sq if sq[t] != pg[t]}
    assert not mismatches, f"row-count mismatches: {mismatches}"

    # ── 3. RLS enforcement as the non-owner app_tenant role ─────────────────
    import sqlalchemy as sa
    url = scratch_db.replace("/battery_platform_e2e", "/battery_platform_e2e")
    tenant_url = url  # same host/creds; role switched below via user override
    parts = url.replace("postgresql+psycopg2://postgres:postgres@", "")
    tenant_url = "postgresql+psycopg2://app_tenant:app_tenant@" + parts
    tenant = sa.create_engine(tenant_url)

    def _tenant_count(sql: str):
        with tenant.connect() as conn:
            return conn.execute(sa.text(sql)).scalar()

    assert _tenant_count("SET app.org_id='1'; SELECT count(*) FROM decisions") == 1
    assert _tenant_count("SET app.org_id='2'; SELECT count(*) FROM decisions") == 1
    # No setting -> RLS denies by default (zero rows, not an error). RESET
    # first because SET persists per session and SQLAlchemy reuses pooled
    # connections (the production pattern is SET LOCAL inside a transaction
    # precisely so the setting can't leak across pooled connections).
    assert _tenant_count("RESET app.org_id; SELECT count(*) FROM decisions") == 0

    # ── 4. Autoincrement survives the copy (sequence sync) ──────────────────
    import db as db_module
    db_module.DB_URL = scratch_db
    db_module.engine = db_module.create_engine(scratch_db)
    db_module.Session = db_module.sessionmaker(bind=db_module.engine)
    org = db_module.create_organization_with_admin("E2E Post-Migration", "e2e_admin", "e2e-pass-1")
    assert org["org_id"] == 3  # demo org = 1, seed org = 2 -> next must be 3
