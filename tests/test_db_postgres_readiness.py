"""
PostgreSQL-readiness tests (perf/production batch, docs/history.md).

src/db.py runs on SQLite by default and on PostgreSQL when DATABASE_URL is
set; scripts/migrate_sqlite_to_postgres.py is the documented copy path.
No live Postgres server exists in this repo, so these tests cover what is
testable without one: the SQLite side stays fully green (covered by the
whole suite), the DATABASE_URL wiring resolves a postgres engine without
falling back to SQLite, and the migration script's --dry-run (which only
touches the SQLite side + prints RLS DDL) works end to end.
"""

import os
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _tiny_sqlite(path: pathlib.Path) -> None:
    """Create a minimal SQLite DB with one org-scoped table + one row."""
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE organizations (id INTEGER PRIMARY KEY, slug TEXT, name TEXT)")
    conn.execute("INSERT INTO organizations (slug, name) VALUES ('demo-org', 'Demo Org')")
    conn.commit()
    conn.close()


def test_database_url_env_builds_postgres_engine(monkeypatch):
    """DATABASE_URL=postgresql… must yield a postgres engine (not SQLite), and the
    sqlite-only check_same_thread kwarg must not be passed to it. Run in a
    subprocess so the module-level engine in src/db.py isn't re-created in
    this test process (which other tests hold references to)."""
    code = (
        "import os, sys; "
        "sys.path.insert(0, r'%s'); "
        "os.environ['DATABASE_URL'] = 'postgresql+psycopg2://u:p@localhost:5432/db'; "
        "import db; "
        "assert db.DB_URL.startswith('postgresql'), db.DB_URL; "
        "assert db.engine.url.get_backend_name() == 'postgresql'; "
        "print('OK')"
    ) % (_ROOT / "src")
    env = dict(os.environ)
    env["DATABASE_URL"] = "postgresql+psycopg2://u:p@localhost:5432/db"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_migration_script_dry_run(tmp_path, monkeypatch):
    """--dry-run inspects the SQLite side and prints RLS DDL without ever
    connecting to PostgreSQL (works even with no server reachable)."""
    sqlite_path = tmp_path / "app.db"
    _tiny_sqlite(sqlite_path)

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@localhost:5432/db")
    out = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "migrate_sqlite_to_postgres.py"),
            "--dry-run",
            "--sqlite",
            str(sqlite_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "organizations" in out.stdout
    assert "ENABLE ROW LEVEL SECURITY" in out.stdout
    assert "current_setting('app.org_id'" in out.stdout


def test_migration_script_refuses_without_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    sqlite_path = tmp_path / "app.db"
    _tiny_sqlite(sqlite_path)
    out = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "migrate_sqlite_to_postgres.py"),
            "--sqlite",
            str(sqlite_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 1
    assert "DATABASE_URL" in out.stderr
