"""
Copy the demo SQLite database into PostgreSQL and print the row-level
security (RLS) DDL for the multi-tenant isolation layer.

Why this exists
---------------
The demo deployment runs on SQLite (single process, single user) via the
``DB_URL`` default in src/db.py. The Production Readiness Roadmap
(docs/history.md) names PostgreSQL as the path for concurrent multi-user
access at scale. This script is the documented, executable version of that
path: it reads the SQLite schema + rows through the same SQLAlchemy ORM
metadata the app uses, recreates them in PostgreSQL, and copies every row
(org-scoped tables included, since ``org_id`` is a real column everywhere).

Validation status (2026-08-28): validated end-to-end against a real
PostgreSQL 18.4 server — the project-local dev instance provisioned by
scripts/postgres_dev.py (bundled binaries via the postgresql-binaries pip
package, data dir .tools/pgdata). The live run caught and fixed five real
bugs the unit tests couldn't (plain INTEGER PKs with no autoincrement
sequence, identity sequences not advanced past copied ids, String-id tables
breaking the sequence sync, the RLS policy throwing on the empty-string
GUC state after RESET, and --apply-rls not being idempotent). The full
pytest suite passes against PostgreSQL (see docs/history.md). This repo's
CI still has no Postgres server, so the live validation is opt-in via
``PG_VALIDATE=1 python -m pytest tests/test_migration_e2e.py``; Postgres
itself must be reachable at ``DATABASE_URL``.

Usage
-----
    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/battery_platform \\
        python scripts/migrate_sqlite_to_postgres.py [--sqlite data/app.db]

After the copy, enable row-level security so an org can never see another
org's rows even if a bug (or a raw SQL statement) forgets to scope by
``org_id`` — the policies below are printed for you to apply, and the app
layer should set ``app.org_id`` via ``SET LOCAL`` at the start of each
transaction (SQLAlchemy event listener) so the current org is always
known to the database:

    ALTER TABLE <table> ENABLE ROW LEVEL SECURITY;
    CREATE POLICY org_isolation ON <table>
      USING (org_id = NULLIF(current_setting('app.org_id', true), '')::int);
"""

from __future__ import annotations

import argparse
import pathlib
import sys as _sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
import _paths  # noqa: F402

from sqlalchemy import create_engine, inspect, text  # noqa: E402
from sqlalchemy.schema import CreateTable  # noqa: E402

DEFAULT_SQLITE = pathlib.Path(__file__).resolve().parent.parent / "data" / "app.db"

# Tables that hold org-owned rows and therefore get an RLS policy. Derived
# from the live schema (any table with an org_id column) rather than
# hardcoded, so a future table automatically joins the RLS set.
def _org_scoped_tables(src_insp) -> list:
    return sorted(
        t for t in src_insp.get_table_names()
        if any(c["name"] == "org_id" for c in src_insp.get_columns(t))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default=str(DEFAULT_SQLITE), help="Path to the SQLite DB file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the DDL that WOULD run without connecting to Postgres",
    )
    parser.add_argument(
        "--apply-rls", action="store_true",
        help="Also ENABLE row-level security + create the per-org policies on the "
        "org-scoped tables, and create a limited 'app_tenant' role with SELECT "
        "grants so RLS enforcement can actually be exercised (the migration "
        "user is the table owner, and owners/superusers bypass RLS).",
    )
    args = parser.parse_args()

    import os
    pg_url = os.environ.get("DATABASE_URL")
    if not pg_url:
        print("DATABASE_URL is not set — refusing to guess a target server.", file=sys.stderr)
        return 1
    if not pg_url.startswith(("postgresql", "postgres")):
        print(f"DATABASE_URL must point at PostgreSQL, got: {pg_url[:20]}...", file=sys.stderr)
        return 1

    sqlite_path = pathlib.Path(args.sqlite)
    if not sqlite_path.exists():
        print(f"SQLite DB not found: {sqlite_path}", file=sys.stderr)
        return 1

    src = create_engine(f"sqlite:///{sqlite_path}")
    dst = create_engine(pg_url)
    src_insp = inspect(src)

    tables = src_insp.get_table_names()
    org_scoped = _org_scoped_tables(src_insp)
    if args.dry_run:
        print(f"[dry-run] would copy {len(tables)} tables to {pg_url.split('@')[-1]}")
        print("\n".join(f"  - {t}" for t in tables))
        print(f"\nRLS DDL for {len(org_scoped)} org-scoped tables to apply after the copy:")
        print(_rls_ddl_comment())
        print("\n".join(_rls_ddl(org_scoped)))
        return 0

    print(f"Copying {len(tables)} tables -> PostgreSQL...")
    with dst.begin() as conn:
        for table_name in tables:
            print(f"  {table_name} ...", end="", flush=True)
            meta_src = src_insp.get_columns(table_name)
            meta = dst.dialect.has_table(conn, table_name)
            if meta:
                print(" exists, skipping", flush=True)
                continue
            # Create the table from the ORM-visible schema (portable types only —
            # this app uses Integer/String/Float/Text, which map cleanly). A
            # single-column integer 'id' primary key becomes SERIAL so
            # autoincrement keeps working after the copy (a plain INTEGER PK
            # would reject every later INSERT that omits the id).
            pk_cols = set(src_insp.get_pk_constraint(table_name).get("constrained_columns") or [])
            cols = ",\n  ".join(
                f"{c['name']} SERIAL"
                if (c["name"] == "id" and pk_cols == {"id"} and str(c["type"]).upper().startswith("INTEGER"))
                else f"{c['name']} {c['type']}"
                for c in meta_src
            )
            conn.execute(text(f"CREATE TABLE {table_name} ({cols})"))
            with src.connect() as src_conn:
                rows = src_conn.execute(text(f"SELECT * FROM {table_name}")).mappings().all()
            if rows:
                col_names = list(rows[0].keys())
                placeholders = ", ".join(":" + c for c in col_names)
                col_list = ", ".join(col_names)
                conn.execute(
                    text(f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})"),
                    [dict(r) for r in rows],
                )
            print(f" {len(rows)} rows", flush=True)

    # Sync identity sequences past the largest copied id. Copied rows keep
    # their explicit ids (demo org = 1, self-service orgs = 2+), so without
    # this the next autoincrement insert would collide with an existing row
    # (a real bug caught by the end-to-end Postgres validation).
    print("Syncing identity sequences...")
    with dst.begin() as conn:
        for table_name in tables:
            # Only INTEGER id columns carry a Postgres serial sequence;
            # String-id tables (buyer_profiles, marketplace_matches, ...)
            # have no sequence and must be skipped.
            id_cols = [
                c for c in src_insp.get_columns(table_name)
                if c["name"] == "id" and str(c["type"]).upper().startswith("INTEGER")
            ]
            if not id_cols:
                continue
            seq_sql = f"pg_get_serial_sequence('{table_name}', 'id')"
            # setval(seq, n, false) makes the NEXT value n — max(id)+1 when
            # rows exist, 1 for an empty table.
            row = conn.execute(text(
                f"SELECT setval({seq_sql}, "
                f"COALESCE((SELECT max(id) FROM {table_name}), 0) + 1, false) "
                f"FROM (SELECT 1) t WHERE {seq_sql} IS NOT NULL"
            )).first()
            if row is not None:
                print(f"  {table_name}.id sequence next -> {row[0]}")

    if args.apply_rls:
        print("\nApplying row-level security policies...")
        with dst.begin() as conn:
            for stmt in _rls_ddl(org_scoped):
                conn.execute(text(stmt))
        with dst.begin() as conn:
            conn.execute(text("DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='app_tenant') THEN CREATE ROLE app_tenant LOGIN PASSWORD 'app_tenant'; END IF; END $$;"))
            conn.execute(text("GRANT USAGE ON SCHEMA public TO app_tenant;"))
            for t in org_scoped:
                conn.execute(text(f"GRANT SELECT ON {t} TO app_tenant;"))
        print("\n[ok] RLS applied. Validate enforcement as a non-owner role, e.g.:")
        print("    SET app.org_id = '1';")
        print("    SELECT count(*) FROM decisions;   -- org 1's rows only")
        print("    RESET app.org_id;")
        print("    SELECT count(*) FROM decisions;   -- 0 (setting missing -> denied)")
    else:
        print("\nDone. Enable row-level security with (or re-run with --apply-rls):")
        print(_rls_ddl_comment())
        print("\n".join(_rls_ddl(org_scoped)))


def _rls_ddl_comment() -> str:
    return (
        "-- Per-org row-level security (defense in depth on top of org_id scoping).\n"
        "-- The app layer should run `SET LOCAL app.org_id = <org>` at the start of\n"
        "-- each transaction; requests without the setting are then denied by default."
    )


def _rls_ddl(org_scoped: list) -> list:
    """Executable RLS statements (one per element — no comments, no embedded
    semicolons, so callers can run them one at a time)."""
    stmts = []
    for t in org_scoped:
        stmts.append(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
        # DROP IF EXISTS keeps --apply-rls idempotent (re-runnable).
        stmts.append(f"DROP POLICY IF EXISTS org_isolation ON {t}")
        # NULLIF('', '') is essential, not cosmetic: for a custom GUC,
        # current_setting() returns an EMPTY STRING (not NULL) after RESET or
        # after a SET LOCAL transaction ends — the bare ::int cast would then
        # throw and deny every query. NULLIF normalizes '' -> NULL -> denied.
        stmts.append(
            f"CREATE POLICY org_isolation ON {t} "
            "USING (org_id = NULLIF(current_setting('app.org_id', true), '')::int)"
        )
    return stmts


if __name__ == "__main__":
    raise SystemExit(main())
