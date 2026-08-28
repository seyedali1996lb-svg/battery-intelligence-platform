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

It is NOT validated against a live PostgreSQL server in this repo's CI —
no Postgres instance exists here, and inventing one would be the same
"substituting assumptions for real data" pattern this project documents
elsewhere. Run it against a real server once, then treat it as validated.
Postgres itself must be reachable at ``DATABASE_URL``.

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
      USING (org_id = current_setting('app.org_id')::int);
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import create_engine, inspect, text  # noqa: E402
from sqlalchemy.schema import CreateTable  # noqa: E402

DEFAULT_SQLITE = pathlib.Path(__file__).resolve().parent.parent / "data" / "app.db"

# Tables that hold org-owned rows and therefore get an RLS policy.
_ORG_SCOPED = [
    "users", "organizations", "settings", "decision_logs", "cohort_tags",
    "trajectory_signatures", "trajectory_edges", "experiment_runs",
    "cell_summaries", "webhook_subscriptions", "buyer_profiles",
    "marketplace_matches", "action_items",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default=str(DEFAULT_SQLITE), help="Path to the SQLite DB file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the DDL that WOULD run without connecting to Postgres",
    )
    args = parser.parse_args()

    import os
    pg_url = os.environ.get("DATABASE_URL")
    if not pg_url:
        print("DATABASE_URL is not set — refusing to guess a target server.", file=sys.stderr)
        return 1
    if not pg_url.startswith(("postgresql", "postgres")):
        print(f"DATABASE_URL must point at PostgreSQL, got: {pg_url[:20]}…", file=sys.stderr)
        return 1

    sqlite_path = pathlib.Path(args.sqlite)
    if not sqlite_path.exists():
        print(f"SQLite DB not found: {sqlite_path}", file=sys.stderr)
        return 1

    src = create_engine(f"sqlite:///{sqlite_path}")
    dst = create_engine(pg_url)
    src_insp = inspect(src)

    tables = src_insp.get_table_names()
    if args.dry_run:
        print(f"[dry-run] would copy {len(tables)} tables to {pg_url.split('@')[-1]}")
        print("\n".join(f"  - {t}" for t in tables))
        print("\nRLS DDL to apply after the copy:")
        print(_rls_ddl())
        return 0

    print(f"Copying {len(tables)} tables → PostgreSQL…")
    with dst.begin() as conn:
        for table_name in tables:
            print(f"  {table_name} …", end="", flush=True)
            meta_src = src_insp.get_columns(table_name)
            meta = dst.dialect.has_table(conn, table_name)
            if meta:
                print(" exists, skipping", flush=True)
                continue
            # Create the table from the ORM-visible schema (portable types only —
            # this app uses Integer/String/Float/Text, which map cleanly).
            cols = ",\n  ".join(
                f"{c['name']} {c['type']}"
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

    print("\nDone. Enable row-level security with:")
    print(_rls_ddl())


def _rls_ddl() -> str:
    lines = [
        "-- Per-org row-level security (defense in depth on top of org_id scoping).",
        "-- The app layer should run `SET LOCAL app.org_id = <org>` at the start of",
        "-- each transaction; requests without the setting are then denied by default.",
    ]
    for t in _ORG_SCOPED:
        lines.append(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;")
        lines.append(
            f"CREATE POLICY org_isolation ON {t} "
            "USING (org_id = current_setting('app.org_id', true)::int);"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
