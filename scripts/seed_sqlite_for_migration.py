"""
Seed a deterministic SQLite DB for validating the Postgres migration
(scripts/migrate_sqlite_to_postgres.py). Two organizations with genuinely
different org-scoped rows (settings, decision logs, an experiment run), so
both the row-count copy AND the row-level-security isolation between orgs
are testable after the migration.

Usage
-----
    python scripts/seed_sqlite_for_migration.py <sqlite_path>

The Demo Org (id 1, seeded by db.init_db()) gets extra org-scoped rows;
a second org ("Migrate Test Co", id 2) is created with its own admin user
and its own rows. Every org-scoped table the migration script lists gets
at least one row per org where the app's public API allows it.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import db  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/seed_sqlite_for_migration.py <sqlite_path>", file=sys.stderr)
        return 1
    target = pathlib.Path(sys.argv[1])
    if target.exists():
        target.unlink()

    # Point the module's engine at the seed file (fresh SQLite).
    db.DB_PATH = target
    db.DB_URL = f"sqlite:///{target}"
    db.engine = db.create_engine(db.DB_URL, connect_args={"check_same_thread": False})
    db.Session = db.sessionmaker(bind=db.engine)
    db.init_db()  # creates schema + seeds Demo Org (id 1) with the 4 demo users

    # ── Org 1 (Demo Org) extra org-scoped rows ───────────────────────────────
    db.set_setting(1, "alert_threshold_soh", 75.0)
    db.set_setting(1, "site_label", "Demo Site A")
    from experiment_registry import log_run
    log_run(
        org_id=1, dataset="nasa", chemistry="NCA", feature_set=["soh_pct"],
        feature_version="5", hyperparams={"n_estimators": 100}, seed=42,
        cell_ids=["B0005"], n_rows=168,
        lco_metrics={"soh_mae": 1.2, "soh_r2": 0.99, "rul_mae": 8.0, "rul_r2": 0.96,
                     "rul_reliable": True, "per_cell": {}},
    )
    db.save_decision(1, {"id": "dec-demo-1", "cell_id": "B0005", "action": "continue",
                         "confidence": "Medium", "soh_pct": 71.4, "timestamp": "2026-08-01T10:00:00",
                         "status": "Completed"}, caller_role="admin")

    # ── Org 2 (Migrate Test Co) — its own admin + rows ──────────────────────
    org2 = db.create_organization_with_admin(
        "Migrate Test Co", "mt_admin", "mt-pass-123", display_name="MT Admin"
    )
    org2_id = org2["org_id"]
    db.set_setting(org2_id, "site_label", "Migrate Site Z")
    log_run(
        org_id=org2_id, dataset="severson", chemistry="LFP", feature_set=["soh_pct"],
        feature_version="5", hyperparams={"n_estimators": 200}, seed=7,
        cell_ids=["b1c13"], n_rows=1200,
        lco_metrics={"soh_mae": 0.9, "soh_r2": 0.995, "rul_mae": 6.0, "rul_r2": 0.98,
                     "rul_reliable": True, "per_cell": {}},
    )
    db.save_decision(org2_id, {"id": "dec-mt-1", "cell_id": "b1c13", "action": "replace",
                               "confidence": "High", "soh_pct": 66.0, "timestamp": "2026-08-02T10:00:00",
                               "status": "Pending"}, caller_role="admin")

    # Report the per-org row counts so the post-migration comparison has
    # exact expected numbers.
    with db.Session() as s:
        counts = {}
        for table in ("organizations", "users", "settings", "experiment_runs", "decisions"):
            counts[table] = s.execute(db.text(f"SELECT count(*) FROM {table}")).scalar()
    print("seeded sqlite:", target)
    print("row counts:", counts)
    print(f"org2 id: {org2_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
