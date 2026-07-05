"""Unit tests for src/db.py — SQLite persistence layer + multi-tenancy.

Each test gets an isolated SQLite file via the db_module fixture below,
so tests never touch the real data/app.db.
"""

import pathlib
import sqlite3

import numpy as np
import pytest
import db as db_module


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point src/db.py at a throwaway SQLite file for the duration of one test."""
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


DEMO_ORG_ID = 1  # seeded by init_db()


def test_demo_org_and_users_seeded(db):
    user = db.get_user_by_username("engineer")
    assert user is not None
    assert user["org_id"] == DEMO_ORG_ID
    assert user["org_name"] == "Demo Org"
    assert user["role"] == "engineer"
    assert db.verify_password("battery", user["password_hash"])
    assert not db.verify_password("wrongpassword", user["password_hash"])

    # Idempotent — calling init_db() again doesn't duplicate the seed
    db.init_db()
    with db.Session() as s:
        assert s.query(db.Organization).count() == 1
        assert s.query(db.User).count() == len(db.DEMO_USERS)


def test_hash_password_salts_each_call(db):
    h1 = db.hash_password("samepassword")
    h2 = db.hash_password("samepassword")
    assert h1 != h2  # different salt each time
    assert db.verify_password("samepassword", h1)
    assert db.verify_password("samepassword", h2)
    assert not db.verify_password("samepassword", "not-a-real-hash")


def test_create_organization_with_admin(db):
    result = db.create_organization_with_admin("Acme Batteries", "alice", "secret123")
    assert "org_id" in result
    user = db.get_user_by_username("alice")
    assert user["org_id"] == result["org_id"]
    assert user["role"] == "admin"
    assert db.verify_password("secret123", user["password_hash"])


def test_create_organization_rejects_duplicate_username(db):
    db.create_organization_with_admin("Acme Batteries", "alice", "secret123")
    dup = db.create_organization_with_admin("Another Org", "alice", "otherpass")
    assert "error" in dup


def test_create_organization_dedupes_slug(db):
    res1 = db.create_organization_with_admin("Acme Batteries", "alice", "secret123")
    res2 = db.create_organization_with_admin("Acme Batteries", "bob", "secret456")
    assert res1["org_id"] != res2["org_id"]


def test_create_user_invites_teammate_into_same_org(db):
    res = db.create_organization_with_admin("Acme Batteries", "alice", "secret123")
    org_id = res["org_id"]
    invite = db.create_user(org_id, "bob", "bobpass1", "fleet", "Bob Chen")
    assert "user_id" in invite
    bob = db.get_user_by_username("bob")
    assert bob["org_id"] == org_id
    assert bob["role"] == "fleet"

    dup = db.create_user(org_id, "bob", "x", "engineer")
    assert "error" in dup


def test_decision_round_trip(db):
    entry = {
        "id": "Cell1_120000", "cell_id": "Cell1", "action": "Continue",
        "confidence": "high", "soh_pct": 95.0, "timestamp": "2026-01-01T00:00",
        "status": "Pending", "outcome_soh": None,
    }
    db.save_decision(DEMO_ORG_ID, entry)
    loaded = db.load_decisions(DEMO_ORG_ID)
    assert len(loaded) == 1
    assert loaded[0]["cell_id"] == "Cell1"

    db.update_decision(DEMO_ORG_ID, "Cell1_120000", status="Approved")
    assert db.load_decisions(DEMO_ORG_ID)[0]["status"] == "Approved"


def test_cohort_tags_round_trip(db):
    db.save_cohort_tag(DEMO_ORG_ID, "Cell1", "Batch-A")
    db.save_cohort_tag(DEMO_ORG_ID, "Cell2", "Batch-B")
    assert db.load_cohort_tags(DEMO_ORG_ID) == {"Cell1": "Batch-A", "Cell2": "Batch-B"}

    # Re-saving the same cell_id overwrites, doesn't duplicate
    db.save_cohort_tag(DEMO_ORG_ID, "Cell1", "Batch-C")
    assert db.load_cohort_tags(DEMO_ORG_ID)["Cell1"] == "Batch-C"


def test_settings_round_trip_and_default(db):
    assert db.get_setting(DEMO_ORG_ID, "webhook_url", "") == ""
    db.set_setting(DEMO_ORG_ID, "webhook_url", "https://example.com/hook")
    assert db.get_setting(DEMO_ORG_ID, "webhook_url") == "https://example.com/hook"
    db.set_setting(DEMO_ORG_ID, "eol_threshold_pct", 85.0)
    assert db.get_setting(DEMO_ORG_ID, "eol_threshold_pct") == 85.0
    db.set_setting(DEMO_ORG_ID, "webhook_events", ["A", "B"])
    assert db.get_setting(DEMO_ORG_ID, "webhook_events") == ["A", "B"]


def test_upload_meta_round_trip(db):
    db.save_upload_meta(DEMO_ORG_ID, {"upload_date": "2026-01-01", "n_cells": 3, "cell_ids": ["A", "B", "C"]}, "upload-abc")
    hist = db.load_upload_meta_history(DEMO_ORG_ID)
    assert hist[0]["n_cells"] == 3
    assert hist[0]["cell_ids"] == ["A", "B", "C"]
    assert hist[0]["joblib_key"] == "upload-abc"


def test_failure_signatures_round_trip(db):
    from trajectory_memory import FailureSignature
    sig = FailureSignature(
        cell_id="Cell1", source="synthetic", eol_cycle=500,
        soh_at_window_start=82.0, failure_mode="LLI",
        feature_names=["fade_rate_30cy", "stress_index"],
        trend_vector=np.array([0.1, -0.2]),
    )
    db.save_failure_signatures(DEMO_ORG_ID, [sig])
    loaded = db.load_failure_signatures(DEMO_ORG_ID)
    assert len(loaded) == 1
    assert loaded[0].cell_id == "Cell1"
    assert loaded[0].feature_names == ["fade_rate_30cy", "stress_index"]
    assert abs(loaded[0].trend_vector[0] - 0.1) < 1e-9


def test_migration_preserves_pre_existing_rows_without_org_id(tmp_path, monkeypatch):
    """
    Simulates a pre-multi-tenancy local DB (tables exist but have no org_id
    column) and confirms init_db() adds the column additively via ALTER
    TABLE, without raising and without losing the existing rows — they
    should land under the seeded Demo Org (org_id=1) via the column default.
    """
    old_db_path = tmp_path / "old_schema.db"
    conn = sqlite3.connect(old_db_path)
    conn.execute(
        "CREATE TABLE decisions (id TEXT PRIMARY KEY, cell_id TEXT, action TEXT, "
        "confidence TEXT, soh_pct REAL, timestamp TEXT, status TEXT, outcome_soh REAL)"
    )
    conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO decisions VALUES ('old1', 'Cell1', 'Continue', 'high', 95.0, '2025-01-01', 'Pending', NULL)"
    )
    conn.execute("INSERT INTO settings VALUES ('eol_threshold_pct', '80.0')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(db_module, "DB_PATH", old_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{old_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))

    db_module.init_db()  # must not raise

    decisions = db_module.load_decisions(DEMO_ORG_ID)
    assert len(decisions) == 1 and decisions[0]["id"] == "old1"
    assert db_module.get_setting(DEMO_ORG_ID, "eol_threshold_pct") == 80.0

    # Demo org/users still seeded alongside the migrated pre-existing data
    assert db_module.get_user_by_username("admin") is not None


def test_cross_org_isolation(db):
    """
    The core acceptance criterion for multi-tenancy: two organizations'
    decisions, cohort tags, settings, and failure signatures never leak
    into each other's load_*() results.
    """
    from trajectory_memory import FailureSignature

    org_a = db.create_organization_with_admin("Org A", "a_admin", "passwordA")["org_id"]
    org_b = db.create_organization_with_admin("Org B", "b_admin", "passwordB")["org_id"]

    db.save_decision(org_a, {"id": "dA", "cell_id": "B0005", "action": "Continue", "timestamp": "2026-01-01"})
    db.save_decision(org_b, {"id": "dB", "cell_id": "B0005", "action": "Inspect", "timestamp": "2026-01-01"})
    assert [d["id"] for d in db.load_decisions(org_a)] == ["dA"]
    assert [d["id"] for d in db.load_decisions(org_b)] == ["dB"]

    db.save_cohort_tag(org_a, "B0005", "batch-A")
    db.save_cohort_tag(org_b, "B0005", "batch-Z")
    assert db.load_cohort_tags(org_a) == {"B0005": "batch-A"}
    assert db.load_cohort_tags(org_b) == {"B0005": "batch-Z"}

    db.set_setting(org_a, "eol_threshold_pct", 75.0)
    db.set_setting(org_b, "eol_threshold_pct", 90.0)
    assert db.get_setting(org_a, "eol_threshold_pct") == 75.0
    assert db.get_setting(org_b, "eol_threshold_pct") == 90.0

    db.save_upload_meta(org_a, {"upload_date": "2026-01-01", "n_cells": 1, "cell_ids": ["X"]}, "key-a")
    db.save_upload_meta(org_b, {"upload_date": "2026-01-01", "n_cells": 2, "cell_ids": ["Y", "Z"]}, "key-b")
    assert len(db.load_upload_meta_history(org_a)) == 1
    assert db.load_upload_meta_history(org_a)[0]["joblib_key"] == "key-a"
    assert len(db.load_upload_meta_history(org_b)) == 1
    assert db.load_upload_meta_history(org_b)[0]["joblib_key"] == "key-b"

    sig_a = FailureSignature(
        cell_id="B0005", source="nasa", eol_cycle=400, soh_at_window_start=80.0,
        failure_mode="LLI", feature_names=["fade_rate_30cy"], trend_vector=np.array([0.1]),
    )
    sig_b = FailureSignature(
        cell_id="B0005", source="nasa", eol_cycle=600, soh_at_window_start=85.0,
        failure_mode="LAM", feature_names=["stress_index"], trend_vector=np.array([0.2]),
    )
    db.save_failure_signatures(org_a, [sig_a])
    db.save_failure_signatures(org_b, [sig_b])
    loaded_a = db.load_failure_signatures(org_a)
    loaded_b = db.load_failure_signatures(org_b)
    assert len(loaded_a) == 1 and loaded_a[0].failure_mode == "LLI"
    assert len(loaded_b) == 1 and loaded_b[0].failure_mode == "LAM"
