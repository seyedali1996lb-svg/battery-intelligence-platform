"""Unit tests for src/db.py — SQLite persistence layer + multi-tenancy.

Each test gets an isolated SQLite file via the db_module fixture below,
so tests never touch the real data/app.db.
"""

import pathlib
import sqlite3

import numpy as np
import pytest
import db as db_module


_TEST_ENCRYPTION_KEY = "03ZJHIomd1hhT9w4FWvNxoN2wqPUnjfg3bSycZqUmgY="  # test-only, not the fallback


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point src/db.py at a throwaway SQLite file for the duration of one
    test, with a real (test-only) SETTINGS_ENCRYPTION_KEY set -- so tests
    exercise the same "properly configured" path a real deployment would,
    rather than incidentally hitting the InsecureCredentialStorageError
    guard meant for the unset-key case (see
    test_set_setting_refuses_new_secret_under_fallback_key below, which
    deliberately unsets it). _fernet is a process-global cache in db.py,
    so it's reset here too -- otherwise a Fernet built from a *different*
    test's key (or the real fallback) would leak into this test."""
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


def test_secret_setting_round_trips_through_encryption(db):
    assert "vrm_api_token" in db._SECRET_SETTING_KEYS
    db.set_setting(DEMO_ORG_ID, "vrm_api_token", "super-secret-token-123")
    assert db.get_setting(DEMO_ORG_ID, "vrm_api_token") == "super-secret-token-123"


def test_get_settings_batches_multiple_keys_matching_get_setting(db):
    """get_settings() must return exactly what N calls to get_setting() would,
    for both plain and encrypted keys, in one query instead of N."""
    db.set_setting(DEMO_ORG_ID, "pinned_cell", "B0005")
    db.set_setting(DEMO_ORG_ID, "webhook_secret", "shh-its-a-secret")
    db.set_setting(DEMO_ORG_ID, "eol_threshold_pct", 85.0)

    batched = db.get_settings(
        DEMO_ORG_ID, keys=["pinned_cell", "webhook_secret", "eol_threshold_pct", "never_set_key"]
    )
    assert batched["pinned_cell"] == db.get_setting(DEMO_ORG_ID, "pinned_cell")
    assert batched["webhook_secret"] == db.get_setting(DEMO_ORG_ID, "webhook_secret") == "shh-its-a-secret"
    assert batched["eol_threshold_pct"] == db.get_setting(DEMO_ORG_ID, "eol_threshold_pct")
    assert "never_set_key" not in batched  # absent, not filled with a default


def test_get_settings_respects_org_isolation(db):
    org_a = db.create_organization_with_admin("Org A", "a_admin", "passwordA")["org_id"]
    org_b = db.create_organization_with_admin("Org B", "b_admin", "passwordB")["org_id"]
    db.set_setting(org_a, "pinned_cell", "CELL-A")
    db.set_setting(org_b, "pinned_cell", "CELL-B")

    result_a = db.get_settings(org_a)
    result_b = db.get_settings(org_b)
    assert result_a["pinned_cell"] == "CELL-A"
    assert result_b["pinned_cell"] == "CELL-B"


def test_get_settings_unfiltered_returns_all_keys_for_org(db):
    db.set_setting(DEMO_ORG_ID, "pinned_cell", "B0005")
    db.set_setting(DEMO_ORG_ID, "webhook_url", "https://example.com/hook")
    result = db.get_settings(DEMO_ORG_ID)
    assert result["pinned_cell"] == "B0005"
    assert result["webhook_url"] == "https://example.com/hook"


def test_secret_setting_is_not_plaintext_in_the_raw_db_row(db):
    db.set_setting(DEMO_ORG_ID, "cmms_api_key", "plaintext-should-not-appear")
    with db.Session() as s:
        row = s.query(db.Setting).filter_by(org_id=DEMO_ORG_ID, key="cmms_api_key").one()
        assert "plaintext-should-not-appear" not in row.value
        assert row.value != db.json.dumps("plaintext-should-not-appear")


def test_set_setting_refuses_new_secret_under_fallback_key(db, monkeypatch):
    """GitGuardian flagged (2026-07-30) that the fallback encryption key is
    public in this repo's source -- set_setting() must refuse to newly
    encrypt a real credential with it rather than silently doing so."""
    monkeypatch.delenv("SETTINGS_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(db, "_fernet", None)
    with pytest.raises(db.InsecureCredentialStorageError):
        db.set_setting(DEMO_ORG_ID, "vrm_api_token", "a-real-looking-token")
    # And it must not have been written at all.
    assert db.get_setting(DEMO_ORG_ID, "vrm_api_token") is None


def test_set_setting_allows_resaving_unchanged_secret_under_fallback_key(db, monkeypatch):
    """A value already stored under the fallback key (e.g. configured
    before SETTINGS_ENCRYPTION_KEY was ever set) must still be re-savable
    -- app/_pages/_settings_config.py calls set_setting() on every rerun,
    not just on a change, so raising here would break the Settings page
    outright for every already-configured org, for no security benefit
    (re-encrypting with the same insecure key exposes nothing new)."""
    monkeypatch.delenv("SETTINGS_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(db, "_fernet", None)
    # Seed a row already encrypted under the fallback key directly, bypassing
    # set_setting() -- that's precisely what the guard would refuse as a
    # brand-new write, so this simulates a value configured before this
    # guard existed rather than exercising the guard itself.
    ciphertext = db.Fernet(db._FALLBACK_ENCRYPTION_KEY.encode()).encrypt(
        db.json.dumps("already-stored-token").encode()
    ).decode()
    with db.Session() as s:
        s.merge(db.Setting(org_id=DEMO_ORG_ID, key="orion_bms_api_key", value=ciphertext))
        s.commit()

    db.set_setting(DEMO_ORG_ID, "orion_bms_api_key", "already-stored-token")  # must not raise
    assert db.get_setting(DEMO_ORG_ID, "orion_bms_api_key") == "already-stored-token"


def test_set_setting_allows_clearing_secret_to_empty_under_fallback_key(db, monkeypatch):
    """Clearing a credential field to "" must not raise -- text_input
    widgets call set_setting() on every rerun including when a field is
    empty, and an empty string carries nothing to newly expose."""
    monkeypatch.delenv("SETTINGS_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(db, "_fernet", None)
    db.set_setting(DEMO_ORG_ID, "cmms_api_key", "")  # must not raise


def test_non_secret_setting_remains_plain_json_at_rest(db):
    db.set_setting(DEMO_ORG_ID, "webhook_url", "https://example.com/hook")
    with db.Session() as s:
        row = s.query(db.Setting).filter_by(org_id=DEMO_ORG_ID, key="webhook_url").one()
        assert row.value == db.json.dumps("https://example.com/hook")


def test_legacy_plaintext_secret_row_is_still_readable():
    """A row written before encryption was added (plain JSON, no Fernet
    envelope) must still be readable — get_setting() falls back to raw
    JSON on InvalidToken rather than losing/erroring on old data."""
    import tempfile
    tmp_path = pathlib.Path(tempfile.mkdtemp())
    test_db_path = tmp_path / "legacy_test.db"
    import sqlalchemy
    engine = sqlalchemy.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})
    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    db_module.Base.metadata.create_all(engine)

    import json as _json
    with Session() as s:
        s.merge(db_module.Setting(org_id=1, key="circunomics_api_key", value=_json.dumps("legacy-plaintext-key")))
        s.commit()

    import pytest as _pytest
    _monkeypatch = _pytest.MonkeyPatch()
    _monkeypatch.setattr(db_module, "engine", engine)
    _monkeypatch.setattr(db_module, "Session", Session)
    try:
        assert db_module.get_setting(1, "circunomics_api_key") == "legacy-plaintext-key"
        # Writing again should re-encrypt it going forward.
        db_module.set_setting(1, "circunomics_api_key", "legacy-plaintext-key")
        with Session() as s:
            row = s.query(db_module.Setting).filter_by(org_id=1, key="circunomics_api_key").one()
            assert row.value != _json.dumps("legacy-plaintext-key")
        assert db_module.get_setting(1, "circunomics_api_key") == "legacy-plaintext-key"
    finally:
        _monkeypatch.undo()


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


# ---------------------------------------------------------------------------
# FleetAsset hierarchy: Organization -> Site -> Fleet -> Pack -> Cell
# ---------------------------------------------------------------------------

def test_demo_org_gets_default_site_and_fleet_on_init(db):
    """init_db() backfills a default Site + Fleet for every org, including
    the seeded Demo Org, without any explicit call needed."""
    sites = db.list_sites(DEMO_ORG_ID)
    assert len(sites) == 1
    assert sites[0]["name"] == "Default Site"

    fleets = db.list_fleets(DEMO_ORG_ID)
    assert len(fleets) == 1
    assert fleets[0]["name"] == "Default Fleet"
    assert fleets[0]["site_id"] == sites[0]["id"]


def test_default_fleet_hierarchy_backfill_is_idempotent(db):
    """Calling init_db() again (or _seed_default_fleet_hierarchy() directly)
    must not create duplicate default sites/fleets."""
    db.init_db()
    db._seed_default_fleet_hierarchy()
    assert len(db.list_sites(DEMO_ORG_ID)) == 1
    assert len(db.list_fleets(DEMO_ORG_ID)) == 1


def test_new_org_gets_default_site_and_fleet_immediately(db):
    """A brand-new org created via signup (not present at the last
    init_db() backfill) must still get its own default Site/Fleet right
    away, not just at the next process restart."""
    org_id = db.create_organization_with_admin("New Co", "newco_admin", "password123")["org_id"]
    sites = db.list_sites(org_id)
    assert len(sites) == 1 and sites[0]["name"] == "Default Site"
    fleets = db.list_fleets(org_id)
    assert len(fleets) == 1 and fleets[0]["name"] == "Default Fleet"


def test_migration_does_not_disturb_pre_existing_data(db):
    """Adding the FleetAsset tables/backfill must not alter any
    pre-existing decisions/settings/etc. -- purely additive."""
    db.save_decision(DEMO_ORG_ID, {"id": "d1", "cell_id": "B0005", "action": "Continue", "timestamp": "2026-01-01"})
    db.set_setting(DEMO_ORG_ID, "eol_threshold_pct", 82.0)

    db.init_db()  # re-run migration/backfill

    decisions = db.load_decisions(DEMO_ORG_ID)
    assert len(decisions) == 1 and decisions[0]["id"] == "d1"
    assert db.get_setting(DEMO_ORG_ID, "eol_threshold_pct") == 82.0


def test_site_fleet_pack_hierarchy_crud(db):
    site = db.create_site(DEMO_ORG_ID, "Warehouse A")
    fleet = db.create_fleet(DEMO_ORG_ID, site["id"], "Forklift Fleet")
    pack = db.create_pack(DEMO_ORG_ID, fleet["id"], "Pack 1")

    assert any(s["id"] == site["id"] for s in db.list_sites(DEMO_ORG_ID))
    assert db.list_fleets(DEMO_ORG_ID, site_id=site["id"]) == [fleet]
    assert db.list_packs(DEMO_ORG_ID, fleet_id=fleet["id"]) == [pack]

    db.add_cell_to_pack(DEMO_ORG_ID, pack["id"], "B0005", position=0)
    db.add_cell_to_pack(DEMO_ORG_ID, pack["id"], "B0006", position=1)
    assert db.list_pack_cells(DEMO_ORG_ID, pack["id"]) == ["B0005", "B0006"]

    db.remove_cell_from_pack(DEMO_ORG_ID, pack["id"], "B0005")
    assert db.list_pack_cells(DEMO_ORG_ID, pack["id"]) == ["B0006"]


def test_fleet_asset_hierarchy_is_org_scoped(db):
    """Two orgs' sites/fleets/packs never leak into each other's listings --
    same cross-tenant isolation guarantee as every other table in this
    module."""
    org_a = db.create_organization_with_admin("Org A", "siteadmin_a", "passwordA")["org_id"]
    org_b = db.create_organization_with_admin("Org B", "siteadmin_b", "passwordB")["org_id"]

    site_a = db.create_site(org_a, "Site A")
    site_b = db.create_site(org_b, "Site B")

    # Each org already has its own default site from signup, plus the new one
    assert {s["name"] for s in db.list_sites(org_a)} == {"Default Site", "Site A"}
    assert {s["name"] for s in db.list_sites(org_b)} == {"Default Site", "Site B"}
    assert site_a["id"] not in [s["id"] for s in db.list_sites(org_b)]


# ---------------------------------------------------------------------------
# Login lockout (Enterprise Readiness audit finding: login.py had no
# rate-limiting at all before this -- unlimited password guesses against any
# username)
# ---------------------------------------------------------------------------

def test_unlocked_account_returns_none(db):
    assert db.is_login_locked_out("engineer") is None


def test_lockout_triggers_after_max_failed_attempts(db):
    for _ in range(db._MAX_FAILED_LOGIN_ATTEMPTS - 1):
        db.record_failed_login("engineer")
        assert db.is_login_locked_out("engineer") is None, "should not lock out before the threshold"
    db.record_failed_login("engineer")  # the Nth attempt crosses the threshold
    assert db.is_login_locked_out("engineer") is not None


def test_successful_login_resets_attempt_counter(db):
    for _ in range(db._MAX_FAILED_LOGIN_ATTEMPTS - 1):
        db.record_failed_login("engineer")
    db.reset_login_attempts("engineer")
    assert db.is_login_locked_out("engineer") is None
    # Confirm the counter itself was cleared, not just the lock -- one more
    # failed attempt right after a reset should not immediately re-lock.
    db.record_failed_login("engineer")
    assert db.is_login_locked_out("engineer") is None


def test_lockout_is_per_username_not_global(db):
    for _ in range(db._MAX_FAILED_LOGIN_ATTEMPTS):
        db.record_failed_login("engineer")
    assert db.is_login_locked_out("engineer") is not None
    assert db.is_login_locked_out("admin") is None


def test_failed_login_against_nonexistent_username_is_a_safe_no_op(db):
    """Locking out (or erroring on) a username that was never registered
    would let an attacker distinguish "real user, wrong password" from "no
    such user" by comparing responses -- must behave identically to a
    normal miss instead."""
    db.record_failed_login("this-username-does-not-exist")
    assert db.is_login_locked_out("this-username-does-not-exist") is None
    db.reset_login_attempts("this-username-does-not-exist")  # must not raise


def test_lockout_expiry_is_in_the_future_when_active(db):
    import datetime
    for _ in range(db._MAX_FAILED_LOGIN_ATTEMPTS):
        db.record_failed_login("engineer")
    locked_until = db.is_login_locked_out("engineer")
    assert locked_until is not None
    assert datetime.datetime.fromisoformat(locked_until) > datetime.datetime.now()


def test_expired_lockout_no_longer_blocks_login(db):
    """A lockout in the past (simulating LOCKOUT_MINUTES having elapsed)
    must not keep blocking sign-in -- writes locked_until directly rather
    than sleeping LOCKOUT_MINUTES in a test."""
    import datetime
    for _ in range(db._MAX_FAILED_LOGIN_ATTEMPTS):
        db.record_failed_login("engineer")
    assert db.is_login_locked_out("engineer") is not None

    with db.Session() as s:
        row = s.query(db.User).filter_by(username="engineer").one()
        row.locked_until = (datetime.datetime.now() - datetime.timedelta(minutes=1)).isoformat()
        s.commit()

    assert db.is_login_locked_out("engineer") is None


def test_login_lockout_columns_migrate_onto_pre_existing_users_table(tmp_path, monkeypatch):
    """Same additive-migration guarantee as test_migration_preserves_pre_existing_rows_without_org_id
    above -- a users table created before this module gained login lockout
    must still get the new columns (and keep its existing rows) on the next
    init_db(), not crash or silently drop the account."""
    test_db_path = tmp_path / "test_legacy.db"
    conn = sqlite3.connect(test_db_path)
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, org_id INTEGER, username TEXT UNIQUE, "
        "password_hash TEXT, role TEXT, display_name TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO users (org_id, username, password_hash, role, display_name, created_at) "
        "VALUES (1, 'legacyuser', 'x', 'engineer', 'Legacy User', '2020-01-01')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()

    user = db_module.get_user_by_username("legacyuser")
    assert user is not None, "pre-existing row must survive the migration"
    assert db_module.is_login_locked_out("legacyuser") is None
    db_module.record_failed_login("legacyuser")  # must not raise on the newly-added columns
