"""Registry-verified cache hits — the self-heal for the 2026-09-13 incident.

A cached bundle carries the experiment_run_id of the log_run() call that
trained it, but the bundle cache's signature (cell IDs + cycle counts +
cache version) knows nothing about the REGISTRY. If a training session's
registry writes went to a different/ephemeral DB, every later app load
serves that bundle as a cache hit, cache hits never re-log, and the
plain-GBRT row stays missing (zhu2022) or stale (nasa/severson/synth on
pre-v12 rows) forever. These tests pin the guard that detects exactly
that state: cached_bundle_run_missing() + run_exists_in_db().
"""

import pytest

import db as db_module
from experiment_registry import PLATFORM_ORG_ID

_TEST_ENCRYPTION_KEY = "03ZJHIomd1hhT9w4FWvNxoN2wqPUnjfg3bSycZqUmgY="  # test-only


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Isolated SQLite DB (same pattern as tests/test_db.py's fixture)."""
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


def _make_bundle(run_id):
    return ({"metrics": {"experiment_run_id": run_id, "soh_r2": 0.9}}, {"c1": 80})


def test_run_exists_in_db_true_for_logged_run(db):
    import experiment_registry as reg
    rid = reg.log_run(
        org_id=PLATFORM_ORG_ID, dataset="t", chemistry="c",
        feature_set=["f"], feature_version="v-test", hyperparams={}, seed=0,
        cell_ids=["c1"], n_rows=1,
        lco_metrics={"soh_r2": 0.9, "rul_reliable": False, "per_cell": {}},
    )
    assert reg.run_exists_in_db(PLATFORM_ORG_ID, rid) is True


def test_run_exists_in_db_false_for_unknown_run(db):
    import experiment_registry as reg
    assert reg.run_exists_in_db(PLATFORM_ORG_ID, "nope_2026-09-13T000000_none_aa") is False


def test_cached_bundle_run_missing_detects_orphaned_run(db):
    """The incident state: a cache-hit bundle whose run_id is not in the DB."""
    import app._data as app_data
    orphaned = _make_bundle("ghost_2026-09-12T111109330767_ffd868")
    assert app_data.cached_bundle_run_missing(orphaned) is True


def test_cached_bundle_run_missing_false_when_row_exists(db):
    import app._data as app_data
    import experiment_registry as reg
    rid = reg.log_run(
        org_id=PLATFORM_ORG_ID, dataset="t", chemistry="c",
        feature_set=["f"], feature_version="v-test", hyperparams={}, seed=0,
        cell_ids=["c1"], n_rows=1,
        lco_metrics={"soh_r2": 0.9, "rul_reliable": False, "per_cell": {}},
    )
    assert app_data.cached_bundle_run_missing(_make_bundle(rid)) is False


def test_cached_bundle_run_missing_false_when_bundle_not_registry_backed():
    """Bundles trained outside the registry (import path, legacy caches)
    carry no experiment_run_id — they are NOT flagged (that state was
    always handled by the normal pipeline)."""
    import app._data as app_data
    no_run_id = ({"metrics": {"soh_r2": 0.9}}, {"c1": 80})
    assert app_data.cached_bundle_run_missing(no_run_id) is False
    assert app_data.cached_bundle_run_missing(None) is False
    assert app_data.cached_bundle_run_missing("garbage") is False


def test_cached_bundle_run_missing_fails_open_on_registry_error(db, monkeypatch):
    """A registry blip must not take the app down: fail-open (treat the
    row as present). The fingerprint column makes any resulting divergence
    visible on the Benchmark page instead."""
    import app._data as app_data
    import experiment_registry as reg

    def _boom(org_id, run_id):
        raise RuntimeError("registry unavailable")
    monkeypatch.setattr(reg, "run_exists_in_db", _boom)
    assert app_data.cached_bundle_run_missing(_make_bundle("any_run_id")) is False


def test_bundle_cache_version_bumped_v5():
    """The one-time cache bust that forces the re-log TODAY. Without it,
    all four fleet signatures still match and the self-heal never fires
    (the guard only runs on a cache hit... which is exactly the stale one)."""
    import bundle_cache as bc
    assert bc.MODEL_VERSION == "v5-registry-verified-bundles"


def test_score_count_handles_numpy_arrays():
    """Regression for the 2026-09-13 calibration-wiring bug: the merge code
    used `len(x or [])` on the pooled conformity scores — a multi-element
    numpy array in a boolean context raises ValueError, so run_lco_quantiles
    succeeded, the NEXT line threw, and the bare except silently replaced
    every measured calibrated-coverage number with 'calibration_attempted:
    true' on all four reference fleets. The count must be taken size-safely."""
    import numpy as np
    from app._data import _score_count
    assert _score_count(np.array([0.5, 1.2, 3.3])) == 3
    assert _score_count(np.array([])) == 0
    assert _score_count(None) == 0
    assert _score_count([1.0, 2.0]) == 2


def test_log_run_persists_fingerprint_and_meta_columns(db):
    """Fingerprints and validity/calibration metadata must survive the
    registry round trip. Regression: log_run() built the fingerprint into
    the record but save_experiment_run() silently DROPPED the column, so
    every logged run read back fingerprint=None — discovered 2026-09-13
    when the re-logged fleet rows arrived unstamped."""
    import experiment_registry as reg
    rid = reg.log_run(
        org_id=PLATFORM_ORG_ID, dataset="t", chemistry="c",
        feature_set=["f"], feature_version="v-test", hyperparams={}, seed=0,
        cell_ids=["c1"], n_rows=1,
        lco_metrics={
            "soh_r2": 0.9, "rul_reliable": False, "per_cell": {},
            "fingerprint": {"dataset": {"dataset_sha256": "abc"}, "environment": {"python": "3.14"}},
            "validity_meta": {"envelope": {"chemistry": "LFP"}},
            "calibration_meta": {"nominal_coverage": 0.8},
        },
    )
    row = reg.get_run(PLATFORM_ORG_ID, rid)
    assert row is not None
    assert row.get("fingerprint") == {"dataset": {"dataset_sha256": "abc"}, "environment": {"python": "3.14"}}
    assert row.get("validity_meta") == {"envelope": {"chemistry": "LFP"}}
    assert row.get("calibration_meta") == {"nominal_coverage": 0.8}
