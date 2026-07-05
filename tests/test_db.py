"""Unit tests for src/db.py — SQLite persistence layer.

Each test gets an isolated SQLite file via the db_module fixture below,
so tests never touch the real data/app.db.
"""

import pathlib
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


def test_decision_round_trip(db):
    entry = {
        "id": "Cell1_120000", "cell_id": "Cell1", "action": "Continue",
        "confidence": "high", "soh_pct": 95.0, "timestamp": "2026-01-01T00:00",
        "status": "Pending", "outcome_soh": None,
    }
    db.save_decision(entry)
    loaded = db.load_decisions()
    assert len(loaded) == 1
    assert loaded[0]["cell_id"] == "Cell1"

    db.update_decision("Cell1_120000", status="Approved")
    assert db.load_decisions()[0]["status"] == "Approved"


def test_cohort_tags_round_trip(db):
    db.save_cohort_tag("Cell1", "Batch-A")
    db.save_cohort_tag("Cell2", "Batch-B")
    assert db.load_cohort_tags() == {"Cell1": "Batch-A", "Cell2": "Batch-B"}

    # Re-saving the same cell_id overwrites, doesn't duplicate
    db.save_cohort_tag("Cell1", "Batch-C")
    assert db.load_cohort_tags()["Cell1"] == "Batch-C"


def test_settings_round_trip_and_default(db):
    assert db.get_setting("webhook_url", "") == ""
    db.set_setting("webhook_url", "https://example.com/hook")
    assert db.get_setting("webhook_url") == "https://example.com/hook"
    db.set_setting("eol_threshold_pct", 85.0)
    assert db.get_setting("eol_threshold_pct") == 85.0
    db.set_setting("webhook_events", ["A", "B"])
    assert db.get_setting("webhook_events") == ["A", "B"]


def test_upload_meta_round_trip(db):
    db.save_upload_meta({"upload_date": "2026-01-01", "n_cells": 3, "cell_ids": ["A", "B", "C"]}, "upload-abc")
    hist = db.load_upload_meta_history()
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
    db.save_failure_signatures([sig])
    loaded = db.load_failure_signatures()
    assert len(loaded) == 1
    assert loaded[0].cell_id == "Cell1"
    assert loaded[0].feature_names == ["fade_rate_30cy", "stress_index"]
    assert abs(loaded[0].trend_vector[0] - 0.1) < 1e-9
