"""Unit tests for src/experiment_registry.py — the GBRT run registry.

Covers log/get/leaderboard-query/replay paths. Same isolated-SQLite-file
fixture pattern as tests/test_db.py so nothing here touches the real
data/app.db.
"""

import pathlib
import sys

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
import pytest
import db as db_module
import experiment_registry as reg
from conftest import make_cycles_df
from batlab.validation.lco import run_lco
from batlab.features.engineering import FEATURE_VERSION


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


def _lco_metrics(soh_mae=1.0, soh_r2=0.8, rul_mae=20.0, rul_r2=0.5, rul_reliable=True, per_cell=None,
                 baseline_soh_r2=None, baseline_per_cell=None):
    return {
        "soh_mae": soh_mae, "soh_r2": soh_r2,
        "rul_mae": rul_mae, "rul_r2": rul_r2,
        "rul_reliable": rul_reliable,
        "per_cell": per_cell or {"CellA": {"soh_mae": soh_mae, "soh_r2": soh_r2,
                                            "rul_mae": rul_mae, "rul_r2": rul_r2}},
        "baseline_soh_r2": baseline_soh_r2,
        "baseline_per_cell": baseline_per_cell,
    }


def _log(db, org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
          rul_mae=20.0, soh_r2=0.8, notes=None):
    return reg.log_run(
        org_id=org_id,
        dataset=dataset,
        chemistry=chemistry,
        feature_set=["cycle_number", "fade_rate_30cy"],
        feature_version=FEATURE_VERSION,
        hyperparams={"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                     "subsample": 0.8, "random_state": 42},
        seed=42,
        cell_ids=["CellA", "CellB"],
        n_rows=300,
        lco_metrics=_lco_metrics(rul_mae=rul_mae, soh_r2=soh_r2),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# log_run / get_run
# ---------------------------------------------------------------------------

def test_log_run_and_get_run_round_trip(db):
    run_id = _log(db, chemistry="LFP", notes="unit test run")
    got = reg.get_run(reg.PLATFORM_ORG_ID, run_id)

    assert got is not None
    assert got["run_id"] == run_id
    assert got["dataset"] == "nasa"
    assert got["chemistry"] == "LFP"
    assert got["feature_set"] == ["cycle_number", "fade_rate_30cy"]
    assert got["feature_version"] == FEATURE_VERSION
    assert got["seed"] == 42
    assert got["cell_ids"] == ["CellA", "CellB"]
    assert got["n_cells"] == 2
    assert got["n_rows"] == 300
    assert got["rul_reliable"] is True
    assert got["fold_metrics"]["CellA"]["soh_r2"] == 0.8
    assert got["notes"] == "unit test run"
    assert got["git_commit"]  # non-empty string, "unknown" is an acceptable value
    assert got["timestamp"]


def test_log_run_persists_trivial_baseline(db):
    """The honest accuracy denominator must survive the DB round-trip, not just
    live in the in-memory bundle — otherwise the Benchmark page's baseline
    column and the model card's model-advantage line would always read '—'."""
    run_id = reg.log_run(
        org_id=reg.PLATFORM_ORG_ID,
        dataset="nasa",
        chemistry="LiCoO2",
        feature_set=["cycle_number"],
        feature_version=FEATURE_VERSION,
        hyperparams={},
        seed=42,
        cell_ids=["CellA", "CellB"],
        n_rows=300,
        lco_metrics=_lco_metrics(
            soh_r2=0.806,
            baseline_soh_r2=0.603,
            baseline_per_cell={"CellA": {"baseline_soh_r2": 0.6}},
        ),
    )
    got = reg.get_run(reg.PLATFORM_ORG_ID, run_id)
    assert got["baseline_soh_r2"] == 0.603
    assert got["baseline_per_cell"] == {"CellA": {"baseline_soh_r2": 0.6}}


def test_log_run_baseline_absent_is_null_not_fabricated(db):
    """A run logged without a baseline (e.g. a cross-chemistry transfer, where
    LCO folds don't exist) stores NULL — never a fabricated number."""
    run_id = _log(db, dataset="nasa_to_severson")
    got = reg.get_run(reg.PLATFORM_ORG_ID, run_id)
    assert got["baseline_soh_r2"] is None
    assert got["baseline_per_cell"] is None


def test_get_run_returns_none_for_unknown_run_id(db):
    assert reg.get_run(reg.PLATFORM_ORG_ID, "does-not-exist") is None


def test_get_run_is_scoped_to_org_id(db):
    """A run logged under one org must not be visible via a different org_id --
    the same isolation guarantee every other org-scoped table in db.py has."""
    run_id = _log(db, org_id=1)
    assert reg.get_run(1, run_id) is not None
    assert reg.get_run(2, run_id) is None


def test_log_run_generates_unique_ids_for_repeated_calls(db):
    id_a = _log(db)
    id_b = _log(db)
    assert id_a != id_b


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------

def test_leaderboard_filters_by_dataset(db):
    _log(db, dataset="nasa")
    _log(db, dataset="severson")
    board = reg.leaderboard(tenant_org_id=None, dataset="severson")
    assert len(board) == 1
    assert board[0]["dataset"] == "severson"


def test_leaderboard_filters_by_chemistry(db):
    _log(db, chemistry="LiCoO2")
    _log(db, chemistry="LFP")
    board = reg.leaderboard(tenant_org_id=None, chemistry="LFP")
    assert len(board) == 1
    assert board[0]["chemistry"] == "LFP"


def test_leaderboard_sorts_ascending_by_default(db):
    _log(db, dataset="a", rul_mae=30.0)
    _log(db, dataset="b", rul_mae=10.0)
    _log(db, dataset="c", rul_mae=20.0)
    board = reg.leaderboard(tenant_org_id=None, sort_by="rul_mae")
    assert [r["dataset"] for r in board] == ["b", "c", "a"]


def test_leaderboard_sorts_descending_when_requested(db):
    _log(db, dataset="a", soh_r2=0.5)
    _log(db, dataset="b", soh_r2=0.9)
    _log(db, dataset="c", soh_r2=0.7)
    board = reg.leaderboard(tenant_org_id=None, sort_by="soh_r2", ascending=False)
    assert [r["dataset"] for r in board] == ["b", "c", "a"]


def test_leaderboard_missing_metric_sorts_last_regardless_of_direction(db):
    """A run missing the sort column (e.g. a cross-dataset transfer run with
    no rul_mae) must never be mistaken for the best result just because
    None compares as falsy -- it always sorts last."""
    _log(db, dataset="has_metric", rul_mae=15.0)
    no_metric_id = reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="missing_metric", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version=FEATURE_VERSION,
        hyperparams={"random_state": 42}, seed=42, cell_ids=["CellA"], n_rows=10,
        lco_metrics={"soh_mae": 1.0, "soh_r2": 0.8, "rul_mae": None, "rul_r2": None,
                     "rul_reliable": False, "per_cell": {}},
    )

    board_asc = reg.leaderboard(tenant_org_id=None, sort_by="rul_mae", ascending=True)
    assert board_asc[-1]["run_id"] == no_metric_id

    board_desc = reg.leaderboard(tenant_org_id=None, sort_by="rul_mae", ascending=False)
    assert board_desc[-1]["run_id"] == no_metric_id


def test_leaderboard_combines_platform_and_tenant_runs(db):
    _log(db, org_id=reg.PLATFORM_ORG_ID, dataset="nasa")
    _log(db, org_id=5, dataset="uploaded")
    board = reg.leaderboard(tenant_org_id=5)
    assert {r["dataset"] for r in board} == {"nasa", "uploaded"}


def test_leaderboard_excludes_other_tenants_uploaded_runs(db):
    _log(db, org_id=5, dataset="uploaded")
    _log(db, org_id=6, dataset="uploaded")
    board = reg.leaderboard(tenant_org_id=5)
    assert len(board) == 1
    assert board[0]["org_id"] == 5


def test_leaderboard_with_no_tenant_shows_only_platform_runs(db):
    _log(db, org_id=reg.PLATFORM_ORG_ID, dataset="nasa")
    _log(db, org_id=5, dataset="uploaded")
    board = reg.leaderboard(tenant_org_id=None)
    assert len(board) == 1
    assert board[0]["dataset"] == "nasa"


# ---------------------------------------------------------------------------
# accuracy_by_source
# ---------------------------------------------------------------------------

def _log_acc(db, dataset, chemistry, soh_r2, baseline, n_cells=4, rul_r2=0.7,
             rul_reliable=True, org_id=reg.PLATFORM_ORG_ID):
    return reg.log_run(
        org_id=org_id,
        dataset=dataset,
        chemistry=chemistry,
        feature_set=["cycle_number"],
        feature_version=FEATURE_VERSION,
        hyperparams={},
        seed=42,
        cell_ids=[f"C{i}" for i in range(n_cells)],
        n_rows=n_cells * 100,
        lco_metrics=_lco_metrics(
            soh_r2=soh_r2,
            rul_r2=rul_r2,
            rul_reliable=rul_reliable,
            baseline_soh_r2=baseline,
        ),
    )


def test_accuracy_by_source_reports_advantage(db):
    _log_acc(db, "nasa", "LiCoO2", soh_r2=0.759, baseline=0.603)
    _log_acc(db, "severson", "LFP", soh_r2=0.986, baseline=-0.758)

    rows = reg.accuracy_by_source(tenant_org_id=None)
    assert len(rows) == 2

    # Sorted by real advantage descending: Severson (+1.744) before NASA (+0.156).
    assert [r["dataset"] for r in rows] == ["severson", "nasa"]

    nasa = next(r for r in rows if r["dataset"] == "nasa")
    assert abs(nasa["advantage"] - 0.156) < 1e-9
    assert nasa["baseline_soh_r2"] == 0.603
    assert nasa["soh_r2"] == 0.759

    sev = next(r for r in rows if r["dataset"] == "severson")
    assert abs(sev["advantage"] - 1.744) < 1e-9
    assert sev["chemistry"] == "LFP"


def test_accuracy_by_source_groups_by_dataset_not_chemistry(db):
    """The synthetic fleet and NASA cells are BOTH 'LiCoO2' — they must not be
    collapsed into one row, or a simulated fleet would masquerade as real cells."""
    _log_acc(db, "nasa", "LiCoO2", soh_r2=0.759, baseline=0.603)
    _log_acc(db, "synth", "LiCoO2", soh_r2=0.998, baseline=-1.967)

    rows = reg.accuracy_by_source(tenant_org_id=None)
    assert len(rows) == 2
    assert {r["dataset"] for r in rows} == {"nasa", "synth"}


def test_accuracy_by_source_excludes_runs_without_baseline(db):
    """A cross-chemistry transfer run has no baseline by construction and must
    be excluded, not shown with a fabricated advantage."""
    _log_acc(db, "nasa", "LiCoO2", soh_r2=0.759, baseline=0.603)
    _log(db, dataset="nasa_to_severson", chemistry="LiCoO2 -> LFP")

    rows = reg.accuracy_by_source(tenant_org_id=None)
    assert [r["dataset"] for r in rows] == ["nasa"]


def test_accuracy_by_source_reports_newest_run_per_group(db):
    """Repeated runs (or a retrain) for the same group: the newest wins — the
    older duplicate must not appear as a second row."""
    _log_acc(db, "nasa", "LiCoO2", soh_r2=0.500, baseline=0.400)
    _log_acc(db, "nasa", "LiCoO2", soh_r2=0.900, baseline=0.600)

    rows = reg.accuracy_by_source(tenant_org_id=None)
    assert len(rows) == 1
    assert rows[0]["soh_r2"] == 0.900
    assert abs(rows[0]["advantage"] - 0.300) < 1e-9


def test_accuracy_by_source_returns_empty_when_no_baselines(db):
    _log(db, dataset="nasa", chemistry="LiCoO2")
    assert reg.accuracy_by_source(tenant_org_id=None) == []


def test_accuracy_by_source_scoped_to_tenant(db):
    _log_acc(db, "uploaded", "LFP", soh_r2=0.9, baseline=0.5, org_id=5)
    assert reg.accuracy_by_source(tenant_org_id=None) == []   # platform only
    assert len(reg.accuracy_by_source(tenant_org_id=5)) == 1  # org 5 sees its own


# ---------------------------------------------------------------------------
# replay_run
# ---------------------------------------------------------------------------

def test_replay_run_reproduces_recorded_metrics(db):
    # Fast fade so both cells cross EOL in-window: the RUL metrics this test
    # replays are computed on observed rows (v12), not None.
    cell_data = {
        "CellA": make_cycles_df(n_cycles=300, fade_per_cycle=0.003),
        "CellB": make_cycles_df(n_cycles=300, fade_per_cycle=0.0035, initial_resistance_ohm=0.06),
    }
    lco = run_lco(cell_data, seed=42)
    run_id = reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number", "fade_rate_30cy"], feature_version=FEATURE_VERSION,
        hyperparams={"random_state": 42}, seed=42,
        cell_ids=list(cell_data.keys()), n_rows=400, lco_metrics=lco,
    )

    result = reg.replay_run(reg.PLATFORM_ORG_ID, run_id, cell_data)

    assert result["environment_match"] is True
    assert result["recorded"]["soh_r2"] == lco["soh_r2"]
    assert result["recorded"]["rul_mae"] == lco["rul_mae"]
    # Same seed + same data + same code -> reproduced numbers match exactly.
    assert result["soh_r2"] == pytest.approx(lco["soh_r2"])
    assert result["rul_mae"] == pytest.approx(lco["rul_mae"])
    assert result["run"]["run_id"] == run_id
    # This test's hyperparams={"random_state": 42} is a partial fixture dict,
    # not a real dict(GBRT_PARAMS) copy -- it genuinely doesn't match every
    # key in the current batlab.validation.lco.GBRT_PARAMS, so this is
    # correctly False, not a bug.
    assert result["hyperparams_match"] is False
    assert result["hyperparams_diff"]["n_estimators"] == (None, 200)


def test_replay_run_hyperparams_match_true_when_recorded_matches_current(db):
    """The replay contract's hyperparams_match check (see
    experiment_registry.py's "The replay contract" docstring section): a
    run logged with the real current batlab.validation.lco.GBRT_PARAMS
    values must classify as a match."""
    from batlab.validation.lco import GBRT_PARAMS as CURRENT_GBRT_PARAMS

    cell_data = {"CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006)}
    lco = run_lco(cell_data, seed=42)
    run_id = reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number", "fade_rate_30cy"], feature_version=FEATURE_VERSION,
        hyperparams=dict(CURRENT_GBRT_PARAMS), seed=42,
        cell_ids=list(cell_data.keys()), n_rows=200, lco_metrics=lco,
    )

    result = reg.replay_run(reg.PLATFORM_ORG_ID, run_id, cell_data)

    assert result["hyperparams_match"] is True
    assert result["hyperparams_diff"] == {}


def test_replay_run_raises_for_unknown_run_id(db):
    with pytest.raises(ValueError, match="No logged run"):
        reg.replay_run(reg.PLATFORM_ORG_ID, "does-not-exist", {})


def test_replay_run_propagates_missing_cell_error(db):
    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008),
    }
    lco = run_lco(cell_data, seed=1)
    run_id = reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version=FEATURE_VERSION,
        hyperparams={"random_state": 1}, seed=1,
        cell_ids=list(cell_data.keys()), n_rows=400, lco_metrics=lco,
    )

    with pytest.raises(ValueError, match="missing"):
        reg.replay_run(reg.PLATFORM_ORG_ID, run_id, {"CellA": cell_data["CellA"]})


# ---------------------------------------------------------------------------
# git commit hash
# ---------------------------------------------------------------------------

def test_git_commit_hash_returns_nonempty_string():
    commit = reg._git_commit_hash()
    assert isinstance(commit, str)
    assert len(commit) > 0


# ---------------------------------------------------------------------------
# Cross-chemistry generalization study
# ---------------------------------------------------------------------------

def test_run_cross_chemistry_transfer_logs_a_real_run(db):
    train_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008, initial_resistance_ohm=0.06),
    }
    eval_data = {"CellC": make_cycles_df(n_cycles=150, fade_per_cycle=0.0007)}

    result = reg.run_cross_chemistry_transfer("fake_train", train_data, "fake_eval", eval_data)

    assert result["n_common_features"] > 0
    run = reg.get_run(reg.PLATFORM_ORG_ID, result["run_id"])
    assert run["dataset"] == "fake_train_to_fake_eval"
    assert run["cell_ids"] == ["CellA", "CellB", "CellC"]
    assert run["rul_reliable"] is False  # never claimed reliable for an out-of-domain transfer
    assert run["fold_metrics"] == {}     # not leave-cell-out
    # CellA/B/C are all synthetic-prefix (LiCoO2), so this is the
    # same-chemistry cross-source flavor of the transfer study.
    assert "Same-chemistry cross-source generalization study" in run["notes"]


def test_run_cross_chemistry_transfer_labels_cross_chemistry_pairs(db):
    """An LFP eval domain (S- prefix) against a LiCoO2 train domain must be
    labelled a CROSS-chemistry study in the notes."""
    train_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
    }
    eval_data = {"S-eval1": make_cycles_df(n_cycles=150, fade_per_cycle=0.0007)}

    result = reg.run_cross_chemistry_transfer("fake_train", train_data, "fake_eval", eval_data)
    run = reg.get_run(reg.PLATFORM_ORG_ID, result["run_id"])
    assert "Cross-chemistry generalization study" in run["notes"]
    assert "LiCoO2 -> LFP" in (run.get("chemistry") or "")


def test_run_cross_chemistry_transfer_raises_on_incompatible_schema(db, monkeypatch):
    """A domain pair sharing too few real feature columns must refuse
    rather than report a number built on 1-2 coincidental columns. In
    practice this codebase's generic capacity/cycle-derived features
    (fade_rate_*, soh_velocity_50cy, ...) are present for almost any
    per-cycle dataset, so the guard is exercised here by shrinking the
    universe of columns get_model_matrix() is allowed to consider --
    simulating the real failure mode (a dataset missing the raw
    quantities most FEATURE_COLUMNS derive from, like Oxford's checkpoint
    schema, which has no cycle_number/resistance_ohm/temperature_c at
    all and can't even reach get_model_matrix -- see
    log_cross_chemistry_unavailable() for how that specific case is
    actually handled)."""
    import batlab.features.engineering as engineering

    monkeypatch.setattr(engineering, "FEATURE_COLUMNS", ["cycle_number"])

    train_data = {"CellA": make_cycles_df(n_cycles=100)}
    eval_data = {"CellB": make_cycles_df(n_cycles=100)}

    with pytest.raises(ValueError, match="usable feature column"):
        reg.run_cross_chemistry_transfer("fake_train", train_data, "fake_eval", eval_data)


def test_log_cross_chemistry_unavailable_records_null_metrics_not_a_fabricated_number(db):
    run_id = reg.log_cross_chemistry_unavailable(
        "nasa", "oxford", "Oxford schema incompatible.", org_id=reg.PLATFORM_ORG_ID,
    )
    run = reg.get_run(reg.PLATFORM_ORG_ID, run_id)
    assert run["dataset"] == "nasa_to_oxford"
    assert run["soh_mae"] is None
    assert run["rul_mae"] is None
    assert run["rul_r2"] is None
    assert "Oxford schema incompatible" in run["notes"]


def test_cross_chemistry_runs_for_train_dataset_filters_by_prefix(db):
    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa_to_severson", chemistry="LiCoO2 -> LFP",
        feature_set=["cycle_number"], feature_version=FEATURE_VERSION,
        hyperparams={"random_state": 42}, seed=42, cell_ids=["A"], n_rows=10,
        lco_metrics=_lco_metrics(),
    )
    reg.log_cross_chemistry_unavailable("nasa", "oxford", "reason", org_id=reg.PLATFORM_ORG_ID)
    reg.log_run(  # unrelated run -- must not be picked up
        org_id=reg.PLATFORM_ORG_ID, dataset="severson", chemistry="LFP",
        feature_set=["cycle_number"], feature_version=FEATURE_VERSION,
        hyperparams={"random_state": 42}, seed=42, cell_ids=["B"], n_rows=10,
        lco_metrics=_lco_metrics(),
    )

    runs = reg.cross_chemistry_runs_for_train_dataset("nasa")
    assert {r["dataset"] for r in runs} == {"nasa_to_severson", "nasa_to_oxford"}


def test_cross_chemistry_runs_for_train_dataset_empty_when_none_logged(db):
    assert reg.cross_chemistry_runs_for_train_dataset("nasa") == []


# ---------------------------------------------------------------------------
# Permanent cross-chemistry benchmark (aggregator)
# ---------------------------------------------------------------------------

def _fake_clock(monkeypatch):
    """Patch experiment_registry's datetime so each log_run() gets a strictly
    increasing timestamp -- the aggregators pick the NEWEST run per pairing by
    comparing ISO timestamp strings, and two log_run() calls in one test can
    otherwise land on the same microsecond."""
    import datetime as _dt
    from types import SimpleNamespace
    ticks = {"n": 0}

    class _Clock:
        @staticmethod
        def now():
            ticks["n"] += 1
            return _dt.datetime(2026, 1, 1, 0, 0, ticks["n"])

    monkeypatch.setattr(reg, "datetime", SimpleNamespace(datetime=_Clock))


def _log_transfer(dataset, soh_r2, rul_r2=-1.0, chemistry="LiCoO2 -> LFP"):
    return reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset=dataset, chemistry=chemistry,
        feature_set=["cycle_number", "fade_rate_30cy"], feature_version=FEATURE_VERSION,
        hyperparams={"random_state": 42}, seed=42, cell_ids=["A"], n_rows=10,
        lco_metrics=_lco_metrics(soh_mae=22.0, soh_r2=soh_r2, rul_mae=1350.0, rul_r2=rul_r2,
                                rul_reliable=False, per_cell={}),
    )


def test_cross_chemistry_benchmark_keeps_newest_run_per_pair(db, monkeypatch):
    _fake_clock(monkeypatch)
    _log_transfer("nasa_to_severson", soh_r2=-34.6)   # older, superseded
    _log_transfer("nasa_to_severson", soh_r2=-9.9)    # newer, must win

    rows = reg.cross_chemistry_benchmark()
    assert len(rows) == 1
    assert rows[0]["train_dataset"] == "nasa"
    assert rows[0]["eval_dataset"] == "severson"
    assert rows[0]["soh_r2"] == pytest.approx(-9.9)
    assert rows[0]["evaluated"] is True
    assert rows[0]["n_features"] == 2


def test_cross_chemistry_benchmark_sorts_worst_transfer_first(db, monkeypatch):
    """The point of this table is the counterexample -- the most negative
    SOH R² must come first, and the honest "not evaluated" rows must be
    present (after the numbers) rather than omitted."""
    _fake_clock(monkeypatch)
    _log_transfer("severson_to_synth", soh_r2=-2.0, chemistry="LFP -> LiCoO2")
    _log_transfer("nasa_to_severson", soh_r2=-34.6)
    reg.log_cross_chemistry_unavailable("nasa", "oxford", "schema incompatible")

    rows = reg.cross_chemistry_benchmark()
    assert [(r["train_dataset"], r["eval_dataset"]) for r in rows] == [
        ("nasa", "severson"), ("severson", "synth"), ("nasa", "oxford"),
    ]
    assert rows[-1]["evaluated"] is False
    assert rows[-1]["soh_r2"] is None
    assert "schema incompatible" in rows[-1]["notes"]


def test_cross_chemistry_benchmark_excludes_plain_lco_runs(db):
    _log(db, dataset="nasa", chemistry="LiCoO2")
    assert reg.cross_chemistry_benchmark() == []


# ---------------------------------------------------------------------------
# run_cross_chemistry_study -- the PERMANENT study (not a one-off script)
# ---------------------------------------------------------------------------

def _study_datasets():
    """Two datasets whose chemistries differ: "CellA" resolves to the synthetic
    LiCoO2 profile, "S-b1c2" to the Severson LFP profile."""
    return {
        "synth":    {"CellA": make_cycles_df(n_cycles=120)},
        "severson": {"S-b1c2": make_cycles_df(n_cycles=120, fade_per_cycle=0.0008)},
    }


def test_run_cross_chemistry_study_only_runs_cross_chemistry_pairs(db, monkeypatch):
    _fake_clock(monkeypatch)
    result = reg.run_cross_chemistry_study(_study_datasets())

    pairs = {r["dataset"] for r in reg.leaderboard(tenant_org_id=None)}
    assert pairs == {"synth_to_severson", "severson_to_synth"}
    assert len(result["evaluated"]) == 2


def test_run_cross_chemistry_study_includes_same_chemistry_cross_source_pairs(db, monkeypatch):
    """Two LiCoO2 sources (different dataset keys) ARE paired now: with more
    than one real source per chemistry (NASA + CALCE), cross-SOURCE transfer
    within a chemistry is its own generalization test — "does the model work
    on cells from a different cycler/form factor even when the cathode
    matches?" — and the notes must say which kind of study a row is."""
    _fake_clock(monkeypatch)
    datasets = {
        "synth": {"CellA": make_cycles_df(n_cycles=120)},
        "other": {"CellB": make_cycles_df(n_cycles=120)},
    }
    result = reg.run_cross_chemistry_study(datasets)
    assert len(result["evaluated"]) == 2
    bench = reg.cross_chemistry_benchmark()
    assert {(b["train_dataset"], b["eval_dataset"]) for b in bench} == {
        ("synth", "other"), ("other", "synth")
    }
    for row in bench:
        assert "Same-chemistry cross-source generalization study" in (row.get("notes") or "")


def test_run_cross_chemistry_study_is_idempotent_per_feature_version(db, monkeypatch):
    """Second call (same FEATURE_VERSION) must be a cheap registry read, not a
    retrain — otherwise every process start re-runs the study."""
    _fake_clock(monkeypatch)
    first = reg.run_cross_chemistry_study(_study_datasets())
    second = reg.run_cross_chemistry_study(_study_datasets())

    assert len(first["evaluated"]) == 2
    assert second["evaluated"] == []
    assert set(second["skipped"]) == {"synth_to_severson", "severson_to_synth"}
    # No duplicate runs accumulated.
    assert len(reg.leaderboard(tenant_org_id=None)) == 2


def test_run_cross_chemistry_study_reruns_when_feature_version_changes(db, monkeypatch):
    """The number depends on the feature set, so a FEATURE_VERSION bump must
    re-run and log a fresh row rather than serving a stale transfer metric."""
    _fake_clock(monkeypatch)
    reg.run_cross_chemistry_study(_study_datasets())

    import batlab.features.engineering as engineering
    monkeypatch.setattr(engineering, "FEATURE_VERSION", "v99-test")

    again = reg.run_cross_chemistry_study(_study_datasets())
    assert len(again["evaluated"]) == 2
    assert again["skipped"] == []


def test_run_cross_chemistry_study_logs_unavailable_pairing_once(db, monkeypatch):
    _fake_clock(monkeypatch)
    datasets = _study_datasets()
    reason = "Oxford's checkpoint schema has no cycle_number "
    first = reg.run_cross_chemistry_study(datasets, unavailable=[("synth", "oxford", reason)])
    second = reg.run_cross_chemistry_study(datasets, unavailable=[("synth", "oxford", reason)])

    assert first["unavailable"][0]["pair"] == "synth_to_oxford"
    assert second["unavailable"] == []  # schema reason is version-independent — once ever
    assert "synth_to_oxford" in second["skipped"]

    rows = reg.cross_chemistry_benchmark()
    oxford = next(r for r in rows if r["eval_dataset"] == "oxford")
    assert oxford["evaluated"] is False
    assert oxford["soh_r2"] is None


# ---------------------------------------------------------------------------
# Hyperparameter divergence -- the shared replay-contract check
# ---------------------------------------------------------------------------

def test_hyperparams_divergence_empty_when_run_matches_current_params(db):
    from batlab.models.gbrt import GBRT_PARAMS
    run_id = reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version=FEATURE_VERSION,
        hyperparams=dict(GBRT_PARAMS), seed=GBRT_PARAMS["random_state"],
        cell_ids=["A"], n_rows=10, lco_metrics=_lco_metrics(),
    )
    run = reg.get_run(reg.PLATFORM_ORG_ID, run_id)
    assert reg.hyperparams_divergence(run) == {}
    assert reg.format_hyperparams_diff({}) == ""


def test_hyperparams_divergence_names_recorded_and_current_values(db):
    run_id = reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="nasa", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version=FEATURE_VERSION,
        hyperparams={"n_estimators": 1}, seed=42,
        cell_ids=["A"], n_rows=10, lco_metrics=_lco_metrics(),
    )
    run = reg.get_run(reg.PLATFORM_ORG_ID, run_id)
    diff = reg.hyperparams_divergence(run)
    assert diff, "a partial/old snapshot must register as diverged"
    for recorded, current in diff.values():
        assert recorded != current
    text = reg.format_hyperparams_diff(diff)
    assert "→" in text


def test_hyperparams_divergence_none_and_missing_snapshot_are_diverged(db):
    """A run with no recorded hyperparameters at all must register as diverged
    (we cannot claim it is reproducible), and must not raise."""
    assert reg.hyperparams_divergence(None)
    assert reg.hyperparams_divergence({})


# ---------------------------------------------------------------------------
# model_kind — GBRT vs PINN side by side
# ---------------------------------------------------------------------------

def _log_kind(db, kind, dataset="nasa", chemistry="LiCoO2", soh_r2=0.8):
    return reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset=dataset, chemistry=chemistry,
        feature_set=["cycle_number"], feature_version=FEATURE_VERSION,
        hyperparams={"model_kind": kind}, seed=42,
        cell_ids=["CellA", "CellB"], n_rows=300,
        lco_metrics=_lco_metrics(soh_r2=soh_r2, baseline_soh_r2=0.5),
        model_kind=kind,
    )


def test_model_kind_defaults_to_gbrt_and_round_trips(db):
    """An unlabelled run is a GBRT fit (the production model) — never NULL,
    so no call site needs a special 'NULL means gbrt' rule."""
    run_id = _log(db)
    assert reg.get_run(reg.PLATFORM_ORG_ID, run_id)["model_kind"] == "gbrt"

    pinn_id = _log_kind(db, "pinn")
    assert reg.get_run(reg.PLATFORM_ORG_ID, pinn_id)["model_kind"] == "pinn"


def test_pinn_runs_never_pollute_per_chemistry_gbrt_accuracy(db):
    """The per-chemistry accuracy table is the PRODUCTION model's number. A
    PINN run on the same dataset must not be averaged into it or mistaken
    for it — silently conflating the two would misstate production accuracy."""
    _log_kind(db, "gbrt", soh_r2=0.90)
    _log_kind(db, "pinn", soh_r2=0.10)  # deliberately much worse

    rows = reg.accuracy_by_source(tenant_org_id=None)
    nasa = [r for r in rows if r["dataset"] == "nasa"]
    assert len(nasa) == 1
    assert nasa[0]["soh_r2"] == 0.90


def test_model_kind_comparison_pairs_kinds_and_reports_signed_delta(db):
    """Both kinds must appear for the same (dataset, chemistry), and the GBRT
    row must carry the PINN-minus-GBRT delta — negative when the PINN loses,
    which is a result, not something to hide."""
    _log_kind(db, "gbrt", soh_r2=0.98)
    _log_kind(db, "pinn", soh_r2=0.40)

    rows = reg.model_kind_comparison(tenant_org_id=None)
    kinds = {r["model_kind"] for r in rows}
    assert kinds == {"gbrt", "pinn"}

    gbrt_row = next(r for r in rows if r["model_kind"] == "gbrt")
    pinn_row = next(r for r in rows if r["model_kind"] == "pinn")
    assert gbrt_row["has_counterpart"] is True
    assert pinn_row["has_counterpart"] is True
    # The delta sits on the PINN row (the GBRT is the reference).
    assert pinn_row["pinn_minus_gbrt_soh_r2"] == pytest.approx(0.40 - 0.98)
    assert gbrt_row["pinn_minus_gbrt_soh_r2"] is None
    # The PINN row still carries its own genuine-advantage number.
    assert pinn_row["advantage"] == pytest.approx(0.40 - 0.5)


def test_model_kind_comparison_marks_absent_counterpart(db):
    _log_kind(db, "gbrt", soh_r2=0.9)
    rows = reg.model_kind_comparison(tenant_org_id=None)
    gbrt_row = next(r for r in rows if r["model_kind"] == "gbrt")
    assert gbrt_row["has_counterpart"] is False
    assert gbrt_row["pinn_minus_gbrt_soh_r2"] is None
    # No counterpart means no head-to-head row at all — only the GBRT row.
    assert {r["model_kind"] for r in rows} == {"gbrt"}


def test_pinn_benchmark_study_logs_a_first_class_run(db, monkeypatch):
    """The study runner must log a PINN run through log_run() with the
    shared baseline, and be idempotent for the current FEATURE_VERSION."""
    import batlab.validation.pinn_lco as pinn_lco

    cells = {
        "CellA": make_cycles_df(n_cycles=120, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=120, fade_per_cycle=0.0008,
                                initial_resistance_ohm=0.06),
    }
    datasets = {"nasa": cells}

    # Build features once so the study doesn't pay the full pipeline here.
    from batlab.features.engineering import build_features
    featured = {"nasa": {cid: build_features(df, cell_id=cid)
                         for cid, df in cells.items()}}

    first = reg.run_pinn_benchmark_study(
        datasets, featured=featured, baselines={"nasa": 0.5},
        org_id=reg.PLATFORM_ORG_ID,
    )
    assert len(first) == 1
    run = reg.get_run(reg.PLATFORM_ORG_ID, first[0]["run_id"])
    assert run["model_kind"] == "pinn"
    assert run["baseline_soh_r2"] == 0.5
    assert run["hyperparams"]["model_kind"] == "pinn"

    # Second call is a cheap registry read — no duplicate run.
    again = reg.run_pinn_benchmark_study(
        datasets, featured=featured, baselines={"nasa": 0.5},
        org_id=reg.PLATFORM_ORG_ID,
    )
    assert again == []
    assert len([r for r in reg.leaderboard(None) if r["model_kind"] == "pinn"]) == 1


# ── Prospective (temporal-holdout) benchmark study ──────────────────────────


def test_run_prospective_benchmark_study_logs_and_is_idempotent(db, monkeypatch):
    """The study logs one run per dataset under the `_prospective` suffix and
    a second call is a cheap registry read — no duplicate runs."""
    _fake_clock(monkeypatch)
    datasets = {
        "synth": {f"Cell{i}": make_cycles_df(n_cycles=200) for i in range(4)},
    }
    first = reg.run_prospective_benchmark_study(datasets, org_id=reg.PLATFORM_ORG_ID)
    assert len(first) == 1
    run = reg.get_run(reg.PLATFORM_ORG_ID, first[0]["run_id"])
    assert run["dataset"] == "synth_prospective"
    assert run["model_kind"] == "gbrt"
    assert "PROSPECTIVE evaluation" in (run["notes"] or "")
    assert "forecasting from curve-fitting" in (run["notes"] or "")
    # The trivial + formula baselines travel with the run.
    assert run["baseline_soh_r2"] is not None

    again = reg.run_prospective_benchmark_study(datasets, org_id=reg.PLATFORM_ORG_ID)
    assert again == []
    pros = [r for r in reg.leaderboard(None) if (r["dataset"] or "").endswith("_prospective")]
    assert len(pros) == 1


def test_prospective_rows_excluded_from_lco_aggregators(db, monkeypatch):
    """A `_prospective` run evaluates a different population (test windows
    under a temporal holdout, not leave-cell-out folds) — it must never be
    averaged into the per-chemistry LCO table or the GBRT-vs-PINN rows."""
    _fake_clock(monkeypatch)
    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="synth", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version="v-test",
        hyperparams={}, seed=42, cell_ids=["CellA"], n_rows=100,
        lco_metrics={"soh_mae": 1.0, "soh_r2": 0.9, "baseline_soh_r2": 0.6},
    )
    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="synth_prospective", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version="v-test",
        hyperparams={}, seed=50, cell_ids=["CellA"], n_rows=100,
        lco_metrics={"soh_mae": 9.0, "soh_r2": -0.5, "baseline_soh_r2": 0.4},
    )

    rows = reg.accuracy_by_source()
    assert [r["dataset"] for r in rows] == ["synth"]
    assert rows[0]["soh_r2"] == 0.9

    kinds = reg.model_kind_comparison()
    assert kinds and kinds[0]["dataset"] == "synth"
    assert not any((r.get("dataset") or "").endswith("_prospective") for r in kinds)


def test_prospective_benchmark_reader_reports_the_gap(db, monkeypatch):
    """The reader must place the prospective number NEXT to its dataset's LCO
    number — the gap between them is the finding."""
    _fake_clock(monkeypatch)
    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="synth", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version="v-test",
        hyperparams={}, seed=42, cell_ids=["CellA"], n_rows=100,
        lco_metrics={"soh_mae": 1.0, "soh_r2": 0.95, "baseline_soh_r2": 0.6},
    )
    datasets = {
        "synth": {f"Cell{i}": make_cycles_df(n_cycles=200) for i in range(4)},
    }
    logged = reg.run_prospective_benchmark_study(datasets, org_id=reg.PLATFORM_ORG_ID)
    assert logged

    rows = reg.prospective_benchmark()
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "synth"
    assert row["lco_soh_r2"] == 0.95
    assert row["soh_r2"] is not None
    assert row["forecasting_gap"] == row["soh_r2"] - 0.95
