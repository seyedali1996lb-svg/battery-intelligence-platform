"""Unit tests for src/experiment_registry.py — the GBRT run registry.

Covers log/get/leaderboard-query/replay paths. Same isolated-SQLite-file
fixture pattern as tests/test_db.py so nothing here touches the real
data/app.db.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

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


def _lco_metrics(soh_mae=1.0, soh_r2=0.8, rul_mae=20.0, rul_r2=0.5, rul_reliable=True, per_cell=None):
    return {
        "soh_mae": soh_mae, "soh_r2": soh_r2,
        "rul_mae": rul_mae, "rul_r2": rul_r2,
        "rul_reliable": rul_reliable,
        "per_cell": per_cell or {"CellA": {"soh_mae": soh_mae, "soh_r2": soh_r2,
                                            "rul_mae": rul_mae, "rul_r2": rul_r2}},
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
# replay_run
# ---------------------------------------------------------------------------

def test_replay_run_reproduces_recorded_metrics(db):
    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008, initial_resistance_ohm=0.06),
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
    assert "Cross-chemistry generalization study" in run["notes"]


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
