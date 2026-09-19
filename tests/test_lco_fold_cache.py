"""Per-fold leave-cell-out caching (batlab.validation.fold_cache).

The cache exists so an interrupted or repeated boot replays completed folds
instead of refitting them. What these tests pin is the pair of properties that
makes that safe rather than merely fast:

  1. a replayed fold is the SAME fold — a second run over a warm cache returns
     a result dict indistinguishable from the first, with zero model fits;
  2. a stale cache can never be served — any change to the cells, the feature
     version, the hyperparameters, the seed or the numeric stack produces a
     different key (so a changed input is recomputed, not replayed).

Every test points BATLAB_LCO_CACHE_DIR at tmp_path, so the real checkout cache
is never written to or read from.
"""

import numpy as np
import pytest

from conftest import make_cycles_df

import batlab.validation.fold_cache as fc
import batlab.validation.lco as lco_mod
from batlab.validation.lco import run_lco


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # conftest pins BATLAB_LCO_CACHE=off for the suite; this file is the one
    # place that turns it back on, always against its own tmp_path so a
    # developer's real checkout cache is never read or written.
    monkeypatch.setenv(fc.CACHE_DIR_ENV, str(tmp_path / "lco"))
    monkeypatch.setenv(fc.CACHE_MODE_ENV, fc._MODE_ON)


def _fleet():
    return {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008, initial_resistance_ohm=0.06),
    }


def _counting_fits():
    """Count every model fit run_lco's folds perform, without changing one."""
    real = lco_mod.fit_forecaster
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    lco_mod.fit_forecaster = counted
    return calls, lambda: setattr(lco_mod, "fit_forecaster", real)


def _identical(a: dict, b: dict, keys) -> None:
    for key in keys:
        av, bv = a[key], b[key]
        if isinstance(av, float) and av != av:
            assert isinstance(bv, float) and bv != bv, key
        else:
            assert av == bv, f"{key}: {av!r} != {bv!r}"


def test_second_run_replays_every_fold_without_fitting():
    """Warm cache → zero fits, identical numbers."""
    fleet = _fleet()

    calls, restore = _counting_fits()
    try:
        first = run_lco(fleet)
    finally:
        restore()
    assert len(calls) > 0, "the cold run must actually fit"

    calls, restore = _counting_fits()
    try:
        second = run_lco(fleet)
    finally:
        restore()

    assert calls == [], "a warm cache must not refit a single fold"
    assert second["fold_cache"]["hits"] == 2
    assert second["fold_cache"]["fitted"] == 0
    assert first["fold_cache"]["hits"] == 0
    assert first["fold_cache"]["fitted"] == 2

    _identical(first, second, (
        "soh_r2", "soh_mae", "rul_r2", "rul_mae", "rul_reliable",
        "rul_label_coverage", "n_rul_observed_rows", "n_rul_extrapolated_rows",
    ))
    assert first["per_cell"].keys() == second["per_cell"].keys()
    for cid, fold in first["per_cell"].items():
        _identical(fold, second["per_cell"][cid], fold.keys())
    assert first["fingerprint"] == second["fingerprint"]


def test_interrupted_run_resumes_only_the_missing_folds(tmp_path):
    """The case this cache exists for: a run killed partway through keeps the
    folds that finished, and the next run fits only what never landed."""
    fleet = _fleet()
    full = run_lco(fleet)
    key_dir = fc.cache_dir() / full["fold_cache"]["key"]

    # Simulate the kill: one fold's result survived, the other never landed.
    assert (key_dir / "CellB.joblib").exists()
    (key_dir / "CellB.joblib").unlink()

    calls, restore = _counting_fits()
    try:
        resumed = run_lco(fleet)
    finally:
        restore()

    assert resumed["fold_cache"]["hits"] == 1
    assert resumed["fold_cache"]["fitted"] == 1
    assert len(calls) == 2, "one fold refit, two fits (SOH + RUL) — not four"
    _identical(full, resumed, ("soh_r2", "soh_mae", "rul_r2", "rul_mae"))
    for cid in full["per_cell"]:
        _identical(full["per_cell"][cid], resumed["per_cell"][cid], full["per_cell"][cid].keys())


def test_changed_cell_content_is_recomputed_not_replayed():
    """A loader/preprocessing fix changes the content digest — and therefore
    the key — so a stale fold can never be served as if it were current."""
    fleet = _fleet()
    first = run_lco(fleet)
    assert first["fold_cache"]["fitted"] == 2

    fleet["CellA"] = make_cycles_df(n_cycles=200, fade_per_cycle=0.0009)
    second = run_lco(fleet)

    assert second["fingerprint"]["dataset"]["dataset_sha256"] != first["fingerprint"]["dataset"]["dataset_sha256"]
    assert second["fold_cache"]["fitted"] == 2, "changed data must be refit"
    assert second["fold_cache"]["key"] != first["fold_cache"]["key"]


def test_changed_seed_or_params_change_the_key():
    cells = _fleet()
    base = {
        "dataset_sha256": "d" * 64, "env_snapshot": {"scikit-learn": "1.9.0"},
        "cell_ids": ["CellA", "CellB"], "include_predictions": False,
    }
    key_a = fc.fold_key(fc.for_run(seed=42, params={"max_depth": 4}, **base)._base, "CellA")
    key_seed = fc.fold_key(fc.for_run(seed=7, params={"max_depth": 4}, **base)._base, "CellA")
    key_params = fc.fold_key(fc.for_run(seed=42, params={"max_depth": 5}, **base)._base, "CellA")
    key_other_cell = fc.fold_key(fc.for_run(seed=42, params={"max_depth": 4}, **base)._base, "CellB")

    assert len({key_a, key_seed, key_params, key_other_cell}) == 4


def test_feature_version_and_environment_are_part_of_the_key(monkeypatch):
    cells = _fleet()
    base = {
        "dataset_sha256": "d" * 64, "env_snapshot": {"scikit-learn": "1.9.0"},
        "cell_ids": ["CellA", "CellB"], "include_predictions": False,
        "seed": 42, "params": {"max_depth": 4},
    }
    before = fc.fold_key(fc.for_run(**base)._base, "CellA")

    monkeypatch.setattr(fc, "_feature_version", lambda: "v99-test-only")
    after_version = fc.fold_key(fc.for_run(**base)._base, "CellA")

    monkeypatch.undo()
    other_env = fc.fold_key(
        fc.for_run(**{**base, "env_snapshot": {"scikit-learn": "2.0.0"}})._base, "CellA"
    )

    assert before != after_version, "a feature-version change must bust the key"
    assert before != other_env, "a different numeric stack can fit a different model"


def test_cache_off_fits_every_time_and_writes_nothing(monkeypatch):
    monkeypatch.setenv(fc.CACHE_MODE_ENV, "off")
    fleet = _fleet()

    calls, restore = _counting_fits()
    try:
        first = run_lco(fleet)
        second = run_lco(fleet)
    finally:
        restore()

    assert len(calls) == 8, "two folds x two fits, twice over with no reuse"
    assert first["fold_cache"]["enabled"] is False
    assert second["fold_cache"]["enabled"] is False
    assert not fc.cache_dir().exists(), "off means off: nothing written"


def test_refresh_mode_recomputes_but_repopulates(monkeypatch):
    fleet = _fleet()
    run_lco(fleet)  # populate

    monkeypatch.setenv(fc.CACHE_MODE_ENV, "refresh")
    calls, restore = _counting_fits()
    try:
        refreshed = run_lco(fleet)
    finally:
        restore()

    assert refreshed["fold_cache"]["hits"] == 0
    assert refreshed["fold_cache"]["fitted"] == 2
    assert len(calls) == 4, "refresh ignores existing entries (2 folds x 2 fits)"

    monkeypatch.delenv(fc.CACHE_MODE_ENV)
    calls, restore = _counting_fits()
    try:
        after = run_lco(fleet)
    finally:
        restore()
    assert calls == [], "and repopulates them: the next run is a full hit again"
    assert after["fold_cache"]["hits"] == 2


def test_custom_forecaster_is_never_cached():
    """A caller-supplied model has no identity this module can key on, so its
    folds must be recomputed every time rather than replayed from a key built
    for the platform's own GBRT."""
    from batlab.harness.forecaster import default_forecaster

    fleet = _fleet()
    first = run_lco(fleet, forecaster=default_forecaster(42))
    second = run_lco(fleet, forecaster=default_forecaster(42))

    assert first["fold_cache"]["enabled"] is False
    assert second["fold_cache"]["enabled"] is False
    assert second["fold_cache"]["hits"] == 0
    # And the platform-default run in the same directory still caches.
    default_run = run_lco(fleet)
    assert default_run["fold_cache"]["enabled"] is True
    assert default_run["fold_cache"]["fitted"] == 2


def test_hits_and_fitted_add_up_to_the_fold_count():
    fleet = {
        f"Cell{i}": make_cycles_df(n_cycles=160, fade_per_cycle=0.0006 + i * 1e-4)
        for i in range(4)
    }
    run_lco(fleet)
    again = run_lco(fleet)
    summary = again["fold_cache"]
    assert summary["hits"] + summary["fitted"] == len(fleet) == 4


def test_describe_reports_what_is_on_disk():
    fleet = _fleet()
    run_lco(fleet)

    info = fc.describe()
    assert info["dir"] == str(fc.cache_dir())
    assert len(info["keys"]) == 1
    entry = info["keys"][0]
    assert entry["folds"] == 2
    assert entry["n_cells"] == 2
    assert entry["dataset_sha256"]


def test_corrupt_fold_file_misses_instead_of_raising():
    """A truncated or unreadable entry is a miss: a cache must never be able to
    fail a training run."""
    fleet = _fleet()
    first = run_lco(fleet)
    key_dir = fc.cache_dir() / first["fold_cache"]["key"]
    (key_dir / "CellA.joblib").write_bytes(b"not a joblib payload")

    recovered = run_lco(fleet)
    assert recovered["fold_cache"]["hits"] == 1
    assert recovered["fold_cache"]["fitted"] == 1
    _identical(first, recovered, ("soh_r2", "soh_mae", "rul_r2", "rul_mae"))


def test_clear_removes_cached_folds():
    run_lco(_fleet())
    assert fc.describe()["keys"]
    removed = fc.clear()
    assert removed >= 1
    assert fc.describe()["keys"] == []


def test_fold_cache_summary_travels_with_the_result():
    fleet = _fleet()
    result = run_lco(fleet)
    summary = result["fold_cache"]
    for key in ("mode", "enabled", "key", "dir", "hits", "fitted"):
        assert key in summary
    # dir is the cache ROOT a human inspects; key names the subdirectory
    # holding this run's folds.
    assert summary["dir"] == str(fc.cache_dir())
    assert (fc.cache_dir() / summary["key"]).is_dir()
    assert np.isfinite(result["soh_r2"])


def test_use_fold_cache_false_refits_even_with_a_warm_cache():
    """Verification callers — replication's independent recompute, the metric
    regression gate — must re-derive the numbers: a check that replays a
    stored answer is not a check."""
    fleet = _fleet()
    run_lco(fleet)  # populate a warm cache

    calls, restore = _counting_fits()
    try:
        verified = run_lco(fleet, use_fold_cache=False)
    finally:
        restore()

    assert len(calls) == 4, "two folds x two fits, actually refit"
    assert verified["fold_cache"]["enabled"] is False
    assert verified["fold_cache"]["hits"] == 0
    assert np.isfinite(verified["soh_r2"])
