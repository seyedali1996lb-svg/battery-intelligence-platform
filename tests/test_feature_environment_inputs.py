"""Optional feature blocks that depend on the ENVIRONMENT, not the data.

`build_features()` opportunistically imports `physics_calibration`, which lives in
the demo application's `src/` — not in the library. So the same statement

    batlab.benchmark(batlab.load("nasa"))

returns a different number depending on whether `src/` happens to be importable:
SOH R² **0.9580** without it, **0.9471** with it (measured 2026-09-19, four NASA
cells). That is the same class of trap as "same dataset name is not same data",
except the axis is the Python path instead of the loader.

These tests pin the two properties that make the trap survivable: the run SAYS
which population it came from, and the fold cache cannot replay one population's
folds for the other.
"""

from __future__ import annotations

import builtins

import pytest

import batlab
from batlab.features.engineering import build_features


def _block_physics_import(monkeypatch):
    """Make `from physics_calibration import ...` fail, as it does in a wheel."""
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "physics_calibration":
            raise ImportError("blocked for this test — simulating a pip install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


@pytest.fixture(scope="module")
def nasa_two():
    cells = batlab.load("nasa", cell_ids=["B0005", "B0006"])
    assert len(cells) == 2
    return cells


def test_build_features_records_whether_the_physics_block_was_available(nasa_two):
    frame = build_features(nasa_two["B0005"], cell_id="B0005")
    assert "physics_features" in frame.attrs
    assert isinstance(frame.attrs["physics_features"], bool)


def test_a_run_says_which_population_its_number_came_from(nasa_two, tmp_path, monkeypatch):
    """The flag is reported, and it tracks the import — both ways."""
    monkeypatch.setenv("BATLAB_LCO_CACHE_DIR", str(tmp_path))

    with_physics = batlab.benchmark(nasa_two)
    assert "physics_features" in with_physics, "the population must travel with the number"
    assert with_physics["physics_features"] is True

    _block_physics_import(monkeypatch)
    without_physics = batlab.benchmark(nasa_two)
    assert without_physics["physics_features"] is False


def test_the_two_populations_are_actually_different(nasa_two, tmp_path, monkeypatch):
    """If this ever stops holding, the flag is noise — and the docstring lies.

    Asserted rather than assumed: the whole reason the flag exists is that the
    measured numbers differ by ~0.01 SOH R² between the two environments.
    """
    monkeypatch.setenv("BATLAB_LCO_CACHE_DIR", str(tmp_path))
    with_physics = batlab.benchmark(nasa_two)["soh_r2"]
    _block_physics_import(monkeypatch)
    without_physics = batlab.benchmark(nasa_two)["soh_r2"]
    assert with_physics != pytest.approx(without_physics, abs=1e-9)


def test_the_fold_cache_cannot_replay_across_the_two_populations(
    nasa_two, tmp_path, monkeypatch
):
    """Different key, so a fold fitted with the block is never served without it.

    This is the correctness half of the fix: without it, a developer's warm cache
    (src/ importable) would hand a pip-installed run folds that describe features
    it never computed — a replayed answer for a different model.
    """
    monkeypatch.setenv("BATLAB_LCO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BATLAB_LCO_CACHE", "on")

    with_physics_key = batlab.benchmark(nasa_two)["fold_cache"]["key"]
    _block_physics_import(monkeypatch)
    without_physics_key = batlab.benchmark(nasa_two)["fold_cache"]["key"]

    assert with_physics_key and without_physics_key
    assert with_physics_key != without_physics_key


def test_a_mixed_fleet_is_not_rounded_up_to_fully_calibrated(nasa_two, monkeypatch):
    """`all()`, not `any()`: one cell without the block means the run did not have it.

    Rounding a mixed fleet up to True would be the exact overclaim the flag exists
    to prevent.
    """
    from batlab.validation import lco as lco_mod

    real_build = lco_mod.build_features
    calls = {"n": 0}

    def _one_cell_less(df, *args, **kwargs):
        frame = real_build(df, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] > 1:
            frame.attrs["physics_features"] = False
        return frame

    monkeypatch.setattr(lco_mod, "build_features", _one_cell_less)
    assert batlab.benchmark(nasa_two)["physics_features"] is False
