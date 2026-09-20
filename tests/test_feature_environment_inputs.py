"""One set of features in every environment.

``build_features()`` used to import ``physics_calibration`` opportunistically
from the demo application's ``src/`` — not from the library — so

    batlab.benchmark(batlab.load("nasa"))

returned a different number depending on whether ``src/`` happened to be
importable: SOH R² **0.9580** without the block, **0.9471** with it (measured
2026-09-19, four NASA cells). Same class of trap as "same dataset name is not
same data", with the Python path as the axis — and a library whose numbers move
with the caller's import path is not a library.

The block now lives in ``batlab.features.physics_calibration``, so that axis is
gone. These tests pin what replaced it:

  * the population no longer tracks the environment (the regression itself),
  * the run still SAYS which population its number came from,
  * and the fold cache still cannot replay one population's folds for another.
"""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

import batlab
from batlab.features.engineering import FEATURE_VERSION, build_features

ROOT = Path(__file__).resolve().parent.parent


def _block_import(monkeypatch, name: str):
    """Make ``import <name>`` fail, as it does where that module is absent."""
    real_import = builtins.__import__

    def _import(module_name, *args, **kwargs):
        if module_name == name:
            raise ImportError(f"blocked for this test: {name}")
        return real_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


def _without_the_physics_block(monkeypatch):
    """Reproduce the pre-0.2.0 pip-install state: the block cannot be computed.

    Makes ``calibrated_feature_series`` raise, which is what a missing module did
    before — so ``build_features()`` takes the same degraded path (all-NaN columns
    and ``physics_features=False``), giving the second population test 4/5 need.
    Note that flipping only the flag would NOT be enough: the flag records how the
    frame was built, and the numbers move with the COLUMNS, not the flag.
    """
    from batlab.features import physics_calibration as pc

    def _unavailable(df, cell_id=None):
        raise ImportError("simulated: physics block unavailable in this environment")

    monkeypatch.setattr(pc, "calibrated_feature_series", _unavailable)


@pytest.fixture(scope="module")
def nasa_two():
    cells = batlab.load("nasa", cell_ids=["B0005", "B0006"])
    assert len(cells) == 2
    return cells


def test_build_features_records_whether_the_physics_block_was_computed(nasa_two):
    frame = build_features(nasa_two["B0005"], cell_id="B0005")
    assert frame.attrs["physics_features"] is True
    assert frame["physics_beta_sei"].notna().any()


def test_the_library_owns_the_block_and_never_imports_the_application_module():
    """The seam is gone, not merely disclosed.

    A bare ``import physics_calibration`` inside the library is what made a
    feature column environment-dependent; if it ever comes back, this fails even
    if the numbers happen to agree on this machine. Checked structurally (AST)
    rather than by grepping text, so prose about the old seam stays allowed.
    """
    import ast

    from batlab.features import engineering, physics_calibration

    assert Path(physics_calibration.__file__).resolve().is_relative_to(ROOT / "batlab")
    for module in (engineering, physics_calibration):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        bare_app_imports = [
            node.names[0].name if isinstance(node, ast.Import) else node.module
            for node in ast.walk(tree)
            if (
                isinstance(node, (ast.Import, ast.ImportFrom))
                and (getattr(node, "module", None) or getattr(node.names[0], "name", "")) == "physics_calibration"
            )
        ]
        assert bare_app_imports == [], (
            f"{module.__name__} imports the demo app's physics_calibration module again — "
            "use batlab.features.physics_calibration"
        )


def test_blocking_the_application_module_does_not_change_the_number(nasa_two, tmp_path, monkeypatch):
    """The original trap: numbers that moved with ``src/`` on sys.path.

    Blocking the *application* module is exactly what a wheel/notebook process
    looks like. The number and the population flag must be identical.
    """
    monkeypatch.setenv("BATLAB_LCO_CACHE_DIR", str(tmp_path))
    before = batlab.benchmark(nasa_two)

    _block_import(monkeypatch, "physics_calibration")
    after = batlab.benchmark(nasa_two)

    # The old code had NO physics block in this process at all (all-NaN columns,
    # SOH R² 0.9580 on four cells); now it computes the same block either way.
    assert after["physics_features"] is True
    assert after["soh_r2"] == pytest.approx(before["soh_r2"], abs=1e-12)


def test_a_run_says_which_population_its_number_came_from(nasa_two, tmp_path, monkeypatch):
    """The flag still tracks a real difference — so it is not decorative.

    Driven from the data side now (a cell whose physics block was not computed),
    which is the difference that survives; the environment can no longer move it.
    """
    monkeypatch.setenv("BATLAB_LCO_CACHE_DIR", str(tmp_path))
    with_physics = batlab.benchmark(nasa_two)
    assert with_physics["physics_features"] is True

    _without_the_physics_block(monkeypatch)
    without_physics = batlab.benchmark(nasa_two)
    assert without_physics["physics_features"] is False
    # If this ever stops holding the flag is noise — the docstring would lie.
    assert without_physics["soh_r2"] != pytest.approx(with_physics["soh_r2"], abs=1e-9)


def test_the_fold_cache_cannot_replay_across_the_two_populations(nasa_two, tmp_path, monkeypatch):
    """Different feature values, different key — never a replayed answer.

    A developer's warm cache (block present) must not hand a block-less run folds
    that describe features it never computed: that would be a replayed answer for
    a different model.
    """
    monkeypatch.setenv("BATLAB_LCO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BATLAB_LCO_CACHE", "on")

    with_physics_key = batlab.benchmark(nasa_two)["fold_cache"]["key"]

    _without_the_physics_block(monkeypatch)
    without_physics_key = batlab.benchmark(nasa_two)["fold_cache"]["key"]

    assert with_physics_key and without_physics_key
    assert with_physics_key != without_physics_key


def test_the_applications_loader_declares_provenance():
    """The app's own loader must declare what it loaded, like every batlab loader.

    ``src/data_loader.build_battery()`` reads the NASA and synthetic CSVs under
    ``data/raw/`` directly, and never declared ``source``/``chemistry`` — which
    the batlab schema requires. Harmless until the physics block's eligibility
    started reading those attrs, at which point an undeclared frame would have
    silently lost the SEI/LAM features the app had always had.
    """
    from data_loader import build_battery

    from batlab.features.physics_calibration import _eligible_for_calibration

    nasa_frame = build_battery(battery_id="NASA_B1", cell_ids=["B0005"])["cells"]["B0005"]["cycles"]
    assert nasa_frame.attrs["source"] == "nasa"
    assert nasa_frame.attrs["chemistry"] == "LiCoO2"
    assert _eligible_for_calibration(nasa_frame, "B0005") is True

    synth_frame = build_battery(battery_id="SYN_B1", cell_ids=["Cell1"])["cells"]["Cell1"]["cycles"]
    assert synth_frame.attrs["source"] == "synthetic"
    assert _eligible_for_calibration(synth_frame, "Cell1") is False


def test_profile_dataset_sources_match_the_loaders_they_describe():
    """The app's provenance table is checked against the real loaders, not itself.

    ``dataset_source`` exists only so the app can declare the same string the
    library's loaders do; if the two ever disagree, an app frame and a library
    frame of the same fleet would be different populations.
    """
    from chemistry_profiles import ChemistryProfile

    nasa_cells = batlab.load("nasa", cell_ids=["B0005"])
    # The Severson loader has no cell_ids argument; every one of its ids is
    # "S-<key>", so any cell it returns carries the same profile.
    severson_cells = batlab.load("severson")
    assert severson_cells, "the committed Severson summaries should load"
    severson_frame = next(iter(severson_cells.values()))

    for cell_id, frame in (("B0005", nasa_cells["B0005"]), ("S-b1c0", severson_frame)):
        profile = ChemistryProfile.for_cell(cell_id)
        assert frame.attrs["source"] == profile.dataset_source
        assert frame.attrs["chemistry"] == profile.short_name


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


def test_installed_without_the_application_on_the_path_the_features_are_populated():
    """End-to-end in a process that only has the library importable.

    This is the cheapest faithful reproduction of "pip install battery-lab, run
    the quickstart": no ``src/`` anywhere on sys.path, one NASA cell, and the
    physics block must be populated (0.9580-era code produced all-NaN columns
    here and 0.9471 in the app — the discrepancy this pins shut).
    """
    code = (
        "import sys, warnings; warnings.filterwarnings('ignore');"
        "sys.path.insert(0, sys.argv[1]);"
        "import batlab;"
        "from batlab.features.engineering import build_features, FEATURE_VERSION;"
        "df = batlab.load('nasa', cell_ids=['B0005'])['B0005'];"
        "f = build_features(df, cell_id='B0005');"
        "print('@@@', FEATURE_VERSION, f.attrs['physics_features'], int(f['physics_beta_sei'].notna().sum()))"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("@@@"))
    version, flag, populated = line[4:].split()
    assert version == FEATURE_VERSION
    assert flag == "True"
    assert int(populated) > 0, "the physics block must not be all-NaN without src/ on the path"
