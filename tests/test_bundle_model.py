"""
Tests for a bundle that CARRIES the model it was published for
(batlab/harness/model_source.py + seal_bundle(model_source=...)).

The request this closes: a bundle published for a model nobody else has —
an uploaded module, or one of the app's catalogue baselines — could be sealed
but not *recomputed*, because the verifier had no way to obtain the model. The
recompute check is the one that re-derives the headline number, so without it
the artifact certified less than it appeared to.

What these tests defend, in order of how much damage their absence would do:

1. The command a reviewer is given has to work on a machine that has nothing
   but the unzipped bundle. Asserted end to end through the real CLI.
2. Loading a carried model must produce the model that was GRADED — not a
   lookalike. Asserted on predictions, bit for bit.
3. A model edited after sealing must be refused, and refused BEFORE it is
   imported: importing a module executes it, so digest-then-decide is the
   whole safety property. Asserted with a tampered file that would leave a
   side effect if it ever ran.
4. A bundle without a carried model must be byte-identical to what it was
   before this feature existed.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_cycles_df

from batlab.harness import seal_bundle, validate_forecaster
from batlab.harness.forecaster import SklearnForecaster, default_forecaster, forecaster_identity
from batlab.harness.model_source import (
    BUNDLE_MODEL_LOADER,
    MODEL_INDEX_NAME,
    import_module_source,
    load_bundle_model,
    write_bundle_model,
)
from batlab.validation.replication import (
    _load_model,
    _recorded_model_kwargs,
    format_verification,
    load_bundle,
    verify_bundle,
)

ROOT = Path(__file__).parent.parent

# A model this repo does not ship: the exact situation an uploaded module puts
# a verifier in. Deterministic, so recompute can be compared bit for bit.
_MODEL_SOURCE = '''\
"""A model that exists only here — the stand-in for an upload."""

from sklearn.linear_model import Ridge

from batlab.harness import SklearnForecaster


def make_model():
    return SklearnForecaster(Ridge(alpha=2.5, random_state=None), scale=True)
'''


def _fleet(n_cells: int = 3, n_cycles: int = 160) -> dict:
    return {
        f"C{i}": make_cycles_df(
            n_cycles=n_cycles,
            fade_per_cycle=0.005 * (1.0 + 0.08 * i),
            initial_resistance_ohm=0.05 + 0.004 * i,
        )
        for i in range(n_cells)
    }


@pytest.fixture(scope="module")
def fleet() -> dict:
    return _fleet()


@pytest.fixture(scope="module")
def carried_model():
    """The factory a verifier would get from the module, loaded in-process."""
    return import_module_source(_MODEL_SOURCE, module_name="carried_test_model")["factory"]


@pytest.fixture(scope="module")
def report_for_a_foreign_model(fleet):
    """The report of a run graded with a model only this test has."""
    from sklearn.linear_model import Ridge

    return validate_forecaster(
        fleet,
        model=SklearnForecaster(Ridge(alpha=2.5, random_state=None), scale=True),
        splits=("lco",),
        intervals=False,
        dataset="synth-eol",
    )


# ---------------------------------------------------------------------------
# 1. The printed command works with only the zip
# ---------------------------------------------------------------------------

def test_a_bundle_carrying_its_model_recomputes_end_to_end(report_for_a_foreign_model, fleet, tmp_path):
    """The whole feature, through the CLI a reviewer would actually run."""
    sealed = seal_bundle(
        report_for_a_foreign_model, fleet, tmp_path / "published",
        dataset="synth-eol", embed_data=True,
        model_source=_MODEL_SOURCE, model_entry_point="make_model",
    )
    assert sealed["model_source"]["loader"]["module"] == "batlab.harness.model_source"
    assert sealed["model_source"]["loader"]["function"] == "load_bundle_model"
    assert sealed["model_source"]["entry_point"] == "make_model"
    assert sealed["model_source"]["file"] == "model/module.py"
    assert "model/module.py" in sealed["files"]
    assert f"model/{MODEL_INDEX_NAME}" in sealed["files"]

    # Unzip it somewhere else entirely — the reviewer's machine, not ours.
    import shutil

    bundle_dir = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(tmp_path / "published", bundle_dir)

    result = subprocess.run(
        [sys.executable, "-m", "batlab.validation.replication", str(bundle_dir),
         "--loader", "batlab.validation.bundle_data:load_bundle_cells",
         "--model", BUNDLE_MODEL_LOADER, "--recompute"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replication verify: PASS" in result.stdout
    assert "[PASS] recompute" in result.stdout


def test_the_cli_resolves_a_recorded_loader_that_lives_in_this_checkout(tmp_path):
    """A reference-fleet bundle records a loader from THIS repo's `src/`.

    Found by running the printed command for real: the synth fleet's recorded
    data path is `experiment_registry:reload_reference_cell_data`, a top-level
    module under `src/`, so a reviewer running the CLI from the repo root got
    "data-identity: ModuleNotFoundError: No module named 'experiment_registry'"
    on a bundle that was entirely correct — the artifact's own command could
    not be run as printed.

    The CLI now resolves the checkout's own loader directories before it
    imports anything a bundle names. This test deletes PYTHONPATH from the
    child environment so it measures the CLI and not the developer's shell.
    """
    import shutil

    import experiment_registry

    from batlab.harness.model_source import import_module_source

    # Sealed EXACTLY as the app seals a reference fleet: the recorded loader is
    # the platform's own reloader for the public dataset the report was graded
    # on, and the model travels because it is not batlab's default.
    cells = experiment_registry.reload_reference_cell_data(dataset="synth")
    factory = import_module_source(_MODEL_SOURCE, module_name="cli_foreign_model")["factory"]
    report = validate_forecaster(cells, model=factory(), splits=("lco",),
                                 intervals=False, dataset="synth")
    seal_bundle(report, cells, tmp_path / "published", dataset="synth",
                loader="experiment_registry:reload_reference_cell_data",
                loader_kwargs={"dataset": "synth"},
                model_source=_MODEL_SOURCE, model_entry_point="make_model")
    bundle_dir = tmp_path / "reference" / "bundle"
    shutil.copytree(tmp_path / "published", bundle_dir)

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-m", "batlab.validation.replication", str(bundle_dir),
         "--loader", "experiment_registry:reload_reference_cell_data",
         "--model", BUNDLE_MODEL_LOADER, "--recompute"],
        cwd=ROOT, capture_output=True, text=True, timeout=600, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replication verify: PASS" in result.stdout
    assert "[PASS] data-identity" in result.stdout
    assert "[PASS] recompute" in result.stdout


def test_checkout_import_paths_are_the_ones_this_repo_needs():
    """The helper is what makes that command work, so pin its shape.

    Idempotent (a second call adds nothing), confined to this checkout, and
    sufficient for both recorded loaders the platform can emit: the reference
    reloader as a top-level module, and the tenant store as `src.*`.
    """
    import importlib

    from batlab.validation.replication import _ensure_checkout_importable

    # Start from the state the CLI actually starts in: neither path present.
    saved = list(sys.path)
    wanted = [str(ROOT / "src"), str(ROOT)]
    try:
        for path in wanted:
            while path in sys.path:
                sys.path.remove(path)

        # Forget them so the import below really resolves through sys.path
        # rather than hitting a module another test already imported.
        for name in ("experiment_registry", "src.uploaded_store"):
            sys.modules.pop(name, None)

        added = _ensure_checkout_importable()
        assert added == wanted
        assert _ensure_checkout_importable() == []  # no unbounded sys.path growth
        assert importlib.import_module("experiment_registry") is not None
        assert importlib.import_module("src.uploaded_store") is not None
    finally:
        sys.path[:] = saved


def test_the_loaded_model_is_the_model_that_was_graded(carried_model, report_for_a_foreign_model, fleet, tmp_path):
    """Same configuration, same fits — not a plausible lookalike.

    The report was graded with `SklearnForecaster(Ridge(alpha=2.5), scale=True)`;
    _MODEL_SOURCE declares that configuration. If the embedded text ever drifted
    from the graded model, the bundle would recompute a different number from a
    different (but equally plausible) model, and only this comparison notices.
    """
    from sklearn.linear_model import Ridge

    sealed = seal_bundle(report_for_a_foreign_model, fleet, tmp_path, dataset="synth-eol",
                         model_source=_MODEL_SOURCE)
    loaded = load_bundle_model(source_file=sealed["model_source"]["file"],
                               entry_point="make_model", bundle_dir=tmp_path)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] * 2.0 + 3.0)

    graded = SklearnForecaster(Ridge(alpha=2.5, random_state=None), scale=True)
    graded.fit(X, y)
    from_bundle = loaded()
    from_bundle.fit(X, y)

    assert np.array_equal(graded.predict(X), from_bundle.predict(X))
    assert forecaster_identity(loaded()) == forecaster_identity(carried_model())


# ---------------------------------------------------------------------------
# 2 + 3. Tampering is refused before anything is imported
# ---------------------------------------------------------------------------

def test_a_model_edited_after_sealing_is_refused_before_it_runs(report_for_a_foreign_model, fleet, tmp_path):
    """Digest first, import second — the only order that can be safe.

    The tampered file leaves a marker if it is ever imported, so this test also
    proves the refusal happened BEFORE execution rather than after.
    """
    seal_bundle(report_for_a_foreign_model, fleet, tmp_path, dataset="synth-eol",
                model_source=_MODEL_SOURCE, embed_data=True)
    marker = tmp_path / "EXECUTED.marker"
    (tmp_path / "model" / "module.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
        "def make_model():\n    return None\n"
    )

    with pytest.raises(ValueError) as exc:
        load_bundle_model(bundle_dir=tmp_path)
    assert "does not match the digest its own index records" in str(exc.value)
    assert not marker.exists(), "the tampered module was imported before being checked"

    # ... and the seal check fails independently, naming the file.
    result = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path, cell_data=fleet,
                           recompute=True)
    seal_check = next(c for c in result["checks"] if c["name"] == "seal")
    assert seal_check["status"] == "fail"
    assert "model/module.py" in seal_check["detail"]


def test_a_model_path_outside_the_bundle_is_refused(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("def make_model():\n    return None\n")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError) as exc:
        load_bundle_model(source_file="../outside.py", bundle_dir=model_dir)
    assert "not inside the bundle" in str(exc.value)


def test_a_carried_module_that_cannot_produce_a_model_says_why(report_for_a_foreign_model, fleet, tmp_path):
    cases = (
        ("def make_model():\n    raise SystemExit(3)\n", "SystemExit"),
        ("import sys\nsys.exit(4)\n", "SystemExit"),
        ("x = 1\n", "does not define make_model"),
        ("def make_model():\n    return None\n", "returned None"),
    )
    for i, (source, expected) in enumerate(cases):
        target = tmp_path / f"case_{i}"
        seal_bundle(report_for_a_foreign_model, fleet, target, dataset="synth-eol",
                    model_source=source)
        with pytest.raises(ValueError) as exc:
            load_bundle_model(bundle_dir=target)
        assert "not usable" in str(exc.value)
        assert expected in str(exc.value)


def test_writing_a_model_never_imports_it(tmp_path):
    """An artifact writer must not run the code it stores."""
    marker = tmp_path / "ran.txt"
    source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran')\n"
        "def make_model():\n    return None\n"
    )
    index = write_bundle_model(source, tmp_path, entry_point="make_model", label="test")
    assert not marker.exists()
    assert index["sha256"]
    assert index["entry_point"] == "make_model"


@pytest.mark.parametrize("bad", ["", "   \n", None])
def test_empty_or_missing_model_source_is_refused(bad, tmp_path):
    with pytest.raises((ValueError, TypeError)):
        write_bundle_model(bad, tmp_path)


@pytest.mark.parametrize("bad_name", ["../module.py", "a/b.py", ".."])
def test_a_model_file_name_that_is_a_path_is_refused(bad_name, tmp_path):
    with pytest.raises(ValueError):
        write_bundle_model("def make_model():\n    return None\n", tmp_path, file_name=bad_name)


# ---------------------------------------------------------------------------
# 4. Backwards compatibility, and the failure message when nothing is carried
# ---------------------------------------------------------------------------

def test_a_bundle_without_a_carried_model_is_unchanged(report_for_a_foreign_model, fleet, tmp_path):
    sealed = seal_bundle(report_for_a_foreign_model, fleet, tmp_path, dataset="synth-eol",
                         loader="batlab.datasets.nasa:load_nasa_cells")
    assert "model_source" not in sealed
    assert not (tmp_path / "model").exists()
    assert sorted(sealed["files"]) == ["benchmark.json", "harness_report.json"]
    assert _recorded_model_kwargs(sealed) == {}


def test_the_recompute_failure_names_the_carried_model_and_how_to_use_it(
    report_for_a_foreign_model, fleet, tmp_path
):
    """No silent gap: the check says exactly what is missing and what fixes it."""
    seal_bundle(report_for_a_foreign_model, fleet, tmp_path, dataset="synth-eol",
                model_source=_MODEL_SOURCE)
    without_model = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path,
                                  cell_data=fleet, recompute=True)
    check = next(c for c in without_model["checks"] if c["name"] == "recompute")
    assert check["status"] == "fail"
    assert BUNDLE_MODEL_LOADER in check["detail"]
    assert "read" in check["detail"].lower()

    # With the model, the same bundle passes — same data, same fold list.
    factory = _load_model(BUNDLE_MODEL_LOADER, bundle_dir=tmp_path,
                          recorded_kwargs=_recorded_model_kwargs(load_bundle(tmp_path)))
    with_model = verify_bundle(load_bundle(tmp_path), bundle_dir=tmp_path, cell_data=fleet,
                               recompute=True, forecaster=factory)
    assert with_model["verdict"] == "pass", format_verification(with_model)


def test_a_plain_model_factory_is_still_passed_through_uncalled(tmp_path):
    """The `--model module:factory` path must not be called by the verifier.

    It is returned untouched so run_lco can call it once per fold; calling it
    here as a probe would be harmless-looking and would change what a factory
    with side effects (or a counter) sees.
    """
    calls = []

    def make_forecaster():
        calls.append(1)
        return default_forecaster(42)

    import types

    module = types.ModuleType("plain_factory_test_module")
    module.make_forecaster = make_forecaster  # type: ignore[attr-defined]
    sys.modules["plain_factory_test_module"] = module
    try:
        loaded = _load_model("plain_factory_test_module:make_forecaster")
        assert loaded is make_forecaster
        assert calls == []
    finally:
        sys.modules.pop("plain_factory_test_module", None)


def test_the_recorded_model_kwargs_reach_the_loader(report_for_a_foreign_model, fleet, tmp_path):
    """`--model <loader>` alone is the whole command: file and entry point come
    from the bundle, not from the reviewer's memory."""
    seal_bundle(report_for_a_foreign_model, fleet, tmp_path, dataset="synth-eol",
                model_source=_MODEL_SOURCE, model_entry_point="make_model")
    bundle = load_bundle(tmp_path)
    kwargs = _recorded_model_kwargs(bundle)
    assert kwargs == {"source_file": "model/module.py", "entry_point": "make_model"}

    loaded = _load_model(BUNDLE_MODEL_LOADER, bundle_dir=tmp_path, recorded_kwargs=kwargs)
    assert callable(loaded)

    # What comes back is a zero-argument FACTORY, and what it returns is a
    # fresh, UNFITTED model — one per fold, exactly like the page's uploads.
    first, second = loaded(), loaded()
    assert first is not second
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0], "c": [3.0, 4.0, 5.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="before fit"):
        first.predict(X)
    first.fit(X, y)
    assert np.asarray(first.predict(X)).shape == (3,)
