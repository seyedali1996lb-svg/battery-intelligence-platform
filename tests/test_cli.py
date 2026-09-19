"""`python -m batlab` / the `batlab` console script.

The point of these tests is that the CLI is a *thin wrapper* over the public
API rather than a second implementation of it: the end-to-end test compares the
CLI's own JSON report against an in-process `batlab.benchmark()` call on the same
fleet, so a drift between the two fails here instead of in a user's script.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import batlab
from batlab.cli import main

def test_version_verb_prints_the_library_version(capsys):
    assert main(["version"]) == 0
    assert batlab.__version__ in capsys.readouterr().out


def test_cite_verb_prints_bibtex(capsys):
    assert main(["cite"]) == 0
    out = capsys.readouterr().out
    assert "@" in out and "batlab" in out


def test_datasets_verb_lists_every_dataset(capsys):
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    for name in batlab.AVAILABLE_DATASETS:
        assert name in out


def test_unknown_dataset_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        main(["benchmark", "--dataset", "not-a-dataset"])


def test_malformed_loader_spec_explains_the_expected_form(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["benchmark", "--loader", "not_a_spec"])
    assert "module:function" in str(excinfo.value)


def test_missing_loader_module_is_reported_not_traced():
    with pytest.raises(SystemExit) as excinfo:
        main(["benchmark", "--loader", "no_such_module_xyz:load"])
    assert "no_such_module_xyz" in str(excinfo.value)


def test_benchmark_json_matches_the_in_process_number(tmp_path):
    """The CLI's number IS `batlab.benchmark()`'s number — compared in ONE process.

    Both halves run in a subprocess on purpose. In this test process `batlab`
    resolves with the repository's `src/` on sys.path (conftest bootstraps it via
    `_paths`), which switches on the optional physics-calibration feature block
    and changes the number — a real, now-disclosed population difference (see
    tests/test_feature_environment_inputs.py). Comparing a CLI subprocess against
    an in-process call would therefore compare two different populations and
    prove nothing.
    """
    out_path = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "batlab", "benchmark",
            "--dataset", "nasa", "--out", str(out_path),
        ],
        capture_output=True, text=True, cwd=str(_repo_root()),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["loader"].endswith("load_nasa_cells")

    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import json, batlab; "
            "lco = batlab.benchmark(batlab.load('nasa')); "
            "print(json.dumps({'lco': lco}, default=float))",
        ],
        capture_output=True, text=True, cwd=str(_repo_root()),
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    in_process = json.loads(probe.stdout.strip().splitlines()[-1])["lco"]

    assert report["lco"]["soh_r2"] == pytest.approx(in_process["soh_r2"], abs=1e-9)
    assert set(report["lco"]) == set(in_process)
    cache = report["lco"]["fold_cache"]
    assert cache["hits"] + cache["fitted"] == 4


def test_json_default_encodes_numpy_scalars_and_arrays():
    import numpy as np

    from batlab.cli import _json_default

    assert _json_default(np.float32(0.5)) == pytest.approx(0.5)
    assert _json_default(np.array([1.0, 2.0])) == [1.0, 2.0]
    with pytest.raises(TypeError):
        _json_default(object())


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[1]
