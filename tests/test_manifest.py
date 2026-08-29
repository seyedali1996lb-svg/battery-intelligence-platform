"""Unit tests for batlab.validation.manifest — reproducible LCO manifests.

Includes regression coverage for the FEATURE_VERSION single-source-of-truth
fix: batlab/validation/manifest.py and src/bundle_cache.py both used to keep
their own separately-maintained copy of this string, in sync only by a
non-blocking CI warning. Both now import it from
batlab.features.engineering instead, so it is structurally impossible for
them to disagree -- these tests guard against a future regression back to a
hardcoded duplicate.
"""

import json
import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
import numpy as np
import pandas as pd
import pytest
import sklearn
from conftest import make_cycles_df

from batlab.validation.manifest import (
    BENCHMARK_SCHEMA,
    BENCHMARK_SCHEMA_VERSION,
    FEATURE_VERSION,
    evaluate_from_manifest,
    export_benchmark_results,
    export_split_manifest,
    load_benchmark_results,
    load_manifest,
)


def test_manifest_feature_version_is_engineering_feature_version():
    """The actual regression this guards: manifest.py must not define its
    own copy of FEATURE_VERSION -- it must be the same object as
    batlab.features.engineering's, not just an equal-by-value string that
    could silently drift."""
    from batlab.features.engineering import FEATURE_VERSION as engineering_version
    assert FEATURE_VERSION is engineering_version


def test_bundle_cache_cache_version_incorporates_feature_version():
    """src/bundle_cache.py's app-level cache signature must bump whenever
    batlab.features.engineering.FEATURE_VERSION does -- confirmed here by
    checking it's embedded in the composed CACHE_VERSION, not a separate
    hardcoded copy."""
    import bundle_cache
    assert FEATURE_VERSION in bundle_cache.CACHE_VERSION
    assert bundle_cache.MODEL_VERSION in bundle_cache.CACHE_VERSION


def test_export_and_load_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    written = export_split_manifest(["CellA", "CellB", "CellC"], path, seed=7)
    loaded = load_manifest(path)
    assert loaded == written
    assert loaded["feature_version"] == FEATURE_VERSION
    assert loaded["seed"] == 7
    assert len(loaded["folds"]) == 3


def test_export_split_manifest_records_installed_library_versions(tmp_path):
    """The reproducibility fix: a manifest now pins numpy/pandas/scikit-learn
    versions at export time, so "reproducible across environments" is
    something evaluate_from_manifest() can actually check, not just an
    unverified claim covering "reproducible within one run"."""
    import numpy, pandas, sklearn

    path = tmp_path / "manifest.json"
    written = export_split_manifest(["CellA", "CellB"], path, seed=1)
    assert written["environment"] == {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }


def test_evaluate_from_manifest_warns_but_does_not_raise_on_library_version_drift(tmp_path):
    """Unlike feature_version, a numpy/pandas/scikit-learn mismatch doesn't
    hard-fail (a routine `pip install -U scikit-learn` shouldn't break every
    existing manifest) -- it's surfaced via a warning and the
    "environment_match"/"environment_diff" result fields instead."""
    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008),
    }
    path = tmp_path / "manifest.json"
    manifest = export_split_manifest(list(cell_data.keys()), path, seed=1)
    manifest["environment"]["scikit-learn"] = "0.0.0-not-the-real-version"
    path.write_text(json.dumps(manifest))

    with pytest.warns(UserWarning, match="scikit-learn"):
        result = evaluate_from_manifest(path, cell_data)

    assert result["environment_match"] is False
    assert result["environment_diff"]["scikit-learn"][0] == "0.0.0-not-the-real-version"


def test_evaluate_from_manifest_reports_environment_match_when_versions_agree(tmp_path):
    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008),
    }
    path = tmp_path / "manifest.json"
    export_split_manifest(list(cell_data.keys()), path, seed=1)

    result = evaluate_from_manifest(path, cell_data)
    assert result["environment_match"] is True
    assert result["environment_diff"] == {}


def test_evaluate_from_manifest_reproduces_lco(tmp_path):
    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008, initial_resistance_ohm=0.06),
    }
    path = tmp_path / "manifest.json"
    export_split_manifest(list(cell_data.keys()), path, seed=42)

    result = evaluate_from_manifest(path, cell_data)
    assert result["manifest_seed"] == 42
    assert result["manifest_feature_version"] == FEATURE_VERSION
    assert set(result["manifest_cell_ids"]) == set(cell_data.keys())


def test_evaluate_from_manifest_rejects_stale_feature_version(tmp_path, monkeypatch):
    """If FEATURE_VERSION ever changes (build_features() behavior changed)
    and someone tries to reproduce an old manifest against the new code,
    that must fail loudly instead of silently reporting numbers that aren't
    actually reproducible."""
    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008),
    }
    path = tmp_path / "manifest.json"
    export_split_manifest(list(cell_data.keys()), path, seed=1, feature_version="stale-version")

    with pytest.raises(ValueError, match="feature_version"):
        evaluate_from_manifest(path, cell_data)


def test_evaluate_from_manifest_rejects_missing_cells(tmp_path):
    path = tmp_path / "manifest.json"
    export_split_manifest(["CellA", "CellB", "CellC"], path, seed=1)

    with pytest.raises(ValueError, match="missing"):
        evaluate_from_manifest(path, {"CellA": make_cycles_df()})


# ---------------------------------------------------------------------------
# Benchmark bundle export (machine-readable interop format)
# ---------------------------------------------------------------------------


def _fake_lco_result():
    """A run_lco()-shaped dict with numpy scalar metrics — exercises the
    JSON serializer's numpy handling without paying for a real LCO run."""
    return {
        "soh_r2": float(np.float64(0.806)),
        "soh_mae": float(np.float64(3.70)),
        "rul_r2": float(np.float64(0.629)),
        "rul_mae": float(np.float64(8.06)),
        "rul_reliable": True,
        "per_cell": {
            "CellA": {"soh_mae": 3.1, "soh_r2": 0.81, "rul_mae": 7.2, "rul_r2": 0.60},
            "CellB": {"soh_mae": 4.3, "soh_r2": 0.80, "rul_mae": 8.9, "rul_r2": 0.66},
        },
    }


def test_export_and_load_benchmark_round_trip(tmp_path):
    path = tmp_path / "benchmark.json"
    written = export_benchmark_results(_fake_lco_result(), path, cell_ids=["CellA", "CellB"], seed=9)
    loaded = load_benchmark_results(path)
    assert loaded == written
    assert loaded["schema"] == BENCHMARK_SCHEMA
    assert loaded["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert loaded["metrics"]["soh_r2"] == 0.806
    assert loaded["metrics"]["rul_reliable"] is True
    assert loaded["seed"] == 9
    assert loaded["feature_version"] == FEATURE_VERSION
    assert len(loaded["folds"]) == 2
    assert loaded["folds"][0]["test_cell"] == "CellA"
    assert loaded["folds"][0]["train_cells"] == ["CellB"]
    assert loaded["per_cell"]["CellA"]["rul_r2"] == 0.60


def test_export_benchmark_pins_environment_and_uses_numpy_clean_serialization(tmp_path):
    """The interop contract: the number ships with the environment and fold
    structure it was produced under, and numpy scalars never break JSON."""
    import numpy as np

    result = _fake_lco_result()
    result["soh_r2"] = np.float64(0.9)  # raw numpy scalar on purpose
    path = tmp_path / "benchmark.json"
    written = export_benchmark_results(result, path)
    assert written["environment"] == {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    assert written["metrics"]["soh_r2"] == 0.9
    # round-trip through real JSON, not just the in-memory dict
    assert load_benchmark_results(path)["metrics"]["soh_r2"] == 0.9


def test_load_benchmark_rejects_wrong_schema_or_version(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"schema": "something-else", "schema_version": 1}))
    with pytest.raises(ValueError, match="Not a batlab benchmark"):
        load_benchmark_results(path)

    path.write_text(json.dumps({"schema": BENCHMARK_SCHEMA, "schema_version": 999}))
    with pytest.raises(ValueError, match="schema_version"):
        load_benchmark_results(path)


def test_export_benchmark_requires_lco_shaped_result(tmp_path):
    with pytest.raises(ValueError, match="per_cell"):
        export_benchmark_results({"soh_r2": 0.8}, tmp_path / "x.json")
    with pytest.raises(ValueError, match="at least 2 cells"):
        export_benchmark_results({"per_cell": {"CellA": {}}}, tmp_path / "x.json")
