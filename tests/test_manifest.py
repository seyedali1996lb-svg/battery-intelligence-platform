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

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest
from conftest import make_cycles_df

from batlab.validation.manifest import (
    FEATURE_VERSION,
    export_split_manifest,
    load_manifest,
    evaluate_from_manifest,
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
