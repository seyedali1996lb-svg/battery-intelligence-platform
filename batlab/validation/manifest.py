"""
Reproducible leave-cell-out benchmark manifests.

Exports the *exact* fold assignments a leave-cell-out (LCO) evaluation
used — which cells were held out, in what order, with what random seed
and feature-engineering version — as a JSON file. Re-running
evaluate_from_manifest() against the same cell data later (a different
machine, a future batlab version) reproduces the identical fold
structure, so a reported R2/MAE number can be checked rather than taken
on faith.

feature_version mismatches hard-fail (see evaluate_from_manifest) because
a build_features() code change is a definite break — the feature values
themselves would differ. numpy/pandas/scikit-learn version drift is
different: GradientBoostingRegressor can (rarely) produce different fits
across library versions with identical code, seed, and input data, but
usually doesn't. That's recorded in the manifest and checked at evaluation
time too, but only warns (via the returned "environment_match" field) — a
hard fail on every routine `pip install -U scikit-learn` would make
manifests too brittle to actually use.

This is the seed of a reproducible benchmark, not a full benchmark
harness — it captures fold structure and provenance, not a leaderboard.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np

from batlab.features.engineering import FEATURE_VERSION
from batlab.validation.lco import run_lco

MANIFEST_SCHEMA_VERSION = 2

# Schema marker + version for the machine-readable benchmark bundle written
# by export_benchmark_results() (see below) — distinct from the split-only
# manifest schema above: a benchmark bundle records BOTH the split structure
# AND the actual reported metrics, so other software can consume the number
# together with the conditions it was produced under.
BENCHMARK_SCHEMA = "batlab-lco-benchmark"
BENCHMARK_SCHEMA_VERSION = 1

# Re-exported from batlab.features.engineering — the single source of truth
# for "which version of build_features() produced this feature data" (see
# that module for why). Bump it there, not here; this name stays for
# backwards-compatible imports (batlab.validation.FEATURE_VERSION).

# Libraries whose version affects whether re-running an LCO fold produces
# identical numbers, not just identical code. Recorded in every exported
# manifest (see _current_environment()) so "reproducible across
# time/environment" is something evaluate_from_manifest() actually checks,
# not just an unverified claim covering "reproducible within one run".
_TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn")


def _current_environment() -> dict[str, str]:
    """Installed versions of the libraries an LCO fit depends on."""
    return {pkg: _pkg_version(pkg) for pkg in _TRACKED_PACKAGES}


def export_split_manifest(
    cell_ids: list[str],
    path: str | Path,
    seed: int = 42,
    feature_version: str = FEATURE_VERSION,
) -> dict:
    """
    Write a JSON manifest describing an LCO benchmark's fold structure.

    Each of the N input cells becomes its own fold (held out once, trained
    on the other N-1) — the same fold structure batlab.validation.lco.run_lco()
    always uses. What this manifest adds is a durable, inspectable record of
    *which* cells, in *what* order, with *what* seed and feature version —
    so evaluate_from_manifest() can later confirm a reported result was
    reproduced under the same conditions, not just re-run blindly.

    Parameters
    ----------
    cell_ids : every cell in the benchmark population, in a fixed order.
    path : where to write the manifest JSON.
    seed : random seed used for every fold's GradientBoostingRegressor.
    feature_version : tag identifying the feature-engineering code version
        used — compared against FEATURE_VERSION at evaluation time so a
        stale manifest is flagged, not silently reused.

    Returns
    -------
    The manifest dict that was written (same content as the JSON file).
    """
    if len(cell_ids) < 2:
        raise ValueError("A leave-cell-out manifest needs at least 2 cells.")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "feature_version": feature_version,
        "environment": _current_environment(),
        "cell_ids": list(cell_ids),
        "folds": [
            {
                "fold": i,
                "test_cell": test_cell,
                "train_cells": [c for c in cell_ids if c != test_cell],
            }
            for i, test_cell in enumerate(cell_ids)
        ],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(path: str | Path) -> dict:
    """Load and minimally validate a split manifest JSON file."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema_version {manifest.get('schema_version')!r} != "
            f"supported version {MANIFEST_SCHEMA_VERSION}. Re-export it with "
            "the current batlab.validation.manifest."
        )
    return manifest


def export_benchmark_results(
    result: dict,
    path: str | Path,
    cell_ids: list[str] | None = None,
    seed: int = 42,
    feature_version: str = FEATURE_VERSION,
) -> dict:
    """
    Write a machine-readable LCO benchmark bundle: the fold structure a
    split manifest records (which cells, what order, what seed/feature
    version) PLUS the actual reported metrics and per-cell fold results.

    This is the interop surface: other software (a BESS optimizer, a
    partner's fleet tool, a benchmark aggregator) can consume a reported
    SOH/RUL accuracy number together with the exact conditions it was
    produced under, without re-implementing this library's LCO semantics.
    `result` is any dict shaped like batlab.validation.lco.run_lco()'s
    return value (soh_r2/soh_mae/rul_r2/rul_mae/rul_reliable/per_cell) —
    the output of evaluate_from_manifest() works unchanged.

    Returns the benchmark dict that was written (same content as the JSON
    file). `cell_ids` fixes the fold enumeration order (default: the order
    of `result["per_cell"]`).
    """
    if "per_cell" not in result:
        raise ValueError("result must be shaped like run_lco()'s return value — "
                         "missing 'per_cell'.")
    if cell_ids is None:
        cell_ids = list(result["per_cell"].keys())
    if len(cell_ids) < 2:
        raise ValueError("A benchmark bundle needs at least 2 cells.")

    benchmark = {
        "schema": BENCHMARK_SCHEMA,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "feature_version": feature_version,
        "environment": _current_environment(),
        "metrics": {
            k: result[k]
            for k in ("soh_r2", "soh_mae", "rul_r2", "rul_mae", "rul_reliable")
            if k in result
        },
        "per_cell": {
            cell: {k: v for k, v in fold.items() if not isinstance(v, np.ndarray)}
            for cell, fold in result["per_cell"].items()
        },
        "folds": [
            {
                "fold": i,
                "test_cell": test_cell,
                "train_cells": [c for c in cell_ids if c != test_cell],
            }
            for i, test_cell in enumerate(cell_ids)
        ],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(benchmark, indent=2, default=_json_default))
    return benchmark


def _json_default(obj):
    """Serialize numpy scalars (e.g. np.float64 metrics) to plain JSON types."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_benchmark_results(path: str | Path) -> dict:
    """Load and validate a benchmark bundle written by export_benchmark_results()."""
    benchmark = json.loads(Path(path).read_text())
    if benchmark.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError(
            f"Not a batlab benchmark bundle (schema {benchmark.get('schema')!r} != "
            f"{BENCHMARK_SCHEMA!r})."
        )
    if benchmark.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"Benchmark schema_version {benchmark.get('schema_version')!r} != "
            f"supported version {BENCHMARK_SCHEMA_VERSION}. Re-export it with "
            "the current batlab.validation.manifest."
        )
    return benchmark


def evaluate_from_manifest(manifest: dict | str | Path, cell_data: dict) -> dict:
    """
    Re-run the LCO evaluation a manifest describes, against `cell_data`.

    Parameters
    ----------
    manifest : a manifest dict, or a path to a manifest JSON file.
    cell_data : {cell_id: DataFrame} — must contain every cell_id in the
        manifest's "cell_ids" list (extra cells are ignored, so a manifest
        built on a subset of a larger loaded fleet can still be reproduced
        without re-loading the exact original set).

    Returns
    -------
    Same shape as batlab.validation.lco.run_lco(), plus the manifest's
    provenance fields ("manifest_seed", "manifest_feature_version",
    "manifest_cell_ids") echoed back so a caller can confirm what was
    actually reproduced, plus "environment_match" (bool) and
    "environment_diff" (dict of {package: (manifest_version,
    installed_version)} for every mismatch, empty if none) recording
    whether numpy/pandas/scikit-learn match what the manifest was
    originally exported under.

    Raises
    ------
    ValueError if cell_data is missing a manifest cell_id, or if the
    installed batlab's FEATURE_VERSION doesn't match the manifest's
    (results would not be reproducible against the code that generated
    the original numbers). Does NOT raise on a numpy/pandas/scikit-learn
    version mismatch — see module docstring for why; check
    "environment_match" in the result instead.
    """
    if isinstance(manifest, (str, Path)):
        manifest = load_manifest(manifest)

    missing = [c for c in manifest["cell_ids"] if c not in cell_data]
    if missing:
        raise ValueError(
            f"cell_data is missing {len(missing)} cell(s) required by this "
            f"manifest: {missing}"
        )

    if manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError(
            f"Manifest feature_version {manifest['feature_version']!r} does not "
            f"match the installed batlab's FEATURE_VERSION {FEATURE_VERSION!r} — "
            "results would not be reproducible against the code that produced "
            "the original manifest. Re-export the manifest with the current "
            "batlab version, or install the matching batlab version."
        )

    manifest_env = manifest.get("environment", {})
    current_env = _current_environment()
    env_diff = {
        pkg: (manifest_env[pkg], current_env[pkg])
        for pkg in _TRACKED_PACKAGES
        if pkg in manifest_env and manifest_env[pkg] != current_env[pkg]
    }
    if env_diff:
        warnings.warn(
            "This manifest was exported under different library versions than "
            f"are currently installed: {env_diff} (format: {{package: (manifest, "
            "installed)}}). GradientBoostingRegressor can produce slightly "
            "different fits across library versions even with identical code, "
            "seed, and data — treat this result as reproduced-in-spirit, not "
            "byte-for-byte confirmed. Install the manifest's exact versions for "
            "a stricter check.",
            stacklevel=2,
        )

    restricted = {cid: cell_data[cid] for cid in manifest["cell_ids"]}
    result = run_lco(restricted, seed=manifest["seed"])
    result["manifest_seed"] = manifest["seed"]
    result["manifest_feature_version"] = manifest["feature_version"]
    result["manifest_cell_ids"] = manifest["cell_ids"]
    result["environment_match"] = not env_diff
    result["environment_diff"] = env_diff
    return result
