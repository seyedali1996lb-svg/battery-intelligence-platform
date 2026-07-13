"""
Reproducible leave-cell-out benchmark manifests.

Exports the *exact* fold assignments a leave-cell-out (LCO) evaluation
used — which cells were held out, in what order, with what random seed
and feature-engineering version — as a JSON file. Re-running
evaluate_from_manifest() against the same cell data later (a different
machine, a future batlab version) reproduces the identical fold
structure, so a reported R2/MAE number can be checked rather than taken
on faith.

This is the seed of a reproducible benchmark, not a full benchmark
harness — it captures fold structure and provenance, not a leaderboard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from batlab.validation.lco import run_lco

MANIFEST_SCHEMA_VERSION = 1

# Bump whenever batlab.features.engineering.build_features() changes in a
# way that would make an old manifest's reported numbers non-reproducible
# against the current code.
FEATURE_VERSION = "batlab-v1"


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
    actually reproduced.

    Raises
    ------
    ValueError if cell_data is missing a manifest cell_id, or if the
    installed batlab's FEATURE_VERSION doesn't match the manifest's
    (results would not be reproducible against the code that generated
    the original numbers).
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

    restricted = {cid: cell_data[cid] for cid in manifest["cell_ids"]}
    result = run_lco(restricted, seed=manifest["seed"])
    result["manifest_seed"] = manifest["seed"]
    result["manifest_feature_version"] = manifest["feature_version"]
    result["manifest_cell_ids"] = manifest["cell_ids"]
    return result
