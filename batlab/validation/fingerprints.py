"""
Dataset and environment fingerprints — pinning WHAT DATA and WHAT CODE
ENVIRONMENT produced a number, so "same dataset" is checkable byte-for-byte
instead of asserted by name.

Why this exists
---------------
Tier 4's NASA study-row incident is the case this module prevents: two runs
both said "NASA", but one loaded 636 rows and the other 580 for the same
four cells through different preprocessing — and the ensemble's apparent
+0.187 win was an artifact of the difference, discovered only by hand.
A fingerprint logged with every run makes that comparison mechanical:
two rows are comparable only if their dataset_fingerprints match.

The per-cell digest reuses the exact normalization holdout.py established
(sorted by cycle_number, fixed float formatting, hashed as CSV text) so
"this run's B0005 is byte-identical to that run's B0005" is a string
comparison, not a judgment call. The environment snapshot extends
manifest.py's tracked-package versioning to the interpreter and platform,
because a fit depends on all of them.

Pure functions over DataFrames and importlib.metadata — no filesystem
writes, no DB — so tests can pin every behavior directly.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from importlib.metadata import version as _pkg_version

import pandas as pd

# Libraries whose version can change a fit's numbers, extended from
# manifest.py's list: the interpreter itself and pybamm (the physics
# calibration inside build_features) are part of the computation too.
_TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "pybamm")


def cell_digest(cycles_df) -> str:
    """Content hash of one cell's cycle table.

    Same normalization as holdout._cell_digest: sorted by cycle_number,
    column-ordered, CSV-serialized at fixed float precision, SHA-256. The
    digest is stable across rebuilds of identical data and insensitive to
    dtype/endianness — but ANY change in the numbers (a loader preprocessing
    fix, a sentinel cleanup, an extra row) changes it, which is the point.

    Accepts the wrapper shapes callers actually pass (build_battery cells:
    {"cell_id", "cycles"}; loader dicts: {"cycles"}) — the same
    normalization rule as lco.unwrap_cell_data, so a digest never silently
    depends on which shape a caller happened to hold.
    """
    if isinstance(cycles_df, dict):
        cycles_df = cycles_df.get("cycles")
    if not isinstance(cycles_df, pd.DataFrame):
        raise TypeError("cell_digest expects a cycles DataFrame (or a {'cycles': df} wrapper)")
    df = cycles_df.copy()
    if "cycle_number" in df.columns:
        df = df.sort_values("cycle_number")
    text = df.to_csv(index=False, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def environment_snapshot() -> dict:
    """The interpreter + library environment a fit runs in.

    Recorded with every logged run (registry `environment` column) and in
    every replication bundle, so "reproduced in a different environment"
    is a checked claim: versions that differ are listed, not guessed at.
    """
    snapshot: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in _TRACKED_PACKAGES:
        try:
            snapshot[pkg] = _pkg_version(pkg)
        except Exception:
            snapshot[pkg] = "not installed"
    return snapshot


def dataset_fingerprint(cell_data: dict) -> dict:
    """Fingerprint a whole dataset: one digest per cell plus a set-level hash.

    The set hash is order-independent (sorted digest concatenation) so the
    same cells in a different dict order produce the same fingerprint —
    what identifies the dataset is its CONTENT, not iteration order.

    Returns {n_cells, cell_ids (sorted), cell_digests {id: sha256},
    dataset_sha256}.
    """
    digests = {cid: cell_digest(df) for cid, df in cell_data.items()}
    ordered = [digests[cid] for cid in sorted(digests)]
    set_hash = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    return {
        "n_cells": len(digests),
        "cell_ids": sorted(digests),
        "cell_digests": digests,
        "dataset_sha256": set_hash,
    }


def compare_fingerprints(a: dict, b: dict) -> dict:
    """Compare two dataset fingerprints.

    Returns {match, n_cells_match, cell_ids_match, changed_cells,
    added_cells, removed_cells}. `changed_cells` names exactly which cells'
    content moved — the mechanical answer to "same dataset name, same data?"
    """
    da = a.get("cell_digests") or {}
    dbb = b.get("cell_digests") or {}
    ids_a, ids_b = set(da), set(dbb)
    changed = sorted(cid for cid in ids_a & ids_b if da[cid] != dbb[cid])
    return {
        "match": bool(
            a.get("dataset_sha256") == b.get("dataset_sha256")
            and not changed
            and ids_a == ids_b
        ),
        "n_cells_match": a.get("n_cells") == b.get("n_cells"),
        "cell_ids_match": ids_a == ids_b,
        "changed_cells": changed,
        "added_cells": sorted(ids_b - ids_a),
        "removed_cells": sorted(ids_a - ids_b),
    }
