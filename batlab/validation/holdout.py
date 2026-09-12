"""Sealed holdout with a hashed, tamper-evident manifest.

Why this exists
---------------
Reproducibility (batlab.validation.manifest) answers "can we re-run this
exact evaluation?". It does NOT answer "was this data ever used to develop
the model?" — a holdout set that leaked into feature tuning, hyperparameter
picking, or a training fold still reproduces perfectly. The only honest
holdout is a SEALED one: the holdout cells are identified and their exact
content is hashed at seal time; from then on every evaluation against the
holdout verifies (a) the cells are byte-identical to what was sealed, and
(b) the training population excludes them entirely.

What sealing gives you
----------------------
- `seal_holdout()` writes a JSON manifest with the holdout cell ids and the
  SHA-256 of each cell's normalized cycle table, plus the training pool's
  ids. The manifest records the platform FEATURE_VERSION so a later
  re-evaluation under changed feature code fails loudly instead of scoring
  numbers that aren't comparable.
- `verify_holdout()` re-hashes the live data and compares: any mutation of a
  holdout cell (more cycles appended, values edited, a cell swapped) breaks
  the hash and is reported, not silently accepted.
- `assert_clean_training_pool()` fails when a training dict contains a cell
  the manifest sealed — the check that makes "never trained on the holdout"
  a verified statement rather than a convention.

Honest scope
------------
Sealing makes a holdout verifiable, not virtuous: it cannot retroactively
prove that no development decision was ever informed by these cells. What
it does prove, mechanically, is that the data evaluated today is the data
that was sealed, and that the training folds exclude it. The remaining
discipline — never iterate on holdout results — stays a process rule, and
the manifest's `sealed_utc` timestamp makes its violation at least
detectable (results claimed before sealing cannot reference the seal).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HOLDOUT_SCHEMA_VERSION = 1
SEAL_ALGORITHM = "sha256"


def _cell_digest(cycles_df: pd.DataFrame) -> str:
    """Content hash of one cell's cycle table.

    Normalized so the digest is stable across rebuilds of the same data:
    sorted by cycle_number, column-ordered, float-formatted to a fixed
    precision (float64 repr is stable; we hash the CSV text rather than the
    in-memory representation to avoid dtype/platform endianness drift).
    """
    df = cycles_df.copy()
    if "cycle_number" in df.columns:
        df = df.sort_values("cycle_number")
    cols = list(df.columns)
    ordered = df[cols]
    text = ordered.to_csv(index=False, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seal_holdout(
    holdout_cell_data: dict,
    training_cell_ids: "list[str] | None" = None,
    path: "str | Path | None" = None,
    label: str = "",
) -> dict:
    """Seal a holdout population: hash its cells and write the manifest.

    Parameters
    ----------
    holdout_cell_data : {cell_id: raw cycles DataFrame} — the population that
        must stay untouched by any development or training use.
    training_cell_ids : the cell ids that ARE allowed for training (the rest
        of the fleet). Stored so `assert_clean_training_pool()` can verify
        membership even if a caller no longer has the training data itself.
    path : where to write the manifest JSON. When None, the manifest is only
        returned (useful for tests).
    label : free-text purpose of this holdout ("second-life bankability", ...).

    Returns the manifest dict (same content as the written file).
    Raises ValueError on an empty holdout or ids shared with the training pool.
    """
    if not holdout_cell_data:
        raise ValueError("seal_holdout(): holdout population is empty.")
    holdout_ids = list(holdout_cell_data.keys())
    training_ids = list(training_cell_ids or [])
    overlap = sorted(set(holdout_ids) & set(training_ids))
    if overlap:
        raise ValueError(
            f"seal_holdout(): cells {overlap} appear in BOTH the holdout and "
            "the training pool — a cell cannot be sealed and trainable."
        )

    manifest = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": SEAL_ALGORITHM,
        "label": label,
        "feature_version": _feature_version(),
        "holdout_cells": {
            cid: {"sha256": _cell_digest(df), "n_cycles": int(len(df))}
            for cid, df in holdout_cell_data.items()
        },
        "training_cell_ids": training_ids,
    }
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _feature_version() -> str:
    from batlab.features.engineering import FEATURE_VERSION
    return FEATURE_VERSION


def load_holdout(path: "str | Path") -> dict:
    """Load and minimally validate a sealed-holdout manifest."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != HOLDOUT_SCHEMA_VERSION:
        raise ValueError(
            f"Holdout manifest schema_version {manifest.get('schema_version')!r} != "
            f"supported version {HOLDOUT_SCHEMA_VERSION}."
        )
    return manifest


def verify_holdout(manifest: dict, holdout_cell_data: dict) -> dict:
    """Re-hash the live holdout cells and compare against the seal.

    Returns {"intact": bool, "missing": [...], "mutated": {...}, "extra": [...]}
    where `mutated` maps cell_id -> (sealed_sha256, live_sha256). `intact` is
    True only when every sealed cell is present and byte-identical.
    """
    sealed = manifest.get("holdout_cells", {})
    missing = [cid for cid in sealed if cid not in holdout_cell_data]
    mutated: dict = {}
    for cid, meta in sealed.items():
        if cid in missing:
            continue
        live = _cell_digest(holdout_cell_data[cid])
        if live != meta.get("sha256"):
            mutated[cid] = (meta.get("sha256"), live)
    extra = [cid for cid in holdout_cell_data if cid not in sealed]
    return {
        "intact": not missing and not mutated,
        "missing": missing,
        "mutated": mutated,
        "extra": extra,
    }


def assert_clean_training_pool(manifest: dict, training_cell_data: dict) -> None:
    """Raise when a training population contains a sealed holdout cell.

    Call this at the top of every training/evaluation entry point that
    consumes the fleet dict — it is the mechanical form of "the holdout was
    never trained on".
    """
    sealed_ids = set(manifest.get("holdout_cells", {}).keys())
    leaked = sorted(sealed_ids & set(training_cell_data.keys()))
    if leaked:
        raise ValueError(
            f"SEALED-HOLDOUT VIOLATION: sealed holdout cell(s) {leaked} found in "
            "the training pool. These cells are sealed for final evaluation only — "
            "remove them from training before this number can be reported."
        )
