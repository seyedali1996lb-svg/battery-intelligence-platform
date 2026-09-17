"""
The raw cycles of a tenant's own upload, persisted so they can be graded and
sealed.

Why this module exists
----------------------
Until now an upload left two durable traces: the engineered frames
(bundle_cache.save_tenant_bundle) and a row of metadata (db.save_upload_meta).
The RAW cycles were dropped when the request ended. That is fine for training
and fine for the dashboard — and fatal for the two things this platform claims
to be good at:

- The validation harness re-derives features from raw cycles and fingerprints
  what it fingerprinted. Handed engineered frames it cannot say which dataset
  they describe, so "Bring your own model" could only ever grade the shared
  reference fleets (see app/_pages/model_validation.py's docstring).
- A sealed replication bundle records per-cell content digests and is verified
  by re-deriving the number from that same data. A bundle over data nobody can
  reload is a claim, not evidence.

So the enabling change is a store, keyed by the upload's own content hash (the
same `upload-<sha256[:20]>` key the caches already use), holding each cell's
cycle table plus a manifest carrying the dataset fingerprint. Everything
downstream — grading a tenant's own fleet, sealing a bundle they can hand to an
auditor — reads from here.

What is written, and why it is CSV
----------------------------------
Each cell is written as `fingerprints.cell_csv_text(df)` — the exact bytes
`cell_digest()` hashes, through that one shared function. So the stored file IS
the fingerprint's input: `sha256(data/uploaded_fleets/<key>/cells/B0005.csv)`
equals `cell_digests["B0005"]` in every report, registry row, and sealed
bundle, with no tooling in between. Storing a pickle or a Parquet table instead
would mean two serializations that must be kept in agreement by hand.

Privacy and retention, stated rather than implied
-------------------------------------------------
This is raw tenant data at rest, in the deployment's own data directory, for
every upload ever analysed (content-addressed, so re-uploading the same file
reuses one directory). Nothing here encrypts it — the platform's at-rest
protection for tenant data is the filesystem's, exactly as for
data/tenant_bundles/ — and "Clear uploaded data" in the app reverts the session,
it does not delete this. What it changes is that a tenant's own cycles can now
travel inside a sealed bundle they choose to share; that choice is offered and
labelled on the page that offers it, because handing someone the bundle then
hands them the data.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
from typing import Any

from batlab.validation.bundle_data import (
    CELLS_INDEX_NAME,
    load_bundle_cells,
    write_bundle_cells,
)
from batlab.validation.fingerprints import dataset_fingerprint

__all__ = [
    "STORE_SCHEMA",
    "STORE_SCHEMA_VERSION",
    "UPLOADED_STORE_DIR",
    "UploadedFleetError",
    "clear_uploaded_cell_data",
    "latest_uploaded_fleet",
    "list_uploaded_fleets",
    "load_uploaded_cell_data",
    "read_manifest",
    "save_uploaded_cell_data",
    "store_dir",
]

STORE_SCHEMA = "batlab-uploaded-fleet"
STORE_SCHEMA_VERSION = 1
CELLS_SUBDIR = "cells"
MANIFEST_NAME = "manifest.json"

# Overridable via BATLAB_UPLOADED_STORE_DIR (same convention as db.py's
# DATABASE_URL). It exists for a real workflow, not only for tests: a bundle
# sealed WITHOUT the embedded cycles records this module as its data path, so
# verifying it somewhere the store lives elsewhere — a restored backup, an
# auditor's copy, CI — needs a way to say where. Tests monkeypatch the module
# attribute instead of setting the env var, so a developer's own uploads are
# never read by the suite.
UPLOADED_STORE_DIR = pathlib.Path(
    os.environ.get("BATLAB_UPLOADED_STORE_DIR") or (pathlib.Path(__file__).parent.parent / "data" / "uploaded_fleets")
)

# A store key becomes a directory name, so it is validated rather than joined.
# The app generates `upload-<20 hex chars>` (app/_pages/import_page.py); the
# regex is deliberately stricter than "anything without a slash" so a key
# arriving from a URL, a form field, or a future caller cannot traverse.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


class UploadedFleetError(ValueError):
    """A persisted upload could not be written or read, with a named reason."""


def _validate_key(upload_key: str) -> str:
    key = str(upload_key or "").strip()
    if not _KEY_RE.match(key) or ".." in key:
        raise UploadedFleetError(
            f"{upload_key!r} is not a valid upload key — expected the "
            "'upload-<hex>' key the import pipeline computes"
        )
    return key


def store_dir(upload_key: str) -> pathlib.Path:
    """The directory holding one upload's raw cycles (not created here)."""
    return UPLOADED_STORE_DIR / _validate_key(upload_key)


def save_uploaded_cell_data(
    org_id: int,
    upload_key: str,
    cell_data: dict,
    *,
    meta: "dict | None" = None,
) -> dict:
    """Persist one upload's raw cycles + manifest. Returns the manifest.

    Idempotent by construction: the key IS the content hash, so re-analysing
    the same file rewrites the same directory with the same bytes. Raises
    UploadedFleetError for an unusable key, an empty fleet, or a cell table
    that is not a DataFrame, so a broken import is reported at the point of the
    mistake rather than as an unverifiable bundle a week later.
    """
    key = _validate_key(upload_key)
    if not isinstance(cell_data, dict) or not cell_data:
        raise UploadedFleetError("no cell cycle tables were supplied")
    for cell_id, df in cell_data.items():
        if not hasattr(df, "columns"):
            raise UploadedFleetError(
                f"cell {cell_id!r} is not a DataFrame — the raw cycles must be "
                "persisted as the tables the feature builder reads"
            )

    directory = store_dir(key)
    write_bundle_cells(cell_data, directory / CELLS_SUBDIR)

    fingerprint = dataset_fingerprint(cell_data)
    manifest: dict[str, Any] = {
        "schema": STORE_SCHEMA,
        "schema_version": STORE_SCHEMA_VERSION,
        "org_id": int(org_id),
        "upload_key": key,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_cells": fingerprint["n_cells"],
        "cell_ids": fingerprint["cell_ids"],
        "n_rows": int(sum(len(df) for df in cell_data.values())),
        "fingerprint": fingerprint,
        "cells_index": f"{CELLS_SUBDIR}/{CELLS_INDEX_NAME}",
        "context": dict(meta or {}),
        "note": (
            "Raw cycle tables for one tenant upload, stored as the exact bytes "
            "cell_digest() hashes. This is what lets the validation harness "
            "grade a model on this fleet and seal a bundle over it; the sealed "
            "bundle's cell_digests are the digests below."
        ),
    }
    # Written as bytes, like the cell tables themselves: no platform line-ending
    # translation, so the manifest is identical on every OS that writes it.
    (directory / MANIFEST_NAME).write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def read_manifest(upload_key: str) -> "dict | None":
    """The manifest for one upload, or None if it was never persisted.

    Returns None (rather than raising) for a missing or unreadable manifest:
    every caller is deciding whether to OFFER this fleet, and a corrupt file
    must read as "not available here", not as a page that crashes.
    """
    try:
        path = store_dir(upload_key) / MANIFEST_NAME
    except UploadedFleetError:
        return None
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema") != STORE_SCHEMA:
        return None
    return manifest


def load_uploaded_cell_data(upload_key: str, expected_org_id: "int | None" = None) -> dict:
    """Reload one upload's raw cycles: {cell_id: cycles DataFrame}.

    `expected_org_id` is the tenant-isolation check — the caller passes the
    session's org and a manifest belonging to another one is refused, so a
    leaked or guessed key is not a cross-tenant read. Omit it only where there
    is no session to check against (scripts, tests).

    Raises UploadedFleetError with the specific reason (unknown key, wrong org,
    missing or tampered cell file) — the page shows it verbatim.
    """
    key = _validate_key(upload_key)
    manifest = read_manifest(key)
    if manifest is None:
        raise UploadedFleetError(
            f"No persisted raw cycles for upload key {key!r} — this fleet can "
            "only be graded after an upload that stored them "
            "(src/uploaded_store.save_uploaded_cell_data)."
        )
    if expected_org_id is not None and int(manifest.get("org_id", -1)) != int(expected_org_id):
        raise UploadedFleetError(
            f"Upload {key!r} belongs to another organization — refusing to load it."
        )

    try:
        cell_data = load_bundle_cells(CELLS_SUBDIR, bundle_dir=store_dir(key))
    except ValueError as exc:
        raise UploadedFleetError(f"Stored cycles for {key!r} are unusable: {exc}") from exc

    # The manifest's fingerprint is what the harness will stamp into the report
    # and the bundle. Re-checking it here means a store that drifted (a partial
    # write, a hand-edit, a file restored from a different upload) is caught as
    # a named identity failure instead of silently grading different data.
    current = dataset_fingerprint(cell_data)
    if current["dataset_sha256"] != (manifest.get("fingerprint") or {}).get("dataset_sha256"):
        raise UploadedFleetError(
            f"Stored cycles for {key!r} no longer match their manifest fingerprint "
            "— the data changed on disk since it was persisted."
        )
    return cell_data


def list_uploaded_fleets(org_id: int) -> list[dict]:
    """Every persisted upload for one org, newest first.

    Unreadable manifests are skipped rather than reported: this is what a
    picker is built from, and one corrupt directory must not hide the others.
    """
    if not UPLOADED_STORE_DIR.exists():
        return []
    found: list[dict] = []
    for path in UPLOADED_STORE_DIR.glob(f"*/{MANIFEST_NAME}"):
        manifest = read_manifest(path.parent.name)
        if manifest is None or int(manifest.get("org_id", -1)) != int(org_id):
            continue
        found.append(manifest)
    found.sort(key=lambda m: str(m.get("created_utc") or ""), reverse=True)
    return found


def latest_uploaded_fleet(org_id: int) -> "dict | None":
    """The org's most recent persisted upload, or None."""
    fleets = list_uploaded_fleets(org_id)
    return fleets[0] if fleets else None


def clear_uploaded_cell_data(upload_key: str) -> bool:
    """Delete one upload's persisted raw cycles. True when something was removed.

    Deliberately NOT wired into the app's "Clear uploaded data", which is
    documented as session-scoped and leaves every persisted artifact in place
    (src/bundle_cache.clear_cache has the same contract). A tenant's raw data
    is worth being able to delete on request, so the capability exists here and
    is called explicitly.
    """
    import shutil

    key = _validate_key(upload_key)
    directory = store_dir(key)
    if not directory.exists():
        return False
    shutil.rmtree(directory)
    return True
