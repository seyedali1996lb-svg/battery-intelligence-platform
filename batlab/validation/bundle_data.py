"""
Raw cycle tables carried INSIDE a replication bundle — so a sealed number can
be re-derived by someone who has the bundle and nothing else.

Why this exists
---------------
Every bundle this platform sealed until now recorded a *pointer* to the data:
`batlab.validation.replication --loader some.module:function`. That is the
right design for a public reference fleet (NASA, Severson, Zhu — reproducible
by anyone with the dataset and this package), and it is exactly the wrong
design for a tenant's own upload, which is not public, is not re-downloadable,
and lives only in the deployment's store. Publishing "seal" as a capability and
then requiring the publisher's filesystem to check it is a weaker claim than it
sounds.

`embed_data=True` on seal_bundle() closes that gap: the raw cycles are written
into the bundle under `cells/`, and the bundle records the bundle-relative
loader below, so

    unzip sealed_bundle.zip -d bundle && \
    python -m batlab.validation.replication bundle \\
        --loader batlab.validation.bundle_data:load_bundle_cells --recompute

re-derives the headline metric with no data path, no deployment, and no
network. The trade is explicit and stated where it is offered: the DATA travels
with the bundle, so anyone you hand it to receives the cycles themselves.

Two serializations, and why
---------------------------
`cell_digest()` hashes its canonical form at `%.10g` — ten significant digits,
which is a *fingerprint*: exact enough that no two different numbers collide,
short enough to be stable and diffable. That form is LOSSY as storage. A cell
table written at `%.10g`, read back, and re-scored lands ~1e-5 away from the
original fit (GBRT splits flip on differences far below that), which is forty
times the replication recompute tolerance — a bundle carrying its own data
would fail its own recompute check. So the stored table is written at `%.17g`,
the precision at which a double survives the round trip bit-exactly, and each
index entry records BOTH digests:

- `sha256` — the file's own bytes, checkable with `sha256sum` and covered by
the bundle's seal;
- `cell_digest` — the frame's canonical fingerprint, equal to the
  `cell_digests` entry in replication.json, which is what verify_bundle()
  compares a third party's own data against.

So the chain is checkable end to end without trusting this module: file bytes →
index entry → canonical digest → replication.json, and a recompute that agrees
with the published metric to 1e-6 rather than to "close enough".

Reading is deliberately strict. A bundle is attacker-controlled input at the
moment a third party verifies it: `index.json` could name `../../../etc/passwd`
or replace a cell's bytes after publication. Both are refused with a message
naming the cell and the reason, and every file's recorded digest is checked on
read — never "load whatever the index says and hash it later".
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd

from batlab.validation.fingerprints import cell_digest

__all__ = [
    "BUNDLE_CELLS_SCHEMA",
    "BUNDLE_CELLS_SCHEMA_VERSION",
    "CELLS_INDEX_NAME",
    "BUNDLE_CELLS_LOADER",
    "cell_file_name",
    "load_bundle_cells",
    "read_bundle_cell_file",
    "table_csv_text",
    "write_bundle_cells",
]

# The 'module:function' spec a bundle records when it carries its own cycles.
# Recorded rather than hard-coded at the call site so the writer, the docs and
# the verifier's printed command cannot spell it three different ways.
BUNDLE_CELLS_LOADER = "batlab.validation.bundle_data:load_bundle_cells"

BUNDLE_CELLS_SCHEMA = "batlab-bundle-cells"
BUNDLE_CELLS_SCHEMA_VERSION = 1
CELLS_INDEX_NAME = "index.json"

# A filename safe to join onto the bundle directory: no separators, no
# traversal, no hidden-or-absolute names. Cell ids come from a user's CSV, so
# "B0005" is the norm but hashing a hostile id into a path is not a risk worth
# taking when an index file can carry the real id instead.
_SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def cell_file_name(cell_id: str, index: int = 0) -> str:
    """A filesystem-safe file name for one cell's table.

    The cell id is used verbatim whenever it is already safe (the readable,
    diffable case); anything else falls back to `cell_<index>`, and the index
    file — not the name — remains the authority on which cell a file holds.
    """
    text = str(cell_id)
    if _SAFE_STEM_RE.match(text) and ".." not in text and not text.endswith(".csv"):
        return f"{text}.csv"
    return f"cell_{index:04d}.csv"


def table_csv_text(cycles_df) -> str:
    """The canonical ordering and formatting, at a precision that round-trips.

    Same normalization as `fingerprints.cell_csv_text` (sorted by
    cycle_number, column-ordered, LF endings) with `%.17g` instead of `%.10g`:
    17 significant digits is the point at which every IEEE double survives
    write-then-read unchanged, so a reloaded table scores identically to the
    one that was sealed. See the module docstring for why the fingerprint's own
    form is deliberately not used here.
    """
    if isinstance(cycles_df, dict):  # same wrapper tolerance as cell_digest
        cycles_df = cycles_df.get("cycles")
    if not isinstance(cycles_df, pd.DataFrame):
        raise TypeError("table_csv_text expects a cycles DataFrame (or a {'cycles': df} wrapper)")
    df = cycles_df.copy()
    if "cycle_number" in df.columns:
        df = df.sort_values("cycle_number")
    return df.to_csv(index=False, float_format="%.17g", lineterminator="\n")


def write_bundle_cells(cell_data: dict, out_dir: "str | Path") -> dict:
    """Write `{cell_id: cycles DataFrame}` into `out_dir` as CSV + an index.

    Returns the index dict that was written (schema, n_cells, cells:
    [{cell_id, file, sha256, cell_digest, n_rows}]). Cell order and file names
    are deterministic — sorted by cell id — so two seals of identical data
    produce byte-identical bundle contents.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries: list = []
    used: set = set()
    for i, cell_id in enumerate(sorted(cell_data)):
        df = cell_data[cell_id]
        name = cell_file_name(cell_id, i)
        if name in used:  # two ids sanitizing onto one name: keep them distinct
            name = f"cell_{i:04d}.csv"
        used.add(name)

        text = table_csv_text(df)
        # write_bytes, not write_text: text mode translates "\n" to the
        # platform's line separator, which would make the file on disk stop
        # being the bytes the digest below is computed over (and would make the
        # same bundle differ between Windows and Linux).
        (out / name).write_bytes(text.encode("utf-8"))
        entries.append({
            "cell_id": str(cell_id),
            "file": name,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            # The fingerprint, in the form every report and bundle records —
            # NOT the file's hash, which is the line above. Both are here so a
            # reviewer never has to guess which one they are comparing.
            "cell_digest": cell_digest(df),
            "n_rows": int(len(df)),
        })

    index = {
        "schema": BUNDLE_CELLS_SCHEMA,
        "schema_version": BUNDLE_CELLS_SCHEMA_VERSION,
        "n_cells": len(entries),
        "cells": entries,
    }
    (out / CELLS_INDEX_NAME).write_bytes(
        (json.dumps(index, indent=2) + "\n").encode("utf-8")
    )
    return index


def read_bundle_cell_file(path: "str | Path", *, expected_sha256: "str | None" = None) -> pd.DataFrame:
    """Read one CSV from a bundle, checking its bytes against the index first.

    A bundle may have been tampered with between publication and verification,
    so the recorded digest is checked BEFORE the table is used — a verifier
    that hashes the data after scoring it has already scored the wrong data.
    """
    p = Path(path)
    raw = p.read_bytes()
    if expected_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"{p.name} does not match the digest its own index records "
                f"(recorded {expected_sha256[:16]}…, found {actual[:16]}…) — the "
                "cell's data was changed after this bundle was sealed."
            )
    # float_precision="round_trip" is NOT optional here. pandas' default
    # converter is lossy: reading back the 17-significant-digit table above
    # yields values 1 ULP off the sealed ones (measured on pandas 3.0.3 —
    # 0.050050000000000004 reads back as 0.05005), and a fit on those flips
    # GBRT splits: a recompute of an embedded bundle landed 6e-5 away from its
    # own reported R², sixty times the replication tolerance, on data that is
    # byte-identical. Round-trip parsing is what makes "same bytes, same
    # number" true at the precision this check claims.
    return pd.read_csv(io.StringIO(raw.decode("utf-8")), float_precision="round_trip")


def load_bundle_cells(cells_dir: str = "cells", bundle_dir: "str | Path | None" = None) -> dict:
    """Load the raw cycles a bundle carries — the bundle-relative data path.

    `cells_dir` is relative to `bundle_dir` unless it is absolute.
    `bundle_dir` is supplied by the verifier (batlab.validation.replication
    injects its own bundle path for loaders that declare this parameter), so a
    bundle verifies from wherever it was unzipped; pass it explicitly when
    calling this directly.

    Raises ValueError, naming the cell and the reason, when the index is
    missing, records a different schema, points outside its own directory, or
    disagrees with any file's bytes. Returns {cell_id: cycles DataFrame} at
    full precision — the frames are meant to be scored, not merely inspected.

    Dtypes are re-inferred by the reader (a whole-number float column returns
    as int64) and deliberately NOT cast back to what the writer held: an
    astype() round trip on a bool column that contained a NaN turns that NaN
    into True, which is real corruption traded for a cosmetic difference.
    Values — and therefore digests, fingerprints and fits — are identical.
    """
    base = Path(bundle_dir) if bundle_dir is not None else Path(".")
    directory = Path(cells_dir)
    if not directory.is_absolute():
        directory = base / directory
    if not directory.is_dir():
        raise ValueError(
            f"no embedded cycles at {directory} — this bundle does not carry its "
            "own data (sealed with embed_data=False), so verification needs "
            "whatever loader the bundle records instead."
        )

    index_path = directory / CELLS_INDEX_NAME
    if not index_path.exists():
        raise ValueError(f"{index_path} is missing — an embedded-cells bundle is unreadable without its index")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{index_path} is not readable JSON: {exc}") from exc
    if index.get("schema") != BUNDLE_CELLS_SCHEMA:
        raise ValueError(
            f"{index_path} declares schema {index.get('schema')!r}, not {BUNDLE_CELLS_SCHEMA!r}"
        )

    cells: dict = {}
    for entry in index.get("cells") or []:
        cell_id = str(entry.get("cell_id", ""))
        name = str(entry.get("file", ""))
        # The traversal check: a tampered index must not be able to point the
        # verifier at a file outside this directory.
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(
                f"the index entry for cell {cell_id!r} names {name!r}, which is not a "
                "plain file name inside the bundle — refusing to read it"
            )
        path = directory / name
        if not path.is_file():
            raise ValueError(f"the index names {name} for cell {cell_id!r}, but that file is missing")
        cells[cell_id] = read_bundle_cell_file(path, expected_sha256=entry.get("sha256"))
    if not cells:
        raise ValueError(f"{index_path} lists no cells")
    return cells
