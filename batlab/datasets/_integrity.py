"""
Checksum verification for auto-downloaded raw dataset archives.

nasa.py, severson.py, and oxford.py each auto-download their source archive
from a third-party host on first use and cache it under data/raw/. Without
an integrity check, a corrupted transfer or a substituted/tampered file at
the source would be parsed silently -- batlab.datasets.schema.validate_schema()
only checks the *shape* of the resulting DataFrame (required columns,
monotonic cycle numbers, etc.), not that the underlying bytes are the real
dataset, so bad input can still produce a schema-valid DataFrame full of
garbage.

Each loader pins the SHA-256 of the exact file its download URL served when
last verified (see each loader's own EXPECTED_SHA256 comment for when). If a
legitimate upstream re-upload ever changes those bytes, verify_sha256() will
raise -- don't silence the error, manually confirm the new download is
legitimate (not a MITM or a compromised host) before updating the pinned
hash.
"""

from __future__ import annotations

import hashlib
import os


def sha256_of(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file on disk, streaming in
    chunk_size-byte reads so multi-hundred-MB archives don't need to fit
    in memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str | os.PathLike, expected_hex: str | None, label: str) -> None:
    """
    Raise RuntimeError if the file at `path` doesn't hash to `expected_hex`.

    Called right after a download completes, and again before parsing any
    already-cached archive found on disk (covers both a corrupted transfer
    and a file that was replaced after the fact). `expected_hex=None` skips
    verification with a warning printed to stdout -- used only for archives
    this project hasn't pinned a hash for yet.
    """
    if expected_hex is None:
        print(
            f"  [warn] no pinned checksum for {label} — integrity not verified. "
            f"({path})"
        )
        return
    actual = sha256_of(path)
    if actual != expected_hex:
        raise RuntimeError(
            f"Checksum mismatch for {label}:\n"
            f"  file:     {path}\n"
            f"  expected: sha256 {expected_hex}\n"
            f"  actual:   sha256 {actual}\n"
            "The file may be corrupted (partial/interrupted download) or the "
            "upstream source may have changed. Delete this file and retry the "
            "download; if the mismatch persists, verify the new file manually "
            "before updating the pinned hash in this loader."
        )
