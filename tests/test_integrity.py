"""Unit tests for batlab.datasets._integrity and its wiring into the
nasa.py/severson.py/oxford.py auto-downloaders.

Regression coverage for the "no checksum verification on any
auto-downloaded dataset" finding: a corrupted or tampered download used to
be parsed silently, since batlab.datasets.schema.validate_schema() only
checks DataFrame *shape*, not that the underlying bytes are the real
dataset — a schema-valid DataFrame could still be built from garbage.
"""

import hashlib
import re

import pytest

from batlab.datasets._integrity import sha256_of, verify_sha256

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"some test content" * 1000)
    assert sha256_of(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_verify_sha256_passes_for_matching_hash(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"content")
    verify_sha256(p, sha256_of(p), "test file")  # must not raise


def test_verify_sha256_raises_for_mismatched_hash(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"content")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_sha256(p, "0" * 64, "test file")


def test_verify_sha256_skips_with_warning_when_no_expected_hash(tmp_path, capsys):
    """expected_hex=None is used for archives this project hasn't pinned a
    hash for yet -- must not raise, but must be visibly non-silent."""
    p = tmp_path / "f.bin"
    p.write_bytes(b"content")
    verify_sha256(p, None, "test file")  # must not raise
    assert "no pinned checksum" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Every loader's pinned hash is well-formed -- catches a typo or truncated
# copy-paste at test-collection time, not just at the next real download.
# ---------------------------------------------------------------------------

def test_nasa_expected_sha256_is_well_formed():
    from batlab.datasets.nasa import EXPECTED_SHA256
    assert _SHA256_RE.match(EXPECTED_SHA256)


def test_severson_expected_sha256_is_well_formed():
    from batlab.datasets.severson import _EXPECTED_SHA256
    assert _SHA256_RE.match(_EXPECTED_SHA256)


def test_oxford_expected_sha256_covers_every_group():
    from batlab.datasets.oxford import _EXPECTED_SHA256, _GROUP_URLS
    assert set(_EXPECTED_SHA256) == set(_GROUP_URLS)
    for group, digest in _EXPECTED_SHA256.items():
        assert _SHA256_RE.match(digest), f"group {group} hash malformed: {digest!r}"


# ---------------------------------------------------------------------------
# Wiring: a cached-but-tampered raw file must be rejected before parsing --
# not only a freshly-downloaded one. These exercise the real loader
# functions against a planted fake file, so no network access is needed
# (the fake content deliberately can't match the real pinned hash).
# ---------------------------------------------------------------------------

def test_download_nasa_zip_rejects_tampered_cached_file(tmp_path):
    from batlab.datasets.nasa import download_nasa_zip
    (tmp_path / "nasa_battery.zip").write_bytes(b"not the real dataset")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download_nasa_zip(str(tmp_path))


def test_oxford_download_group_zip_rejects_tampered_cached_file(tmp_path, monkeypatch):
    from batlab.datasets import oxford
    monkeypatch.setattr(oxford, "_RAW_DIR", tmp_path)
    (tmp_path / "Group_2.zip").write_bytes(b"not the real dataset")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        oxford._download_group_zip(2)


def test_severson_download_and_cache_rejects_tampered_cached_file(tmp_path, monkeypatch):
    from batlab.datasets import severson
    monkeypatch.setattr(severson, "_RAW_DIR", tmp_path)
    (tmp_path / "batch1.mat").write_bytes(b"not the real dataset")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        severson._download_and_cache()
