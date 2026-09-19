"""Where loaders put raw files — the difference between a clone and a wheel.

Every loader used to compute ``<module>/../../data/raw/<name>``. In a checkout
that is the committed data; in a pip-installed wheel it is ``site-packages/...``,
which is not the user's directory, may be read-only, and pollutes an install
that is supposed to be replaceable. These tests pin the resolution order that
replaced it.
"""

from __future__ import annotations

import pathlib

import pytest

from batlab.datasets import _paths
from batlab.datasets._paths import raw_data_dir

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_override_wins_and_is_read_per_call(monkeypatch, tmp_path):
    """`BATLAB_DATA_DIR` is honoured, and takes effect without a re-import."""
    monkeypatch.setenv(_paths.DATA_DIR_ENV, str(tmp_path / "elsewhere"))
    assert raw_data_dir() == tmp_path / "elsewhere"
    assert raw_data_dir("severson") == tmp_path / "elsewhere" / "severson"

    # Changing the override changes the answer on the next call: the lookups are
    # per-call on purpose, so a test (or a user) is not stuck with whatever the
    # environment said at import time.
    monkeypatch.setenv(_paths.DATA_DIR_ENV, str(tmp_path / "other"))
    assert raw_data_dir("nasa") == tmp_path / "other" / "nasa"


def test_checkout_layout_is_used_when_a_pyproject_sits_beside_it(monkeypatch):
    monkeypatch.delenv(_paths.DATA_DIR_ENV, raising=False)
    assert raw_data_dir() == REPO_ROOT / "data" / "raw"
    # The committed fleet must stay reachable from a checkout, unchanged.
    assert raw_data_dir("severson").is_dir(), "committed Severson CSVs missing"


def test_installed_layout_falls_back_to_a_user_cache(monkeypatch, tmp_path):
    """Simulate the wheel case: no pyproject beside the package.

    `_CHECKOUT_ROOT` is the only thing that distinguishes an installed
    distribution from a source checkout, so pointing it at an empty directory is
    exactly the wheel situation — and the answer must NOT be inside it.
    """
    monkeypatch.delenv(_paths.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(_paths, "_CHECKOUT_ROOT", tmp_path / "site-packages")
    resolved = raw_data_dir("nasa")
    assert tmp_path not in resolved.parents
    assert resolved.name == "nasa"
    # ... and it is somewhere in the user's own cache/home, not a temp dir.
    assert resolved.is_relative_to(pathlib.Path.home())


def test_every_loader_resolves_through_the_shared_helper():
    """No loader may reintroduce a module-relative data path.

    A source-level guard rather than a behavioural one, because the failure mode
    is a line somebody pastes back in later; the behavioural tests above cannot
    see a loader that computes its own path for a codepath they do not exercise.
    """
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "batlab" / "datasets").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "data" in text and '"raw"' in text and "raw_data_dir" not in text:
            # A module that both reaches for data/raw and never calls the helper
            # — the shape of the bug this module removed.
            if path.name != "_paths.py" and "parent.parent.parent" in text:
                offenders.append(path.name)
    assert not offenders, f"loaders bypassing raw_data_dir(): {offenders}"
