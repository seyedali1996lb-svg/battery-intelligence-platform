"""Where a dataset loader keeps its raw files.

Before this module every loader hard-coded ``<module>../../data/raw/<name>``.
That is correct in a checkout and *wrong* in a wheel: ``pip install
battery-lab`` puts the package in ``site-packages``, so the same expression
resolves to ``site-packages/data/raw`` — a directory that is not the user's, may
be read-only, and that a loader would either fail to write into or silently
pollute. A library that only works when cloned is the thing this release is
fixing, so the resolution lives in one place:

1. **``BATLAB_DATA_DIR``** — an explicit override, first because a user who keeps
   datasets on another volume should not have to fight the default, and because
   it is the one knob a test needs to stay hermetic (no test writes to, or reads
   from, a developer's real dataset cache).
2. **``<checkout>/data/raw``** — recognised by ``pyproject.toml`` sitting beside
   it, so a clone finds the committed CSVs exactly where they have always been
   and nothing about local development changes.
3. **A per-user cache directory** — ``%LOCALAPPDATA%/batlab/data/raw`` on
   Windows, ``$XDG_CACHE_HOME/batlab/data/raw`` or ``~/.cache/batlab/data/raw``
   elsewhere — which is writable, is the user's, and survives reinstalls.
"""

from __future__ import annotations

import os
import pathlib
import sys

__all__ = ["DATA_DIR_ENV", "raw_data_dir"]

DATA_DIR_ENV = "BATLAB_DATA_DIR"

# batlab/datasets/_paths.py -> the checkout root two levels up. Used only to
# RECOGNISE a source checkout (pyproject.toml beside data/); a wheel resolves
# through it to site-packages and simply finds no pyproject, which is the signal
# to fall through to the user cache.
_CHECKOUT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _user_data_root() -> pathlib.Path:
    """The per-user cache root for downloaded datasets (see module docstring)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return pathlib.Path(base) / "batlab" / "data"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return pathlib.Path(xdg) / "batlab" / "data"
    return pathlib.Path.home() / ".cache" / "batlab" / "data"


def raw_data_dir(*parts: str) -> pathlib.Path:
    """Return the raw-data directory for a loader, plus any sub-path.

    ``raw_data_dir("severson")`` is the Severson loader's cache directory.
    Environment lookups happen per call, so an override takes effect without
    re-importing a loader.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        base = pathlib.Path(override)
    elif (_CHECKOUT_ROOT / "pyproject.toml").exists():
        base = _CHECKOUT_ROOT / "data" / "raw"
    else:
        base = _user_data_root() / "raw"
    return base.joinpath(*parts) if parts else base
