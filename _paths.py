"""
Centralized sys.path setup for the Battery Intelligence Platform.

This module ensures that src/ (core modules like db, api, data_loader,
chemistry_profiles, etc.) and app/ (UI helpers, pages) are importable
as bare names from anywhere in the project.

Usage — pick ONE of these entry points, then all other modules resolve:

    # At the top of your module (before any src/app imports):
    from _paths import setup; setup()

    # Or, if _paths is not on your path yet (e.g. in tests/conftest.py):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from _paths import setup; setup()

The three directories added are:
  - <project_root>/src/   — core platform modules (bare imports: from db import ...)
  - <project_root>/app/   — Streamlit app helpers  (bare imports: from utils import ...)
  - <project_root>/       — project root            (for _paths itself, batlab, etc.)
"""

from __future__ import annotations

import os
import sys

# Absolute path to the project root.
# Walk up from this file's directory until we find pyproject.toml or .git,
# so the app works even when launched from a subdirectory (e.g.
# ``streamlit run app/main.py`` from anywhere).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT != os.path.dirname(_ROOT):  # stop at filesystem root
    if os.path.isfile(os.path.join(_ROOT, "pyproject.toml")) or os.path.isdir(os.path.join(_ROOT, ".git")):
        break
    _ROOT = os.path.dirname(_ROOT)
_SRC = os.path.join(_ROOT, "src")
_APP = os.path.join(_ROOT, "app")
_SCRIPTS = os.path.join(_ROOT, "scripts")

# Guard: only add each path once, even if setup() is called multiple times.
_SETUP_DONE = False


def setup() -> None:
    """Add project root, src/, and app/ to sys.path (idempotent)."""
    global _SETUP_DONE  # noqa: PLW0603
    if _SETUP_DONE:
        return
    for _p in (_ROOT, _SRC, _APP, _SCRIPTS):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    _SETUP_DONE = True


# Call setup() on import so that `import _paths` alone is enough for
# the most common use case.  Modules that need src/ or app/ on the path
# just need `import _paths` (or `from _paths import setup`) at the top,
# and the path entries are guaranteed to be present before any subsequent
# import statement in that file executes.
setup()
