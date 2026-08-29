"""
Allow running the Streamlit app from any directory::

    python -m app              # from anywhere
    streamlit run app/main.py  # from anywhere

This file ensures the repo root is on sys.path (via _paths.py) before
launching Streamlit, so relative imports and data-file paths resolve
correctly regardless of CWD.
"""

from __future__ import annotations

import os
import sys

# Ensure the repo root is importable (for _paths.py itself).
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import _paths  # noqa: F401 — adds src/, app/, scripts/ to sys.path

# Now launch Streamlit pointing at main.py.
_main = os.path.join(os.path.dirname(__file__), "main.py")
os.execvp("streamlit", ["streamlit", "run", _main])
