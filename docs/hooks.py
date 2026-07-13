"""
mkdocs build hook: copies notebooks/*.ipynb into docs/notebooks/ before every
build, so the four worked notebooks render via mkdocs-jupyter without a
second, divergence-prone copy living permanently in the repo — the
top-level notebooks/ directory (also used by CI, see
notebooks/_run_ci.py and notebooks/README.md) stays the single source
of truth; docs/notebooks/ is a generated build artifact, gitignored.
"""

import pathlib
import shutil

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "notebooks"
DEST = REPO_ROOT / "docs" / "notebooks"


def on_pre_build(config, **kwargs):
    DEST.mkdir(parents=True, exist_ok=True)
    for nb in SRC.glob("*.ipynb"):
        shutil.copy2(nb, DEST / nb.name)
