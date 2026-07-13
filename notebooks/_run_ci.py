"""
Notebook CI runner.

Executes every notebooks/*.ipynb top-to-bottom and writes outputs back
in place, so a broken notebook fails the build the same way a broken test
would. Cells tagged "skip-ci" (Jupyter cell metadata -> tags) are removed
before execution, not just skipped-with-a-warning — for a cell that needs
a large one-time download this repo doesn't commit (e.g. a full raw
dataset zip), so CI never attempts it.

Deliberately not `jupyter nbconvert --execute` with
--TagRemovePreprocessor.remove_cell_tags: that CLI combination does NOT
actually strip tagged cells before execution in the nbconvert/nbclient
versions this was verified against (the preprocessor isn't wired into the
"notebook" exporter's default execution chain) -- confirmed by testing
before writing this script. This uses nbclient directly instead, where
the strip-then-execute order is explicit and verified.

Usage: python notebooks/_run_ci.py
Exit code is non-zero if any notebook raises during execution.
"""

import pathlib
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS_DIR = pathlib.Path(__file__).parent
SKIP_TAG = "skip-ci"


def run_notebook(path: pathlib.Path) -> bool:
    nb = nbformat.read(path, as_version=4)

    before = len(nb["cells"])
    nb["cells"] = [
        c for c in nb["cells"]
        if SKIP_TAG not in c.get("metadata", {}).get("tags", [])
    ]
    skipped = before - len(nb["cells"])
    if skipped:
        print(f"  ({skipped} cell(s) tagged '{SKIP_TAG}' removed before execution)")

    # Run with the notebook's own directory as cwd, matching how a user
    # opening it in Jupyter would run it (relative paths like
    # "sample_data/my_lab_cell.csv" resolve the same way either way).
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"  FAILED: {exc}")
        return False

    nbformat.write(nb, path)
    return True


def main() -> int:
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.")
        return 1

    ok = True
    for nb_path in notebooks:
        print(f"=== {nb_path.name} ===")
        if not run_notebook(nb_path):
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
