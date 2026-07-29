"""Structural guard against the "hand-rolled dataset classification" bug class.

This exact bug — a hand-rolled is_nasa/startswith("S-")-style chain that
silently misses a branch for a dataset added later — has caused real,
shipped mislabeling bugs multiple times across this project's history
(Severson cells shown as "Synthetic"/"Uploaded" in Fleet/Settings/Health/
Compliance/EOL Economics; see docs/history.md and src/chemistry_profiles.py's
ChemistryProfile docstring). It was even independently re-fixed once before
for Severson in app/utils.py's provenance helpers, and the same gap silently
reopened once Oxford was added later — proof that fixing instances one at a
time doesn't stick without a structural guard.

ChemistryProfile.for_cell(cell_id) (src/chemistry_profiles.py) is the one
canonical classifier. This test scans app/main.py, app/_pages/*.py,
app/utils.py, and every src/*.py module (its own implementation excepted)
for the patterns that keep causing this and fails if any reappear. The
same bug class was found and fixed (2026-07-29) not just in page-level
display code but in app/main.py's per-source dict splitting, the FastAPI
backend (src/api.py), the Copilot's bundle selection (src/battery_copilot.py
-- where it picked the wrong trained model for real Severson cells, not
just a display label), the knowledge graph (src/knowledge_graph.py), and
trajectory-memory failure-signature matching (src/trajectory_memory.py,
where it risked comparing an uploaded/Oxford cell's degradation curve
against synthetic failure signatures -- the exact cross-chemistry mismatch
this module's own docstring says it exists to prevent).
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from chemistry_profiles import ChemistryProfile

REPO_ROOT = pathlib.Path(__file__).parent.parent
PAGES_DIR = REPO_ROOT / "app" / "_pages"
UTILS_FILE = REPO_ROOT / "app" / "utils.py"
MAIN_FILE = REPO_ROOT / "app" / "main.py"
SRC_DIR = REPO_ROOT / "src"
# The canonical implementation itself necessarily contains the real prefix
# checks -- everything else must delegate to it instead of reimplementing.
CANONICAL_FILE = SRC_DIR / "chemistry_profiles.py"

# Substrings that indicate a hand-rolled dataset-membership/prefix check
# bypassing ChemistryProfile.for_cell(). Intentionally excludes plain
# "NASA_CELL_IDS" (the constant is still legitimately defined and iterated
# in app/utils.py and app/main.py) — only the classification-branch forms
# are forbidden.
FORBIDDEN_PATTERNS = [
    re.compile(r"\bin NASA_CELL_IDS\b"),
    re.compile(r'\.startswith\(\s*["\']S-["\']\s*\)'),
    re.compile(r'\.startswith\(\s*["\']OX-["\']\s*\)'),
]
# "cid for cid in NASA_CELL_IDS" (enumerate every known NASA id, e.g. to
# check which CSVs are present on disk) is not a per-cell classification
# branch and is legitimate — only exempted when it's exactly this shape.
_ENUMERATION_RE = re.compile(r"\bfor\s+\w+\s+in\s+NASA_CELL_IDS\b")


def _checked_files():
    yield UTILS_FILE
    yield MAIN_FILE
    yield from sorted(PAGES_DIR.glob("*.py"))
    yield from sorted(p for p in SRC_DIR.glob("*.py") if p != CANONICAL_FILE)


def test_no_hand_rolled_source_classification_in_app_pages():
    offenders = []
    for path in _checked_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _ENUMERATION_RE.search(line):
                continue
            if any(p.search(line) for p in FORBIDDEN_PATTERNS):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {stripped}")

    assert not offenders, (
        "Hand-rolled dataset-classification logic found outside "
        "ChemistryProfile.for_cell() -- classify via "
        "ChemistryProfile.for_cell(cell_id).source_kind / .source_label / "
        ".provenance instead (see this test's module docstring for why):\n"
        + "\n".join(offenders)
    )


def test_for_cell_classifies_every_known_dataset_prefix():
    cases = {
        "B0005":            ("nasa", "NASA", "measured"),
        "S-1":              ("severson", "Severson", "measured"),
        "OX-3":             ("oxford", "Oxford", "measured"),
        "Cell_01":          ("synth", "Synthetic", "synthetic"),
        "OxBat_02":         ("synth", "Synthetic", "synthetic"),
        "my-uploaded-cell": ("upload", "Uploaded", "measured"),
    }
    for cell_id, (source_kind, source_label, provenance) in cases.items():
        profile = ChemistryProfile.for_cell(cell_id)
        assert profile.source_kind == source_kind, cell_id
        assert profile.source_label == source_label, cell_id
        assert profile.provenance == provenance, cell_id
