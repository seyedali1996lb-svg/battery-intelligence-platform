"""Structural guard against reintroducing sub-AA text contrast.

An Accessibility audit (2026-07-30) found #4a5568 and #718096 -- both
originally used as "muted"/secondary text colors throughout the app's raw
HTML (st.markdown(unsafe_allow_html=True)) -- fail WCAG AA's 4.5:1 minimum
contrast ratio for normal text against every dark card background actually
used in this app (#0e1117/#1a202c/#1e2a38/#111827/#2d3748), computed via the
real WCAG relative-luminance formula: ratios ranged 1.59-2.51 (#4a5568) and
2.99-4.71 (#718096), both well under 4.5:1 in most contexts. This wasn't one
mistaken component -- both colors were the established convention, copied
into 172 occurrences across 19 files. Fixed by replacing both with #a0aec0
(5.32-8.38:1 against the same backgrounds, comfortably passing everywhere).

This test doesn't re-derive contrast ratios (that's a design decision, not
something to hardcode a threshold for); it just fails if either banned color
value creeps back into app source, the same "structural guard, not a
one-time fix" pattern already established for dataset classification (see
test_source_classification_guard.py).
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).parent.parent
APP_DIR = REPO_ROOT / "app"

BANNED_COLORS = ["#4a5568", "#718096"]
_COLOR_RE = re.compile(
    r"\bcolor:\s*(" + "|".join(re.escape(c) for c in BANNED_COLORS) + r")\b",
    re.IGNORECASE,
)


def _app_py_files():
    return sorted(p for p in APP_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_sub_aa_contrast_text_colors_in_app():
    offenders = []
    for path in _app_py_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _COLOR_RE.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Text color(s) that fail WCAG AA contrast (4.5:1) against this app's "
        "dark card backgrounds found -- use #a0aec0 (or another color "
        "verified >= 4.5:1 against #0e1117/#1a202c/#1e2a38/#111827/#2d3748) "
        "instead (see this test's module docstring for why):\n"
        + "\n".join(offenders)
    )
