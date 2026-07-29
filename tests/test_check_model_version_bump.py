"""
Unit tests for scripts/check_model_version_bump.py's check_needs_bump() --
a real self-test that this CI warning actually catches the case it exists
to catch, rather than trusting the logic works because it "looks right" in
a YAML file (this project's history includes at least one CI check that
silently never ran at all -- pytest itself, for a long stretch before
someone happened to add a real test step). check_needs_bump() is pure
logic (no subprocess, no git) so it's directly testable without a real
repo or commit history.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from check_model_version_bump import check_needs_bump, MODEL_FILE, BUNDLE_CACHE_FILE


def test_flags_when_model_changed_without_bundle_cache():
    assert check_needs_bump([MODEL_FILE]) is True


def test_does_not_flag_when_both_changed_together():
    assert check_needs_bump([MODEL_FILE, BUNDLE_CACHE_FILE]) is False


def test_does_not_flag_when_only_bundle_cache_changed():
    assert check_needs_bump([BUNDLE_CACHE_FILE]) is False


def test_does_not_flag_when_neither_changed():
    assert check_needs_bump(["app/_pages/health.py", "README.md"]) is False


def test_does_not_flag_on_empty_diff():
    assert check_needs_bump([]) is False


def test_ignores_unrelated_files_changed_alongside_model_file():
    """Regression guard for a plausible off-by-something bug: the check
    must key on exact file identity, not just "something in batlab/models/
    changed" or "something under src/ changed"."""
    assert check_needs_bump([MODEL_FILE, "batlab/models/__init__.py", "src/db.py"]) is True
    assert check_needs_bump(["batlab/models/__init__.py", BUNDLE_CACHE_FILE]) is False
