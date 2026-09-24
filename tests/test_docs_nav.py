"""Docs must stay reachable and documented — enforced, not hoped for.

Two failure modes this file retires:

* A page exists but is in no ``mkdocs.yml`` nav entry, so it is published and
  unreachable (and ``--strict`` never notices, because nothing links it).
* A public function ships without a docstring, so its MkDocs API-reference entry
  renders as an empty heading.

Both are cheap to check with no MkDocs install, which is why they run in the main
suite rather than in the docs job that has the toolchain.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# Module -> its public surface to check for docstrings. `batlab.harness` resolves
# its names lazily (module __getattr__), which getattr() follows happily.
_PUBLIC_MODULES = (
    "batlab",
    "batlab.cli",
    "batlab.datasets",
    "batlab.features",
    "batlab.models",
    "batlab.results",
    "batlab.validation",
    "batlab.harness",
)


def _nav_entries() -> list[str]:
    """Every file path referenced by mkdocs.yml's nav, in order."""
    text = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    nav_match = re.search(r"^nav:\n(.*?)(?=^\S)", text, flags=re.MULTILINE | re.DOTALL)
    assert nav_match, "mkdocs.yml has no nav: block"
    return re.findall(r":\s*([^\s:]+\.(?:md|ipynb))\s*$", nav_match.group(1), flags=re.MULTILINE)


def test_nav_is_not_empty():
    assert len(_nav_entries()) > 10, "nav parsing broke — nothing was found"


@pytest.mark.parametrize("entry", _nav_entries())
def test_every_nav_entry_exists(entry):
    """Every nav target exists — against what a fresh checkout actually has.

    ``docs/notebooks/`` is build output: ``docs/hooks.py`` copies the committed
    top-level ``notebooks/`` there during ``mkdocs build``, and it is
    gitignored to keep the top-level directory the single source of truth. A
    fresh checkout — which is what CI's lint job tests, before any build has
    run — therefore has no copy under ``docs/``, so notebook entries are
    asserted against their committed source; the docs job's
    ``mkdocs build --strict`` is what still proves the hook generates the copy
    that renders.
    """
    root = REPO_ROOT if entry.startswith("notebooks/") else DOCS
    target = root / entry
    assert target.exists(), f"mkdocs.yml nav points at missing {target.relative_to(REPO_ROOT)}"


def test_every_top_level_docs_page_is_in_the_nav():
    """A new top-level page that nobody linked is a page nobody can find."""
    listed = {entry for entry in _nav_entries() if "/" not in entry}
    actual = {p.name for p in DOCS.glob("*.md")}
    missing = sorted(actual - listed)
    assert not missing, f"docs/*.md not reachable from mkdocs.yml nav: {missing}"


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_public_names_carry_docstrings(module_name):
    """Every public function/class is documented; constants are exempt.

    Constants are exempt on purpose: a module-level ``RUL_RELIABLE_FLOOR = 0.3``
    is documented by the prose that introduces it and by its own name, and
    inventing a docstring for it would document nothing.
    """
    module = importlib.import_module(module_name)
    undocumented: list[str] = []
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name)
        if not (inspect.isroutine(obj) or inspect.isclass(obj)):
            continue
        if not (getattr(obj, "__doc__") or "").strip():
            undocumented.append(name)
    assert not undocumented, f"{module_name}: public names without docstrings: {undocumented}"


def test_performance_page_documents_the_fold_cache():
    """The page README/link targets must exist and cover the cache's knobs."""
    page = (DOCS / "performance.md").read_text(encoding="utf-8")
    for expected in ("BATLAB_LCO_CACHE", "BATLAB_BOOT_LAYERS", "BATLAB_LCO_CACHE_DIR"):
        assert expected in page, f"docs/performance.md does not mention {expected}"
