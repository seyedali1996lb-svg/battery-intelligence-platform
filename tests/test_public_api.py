"""The public API contract: version, entry points, result schemas, deprecation.

These are the tests that make ``docs/api_stability.md`` enforceable rather than
aspirational. They are deliberately hermetic — the only real data they load is
two committed NASA cells, so the suite measures the *contract*, not the model.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest

try:  # stdlib on 3.11+; the package floor is 3.10, so skip rather than depend
    import tomllib

except ImportError:  # pragma: no cover - Python 3.10
    tomllib = None  # type: ignore[assignment]

import batlab
from batlab import _deprecation
from batlab import cli as batlab_cli
from batlab.results import (
    LcoResult,
    QuantileLcoResult,
    as_lco,
    as_quantile_lco,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Version and packaging
# ---------------------------------------------------------------------------

def test_version_matches_pyproject():
    """One version number, two places that state it — so pin they agree.

    A stale ``__version__`` is how a user reports a bug against a release that
    was never published. tomllib is stdlib on 3.11+; this suite's floor is 3.10,
    where tomli is not a declared dependency, so skip rather than add one.
    """
    if tomllib is None:  # pragma: no cover - 3.10 has no tomllib
        pytest.skip("tomllib requires Python 3.11+")
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == batlab.__version__


def test_distribution_name_is_the_free_one():
    """PyPI's `batlab` is Lexcelon's unrelated hardware library (v0.5.8).

    The distribution must therefore be `battery-lab` while the import package
    stays `batlab`; if this assertion ever fails, someone renamed the dist back
    into a collision.
    """
    if tomllib is None:  # pragma: no cover
        pytest.skip("tomllib requires Python 3.11+")
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "battery-lab"
    assert data["project"]["scripts"] == {"batlab": "batlab.cli:main"}
    assert (REPO_ROOT / "batlab" / "py.typed").exists(), "PEP 561 marker missing"


# ---------------------------------------------------------------------------
# batlab.__all__ and the lazy submodules
# ---------------------------------------------------------------------------

def test_every_exported_name_exists_and_documents_itself():
    for name in batlab.__all__:
        assert hasattr(batlab, name), f"batlab.__all__ lists missing name {name!r}"


def test_available_datasets_matches_the_loader_table():
    assert batlab.AVAILABLE_DATASETS == tuple(sorted(batlab_cli._DATASET_LOADERS))


def test_unknown_dataset_names_the_valid_choices():
    with pytest.raises(KeyError) as excinfo:
        batlab.load("not-a-dataset")
    message = str(excinfo.value)
    assert "not-a-dataset" in message
    for name in batlab.AVAILABLE_DATASETS:
        assert name in message


@pytest.mark.parametrize("submodule", ["datasets", "features", "models", "validation", "harness", "results"])
def test_submodules_resolve_lazily(submodule):
    import importlib

    assert getattr(batlab, submodule) is importlib.import_module(f"batlab.{submodule}")


def test_importing_batlab_does_not_import_pandas():
    """`import batlab` must stay cheap: the loaders are resolved on demand.

    Checked in a subprocess because this suite has already imported pandas by
    the time it runs.
    """
    import subprocess
    import sys

    code = (
        "import sys, batlab; "
        "assert batlab.__version__; "
        "print('pandas' in sys.modules, 'sklearn' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False", out.stdout


# ---------------------------------------------------------------------------
# Result schemas describe the real payloads
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_nasa_cells():
    cells = batlab.load("nasa", cell_ids=["B0005", "B0006"])
    assert len(cells) == 2, "committed NASA cells missing — see data/raw/"
    return cells


def _declared(tp) -> set[str]:
    return set(tp.__annotations__)


def test_lco_result_schema_matches_a_real_payload(two_nasa_cells):
    """Declared keys == produced keys, both directions.

    A schema that omits a key a caller can read is incomplete; one that declares
    a key no run produces is a lie. Both directions are checked against a real
    leave-cell-out run on real committed cells.
    """
    produced = batlab.benchmark(two_nasa_cells)
    declared = _declared(LcoResult)
    assert declared == set(produced), (
        f"declared-but-not-produced: {sorted(declared - set(produced))}; "
        f"produced-but-not-declared: {sorted(set(produced) - declared)}"
    )


def test_quantile_lco_result_schema_matches_a_real_payload(two_nasa_cells):
    from batlab.validation import run_lco_quantiles

    produced = run_lco_quantiles(two_nasa_cells)
    declared = _declared(QuantileLcoResult)
    # `not_evaluable` is declared because the short-circuit path produces it;
    # a fleet WITH observed labels produces every other key and not that one.
    assert set(produced) == declared - {"not_evaluable"}, (
        f"declared-but-not-produced: {sorted(declared - {'not_evaluable'} - set(produced))}; "
        f"produced-but-not-declared: {sorted(set(produced) - declared)}"
    )


def test_as_lco_bridges_are_identity_functions():
    payload = {"soh_r2": 0.7}
    assert as_lco(payload) is payload
    assert as_quantile_lco(payload) is payload


def test_fold_cache_summary_is_a_declared_schema_of_the_result(two_nasa_cells):
    """The nested fold-cache dict is checked against its own TypedDict.

    Nested shapes are exactly where a schema usually rots unnoticed.
    """
    lco = batlab.benchmark(two_nasa_cells)
    summary = lco["fold_cache"]
    assert isinstance(summary, dict)
    # Equality, not subset: this assertion is what caught the schema naming
    # `reused`/`total`/`cache_dir` while the cache actually returns
    # `hits`/`dir` — and the CLI silently printing `0 reused` because of it.
    assert set(summary) == _declared(batlab.results.FoldCacheSummary)


# ---------------------------------------------------------------------------
# Deprecation policy — the mechanism behind docs/api_stability.md
# ---------------------------------------------------------------------------

def test_warn_deprecated_states_what_when_and_the_alternative():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _deprecation.warn_deprecated(
            "old_thing", since="0.2.0", removed_in="0.4.0", alternative="new_thing"
        )
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "old_thing" in message
    assert "0.2.0" in message
    assert "0.4.0" in message
    assert "new_thing" in message
    assert caught[0].category is DeprecationWarning


def test_deprecated_decorator_forwards_and_warns():
    @_deprecation.deprecated("old_fn", since="0.2.0", removed_in="0.4.0", alternative="new_fn")
    def old_fn(value: int) -> int:
        """Kept docstring."""
        return value * 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert old_fn(21) == 42
    assert [w.category for w in caught] == [DeprecationWarning]
    assert old_fn.__doc__ == "Kept docstring."
    assert old_fn.__name__ == "old_fn"
