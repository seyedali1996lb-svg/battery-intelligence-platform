"""
batlab — a citable, honest research library for battery degradation analysis.

Standardized dataset loaders (``batlab.datasets``), literature-cited feature
engineering (``batlab.features``), leave-cell-out-validated GBRT SOH/RUL models
(``batlab.models``), reproducible benchmark manifests and the model-agnostic
validation harness (``batlab.validation``, ``batlab.harness``).

Three entry points cover the common path — load a public fleet, run the
platform's own validation, or grade *any* forecaster by the same rules:

    import batlab

    cells = batlab.load("nasa")                  # {cell_id: DataFrame}

    lco = batlab.benchmark(cells)                # leave-cell-out metrics
    print(lco["soh_r2"], lco["rul_reliable"])

    report = batlab.validate(cells, model=my_model)   # model=None -> batlab's GBRT
    print(report["verdict"]["summary"])
    for claim in report["verdict"]["withheld"]:
        print("  withheld:", claim)

    print(batlab.cite())                         # BibTeX for the library

Distributed as the PyPI distribution ``battery-lab`` (PyPI's ``batlab`` is an
unrelated hardware library); the import package is ``batlab``.

The names of the public API are listed in ``batlab.__all__`` and the stability
contract for them is ``docs/api_stability.md``: additive changes land in minor
releases, and nothing is removed without a DeprecationWarning for at least one
minor release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batlab.cite import cite

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import DataFrame

    from batlab.harness import ForecasterLike

__version__ = "0.2.0"

# name -> "module:function". Strings, not imports: `import batlab` must not pull
# in pandas, scikit-learn or a network-touching loader just to print a version.
_LOADERS: dict[str, str] = {
    "nasa": "batlab.datasets.nasa:load_nasa_cells",
    "severson": "batlab.datasets.severson:load_severson_cells",
    "zhu2022": "batlab.datasets.zhu2022:load_zhu2022_cells",
    "oxford": "batlab.datasets.oxford:load_oxford_cells",
    "calce": "batlab.datasets.calce:load_calce_cells",
}

"""The dataset keys :func:`load` accepts, sorted."""
AVAILABLE_DATASETS: tuple[str, ...] = tuple(sorted(_LOADERS))

__all__ = [
    "AVAILABLE_DATASETS",
    "__version__",
    "benchmark",
    "cite",
    "load",
    "validate",
]

_SUBMODULES = ("datasets", "features", "harness", "models", "results", "validation")


def load(name: str, **kwargs: Any) -> "dict[str, DataFrame]":
    """Load one of the five supported public datasets.

    ``name`` is one of :data:`AVAILABLE_DATASETS`; anything else raises
    ``KeyError`` naming the valid choices rather than importing the wrong
    loader. ``kwargs`` go straight to that dataset's loader (``data_dir``,
    ``cell_ids``, ``force_download`` where the loader supports them).

    Returns ``{cell_id: DataFrame}`` in the one standardized schema — see
    ``batlab.datasets.schema``. A loader whose raw files are absent may download
    them (checksum-verified) or return ``{}`` where the source requires manual
    acquisition; ``batlab datasets --load`` reports which is which.
    """
    import importlib

    spec = _LOADERS.get(name)
    if spec is None:
        raise KeyError(
            f"unknown dataset {name!r}; available: {', '.join(AVAILABLE_DATASETS)}"
        )
    module_name, func_name = spec.split(":", 1)
    loader = getattr(importlib.import_module(module_name), func_name)
    cells: dict[str, DataFrame] = loader(**kwargs)
    return cells


def benchmark(cells: "dict[str, DataFrame]", **kwargs: Any) -> dict[str, Any]:
    """Leave-cell-out metrics for batlab's own GBRT on ``cells``.

    A one-line alias for :func:`batlab.validation.run_lco` — the same folds, the
    same observed-label-only RUL rule, the same confidence intervals. The
    returned dict's schema is :class:`batlab.results.LcoResult`; wrap it with
    :func:`batlab.results.as_lco` to type it at a call site.
    """
    from batlab.validation import run_lco

    return run_lco(cells, **kwargs)


def validate(
    cells: "dict[str, DataFrame]",
    model: "ForecasterLike | None" = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Grade a forecaster with the six harness checks.

    ``model=None`` grades this platform's own GBRT, which reproduces the
    published numbers exactly. Any sklearn estimator, PyTorch module, or
    callable is graded by the *same* rules — leakage lint, label provenance,
    leave-cell-out, interval calibration, the prospective split, and the metric
    gate — and the report names the claims the run supports and the claims it
    withholds.

    Thin alias for :func:`batlab.harness.validate_forecaster`.
    """
    from batlab.harness import validate_forecaster

    return validate_forecaster(cells, model=model, **kwargs)


def __getattr__(name: str) -> Any:
    """Resolve ``batlab.datasets`` / .features / .models / .validation / .harness
    / .results on first attribute access, so `import batlab` stays cheap while
    `batlab.validation.run_lco` still works without an explicit submodule import.
    """
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"batlab.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> "list[str]":
    return sorted(set(__all__) | set(_SUBMODULES))
