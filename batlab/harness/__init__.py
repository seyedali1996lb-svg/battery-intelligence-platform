"""
batlab.harness — grade any battery-degradation forecaster by this platform's
validation rules.

    from batlab.harness import validate_forecaster

    report = validate_forecaster(cells, model=my_model)
    print(report["verdict"]["summary"])
    for claim in report["verdict"]["claims"]:
        print(" ", claim)
    for withheld in report["verdict"]["withheld"]:
        print("  withheld:", withheld)

Public surface
--------------
    validate_forecaster(cell_data, model=None, ...)   the whole harness, one call
    seal_bundle(report, cell_data, out_dir, ...)      a verifiable sealed bundle
    format_report(report)                             terminal text of a report
    Forecaster / IntervalForecaster                   the two-method model protocol
    SklearnForecaster / SklearnIntervalForecaster     sklearn-compatible estimators
    CallableForecaster / torch_forecaster             anything else (incl. PyTorch)
    as_factory(model)                                 normalize any of the above

Command line
------------
    python -m batlab.harness --loader batlab.datasets.nasa:load_nasa_cells

Exports are resolved lazily (module `__getattr__`, the same pattern the
platform's `src/__init__.py` uses) because batlab.validation imports the
forecaster protocol from this package: importing it eagerly here would make
`import batlab.validation.lco` re-enter a half-initialized module.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CallableForecaster",
    "DEFAULT_SPLITS",
    "Forecaster",
    "ForecasterFactory",
    "ForecasterLike",
    "HARNESS_SCHEMA",
    "HARNESS_SCHEMA_VERSION",
    "IntervalForecaster",
    "SklearnForecaster",
    "SklearnIntervalForecaster",
    "as_factory",
    "default_forecaster",
    "default_interval_forecaster",
    "fit_forecaster",
    "forecaster_identity",
    "format_report",
    "has_predict_interval",
    "seal_bundle",
    "torch_forecaster",
    "validate_forecaster",
]

_FROM_FORECASTER = (
    "CallableForecaster",
    "Forecaster",
    "ForecasterFactory",
    "ForecasterLike",
    "IntervalForecaster",
    "SklearnForecaster",
    "SklearnIntervalForecaster",
    "as_factory",
    "default_forecaster",
    "default_interval_forecaster",
    "fit_forecaster",
    "forecaster_identity",
    "has_predict_interval",
    "torch_forecaster",
)

_FROM_HARNESS = (
    "DEFAULT_SPLITS",
    "HARNESS_SCHEMA",
    "HARNESS_SCHEMA_VERSION",
    "format_report",
    "seal_bundle",
    "validate_forecaster",
)


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (see the module docstring)."""
    if name in _FROM_FORECASTER:
        from batlab.harness import forecaster

        return getattr(forecaster, name)
    if name in _FROM_HARNESS:
        from batlab.harness import harness

        return getattr(harness, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> "list[str]":
    return sorted(__all__)
