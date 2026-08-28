"""Leave-cell-out cross-validation, the per-cell reliability gate, quantile-interval calibration, and reproducible manifests."""

from batlab.validation.lco import RUL_RELIABLE_FLOOR, run_lco
from batlab.validation.manifest import (
    FEATURE_VERSION,
    evaluate_from_manifest,
    export_benchmark_results,
    export_split_manifest,
    load_benchmark_results,
    load_manifest,
)
from batlab.validation.calibration import (
    NOMINAL_INTERVAL_COVERAGE,
    empirical_coverage,
    interval_width_mean,
    recalibrate_lco_intervals,
    run_lco_quantiles,
)

__all__ = [
    "run_lco",
    "RUL_RELIABLE_FLOOR",
    "export_split_manifest",
    "load_manifest",
    "evaluate_from_manifest",
    "export_benchmark_results",
    "load_benchmark_results",
    "run_lco_quantiles",
    "recalibrate_lco_intervals",
    "empirical_coverage",
    "interval_width_mean",
    "NOMINAL_INTERVAL_COVERAGE",
    "FEATURE_VERSION",
]
