"""Leave-cell-out cross-validation, the per-cell reliability gate, quantile-interval calibration, reproducible manifests, the accuracy regression gate, and sealed replication bundles."""

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
from batlab.validation.metric_gate import (
    evaluate_gate,
    load_expectations as load_gate_expectations,
)
from batlab.validation.fingerprints import (
    cell_digest,
    dataset_fingerprint,
    environment_snapshot,
)
from batlab.validation.survival import (
    cell_event_type,
    censored_rul_readout,
    kaplan_meier_survival,
    rule_of_three_bound,
)
from batlab.validation.fold_cache import (
    cache_dir as fold_cache_dir,
    clear as clear_fold_cache,
    describe as describe_fold_cache,
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
    "evaluate_gate",
    "load_gate_expectations",
    "cell_digest",
    "dataset_fingerprint",
    "environment_snapshot",
    "cell_event_type",
    "kaplan_meier_survival",
    "rule_of_three_bound",
    "censored_rul_readout",
    "fold_cache_dir",
    "clear_fold_cache",
    "describe_fold_cache",
]
