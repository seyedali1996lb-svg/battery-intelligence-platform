# batlab.validation

::: batlab.validation.lco
    options:
      show_root_heading: true
      members:
        - run_lco
        - RUL_RELIABLE_FLOOR

::: batlab.validation.manifest
    options:
      show_root_heading: true
      members:
        - export_split_manifest
        - load_manifest
        - evaluate_from_manifest
        - export_benchmark_results
        - load_benchmark_results
        - FEATURE_VERSION

::: batlab.validation.calibration
    options:
      show_root_heading: true
      members:
        - run_lco_quantiles
        - recalibrate_lco_intervals
        - empirical_coverage
        - interval_width_mean
        - NOMINAL_INTERVAL_COVERAGE

::: batlab.validation.prospective
    options:
      show_root_heading: true
      members:
        - run_prospective
        - split_prospective
        - trivial_soh_baseline_prospective
        - rul_formula_baseline_prospective

::: batlab.validation.metric_gate
    options:
      show_root_heading: true
      members:
        - evaluate_gate
        - check_metric
        - load_expectations
        - format_gate_report

::: batlab.validation.replication
    options:
      show_root_heading: true
      members:
        - verify_bundle
        - load_bundle
        - format_verification

# batlab.harness

The model-agnostic validation harness — see the
[harness guide](../harness.md) for usage, the gate file format, and the
honest limits of each number it reports.

::: batlab.harness.forecaster
    options:
      show_root_heading: true
      members:
        - Forecaster
        - IntervalForecaster
        - SklearnForecaster
        - SklearnIntervalForecaster
        - CallableForecaster
        - torch_forecaster
        - as_factory
        - fit_forecaster
        - has_predict_interval
        - forecaster_identity
        - default_forecaster
        - default_interval_forecaster

::: batlab.harness.harness
    options:
      show_root_heading: true
      members:
        - validate_forecaster
        - seal_bundle
        - format_report
