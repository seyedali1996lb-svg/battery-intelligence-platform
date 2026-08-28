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
