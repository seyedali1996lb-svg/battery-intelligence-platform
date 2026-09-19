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

::: batlab.validation.survival
    options:
      show_root_heading: true
      members:
        - cell_event_type
        - kaplan_meier_survival
        - rule_of_three_bound
        - censored_rul_readout

::: batlab.validation.fold_cache
    options:
      show_root_heading: true
      members:
        - cache_mode
        - cache_dir
        - fold_key
        - cache_key_id
        - for_run
        - clear
        - describe

::: batlab.validation.bundle_data
    options:
      show_root_heading: true
      members:
        - write_bundle_cells
        - load_bundle_cells
        - read_bundle_cell_file
        - table_csv_text
        - cell_file_name

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

::: batlab.harness.model_source
    options:
      show_root_heading: true
      members:
        - write_bundle_model
        - load_bundle_model
        - import_module_source
        - BUNDLE_MODEL_LOADER
        - ModuleSourceError

::: batlab.harness.sandbox
    options:
      show_root_heading: true
      members:
        - sandbox_forecaster
        - SandboxedModelFactory
        - SandboxLimits
        - SandboxError
        - describe_enforcement
