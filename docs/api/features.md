# batlab.features

::: batlab.features.engineering
    options:
      show_root_heading: true
      members:
        - build_features
        - get_model_matrix
        - feature_summary
        - FEATURE_COLUMNS

::: batlab.features.knee_detection
    options:
      show_root_heading: true
      members:
        - detect_knee
        - degradation_phases

::: batlab.features.dqdv
    options:
      show_root_heading: true
      members:
        - simulate_vq_curve
        - extract_dqdv_features
        - add_dqdv_features

::: batlab.features.physics_calibration
    options:
      show_root_heading: true
      members:
        - calibrate_cell
        - calibrated_feature_series
        - dominant_mode
        - fit_two_term_fade
        - fit_resistance_growth
        - physics_ml_agreement
        - physics_gbrt_divergence_report
        - ANCHOR_PARAM_SETS
        - register_anchor_param_set
        - register_mechanism_classifier
