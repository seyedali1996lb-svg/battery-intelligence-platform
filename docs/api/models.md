# batlab.models

::: batlab.models.gbrt
    options:
      show_root_heading: true
      members:
        - train_models
        - predict
        - feature_importance_df
        - top_drivers
        - GBRT_PARAMS
        - GBRT_QUANTILE_PARAMS

::: batlab.models.hierarchical
    options:
      show_root_heading: true
      members:
        - hierarchical_hyperparams
        - prior_from_fleet
        - cell_local_stats
        - shrunk_log_rate
        - posterior_variance
        - fit_hierarchical
        - forecast_soh
        - project_future_soh
        - export_priors
        - priors_for_chemistry

::: batlab.models.attribution
    options:
      show_root_heading: true
      members:
        - occlusion_attribution
        - mean_attribution
        - top_attributions
