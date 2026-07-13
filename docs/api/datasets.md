# batlab.datasets

::: batlab.datasets.schema
    options:
      show_root_heading: true
      members:
        - REQUIRED_CYCLE_COLUMNS
        - REQUIRED_CHECKPOINT_COLUMNS
        - OPTIONAL_COLUMNS
        - REQUIRED_ATTRS
        - SchemaError
        - validate_schema
        - compute_soh_pct

::: batlab.datasets.nasa
    options:
      show_root_heading: true
      members:
        - load_nasa_cells

::: batlab.datasets.severson
    options:
      show_root_heading: true
      members:
        - load_severson_cells
        - any_cached
        - download_and_prepare

::: batlab.datasets.oxford
    options:
      show_root_heading: true
      members:
        - load_oxford_cells
        - any_cached
        - download_and_prepare

::: batlab.datasets.calce
    options:
      show_root_heading: true
      members:
        - load_calce_cells
        - CalceDataNotFoundError
        - any_cached
