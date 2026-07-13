"""
Standardized cycle-level DataFrame schema for battery degradation data.

Every loader in `batlab.datasets` returns a dict of {cell_id: DataFrame}
where each DataFrame satisfies this contract. This is the single point of
truth other batlab code (features, models, validation) can rely on instead
of each consumer re-deriving "what columns might this dataset have."

Relationship to TRI's `beep` (Battery Evaluation and Early Prediction)
------------------------------------------------------------------------
`beep` is the closest prior art for a standardized battery-cycling schema
in Python. From its public documentation and structuring pipeline (as of
this library's design pass — verify against the current beep docs before
citing specifics, since live web access was unavailable while writing
this module):

  - beep's raw structuring keeps per-*datapoint* (sub-cycle) data
    (`cycle_index`, `step_index`, `voltage`, `current`, `charge_capacity`,
    `discharge_capacity`, `temperature`, `date_time`) and derives a
    per-cycle `summary` table from it (`cycle_index`, `discharge_capacity`,
    `charge_capacity`, `dc_internal_resistance`,
    `temperature_maximum/average/minimum`, `date_time_iso`, ...).
  - This library only standardizes the *summary* level (one row per
    cycle) — batlab does not attempt to standardize raw per-datapoint
    time series, since NASA/Severson/Oxford don't expose it in a common
    shape and this library's models (GBRT SOH/RUL) only ever consume
    cycle-level features.

Where this schema maps to beep's summary table:
    this schema        beep summary (approx.)
    ----------------    -----------------------
    cycle_number    ->  cycle_index          (both 1-indexed... verify: beep is 0-indexed)
    capacity_ah     ->  discharge_capacity
    resistance_ohm  ->  dc_internal_resistance
    temperature_c   ->  temperature_average
    test_date       ->  date_time_iso

Where it diverges:
    - beep separates charge_capacity and discharge_capacity; this schema
      only tracks discharge capacity (`capacity_ah`), since none of the
      four datasets this library ships loaders for report both in a
      consistently comparable way, and RUL/SOH modeling here is
      discharge-capacity-based throughout.
    - This schema adds `soh_pct` as a first-class, always-present column
      (capacity relative to the cell's own first cycle) — beep leaves
      SOH computation to downstream consumers. Standardizing it here
      means every loader is directly usable without a separate
      normalization step.
    - This schema adds `coulombic_efficiency` as a named optional column;
      beep computes it ad hoc from charge/discharge capacity ratios where
      both are present.
    - This library does not attempt beep's raw per-datapoint structuring
      or its Arbin/Maccor/BioLogic multi-vendor-file-format parsing —
      batlab's loaders each parse one specific published dataset's native
      format directly into the summary-level schema below.

Two schema "kinds"
-------------------
Most datasets (NASA, Severson, CALCE) are DENSE: one row per real charge/
discharge cycle, hundreds to thousands of rows per cell. `kind="cycle"`.

The Oxford Path-Dependent dataset is SPARSE: only ~10-14 Reference
Performance Test checkpoints exist per cell (see batlab/datasets/oxford.py
for why) — it is not meaningfully "per cycle" and forcing it into the
dense shape would misrepresent how little data actually exists per cell.
`kind="checkpoint"` uses `checkpoint_index` instead of `cycle_number` and
drops the cycle-level-only optional columns (resistance/temperature/CE
are not resolvable at RPT granularity for this source).
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

# Required for kind="cycle" (dense per-cycle data: NASA, Severson, CALCE).
REQUIRED_CYCLE_COLUMNS: dict[str, str] = {
    "cycle_number": "int — 1-indexed cycle count, monotonically increasing",
    "capacity_ah":  "float — measured discharge capacity, ampere-hours",
    "soh_pct":      "float — capacity_ah relative to this cell's own first "
                     "measured cycle, as a percentage (100.0 at cycle 1)",
}

# Required for kind="checkpoint" (sparse reference-test data: Oxford).
REQUIRED_CHECKPOINT_COLUMNS: dict[str, str] = {
    "checkpoint_index": "int — 0-indexed reference-test checkpoint (0 = "
                          "beginning of life), NOT a cycle count",
    "capacity_ah":       "float — measured discharge capacity at this checkpoint",
    "soh_pct":           "float — capacity_ah relative to checkpoint 0, as a percentage",
}

# Optional for either kind — present only when the source data supports it.
# Loaders must NOT fabricate these; omit the column (or leave it all-NaN)
# rather than guess.
OPTIONAL_COLUMNS: dict[str, str] = {
    "resistance_ohm":        "float — internal/ohmic resistance, ohms",
    "temperature_c":         "float — mean cell temperature during the cycle, Celsius",
    "coulombic_efficiency":  "float — discharge/charge capacity ratio, dimensionless "
                              "(~0.99+ for a healthy cell)",
    "c_rate":                "float — discharge C-rate, dimensionless (protocol-known "
                              "or measured; NaN if genuinely unknown)",
    "test_date":             "str (ISO 8601) or datetime64 — wall-clock date of the "
                              "cycle/checkpoint, if the source records it",
}

ALL_KNOWN_COLUMNS = {**REQUIRED_CYCLE_COLUMNS, **REQUIRED_CHECKPOINT_COLUMNS, **OPTIONAL_COLUMNS}

# DataFrame-level metadata every loader must set via df.attrs (not columns —
# these are per-cell/per-source facts, not per-row measurements, so
# duplicating them into every row would be wasteful and easy to typo out of
# sync). See Cell/cite.py for how these get consumed.
REQUIRED_ATTRS = {
    "cell_id":    "str — this cell's identifier within its source dataset",
    "source":     "str — short dataset key, e.g. 'nasa', 'severson2019', 'oxford2020', 'calce'",
    "chemistry":  "str — cell chemistry, e.g. 'LiCoO2', 'LFP', 'NCA'",
    "citation":   "str — BibTeX citation key for this dataset (see batlab.cite)",
    "license":    "str — the dataset's redistribution/use license, verbatim",
}


class SchemaError(ValueError):
    """Raised by validate_schema() when a DataFrame does not satisfy the
    batlab cycle/checkpoint schema contract. Always includes the specific
    missing/invalid field(s) in the message — never a bare 'invalid data'."""


def validate_schema(df: pd.DataFrame, kind: str = "cycle") -> None:
    """
    Validate that `df` satisfies the batlab schema contract for `kind`
    ("cycle" or "checkpoint"). Raises SchemaError with a specific,
    actionable message on any violation. Returns None on success (call
    site pattern: `validate_schema(df)` as a guard, not `if validate_schema(df):`).

    This checks structure and sanity, not exact numeric correctness —
    it cannot tell you your capacity values are physically wrong, only
    that the DataFrame is shaped the way downstream batlab code expects.
    """
    if kind == "cycle":
        required = REQUIRED_CYCLE_COLUMNS
        index_col = "cycle_number"
    elif kind == "checkpoint":
        required = REQUIRED_CHECKPOINT_COLUMNS
        index_col = "checkpoint_index"
    else:
        raise SchemaError(f"Unknown schema kind {kind!r} — must be 'cycle' or 'checkpoint'.")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Missing required column(s) for kind={kind!r}: {missing}. "
            f"Required: {list(required)}. Got: {list(df.columns)}."
        )

    if df.empty:
        raise SchemaError("DataFrame has zero rows — a loader must return at least one "
                           "cycle/checkpoint per cell, or omit that cell entirely.")

    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(
                f"Column {col!r} must be numeric, got dtype {df[col].dtype!r}."
            )
        if df[col].isna().all():
            raise SchemaError(f"Required column {col!r} is entirely NaN.")

    idx = df[index_col]
    if not idx.is_monotonic_increasing:
        raise SchemaError(
            f"{index_col!r} must be sorted ascending — loaders should sort_values() "
            f"before returning. First few values: {idx.head(5).tolist()}"
        )
    if idx.min() < 0:
        raise SchemaError(f"{index_col!r} must be >= 0, found minimum {idx.min()}.")

    soh = df["soh_pct"]
    # Allow modest measurement noise above 100% (e.g. a slightly higher
    # capacity reading on cycle 2 than cycle 1) but not wildly invalid values.
    if soh.min() < -1 or soh.max() > 110:
        raise SchemaError(
            f"soh_pct out of a physically plausible range: min={soh.min():.2f}, "
            f"max={soh.max():.2f} (expected roughly 0-110)."
        )

    for col, dtype_hint in OPTIONAL_COLUMNS.items():
        if col in df.columns and not df[col].isna().all():
            if col != "test_date" and not pd.api.types.is_numeric_dtype(df[col]):
                raise SchemaError(f"Optional column {col!r} is present but not numeric "
                                   f"(dtype {df[col].dtype!r}) — {dtype_hint}")

    missing_attrs = [a for a in REQUIRED_ATTRS if a not in df.attrs]
    if missing_attrs:
        raise SchemaError(
            f"Missing required df.attrs entr(y/ies): {missing_attrs}. "
            f"Required attrs: {list(REQUIRED_ATTRS)}. A loader must set these on "
            f"every DataFrame it returns, e.g. df.attrs['source'] = 'nasa'."
        )


def compute_soh_pct(capacity_ah: pd.Series) -> pd.Series:
    """
    Shared helper so every loader computes State of Health the same way:
    capacity relative to this cell's own first measured value, as a
    percentage. Centralizing this one small formula avoids the exact bug
    class this project has hit before — a formula reimplemented slightly
    differently (or omitted) in each new loader.
    """
    first = float(capacity_ah.iloc[0])
    if first <= 0:
        raise SchemaError(f"First capacity_ah value must be > 0 to compute soh_pct, got {first}.")
    return capacity_ah / first * 100.0
