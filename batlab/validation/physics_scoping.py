"""
Physics-parameter provenance and scoping (Tier 4 item 5).

Why this exists
---------------
The GBRT consumes four physics-calibration features per cell —
`physics_beta_sei`, `physics_beta_lam`, `physics_k_r`,
`physics_fit_r2` (src/physics_calibration.py's scipy fit of a two-term
degradation law against the cell's own history) — and the Benchmark page
shows the physics fit's R² next to the GBRT's. The request these numbers
invite is: "are these THE cell's SEI/LAM rates?" They are not, and the
distance between what they are and what a reader might assume is exactly
the gap item 5 closes:

  - they are FITS to cycling data (capacity/resistance trajectories),
    not independent measurements. No EIS spectroscopy, no post-mortem
    (teardown, SEM/EDS, coin-cell harvest) exists for any platform cell;
  - the two-term law (sqrt(n) SEI + linear LAM) is a widely used
    simplification. Real degradation is a superposition of more
    mechanisms with different voltage/temperature dependence; a good
    fit_r2 means the two-term projection describes THIS curve, not that
    the physical attribution (how much loss is SEI vs LAM) is unique.
    Structural identifiability of beta_sei/beta_lam from capacity data
    alone is limited — the fit can trade the two terms against each
    other along the curve;
  - `physics_k_r` is fit from the resistance channel, giving an
    independent corroboration signal — but from the same cycling record,
    not from an independent instrument.

So the parameters are honestly "physics-INFORMED fits with quantified
goodness-of-fit", a strictly weaker and clearly-stated claim.
`physics_parameter_provenance()` is the single source of that statement,
in the per-source form every surface needs (NASA/Severson have the
features; Zhu/CALCE/synthetic do not), plus per-cell quality tiers that
surfaces can use to decide how much weight a physics-derived read may
carry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sources whose loaders populate the physics features. Keyed in THIS module's
# dataset-key vocabulary ("nasa"/"severson", as everywhere else in the registry
# and these tables) — not the loaders' df.attrs["source"] vocabulary that
# physics_calibration.ANCHOR_PARAM_SETS uses ("nasa"/"severson2019"), which is
# the eligibility rule itself: a (source, chemistry) pair with a PyBaMM anchor
# AND enough usable history to refit the two-term model causally.
PHYSICS_FEATURE_SOURCES = ("nasa", "severson")

PROVENANCE_STATEMENT = (
    "Physics parameters (physics_beta_sei / beta_lam / k_r, physics_fit_r2) "
    "are scipy FITS of a two-term degradation law (sqrt(n) SEI + linear LAM) "
    "to each cell's own cycling record — NOT independent measurements. No EIS "
    "spectroscopy or post-mortem teardown exists for any platform cell, so the "
    "physical attribution is a model-based decomposition, structurally unable "
    "to fully separate the two terms from capacity data alone. physics_k_r is "
    "fit from the resistance channel (same record, independent signal path). "
    "physics_fit_r2 quantifies how well the two-term projection describes the "
    "observed curve — a good fit does not uniquely identify the mechanism mix."
)

# Per-cell quality tiers from physics_fit_r2: how much weight a
# physics-derived read may carry on that cell.
TIER_STRONG_FLOOR = 0.90     # fit describes the curve well
TIER_USABLE_FLOOR = 0.70     # directionally usable
# below: "weak — treat the decomposition as illustrative only"


def physics_parameter_provenance(cell_data: dict, source_kind: "str | None" = None) -> dict:
    """Per-source provenance statement + per-cell quality tiers.

    Args:
        cell_data: {cell_id: FEATURED DataFrame} (needs the physics_*
                   columns; raw frames simply yield absent-tier cells).
        source_kind: optional dataset key for the per-source statement.

    Returns:
        {
          "provenance_statement": str,
          "source_kind": str | None,
          "features_available": bool,   # does ANY cell carry the columns
          "per_cell": {cell_id: {
              "physics_fit_r2": float | None,
              "tier": "strong" | "usable" | "weak" | "absent",
              "tier_note": str,
          }},
          "tier_counts": {tier: count},
          "validation_gap": str,   # what EIS/post-mortem would add
        }
    """
    per_cell = {}
    tier_counts = {"strong": 0, "usable": 0, "weak": 0, "absent": 0}
    any_available = False

    for cell_id, df in (cell_data or {}).items():
        r2 = None
        if df is not None and "physics_fit_r2" in df.columns:
            vals = pd.to_numeric(df["physics_fit_r2"], errors="coerce").dropna()
            if len(vals):
                r2 = float(vals.iloc[-1])
                any_available = True
        if r2 is None:
            tier, note = "absent", (
                "no physics features for this cell — decomposition not computed"
            )
        elif r2 >= TIER_STRONG_FLOOR:
            tier, note = "strong", (
                "two-term fit describes the curve well; attribution is "
                "model-based and remains non-unique"
            )
        elif r2 >= TIER_USABLE_FLOOR:
            tier, note = "usable", (
                "fit captures the trend but not the detail; lean on the "
                "direction, not the split"
            )
        else:
            tier, note = "weak", (
                "poor fit — treat the SEI/LAM decomposition as illustrative "
                "only, not as a mechanism measurement"
            )
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        per_cell[cell_id] = {
            "physics_fit_r2": r2,
            "tier": tier,
            "tier_note": note,
        }

    return {
        "provenance_statement": PROVENANCE_STATEMENT,
        "source_kind": source_kind,
        "features_available": any_available,
        "per_cell": per_cell,
        "tier_counts": tier_counts,
        "validation_gap": (
            "Validating the attribution against ground truth would require "
            "EIS (to separate ohmic/SEI/charge-transfer resistance) and "
            "post-mortem teardown (to measure active-material loss directly). "
            "Neither exists for any platform cell; until then the parameters "
            "are corroboration signals within one modeling framework, not "
            "independent physical measurements."
        ),
    }
