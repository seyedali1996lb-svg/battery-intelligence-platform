"""
Warranty-breach risk scoring.

Estimates how many cycles remain before a cell's SOH crosses a warranty
floor, and (roughly) how likely that is to happen within a given horizon —
built entirely on values this platform already computes (current SOH,
fade_rate_50cy, and the GBRT RUL model's point + Q10/Q90 quantile
predictions), not a new trained model.

Why this isn't just "reuse rul_pred": the RUL model is trained (and
leave-cell-out validated) to predict cycles until SOH crosses
eol_threshold_pct — 80% by default, this platform's own EOL convention.
Most real warranty terms use a DIFFERENT floor (a manufacturer's stated
"70% capacity retention for N cycles/years" guarantee, or something
uploaded-fleet-specific). Silently reusing rul_pred as if it answered "when
will this breach warranty" would misrepresent a number validated for one
threshold as if it were validated for another. Two honestly-distinct
estimates are exposed instead — see warranty_breach_estimate()'s docstring.
"""

from __future__ import annotations


def cycles_to_soh_floor_linear(
    current_soh_pct: float,
    fade_rate_pct_per_cycle: float,
    floor_soh_pct: float,
) -> "float | None":
    """
    Cycles remaining until SOH crosses floor_soh_pct, extrapolating the
    CURRENT fade rate linearly forward. Always computable but generally
    optimistic once a cell is near or past its knee point, since real
    fade accelerates there and this assumes it stays constant.

    Returns 0.0 if already at/below the floor, None if the fade rate is
    ~0 (extrapolation would imply an infinite number of cycles).
    """
    if current_soh_pct <= floor_soh_pct:
        return 0.0
    if fade_rate_pct_per_cycle <= 1e-9:
        return None
    return (current_soh_pct - floor_soh_pct) / fade_rate_pct_per_cycle


def warranty_breach_estimate(
    current_soh_pct: float,
    fade_rate_pct_per_cycle: float,
    warranty_floor_soh_pct: float,
    rul_pred: "float | None" = None,
    rul_q10: "float | None" = None,
    rul_q90: "float | None" = None,
    eol_threshold_pct: float = 80.0,
    rul_reliable: bool = False,
) -> dict:
    """
    Estimate cycles remaining until current_soh_pct crosses
    warranty_floor_soh_pct.

    Returns a dict with:
      breached:               True if already at/below the floor.
      linear_estimate:        cycles-to-floor from cycles_to_soh_floor_linear()
                               above — always present (None only if fade
                               rate is ~0).
      model_scaled_estimate:  cycles-to-floor derived from the GBRT RUL
                               model's own (LCO-validated) point estimate,
                               proportionally rescaled from eol_threshold_pct
                               to warranty_floor_soh_pct — see below. None
                               when rul_pred/rul_reliable aren't usable.
      model_scaled_q10/q90:   the same rescaling applied to the model's
                               Q10/Q90 interval, forming an 80% band around
                               model_scaled_estimate.
      confidence:             "model" when model_scaled_estimate is
                               available, "linear_only" otherwise — callers
                               should visibly flag linear_only estimates as
                               less trustworthy (no leave-cell-out validation
                               backs a pure linear extrapolation), mirroring
                               RUL's own "Calibrating" confidence tag.

    The rescaling behind model_scaled_estimate: the RUL model predicts
    cycles to reach eol_threshold_pct. Assuming the fade curve's local
    SHAPE near the warranty floor resembles its shape near eol_threshold_pct
    (a real approximation, not exact — it breaks down for a floor far from
    eol_threshold_pct, e.g. requesting a 95% floor while eol_threshold_pct
    is 80%), cycles-to-floor is estimated by scaling rul_pred by the ratio
    of remaining SOH drop needed: (current - floor) / (current - eol_threshold).
    This preserves the model's learned nonlinearity and validated
    uncertainty band instead of a fresh, cruder linear projection — but it
    is still an approximation of a quantity the model was never directly
    trained or LCO-validated against, which is why it's kept visibly
    separate from linear_estimate rather than presented as equally certain.
    """
    if current_soh_pct <= warranty_floor_soh_pct:
        return {
            "breached": True,
            "linear_estimate": 0.0,
            "model_scaled_estimate": 0.0,
            "model_scaled_q10": 0.0,
            "model_scaled_q90": 0.0,
            "confidence": "model" if rul_reliable else "linear_only",
        }

    linear_estimate = cycles_to_soh_floor_linear(current_soh_pct, fade_rate_pct_per_cycle, warranty_floor_soh_pct)

    remaining_to_eol = current_soh_pct - eol_threshold_pct
    can_scale = (
        rul_reliable and rul_pred is not None and remaining_to_eol > 1e-9
    )
    model_scaled_estimate = model_scaled_q10 = model_scaled_q90 = None
    if can_scale:
        remaining_to_floor = current_soh_pct - warranty_floor_soh_pct
        scale = remaining_to_floor / remaining_to_eol
        model_scaled_estimate = max(0.0, rul_pred * scale)  # pyright: ignore[reportOptionalOperand]
        if rul_q10 is not None:
            model_scaled_q10 = max(0.0, rul_q10 * scale)
        if rul_q90 is not None:
            model_scaled_q90 = max(0.0, rul_q90 * scale)

    return {
        "breached": False,
        "linear_estimate": linear_estimate,
        "model_scaled_estimate": model_scaled_estimate,
        "model_scaled_q10": model_scaled_q10,
        "model_scaled_q90": model_scaled_q90,
        "confidence": "model" if model_scaled_estimate is not None else "linear_only",
    }


def probability_of_breach_by(cycles_horizon: float, q10: "float | None", q90: "float | None") -> "float | None":
    """
    Rough probability that the cell crosses its warranty floor within
    cycles_horizon cycles, treating the Q10-Q90 interval as a uniform
    distribution over cycles-to-breach — a deliberately simple
    approximation (GBRT quantile regression only promises calibrated
    MARGINAL quantiles, not a full distribution shape), not a proper
    survival-analysis estimate. Returns None if q10/q90 aren't available.

    cycles_horizon is typically the warranty's own stated cycle/time limit
    (converted to cycles) — "how likely is this cell to breach warranty
    before the warranty period itself ends."
    """
    if q10 is None or q90 is None:
        return None
    if q90 <= q10:
        return 1.0 if cycles_horizon >= q10 else 0.0
    frac = (cycles_horizon - q10) / (q90 - q10)
    return max(0.0, min(1.0, frac))


def estimated_breach_date(cycles_to_breach: "float | None", cycles_per_year: "float | None"):
    """
    Project cycles_to_breach into a calendar date, given the cell's
    historical cycling rate (cycles_per_year — the caller derives this
    from its own cycle_number/test_date history; this function does no
    date math beyond the final addition). Returns None if either input is
    missing or cycles_per_year is ~0 (would imply an undated/infinite
    horizon).
    """
    import datetime

    if cycles_to_breach is None or not cycles_per_year or cycles_per_year <= 1e-9:
        return None
    days = (cycles_to_breach / cycles_per_year) * 365.25
    return datetime.date.today() + datetime.timedelta(days=days)
