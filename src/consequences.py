"""
Phase 4 Consequences: actionable outputs grounded strictly on bundle values.

All functions derive outputs from SOH, RUL, fade rate, and fleet stats already
computed by the pipeline. No new models, no inferred values.
"""

SOH_HEALTHY       = 90.0   # % — above this: healthy
SOH_EOL           = 80.0   # % — below this: end of life
RUL_CRITICAL      = 50     # cycles — if calibrated & below this: critical
SL_UPPER          = 85.0   # % SOH — above this: still primary life
SL_FLOOR          = 65.0   # % SOH — below this: beyond second-life use


# ---------------------------------------------------------------------------
# Action recommendation
# ---------------------------------------------------------------------------

def action_recommendation(
    soh: float,
    rul_reliable: bool,
    rul_pred: float | None,
    fade_30_mah_cy: float,
    fleet_fade_median: float | None = None,
) -> dict:
    """
    Return urgency level and human-readable recommendation.

    Keys: level ("healthy"|"degrading"|"eol"|"critical"),
          title, description, suggestions (list[str])
    """
    suggestions: list[str] = []

    if soh > SOH_HEALTHY:
        level       = "healthy"
        title       = "Healthy — Continue Normal Operation"
        description = (
            f"SOH is {soh:.1f}%, well above the 80% end-of-life threshold. "
            "No immediate action required."
        )
    elif soh > SOH_EOL:
        level       = "degrading"
        title       = "Degrading — Plan for Replacement"
        description = (
            f"SOH is {soh:.1f}%, in the degrading zone (80–90%). "
            "The cell is still operational but approaching end of useful life."
        )
    else:
        level       = "eol"
        title       = "End of Life — Schedule Replacement"
        description = (
            f"SOH is {soh:.1f}%, at or below the 80% EOL threshold. "
            "Continued operation risks capacity shortfall. Replacement recommended."
        )

    # Override to critical if calibrated RUL is very low
    if rul_reliable and rul_pred is not None and rul_pred < RUL_CRITICAL:
        level       = "critical"
        title       = "Critical — Replacement Urgent"
        description = (
            f"SOH is {soh:.1f}% and the calibrated model estimates only "
            f"{int(rul_pred)} cycles remaining. Schedule replacement immediately."
        )

    # Stress-specific suggestions
    if fleet_fade_median is not None and fleet_fade_median > 0:
        if fade_30_mah_cy > fleet_fade_median * 1.3:
            suggestions.append(
                "Fade rate is 30%+ above fleet median. Reducing charge C-rate or "
                "depth of discharge can slow degradation meaningfully."
            )

    if level in ("degrading", "eol", "critical"):
        suggestions.append(
            "Avoid operating above 35°C — thermal stress is the primary accelerator "
            "of SEI growth and capacity loss at this SOH level."
        )

    if rul_reliable and rul_pred is not None and rul_pred < 200:
        suggestions.append(
            f"Approximately {int(rul_pred)} cycles estimated remaining. "
            "Pre-ordering a replacement cell now avoids unplanned downtime."
        )

    return {
        "level":       level,
        "title":       title,
        "description": description,
        "suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def cost_model(
    rul_reliable: bool,
    rul_pred: float | None,
    replacement_cost: float,
    value_per_cycle: float,
) -> dict:
    """
    Compute replacement economics.

    Keys: rul_reliable, rul_pred, replacement_cost, value_per_cycle,
          remaining_value, cost_per_remaining_cycle, replace_now_net
    """
    result: dict = {
        "rul_reliable":            rul_reliable,
        "rul_pred":                rul_pred,
        "replacement_cost":        replacement_cost,
        "value_per_cycle":         value_per_cycle,
        "remaining_value":         None,
        "cost_per_remaining_cycle": None,
        "replace_now_net":         None,
    }

    if not rul_reliable or rul_pred is None:
        return result

    rul = max(rul_pred, 1.0)
    remaining_value         = rul * value_per_cycle
    cost_per_remaining_cycle = replacement_cost / rul
    # Positive → worth running to EOL (remaining value > cost of replacing now)
    replace_now_net          = remaining_value - replacement_cost

    result["remaining_value"]          = remaining_value
    result["cost_per_remaining_cycle"] = cost_per_remaining_cycle
    result["replace_now_net"]          = replace_now_net
    return result


# ---------------------------------------------------------------------------
# Second-life score
# ---------------------------------------------------------------------------

def second_life_score(
    soh: float,
    fade_30_mah_cy: float,
    fleet_fade_median: float | None,
) -> dict:
    """
    Score second-life suitability for stationary storage (0–100).

    Keys: score (int|None), category, description, notes (list[str])
    """
    notes: list[str] = []

    if soh > SL_UPPER:
        return {
            "score":       None,
            "category":    "Primary Life",
            "description": (
                f"SOH is {soh:.1f}% — still within primary use range. "
                "Second-life suitability will be assessed as SOH approaches 80–85%."
            ),
            "notes": notes,
        }

    if soh < SL_FLOOR:
        return {
            "score":       0,
            "category":    "Beyond Second Life",
            "description": (
                f"SOH is {soh:.1f}%, below the 65% floor typical for second-life "
                "applications. Evaluate for material recycling."
            ),
            "notes": notes,
        }

    # SOH component: 0–60 pts  (SL_FLOOR → 0,  SL_UPPER → 60)
    soh_pts = (soh - SL_FLOOR) / (SL_UPPER - SL_FLOOR) * 60.0

    # Fade rate component: 0–40 pts  (slow fade = more points)
    if fleet_fade_median is not None and fleet_fade_median > 0:
        ratio     = fade_30_mah_cy / fleet_fade_median
        fade_pts  = max(0.0, min(40.0, (2.0 - ratio) * 40.0))
        if ratio < 0.8:
            notes.append("Fade rate is below fleet median — positive indicator for second-life longevity.")
        elif ratio > 1.5:
            notes.append("Fade rate is above fleet median — second-life capacity will diminish faster than average.")
    else:
        fade_pts = 20.0  # neutral when no fleet reference

    score = int(max(0, min(100, soh_pts + fade_pts)))

    if score >= 70:
        category    = "Excellent Candidate"
        description = (
            f"SOH {soh:.1f}% with a slow fade rate — well suited for stationary "
            "storage (grid buffer, UPS, residential ESS). Expected to support "
            "200–500 additional cycles at ≥70% SOH."
        )
    elif score >= 50:
        category    = "Good Candidate"
        description = (
            f"SOH {soh:.1f}% — suitable for low-demand stationary applications. "
            "Not recommended for high-cycle or high-rate secondary use."
        )
    elif score >= 30:
        category    = "Marginal"
        description = (
            f"SOH {soh:.1f}% — limited second-life value. Suitable only for very "
            "low-demand backup applications with infrequent cycling."
        )
    else:
        category    = "Not Recommended"
        description = (
            f"SOH {soh:.1f}% with high fade rate — second-life use not recommended. "
            "Evaluate for cathode/anode material recycling."
        )

    notes.append(
        "Assumes stationary storage conditions: 70–80% SOH operating window, "
        "C/5 charge rate, ≤25°C ambient."
    )

    return {
        "score":       score,
        "category":    category,
        "description": description,
        "notes":       notes,
    }
