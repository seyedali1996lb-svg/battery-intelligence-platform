"""
Phase 6 — Recommendation engine.

Classifies a cell into one of four actions based on pipeline outputs and
Phase 4 fit scores. All thresholds are explicit constants at the top of
this file — nothing is buried in a scoring function.

Action hierarchy (evaluated top-down):
  1. Continue Operation  — SOH > 85% AND fade not accelerating
  2. Schedule Inspection — SOH 80–85%, OR fade accelerating >2× baseline
                           even while SOH is still healthy
  3. Route to Second-Life — SOH 70–80% AND at least one Phase 4 app is
                             "fit" or "marginal"
  4. Recycle             — SOH < 70%, OR no application fit found

Confidence reflects TWO separate uncertainty sources:
  - RUL reliability (per-cell fold R² vs floor) — affects cycle-timeline
    certainty only, not the SOH-based action itself
  - Fit-score routing uncertainty — when action = second_life or recycle
    because of application_fit() output, that output carries Phase 4's
    amber "Cited estimate" badges. The confidence label must inherit that.

These are combined but kept named separately so the UI can explain each.
"""

# ---------------------------------------------------------------------------
# Explicit thresholds — change here, changes everywhere
# ---------------------------------------------------------------------------

SOH_PRIMARY_FLOOR   = 85.0   # above this → primary life (Continue or Inspect)
SOH_INSPECT_FLOOR   = 80.0   # 80–85% = inspection band
SOH_SECONDLIFE_FLOOR = 70.0  # 70–80% = evaluate second-life
                              # below 70% → Recycle regardless

FADE_ACCEL_RATIO    = 2.0    # fade_30cy / fade_50cy above this → "accelerating"
                              # relative trigger: recent rate vs longer-window baseline

BOUNDARY_MARGIN     = 3.0    # ±pp around a threshold that counts as "near boundary"


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify(
    soh: float,
    fade_30: float,
    fade_50: float,
    rul_reliable: bool,
    rul_pred: float | None,
    fit_scores: dict,
) -> dict:
    """
    Returns a structured result dict consumed by the Recommendations page.

    fit_scores: output of consequences.application_fit() — dict keyed by
                app key, each entry has "fit" ("fit"|"marginal"|"not_fit"),
                "name", "reasons", etc.
    """
    # ── Fade acceleration signal ──
    fade_ratio = (fade_30 / fade_50) if (fade_50 and fade_50 > 0) else 1.0
    fade_accelerating = fade_ratio > FADE_ACCEL_RATIO

    # ── Action (evaluated in priority order) ──
    fit_driven     = False
    best_app_key   = None
    best_app       = None
    action_reasons = []

    if soh > SOH_PRIMARY_FLOOR and not fade_accelerating:
        action = "continue"
        action_reasons.append(f"SOH {soh:.1f}% is above the {SOH_PRIMARY_FLOOR:.0f}% primary-life threshold.")
        action_reasons.append(f"Fade rate is stable ({fade_ratio:.1f}× baseline — below the {FADE_ACCEL_RATIO:.0f}× acceleration flag).")

    elif soh > SOH_PRIMARY_FLOOR and fade_accelerating:
        action = "inspect"
        action_reasons.append(f"SOH {soh:.1f}% is in primary life, but fade rate is {fade_ratio:.1f}× its own baseline — above the {FADE_ACCEL_RATIO:.0f}× acceleration threshold.")
        action_reasons.append("Accelerating fade warrants inspection even before the SOH window.")

    elif soh >= SOH_INSPECT_FLOOR:
        action = "inspect"
        action_reasons.append(f"SOH {soh:.1f}% is in the {SOH_INSPECT_FLOOR:.0f}–{SOH_PRIMARY_FLOOR:.0f}% inspection band.")
        if fade_accelerating:
            action_reasons.append(f"Fade rate is also accelerating ({fade_ratio:.1f}× baseline) — increasing urgency.")
        else:
            action_reasons.append(f"Fade rate is stable ({fade_ratio:.1f}× baseline).")

    elif soh >= SOH_SECONDLIFE_FLOOR:
        fit_driven = True
        ranked = sorted(
            fit_scores.items(),
            key=lambda kv: {"fit": 2, "marginal": 1, "not_fit": 0}[kv[1]["fit"]],
            reverse=True,
        )
        best_app_key, best_app = ranked[0]
        if best_app["fit"] in ("fit", "marginal"):
            action = "second_life"
            action_reasons.append(f"SOH {soh:.1f}% is in the second-life evaluation window ({SOH_SECONDLIFE_FLOOR:.0f}–{SOH_INSPECT_FLOOR:.0f}%).")
            action_reasons.append(f"Best application fit: {best_app['name']} ({best_app['fit']}).")
        else:
            action = "recycle"
            action_reasons.append(f"SOH {soh:.1f}% is in the second-life window but no application returned a viable fit.")
            action_reasons.append("Recycling is recommended when no second-life use case applies.")

    else:
        action = "recycle"
        action_reasons.append(f"SOH {soh:.1f}% is below the {SOH_SECONDLIFE_FLOOR:.0f}% second-life floor.")
        action_reasons.append("No second-life application accepts cells at this SOH level.")

    # ── Confidence ──
    # Near a decision boundary?
    near_boundary = any(
        abs(soh - threshold) <= BOUNDARY_MARGIN
        for threshold in [SOH_PRIMARY_FLOOR, SOH_INSPECT_FLOOR, SOH_SECONDLIFE_FLOOR]
    )

    confidence_reasons = []
    if not rul_reliable:
        confidence_reasons.append(
            f"RUL is not calibrated for this cell (fold R² below {0.30} reliability floor). "
            "Any cycle-count timeline shown below is omitted or marked accordingly."
        )
    if fit_driven:
        confidence_reasons.append(
            "Routing decision used Phase 4 application fit scores — those scores carry "
            "cited-estimate uncertainty (amber badges). The action is grounded in literature "
            "thresholds, not a validated model output."
        )
    if near_boundary:
        confidence_reasons.append(
            f"SOH {soh:.1f}% is within {BOUNDARY_MARGIN:.0f}pp of a decision threshold. "
            "A small change in SOH estimate could shift the recommendation."
        )

    rul_issue      = not rul_reliable
    fit_issue      = fit_driven
    boundary_issue = near_boundary
    n_issues       = sum([rul_issue, fit_issue, boundary_issue])

    if n_issues == 0:
        confidence = "high"
    elif n_issues == 1 and rul_issue:
        # RUL alone: action is still SOH-based and clear — moderate reduction
        confidence = "medium"
    elif n_issues == 1:
        # Fit-driven or near boundary, but RUL is calibrated
        confidence = "lower"
    else:
        # Two or more factors compounding — e.g. B0018: RUL uncalibrated + fit-driven
        confidence = "uncertain"

    return {
        "action":            action,
        "confidence":        confidence,
        "confidence_reasons": confidence_reasons,
        "action_reasons":    action_reasons,
        "fade_accelerating": fade_accelerating,
        "fade_ratio":        fade_ratio,
        "fit_driven":        fit_driven,
        "best_app_key":      best_app_key,
        "best_app":          best_app,
        "fit_scores":        fit_scores,
    }
