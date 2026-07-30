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


# ---------------------------------------------------------------------------
# Degradation mechanism diagnosis (LLI vs LAM)
# ---------------------------------------------------------------------------

def diagnose_mechanism(df) -> dict:
    """
    Classifies a cell's dominant degradation mechanism — Loss of Lithium
    Inventory (LLI) vs Loss of Active Material (LAM) — from three signals:
    coulombic-efficiency trend, capacity-fade curve shape, and internal-
    resistance rise rate. Any subset of the three columns may be missing;
    the verdict's confidence reflects how many signals were available.

    df: a cell's featured DataFrame — used columns (all optional):
        cycle_number, coulombic_efficiency, soh_pct, resistance_normalized.
    """
    import numpy as _np_mech

    signals = {}
    confidence_notes = []

    # Signal 1: CE trend slope → LLI indicator (works for all chemistries)
    if "coulombic_efficiency" in df.columns:
        ce_valid = df[["cycle_number", "coulombic_efficiency"]].dropna()
        if len(ce_valid) >= 10:
            ce_arr = ce_valid["coulombic_efficiency"].values
            cy_arr = ce_valid["cycle_number"].values.astype(float)
            ce_slope = _np_mech.polyfit(cy_arr, ce_arr, 1)[0]
            ce_deficit_pct = (1.0 - ce_arr.mean()) * 100
            signals["lli_ce_slope"] = ce_slope
            signals["lli_ce_deficit"] = ce_deficit_pct
            confidence_notes.append("CE trend")

    # Signal 2: Capacity fade linearity — linear fade = LLI, accelerating = LAM
    if "soh_pct" in df.columns:
        soh_valid = df[["cycle_number", "soh_pct"]].dropna()
        if len(soh_valid) >= 20:
            cy_s = soh_valid["cycle_number"].values.astype(float)
            soh_s = soh_valid["soh_pct"].values
            # Fit linear + quadratic — if quadratic term >> 0, accelerating (LAM)
            cy_norm = (cy_s - cy_s.min()) / max(cy_s.max() - cy_s.min(), 1)
            p2 = _np_mech.polyfit(cy_norm, soh_s, 2)
            nonlinearity = float(p2[0])  # positive curvature = accelerating fade
            fade_total = float(soh_s[0] - soh_s[-1]) if len(soh_s) > 0 else 0
            signals["lam_nonlinearity"] = nonlinearity
            signals["fade_total"] = fade_total
            confidence_notes.append("fade shape")

    # Signal 3: Resistance rise rate → SEI/LAM indicator
    if "resistance_normalized" in df.columns:
        r_valid = df[["cycle_number", "resistance_normalized"]].dropna()
        if len(r_valid) >= 10:
            r_slope = _np_mech.polyfit(
                r_valid["cycle_number"].values.astype(float),
                r_valid["resistance_normalized"].values, 1
            )[0]
            signals["resistance_slope"] = r_slope * 1000  # Ω/1000cy
            confidence_notes.append("resistance")

    # ── Classification logic ──
    lli_score = 0
    lam_score = 0

    ce_slope_val = signals.get("lli_ce_slope", 0)
    if ce_slope_val < -1e-6:   # declining CE → LLI
        lli_score += 3
    ce_def = signals.get("lli_ce_deficit", 0)
    if ce_def > 0.05:          # CE deficit > 0.05% → active LLI
        lli_score += 2

    nonlin = signals.get("lam_nonlinearity", 0)
    if nonlin < -0.5:          # accelerating fade → LAM
        lam_score += 3
    elif nonlin > 0.5:         # decelerating (stabilising) → LLI
        lli_score += 1

    r_slope = signals.get("resistance_slope", 0)
    if r_slope > 0.02:         # fast resistance rise → LAM (particle contact loss)
        lam_score += 2
    elif 0.005 < r_slope <= 0.02:
        lli_score += 1         # moderate rise → SEI (LLI-associated)

    # Verdict
    total = lli_score + lam_score
    if total == 0:
        verdict = "Insufficient data"
        verdict_color = "#718096"
        verdict_icon = "○"
        verdict_body = "Not enough degradation signals to classify mechanism. Continue cycling."
    elif lli_score > lam_score * 1.5:
        verdict = "LLI — Loss of Lithium Inventory"
        verdict_color = "#f6ad55"
        verdict_icon = "◑"
        verdict_body = (
            "Declining coulombic efficiency and/or near-linear capacity fade indicate "
            "that cyclable lithium is being consumed by SEI layer growth. "
            "Root cause: calendar aging, elevated temperature, or overcharge events. "
            "Mitigation: reduce SOC window, lower charge temperature, avoid prolonged full-charge storage."
        )
    elif lam_score > lli_score * 1.5:
        verdict = "LAM — Loss of Active Material"
        verdict_color = "#fc8181"
        verdict_icon = "◕"
        verdict_body = (
            "Accelerating capacity fade and/or fast resistance rise indicate electrode "
            "active sites are being lost. Root cause: high C-rate stress, mechanical particle "
            "cracking, or lithium plating followed by dead lithium formation. "
            "Mitigation: reduce peak charge/discharge rate, improve thermal management."
        )
    else:
        verdict = "Mixed LLI + LAM"
        verdict_color = "#b794f4"
        verdict_icon = "●"
        verdict_body = (
            "Both LLI and LAM mechanisms are active simultaneously. "
            "Typical of aged cells under combined calendar and cycle stress. "
            "Prioritise temperature control (targets LLI) while also reducing peak C-rate (targets LAM)."
        )

    # Confidence from number of available signals
    conf = len(confidence_notes)
    conf_label = {0: "No data", 1: "Low", 2: "Medium", 3: "High"}.get(conf, "High")
    conf_color = {"No data": "#4a5568", "Low": "#f6ad55", "Medium": "#68d391", "High": "#68d391"}.get(conf_label, "#68d391")

    return {
        "verdict":           verdict,
        "verdict_color":     verdict_color,
        "verdict_icon":      verdict_icon,
        "verdict_body":      verdict_body,
        "confidence_label":  conf_label,
        "confidence_color":  conf_color,
        "confidence_notes":  confidence_notes,
        "lli_score":         lli_score,
        "lam_score":         lam_score,
        "signals":           signals,
    }


def mechanism_corroboration_note(action: str, mechanism: dict) -> "str | None":
    """
    Cross-check the recommended action's implied urgency against the
    mechanism classifier's verdict (Decision Support review finding:
    classify() picks the action from SOH/fade/RUL/fit-scores alone -- it
    never sees diagnose_mechanism()'s output, so the two are independent
    analytical surfaces that could point in different directions with no
    arbitration between them). This does not change the recommended
    action or its confidence badge -- it only makes a real disagreement
    visible where it was previously silent, the same additive, low-risk
    pattern already used for the trajectory-match and financial-NPV
    reconciliations elsewhere on this page.

    Returns a caution string when a lower-urgency action ("continue" or
    "inspect") coincides with a mechanism verdict that indicates
    accelerating, higher-risk degradation (LAM present), at Medium+
    confidence. Returns None for the common case where nothing disagrees
    (including whenever the mechanism classifier itself has too little
    signal to be worth second-guessing the recommendation over).
    """
    if mechanism.get("confidence_label") in ("No data", "Low"):
        return None
    verdict = mechanism.get("verdict", "")
    if action in ("continue", "inspect") and "LAM" in verdict:
        return (
            f"Mechanism classifier detects {verdict} ({mechanism['confidence_label']} confidence) "
            f"— an accelerating-fade pattern inconsistent with the smooth trend this recommendation "
            f"assumes. Treat with elevated caution."
        )
    return None


def mechanism_second_life_caution(fit_scores: dict, mechanism: dict) -> "str | None":
    """
    Cross-check consequences.application_fit()'s second-life fit scores
    against the mechanism classifier's verdict (Circular Economy Coverage
    finding: application_fit()'s SOH/fade-ratio bands have no way to see
    *why* a cell is fading -- a cell scored "fit" or "marginal" for reuse
    could still be LAM-dominant, meaning its active material/particle
    structure is degraded, not just its lithium inventory. LAM's fade is
    typically less predictable going forward than LLI's -- see
    diagnose_mechanism()'s own verdict_body for the mechanistic detail --
    so a second-life deployment sized around today's SOH/fade-rate numbers
    carries more risk than those numbers alone suggest. Same additive,
    non-score-changing pattern as mechanism_corroboration_note(): this
    never changes a fit score, action, or confidence -- it only makes a
    real disagreement visible where it was previously silent.

    Returns a caution string when the best-ranked application in
    fit_scores is "fit" or "marginal" and the mechanism verdict contains
    "LAM" at Medium+ confidence. Returns None otherwise (including
    whenever the mechanism classifier has too little signal to be worth
    second-guessing the fit score over).
    """
    if mechanism.get("confidence_label") in ("No data", "Low"):
        return None
    if not fit_scores:
        return None
    _, best_app = max(
        fit_scores.items(),
        key=lambda kv: {"fit": 2, "marginal": 1, "not_fit": 0}[kv[1]["fit"]],
    )
    if best_app["fit"] not in ("fit", "marginal"):
        return None
    verdict = mechanism.get("verdict", "")
    if "LAM" not in verdict:
        return None
    return (
        f"Mechanism classifier detects {verdict} ({mechanism['confidence_label']} confidence) "
        f"— {best_app['name']} is scored {best_app['fit']} on today's SOH/fade-rate numbers, but "
        f"active-material degradation typically fades less predictably going forward than lithium-"
        f"inventory loss. Treat this fit score as more uncertain than its badge alone conveys."
    )


def physics_ml_agreement_note(agreement: dict) -> "str | None":
    """
    Caution-note wrapper around src/physics_calibration.py's
    physics_ml_agreement() — same exact contract as
    mechanism_corroboration_note() above: returns None in the common,
    unremarkable case (physics and ML agree, or either side has too little
    data to compare), and a short plain-English caution string only when
    the two independently-derived mechanism verdicts genuinely disagree.

    `agreement` is the dict returned by
    physics_calibration.physics_ml_agreement(cell_id, df) — this function
    does not recompute anything, it only decides whether that comparison
    is noteworthy enough to surface as a caution (mirroring the split
    between an always-on diagnostic and a caution-only note that
    physics_ml_agreement()'s own docstring describes).

    This is the same reused pattern as mechanism_corroboration_note(): two
    independent analytical surfaces (there, the action classifier vs. the
    mechanism classifier; here, a scipy physics fit vs. the mechanism
    classifier) that could silently diverge with no arbitration between
    them — surfaced honestly rather than resolved by picking one.
    """
    if agreement.get("agree") is not False:
        return None
    physics = agreement.get("physics", {})
    ml = agreement.get("ml", {})
    return (
        f"Physics-fitted degradation mode ({physics.get('dominant_mode_label', '?')}) disagrees "
        f"with the ML mechanism classifier's verdict ({ml.get('verdict', '?')}, "
        f"{ml.get('confidence_label', '?')} confidence) — two independently-derived signals point "
        f"different ways. Treat either mechanism call with elevated caution until they converge."
    )
