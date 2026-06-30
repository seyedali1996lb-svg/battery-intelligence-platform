"""
Battery Intelligence Copilot — Option A: Template-based narration.

The Copilot explains numbers that already exist in the model bundle.
It never calculates, estimates, or infers values not present in the data.

Entry points:
  build_cell_context(cell_id, featured_dfs, bundles) -> dict
  answer_health(ctx)              -> str
  answer_prediction_drivers(ctx)  -> str
  answer_rul(ctx)                 -> str
  answer_compare(ctx_a, ctx_b)    -> str

The context dict is the only source of truth any answer function may use.
If a value is absent from the context, it must not appear in any response.

RUL reliability is gated per cell using per_cell_rul_reliable from the
bundle — never the dataset average. B0018 (fold R²=0.22) gets an honest
explanation; B0005/B0006/B0007 (0.76–0.87) get their numbers.
"""

from __future__ import annotations

from lco_eval import RUL_RELIABLE_FLOOR

# ---------------------------------------------------------------------------
# Physical explanations for known features
# These are established battery science, not generated text.
# ---------------------------------------------------------------------------

FEATURE_PHYSICS = {
    "resistance_ohm": (
        "Internal resistance is the most direct measurable proxy for SEI "
        "(solid electrolyte interphase) layer thickness. The SEI grows on the "
        "anode with every charge/discharge cycle, consuming lithium inventory "
        "and raising impedance. Both capacity fade and resistance growth are "
        "caused by the same underlying mechanism, which is why resistance is "
        "highly predictive of state of health."
    ),
    "resistance_normalized": (
        "Normalised resistance is the ratio of current resistance to the "
        "cell's own initial resistance. Starting at 1.0 and rising, it captures "
        "relative degradation independently of the measurement method — making "
        "it comparable across cells with different absolute resistance scales."
    ),
    "resistance_trend_30cy": (
        "The 30-cycle resistance trend shows whether resistance is growing "
        "faster or slower than it was recently. Accelerating growth is an early "
        "signal that the cell is entering a more rapid degradation phase."
    ),
    "cycle_number": (
        "Cycle age is the baseline predictor — older cells have experienced "
        "more cumulative degradation. It dominates model importance when all "
        "training cells share similar operating conditions, because the model "
        "cannot distinguish them by stress history and falls back to age."
    ),
    "fade_rate_50cy": (
        "The 50-cycle capacity fade rate measures usable capacity lost per "
        "cycle, averaged over the last 50 cycles. A rising fade rate signals "
        "degradation is accelerating — the early warning sign before a steep "
        "SOH drop becomes visible."
    ),
    "fade_rate_30cy": (
        "The 30-cycle fade rate is a shorter-window view of capacity loss per "
        "cycle — more responsive to recent operating changes than the 50-cycle "
        "measure, but noisier."
    ),
    "fade_rate_10cy": (
        "The 10-cycle fade rate captures very recent capacity loss. It reacts "
        "quickly to short-term events like temperature excursions or aggressive "
        "cycling, but fluctuates more from cycle to cycle."
    ),
    "fade_acceleration": (
        "Fade acceleration is the second derivative of capacity — it measures "
        "whether the rate of fade is itself increasing. A positive value means "
        "degradation is speeding up, which is a leading indicator of "
        "approaching the rapid-fade phase seen near end of life."
    ),
    "soh_velocity_50cy": (
        "SOH velocity is the rate of health loss per cycle over a 50-cycle "
        "window. A more negative value means health is declining faster. "
        "It complements fade rate by working in percentage-of-health space "
        "rather than absolute capacity."
    ),
    "temp_rolling_30cy": (
        "The 30-cycle rolling temperature captures the cell's recent thermal "
        "operating regime. Temperature drives degradation through Arrhenius "
        "kinetics — roughly, each 10°C increase doubles the rate of SEI growth. "
        "It is most informative when cells operate at genuinely different "
        "temperatures, as in the synthetic dataset."
    ),
}

FEATURE_LABELS = {
    "cycle_number":          "Cycle age",
    "fade_rate_10cy":        "Fade rate (10-cycle)",
    "fade_rate_30cy":        "Fade rate (30-cycle)",
    "fade_rate_50cy":        "Fade rate (50-cycle)",
    "fade_acceleration":     "Fade acceleration",
    "soh_velocity_50cy":     "SOH velocity",
    "resistance_ohm":        "Internal resistance",
    "resistance_normalized": "Resistance (normalised)",
    "resistance_trend_30cy": "Resistance trend",
    "temp_rolling_30cy":     "Temperature (30-cycle avg)",
}

NASA_CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]


# ---------------------------------------------------------------------------
# Context builder — the only source of truth any answer may reference
# ---------------------------------------------------------------------------

def build_cell_context(
    cell_id: str,
    featured_dfs: dict,
    bundles: dict,
) -> dict:
    """
    Extract all grounded facts for one cell into a plain dict.

    Nothing in this dict is inferred or calculated here — every value
    comes directly from the model bundle or the featured DataFrame.
    Answer functions may only use what this dict contains.
    """
    from data_loader import CELL_STRESS_PROFILES, _stress_factor
    from model import feature_importance_df

    is_nasa = cell_id in NASA_CELL_IDS
    bundle  = bundles["nasa"] if is_nasa else bundles["synth"]

    df     = featured_dfs[cell_id]
    latest = df.iloc[-1]

    # ── SOH and status ──
    soh = float(latest["soh_pct"])
    if soh >= 90:
        status      = "Healthy"
        status_note = "above the 90% threshold for healthy operation"
    elif soh >= 80:
        status      = "Degrading"
        status_note = "between 80% and 90% — still operational but declining"
    else:
        status      = "End of Life"
        status_note = "below the 80% end-of-life threshold"

    # ── Cycle ──
    cycle = int(latest["cycle_number"])

    # ── Fade rate (30-cycle, converted to mAh/cy) ──
    fade_30 = float(latest.get("fade_rate_30cy", float("nan"))) * 1000

    # ── Resistance ──
    resistance = float(latest.get("resistance_ohm", float("nan")))

    # ── EOL ──
    eol_rows    = df[df["is_eol"]]
    eol_at      = int(eol_rows["cycle_number"].iloc[0]) if len(eol_rows) else None
    eol_reached = (eol_at is not None) and (eol_at <= cycle)

    # ── Per-cell RUL reliability — must use the individual fold, not the average ──
    per_cell_ok = bundle["metrics"].get("per_cell_rul_reliable", {})
    rul_reliable = per_cell_ok.get(
        cell_id,
        bundle["metrics"].get("rul_reliable", False),
    )

    lco_per_cell = bundle["metrics"].get("lco_per_cell", {})
    cell_fold    = lco_per_cell.get(cell_id, {})
    rul_fold_r2  = cell_fold.get("rul_r2", None)
    soh_fold_r2  = cell_fold.get("soh_r2", None)

    # RUL value is only included when reliable — absent otherwise
    rul_pred = float(latest["rul_pred"]) if rul_reliable else None

    # ── Feature importance (top 5, SOH model) ──
    fi           = feature_importance_df(bundle, "soh")
    top_features = fi.head(5).to_dict(orient="records")

    # ── Source metadata ──
    if is_nasa:
        source         = "NASA PCoE real measured data"
        operating_note = "discharged at 2A to ~2.7V cutoff at ~24°C lab conditions"
    else:
        profile        = CELL_STRESS_PROFILES.get(cell_id, {})
        sf             = _stress_factor(
            profile.get("temp_mean", 25),
            profile.get("c_rate",    1.0),
            profile.get("dod",       1.0),
        )
        source         = "physics-informed synthetic data"
        operating_note = (
            f"T={profile.get('temp_mean', 25):.0f}°C, "
            f"C-rate={profile.get('c_rate', 1.0):.1f}C, "
            f"DoD={profile.get('dod', 1.0)*100:.0f}%, "
            f"stress factor {sf:.2f}x baseline"
        )

    return {
        "cell_id":       cell_id,
        "source":        source,
        "operating_note": operating_note,
        "is_nasa":       is_nasa,
        "soh":           soh,
        "status":        status,
        "status_note":   status_note,
        "cycle":         cycle,
        "fade_30_mah_cy": fade_30,
        "resistance_ohm": resistance,
        "eol_at":        eol_at,
        "eol_reached":   eol_reached,
        "rul_reliable":  rul_reliable,
        "rul_pred":      rul_pred,
        "rul_fold_r2":   rul_fold_r2,
        "soh_fold_r2":   soh_fold_r2,
        "lco_soh_r2":    bundle["metrics"].get("lco_soh_r2"),
        "top_features":  top_features,
    }


def context_summary(ctx: dict) -> str:
    """Plain-text summary of what the context contains — shown in the UI expander."""
    lines = [
        f"Cell:             {ctx['cell_id']}",
        f"Source:           {ctx['source']}",
        f"Operating:        {ctx['operating_note']}",
        f"SOH:              {ctx['soh']:.1f}%  ({ctx['status']})",
        f"Cycle:            {ctx['cycle']}",
        f"Fade rate (30cy): {ctx['fade_30_mah_cy']:.3f} mAh/cycle",
        f"Resistance:       {ctx['resistance_ohm']:.4f} ohm",
        f"EOL reached:      {ctx['eol_reached']}  (at cycle {ctx['eol_at']})" if ctx['eol_at'] else "EOL reached:      no",
        f"RUL reliable:     {ctx['rul_reliable']}  (fold R²={ctx['rul_fold_r2']:.2f})" if ctx['rul_fold_r2'] is not None else f"RUL reliable:     {ctx['rul_reliable']}",
        f"RUL estimate:     {ctx['rul_pred']:.0f} cycles" if ctx['rul_pred'] is not None else "RUL estimate:     withheld (not reliable)",
        f"SOH fold R²:      {ctx['soh_fold_r2']:.3f}" if ctx['soh_fold_r2'] is not None else "SOH fold R²:      n/a",
    ]
    lines.append("Top SOH drivers:")
    for f in ctx["top_features"]:
        label = FEATURE_LABELS.get(f["feature"], f["feature"])
        lines.append(f"  {label}: {f['importance_pct']:.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer functions — each takes a context dict, returns a markdown string.
# No value may appear in the output that is not present in the context dict.
# ---------------------------------------------------------------------------

def answer_health(ctx: dict) -> str:
    """Explain the cell's current health state."""
    cell  = ctx["cell_id"]
    soh   = ctx["soh"]
    cycle = ctx["cycle"]
    fade  = ctx["fade_30_mah_cy"]

    if ctx["eol_reached"]:
        eol_line = (
            f"{cell} crossed the end-of-life threshold at cycle {ctx['eol_at']} "
            f"and is now {cycle - ctx['eol_at']} cycles past it."
        )
    elif ctx["eol_at"]:
        remaining = ctx["eol_at"] - cycle
        eol_line = (
            f"Based on the observed fade trajectory, the end-of-life threshold "
            f"is projected at cycle {ctx['eol_at']} — approximately {remaining} cycles from now."
        )
    else:
        eol_line = (
            "End of life has not been reached and is not projected within "
            "the current data window."
        )

    top_driver = ""
    if ctx["top_features"]:
        f0    = ctx["top_features"][0]
        name  = FEATURE_LABELS.get(f0["feature"], f0["feature"])
        top_driver = (
            f"The model's strongest signal for this assessment is "
            f"**{name}**, which accounts for {f0['importance_pct']:.0f}% "
            f"of the SOH prediction."
        )

    source_note = f"{ctx['source']} ({ctx['operating_note']})"

    return (
        f"**{cell}** is currently at **{soh:.1f}% state of health** — "
        f"{ctx['status_note']}.\n\n"
        f"This is based on cycle {cycle} of {source_note}. "
        f"Over the last 30 cycles, the cell has been losing approximately "
        f"**{fade:.2f} mAh of capacity per cycle**.\n\n"
        f"{eol_line}\n\n"
        f"{top_driver}"
    ).strip()


def answer_prediction_drivers(ctx: dict) -> str:
    """Explain why the model predicts this SOH — feature by feature."""
    cell = ctx["cell_id"]

    if not ctx["top_features"]:
        return f"Feature importance data is not available for {cell}."

    soh_fold = ctx["soh_fold_r2"]
    lco_r2   = ctx["lco_soh_r2"]

    if soh_fold is not None:
        accuracy_note = (
            f"When validated with leave-cell-out cross-validation — training on all "
            f"other cells and testing on {cell} alone — the SOH model achieved "
            f"**R²={soh_fold:.3f}**, meaning it explained "
            f"{soh_fold * 100:.0f}% of the SOH variance for this specific cell."
        )
    elif lco_r2 is not None:
        accuracy_note = (
            f"Across all cells in the dataset, the SOH model's leave-cell-out "
            f"accuracy is R²={lco_r2:.3f}."
        )
    else:
        accuracy_note = ""

    lines = [
        f"The SOH model's prediction for **{cell}** is driven by these signals, "
        f"ranked by how much each one contributed to reducing prediction error:\n"
    ]

    for i, f in enumerate(ctx["top_features"], 1):
        name    = FEATURE_LABELS.get(f["feature"], f["feature"])
        pct     = f["importance_pct"]
        physics = FEATURE_PHYSICS.get(f["feature"], "")
        lines.append(f"**{i}. {name} — {pct:.1f}%**")
        if physics:
            lines.append(physics)
        lines.append("")

    if accuracy_note:
        lines.append(accuracy_note)

    return "\n".join(lines).strip()


def answer_rul(ctx: dict) -> str:
    """
    Explain remaining useful life — or explain honestly why it isn't shown.
    This is the most safety-critical response: if rul_reliable is False,
    no cycle estimate is generated under any circumstances.
    """
    cell = ctx["cell_id"]

    # ── Not reliable — full honest explanation ──
    if not ctx["rul_reliable"]:
        fold_r2 = ctx["rul_fold_r2"]
        r2_str  = f"R²={fold_r2:.2f}" if fold_r2 is not None else "below the reliability threshold"
        return (
            f"**A reliable remaining-life estimate is not available for {cell}.**\n\n"
            f"During leave-cell-out validation — where the model was trained on all "
            f"other cells and tested on {cell} alone — the RUL model achieved "
            f"**{r2_str}** for this cell. The reliability threshold is R²>={RUL_RELIABLE_FLOOR}.\n\n"
            f"An R² below this floor means the model's remaining-life predictions for "
            f"{cell} were less accurate than simply guessing the average remaining "
            f"life across similar cells. Displaying a cycle count would create false "
            f"confidence in a number the model cannot support.\n\n"
            f"This does not mean {cell}'s data is wrong — it means the model has not "
            f"seen enough cells with a similar degradation profile to generalise "
            f"reliably to this one. As more cells are added to the training set, "
            f"this may change."
        )

    # ── Reliable — give the number with appropriate context ──
    rul  = ctx["rul_pred"]
    fold = ctx["rul_fold_r2"]

    confidence_note = (
        f" (model fold R²={fold:.2f} on this cell)" if fold is not None else ""
    )

    if ctx["eol_reached"]:
        return (
            f"**{cell} has already reached end of life**, crossing the 80% SOH "
            f"threshold at cycle {ctx['eol_at']}. It is currently at cycle {ctx['cycle']}.\n\n"
            f"The model's remaining-life estimate is {rul:.0f} cycles, which at this "
            f"stage reflects how much further below EOL the cell continues to operate — "
            f"not a forward-looking projection{confidence_note}."
        )

    return (
        f"**{cell}** has an estimated **{rul:.0f} cycles of useful life remaining** "
        f"before reaching the 80% SOH end-of-life threshold{confidence_note}.\n\n"
        f"This estimate is based on the cell's current capacity fade rate and "
        f"resistance trajectory at cycle {ctx['cycle']}. It assumes operating "
        f"conditions remain similar to those observed so far — significant changes "
        f"in temperature, charge rate, or depth of discharge would shift the actual "
        f"remaining life."
    )


def answer_compare(ctx_a: dict, ctx_b: dict) -> str:
    """Side-by-side comparison of two cells on health, fade, RUL, and top driver."""
    a = ctx_a["cell_id"]
    b = ctx_b["cell_id"]

    # ── SOH comparison ──
    soh_diff  = ctx_a["soh"] - ctx_b["soh"]
    better    = a if soh_diff > 0 else b
    worse     = b if soh_diff > 0 else a
    ctx_worse = ctx_b if soh_diff > 0 else ctx_a
    diff_abs  = abs(soh_diff)

    lines = [
        f"**{a}** ({ctx_a['soh']:.1f}% SOH · {ctx_a['status']}) vs "
        f"**{b}** ({ctx_b['soh']:.1f}% SOH · {ctx_b['status']})\n",

        f"**{better}** is in better condition by **{diff_abs:.1f} percentage points**. "
        f"**{worse}** is at {ctx_worse['soh']:.1f}% — {ctx_worse['status_note']}.\n",
    ]

    # ── Fade rate ──
    fa, fb       = ctx_a["fade_30_mah_cy"], ctx_b["fade_30_mah_cy"]
    faster       = a if fa > fb else b
    fade_diff    = abs(fa - fb)
    lines.append(
        f"**Degradation rate:** {a} is losing {fa:.2f} mAh/cycle vs "
        f"{fb:.2f} mAh/cycle for {b} (30-cycle rolling average). "
        f"**{faster}** is currently degrading faster by {fade_diff:.2f} mAh/cycle.\n"
    )

    # ── Data source note when mixing types ──
    if ctx_a["is_nasa"] != ctx_b["is_nasa"]:
        nasa_cell  = a if ctx_a["is_nasa"] else b
        synth_cell = b if ctx_a["is_nasa"] else a
        lines.append(
            f"**Source note:** {nasa_cell} is real measured data (NASA PCoE); "
            f"{synth_cell} is physics-informed synthetic data. "
            f"SOH % is directly comparable — both are relative to each cell's own "
            f"initial capacity. Absolute resistance values are not comparable "
            f"across measurement methods.\n"
        )

    # ── RUL ──
    def rul_line(ctx: dict) -> str:
        cid = ctx["cell_id"]
        if ctx["rul_reliable"] and ctx["rul_pred"] is not None:
            fold = ctx["rul_fold_r2"]
            note = f" (fold R²={fold:.2f})" if fold is not None else ""
            return f"{cid}: ~{ctx['rul_pred']:.0f} cycles remaining{note}"
        else:
            fold   = ctx["rul_fold_r2"]
            r2_str = f"fold R²={fold:.2f}" if fold is not None else "below reliability floor"
            return f"{cid}: not calibrated ({r2_str})"

    lines.append(
        f"**Remaining life:** {rul_line(ctx_a)} · {rul_line(ctx_b)}\n"
    )

    # ── Top driver comparison ──
    if ctx_a["top_features"] and ctx_b["top_features"]:
        f0a   = ctx_a["top_features"][0]
        f0b   = ctx_b["top_features"][0]
        na    = FEATURE_LABELS.get(f0a["feature"], f0a["feature"])
        nb    = FEATURE_LABELS.get(f0b["feature"], f0b["feature"])
        if f0a["feature"] == f0b["feature"]:
            lines.append(
                f"**Top driver (both cells):** {na} — "
                f"{f0a['importance_pct']:.0f}% for {a}, {f0b['importance_pct']:.0f}% for {b}. "
                f"Both models are responding to the same primary signal."
            )
        else:
            lines.append(
                f"**Top drivers differ:** {na} leads for {a} "
                f"({f0a['importance_pct']:.0f}%); {nb} leads for {b} "
                f"({f0b['importance_pct']:.0f}%). This reflects different operating "
                f"histories, training data, or measurement scales between the two cells."
            )

    return "\n".join(lines).strip()
