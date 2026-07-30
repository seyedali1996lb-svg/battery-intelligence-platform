"""
Battery Intelligence Copilot — Option A: Template-based narration.

The Copilot explains numbers that already exist in the model bundle.
It never calculates, estimates, or infers values not present in the data.

Cell-level entry points:
  build_cell_context(cell_id, featured_dfs, bundles) -> dict
  answer_health(ctx)                        -> str
  answer_prediction_drivers(ctx)            -> str
  answer_rul(ctx)                           -> str
  answer_compare(ctx_a, ctx_b)              -> str
  answer_anomaly(ctx, fleet_stats)          -> str
  answer_recent_trajectory(ctx, df)         -> str
  answer_fleet_compare(ctx, fleet_stats)    -> str

Fleet-level entry points:
  build_fleet_stats(featured_dfs, bundles)  -> dict
  answer_alerts(fleet_stats)                -> str

Navigation helpers:
  QUERY_LABELS   — display label for each query key
  FOLLOW_UP_MAP  — dict[query_key, list[query_key]] of suggested next questions

The context dict is the only source of truth any answer function may use.
Values absent from it cannot appear in any response.

RUL reliability is gated per cell using per_cell_rul_reliable from the
bundle — never the dataset average. B0018 (fold R2=0.22) gets an honest
explanation; B0005/B0006/B0007 (0.76-0.87) get their numbers.
"""

from __future__ import annotations

import math
import numpy as np

from batlab.validation.lco import RUL_RELIABLE_FLOOR
from copilot_templates import TEMPLATES as _T
from battery_knowledge import get_feature_citation
from chemistry_profiles import ChemistryProfile

# ---------------------------------------------------------------------------
# Navigation constants
# ---------------------------------------------------------------------------

QUERY_LABELS: dict[str, str] = {
    "health":              "What is this cell's health?",
    "drivers":             "Why does the model predict this SOH?",
    "rul":                 "How much life is left?",
    "compare":             "Compare to another cell",
    "recent":              "What happened in the last 20 cycles?",
    "anomaly":             "Is this cell behaving unusually?",
    "fleet_compare":       "How does this cell rank in the fleet?",
    "mechanism":           "What degradation mechanism is this cell showing?",
    "alerts":              "Fleet alert summary",
    # Business queries
    "replacement_budget":  "What will replacement cost over the next 12 months?",
    "fleet_risk":          "What is the business risk in my fleet?",
    "business_case":       "Should I replace or repurpose these cells?",
}

# Suggested follow-up queries after each answer — cell-agnostic queries like
# "alerts" only appear as follow-ups, not as mandatory first choices.
FOLLOW_UP_MAP: dict[str, list[str]] = {
    "health":             ["drivers", "mechanism",       "recent"],
    "drivers":            ["health",  "anomaly",         "recent"],
    "rul":                ["health",  "fleet_compare",   "compare"],
    "compare":            ["health",  "fleet_compare",   "rul"],
    "recent":             ["anomaly", "drivers",         "rul"],
    "anomaly":            ["recent",  "drivers",         "fleet_compare"],
    "fleet_compare":      ["health",  "rul",             "anomaly"],
    "mechanism":          ["health",  "drivers",         "recent"],
    "alerts":             ["fleet_compare", "health",   "rul"],
    "replacement_budget": ["fleet_risk", "business_case", "alerts"],
    "fleet_risk":         ["replacement_budget", "business_case", "alerts"],
    "business_case":      ["replacement_budget", "fleet_risk", "rul"],
}

# ---------------------------------------------------------------------------
# Physical explanations for known features
# These are established battery science, not generated text.
# ---------------------------------------------------------------------------

FEATURE_PHYSICS: dict[str, str] = {
    "resistance_ohm": (
        "Internal resistance is the most direct measurable proxy for SEI "
        "(solid electrolyte interphase) layer thickness. The SEI grows on the "
        "anode with every charge/discharge cycle, consuming lithium inventory "
        "and raising impedance. Both capacity fade and resistance growth share "
        "the same underlying mechanism, which is why resistance is highly "
        "predictive of state of health."
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
        "degradation is speeding up, which is a leading indicator of approaching "
        "the rapid-fade phase seen near end of life."
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
        "kinetics — roughly, each 10 degrees C increase doubles the rate of "
        "SEI growth. It is most informative when cells operate at genuinely "
        "different temperatures, as in the synthetic dataset."
    ),
}

FEATURE_LABELS: dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Cell context builder — the only source of truth any answer may reference
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
    from batlab.models.gbrt import feature_importance_df

    # Was an is_nasa/CELL_STRESS_PROFILES-membership/else-upload chain with no
    # Severson case at all -- a Severson cell fell to the "else" branch, which
    # picked bundles["upload"] (or bundles["synth"] as its own fallback) for
    # RUL-reliability/SHAP lookups, and later told the user "Source: user-
    # uploaded data" for a cell nobody uploaded. ChemistryProfile.for_cell()
    # gives every real dataset its own bucket.
    source_kind = ChemistryProfile.for_cell(cell_id).source_kind
    is_nasa     = source_kind == "nasa"   # kept for backward compat with callers below

    bundle = bundles.get(source_kind)
    if bundle is None:
        bundle = bundles.get("nasa") or next(iter(bundles.values()))

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

    cycle    = int(latest["cycle_number"])
    fade_30  = float(latest.get("fade_rate_30cy", float("nan"))) * 1000
    resistance = float(latest.get("resistance_ohm", float("nan")))

    eol_rows    = df[df["is_eol"]]
    eol_at      = int(eol_rows["cycle_number"].iloc[0]) if len(eol_rows) else None
    eol_reached = (eol_at is not None) and (eol_at <= cycle)

    # Per-cell RUL reliability — individual fold, not the dataset average
    per_cell_ok  = bundle["metrics"].get("per_cell_rul_reliable", {})
    rul_reliable = per_cell_ok.get(cell_id, bundle["metrics"].get("rul_reliable", False))

    lco_per_cell = bundle["metrics"].get("lco_per_cell", {})
    cell_fold    = lco_per_cell.get(cell_id, {})
    rul_fold_r2  = cell_fold.get("rul_r2", None)
    soh_fold_r2  = cell_fold.get("soh_r2", None)
    rul_pred     = float(latest["rul_pred"]) if rul_reliable else None
    rul_q10      = float(latest["rul_q10"])  if (rul_reliable and "rul_q10" in latest.index) else None
    rul_q90      = float(latest["rul_q90"])  if (rul_reliable and "rul_q90" in latest.index) else None

    fi           = feature_importance_df(bundle, "soh")
    top_features = fi.head(5).to_dict(orient="records")

    if source_kind == "nasa":
        source         = "NASA PCoE real measured data"
        operating_note = "discharged at 2A to ~2.7V cutoff at ~24 C lab conditions"
    elif source_kind == "severson":
        source         = "Severson et al. 2019 (Nature Energy) real measured data"
        operating_note = "fast-charging protocol search, various C-rates, 30 C chamber"
    elif source_kind == "upload":
        source         = "user-uploaded data"
        operating_note = "operating conditions from uploaded dataset"
    else:
        profile        = CELL_STRESS_PROFILES.get(cell_id, {})
        sf             = _stress_factor(
            profile.get("temp_mean", 25),
            profile.get("c_rate",    1.0),
            profile.get("dod",       1.0),
        )
        source         = "physics-informed synthetic data"
        operating_note = (
            f"T={profile.get('temp_mean', 25):.0f} C, "
            f"C-rate={profile.get('c_rate', 1.0):.1f}C, "
            f"DoD={profile.get('dod', 1.0)*100:.0f}%, "
            f"stress factor {sf:.2f}x baseline"
        )

    data_mode = source_kind

    return {
        "cell_id":        cell_id,
        "source":         source,
        "operating_note": operating_note,
        "data_mode":      data_mode,
        "is_nasa":        is_nasa,
        "source_kind":    source_kind,
        "soh":            soh,
        "status":         status,
        "status_note":    status_note,
        "cycle":          cycle,
        "fade_30_mah_cy": fade_30,
        "resistance_ohm": resistance,
        "eol_at":         eol_at,
        "eol_reached":    eol_reached,
        "rul_reliable":   rul_reliable,
        "rul_pred":       rul_pred,
        "rul_q10":        rul_q10,
        "rul_q90":        rul_q90,
        "rul_fold_r2":    rul_fold_r2,
        "soh_fold_r2":    soh_fold_r2,
        "lco_soh_r2":     bundle["metrics"].get("lco_soh_r2"),
        "top_features":   top_features,
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
        (f"EOL reached:      {ctx['eol_reached']}  (at cycle {ctx['eol_at']})"
         if ctx["eol_at"] else "EOL reached:      no"),
        (f"RUL reliable:     {ctx['rul_reliable']}  (fold R2={ctx['rul_fold_r2']:.2f})"
         if ctx["rul_fold_r2"] is not None else f"RUL reliable:     {ctx['rul_reliable']}"),
        (f"RUL estimate:     {ctx['rul_pred']:.0f} cycles"
         if ctx["rul_pred"] is not None else "RUL estimate:     withheld (not reliable)"),
        (f"SOH fold R2:      {ctx['soh_fold_r2']:.3f}"
         if ctx["soh_fold_r2"] is not None else "SOH fold R2:      n/a"),
    ]
    lines.append("Top SOH drivers:")
    for f in ctx["top_features"]:
        label = FEATURE_LABELS.get(f["feature"], f["feature"])
        lines.append(f"  {label}: {f['importance_pct']:.1f}%")
    return "\n".join(lines)


def answer_query(query: str, ctx: dict) -> str:
    """
    Free-text query entry point. Grounds the answer in the cell's own
    context (context_summary) and augments it with retrieved battery-
    domain knowledge (src/copilot_retrieval.py) relevant to the question —
    real retrieval over an authored corpus, not keyword-to-template
    routing. Returns a plain-text answer usable directly, or as the
    grounding template passed to this module's own llm_answer() for rephrasing.
    """
    from copilot_retrieval import retrieve

    base = context_summary(ctx)
    related = retrieve(query, top_k=2)
    if not related:
        return base
    related_block = "\n\n".join(related)
    return f"{base}\n\nRelated background:\n{related_block}"


# ---------------------------------------------------------------------------
# Fleet stats builder — aggregates across all cells for anomaly & fleet answers
# ---------------------------------------------------------------------------

def build_fleet_stats(org_id: int, featured_dfs: dict, bundles: dict) -> dict:
    """
    Compute fleet-level aggregates. Reads precomputed CellSummary rows
    (src/cell_store.py's build_summary()) instead of every cell's full
    per-cycle DataFrame -- see that module's docstring. featured_dfs is
    still used to scope the read to the currently active data-mode fleet
    (its .keys() list cell_ids without touching any full DataFrame).

    Source-split stats (nasa vs synth) are kept separate because resistance
    scales are incompatible across measurement methods.
    """
    import db as _db_fs

    active_ids = set(featured_dfs.keys())
    summaries_by_id = {
        r["cell_id"]: r for r in _db_fs.get_cell_summaries(org_id) if r["cell_id"] in active_ids
    }

    rows: list[dict] = []

    for cell_id in featured_dfs.keys():
        summary = summaries_by_id.get(cell_id)
        if summary is None:
            continue
        # Same fix as build_cell_context() above -- Severson cells used to
        # fall through to the "uploaded" bucket here too, both picking the
        # wrong model bundle and grouping them with the wrong peers in
        # answer_anomaly()/answer_fleet_compare() below.
        source = ChemistryProfile.for_cell(cell_id).source_kind
        bundle = bundles.get(source)

        if bundle is None:
            continue
        per_cell_ok  = bundle["metrics"].get("per_cell_rul_reliable", {})
        rul_reliable = per_cell_ok.get(cell_id, bundle["metrics"].get("rul_reliable", False))

        soh      = summary.get("soh_pct")
        if soh is None:
            continue
        fade_30  = (summary.get("fade_rate_30cy") or float("nan")) * 1000
        resist   = summary.get("resistance_ohm")
        resist   = float(resist) if resist is not None else float("nan")
        cycle    = summary["cycle_number"]
        rul_pred = summary.get("rul_pred") if rul_reliable else None

        rows.append({
            "cell_id":     cell_id,
            "source":      source,
            "soh":         float(soh),
            "cycle":       cycle,
            "fade_30":     fade_30,
            "resistance":  resist,
            "rul_reliable": rul_reliable,
            "rul_pred":    rul_pred,
        })

    def _mean(vals):
        clean = [v for v in vals if not math.isnan(v)]
        return float(np.mean(clean)) if clean else float("nan")

    sohs  = [r["soh"]    for r in rows]
    fades = [r["fade_30"] for r in rows]

    eol_cells       = [r["cell_id"] for r in rows if r["soh"] < 80]
    degrading_cells = [r["cell_id"] for r in rows if 80 <= r["soh"] < 90]
    unreliable_rul  = [r["cell_id"] for r in rows if not r["rul_reliable"]]

    sorted_by_soh  = sorted(rows, key=lambda r: r["soh"])
    sorted_by_fade = sorted(
        rows,
        key=lambda r: r["fade_30"] if not math.isnan(r["fade_30"]) else -1,
        reverse=True,
    )

    return {
        "rows":            rows,
        "n_cells":         len(rows),
        "soh_mean":        _mean(sohs),
        "soh_median":      float(np.median(sohs)),
        "soh_min":         float(min(sohs)) if sohs else float("nan"),
        "soh_max":         float(max(sohs)) if sohs else float("nan"),
        "fade_mean":       _mean(fades),
        "eol_cells":       eol_cells,
        "degrading_cells": degrading_cells,
        "sorted_by_soh":   sorted_by_soh,   # worst first
        "sorted_by_fade":  sorted_by_fade,  # fastest first
        "unreliable_rul":  unreliable_rul,
    }


# ---------------------------------------------------------------------------
# Cell-level answer functions
# ---------------------------------------------------------------------------

def answer_health(ctx: dict) -> str:
    """Explain the cell's current health state."""
    cell  = ctx["cell_id"]
    cycle = ctx["cycle"]

    intro = _T["health_intro"].format_map({
        "cell": cell, "soh": ctx["soh"], "status_note": ctx["status_note"],
        "cycle": cycle, "source": ctx["source"],
        "operating_note": ctx["operating_note"], "fade": ctx["fade_30_mah_cy"],
    })

    if ctx["eol_reached"]:
        eol_line = _T["health_eol_reached"].format_map(
            {"cell": cell, "eol_at": ctx["eol_at"], "past_eol": cycle - ctx["eol_at"]}
        )
    elif ctx["eol_at"]:
        eol_line = _T["health_eol_projected"].format_map(
            {"eol_at": ctx["eol_at"], "remaining": ctx["eol_at"] - cycle}
        )
    else:
        eol_line = _T["health_eol_none"]

    top_driver = ""
    if ctx["top_features"]:
        f0   = ctx["top_features"][0]
        name = FEATURE_LABELS.get(f0["feature"], f0["feature"])
        top_driver = _T["health_top_driver"].format_map(
            {"name": name, "pct": f0["importance_pct"]}
        )

    return f"{intro}\n\n{eol_line}\n\n{top_driver}".strip()


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
            f"**R2={soh_fold:.3f}**, explaining {soh_fold * 100:.0f}% of SOH variance "
            f"for this specific cell."
        )
    elif lco_r2 is not None:
        accuracy_note = (
            f"Across all cells in the dataset, the SOH model's leave-cell-out "
            f"accuracy is R2={lco_r2:.3f}."
        )
    else:
        accuracy_note = ""

    lines = [
        f"The SOH model's prediction for **{cell}** is driven by these signals, "
        f"ranked by contribution to prediction accuracy:\n"
    ]
    for i, f in enumerate(ctx["top_features"], 1):
        name     = FEATURE_LABELS.get(f["feature"], f["feature"])
        pct      = f["importance_pct"]
        physics  = FEATURE_PHYSICS.get(f["feature"], "")
        citation = get_feature_citation(f["feature"])
        lines.append(f"**{i}. {name} — {pct:.1f}%**")
        if physics:
            lines.append(physics)
        if citation:
            lines.append(f"*Source: {citation['title']} (doi:{citation['doi']}) — {citation['relevance']}*")
        lines.append("")

    if accuracy_note:
        lines.append(accuracy_note)

    return "\n".join(lines).strip()


def answer_rul(ctx: dict) -> str:
    """
    Explain remaining useful life — or explain honestly why it is not shown.
    If rul_reliable is False, no cycle estimate is generated under any circumstances.
    Includes 80% prediction interval (Q10–Q90) when quantile models are available.
    """
    cell = ctx["cell_id"]

    if not ctx["rul_reliable"]:
        fold_r2 = ctx["rul_fold_r2"]
        r2_str  = f"R²={fold_r2:.2f}" if fold_r2 is not None else "below the reliability threshold"
        return _T["rul_not_calibrated"].format_map(
            {"cell": cell, "r2_str": r2_str, "floor": RUL_RELIABLE_FLOOR}
        )

    rul  = ctx["rul_pred"]
    fold = ctx["rul_fold_r2"]
    conf = f" (fold R²={fold:.2f} on this cell)" if fold is not None else ""

    if ctx["eol_reached"]:
        body = _T["rul_eol_reached"].format_map(
            {"cell": cell, "eol_at": ctx["eol_at"],
             "cycle": ctx["cycle"], "rul": rul, "conf": conf}
        )
    else:
        body = _T["rul_normal"].format_map(
            {"cell": cell, "rul": rul, "conf": conf, "cycle": ctx["cycle"]}
        )

    # Append prediction interval note if available
    q10, q90 = ctx.get("rul_q10"), ctx.get("rul_q90")
    if q10 is not None and q90 is not None and q90 > q10:
        interval_note = "\n\n" + _T["rul_interval_note"].format_map(
            {"q10": q10, "q90": q90}
        )
    else:
        interval_note = ""

    return body + interval_note


def answer_mechanism(ctx: dict, mechanism: "dict | None", literature: "list[dict] | None" = None) -> str:
    """
    Explain this cell's degradation-mechanism verdict (LLI vs LAM).

    `mechanism` is the Battery Knowledge Graph's exhibits edge for this cell
    (src/knowledge_graph.py's get_or_compute_mechanism()) — the same shared
    verdict Health's compact card and the Cell Workbench's Decide & Ask view
    read, not an independent recomputation. Before this function existed,
    the Copilot had no mechanism awareness at all — a real gap: Health and
    Decide & Ask both surfaced the LLI/LAM verdict, but asking the Copilot
    about it returned nothing.

    `literature` (optional) is the list of corroborating LiteratureSource
    entries from knowledge_graph.literature_for_mechanism() — appended as
    grounding background, same pattern as answer_query()'s retrieval block.
    """
    cell = ctx["cell_id"]

    if not mechanism:
        return (
            f"No degradation-mechanism verdict is available yet for {cell} — "
            f"it may not be part of the platform's tracked fleet."
        )

    verdict = mechanism.get("verdict", "Insufficient data")
    conf    = mechanism.get("confidence_label", "No data")

    if verdict == "Insufficient data" or conf == "No data":
        return (
            f"**{cell}** does not yet have enough degradation signal to classify a dominant "
            f"mechanism — this needs coulombic-efficiency, SOH-trend, and/or resistance-trend "
            f"data (at least 10-20 cycles of history). Continue cycling to accumulate signal."
        )

    lli = mechanism.get("lli_score", 0)
    lam = mechanism.get("lam_score", 0)
    notes = mechanism.get("confidence_notes", [])
    body  = mechanism.get("verdict_body", "")

    lines = [
        f"**{cell}**'s dominant degradation mechanism is **{verdict}** "
        f"({conf} confidence — LLI score {lli}, LAM score {lam}, "
        f"from {', '.join(notes) if notes else 'no'} signal(s)).",
        body,
    ]
    if literature:
        lines.append("**Related background:**")
        for doc in literature:
            lines.append(f"- {doc.get('text', '')}")
    return "\n\n".join(l for l in lines if l)


def answer_compare(ctx_a: dict, ctx_b: dict) -> str:
    """Side-by-side comparison of two cells on health, fade, RUL, and top driver."""
    a = ctx_a["cell_id"]
    b = ctx_b["cell_id"]

    soh_diff  = ctx_a["soh"] - ctx_b["soh"]
    better    = a if soh_diff > 0 else b
    worse     = b if soh_diff > 0 else a
    ctx_worse = ctx_b if soh_diff > 0 else ctx_a
    diff_abs  = abs(soh_diff)

    lines = [
        f"**{a}** ({ctx_a['soh']:.1f}% SOH · {ctx_a['status']}) vs "
        f"**{b}** ({ctx_b['soh']:.1f}% SOH · {ctx_b['status']})\n",
        f"**{better}** is in better condition by **{diff_abs:.1f} pp**. "
        f"**{worse}** is at {ctx_worse['soh']:.1f}% — {ctx_worse['status_note']}.\n",
    ]

    fa, fb     = ctx_a["fade_30_mah_cy"], ctx_b["fade_30_mah_cy"]
    faster     = a if fa > fb else b
    fade_diff  = abs(fa - fb)
    lines.append(
        f"**Degradation rate:** {a} is losing {fa:.2f} mAh/cycle vs "
        f"{fb:.2f} mAh/cycle for {b} (30-cycle rolling average). "
        f"**{faster}** is currently degrading faster by {fade_diff:.2f} mAh/cycle.\n"
    )

    if ctx_a["source_kind"] != ctx_b["source_kind"]:
        lines.append(
            f"**Source note:** {a} is {ctx_a['source']}; {b} is {ctx_b['source']}. "
            f"SOH % is directly comparable — both are relative to each cell's own "
            f"initial capacity. Absolute resistance values are not comparable across "
            f"measurement methods or model training boundaries.\n"
        )

    def _rul_str(ctx: dict) -> str:
        if ctx["rul_reliable"] and ctx["rul_pred"] is not None:
            fold = ctx["rul_fold_r2"]
            note = f" (fold R2={fold:.2f})" if fold is not None else ""
            return f"{ctx['cell_id']}: ~{ctx['rul_pred']:.0f} cycles remaining{note}"
        fold   = ctx["rul_fold_r2"]
        r2_str = f"fold R2={fold:.2f}" if fold is not None else "below reliability floor"
        return f"{ctx['cell_id']}: not calibrated ({r2_str})"

    lines.append(f"**Remaining life:** {_rul_str(ctx_a)} · {_rul_str(ctx_b)}\n")

    if ctx_a["top_features"] and ctx_b["top_features"]:
        f0a = ctx_a["top_features"][0]
        f0b = ctx_b["top_features"][0]
        na  = FEATURE_LABELS.get(f0a["feature"], f0a["feature"])
        nb  = FEATURE_LABELS.get(f0b["feature"], f0b["feature"])
        if f0a["feature"] == f0b["feature"]:
            lines.append(
                f"**Top driver (both cells):** {na} — "
                f"{f0a['importance_pct']:.0f}% for {a}, {f0b['importance_pct']:.0f}% for {b}. "
                f"Both models respond to the same primary signal."
            )
        else:
            lines.append(
                f"**Top drivers differ:** {na} leads for {a} ({f0a['importance_pct']:.0f}%); "
                f"{nb} leads for {b} ({f0b['importance_pct']:.0f}%). This reflects different "
                f"operating histories, training data, or measurement scales."
            )

    return "\n".join(lines).strip()


def answer_anomaly(ctx: dict, fleet_stats: dict) -> str:
    """
    Is this cell behaving unusually compared to its peers?
    Peers are cells from the same data source (nasa/severson/synth/upload) to
    avoid the resistance-scale incompatibility across measurement methods.
    """
    cell   = ctx["cell_id"]
    source = ctx["source_kind"]

    peers = [r for r in fleet_stats["rows"] if r["source"] == source and r["cell_id"] != cell]

    if not peers:
        return (
            f"There are no other cells of the same type ({ctx['source']}) to compare "
            f"{cell} against. Add more real NASA cells or synthetic cells to enable "
            f"anomaly detection within this data source."
        )

    peer_sohs  = [r["soh"]    for r in peers]
    peer_fades = [r["fade_30"] for r in peers if not math.isnan(r["fade_30"])]

    peer_soh_mean  = float(np.mean(peer_sohs))
    peer_soh_std   = float(np.std(peer_sohs)) if len(peer_sohs) > 1 else 0.0
    soh_diff       = ctx["soh"] - peer_soh_mean

    anomalies: list[str] = []
    normals:   list[str] = []

    # SOH vs peers
    soh_threshold = max(5.0, peer_soh_std * 1.5)
    if abs(soh_diff) >= soh_threshold:
        direction = "higher" if soh_diff > 0 else "lower"
        anomalies.append(
            f"SOH is {abs(soh_diff):.1f} pp {direction} than the peer mean "
            f"({ctx['soh']:.1f}% vs {peer_soh_mean:.1f}%) — "
            f"{'unusually good' if soh_diff > 0 else 'unusually poor'}"
        )
    else:
        normals.append(
            f"SOH ({ctx['soh']:.1f}%) is within normal range of peers "
            f"(peer mean {peer_soh_mean:.1f}%, diff {soh_diff:+.1f} pp)"
        )

    # Fade rate vs peers
    if peer_fades and not math.isnan(ctx["fade_30_mah_cy"]):
        peer_fade_mean = float(np.mean(peer_fades))
        peer_fade_std  = float(np.std(peer_fades)) if len(peer_fades) > 1 else 0.0
        fade_ratio     = ctx["fade_30_mah_cy"] / peer_fade_mean if peer_fade_mean > 0 else 1.0

        fade_threshold = max(1.5, 1.0 + peer_fade_std / peer_fade_mean if peer_fade_mean > 0 else 1.5)
        if fade_ratio >= fade_threshold:
            anomalies.append(
                f"Degrading {fade_ratio:.1f}x faster than peers "
                f"({ctx['fade_30_mah_cy']:.2f} vs {peer_fade_mean:.2f} mAh/cycle) — "
                f"flag for inspection"
            )
        elif fade_ratio <= 1 / fade_threshold:
            anomalies.append(
                f"Degrading {1/fade_ratio:.1f}x slower than peers "
                f"({ctx['fade_30_mah_cy']:.2f} vs {peer_fade_mean:.2f} mAh/cycle) — "
                f"unusually well-preserved"
            )
        else:
            normals.append(
                f"Fade rate ({ctx['fade_30_mah_cy']:.2f} mAh/cycle) is within "
                f"normal range of peers (peer mean {peer_fade_mean:.2f} mAh/cycle)"
            )

    # Compose
    n_peers = len(peers)
    peer_ids = ", ".join(r["cell_id"] for r in peers)

    if anomalies:
        lines = [
            f"**{cell} shows unusual behaviour** compared to its "
            f"{n_peers} peer cell(s) ({peer_ids}):\n"
        ]
        for a in anomalies:
            lines.append(f"- {a}")
        if normals:
            lines.append("\nWithin normal range:")
            for n in normals:
                lines.append(f"- {n}")
    else:
        lines = [
            f"**{cell} is not behaving unusually** compared to its "
            f"{n_peers} peer cell(s) ({peer_ids}).\n"
        ]
        for n in normals:
            lines.append(f"- {n}")

    lines.append(
        f"\n*Peers: cells from the same data source ({ctx['source']}). "
        f"Cross-source comparison is not meaningful due to incompatible resistance scales.*"
    )
    return "\n".join(lines)


def answer_recent_trajectory(ctx: dict, df, window: int = 20) -> str:
    """
    What happened in the last `window` cycles?
    Compares the recent window to the prior window of the same length.
    All values come from the DataFrame columns already computed by the pipeline.
    """
    import pandas as pd

    cell = ctx["cell_id"]

    actual_window = window if len(df) >= window + 5 else max(5, len(df) // 2)

    recent = df.tail(actual_window)
    prior_end   = len(df) - actual_window
    prior_start = max(0, prior_end - actual_window)
    prior  = df.iloc[prior_start:prior_end]

    cycle_start = int(recent.iloc[0]["cycle_number"])
    cycle_end   = int(recent.iloc[-1]["cycle_number"])

    soh_start  = float(recent.iloc[0]["soh_pct"])
    soh_end    = float(recent.iloc[-1]["soh_pct"])
    soh_change = soh_end - soh_start

    # Fade rate: compare window means
    fade_recent = float(recent["fade_rate_30cy"].mean()) * 1000
    fade_prior  = float(prior["fade_rate_30cy"].mean()) * 1000 if len(prior) > 0 else fade_recent
    fade_delta  = fade_recent - fade_prior

    res_start      = float(recent.iloc[0]["resistance_ohm"])
    res_end        = float(recent.iloc[-1]["resistance_ohm"])
    res_change_pct = (res_end - res_start) / res_start * 100 if res_start > 0 else 0.0

    # ── SOH line ──
    if soh_change < -2.0:
        soh_line   = f"SOH dropped {abs(soh_change):.1f} pp (from {soh_start:.1f}% to {soh_end:.1f}%)"
        soh_signal = "concerning"
    elif soh_change < -0.5:
        soh_line   = f"SOH declined {abs(soh_change):.1f} pp (from {soh_start:.1f}% to {soh_end:.1f}%)"
        soh_signal = "expected"
    else:
        soh_line   = f"SOH held near {soh_end:.1f}% (changed {soh_change:+.2f} pp)"
        soh_signal = "stable"

    # ── Fade rate line ──
    if fade_delta > 0.3:
        fade_line = (
            f"Fade rate is **accelerating** — up {fade_delta:.2f} mAh/cycle vs the "
            f"prior {actual_window} cycles (now {fade_recent:.2f} mAh/cycle)"
        )
    elif fade_delta < -0.3:
        fade_line = (
            f"Fade rate is **slowing** — down {abs(fade_delta):.2f} mAh/cycle vs the "
            f"prior {actual_window} cycles (now {fade_recent:.2f} mAh/cycle)"
        )
    else:
        fade_line = f"Fade rate is stable at approximately {fade_recent:.2f} mAh/cycle"

    # ── Resistance line ──
    if res_change_pct > 5.0:
        res_line = (
            f"Internal resistance increased {res_change_pct:.1f}% "
            f"({res_start*1000:.1f} to {res_end*1000:.1f} mOhm) — accelerating SEI growth"
        )
    elif res_change_pct > 1.0:
        res_line = (
            f"Internal resistance rose {res_change_pct:.1f}% "
            f"({res_start*1000:.1f} to {res_end*1000:.1f} mOhm) — normal aging"
        )
    else:
        res_line = (
            f"Internal resistance stable "
            f"({res_start*1000:.1f} to {res_end*1000:.1f} mOhm)"
        )

    overall = (
        "requires attention" if soh_signal == "concerning" or fade_delta > 0.5
        else ("progressing normally" if soh_signal == "expected" else "stable")
    )

    return (
        f"**Cycles {cycle_start}–{cycle_end} ({actual_window} cycles): "
        f"{cell} is {overall}.**\n\n"
        f"- {soh_line}\n"
        f"- {fade_line}\n"
        f"- {res_line}"
    )


def answer_fleet_compare(ctx: dict, fleet_stats: dict) -> str:
    """
    How does this cell rank among its peers?

    Peers are restricted to the same data mode (nasa / synth / uploaded).
    Cross-mode comparison stays out of scope because resistance scales are
    incompatible across measurement methods and model training boundaries.
    """
    cell      = ctx["cell_id"]
    cell_mode = ctx.get("data_mode", ctx["source_kind"])

    # Filter to same-mode peers only
    rows = [r for r in fleet_stats["rows"] if r.get("source", "nasa") == cell_mode]
    n    = len(rows)

    if n == 0:
        return (
            f"No other cells from the same data source ({cell_mode}) found in the fleet. "
            f"Cross-mode comparison stays out of scope."
        )

    # Exclude self to count actual peers
    peers = [r for r in rows if r["cell_id"] != cell]
    if len(peers) < 2:
        peer_ids = ", ".join(r["cell_id"] for r in peers) if peers else "none"
        return (
            f"Not enough same-mode cells for a fleet comparison.\n\n"
            f"**{cell}** is in {cell_mode} mode with only {len(peers)} other "
            f"cell{'s' if len(peers) != 1 else ''} ({peer_ids or 'none yet'}). "
            f"Fleet averages computed from fewer than 2 peers reduce to pairwise comparison "
            f"— 'faster than the fleet average' would mean 'faster than one other cell,' "
            f"which is not a meaningful fleet statistic.\n\n"
            f"Upload at least 3 cells in the same mode to enable fleet ranking."
            if cell_mode == "uploaded" else
            f"Not enough {cell_mode} cells for a fleet comparison.\n\n"
            f"**{cell}** has only {len(peers)} peer cell{'s' if len(peers) != 1 else ''} "
            f"in this mode ({peer_ids or 'none'}). "
            f"A minimum of 2 peers (3 total cells in mode) is required for fleet statistics "
            f"to be meaningful."
        )

    sorted_rows = sorted(rows, key=lambda r: r["soh"])  # worst first
    rank = next((i + 1 for i, r in enumerate(sorted_rows) if r["cell_id"] == cell), None)

    sohs        = [r["soh"] for r in rows]
    soh_median  = float(np.median(sohs))
    soh_min     = float(min(sohs))
    soh_max     = float(max(sohs))

    soh_diff_median = ctx["soh"] - soh_median
    median_dir      = "above" if soh_diff_median >= 0 else "below"

    if rank is None:
        rank_desc = f"not ranked (cell not found in same-mode fleet rows)"
    elif rank == 1:
        rank_desc = f"the lowest SOH among {cell_mode} cells (rank 1 of {n} — worst)"
    elif rank == n:
        rank_desc = f"the highest SOH among {cell_mode} cells (rank {n} of {n} — best)"
    elif rank <= n // 3:
        rank_desc = f"in the bottom third of {cell_mode} cells (rank {rank} of {n})"
    elif rank >= n - n // 3 + 1:
        rank_desc = f"in the top third of {cell_mode} cells (rank {rank} of {n})"
    else:
        rank_desc = f"mid-fleet among {cell_mode} cells (rank {rank} of {n})"

    # Fade rank within same-mode peers
    all_fades = [r["fade_30"] for r in rows if not math.isnan(r.get("fade_30", float("nan")))]
    fade_rank = sum(1 for f in all_fades if f > ctx["fade_30_mah_cy"]) + 1

    if fade_rank == 1:
        fade_rank_desc = f"fastest-degrading among {cell_mode} cells"
    elif fade_rank <= n // 3:
        fade_rank_desc = f"among the faster-degrading {cell_mode} cells (fade rank {fade_rank} of {n})"
    elif fade_rank >= n - n // 3 + 1:
        fade_rank_desc = f"among the slower-degrading {cell_mode} cells (fade rank {fade_rank} of {n})"
    else:
        fade_rank_desc = f"mid-fleet in degradation speed among {cell_mode} cells (fade rank {fade_rank} of {n})"

    # RUL context — same-mode peers only
    if ctx["rul_reliable"] and ctx["rul_pred"] is not None:
        peer_ruls = [
            r["rul_pred"] for r in rows
            if r["rul_reliable"] and r["rul_pred"] is not None and r["cell_id"] != cell
        ]
        if peer_ruls:
            mean_rul = float(np.mean(peer_ruls))
            rul_diff = ctx["rul_pred"] - mean_rul
            rul_dir  = "more" if rul_diff > 0 else "fewer"
            rul_line = (
                f"RUL estimate of {ctx['rul_pred']:.0f} cycles is "
                f"{abs(rul_diff):.0f} cycles {rul_dir} than the {cell_mode} fleet mean "
                f"for calibrated cells ({mean_rul:.0f} cycles)."
            )
        else:
            rul_line = f"RUL estimate: {ctx['rul_pred']:.0f} cycles (only calibrated {cell_mode} cell in fleet)."
    else:
        fold   = ctx["rul_fold_r2"]
        r2_str = f"fold R2={fold:.2f}" if fold is not None else "below floor"
        rul_line = f"RUL not calibrated ({r2_str}) — cannot contribute to fleet RUL ranking."

    return (
        f"**{cell} has {rank_desc}** with {ctx['soh']:.1f}% SOH.\n\n"
        f"{cell_mode.capitalize()} fleet SOH ranges from {soh_min:.1f}% to "
        f"{soh_max:.1f}%, median {soh_median:.1f}%. "
        f"{cell} sits {abs(soh_diff_median):.1f} pp {median_dir} the median.\n\n"
        f"{cell} is the {fade_rank_desc} at {ctx['fade_30_mah_cy']:.2f} mAh/cycle.\n\n"
        f"{rul_line}\n\n"
        f"*Cross-mode comparison stays out of scope — resistance scales are incompatible "
        f"across NASA, synthetic, and uploaded data sources.*"
    )


# ---------------------------------------------------------------------------
# Fleet-level answer (no specific cell required)
# ---------------------------------------------------------------------------

def answer_fleet_query(query: str, fleet_stats: dict) -> str:
    """
    Free-text natural-language front door for fleet-level questions — the
    "Ask the fleet" bar on the Fleet page. Deliberately restricted to
    fleet-level answer functions (answer_alerts, answer_replacement_budget,
    answer_fleet_risk) that need only fleet_stats, not a pre-selected cell —
    build_cell_context() indexes featured_dfs[cell_id] directly and cannot
    run without one, so a genuinely cell-agnostic "ask anything" bar isn't
    possible here. Per-cell questions ("why is B0018 degrading faster?")
    are out of scope for this entry point; the per-cell Copilot on the
    Decide & Ask page answers those instead.

    Unmatched queries fall back to TF-IDF retrieval over the battery-
    knowledge corpus plus an honest list of what this bar *can* answer —
    never a fabricated answer for a question outside its scope.

    Every routed answer appends an explicit reliability/calibration caveat
    in the same breath as the numbers (which cells are excluded from a
    fleet-wide RUL figure because their RUL isn't calibrated) rather than
    a clean-sounding number with the caveat buried elsewhere.
    """
    q = query.lower().strip()
    unreliable = fleet_stats.get("unreliable_rul", [])
    n = fleet_stats.get("n_cells", 0)

    caveat = ""
    if unreliable:
        shown = ", ".join(unreliable[:5]) + (", …" if len(unreliable) > 5 else "")
        caveat = (
            f"\n\n*Reliability note: {len(unreliable)} of {n} cells ({shown}) have RUL "
            f"predictions below the reliability floor — excluded from any RUL-based "
            f"figure above.*"
        )

    if any(k in q for k in ("budget", "cost", "spend", "afford", "expensive")):
        return answer_replacement_budget(fleet_stats) + caveat
    if any(k in q for k in ("risk", "exposure", "danger", "business case")):
        return answer_fleet_risk(fleet_stats) + caveat
    if any(k in q for k in ("alert", "summary", "overview", "eol", "end of life", "flag", "attention", "status")):
        return answer_alerts(fleet_stats) + caveat

    from copilot_retrieval import retrieve
    related = retrieve(query, top_k=2)
    fallback = (
        "I can answer fleet-level questions about:\n\n"
        "- **Alerts** — \"What are the current fleet alerts?\"\n"
        "- **Replacement budget** — \"What will replacement cost over the next 12 months?\"\n"
        "- **Fleet risk** — \"What is the business risk in my fleet?\"\n\n"
        "For questions about one specific cell (e.g. why it's degrading faster than others), "
        "use the per-cell Copilot on the Decide & Ask page instead."
    )
    if related:
        fallback += "\n\nRelated background:\n\n" + "\n\n".join(related)
    return fallback


def answer_alerts(fleet_stats: dict) -> str:
    """Fleet-wide alert summary — identifies EOL cells, fastest fading, and uncalibrated RUL."""
    eol       = fleet_stats["eol_cells"]
    degrading = fleet_stats["degrading_cells"]
    fastest   = fleet_stats["sorted_by_fade"][:3]
    unreliable = fleet_stats["unreliable_rul"]
    n         = fleet_stats["n_cells"]
    healthy   = n - len(eol) - len(degrading)

    lines = [_T["alerts_header"].format_map({"n": n}) + "\n"]

    if eol:
        lines.append(_T["alerts_eol"].format_map({
            "count": len(eol), "plural": "s" if len(eol) > 1 else "",
            "cells": ", ".join(f"**{c}**" for c in eol),
        }))
    if degrading:
        lines.append(_T["alerts_degrading"].format_map({
            "count": len(degrading), "plural": "s" if len(degrading) > 1 else "",
            "cells": ", ".join(f"**{c}**" for c in degrading),
        }))
    if not eol and not degrading:
        lines.append(_T["alerts_no_eol"])

    lines.append("\n**Fastest degrading (30-cycle fade rate):**")
    for r in fastest:
        status = "End of Life" if r["soh"] < 80 else ("Degrading" if r["soh"] < 90 else "Healthy")
        lines.append(
            f"- **{r['cell_id']}**: {r['fade_30']:.2f} mAh/cycle "
            f"({status}, {r['soh']:.1f}% SOH)"
        )

    lines.append("")
    if unreliable:
        lines.append(_T["alerts_unreliable_rul"].format_map({
            "cells": ", ".join(f"**{c}**" for c in unreliable),
            "floor": RUL_RELIABLE_FLOOR,
        }))
    else:
        lines.append(_T["alerts_all_calibrated"])

    lines.append("\n" + _T["alerts_footer"].format_map({
        "n_eol": len(eol), "n_deg": len(degrading), "n_healthy": healthy,
    }))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Business-surface answers (plain English, no technical jargon)
# ---------------------------------------------------------------------------

_REPLACEMENT_COST_USD = 150   # per cell — used by fleet-level budget/risk narratives below


def answer_replacement_budget(fleet_stats: dict) -> str:
    """Plain-English 12-month replacement cost forecast."""
    eol       = fleet_stats["eol_cells"]
    degrading = fleet_stats["degrading_cells"]
    n         = fleet_stats["n_cells"]

    # Cells needing replacement now (EOL) + likely to need it within 12 months
    # (degrading cells — conservative assumption: half reach EOL within a year)
    certain_now   = len(eol)
    likely_12m    = len(degrading)
    total_replace = certain_now + likely_12m

    cost_now  = certain_now   * _REPLACEMENT_COST_USD
    cost_12m  = total_replace * _REPLACEMENT_COST_USD

    lines = [
        f"## 12-Month Replacement Budget\n",
        f"Based on the current health of your **{n} cells**, here is the spending outlook:\n",
        f"- **Immediate replacements needed:** {certain_now} cell{'s' if certain_now != 1 else ''} "
        f"are below 80% SOH (end of life). Cost: **${cost_now:,.0f}**",
        f"- **Within 12 months:** a further {likely_12m} cell{'s' if likely_12m != 1 else ''} "
        f"{'are' if likely_12m != 1 else 'is'} degrading and likely to reach end of life. "
        f"Combined budget: **${cost_12m:,.0f}**",
        f"- **Best case:** if degrading cells stabilise, minimum spend is **${cost_now:,.0f}**",
        f"\n*Estimate uses ${_REPLACEMENT_COST_USD}/cell. Adjust in Settings for your actual unit cost.*",
    ]

    if total_replace == 0:
        lines = [
            f"## 12-Month Replacement Budget\n",
            f"Good news — all **{n} cells** are currently healthy (above 90% SOH). "
            f"No replacements are expected within the next 12 months based on current trends.\n",
            f"*If conditions change, re-check this estimate quarterly.*",
        ]

    return "\n".join(lines)


def answer_fleet_risk(fleet_stats: dict) -> str:
    """Plain-English fleet risk assessment for business stakeholders."""
    eol        = fleet_stats["eol_cells"]
    degrading  = fleet_stats["degrading_cells"]
    unreliable = fleet_stats["unreliable_rul"]
    n          = fleet_stats["n_cells"]
    fastest    = fleet_stats["sorted_by_fade"][:3]

    at_risk = len(eol) + len(degrading)
    pct     = 100 * at_risk / max(n, 1)

    if pct >= 30:
        risk_level, risk_color = "HIGH", "critical"
    elif pct >= 10:
        risk_level, risk_color = "MEDIUM", "moderate"
    else:
        risk_level, risk_color = "LOW", "low"

    lines = [
        f"## Fleet Business Risk Assessment\n",
        f"**Overall risk: {risk_level}** — {at_risk} of {n} cells ({pct:.0f}%) "
        f"are at or approaching end of life.\n",
    ]

    if eol:
        lines.append(
            f"**Operational risk:** {len(eol)} cell{'s' if len(eol)!=1 else ''} "
            f"({', '.join(eol[:5])}{', …' if len(eol)>5 else ''}) have already passed "
            f"the 80% SOH threshold. Continuing to operate these increases the risk of "
            f"in-service failure and may void warranty or compliance coverage."
        )

    if degrading:
        lines.append(
            f"\n**Near-term risk:** {len(degrading)} cell{'s' if len(degrading)!=1 else ''} "
            f"are in the 80–90% SOH range and degrading. These may not survive another "
            f"full operational cycle without dropping below the safety threshold."
        )

    lines.append(f"\n**Top 3 fastest-degrading cells:**")
    for r in fastest:
        lines.append(
            f"- **{r['cell_id']}**: losing {r['fade_30']:.2f} mAh per cycle "
            f"(currently {r['soh']:.1f}% SOH)"
        )

    if unreliable:
        lines.append(
            f"\n**Forecast uncertainty:** RUL predictions for "
            f"{', '.join(f'**{c}**' for c in unreliable[:3])}"
            f"{'…' if len(unreliable)>3 else ''} are based on limited data. "
            f"Treat these as directional rather than precise."
        )

    lines.append(
        f"\n*Replacement cost exposure: **${at_risk * _REPLACEMENT_COST_USD:,.0f}** "
        f"(${_REPLACEMENT_COST_USD}/cell × {at_risk} at-risk cells).*"
    )

    return "\n".join(lines)


def answer_cell_story(story: dict) -> str:
    """Plain-English single-cell narrative for the Decide & Ask page.

    Mirrors answer_fleet_risk()'s job at the single-cell decision level
    (Data Storytelling review finding, the one item left open in the
    2026-07-30 self-audit: Decide & Ask showed 4 separate cards -- hero
    recommendation, mechanism verdict, application fit scores, NPV
    comparison -- with no synthesized "here's the story" paragraph tying
    them together. A reader had to piece together *why* a recommendation
    was made and whether the financial numbers actually agreed with it.

    This function performs NO new classification, thresholding, or
    financial math of its own -- it only narrates values the caller already
    computed via classify()/diagnose_mechanism()/application_fit()/the
    page's own NPV comparison, so it can never drift from or disagree with
    the cards it summarizes (the exact bug class this project has fixed
    repeatedly elsewhere: a second independent derivation of the same
    verdict).

    story: {
        "cell_id": str, "soh": float, "action_label": str,
        "confidence_label": str,        # a design_system.CONF_META display
                                         # string, e.g. "High confidence" --
                                         # already reads naturally in "(...)"
        "mechanism": dict | None,       # diagnose_mechanism() output
        "fit_scores": dict,             # application_fit() output
        "financial_best_label": str, "financial_best_npv": float,
        "financial_disagrees": bool,
        "npv_max_label": str, "npv_max_value": float,
    }
    """
    cell_id          = story["cell_id"]
    soh               = story["soh"]
    action_label      = story["action_label"]
    confidence_label  = story["confidence_label"]
    mechanism         = story.get("mechanism")
    fit_scores        = story.get("fit_scores") or {}

    sentences = [
        f"**{cell_id}** is at **{soh:.1f}% SOH** — the recommended action is "
        f"**{action_label}** ({confidence_label})."
    ]

    if mechanism and mechanism.get("verdict") not in (None, "Insufficient data") \
            and mechanism.get("confidence_label") not in (None, "No data"):
        sentences.append(
            f"Degradation is driven primarily by **{mechanism['verdict']}** "
            f"({mechanism['confidence_label']} confidence)."
        )

    if fit_scores:
        fit_apps = [v["name"] for v in fit_scores.values() if v["fit"] == "fit"]
        marginal_apps = [v["name"] for v in fit_scores.values() if v["fit"] == "marginal"]
        if fit_apps:
            sentences.append(
                f"For second-life reuse, it is a good fit for {', '.join(fit_apps)}."
            )
        elif marginal_apps:
            sentences.append(
                f"For second-life reuse, it is a marginal fit for {', '.join(marginal_apps)} "
                f"— worth evaluating case by case."
            )
        else:
            sentences.append("It does not fit any tracked second-life application today.")

    fb_label = story.get("financial_best_label")
    fb_npv   = story.get("financial_best_npv")
    if fb_label is not None and fb_npv is not None:
        if story.get("financial_disagrees"):
            sentences.append(
                f"Financially, **{story['npv_max_label']}** has the higher 5-yr NPV "
                f"(${story['npv_max_value']:,.0f}), but the recommendation above is followed "
                f"instead of the pure NPV-maximizing option — see \"Compare alternatives\" "
                f"below for the full picture."
            )
        else:
            sentences.append(
                f"Financially, **{fb_label}** is also the highest-NPV option "
                f"(${fb_npv:,.0f} over 5 years) — the operational and financial "
                f"recommendations agree."
            )

    return " ".join(sentences)


def answer_business_case(ctx: dict, result: dict, financials: "dict | None", assumptions: dict) -> str:
    """Replace vs. wait vs. repurpose decision in plain English.

    Grounded in the same classify()/financial_comparison() results Decide &
    Ask itself uses -- not an independent SOH/cost ladder. This used to
    hardcode its own 90%/80% SOH thresholds and a flat replacement cost via
    _REPLACEMENT_COST_USD, entirely separate from
    consequences.ASSUMPTIONS/application_fit()/classify()/
    financial_comparison() -- the same "independently recomputed ladder"
    bug class this project has fixed repeatedly elsewhere (e.g.
    consequences.best_fit_application()). A cell could get one action from
    Decide & Ask and a different-sounding recommendation from this Copilot
    chip for the same cell. (It also silently read a "rul_cycles" key
    build_cell_context() never sets -- always None -- so the months-
    remaining estimate below never actually fired; classify()'s own
    action_reasons replace that entirely now.)

    ctx: build_cell_context() output — only cell_id/soh are read (raw
        facts); the decision itself comes from `result`, not from ctx.
    result: classify()'s output — action, action_reasons, confidence.
    financials: financial_comparison()'s output, or None when this cell's
        chemistry has no CELL_NOMINAL_KWH entry (e.g. an unspecified user
        upload) — financial framing is omitted rather than guessing a
        capacity or crashing.
    assumptions: {key: value} flattened from consequences.ASSUMPTIONS.
    """
    cell_id = ctx["cell_id"]
    soh     = ctx["soh"]
    action  = result["action"]

    from design_system import ACTION_META
    action_label = ACTION_META[action][0]

    lines = [
        f"## Replace or Repurpose? — {cell_id}\n",
        f"**Recommendation: {action_label}** — {cell_id} is at {soh:.1f}% SOH.\n",
    ]
    for reason in result.get("action_reasons", []):
        lines.append(f"- {reason}")

    if financials is not None:
        lines += [
            f"\n**Replace now:** ${financials['new_cell_cost']:.0f} upfront. "
            f"Eliminates unplanned failure risk.",
            f"**Wait:** carries the risk of continuing to operate a degrading cell.",
            f"**Repurpose (second-life):** ~${financials['sl_net']:.2f} residual value "
            f"if the application tolerates this cell's remaining capacity.",
            f"**Recycle instead:** ~${financials['recycle_value']:.2f} material recovery value.",
        ]
    else:
        lines.append(
            "\n*Financial figures aren't available for this cell's chemistry — "
            "nominal capacity isn't specified for this data source.*"
        )

    lines.append(
        f"\n*Figures use this platform's assumption defaults "
        f"(${assumptions.get('new_cell_cost', 0):.0f}/cell replacement). Update in Settings.*"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-powered Copilot — Claude Haiku, constrained to bundle values
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Battery Intelligence Copilot, an expert assistant embedded in a
battery health monitoring platform. You answer questions about battery cells based ONLY on the
data values provided in the user message. You must NEVER invent, estimate, or hallucinate numbers.
If a value is not present in the data, say it is not available.
Use plain English. Be concise — 3-5 sentences maximum unless asked to elaborate.
Do not use jargon the user didn't use. Never recommend specific financial products or services.
Format numbers exactly as given. If uncertain, say so explicitly."""

_QUERY_SYSTEM_HINTS = {
    "health":             "Focus on SOH value, trend, and what the fade rate means in plain English.",
    "prediction_drivers": "Explain the top model features in plain English without statistical jargon.",
    "rul":                "Explain remaining life and uncertainty in plain English. Mention reliability flag.",
    "compare":            "Compare the two cells factually. Avoid superlatives.",
    "recent_trajectory":  "Describe the trajectory trend from the recent cycles data.",
    "anomaly":            "Explain the anomaly flags and their IEC 62619 significance.",
    "fleet_compare":      "Place this cell in the context of the fleet statistics.",
    "alerts":             "List the most important fleet alerts in priority order.",
    "replacement_budget": "Give the 12-month budget estimate with best and worst case.",
    "fleet_risk":         "Assess overall fleet risk level with cost exposure.",
    "business_case":      "Give a clear Replace/Wait/Repurpose recommendation with NPV reasoning.",
}


def llm_answer(query: str, template_answer: str, api_key: str) -> str:
    """Call Claude Haiku with the template answer as grounding context.

    The template answer is injected verbatim as the data source — the LLM
    is only allowed to rephrase and clarify what the template already contains.
    If the API call fails for any reason, the template answer is returned as-is.
    """
    if not api_key:
        return template_answer
    try:
        import anthropic as _ant
        _client = _ant.Anthropic(api_key=api_key)
        _hint = _QUERY_SYSTEM_HINTS.get(query, "")
        _system = _SYSTEM_PROMPT + (f"\n\nAdditional guidance: {_hint}" if _hint else "")
        _user_msg = (
            f"Here is the data summary for this question:\n\n"
            f"---\n{template_answer}\n---\n\n"
            f"Rewrite this as a clear, conversational response. "
            f"Use exactly the numbers shown above — do not change any values. "
            f"Do not add any information not present in the data summary."
        )
        _resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=_system,
            messages=[{"role": "user", "content": _user_msg}],
        )
        return _resp.content[0].text.strip()
    except Exception:
        return template_answer
