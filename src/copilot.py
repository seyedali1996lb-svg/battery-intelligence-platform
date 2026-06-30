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

from lco_eval import RUL_RELIABLE_FLOOR

# ---------------------------------------------------------------------------
# Navigation constants
# ---------------------------------------------------------------------------

QUERY_LABELS: dict[str, str] = {
    "health":        "What is this cell's health?",
    "drivers":       "Why does the model predict this SOH?",
    "rul":           "How much life is left?",
    "compare":       "Compare to another cell",
    "recent":        "What happened in the last 20 cycles?",
    "anomaly":       "Is this cell behaving unusually?",
    "fleet_compare": "How does this cell rank in the fleet?",
    "alerts":        "Fleet alert summary",
}

# Suggested follow-up queries after each answer — cell-agnostic queries like
# "alerts" only appear as follow-ups, not as mandatory first choices.
FOLLOW_UP_MAP: dict[str, list[str]] = {
    "health":        ["drivers", "recent",        "fleet_compare"],
    "drivers":       ["health",  "anomaly",        "recent"],
    "rul":           ["health",  "fleet_compare",  "compare"],
    "compare":       ["health",  "fleet_compare",  "rul"],
    "recent":        ["anomaly", "drivers",         "rul"],
    "anomaly":       ["recent",  "drivers",         "fleet_compare"],
    "fleet_compare": ["health",  "rul",             "anomaly"],
    "alerts":        ["fleet_compare", "health",   "rul"],
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

NASA_CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]


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

    fi           = feature_importance_df(bundle, "soh")
    top_features = fi.head(5).to_dict(orient="records")

    if is_nasa:
        source         = "NASA PCoE real measured data"
        operating_note = "discharged at 2A to ~2.7V cutoff at ~24 C lab conditions"
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

    return {
        "cell_id":        cell_id,
        "source":         source,
        "operating_note": operating_note,
        "is_nasa":        is_nasa,
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


# ---------------------------------------------------------------------------
# Fleet stats builder — aggregates across all cells for anomaly & fleet answers
# ---------------------------------------------------------------------------

def build_fleet_stats(featured_dfs: dict, bundles: dict) -> dict:
    """
    Compute fleet-level aggregates from already-computed DataFrames.
    No new model inference — every value is a simple summary statistic
    over values already present in featured_dfs and the bundles.

    Source-split stats (nasa vs synth) are kept separate because resistance
    scales are incompatible across measurement methods.
    """
    rows: list[dict] = []

    for cell_id, df in featured_dfs.items():
        is_nasa  = cell_id in NASA_CELL_IDS
        bundle   = bundles["nasa"] if is_nasa else bundles["synth"]
        if bundle is None:
            continue
        per_cell_ok  = bundle["metrics"].get("per_cell_rul_reliable", {})
        rul_reliable = per_cell_ok.get(cell_id, bundle["metrics"].get("rul_reliable", False))

        latest   = df.iloc[-1]
        soh      = float(latest["soh_pct"])
        fade_30  = float(latest.get("fade_rate_30cy", float("nan"))) * 1000
        resist   = float(latest.get("resistance_ohm", float("nan")))
        cycle    = int(latest["cycle_number"])
        rul_pred = float(latest["rul_pred"]) if rul_reliable else None

        rows.append({
            "cell_id":     cell_id,
            "source":      "nasa" if is_nasa else "synth",
            "soh":         soh,
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
        f0   = ctx["top_features"][0]
        name = FEATURE_LABELS.get(f0["feature"], f0["feature"])
        top_driver = (
            f"The model's strongest signal for this assessment is **{name}**, "
            f"which accounts for {f0['importance_pct']:.0f}% of the SOH prediction."
        )

    return (
        f"**{cell}** is currently at **{soh:.1f}% state of health** — "
        f"{ctx['status_note']}.\n\n"
        f"This is based on cycle {cycle} of {ctx['source']} ({ctx['operating_note']}). "
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
    Explain remaining useful life — or explain honestly why it is not shown.
    If rul_reliable is False, no cycle estimate is generated under any circumstances.
    """
    cell = ctx["cell_id"]

    if not ctx["rul_reliable"]:
        fold_r2 = ctx["rul_fold_r2"]
        r2_str  = f"R2={fold_r2:.2f}" if fold_r2 is not None else "below the reliability threshold"
        return (
            f"**A reliable remaining-life estimate is not available for {cell}.**\n\n"
            f"During leave-cell-out validation — where the model was trained on all "
            f"other cells and tested on {cell} alone — the RUL model achieved "
            f"**{r2_str}** for this cell. The reliability threshold is R2>={RUL_RELIABLE_FLOOR}.\n\n"
            f"An R2 below this floor means the model's remaining-life predictions for "
            f"{cell} were less accurate than simply guessing the average remaining "
            f"life across similar cells. Displaying a cycle count would create false "
            f"confidence in a number the model cannot support.\n\n"
            f"This does not mean {cell}'s data is wrong — it means the model has not "
            f"seen enough cells with a similar degradation profile to generalise "
            f"reliably to this one. As more cells are added to the training set, "
            f"this may change."
        )

    rul  = ctx["rul_pred"]
    fold = ctx["rul_fold_r2"]
    conf = f" (model fold R2={fold:.2f} on this cell)" if fold is not None else ""

    if ctx["eol_reached"]:
        return (
            f"**{cell} has already reached end of life**, crossing the 80% SOH "
            f"threshold at cycle {ctx['eol_at']}. It is currently at cycle {ctx['cycle']}.\n\n"
            f"The model's remaining-life estimate is {rul:.0f} cycles, which at this "
            f"stage reflects how much further below EOL the cell continues to operate — "
            f"not a forward-looking projection{conf}."
        )

    return (
        f"**{cell}** has an estimated **{rul:.0f} cycles of useful life remaining** "
        f"before reaching the 80% SOH end-of-life threshold{conf}.\n\n"
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

    if ctx_a["is_nasa"] != ctx_b["is_nasa"]:
        nasa_cell  = a if ctx_a["is_nasa"] else b
        synth_cell = b if ctx_a["is_nasa"] else a
        lines.append(
            f"**Source note:** {nasa_cell} is real measured data (NASA PCoE); "
            f"{synth_cell} is physics-informed synthetic data. SOH % is directly "
            f"comparable — both are relative to each cell's own initial capacity. "
            f"Absolute resistance values are not comparable across measurement methods.\n"
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
    Peers are cells from the same data source (nasa or synth) to avoid
    the resistance-scale incompatibility between the two datasets.
    """
    cell   = ctx["cell_id"]
    source = "nasa" if ctx["is_nasa"] else "synth"

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
    """How does this cell rank in the fleet by SOH, fade rate, and RUL?"""
    cell = ctx["cell_id"]
    rows = fleet_stats["rows"]
    n    = fleet_stats["n_cells"]

    sorted_rows = fleet_stats["sorted_by_soh"]  # worst first
    rank        = next((i + 1 for i, r in enumerate(sorted_rows) if r["cell_id"] == cell), None)

    soh_diff_median = ctx["soh"] - fleet_stats["soh_median"]
    median_dir      = "above" if soh_diff_median >= 0 else "below"

    if rank == 1:
        rank_desc = f"the lowest SOH in the fleet (rank 1 of {n} — worst)"
    elif rank == n:
        rank_desc = f"the highest SOH in the fleet (rank {n} of {n} — best)"
    elif rank <= n // 3:
        rank_desc = f"in the bottom third of the fleet (rank {rank} of {n})"
    elif rank >= n - n // 3 + 1:
        rank_desc = f"in the top third of the fleet (rank {rank} of {n})"
    else:
        rank_desc = f"mid-fleet (rank {rank} of {n})"

    # Fade rank across fleet
    all_fades  = [r["fade_30"] for r in rows if not math.isnan(r.get("fade_30", float("nan")))]
    fade_rank  = sum(1 for f in all_fades if f > ctx["fade_30_mah_cy"]) + 1  # 1 = fastest

    if fade_rank == 1:
        fade_rank_desc = f"fastest-degrading cell in the fleet"
    elif fade_rank <= n // 3:
        fade_rank_desc = f"among the faster-degrading cells (fade rank {fade_rank} of {n})"
    elif fade_rank >= n - n // 3 + 1:
        fade_rank_desc = f"among the slower-degrading cells (fade rank {fade_rank} of {n})"
    else:
        fade_rank_desc = f"mid-fleet in degradation speed (fade rank {fade_rank} of {n})"

    # RUL context
    if ctx["rul_reliable"] and ctx["rul_pred"] is not None:
        peer_ruls = [
            r["rul_pred"] for r in rows
            if r["rul_reliable"] and r["rul_pred"] is not None and r["cell_id"] != cell
        ]
        if peer_ruls:
            mean_rul  = float(np.mean(peer_ruls))
            rul_diff  = ctx["rul_pred"] - mean_rul
            rul_dir   = "more" if rul_diff > 0 else "fewer"
            rul_line  = (
                f"RUL estimate of {ctx['rul_pred']:.0f} cycles is "
                f"{abs(rul_diff):.0f} cycles {rul_dir} than the fleet mean "
                f"for calibrated cells ({mean_rul:.0f} cycles)."
            )
        else:
            rul_line = f"RUL estimate: {ctx['rul_pred']:.0f} cycles (only calibrated cell in fleet)."
    else:
        fold   = ctx["rul_fold_r2"]
        r2_str = f"fold R2={fold:.2f}" if fold is not None else "below floor"
        rul_line = f"RUL not calibrated ({r2_str}) — cannot contribute to fleet RUL ranking."

    return (
        f"**{cell} has {rank_desc}** with {ctx['soh']:.1f}% SOH.\n\n"
        f"Fleet SOH ranges from {fleet_stats['soh_min']:.1f}% to "
        f"{fleet_stats['soh_max']:.1f}%, median {fleet_stats['soh_median']:.1f}%. "
        f"{cell} sits {abs(soh_diff_median):.1f} pp {median_dir} the median.\n\n"
        f"{cell} is the {fade_rank_desc} at {ctx['fade_30_mah_cy']:.2f} mAh/cycle.\n\n"
        f"{rul_line}"
    )


# ---------------------------------------------------------------------------
# Fleet-level answer (no specific cell required)
# ---------------------------------------------------------------------------

def answer_alerts(fleet_stats: dict) -> str:
    """Fleet-wide alert summary — identifies EOL cells, fastest fading, and uncalibrated RUL."""
    eol       = fleet_stats["eol_cells"]
    degrading = fleet_stats["degrading_cells"]
    fastest   = fleet_stats["sorted_by_fade"][:3]
    unreliable = fleet_stats["unreliable_rul"]
    n         = fleet_stats["n_cells"]
    healthy   = n - len(eol) - len(degrading)

    lines = [f"**Fleet alert summary — {n} cells monitored**\n"]

    if eol:
        cell_str = ", ".join(f"**{c}**" for c in eol)
        lines.append(
            f"**End of Life ({len(eol)} cell{'s' if len(eol) > 1 else ''}):** "
            f"{cell_str} — below 80% SOH, past the replacement threshold."
        )
    if degrading:
        cell_str = ", ".join(f"**{c}**" for c in degrading)
        lines.append(
            f"**Degrading ({len(degrading)} cell{'s' if len(degrading) > 1 else ''}):** "
            f"{cell_str} — between 80% and 90% SOH, monitor closely."
        )
    if not eol and not degrading:
        lines.append("No cells are at or past end of life.")

    lines.append("\n**Fastest degrading (30-cycle fade rate):**")
    for r in fastest:
        status = "End of Life" if r["soh"] < 80 else ("Degrading" if r["soh"] < 90 else "Healthy")
        fade   = r["fade_30"]
        lines.append(
            f"- **{r['cell_id']}**: {fade:.2f} mAh/cycle "
            f"({status}, {r['soh']:.1f}% SOH)"
        )

    lines.append("")
    if unreliable:
        cell_str = ", ".join(f"**{c}**" for c in unreliable)
        lines.append(
            f"**RUL not calibrated:** {cell_str} — remaining-life estimates are withheld "
            f"(model fold R2 below {RUL_RELIABLE_FLOOR} on leave-cell-out validation)."
        )
    else:
        lines.append("All cells have calibrated RUL estimates.")

    lines.append(
        f"\n**Fleet health:** {len(eol)} at EOL · {len(degrading)} degrading · {healthy} healthy"
    )

    return "\n".join(lines)
