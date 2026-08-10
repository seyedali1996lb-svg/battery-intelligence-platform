"""
Residual-value / bankability report.

A financing-grade asset-condition summary — SOH, RUL quantiles, second-life
fit, and NPV comparison packaged as one document, for anyone trying to
raise capital against (or price) a second-life battery deployment. Reuses
src/consequences.py's existing application_fit()/best_fit_application()/
financial_comparison()/eol_r_code_recommendation() rather than re-deriving
their math — same "single source of truth" discipline as src/spine_export.py
and src/optimade_export.py, which this module's shape otherwise mirrors
(precomputed values in, one build_*() function, field dicts in the same
{label,value,state,note} shape src/passport.py already uses so
src/report_pdf.py's existing _field_table()/styling can render this too).

This is explicitly NOT investment advice, a credit rating, or a guarantee
of future performance — see the disclaimer field every consumer (UI, PDF)
must surface prominently, not bury in a footnote. It is an informational
summary of this platform's own model outputs (SOH estimate, RUL
prediction with leave-cell-out-validated uncertainty, second-life fit
score, NPV comparison under stated assumptions) — real analytical work,
but not a substitute for physical inspection, a licensed appraisal, or a
rating agency's own diligence process.
"""


def build_bankability_report(
    cell_id: str,
    source: str,
    chemistry: str,
    soh: float,
    fade_30_mah_cy: float,
    fleet_fade_median: "float | None",
    cycle_count: int,
    n_lco_cells: "int | None",
    lco_soh_r2: "float | None",
    rul_reliable: bool,
    rul_q10: "float | None",
    rul_pred: "float | None",
    rul_q90: "float | None",
    mechanism: "dict | None",
    sop_pct: "float | None" = None,
    assumptions: "dict | None" = None,
    experiment_run_id: "str | None" = None,
    git_commit: "str | None" = None,
) -> dict:
    """
    Build the bankability report dict for one cell.

    Computes financial_comparison() internally from consequences.ASSUMPTIONS
    (optionally overridden per-key via `assumptions`, same override
    convention as src/spine_export.py's own build_second_life_export()) —
    reused, not re-derived math, and no risk of the caller passing a
    financial dict computed under different assumption values than what's
    shown elsewhere on the page.

    Returns {"cell_id", "identity": [...], "condition": [...],
    "second_life": [...], "financial": [...], "provenance": [...],
    "disclaimer": str} — each list a field-dict list in the same
    {"label","value","state","note"} shape src/passport.py uses, so
    src/report_pdf.py's existing _field_table() renders this unchanged.
    """
    from consequences import ASSUMPTIONS, application_fit, best_fit_application, eol_r_code_recommendation, financial_comparison

    a = {k: v["value"] for k, v in ASSUMPTIONS.items()}
    if assumptions:
        a.update(assumptions)
    financial = financial_comparison(
        soh, source, a["recycling_value"], a["new_cell_cost"],
        a["second_life_value_per_kwh"], a["repack_cost"],
    )

    identity = [
        {"label": "Cell / asset ID", "value": cell_id, "state": "available"},
        {"label": "Chemistry", "value": chemistry, "state": "available"},
        {"label": "Data source", "value": source, "state": "available"},
        {"label": "Cycle count", "value": str(cycle_count), "state": "available"},
    ]

    condition = [
        {"label": "State of Health (SOH)", "value": f"{soh:.1f}%", "state": "available",
         "note": f"Leave-cell-out validated R² = {lco_soh_r2:.2f} across n={n_lco_cells} cells" if (lco_soh_r2 is not None and n_lco_cells) else "Leave-cell-out validation sample size not available"},
        {"label": "Fade rate (30-cycle)", "value": f"{fade_30_mah_cy * 1000:.2f} mAh/cycle", "state": "available"},
    ]
    if sop_pct is not None:
        condition.append({"label": "Peak power capability (State of Power)", "value": f"{sop_pct:.0f}% of initial", "state": "available"})
    if rul_reliable and rul_pred is not None:
        condition.append({
            "label": "Remaining Useful Life (RUL)",
            "value": f"{rul_pred:.0f} cycles (P50)" + (f", {rul_q10:.0f}–{rul_q90:.0f} cycles (P10–P90)" if rul_q10 is not None and rul_q90 is not None else ""),
            "state": "available",
            "note": "GradientBoostingRegressor, leave-cell-out validated — not a fleet-scale guarantee, see provenance section",
        })
    else:
        condition.append({
            "label": "Remaining Useful Life (RUL)", "value": "Not reliable at this cycle count",
            "state": "unavailable",
            "note": "Model has not cleared this platform's per-cell reliability floor yet",
        })
    if mechanism:
        condition.append({
            "label": "Degradation mechanism", "value": mechanism.get("verdict", "insufficient_data"),
            "state": "estimated" if mechanism.get("verdict") != "insufficient_data" else "unavailable",
            "note": f"Confidence: {mechanism.get('confidence', 'n/a')}",
        })

    fit_scores = application_fit(soh, fade_30_mah_cy, fleet_fade_median, sop_pct=sop_pct)
    best_key, best_result = best_fit_application(fit_scores)
    r_code = eol_r_code_recommendation(soh, fade_30_mah_cy, sop_pct=sop_pct)
    second_life = [
        {"label": "Best-fit second-life application", "value": best_result["name"], "state": "estimated",
         "note": f"Fit: {best_result['fit']} — " + "; ".join(best_result["reasons"][:2])},
        {"label": "End-of-life recommendation (IEC 62902)", "value": r_code["r_code"], "state": "estimated"},
    ]

    financial_fields = [
        {
            "label": label, "value": f"${financial[key]:,.2f}", "state": "estimated",
            "note": "Illustrative/cited assumption inputs — see the app's Assumption Register for sourcing",
        }
        for key, label in (
            ("sl_net", "Second-life reuse value (net of repack)"),
            ("recycle_value", "Immediate recycle value"),
            ("new_cell_cost", "New cell replacement cost"),
        )
    ]

    provenance = [
        {"label": "Experiment run ID", "value": experiment_run_id or "Not available", "state": "available" if experiment_run_id else "unavailable",
         "note": "Traces this report's SOH/RUL predictions back to the exact logged training run — see the app's 'Regenerate this report' action"},
        {"label": "Git commit", "value": git_commit or "Not available", "state": "available" if git_commit else "unavailable"},
    ]

    disclaimer = (
        "This is an informational asset-condition summary generated from this platform's own "
        "model outputs (State of Health estimate, leave-cell-out-validated Remaining Useful "
        "Life prediction, second-life application fit score, and NPV comparison under stated "
        "assumptions). IT IS NOT INVESTMENT ADVICE, A CREDIT RATING, AN APPRAISAL, OR A "
        "GUARANTEE OF FUTURE PERFORMANCE. It does not substitute for physical inspection, a "
        "licensed appraisal, or a rating agency's own diligence process. Anyone using this "
        "document to raise capital or price a transaction should independently verify every "
        "figure and consult qualified financial/legal advisors."
    )

    return {
        "cell_id": cell_id,
        "identity": identity,
        "condition": condition,
        "second_life": second_life,
        "financial": financial_fields,
        "provenance": provenance,
        "disclaimer": disclaimer,
    }
