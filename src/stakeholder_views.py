"""
Shared multi-stakeholder fleet view.

The same cell's data, sliced three different ways for three different
real parties who'd each want a different subset: the OEM (manufacturer,
cares about degradation mechanism as a quality signal and warranty
exposure), the operator (current owner/fleet manager, cares about
action/maintenance/financials), and the recycler (cares about chemistry,
material content, and the end-of-life pathway, not the operator's day-
to-day decision).

Every field below is read from values this platform already computed
(mechanism verdict, warranty_breach_estimate(), application_fit()/
eol_r_code_recommendation(), financial_comparison(), recommend_recyclers())
— nothing here recomputes or re-derives; same single-source-of-truth
discipline as src/spine_export.py/src/bankability_report.py.

This is a real, reusable slicing of cell data (used by both the
Streamlit "Stakeholder View" tab and 3 new REST API endpoints in
src/api.py — GET /cells/{id}/view/{oem|operator|recycler} — the actual
mechanism by which the same cell's data could be shared with a real
external OEM/operator/recycler party, gated by this platform's existing
JWT auth). It is NOT a new login system for external parties, no new
role, no new authentication — those three stakeholders use the same
REST API a real integration partner would.

Each build_*_view() returns a list of field dicts in the same
{"label","value","state","note"} shape src/passport.py's groups use, so
the same rendering helpers (Streamlit or app/utils.py's render_card-based
table, or src/report_pdf.py's _field_table() if a PDF is ever added)
work unchanged across the three.
"""


def build_oem_view(
    cell_id: str,
    chemistry: str,
    source: str,
    soh: float,
    cycle_count: int,
    fade_30_mah_cy: float,
    mechanism: "dict | None",
    rul_reliable: bool,
    rul_pred: "float | None",
    rul_q10: "float | None",
    rul_q90: "float | None",
    warranty_floor_soh_pct: float = 70.0,
) -> list:
    """
    What a manufacturer would actually want to know about a fielded
    unit: is it degrading the way the design intended (mechanism
    verdict — a real quality signal, not just a number), and is it on
    track to breach its own warranty terms. Deliberately does NOT
    include the operator's maintenance action or financial figures —
    an OEM cares about fleet-wide product performance, not one
    operator's replace/repurpose decision.
    """
    from warranty import warranty_breach_estimate

    fields = [
        {"label": "Cell / unit ID", "value": cell_id, "state": "available"},
        {"label": "Chemistry", "value": chemistry, "state": "available"},
        {"label": "Data source", "value": source, "state": "available"},
        {"label": "Cycle count", "value": str(cycle_count), "state": "available"},
        {"label": "State of Health", "value": f"{soh:.1f}%", "state": "available"},
        {"label": "Fade rate (30-cycle)", "value": f"{fade_30_mah_cy * 1000:.2f} mAh/cycle", "state": "available"},
    ]
    if mechanism:
        fields.append({
            "label": "Degradation mechanism (quality signal)",
            "value": mechanism.get("verdict", "insufficient_data"),
            "state": "estimated" if mechanism.get("verdict") != "insufficient_data" else "unavailable",
            "note": f"Confidence: {mechanism.get('confidence', 'n/a')} — a real deviation from the expected fade pattern for this chemistry is a genuine early-warning signal for a design/manufacturing issue, not just an operational one",
        })

    wr = warranty_breach_estimate(
        current_soh_pct=soh, fade_rate_pct_per_cycle=fade_30_mah_cy * 100,
        warranty_floor_soh_pct=warranty_floor_soh_pct,
        rul_pred=rul_pred, rul_q10=rul_q10, rul_q90=rul_q90, rul_reliable=rul_reliable,
    )
    if wr["breached"]:
        fields.append({"label": f"Warranty risk ({warranty_floor_soh_pct:.0f}% floor, illustrative)", "value": "Already breached", "state": "estimated"})
    else:
        est = wr["model_scaled_estimate"] if wr["confidence"] == "model" else wr["linear_estimate"]
        fields.append({
            "label": f"Warranty risk ({warranty_floor_soh_pct:.0f}% floor, illustrative)",
            "value": f"~{est:.0f} cycles remaining" if est is not None else "Not computable",
            "state": "estimated" if wr["confidence"] == "model" else "unavailable",
            "note": "Model-scaled estimate reuses the LCO-validated RUL model" if wr["confidence"] == "model"
                    else "Linear extrapolation only — not leave-cell-out validated for this floor",
        })
    return fields


def build_operator_view(
    cell_id: str,
    chemistry: str,
    source: str,
    soh: float,
    cycle_count: int,
    fade_30_mah_cy: float,
    fade_50_mah_cy: float,
    fleet_fade_median: "float | None",
    rul_reliable: bool,
    rul_pred: "float | None",
    rul_q10: "float | None",
    rul_q90: "float | None",
    sop_pct: "float | None" = None,
) -> list:
    """
    What the current owner/fleet operator actually needs to act on
    today: the recommendation engine's own verdict, and the real dollar
    comparison behind it. Reuses recommendations.classify()/
    consequences.application_fit()/financial_comparison() directly,
    never re-derives a second opinion. fade_50_mah_cy is required (not
    approximated from fade_30) because classify()'s fade-acceleration
    signal is the ratio of the two windows — collapsing them to one
    value would silently disable that signal.
    """
    from consequences import ASSUMPTIONS, application_fit, financial_comparison
    from recommendations import classify

    fit_scores = application_fit(soh, fade_30_mah_cy, fleet_fade_median, sop_pct=sop_pct)
    result = classify(soh, fade_30_mah_cy, fade_50_mah_cy, rul_reliable, rul_pred if rul_reliable else None, fit_scores)

    a = {k: v["value"] for k, v in ASSUMPTIONS.items()}
    financial = financial_comparison(
        soh, source, a["recycling_value"], a["new_cell_cost"],
        a["second_life_value_per_kwh"], a["repack_cost"],
    )

    fields = [
        {"label": "Cell / unit ID", "value": cell_id, "state": "available"},
        {"label": "Chemistry", "value": chemistry, "state": "available"},
        {"label": "State of Health", "value": f"{soh:.1f}%", "state": "available"},
        {"label": "Recommended action", "value": result["action"].replace("_", " ").title(), "state": "available",
         "note": f"Confidence: {result['confidence']}"},
    ]
    if rul_reliable and rul_pred is not None:
        fields.append({
            "label": "Remaining Useful Life",
            "value": f"{rul_pred:.0f} cycles (P50)" + (f", {rul_q10:.0f}–{rul_q90:.0f} (P10–P90)" if rul_q10 is not None and rul_q90 is not None else ""),
            "state": "available",
        })
    else:
        fields.append({"label": "Remaining Useful Life", "value": "Not reliable at this cycle count", "state": "unavailable"})
    fields.extend([
        {"label": "Second-life reuse value (net)", "value": f"${financial['sl_net']:,.2f}", "state": "estimated"},
        {"label": "Immediate recycle value", "value": f"${financial['recycle_value']:,.2f}", "state": "estimated"},
        {"label": "New cell replacement cost", "value": f"${financial['new_cell_cost']:,.2f}", "state": "estimated"},
    ])
    return fields


def build_recycler_view(
    cell_id: str,
    chemistry: str,
    soh: float,
    fade_30_mah_cy: float,
    sop_pct: "float | None" = None,
    user_region: "str | None" = None,
) -> list:
    """
    What a recycler actually needs: chemistry (determines process line),
    condition (determines R-code/pathway), and a routing recommendation
    — not the operator's financial comparison or the OEM's warranty
    exposure, neither of which changes how a recycler processes the
    cell. Reuses eol_r_code_recommendation() and recommend_recyclers()
    directly.
    """
    from consequences import eol_r_code_recommendation
    from recycler_directory import recommend_recyclers

    r_code = eol_r_code_recommendation(soh, fade_30_mah_cy, sop_pct=sop_pct)
    fields = [
        {"label": "Cell / unit ID", "value": cell_id, "state": "available"},
        {"label": "Chemistry", "value": chemistry, "state": "available"},
        {"label": "State of Health", "value": f"{soh:.1f}%", "state": "available"},
        {"label": "End-of-life recommendation (IEC 62902)", "value": r_code["r_code"], "state": "estimated"},
    ]
    if r_code["r_code"].startswith(("R4", "R5")):
        matches = recommend_recyclers(chemistry, user_region=user_region, top_n=3)
        if matches:
            fields.append({
                "label": "Recommended recyclers", "value": ", ".join(m["name"] for m in matches),
                "state": "estimated",
                "note": "Point-in-time research snapshot (2026-08) — verify current operating status before routing a real cell",
            })
    else:
        fields.append({
            "label": "Recommended recyclers", "value": "Not applicable",
            "state": "unavailable", "note": "This cell's recommendation doesn't call for recycling yet",
        })
    return fields


STAKEHOLDER_BUILDERS = {
    "oem": build_oem_view,
    "operator": build_operator_view,
    "recycler": build_recycler_view,
}
