"""Page: Decision (merged Recommendations + EOL Economics)."""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils import _action_bar, _md_html, _empty_state, render_card, metric_tile_html
from design_system import make_badge, ACTION_META, CONF_META
from chemistry_profiles import ChemistryProfile

from _pages.consequences import page_consequences


# ---------------------------------------------------------------------------
# Page: Decision  (merged Recommendations + EOL Economics)
# ---------------------------------------------------------------------------

def page_decision(
    selected: str,
    df: pd.DataFrame,
    featured_dfs: dict,
    bundles: dict,
    rul_reliable: bool,
    graph=None,
):
    """Merged Recommendations + EOL Economics page.

    Layout:
      1. Hero recommendation card (action + confidence)
      2. NPV 3-column decision table (Replace / Wait / Repurpose)
      3. Maintenance calendar
      4. Application fit scores
      5. Log decision + SoC window
      Full economics in expander below.
    """
    _action_bar("decision")
    from recommendations import classify, SOH_PRIMARY_FLOOR, SOH_INSPECT_FLOOR, SOH_SECONDLIFE_FLOOR
    from consequences import (
        ASSUMPTIONS, SECOND_LIFE_APPS, CELL_NOMINAL_KWH,
        application_fit, financial_comparison, sustainability_snapshot, breakeven_curve,
    )

    latest          = df.iloc[-1]
    soh             = float(latest["soh_pct"])
    fade_30         = float(latest.get("fade_rate_30cy", 0.0))
    fade_50         = float(latest.get("fade_rate_50cy", 0.0))
    _profile        = ChemistryProfile.for_cell(selected)
    source          = _profile.source_kind
    rul_pred_raw    = latest.get("rul_pred", None)
    rul_pred        = float(rul_pred_raw) if (rul_reliable and rul_pred_raw is not None) else None
    rul_q10_raw     = latest.get("rul_q10", None)
    rul_q10         = float(rul_q10_raw) if (rul_reliable and rul_q10_raw is not None) else None
    rul_q90_raw     = latest.get("rul_q90", None)
    rul_q90         = float(rul_q90_raw) if (rul_reliable and rul_q90_raw is not None) else None

    # Reads precomputed CellSummary rows instead of every peer cell's full
    # per-cycle DataFrame -- see src/cell_store.py's module docstring.
    import db as _db_decision
    _active_ids_dec = set(featured_dfs.keys())
    peer_fades = [
        float(r["fade_rate_30cy"] or 0)
        for r in _db_decision.get_cell_summaries(st.session_state["auth_org_id"])
        if r["cell_id"] in _active_ids_dec
        and ChemistryProfile.for_cell(r["cell_id"]).source_kind == _profile.source_kind
        and r["cell_id"] != selected
    ]
    fleet_fade_median = float(pd.Series(peer_fades).median()) if peer_fades else None
    fit_scores = application_fit(soh, fade_30, fleet_fade_median)
    result     = classify(soh, fade_30, fade_50, rul_reliable, rul_pred, fit_scores)

    action          = result["action"]
    action_label, action_colour, action_bg = ACTION_META[action]
    conf_colour, conf_label = CONF_META[result["confidence"]]

    st.markdown(f"# What should I do with {selected}?")

    # Mechanism verdict read from the shared Battery Knowledge Graph exhibits
    # edge (src/knowledge_graph.py) — the same edge Health's compact card and
    # the Copilot's "mechanism" answer read, so this page can no longer
    # silently disagree with either about the same cell's verdict. Computed
    # before the hero card so a real disagreement between the mechanism
    # verdict and the SOH-based recommendation (Decision Support review
    # finding: classify() never sees this signal) can be surfaced as a
    # caution line inside the recommendation itself, not just displayed
    # adjacent to it as unrelated context.
    try:
        from recommendations import mechanism_corroboration_note, mechanism_second_life_caution
        if graph is not None:
            from knowledge_graph import get_or_compute_mechanism
            _dec_mech = get_or_compute_mechanism(graph, selected, df)
        else:
            from recommendations import diagnose_mechanism
            _dec_mech = diagnose_mechanism(df)
        _corroboration_note = mechanism_corroboration_note(action, _dec_mech)
        _second_life_caution = mechanism_second_life_caution(fit_scores, _dec_mech)
    except Exception:
        _dec_mech = None
        _corroboration_note = None
        _second_life_caution = None

    # ── 1. Hero recommendation ──────────────────────────────────────────────
    reason_html = "".join(
        f"<div style='margin-top:6px;font-size:13px;color:{action_colour}cc'>· {r}</div>"
        for r in result["action_reasons"]
    )
    if _corroboration_note:
        reason_html += (
            f"<div style='margin-top:8px;font-size:13px;color:#f6ad55;"
            f"border-top:1px solid {action_colour}33;padding-top:8px'>⚠ {_corroboration_note}</div>"
        )
    _md_html(
        f"<div style='background:{action_bg};border:2px solid {action_colour}55;"
        f"border-radius:14px;padding:24px 28px;margin-bottom:20px'>"
        f"<div style='font-size:10px;font-weight:700;color:{action_colour}99;"
        f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Recommended Action</div>"
        f"<div style='font-size:32px;font-weight:800;color:{action_colour}'>{action_label}</div>"
        f"<div style='margin-top:8px'>"
        f"<span style='background:{conf_colour}22;border:1px solid {conf_colour}55;color:{conf_colour};"
        f"font-size:11px;font-weight:700;padding:3px 10px;border-radius:10px'>{conf_label}</span>"
        f"</div>{reason_html}</div>"
    )

    # ── Compact mechanism verdict (U3: merged from Health page's LLI/LAM classifier) ──
    if _dec_mech is not None:
        st.markdown(
            f"<div style='background:#111827;border-left:3px solid {_dec_mech['verdict_color']};"
            f"border-radius:6px;padding:8px 14px;margin:-12px 0 20px;font-size:12px;color:#a0aec0'>"
            f"⚗️ <strong style='color:{_dec_mech['verdict_color']}'>{_dec_mech['verdict']}</strong>"
            f" <span style='color:#a0aec0'>({_dec_mech['confidence_label']} confidence)</span>"
            f" — {_dec_mech['verdict_body'].split('.')[0]}."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── 2. NPV Decision Table (3 options, no chart) ─────────────────────────
    st.markdown("<h4 class='section-header'>Financial Decision</h4>", unsafe_allow_html=True)
    # This 5-yr NPV Scenario Planner deliberately uses its own assumption set,
    # NOT consequences.ASSUMPTIONS — verified this is intentional, not drift,
    # before touching it: _repl_cost is a BNEF 2024 *installed/system-level*
    # replacement-cost estimate ($100-200/cell) for a long-run operational
    # DCF decision, while ASSUMPTIONS["new_cell_cost"] ($20, range $5-60) is
    # a *bare-cell spot price* citation used by the separate, point-in-time
    # Reuse/Recycle/Buy-new comparison (financial_comparison(), also shown
    # on this platform — e.g. EOL Economics' top cards, the Copilot's
    # "business case" chip). Both figures are real and correctly sourced for
    # their own purpose; they are not the same real-world quantity, so they
    # are NOT unified here — see the "ⓘ Assumptions" popover below, which
    # now says so explicitly, so a reader bouncing between the two doesn't
    # mistake this for the same bug class as consequences.best_fit_application()
    # (an independently re-derived ladder that SHOULD agree with the original).
    # consequences.py's own NPV Scenario Planner section uses the identical
    # $80/$150/$30 figures for the same reason — keep both in sync if either
    # changes.
    _npv_rate   = 0.08
    _energy_usd = 80.0   # IEA 2024 LCOS range $60-140/kWh·yr — NPV Scenario Planner only
    _repl_cost  = 150.0  # BNEF 2024 range $100-200/cell (installed) — NPV Scenario Planner only
    _years      = list(range(1, 6))
    def _pv(r, t): return 1.0 / ((1 + r) ** t)
    _rul_yrs = min((rul_pred / 200.0 / 12.0) if rul_pred else 1.5, 5.0)

    _cap_now  = CELL_NOMINAL_KWH.get(source, 0.0057)
    _a_npv = sum(_cap_now * _energy_usd * _pv(_npv_rate, t) for t in _years) - _repl_cost

    _b_cap = _cap_now * (soh / 100.0)
    _b_npv = (
        sum(_b_cap * _energy_usd * _pv(_npv_rate, t) for t in range(1, max(1, int(_rul_yrs)) + 1))
        + sum(_cap_now * _energy_usd * _pv(_npv_rate, t) for t in range(max(1, int(_rul_yrs)) + 1, 6))
        - _repl_cost * _pv(_npv_rate, _rul_yrs)
    )

    _sl_cap   = _cap_now * (soh / 100.0) * 0.85
    _repack   = 30.0   # NPV Scenario Planner only — see note above _npv_rate; not ASSUMPTIONS["repack_cost"]
    _c_npv    = sum(_sl_cap * _energy_usd * 0.6 * _pv(_npv_rate, t) for t in _years) - _repack

    _opts = [
        ("Replace Now",          _a_npv, "#fc8181",
         "Immediate outlay. New cell, full life ahead.", f"${_repl_cost:.0f} upfront"),
        ("Wait to EOL",          _b_npv, "#f6ad55",
         f"~{_rul_yrs:.1f} yr remaining at current fade rate.", "Risk of degraded performance"),
        ("Repurpose (2nd life)", _c_npv, "#68d391",
         "Lower-demand application (stationary storage).", f"${_repack:.0f} repack cost"),
    ]
    _npv_max = max(_opts, key=lambda x: x[1])

    # The financial widget used to badge whichever option had the highest
    # NPV as "★ Recommended", completely independent of the operational
    # verdict above it -- live-reproduced: the hero card said "Continue
    # Operation, High confidence" while this widget's own "★ Recommended"
    # badge sat on "Repurpose (2nd life)" at a negative NPV, two contradictory
    # "recommended" badges on one screen. The financial section must always
    # foreground the option consistent with the operational headline instead;
    # the NPV-maximizing option is still shown, but only as a labelled
    # alternative when it actually differs.
    _ACTION_TO_NPV_LABEL = {
        "continue":     "Wait to EOL",
        "inspect":      "Wait to EOL",
        "second_life":  "Repurpose (2nd life)",
        "recycle":      "Replace Now",
    }
    _aligned_lbl = _ACTION_TO_NPV_LABEL.get(action, _npv_max[0])
    _aligned_opt = next((o for o in _opts if o[0] == _aligned_lbl), _npv_max)
    _best_lbl, _best_npv_v, _best_col_v, _best_desc, _best_cost_note = _aligned_opt
    _financial_disagrees = _aligned_opt[0] != _npv_max[0]

    _badge_text = "★ Consistent with recommendation" if _financial_disagrees else "★ Recommended"
    _md_html(
        f"<div style='background:#1e2a38;border:2px solid {_best_col_v};"
        f"border-radius:12px;padding:20px 24px;margin-bottom:10px'>"
        f"<div style='font-size:10px;font-weight:700;color:{_best_col_v};"
        f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px'>{_badge_text}</div>"
        f"<div style='font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:8px'>{_best_lbl}</div>"
        f"<div style='font-size:40px;font-weight:800;color:{_best_col_v}'>${_best_npv_v:,.0f}</div>"
        f"<div style='font-size:10px;color:#a0aec0;margin-top:2px'>5-yr NPV at 8% discount rate</div>"
        f"<div style='font-size:12px;color:#a0aec0;margin-top:10px;line-height:1.5'>{_best_desc}</div>"
        f"<div style='font-size:11px;color:#a0aec0;margin-top:4px'>{_best_cost_note}</div>"
        + (
            f"<div style='font-size:11px;color:#94a3b8;margin-top:10px;padding-top:10px;"
            f"border-top:1px solid #2d3748'>{_npv_max[0]} has the higher 5-yr NPV "
            f"(${_npv_max[1]:,.0f}), but this platform's operational recommendation above "
            f"is {action_label} — see Compare alternatives for the full picture.</div>"
            if _financial_disagrees else ""
        )
        + f"</div>"
    )
    with st.expander("Compare alternatives"):
        _alts = [o for o in _opts if o[0] != _best_lbl]
        _nd_cols = st.columns(len(_alts))
        for _col, (_lbl, _npv_v, _col_v, _desc, _cost_note) in zip(_nd_cols, _alts):
            with _col:
                _tile_label = _lbl + (" · highest NPV" if _lbl == _npv_max[0] else "")
                render_card(
                    metric_tile_html(_tile_label, f"${_npv_v:,.0f}", "5-yr NPV",
                                      value_color=_col_v, value_size="26px")
                    + f"<div style='font-size:11px;color:#8896a8;margin-top:8px;line-height:1.5'>{_desc}</div>"
                    f"<div style='font-size:10px;color:#a0aec0;margin-top:4px'>{_cost_note}</div>",
                    extra_style="height:100%",
                )

    with st.popover("ⓘ Assumptions", use_container_width=False):
        st.markdown(
            f"<div style='font-size:12px;color:#a0aec0;line-height:1.8'>"
            f"<strong style='color:#e2e8f0'>5-yr NPV · {_npv_rate*100:.0f}% discount rate</strong><br>"
            f"WACC assumption — adjust in the NPV Scenario Planner below.<br><br>"
            f"<strong style='color:#e2e8f0'>$80/kWh·yr energy value</strong> "
            f"{make_badge('Illustrative', '#718096')}<br>"
            f"IEA 2024 LCOS range $60–140/kWh·yr.<br><br>"
            f"<strong style='color:#e2e8f0'>${_repl_cost:.0f}/cell replacement</strong> "
            f"{make_badge('Illustrative', '#718096')}<br>"
            f"BNEF 2024 range $100–200/cell.<br><br>"
            f"<strong style='color:#e2e8f0'>${_repack:.0f} repack cost</strong> "
            f"{make_badge('Illustrative', '#718096')}<br>"
            f"Engineering estimate.<br><br>"
            f"<span style='color:#8896a8;font-size:11px'>These figures are a separate, longer-run "
            f"assumption set for this 5-yr NPV model — not the same as the "
            f"${ASSUMPTIONS['new_cell_cost']['value']:.0f}/cell "
            f"bare-cell spot price used in the Reuse/Recycle/Buy-new comparison elsewhere on this "
            f"platform (EOL Economics, the Copilot's \"business case\" chip). ${_repl_cost:.0f} here reflects "
            f"an installed/system-level replacement estimate, not a component price — the two are "
            f"intentionally different figures for different decisions, not a discrepancy.</span><br><br>"
            f"<em style='color:#a0aec0'>Not financial advice.</em>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── The story: one paragraph synthesizing the hero, mechanism, fit, and ──
    # NPV cards above (Data Storytelling review finding — these used to sit
    # as 4 unconnected cards with no "here's what's happening and why"
    # narrative tying them together). Reads only already-computed values, so
    # it can't drift from what the cards above already say.
    from battery_copilot import answer_cell_story
    _story_text = answer_cell_story({
        "cell_id": selected,
        "soh": soh,
        "action_label": action_label,
        "confidence_label": conf_label,
        "mechanism": _dec_mech,
        "fit_scores": fit_scores,
        "financial_best_label": _best_lbl,
        "financial_best_npv": _best_npv_v,
        "financial_disagrees": _financial_disagrees,
        "npv_max_label": _npv_max[0],
        "npv_max_value": _npv_max[1],
    })
    st.markdown(
        f"<div style='font-size:13px;color:#a0aec0;line-height:1.7;margin:12px 0 16px;"
        f"padding:14px 18px;background:#1a202c;border-radius:8px;"
        f"border-left:3px solid #2d3748'>{_story_text}</div>",
        unsafe_allow_html=True,
    )

    # ── 📋 Supporting details (U2 density reduction): maintenance calendar, ──
    # application fit, second-life marketplace, confidence-reason callouts ──
    with st.expander("📋 Supporting details", expanded=False):
        # ── 3. Maintenance Calendar ─────────────────────────────────────────
        st.markdown("<h4 class='section-header'>Maintenance Calendar</h4>", unsafe_allow_html=True)
        _eol_thr    = float(st.session_state.get("eol_threshold_pct", 80.0))
        _cur_soh    = float(df["soh_pct"].iloc[-1])
        _fp_cy      = float(df["fade_rate_50cy"].iloc[-1]) * 100 / (float(df["capacity_ah"].iloc[0]) + 1e-9)
        _rul_cy     = max(0, (_cur_soh - _eol_thr) / _fp_cy) if _fp_cy > 1e-6 else None

        if "test_date" in df.columns and df["test_date"].notna().any():
            _dates = pd.to_datetime(df["test_date"].dropna())
            _cpd   = len(df) / max((_dates.iloc[-1] - _dates.iloc[0]).days, 1)
        else:
            _cpd   = 1.0

        if _rul_cy is not None and _rul_cy > 0:
            from datetime import date as _date, timedelta as _td
            _days    = _rul_cy / _cpd
            _rep_dt  = _date.today() + _td(days=_days)
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Recommended Replacement", _rep_dt.strftime("%B %Y"))
            _c2.metric("Cycles Remaining", f"{_rul_cy:.0f}")
            _c3.metric("Days Remaining",   f"{_days:.0f}")
        else:
            _empty_state(
                "Replacement timeline unavailable",
                "Insufficient cycle history to compute a fade-based schedule. "
                "At least 50 cycles with a measurable fade trend are required.",
                "", "📅",
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── 4. Application Fit ──────────────────────────────────────────────
        if soh <= 90:
            st.markdown("<h4 class='section-header'>Application Fit Scores</h4>", unsafe_allow_html=True)
            _af_cols = st.columns(min(len(fit_scores), 4))
            _af_colour = {"fit": "#48bb78", "marginal": "#f6ad55", "not_fit": "#fc8181"}
            for _i, (_app_key, _app) in enumerate(fit_scores.items()):
                with _af_cols[_i % len(_af_cols)]:
                    render_card(
                        f"<div style='font-size:11px;color:#a0aec0'>{_app['short']}</div>"
                        f"<div style='font-size:16px;font-weight:700;color:{_af_colour[_app['fit']]};margin-top:4px'>"
                        f"{_app['fit'].replace('_', ' ').title()}</div>",
                        padding="14px 16px",
                    )
            if _second_life_caution:
                st.markdown(
                    f"<div style='font-size:12px;color:#f6ad55;margin-top:8px'>⚠ {_second_life_caution}</div>",
                    unsafe_allow_html=True,
                )

        # ── 4b. Second-life marketplace ──────────────────────────────────────
        if action in ("second_life", "recycle") or soh <= 85:
            st.markdown("<h4 class='section-header'>Second-Life Marketplace</h4>", unsafe_allow_html=True)
            _sl_eligible = soh >= 70.0
            _sl_col = "#68d391" if _sl_eligible else "#718096"
            render_card(
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
                f"<div>"
                f"<div style='font-size:14px;font-weight:700;color:{_sl_col}'>"
                f"{'Eligible for second-life listing' if _sl_eligible else 'Below second-life floor (SOH < 70%)'}</div>"
                f"<div style='font-size:12px;color:#8896a8;margin-top:6px;line-height:1.6'>"
                f"Cell {selected} · SOH {soh:.1f}% · Est. residual capacity {soh * _cap_now / 100 * 1000:.0f} mAh<br>"
                f"Suitable for: stationary storage, UPS backup, low-power IoT applications"
                f"</div>"
                f"</div>"
                f"<div style='font-size:22px;color:{_sl_col};margin-left:16px'>♻</div>"
                f"</div>",
                border_color=f"{_sl_col}44",
                padding="18px 22px",
                extra_style="margin-bottom:12px",
            )
            if _sl_eligible:
                _mk_col1, _mk_col2 = st.columns(2)
                if _mk_col1.button(
                    "List on Circunomics →", key="sl_list_circ",
                    help="Opens second-life battery exchange (demo — not a live API call)",
                    use_container_width=True,
                ):
                    _circ_chemistry = ChemistryProfile.for_cell(selected).short_name
                    _circ_capacity_ah = round(_cap_now * (soh / 100) * 1000 / 3.7, 2)
                    _circ_asking_usd = round(_c_npv * 0.4, 2)
                    _circ_api_key = st.session_state.get("circunomics_api_key", "")
                    _circ_result = None
                    if _circ_api_key:
                        from circunomics_adapter import list_cell_on_circunomics
                        _circ_result = list_cell_on_circunomics(
                            selected, soh, _circ_chemistry, _circ_capacity_ah,
                            _circ_asking_usd, _circ_api_key,
                        )
                    from adapter_contract import classify_result
                    _circ_state = classify_result(_circ_result)
                    if _circ_state == "success":
                        st.success(
                            f"Listing submitted to Circunomics for {selected} at "
                            f"${_circ_asking_usd:.2f}."
                        )
                    else:
                        _listing = {
                            "cell_id":        selected,
                            "soh_pct":        round(soh, 1),
                            "chemistry":      _circ_chemistry,
                            "capacity_ah":    _circ_capacity_ah,
                            "asking_usd":     _circ_asking_usd,
                            "listed_at":      datetime.datetime.now().isoformat(),
                            "platform":       "battery-intelligence-platform",
                            "note":           "Demo listing — not submitted to a live exchange",
                        }
                        if "sl_listings" not in st.session_state:
                            st.session_state["sl_listings"] = []
                        st.session_state["sl_listings"].append(_listing)
                        if _circ_state == "error":
                            st.error(f"Circunomics submission failed ({_circ_result['error']}) — saved as a local demo listing instead.")
                        else:
                            st.success(
                                f"Listing created for {selected} at ${_listing['asking_usd']:.2f} "
                                f"(demo — no real API call made). Configure a Circunomics API key in "
                                f"Settings to submit real listings."
                            )
                if _mk_col2.button(
                    "List on Battery-Lifecycle.com →", key="sl_list_blc",
                    help="Second-life exchange for industrial battery packs (demo)",
                    use_container_width=True,
                ):
                    st.info(
                        "Battery Lifecycle Company integration requires an API key. "
                        "In production: POST /api/v1/listings with cell ID, SOH, chemistry, "
                        "capacity, and asking price. See Settings to configure the endpoint."
                    )
            _sl_listings = st.session_state.get("sl_listings", [])
            if _sl_listings:
                with st.expander(f"Active listings ({len(_sl_listings)})"):
                    import pandas as _pd_sl
                    st.dataframe(
                        _pd_sl.DataFrame(_sl_listings),
                        use_container_width=True, hide_index=True,
                    )
                    st.download_button(
                        "Export listings",
                        data=_pd_sl.DataFrame(_sl_listings).to_csv(index=False).encode(),
                        file_name="secondlife_listings.csv",
                        mime="text/csv",
                        key="sl_listings_export",
                    )

        # ── Confidence-reason callouts ────────────────────────────────────────
        if result["confidence_reasons"]:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for note in result["confidence_reasons"]:
                note_colour = "#b7791f" if "fit scores" in note else "#718096"
                st.markdown(
                    f"<div style='background:{note_colour}11;border:1px solid {note_colour}33;"
                    f"border-radius:8px;padding:10px 16px;margin-bottom:8px;"
                    f"font-size:12px;color:#a0aec0'>{note}</div>",
                    unsafe_allow_html=True,
                )

    # ── 5. Log Decision ─────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _log_col, _cmms_col, _spine_col, _ = st.columns([1, 1, 1, 1])
    if _log_col.button("Log Decision", key="dec_log_btn", use_container_width=True):
        import db as _db
        _entry = {
            "id":         f"{selected}_{datetime.datetime.now().strftime('%H%M%S')}",
            "cell_id":    selected,
            "action":     action_label,
            "confidence": conf_label,
            "soh_pct":    round(soh, 1),
            "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status":     "Pending",
            "outcome_soh": None,
        }
        if "decision_log" not in st.session_state:
            st.session_state["decision_log"] = []
        st.session_state["decision_log"].append(_entry)
        _db.save_decision(st.session_state["auth_org_id"], _entry)
        st.success(f"Logged: {action_label} for {selected}")

    # ── U4: CMMS/ERP write-back ──────────────────────────────────────────────
    _cmms_api_key = st.session_state.get("cmms_api_key", "")
    if _cmms_col.button(
        "Create CMMS ticket", key="dec_cmms_btn", use_container_width=True,
        disabled=not _cmms_api_key,
        help="Create a maintenance ticket in your CMMS/ERP system for this recommendation. "
             "Configure a CMMS API key in Settings to enable this."
             if not _cmms_api_key else
             "Creates a maintenance ticket in your configured CMMS/ERP system.",
    ):
        from cmms_adapter import create_maintenance_ticket
        from adapter_contract import classify_result
        _cmms_base_url = st.session_state.get("cmms_api_base_url", "") or "https://api.example-cmms.com/v1"
        _cmms_title = f"{action_label} — {selected}"
        _cmms_desc = (
            f"Recommended action: {action_label} (confidence: {conf_label}). "
            f"SOH: {soh:.1f}%. " + " ".join(result["action_reasons"])
        )
        _cmms_priority = "high" if action in ("recycle",) else "medium" if action == "second_life" else "low"
        _cmms_result = create_maintenance_ticket(
            selected, _cmms_title, _cmms_desc, _cmms_priority, _cmms_api_key,
            api_base_url=_cmms_base_url,
        )
        _cmms_state = classify_result(_cmms_result)
        if _cmms_state == "unconfigured":
            st.warning("No CMMS API key configured — set one up in Settings first.")
        elif _cmms_state == "error":
            st.error(f"CMMS ticket creation failed: {_cmms_result['error']}")
        else:
            st.success(f"CMMS ticket created for {selected}.")

    # Spine-compatible second-life data export — lets a grid-modeling team
    # import this cell's health + second-life economics into a SpineDB-backed
    # model (e.g. SpineOpt/FlexTool) instead of hand-transcribing values off
    # this page. See src/spine_export.py's module docstring for schema
    # provenance and explicit scope boundaries (no trajectory, no live DB link).
    import json as _json_spine
    from spine_export import build_second_life_export, to_export_document
    from batlab.datasets.schema import condition_completeness
    _condition_completeness = condition_completeness(df)
    _spine_data = build_second_life_export(
        selected, source, _profile.passport_chemistry, soh, fade_30,
        fleet_fade_median, rul_reliable,
        rul_q10=rul_q10, rul_pred=rul_pred, rul_q90=rul_q90,
        mechanism=_dec_mech,
    )
    _spine_doc = to_export_document(_spine_data, selected, condition_completeness=_condition_completeness)
    _spine_col.download_button(
        "Export for grid modeling",
        data=_json_spine.dumps(_spine_doc, indent=2).encode(),
        file_name=f"{selected}_spine_export.json",
        mime="application/json",
        key="dec_spine_export_btn",
        use_container_width=True,
        help="Download this cell's SOH/RUL/second-life economics as a Spine "
             "Toolbox-compatible JSON file (spinedb_api import_data() format) "
             "for use in a grid-storage modeling tool.",
    )

    # E4: Audit trail with status chips and outcome tracking
    _dlog = st.session_state.get("decision_log", [])
    if _dlog:
        with st.expander(f"Decision Audit Trail ({len(_dlog)} entries)", expanded=False):
            _STATUS_COLOURS = {
                "Pending": "#718096", "Approved": "#63b3ed",
                "Completed": "#48bb78", "Verified": "#9f7aea",
            }
            for _i, _dl in enumerate(_dlog):
                _sc   = _STATUS_COLOURS.get(_dl.get("status", "Pending"), "#718096")
                _sc22 = _sc + "22"
                _cols_e4 = st.columns([3, 2, 2, 2, 1])
                _cols_e4[0].markdown(
                    f"<div style='padding:6px 0'>"
                    f"<div style='font-size:13px;font-weight:600;color:#e2e8f0'>"
                    f"{_dl['cell_id']} — {_dl['action']}</div>"
                    f"<div style='font-size:11px;color:#a0aec0'>{_dl['timestamp']} · SOH {_dl['soh_pct']}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                _new_status = _cols_e4[1].selectbox(
                    "Status", options=["Pending", "Approved", "Completed", "Verified"],
                    index=["Pending", "Approved", "Completed", "Verified"].index(
                        _dl.get("status", "Pending")),
                    key=f"e4_status_{_i}",
                    label_visibility="collapsed",
                )
                if _new_status != _dl.get("status"):
                    st.session_state["decision_log"][_i]["status"] = _new_status
                    import db as _db
                    _db.update_decision(st.session_state["auth_org_id"], _dl["id"], status=_new_status)
                    st.rerun()
                _cols_e4[2].markdown(
                    f"<div style='padding:8px 0'>"
                    f"<span style='background:{_sc22};border:1px solid {_sc}44;color:{_sc};"
                    f"font-size:11px;font-weight:700;padding:3px 10px;border-radius:10px'>"
                    f"{_dl.get('status', 'Pending')}</span></div>",
                    unsafe_allow_html=True,
                )
                # Outcome tracking: prompt for SOH when marked Completed
                if _dl.get("status") == "Completed" and _dl.get("outcome_soh") is None:
                    _new_soh = _cols_e4[3].number_input(
                        "Actual SOH after action (%)", min_value=0.0, max_value=100.0,
                        value=float(_dl["soh_pct"]), step=0.5,
                        key=f"e4_soh_{_i}", label_visibility="collapsed",
                    )
                    if _cols_e4[4].button("Save", key=f"e4_save_{_i}"):
                        st.session_state["decision_log"][_i]["outcome_soh"] = round(_new_soh, 1)
                        st.session_state["decision_log"][_i]["status"] = "Verified"
                        import db as _db
                        _db.update_decision(st.session_state["auth_org_id"], _dl["id"], outcome_soh=round(_new_soh, 1), status="Verified")
                        st.rerun()
                elif _dl.get("outcome_soh") is not None:
                    _delta_soh = _dl["outcome_soh"] - _dl["soh_pct"]
                    _delta_col = "#48bb78" if _delta_soh >= 0 else "#fc8181"
                    _cols_e4[3].markdown(
                        f"<div style='padding:8px 0;font-size:12px;color:{_delta_col}'>"
                        f"Outcome: {_dl['outcome_soh']}% "
                        f"({_delta_soh:+.1f}%)</div>",
                        unsafe_allow_html=True,
                    )
            import pandas as _pd_log
            st.download_button(
                "Export log as CSV",
                data=_pd_log.DataFrame(_dlog).to_csv(index=False).encode(),
                file_name="battery_decision_log.csv",
                mime="text/csv",
                key="dec_export_log",
            )

    # ── Full Economics (expander) ────────────────────────────────────────────
    # mode_landing_ess: set (and cleared) by the "Plan a storage deployment"
    # use-case landing in app/main.py — read here (not popped) so this outer
    # expander also opens on that one arrival rerun; page_consequences() pops
    # the same flag for its own nested Solar + Storage Sizing expander.
    _ess_landing = st.session_state.get("mode_landing_ess", False)
    with st.expander("Full economics & breakeven analysis", expanded=_ess_landing):
        page_consequences(selected, df, featured_dfs, bundles, rul_reliable)

    # ── Inline Copilot panel (merged Decision + Copilot) ─────────────────────
    st.markdown("<h4 class='section-header'>Ask about this cell</h4>", unsafe_allow_html=True)
    st.caption("Explains the recommendation above in plain language — grounded only on values already computed by the model pipeline. It does not make the decision; the recommendation engine above does.")
    _dc_bundle = bundles.get(ChemistryProfile.for_cell(selected).source_kind)
    if _dc_bundle:
        try:
            from battery_copilot import build_cell_context, answer_query, llm_answer
            _dc_ctx = build_cell_context(selected, featured_dfs, bundles)
            _dc_input = st.text_input(
                "Question", placeholder="e.g. 'Why is this cell degrading faster than others?'",
                key="decision_copilot_input", label_visibility="collapsed",
            )
            _dc_chips_row = st.columns(4)
            _dc_presets = [
                ("What should I do?",   "what_to_do"),
                ("Replacement budget?", "budget"),
                ("What's the risk?",    "risk"),
                ("Repurpose options?",  "repurpose"),
            ]
            for _dci, (_dclbl, _dckey) in enumerate(_dc_presets):
                if _dc_chips_row[_dci].button(_dclbl, key=f"dc_chip_{_dckey}", use_container_width=True):
                    st.session_state["decision_copilot_input"] = _dclbl
                    st.rerun()
            if _dc_input:
                _api_key_dc = st.session_state.get("anthropic_api_key", "")
                _template_dc = answer_query(_dc_input, _dc_ctx)
                _dc_answer = llm_answer(_dc_input, _template_dc, _api_key_dc) if _api_key_dc else _template_dc
                _badge_dc = make_badge("Claude Haiku", "#667eea") if _api_key_dc else make_badge("Template", "#718096")
                render_card(
                    f"<div style='font-size:10px;color:#a0aec0;margin-bottom:8px'>{_badge_dc} · {selected}</div>"
                    f"<div style='font-size:13px;color:#e2e8f0;line-height:1.7'>{_dc_answer}</div>",
                    padding="16px 20px",
                    extra_style="margin-top:8px",
                )
        except Exception as _dc_e:
            st.caption(f"Copilot unavailable: {_dc_e}")


