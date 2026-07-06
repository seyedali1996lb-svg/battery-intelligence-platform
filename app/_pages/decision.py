"""Page: Decision (merged Recommendations + EOL Economics)."""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils import _action_bar, _md_html, _empty_state, NASA_CELL_IDS
from design_system import make_badge, ACTION_META, CONF_META
from data_loader import CELL_STRESS_PROFILES

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
    is_nasa         = selected in NASA_CELL_IDS
    source          = "nasa" if is_nasa else "synth"
    rul_pred_raw    = latest.get("rul_pred", None)
    rul_pred        = float(rul_pred_raw) if (rul_reliable and rul_pred_raw is not None) else None

    peer_fades = [
        float(fdf.iloc[-1].get("fade_rate_30cy", 0))
        for cid, fdf in featured_dfs.items()
        if (cid in NASA_CELL_IDS) == is_nasa and cid != selected
    ]
    fleet_fade_median = float(pd.Series(peer_fades).median()) if peer_fades else None
    fit_scores = application_fit(soh, fade_30, fleet_fade_median)
    result     = classify(soh, fade_30, fade_50, rul_reliable, rul_pred, fit_scores)

    action          = result["action"]
    action_label, action_colour, action_bg = ACTION_META[action]
    conf_colour, conf_label = CONF_META[result["confidence"]]

    st.markdown(f"# What should I do with {selected}?")

    # ── 1. Hero recommendation ──────────────────────────────────────────────
    reason_html = "".join(
        f"<div style='margin-top:6px;font-size:13px;color:{action_colour}cc'>· {r}</div>"
        for r in result["action_reasons"]
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
    try:
        from recommendations import diagnose_mechanism
        _dec_mech = diagnose_mechanism(df)
        st.markdown(
            f"<div style='background:#111827;border-left:3px solid {_dec_mech['verdict_color']};"
            f"border-radius:6px;padding:8px 14px;margin:-12px 0 20px;font-size:12px;color:#a0aec0'>"
            f"⚗️ <strong style='color:{_dec_mech['verdict_color']}'>{_dec_mech['verdict']}</strong>"
            f" <span style='color:#4a5568'>({_dec_mech['confidence_label']} confidence)</span>"
            f" — {_dec_mech['verdict_body'].split('.')[0]}."
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # ── 2. NPV Decision Table (3 options, no chart) ─────────────────────────
    st.markdown("<div class='section-header'>Financial Decision</div>", unsafe_allow_html=True)
    _npv_rate   = 0.08
    _energy_usd = 80.0
    _repl_cost  = 150.0
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
    _repack   = 30.0
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
        f"<div style='font-size:10px;color:#718096;margin-top:2px'>5-yr NPV at 8% discount rate</div>"
        f"<div style='font-size:12px;color:#a0aec0;margin-top:10px;line-height:1.5'>{_best_desc}</div>"
        f"<div style='font-size:11px;color:#4a5568;margin-top:4px'>{_best_cost_note}</div>"
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
                _md_html(
                    f"<div style='background:#1e2a38;border:1px solid #2d3748;"
                    f"border-radius:10px;padding:16px 18px;height:100%'>"
                    f"<div style='font-size:10px;font-weight:700;color:#4a5568;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin-bottom:6px'>{_lbl}"
                    + (" · highest NPV" if _lbl == _npv_max[0] else "")
                    + f"</div>"
                    f"<div style='font-size:26px;font-weight:800;color:{_col_v}'>${_npv_v:,.0f}</div>"
                    f"<div style='font-size:10px;color:#718096;margin-top:2px'>5-yr NPV</div>"
                    f"<div style='font-size:11px;color:#8896a8;margin-top:8px;line-height:1.5'>{_desc}</div>"
                    f"<div style='font-size:10px;color:#4a5568;margin-top:4px'>{_cost_note}</div>"
                    f"</div>"
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
            f"<em style='color:#4a5568'>Not financial advice.</em>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── 📋 Supporting details (U2 density reduction): maintenance calendar, ──
    # application fit, second-life marketplace, confidence-reason callouts ──
    with st.expander("📋 Supporting details", expanded=False):
        # ── 3. Maintenance Calendar ─────────────────────────────────────────
        st.markdown("<div class='section-header'>Maintenance Calendar</div>", unsafe_allow_html=True)
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
            st.markdown("<div class='section-header'>Application Fit Scores</div>", unsafe_allow_html=True)
            _af_cols = st.columns(min(len(fit_scores), 4))
            _af_colour = {"fit": "#48bb78", "marginal": "#f6ad55", "not_fit": "#fc8181"}
            for _i, (_app_key, _app) in enumerate(fit_scores.items()):
                with _af_cols[_i % len(_af_cols)]:
                    st.markdown(
                        f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                        f"padding:14px 16px'>"
                        f"<div style='font-size:11px;color:#4a5568'>{_app['short']}</div>"
                        f"<div style='font-size:16px;font-weight:700;color:{_af_colour[_app['fit']]};margin-top:4px'>"
                        f"{_app['fit'].replace('_', ' ').title()}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── 4b. Second-life marketplace ──────────────────────────────────────
        if action in ("second_life", "recycle") or soh <= 85:
            st.markdown("<div class='section-header'>Second-Life Marketplace</div>", unsafe_allow_html=True)
            _sl_eligible = soh >= 70.0
            _sl_col = "#68d391" if _sl_eligible else "#718096"
            _md_html(
                f"<div style='background:#1e2a38;border:1px solid {_sl_col}44;border-radius:10px;"
                f"padding:18px 22px;margin-bottom:12px'>"
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
                f"</div>"
                f"</div>"
            )
            if _sl_eligible:
                _mk_col1, _mk_col2 = st.columns(2)
                if _mk_col1.button(
                    "List on Circunomics →", key="sl_list_circ",
                    help="Opens second-life battery exchange (demo — not a live API call)",
                    use_container_width=True,
                ):
                    _circ_chemistry = "LFP" if selected.startswith("S-") else "LiCoO2"
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
                    if _circ_result is not None and "error" not in _circ_result:
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
                        if _circ_result is not None:
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
    _log_col, _cmms_col, _ = st.columns([1, 1, 2])
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
        if _cmms_result is None:
            st.warning("No CMMS API key configured — set one up in Settings first.")
        elif "error" in _cmms_result:
            st.error(f"CMMS ticket creation failed: {_cmms_result['error']}")
        else:
            st.success(f"CMMS ticket created for {selected}.")

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
                    f"<div style='font-size:11px;color:#4a5568'>{_dl['timestamp']} · SOH {_dl['soh_pct']}%</div>"
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
    with st.expander("Full economics & breakeven analysis", expanded=False):
        page_consequences(selected, df, featured_dfs, bundles, rul_reliable)

    # ── Inline Copilot panel (merged Decision + Copilot) ─────────────────────
    st.markdown("<div class='section-header'>Ask about this cell</div>", unsafe_allow_html=True)
    if selected in NASA_CELL_IDS:
        _dc_bundle = bundles.get("nasa")
    elif selected.startswith("S-"):
        _dc_bundle = bundles.get("severson")
    elif selected in CELL_STRESS_PROFILES:
        _dc_bundle = bundles.get("synth")
    else:
        _dc_bundle = bundles.get("upload")
    if _dc_bundle:
        try:
            from battery_copilot import build_cell_context, answer_query
            from copilot_llm import llm_answer
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
                _md_html(
                    f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                    f"padding:16px 20px;margin-top:8px'>"
                    f"<div style='font-size:10px;color:#4a5568;margin-bottom:8px'>{_badge_dc} · {selected}</div>"
                    f"<div style='font-size:13px;color:#e2e8f0;line-height:1.7'>{_dc_answer}</div>"
                    f"</div>"
                )
        except Exception as _dc_e:
            st.caption(f"Copilot unavailable: {_dc_e}")


