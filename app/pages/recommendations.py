"""Page: Recommendations"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from utils import _md_html, NASA_CELL_IDS
from design_system import (
    BADGE_VALIDATED, BADGE_ESTIMATE, make_badge,
    ACTION_META, CONF_META, provenance_banner,
)


def page_recommendations(
    selected: str,
    df: pd.DataFrame,
    featured_dfs: dict,
    bundles: dict,
    rul_reliable: bool,
):
    from recommendations import classify, SOH_PRIMARY_FLOOR, SOH_INSPECT_FLOOR, SOH_SECONDLIFE_FLOOR
    from consequences import ASSUMPTIONS, application_fit, financial_comparison, CELL_NOMINAL_KWH

    latest   = df.iloc[-1]
    soh      = float(latest["soh_pct"])
    fade_30  = float(latest.get("fade_rate_30cy", 0.0))
    fade_50  = float(latest.get("fade_rate_50cy", 0.0))
    is_nasa  = selected in NASA_CELL_IDS
    source   = "nasa" if is_nasa else "synth"

    rul_pred_raw = latest.get("rul_pred", None)
    rul_pred     = float(rul_pred_raw) if (rul_reliable and rul_pred_raw is not None) else None

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

    # Alias for thermal runaway section
    cell_id = selected

    st.markdown("# Recommendations")
    st.markdown(f"##### Decision summary · {selected}")

    # ── Maintenance Calendar ──
    st.markdown("<div class='section-header'>📅 Maintenance Calendar</div>", unsafe_allow_html=True)
    _eol_threshold = float(st.session_state.get("eol_threshold_pct", 80.0))
    _current_soh = float(df["soh_pct"].iloc[-1])
    _fade_per_cycle_pct = float(df["fade_rate_50cy"].iloc[-1]) * 100 / (float(df["capacity_ah"].iloc[0]) + 1e-9)
    if _fade_per_cycle_pct > 1e-6:
        _rul_cycles = max(0, (_current_soh - _eol_threshold) / _fade_per_cycle_pct)
    else:
        _rul_cycles = None

    if "test_date" in df.columns and df["test_date"].notna().any():
        _dates = pd.to_datetime(df["test_date"].dropna())
        _span_days = max((_dates.iloc[-1] - _dates.iloc[0]).days, 1)
        _cycles_per_day = len(df) / _span_days
    else:
        _cycles_per_day = 1.0

    if _rul_cycles is not None and _rul_cycles > 0:
        from datetime import date as _date, timedelta as _timedelta
        _days_to_eol = _rul_cycles / _cycles_per_day
        _replacement_date = _date.today() + _timedelta(days=_days_to_eol)
        _cal1, _cal2, _cal3 = st.columns(3)
        _cal1.metric("Recommended Replacement", _replacement_date.strftime("%B %Y"))
        _cal2.metric("Cycles Remaining", f"{_rul_cycles:.0f}")
        _cal3.metric("Days Remaining", f"{_days_to_eol:.0f}")

        st.markdown("**⚠️ Cost of Delay**")
        _monthly_extra_fade_pct = _fade_per_cycle_pct * _cycles_per_day * 30.44
        _delay_data = []
        for _months_delay in [0, 1, 3, 6, 12]:
            _extra_fade = _monthly_extra_fade_pct * _months_delay
            _soh_at_delay = max(60, _current_soh - _fade_per_cycle_pct * (_rul_cycles + _cycles_per_day * 30.44 * _months_delay))
            _value_penalty_pct = max(0, (_eol_threshold - _soh_at_delay) * 2)
            _delay_data.append({
                "Delay": f"+{_months_delay}mo" if _months_delay > 0 else "On time",
                "SOH at replacement": f"{_soh_at_delay:.1f}%",
                "Extra degradation": f"{_extra_fade:.2f}%",
                "Est. residual value loss": f"{_value_penalty_pct:.0f}%",
                "Risk": "Low" if _months_delay == 0 else ("Medium" if _months_delay <= 3 else "High"),
            })
        st.dataframe(pd.DataFrame(_delay_data), use_container_width=True, hide_index=True)
        st.caption("Residual value estimate: each % SOH below EOL threshold at replacement ≈ 2% additional value penalty (illustrative — no market data).")
    else:
        st.info("Insufficient fade data to compute replacement timeline for this cell.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Hero card ──
    reason_html = "".join(
        f"<div style='margin-top:6px;font-size:13px;color:{action_colour}cc'>"
        f"· {r}</div>"
        for r in result["action_reasons"]
    )
    st.markdown(
        f"<div style='background:{action_bg};border:2px solid {action_colour}55;"
        f"border-radius:14px;padding:28px 32px;margin-bottom:24px'>"
        f"<div style='font-size:11px;font-weight:700;color:{action_colour}99;"
        f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px'>"
        f"Primary Recommendation</div>"
        f"<div style='font-size:32px;font-weight:800;color:{action_colour}'>"
        f"{action_label}</div>"
        f"<div style='margin-top:10px'>"
        f"<span style='background:{conf_colour}22;border:1px solid {conf_colour}55;"
        f"color:{conf_colour};font-size:11px;font-weight:700;padding:2px 10px;"
        f"border-radius:10px;letter-spacing:0.06em'>{conf_label}</span>"
        f"</div>"
        f"{reason_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if result["confidence_reasons"]:
        for note in result["confidence_reasons"]:
            is_fit = "fit scores" in note
            note_colour = "#b7791f" if is_fit else "#718096"
            badge_html  = BADGE_ESTIMATE if is_fit else make_badge("Reduced certainty", "#718096")
            st.markdown(
                f"<div style='background:{note_colour}11;border:1px solid {note_colour}33;"
                f"border-radius:8px;padding:10px 16px;margin-bottom:8px;"
                f"font-size:12px;color:#a0aec0;line-height:1.6'>"
                f"{badge_html}&nbsp; {note}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── SoC Window ──
    st.markdown("<div class='section-header'>🔋 Optimal Charging Window (SoC)</div>", unsafe_allow_html=True)
    _fade_pct_cy = fade_30 * 100
    if soh >= 95 and _fade_pct_cy < 0.02:
        _soc_lo, _soc_hi = 0, 100
        _soc_label, _soc_colour = "Full range OK", "#68d391"
        _soc_reason = "Cell is healthy with low fade rate. 0–100% is acceptable for this usage profile."
    elif soh >= 90:
        _soc_lo, _soc_hi = 10, 90
        _soc_label, _soc_colour = "Light restriction", "#a3e635"
        _soc_reason = "Avoid deep discharge and sustained 100% SOC. Restricting top/bottom 10% extends cycle life by ~20% (NREL 2022)."
    elif soh >= 80:
        _soc_lo, _soc_hi = 20, 80
        _soc_label, _soc_colour = "Standard protection", "#f6ad55"
        _soc_reason = "20–80% is Tesla's default adaptive charging target for degraded cells. Minimises lithium plating risk at charge top and copper dissolution at deep discharge."
    elif soh >= 70:
        _soc_lo, _soc_hi = 25, 75
        _soc_label, _soc_colour = "Conservative window", "#f6ad55"
        _soc_reason = "Cell approaching EOL. Narrowing window to 25–75% maximises residual value and reduces thermal risk per IEC 62619."
    else:
        _soc_lo, _soc_hi = 30, 70
        _soc_label, _soc_colour = "Deep restriction", "#fc8181"
        _soc_reason = "Cell significantly degraded. 30–70% minimises lithium plating and reduces thermal runaway risk."

    _soc_c1, _soc_c2, _soc_c3 = st.columns([1, 2, 1])
    with _soc_c1:
        st.metric("Lower Limit", f"{_soc_lo}% SOC")
    with _soc_c2:
        _bar_left  = _soc_lo
        _bar_width = _soc_hi - _soc_lo
        _md_html(
            f"<div style='margin:8px 0'>"
            f"<div style='font-size:11px;color:#4a5568;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.08em'>Recommended Window</div>"
            f"<div style='position:relative;background:#1a202c;border-radius:6px;height:28px;overflow:hidden'>"
            f"<div style='position:absolute;left:{_bar_left}%;width:{_bar_width}%;height:100%;"
            f"background:{_soc_colour}33;border:2px solid {_soc_colour};border-radius:4px'></div>"
            f"<div style='position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);"
            f"font-size:12px;font-weight:700;color:{_soc_colour};text-align:center'>"
            f"{_soc_lo}% → {_soc_hi}%</div></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:10px;color:#4a5568;margin-top:3px'>"
            f"<span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>"
            f"</div>"
        )
    with _soc_c3:
        st.metric("Upper Limit", f"{_soc_hi}% SOC")
    _md_html(f"<div style='font-size:12px;color:#718096;margin:8px 0 4px;padding:10px 14px;background:#1e2a38;border-radius:8px;border-left:3px solid {_soc_colour}'><strong style='color:{_soc_colour}'>{_soc_label}</strong> — {_soc_reason}</div>")

    # ── Thermal Runaway ──
    st.markdown("<div class='section-header'>⚠️ Thermal Runaway Risk Proxy</div>", unsafe_allow_html=True)
    _md_html(provenance_banner(
        "simulated" if cell_id in NASA_CELL_IDS else "synthetic",
        "<strong>NOT a certified safety assessment.</strong> "
        "This score is a heuristic weighted sum of three observable signals "
        "(resistance growth 45%, fade acceleration 40%, temperature 15%). "
        "The weights have no experimental validation and no peer-reviewed basis. "
        "A certified thermal runaway risk assessment requires accelerating rate calorimetry (ARC), "
        "nail penetration tests, and EIS characterisation under abuse conditions — "
        "none of which this platform performs. "
        "Do not use this score for safety-critical decisions."
    ))
    _r_norm = float(latest.get("resistance_normalized", 1.0))
    _fade_acc = float(latest.get("fade_acceleration", 0.0))
    _temp_now = float(latest.get("temperature_c", 25.0))
    _r_score   = min(100, max(0, (_r_norm - 1.0) / 0.6 * 100))
    _acc_score = min(100, max(0, -_fade_acc / 0.0008 * 100))
    _t_score   = min(100, max(0, (_temp_now - 25.0) / 20.0 * 100))
    _tr_score  = round(min(100, max(0, 0.45 * _r_score + 0.40 * _acc_score + 0.15 * _t_score)), 1)
    if _tr_score >= 65:
        _tr_label, _tr_colour, _tr_bg = "High", "#fc8181", "rgba(252,129,129,0.08)"
        _tr_action = "Immediate inspection recommended. Prioritise this cell for replacement."
    elif _tr_score >= 35:
        _tr_label, _tr_colour, _tr_bg = "Moderate", "#f6ad55", "rgba(246,173,85,0.08)"
        _tr_action = "Monitor closely. Reduce operating temperature and C-rate if possible."
    else:
        _tr_label, _tr_colour, _tr_bg = "Low", "#68d391", "rgba(104,211,145,0.08)"
        _tr_action = "No elevated thermal risk. Continue standard monitoring."
    _tr1, _tr2, _tr3, _tr4 = st.columns(4)
    _tr1.metric("TR Risk Score", f"{_tr_score:.0f} / 100")
    _tr2.metric("Resistance Factor", f"{_r_norm:.2f}×", help=f"{_r_score:.0f}/100 — contributes 45%")
    _tr3.metric("Fade Acceleration", f"{_fade_acc*1000:.3f} mAh/cy²", help=f"{_acc_score:.0f}/100 — contributes 40%")
    _tr4.metric("Temp. Factor", f"{_temp_now:.1f}°C", help=f"{_t_score:.0f}/100 — contributes 15%")
    _md_html(
        f"<div style='background:{_tr_bg};border:1px solid {_tr_colour}44;border-radius:10px;"
        f"padding:14px 20px;margin:8px 0'>"
        f"<span style='font-size:14px;font-weight:700;color:{_tr_colour}'>{_tr_label} Risk</span>"
        f"<span style='font-size:12px;color:#a0aec0;margin-left:14px'>{_tr_action}</span>"
        f"<div style='margin-top:8px;background:#0e1117;border-radius:4px;height:8px'>"
        f"<div style='width:{_tr_score}%;background:{_tr_colour};height:8px;border-radius:4px'></div>"
        f"</div></div>"
    )

    # ── Supporting Evidence ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin:24px 0 16px'>Supporting Evidence</div>",
        unsafe_allow_html=True,
    )

    ev1, ev2, ev3 = st.columns(3)

    with ev1:
        soh_status_label, soh_colour = (
            ("Healthy", "#68d391")        if soh >= 90 else
            ("Degrading", "#f6e05e")      if soh >= 80 else
            ("End of Life", "#fc8181")
        )
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
            f"padding:16px 20px'>"
            f"<div style='font-size:11px;color:#4a5568'>State of Health</div>"
            f"<div style='font-size:28px;font-weight:700;color:{soh_colour};margin-top:4px'>{soh:.1f}%</div>"
            f"<div style='font-size:12px;color:{soh_colour}99;margin-top:2px'>{soh_status_label}</div>"
            f"<div style='margin-top:10px'>{BADGE_VALIDATED}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with ev2:
        if rul_pred is not None:
            rul_val    = f"{rul_pred:.0f} cy"
            rul_colour = "#e2e8f0"
            rul_note   = "Leave-cell-out validated"
        else:
            rul_val    = "Not calibrated"
            rul_colour = "#718096"
            rul_note   = f"Fold R² below {0.30} floor"
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
            f"padding:16px 20px'>"
            f"<div style='font-size:11px;color:#4a5568'>Est. Remaining Useful Life</div>"
            f"<div style='font-size:28px;font-weight:700;color:{rul_colour};margin-top:4px'>{rul_val}</div>"
            f"<div style='font-size:12px;color:#4a5568;margin-top:2px'>{rul_note}</div>"
            f"<div style='margin-top:10px'>{BADGE_VALIDATED}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with ev3:
        fade_label  = "Accelerating ⚠" if result["fade_accelerating"] else "Stable"
        fade_colour = "#fc8181" if result["fade_accelerating"] else "#68d391"
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
            f"padding:16px 20px'>"
            f"<div style='font-size:11px;color:#4a5568'>Fade Rate (30-cycle)</div>"
            f"<div style='font-size:28px;font-weight:700;color:#e2e8f0;margin-top:4px'>"
            f"{fade_30*1000:.2f} <span style='font-size:14px;color:#718096'>mAh/cy</span></div>"
            f"<div style='font-size:12px;color:{fade_colour};margin-top:2px'>"
            f"{fade_label} ({result['fade_ratio']:.1f}× baseline)</div>"
            f"<div style='margin-top:10px'>{BADGE_VALIDATED}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Action Timeline ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin:24px 0 16px'>Action Timeline</div>",
        unsafe_allow_html=True,
    )

    if action == "continue":
        next_soh = SOH_PRIMARY_FLOOR - 2.0
        if rul_pred is not None and fade_30 > 0:
            cycles_to_inspect = (soh - next_soh) / (fade_30 * 1000 / 100) if fade_30 > 0 else None
        else:
            cycles_to_inspect = None
        cycle_note = (
            f"Estimated ~{cycles_to_inspect:.0f} cycles at current fade rate."
            if cycles_to_inspect is not None
            else "Cycle count not estimated — RUL not calibrated for this cell."
        )
        cycle_badge = BADGE_VALIDATED if rul_pred is not None else make_badge("Not calibrated", "#718096")
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:20px 24px'>"
            f"<div style='font-size:14px;font-weight:600;color:#e2e8f0'>Next action: monitor fade rate</div>"
            f"<div style='font-size:13px;color:#a0aec0;margin-top:8px;line-height:1.7'>"
            f"Schedule inspection when SOH reaches {next_soh:.0f}%. {cycle_note}"
            f"</div>"
            f"<div style='margin-top:10px'>{cycle_badge}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    elif action == "inspect":
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:20px 24px'>"
            f"<div style='font-size:14px;font-weight:600;color:#e2e8f0'>Recommended inspection checks</div>"
            f"<div style='font-size:13px;color:#a0aec0;margin-top:8px;line-height:1.9'>"
            f"· Confirm SOH trajectory — is fade rate stable, marginal, or accelerating?<br>"
            f"· Check resistance trend — rising resistance at this SOH often precedes rapid fade<br>"
            f"· Evaluate second-life fit (SOH target {SOH_SECONDLIFE_FLOOR:.0f}–{SOH_PRIMARY_FLOOR:.0f}%) — prepare transition plan<br>"
            f"· If fade is accelerating ({result['fade_ratio']:.1f}× baseline currently), advance the inspection timeline"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    elif action == "second_life":
        a   = {k: v["value"] for k, v in ASSUMPTIONS.items()}
        fc  = financial_comparison(
            soh=soh, source=source,
            recycling_value=a["recycling_value"],
            new_cell_cost=a["new_cell_cost"],
            sl_value_per_kwh=a["second_life_value_per_kwh"],
            repack_cost=a["repack_cost"],
        )
        best = result["best_app"]
        if rul_reliable and fade_30 > 0:
            cycles_to_eol = (soh - SOH_SECONDLIFE_FLOOR) / (fade_30 * 1000 / 100)
            sl_timeline_html = (
                f"<div style='font-size:13px;color:#a0aec0;margin-top:10px'>"
                f"Estimated ~<strong style='color:#e2e8f0'>{cycles_to_eol:.0f} cycles</strong> "
                f"remaining before reaching the {SOH_SECONDLIFE_FLOOR:.0f}% second-life floor "
                f"at the current fade rate."
                f"</div>"
            )
        else:
            sl_timeline_html = (
                f"<div style='font-size:12px;color:#718096;margin-top:10px'>"
                f"Timeline unavailable — RUL not calibrated for this cell."
                f"</div>"
            )
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:20px 24px'>"
            f"<div style='font-size:14px;font-weight:600;color:#e2e8f0'>"
            f"Best application: {best['name']}</div>"
            f"<div style='font-size:13px;color:#a0aec0;margin-top:8px;line-height:1.7'>"
            f"Fit: <strong style='color:#e2e8f0'>{best['fit']}</strong>"
            f" &nbsp;·&nbsp; SOH range: {best['soh_min']:.0f}–{best['soh_max']:.0f}%"
            f"</div>"
            f"<div style='font-size:13px;color:#a0aec0;margin-top:6px;line-height:1.7'>"
            f"{' '.join(best['reasons'])}"
            f"</div>"
            f"{sl_timeline_html}"
            f"<div style='display:flex;gap:32px;margin-top:16px'>"
            f"<div><div style='font-size:11px;color:#4a5568'>Second-life net value</div>"
            f"<div style='font-size:22px;font-weight:700;color:#63b3ed'>${fc['sl_net']:.2f}</div>"
            f"<div style='margin-top:6px'>{BADGE_ESTIMATE}</div></div>"
            f"<div><div style='font-size:11px;color:#4a5568'>vs. Recycle now</div>"
            f"<div style='font-size:22px;font-weight:700;color:#f6ad55'>${fc['recycle_value']:.2f}</div>"
            f"<div style='margin-top:6px'>{BADGE_ESTIMATE}</div></div>"
            f"</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:12px'>"
            f"Financial figures use Phase 4 assumption defaults — adjust on the Consequences page.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    elif action == "recycle":
        a        = {k: v["value"] for k, v in ASSUMPTIONS.items()}
        rec_val  = a["recycling_value"]
        mat_val  = a["material_recovery"]
        reason = (
            "No viable second-life application identified based on estimated fit thresholds (cited-estimate uncertainty applies)."
            if result["fit_driven"]
            else f"SOH {soh:.1f}% is below the {SOH_SECONDLIFE_FLOOR:.0f}% second-life floor."
        )
        if rul_pred is not None:
            rec_timeline_html = (
                f"<div style='font-size:13px;color:#a0aec0;margin-top:10px'>"
                f"Est. <strong style='color:#e2e8f0'>{rul_pred:.0f} cycles</strong> remaining "
                f"before reaching 80% EOL threshold — proceed with recycling pathway now."
                f"</div>"
            )
        else:
            rec_timeline_html = (
                f"<div style='font-size:12px;color:#718096;margin-top:10px'>"
                f"Timeline unavailable — RUL not calibrated for this cell."
                f"</div>"
            )
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:20px 24px'>"
            f"<div style='font-size:14px;font-weight:600;color:#e2e8f0'>Estimated recovery value</div>"
            f"<div style='font-size:13px;color:#a0aec0;margin-top:8px;line-height:1.7'>{reason}</div>"
            f"{rec_timeline_html}"
            f"<div style='display:flex;gap:32px;margin-top:16px'>"
            f"<div><div style='font-size:11px;color:#4a5568'>Cell recycling value</div>"
            f"<div style='font-size:22px;font-weight:700;color:#f6ad55'>${rec_val:.2f}</div>"
            f"<div style='margin-top:6px'>{BADGE_ESTIMATE}</div></div>"
            f"<div><div style='font-size:11px;color:#4a5568'>Material recovery</div>"
            f"<div style='font-size:22px;font-weight:700;color:#f6ad55'>${mat_val:.2f}</div>"
            f"<div style='margin-top:6px'>{BADGE_ESTIMATE}</div></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Uncertainty acknowledgment ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin:24px 0 16px'>What this recommendation does not account for</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;"
        "padding:16px 24px;font-size:13px;color:#718096;line-height:1.9'>"
        "This recommendation is based on observed cycle data and leave-cell-out validated "
        "model outputs. It does not account for: operating conditions going forward (temperature, "
        "C-rate, and depth-of-discharge significantly affect future degradation); real-world "
        "variation in second-life market pricing; manufacturer safety specifications for "
        "continued use or repacking; or any events since the last recorded cycle. "
        "Treat it as a data-driven starting point for a human decision, not a directive."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── All application fit scores ──
    with st.expander("All application fit scores", expanded=False):
        fit_cols = st.columns(len(fit_scores))
        for col, (app_key, app) in zip(fit_cols, fit_scores.items()):
            colour = {"fit": "#68d391", "marginal": "#f6e05e", "not_fit": "#fc8181"}[app["fit"]]
            with col:
                st.markdown(
                    f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                    f"padding:14px 16px'>"
                    f"<div style='font-size:11px;color:#4a5568'>{app['short']}</div>"
                    f"<div style='font-size:16px;font-weight:700;color:{colour};margin-top:4px'>"
                    f"{app['fit'].replace('_', ' ').title()}</div>"
                    f"<div style='font-size:11px;color:#4a5568;margin-top:8px;line-height:1.6'>"
                    f"{'<br>'.join(app['reasons'])}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
