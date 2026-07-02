"""Page: Overview"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from utils import (
    base_layout, LEGEND_H, PLOTLY_CONFIG, _md_html, soh_status, friendly,
    _soh_sparkline_svg, _cell_provenance, _analysis_provenance, NASA_CELL_IDS,
)
from data_loader import CELL_STRESS_PROFILES, _stress_factor


def page_overview(df: pd.DataFrame, split_cycle: int, cell_id: str,
                  rul_reliable: bool = True, bundle: dict | None = None):
    st.markdown("# Overview")

    latest         = df.iloc[-1]
    current_soh    = latest["soh_pct"]
    current_rul    = latest["rul_pred"]
    current_cycle  = int(latest["cycle_number"])
    total_fade     = latest["capacity_fade_ah"]
    confidence     = latest["confidence_tag"]
    status_label, status_colour = soh_status(current_soh)

    rul_q10 = float(latest["rul_q10"]) if "rul_q10" in latest.index else None
    rul_q90 = float(latest["rul_q90"]) if "rul_q90" in latest.index else None

    sop_pct = float(latest["sop_pct"]) if "sop_pct" in latest.index else None

    fold_r2 = None
    if bundle is not None:
        cell_fold = bundle["metrics"].get("lco_per_cell", {}).get(cell_id, {})
        fold_r2   = cell_fold.get("rul_r2", None)

    rul_calibrating = (not rul_reliable) or (confidence == "Calibrating")
    rul_display     = "—" if rul_calibrating else f"{current_rul:.0f}"
    rul_sub         = "not calibrated" if not rul_reliable else "cycles to 80% SOH"

    app_eol = float(st.session_state.get("eol_threshold_pct", 80.0))
    adj_rul = current_rul
    if not rul_calibrating and current_rul is not None and app_eol != 80.0:
        fade_50 = float(latest.get("fade_rate_50cy", 0)) * 100
        if fade_50 > 1e-6:
            adj_rul = max(0, (current_soh - app_eol) / fade_50)
        rul_display = f"{adj_rul:.0f}"
        rul_sub     = f"cycles to {app_eol:.0f}% SOH (app threshold)"

    conf_html = (
        "<span class='tag-calibrating'>CALIBRATING</span>"
        if rul_calibrating
        else "<span class='tag-model'>MODEL</span>"
    )
    rul_hero = "Not calibrated" if not rul_reliable else f"Est. {current_rul:.0f} cycles remaining"

    if rul_calibrating and not rul_reliable:
        if fold_r2 is not None:
            conf_reason = (
                "RUL not calibrated — model accuracy insufficient for this cell's "
                f"degradation profile (held-out fold R²={fold_r2:.2f}, below 0.30 reliability floor)"
            )
        else:
            conf_reason = (
                "RUL not calibrated — model accuracy insufficient for this cell's degradation profile"
            )
    elif not rul_calibrating and fold_r2 is not None:
        conf_reason = (
            f"RUL predictions were tested against data this cell never trained on "
            f"(leave-cell-out validation) — fold R²={fold_r2:.2f} (reliable above 0.30 floor)"
        )
    else:
        conf_reason = None

    if "test_date" in df.columns and df["test_date"].notna().any():
        dates = pd.to_datetime(df["test_date"].dropna())
        span_days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
        cycles_per_day = len(df) / span_days
        rate_note = f"estimated at {cycles_per_day:.1f} cycle/day from data"
    else:
        cycles_per_day = 1.0
        rate_note = "estimated at 1 cycle/day assumption"
    months_remaining = None
    if not rul_calibrating and current_rul is not None and cycles_per_day > 0:
        months_remaining = current_rul / cycles_per_day / 30.44

    is_nasa = cell_id in NASA_CELL_IDS
    if is_nasa:
        source_tag = "NASA real · 24°C · 2A discharge"
    else:
        p  = CELL_STRESS_PROFILES.get(cell_id, {})
        sf = _stress_factor(p.get("temp_mean",25), p.get("c_rate",1.0), p.get("dod",1.0))
        source_tag = f"Synthetic · Stress {sf:.2f}x baseline"

    sparkline_svg = _soh_sparkline_svg(df["soh_pct"])
    interval_html = ""
    if not rul_calibrating and rul_q10 is not None and rul_q90 is not None and rul_q90 > rul_q10:
        interval_html = (
            f"<div style='font-size:11px;color:#718096;margin-top:4px'>"
            f"80% interval: <strong style='color:#a0aec0'>{rul_q10:.0f}–{rul_q90:.0f} cycles</strong>"
            f"</div>"
        )

    _md_html(
        f"""
        <div class="hero-card">
            <div class="hero-label">Battery Status · {cell_id}</div>
            <div class="hero-value {status_colour}">{status_label}</div>
            <div class="hero-sub">
                SOH: <strong style="color:#e2e8f0">{current_soh:.1f}%</strong>
                &nbsp;{sparkline_svg}&nbsp;
                &nbsp;·&nbsp; {rul_hero}
                &nbsp;·&nbsp; {source_tag}
                &nbsp;·&nbsp; {conf_html}
            </div>
            {interval_html}
        </div>
        """
    )

    if conf_reason:
        conf_c = "#fc8181" if (rul_calibrating and not rul_reliable) else "#4a5568"
        st.markdown(
            f"<div style='font-size:12px;color:{conf_c};margin:-8px 0 12px;"
            f"padding:6px 12px;background:#1a202c;border-radius:6px;"
            f"border-left:3px solid {conf_c}'>{conf_reason}</div>",
            unsafe_allow_html=True,
        )

    is_nasa_src = cell_id in NASA_CELL_IDS
    src_val = "NASA" if is_nasa_src else f"{sf:.2f}x"
    src_sub = "real measured" if is_nasa_src else "vs baseline (synthetic)"
    sop_display = f"{sop_pct:.0f}%" if sop_pct is not None else "—"
    months_html = (
        f"<div class='metric-chip-sub' style='margin-top:2px;color:#4a5568'>"
        f"~{months_remaining:.0f} months · {rate_note}</div>"
        if months_remaining is not None else ""
    )
    _cum_kwh = float(latest["cumulative_kwh"]) if "cumulative_kwh" in latest.index else None
    kwh_display = f"{_cum_kwh:.2f} kWh" if _cum_kwh is not None else "—"
    _eq_cy = float(latest["equivalent_cycles"]) if "equivalent_cycles" in latest.index and not pd.isna(latest["equivalent_cycles"]) else None
    eq_cy_display = f"{_eq_cy:,.0f}" if _eq_cy is not None else "—"
    eq_cy_sub = "stress-normalized (CATL/BYD metric)" if _eq_cy is not None else "not available"
    _ce = float(latest["coulombic_efficiency"]) if "coulombic_efficiency" in latest.index else None
    ce_display = f"{_ce*100:.3f}%" if _ce is not None else "—"

    # ── Plain-English action sentence ──
    _fade_50 = float(latest.get("fade_rate_50cy", 0)) * 100
    _fade_accel = float(latest.get("fade_rate_30cy", 0)) > float(latest.get("fade_rate_10cy", 0)) * 1.3
    if not rul_calibrating and months_remaining is not None:
        if months_remaining < 2:
            _plain_sentence = f"This cell is near end of life — replacement within weeks is likely needed."
        elif months_remaining < 6:
            _plain_sentence = f"This cell will need replacement in approximately {months_remaining:.0f} months if current usage continues."
        else:
            _trend = " Fade is accelerating — monitor closely." if _fade_accel else ""
            _plain_sentence = f"This cell has an estimated {months_remaining:.0f} months of useful life remaining.{_trend}"
    elif rul_calibrating:
        _plain_sentence = f"SOH is {current_soh:.1f}% — RUL cannot be estimated for this cell (model not calibrated). Monitor fade rate trend."
    else:
        _plain_sentence = f"SOH is {current_soh:.1f}% — {status_label.lower()} condition."

    _sentence_color = "#fc8181" if current_soh < 80 else ("#f6ad55" if current_soh < 85 else "#48bb78")
    _md_html(
        f"<div style='font-size:14px;color:{_sentence_color};margin:-4px 0 20px;"
        f"padding:10px 16px;background:rgba(0,0,0,0.2);border-radius:8px;"
        f"border-left:3px solid {_sentence_color}'>{_plain_sentence}</div>"
    )

    # ── 3 primary metric chips (SOH, RUL, Fade Rate) ──
    _fade_rate_30 = float(latest.get("fade_rate_30cy", 0)) * 1000
    _res_now = float(latest.get("resistance_ohm", 0)) * 1000
    _md_html(
        f"""<div class="metric-row">
        <div class="metric-chip">
            <div class="metric-chip-label">State of Health</div>
            <div class="metric-chip-value" style="color:{_sentence_color}">{current_soh:.1f}%</div>
            <div class="metric-chip-sub">vs 100% at cycle 1 · {current_cycle:,} cycles completed</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Remaining Useful Life</div>
            <div class="metric-chip-value">{rul_display}</div>
            <div class="metric-chip-sub">{rul_sub}</div>{months_html}
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Fade Rate (30-cycle avg)</div>
            <div class="metric-chip-value" style="font-size:28px">{'↑' if _fade_accel else ''}{_fade_rate_30:.2f}</div>
            <div class="metric-chip-sub">mAh/cycle · {'accelerating' if _fade_accel else 'stable'}</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Internal Resistance</div>
            <div class="metric-chip-value" style="font-size:28px">{_res_now:.1f}</div>
            <div class="metric-chip-sub">mΩ current</div>
        </div>
        </div>"""
    )

    # ── Secondary metrics in collapsible expander ──
    with st.expander("Cell details — throughput, efficiency, stress", expanded=False):
        _md_html(
            f"""<div class="metric-row">
            <div class="metric-chip"><div class="metric-chip-label">Capacity Lost</div><div class="metric-chip-value" style="font-size:22px">{total_fade*1000:.0f} mAh</div><div class="metric-chip-sub">since commissioning</div></div>
            <div class="metric-chip"><div class="metric-chip-label">Peak Power (SoP)</div><div class="metric-chip-value" style="font-size:22px">{sop_display}</div><div class="metric-chip-sub">of initial capability</div></div>
            <div class="metric-chip"><div class="metric-chip-label">Energy Delivered</div><div class="metric-chip-value" style="font-size:22px">{kwh_display}</div><div class="metric-chip-sub">cumulative throughput</div></div>
            <div class="metric-chip"><div class="metric-chip-label">Equiv. Cycles</div><div class="metric-chip-value" style="font-size:22px">{eq_cy_display}</div><div class="metric-chip-sub">{eq_cy_sub}</div></div>
            <div class="metric-chip"><div class="metric-chip-label">Coulombic Eff.</div><div class="metric-chip-value" style="font-size:22px">{ce_display}</div><div class="metric-chip-sub">last cycle Q_d / Q_c</div></div>
            </div>"""
        )

    # ── Contextual action bar ──
    _action_links = [
        ("Deep health analysis", "health"),
        ("Degradation drivers (SHAP)", "insights"),
        ("Recommendations", "recommendations"),
        ("Fleet comparison", "fleet"),
    ]
    _link_html = " &nbsp;·&nbsp; ".join(
        f"<span style='cursor:pointer;color:#63b3ed;text-decoration:underline dotted' "
        f"onclick=\"window.location.href='?page={key}'\">{label}</span>"
        for label, key in _action_links
    )
    _md_html(
        f"<div style='font-size:12px;color:#4a5568;margin-bottom:20px'>"
        f"From this view you can → {_link_html}</div>"
    )
    # Streamlit button version (onclick JS doesn't work in Streamlit — use buttons instead)
    _act_cols = st.columns(len(_action_links))
    for (_label, _key), _col in zip(_action_links, _act_cols):
        with _col:
            if st.button(_label, key=f"ov_link_{_key}", use_container_width=True, type="secondary"):
                st.session_state["page"] = _key
                st.rerun()

    if "cumulative_days" in df.columns:
        with st.expander("📅 Calendar Age Analysis", expanded=False):
            try:
                cal_last = float(df["cumulative_days"].iloc[-1])
                avg_rest = float(df["days_between_cycles"].mean()) if "days_between_cycles" in df.columns else 1.0
                total_fade_ah = float(df["capacity_ah"].iloc[0]) - float(df["capacity_ah"].iloc[-1])
                calendar_fade_pct = min(35, max(5, 100 * (cal_last * 0.00003 * 0.5) / max(total_fade_ah, 1e-6)))

                cal_c1, cal_c2, cal_c3 = st.columns(3)
                with cal_c1:
                    st.metric("Total Calendar Time", f"{cal_last:.0f} days", help="since first cycle")
                    st.markdown("<div style='font-size:11px;color:#718096;margin-top:-8px'>since first cycle</div>", unsafe_allow_html=True)
                with cal_c2:
                    st.metric("Avg Rest Between Cycles", f"{avg_rest:.1f} days", help="mean rest period")
                    st.markdown("<div style='font-size:11px;color:#718096;margin-top:-8px'>mean rest period</div>", unsafe_allow_html=True)
                with cal_c3:
                    st.metric("Calendar vs Cycle Fade", f"~{calendar_fade_pct:.0f}% calendar", help="remainder from cycling")
                    st.markdown("<div style='font-size:11px;color:#718096;margin-top:-8px'>remainder from cycling</div>", unsafe_allow_html=True)

                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=df["cycle_number"], y=df["cumulative_days"],
                    name="Actual calendar time", line=dict(color="#63b3ed", width=2),
                    hovertemplate="Cycle %{x}: %{y:.1f} days<extra>Calendar</extra>",
                ))
                max_cy = int(df["cycle_number"].max())
                fig_cal.add_trace(go.Scatter(
                    x=[0, max_cy], y=[0, max_cy],
                    name="1 cycle/day baseline", line=dict(color="#4a5568", width=1.5, dash="dash"),
                    hoverinfo="skip",
                ))
                fig_cal.update_layout(
                    **base_layout(
                        height=260, legend=LEGEND_H,
                        xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                        yaxis=dict(title="Days elapsed", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                    ),
                )
                fig_cal.update_layout(title=dict(text="Calendar Time vs Cycle Count", font=dict(size=12, color="#a0aec0"), x=0))
                st.plotly_chart(fig_cal, use_container_width=True)
            except Exception as _e:
                st.info(f"Calendar age analysis unavailable: {_e}")

    st.markdown("<div class='section-header'>State of Health — Full History</div>", unsafe_allow_html=True)

    df_train = df[df["cycle_number"] <= split_cycle]
    df_test  = df[df["cycle_number"] >  split_cycle]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["soh_pct"],
        name="Actual SOH", line=dict(color="#3a4a5e", width=1), mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f}%<extra>Actual</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["soh_rolling_avg"],
        name="10-cycle avg", line=dict(color="#63b3ed", width=2), mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f}%<extra>10-cy avg</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_test["cycle_number"], y=df_test["soh_pred"],
        name="Model (test)", line=dict(color="#48bb78", width=2, dash="dot"), mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f}%<extra>Model</extra>",
    ))
    fig.add_vline(
        x=split_cycle, line_dash="dot", line_color="#4a5568", line_width=1,
        annotation_text=f"Train → Test (cy {split_cycle})",
        annotation_position="top left",
        annotation_font_color="#4a5568", annotation_font_size=11,
    )
    last_cycle = df.cycle_number.iloc[-1]
    last_soh   = df.soh_pct.iloc[-1]
    eol_line   = float(st.session_state.get("eol_threshold_pct", 80.0))
    nominal_rate    = float(df.fade_rate_50cy.iloc[-1]) * 100
    optimistic_rate = nominal_rate * 0.6
    pessimistic_rate = nominal_rate * 1.5

    def _proj(rate):
        if rate > 1e-9:
            n_steps = min(500, int((last_soh - eol_line) / rate) + 10)
        else:
            n_steps = 200
        n_steps = max(n_steps, 2)
        proj_cycles = np.arange(last_cycle, last_cycle + n_steps)
        proj_soh    = last_soh - rate * np.arange(n_steps)
        proj_soh    = np.clip(proj_soh, 60.0, None)
        return proj_cycles.tolist(), proj_soh.tolist()

    for rate, name, color in [
        (nominal_rate,     "Nominal projection",          "#63b3ed"),
        (optimistic_rate,  "Optimistic (−40% stress)",    "#48bb78"),
        (pessimistic_rate, "Pessimistic (+50% stress)",   "#fc8181"),
    ]:
        px, py = _proj(rate)
        fig.add_trace(go.Scatter(
            x=px, y=py, name=name, mode="lines",
            line=dict(dash="dash", width=1.5, color=color),
            opacity=0.7,
            hovertemplate="Cycle %{x}: %{y:.1f}%<extra>" + name + "</extra>",
        ))

    fig.add_hline(
        y=eol_line, line_dash="dot", line_color="#e53e3e", line_width=1,
        annotation_text=f"EOL threshold ({eol_line:.0f}%)",
        annotation_position="bottom right",
    )
    y_min = max(df["soh_pct"].min() - 2, 60)
    fig.update_layout(
        **base_layout(
            height=340, legend=LEGEND_H,
            xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                       zeroline=False, range=[y_min, 101]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    _ov_prov = _cell_provenance(cell_id)
    st.caption(
        "Projections assume constant fade rate. Optimistic = 40% stress reduction. Pessimistic = 50% stress increase. "
        + ("Capacity data: ● MEASURED (NASA PCoE). Projections: ◐ SIMULATED (linear extrapolation)."
           if _ov_prov == "measured"
           else "All data: ○ SYNTHETIC — no physical measurements underlie any value.")
    )
