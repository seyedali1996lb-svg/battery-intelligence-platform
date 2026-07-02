"""Page: Sustainability"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import _md_html, NASA_CELL_IDS, LEGEND_H, base_layout
from design_system import (
    BADGE_ESTIMATE, BADGE_ILLUST, make_badge, make_state_badge, section_header_html,
)


def page_sustainability(selected: str, df: pd.DataFrame):
    from consequences import ASSUMPTIONS, sustainability_snapshot, CELL_NOMINAL_KWH
    from sustainability import (
        CRITICAL_MATERIALS, EU_RECYCLED_TARGETS, EU_GREEN_DEAL_FIELDS,
        material_content_for_cell,
    )

    is_nasa  = selected in NASA_CELL_IDS
    source   = "nasa" if is_nasa else "synth"
    latest   = df.iloc[-1]
    soh      = float(latest["soh_pct"])
    cycles   = int(latest["cycle_number"])
    cell_kwh = CELL_NOMINAL_KWH[source]

    def _section(title: str):
        st.markdown(section_header_html(title), unsafe_allow_html=True)

    st.markdown("# Sustainability")
    st.markdown(f"##### Lifecycle carbon + circularity · {selected}")

    st.markdown(
        f"<div style='background:rgba(183,121,31,0.07);border:1px solid rgba(183,121,31,0.25);"
        f"border-radius:10px;padding:14px 20px;margin-bottom:28px;"
        f"font-size:13px;color:#718096;line-height:1.7'>"
        f"<strong style='color:#d69e2e'>Figure transparency.</strong> "
        f"All CO₂ and material figures are estimates from literature sources — "
        f"not measurements from this specific cell. Each figure is labeled "
        f"{BADGE_ESTIMATE} or {BADGE_ILLUST} at the point of display. "
        f"No aggregated sustainability score is shown: individual labeled figures "
        f"are more honest than any index that mixes sources with different confidence levels."
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Adjust assumptions", expanded=False):
        sl_col1, sl_col2 = st.columns(2)
        with sl_col1:
            co2_val = st.slider(
                f"CO₂ to make one new cell ({ASSUMPTIONS['co2_manufacture']['unit']})",
                min_value=float(ASSUMPTIONS["co2_manufacture"]["slider_range"][0]),
                max_value=float(ASSUMPTIONS["co2_manufacture"]["slider_range"][1]),
                value=float(ASSUMPTIONS["co2_manufacture"]["value"]), step=0.05,
                key="sus7_co2_mfg",
                help=ASSUMPTIONS["co2_manufacture"]["source"],
            )
        with sl_col2:
            grid_val = st.slider(
                f"Grid carbon intensity ({ASSUMPTIONS['grid_carbon_intensity']['unit']})",
                min_value=float(ASSUMPTIONS["grid_carbon_intensity"]["slider_range"][0]),
                max_value=float(ASSUMPTIONS["grid_carbon_intensity"]["slider_range"][1]),
                value=float(ASSUMPTIONS["grid_carbon_intensity"]["value"]), step=0.01,
                key="sus7_grid",
                help=ASSUMPTIONS["grid_carbon_intensity"]["source"],
            )
        sl_col3, sl_col4 = st.columns(2)
        with sl_col3:
            mat_val = st.slider(
                f"Material recovery value ({ASSUMPTIONS['material_recovery']['unit']})",
                min_value=float(ASSUMPTIONS["material_recovery"]["slider_range"][0]),
                max_value=float(ASSUMPTIONS["material_recovery"]["slider_range"][1]),
                value=float(ASSUMPTIONS["material_recovery"]["value"]), step=0.05,
                key="sus7_mat",
                help=ASSUMPTIONS["material_recovery"]["source"],
            )
        with sl_col4:
            extension_val = st.slider(
                f"Second-life extension ({ASSUMPTIONS['second_life_extension']['unit']})",
                min_value=float(ASSUMPTIONS["second_life_extension"]["slider_range"][0]),
                max_value=float(ASSUMPTIONS["second_life_extension"]["slider_range"][1]),
                value=float(ASSUMPTIONS["second_life_extension"]["value"]), step=0.1,
                key="sus7_extension",
                help=ASSUMPTIONS["second_life_extension"]["source"],
            )

    sus = sustainability_snapshot(source=source, co2_per_cell=co2_val, material_recovery=mat_val)
    use_phase_co2 = cell_kwh * cycles * grid_val

    _section("CO₂ Impact — Reuse vs Recycle vs New Cell")

    h1, h2, h3, h4 = st.columns(4)
    hero_tiles = [
        (h1, "Manufacturing CO₂\n(one new cell)",    f"{co2_val:.2f} kg", "#f6ad55", BADGE_ESTIMATE),
        (h2, "Use phase CO₂\n(to date, this cell)",  f"{use_phase_co2:.2f} kg", "#718096", BADGE_ILLUST),
        (h3, "Reuse saves\n(vs making a new cell)",  f"{sus['co2_avoided_by_reuse']:.2f} kg", "#68d391", BADGE_ESTIMATE),
        (h4, "Recycle credit\n(15% cathode, Dunn 2015)", f"{sus['co2_recycling_credit']:.2f} kg", "#f6e05e", BADGE_ESTIMATE),
    ]
    for col, label, val, colour, badge in hero_tiles:
        label_lines = label.split("\n")
        label_html = "<br>".join(label_lines)
        with col:
            st.markdown(
                f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                f"padding:16px 20px;height:100%'>"
                f"<div style='font-size:11px;color:#4a5568;line-height:1.5'>{label_html}</div>"
                f"<div style='font-size:26px;font-weight:700;color:{colour};margin-top:6px'>{val}</div>"
                f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>CO₂e</div>"
                f"<div style='margin-top:10px'>{badge}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _section("Lifecycle Carbon Chart")

    st.markdown(
        f"<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        f"Manufacturing and EOL figures are {BADGE_ESTIMATE} from IVL 2019 and Dunn et al. 2015. "
        f"Use-phase CO₂ is {BADGE_ILLUST} — it depends entirely on your grid's carbon intensity "
        f"(set via slider above). Drag the grid slider to see how dominant the use phase becomes "
        f"on a coal grid vs a renewable grid."
        f"</div>",
        unsafe_allow_html=True,
    )

    lc_norm = st.radio(
        "Chart normalisation",
        ["Per cell", "Per kWh delivered"],
        index=0, horizontal=True, key="sus7_lc_norm",
        help="Per kWh delivered divides all bars by total energy throughput for each scenario.",
    )

    reuse_cycles     = cycles * extension_val
    new_cell_cycles  = cycles
    reuse_use_co2    = cell_kwh * reuse_cycles * grid_val
    new_cell_use_co2 = cell_kwh * new_cell_cycles * grid_val

    scenarios       = ["Recycle now", f"Reuse (×{extension_val:.1f} cycles)", "New cell (counterfactual)"]
    mfg_bars        = [co2_val, co2_val, co2_val]
    use_bars        = [use_phase_co2, reuse_use_co2, new_cell_use_co2]
    eol_credit_bars = [
        -sus["co2_recycling_credit"],
        -(sus["co2_recycling_credit"] + sus["co2_avoided_by_reuse"]),
        -sus["co2_recycling_credit"],
    ]

    if lc_norm == "Per kWh delivered":
        kwh_denominators = [cell_kwh * cycles, cell_kwh * reuse_cycles, cell_kwh * new_cell_cycles]
        mfg_bars        = [v / d for v, d in zip(mfg_bars, kwh_denominators)]
        use_bars        = [v / d for v, d in zip(use_bars, kwh_denominators)]
        eol_credit_bars = [v / d for v, d in zip(eol_credit_bars, kwh_denominators)]
        yaxis_label, bar_suffix = "kg CO₂e per kWh delivered", " kg/kWh"
    else:
        yaxis_label, bar_suffix = "kg CO₂e per cell", " kg"

    fig_lc = go.Figure()
    fig_lc.add_trace(go.Bar(name="Manufacturing CO₂", x=scenarios, y=mfg_bars, marker_color="#f6ad55",
        text=[f"{v:.3f}{bar_suffix}" for v in mfg_bars], textposition="inside", textfont=dict(size=10, color="#1a202c")))
    fig_lc.add_trace(go.Bar(name="Use phase CO₂ (illustrative)", x=scenarios, y=use_bars, marker_color="#718096",
        text=[f"{v:.3f}{bar_suffix}" for v in use_bars], textposition="inside", textfont=dict(size=10, color="#e2e8f0")))
    fig_lc.add_trace(go.Bar(name="EOL credit (negative = saving)", x=scenarios, y=eol_credit_bars, marker_color="#68d391",
        text=[f"{v:.3f}{bar_suffix}" for v in eol_credit_bars], textposition="inside", textfont=dict(size=10, color="#1a202c")))
    fig_lc.update_layout(
        **base_layout(
            barmode="relative",
            xaxis=dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            yaxis=dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=True, zerolinecolor="#4a5568",
                       title=dict(text=yaxis_label, font=dict(size=11))),
            height=360,
        )
    )
    fig_lc.update_layout(legend=LEGEND_H, title=dict(
        text="Lifecycle CO₂ — three end-of-life scenarios", font=dict(size=13, color="#a0aec0"), x=0))
    st.plotly_chart(fig_lc, use_container_width=True)
    st.markdown(
        f"<div style='font-size:11px;color:#4a5568;margin-top:-8px;margin-bottom:4px'>"
        f"'Reuse' use-phase uses {extension_val:.1f}× current cycles ({cycles} → {reuse_cycles:.0f} cycles). "
        f"Second-life extension slider is {BADGE_ILLUST}. "
        f"'New cell' use-phase uses the same {cycles}-cycle baseline at current grid intensity. "
        f"All use-phase figures are {BADGE_ILLUST}."
        f"</div>",
        unsafe_allow_html=True,
    )

    _section("Degradation Rate & Carbon — What the Trend Means")

    fade_30 = float(latest.get("fade_rate_30cy", float("nan")))
    if not (fade_30 != fade_30):
        if fade_30 * 1000 > 5.0:
            fade_signal = "accelerating"
            fade_implication = (
                "A fast fade rate shortens the useful life phase. "
                "Because manufacturing CO₂ is fixed at cell production and amortised across the "
                "full operating lifetime, a shorter useful life leaves more of that carbon "
                "unrecovered per unit of energy delivered. "
                "A cell degrading quickly reaches the recycling decision point sooner — "
                "meaning less time in which the reuse CO₂ saving accumulates."
            )
            fade_colour = "#fc8181"
        elif fade_30 * 1000 > 2.0:
            fade_signal = "moderate"
            fade_implication = (
                "A moderate fade rate means the manufacturing carbon is being amortised "
                "at a reasonable pace across the useful life. "
                "Extending service life through second-life deployment would increase that "
                "amortisation further, reducing the manufacturing carbon burden per kWh delivered."
            )
            fade_colour = "#f6e05e"
        else:
            fade_signal = "slow"
            fade_implication = (
                "A slow, stable fade rate maximises the amortisation of manufacturing CO₂ "
                "across a long useful life. "
                "This cell is recovering its embodied carbon effectively: a longer service "
                "life means the fixed manufacturing cost is spread across more kWh delivered."
            )
            fade_colour = "#68d391"

        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
            f"padding:18px 22px;margin-bottom:4px'>"
            f"<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            f"letter-spacing:0.08em;margin-bottom:8px'>Fade rate impact on carbon amortisation</div>"
            f"<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:12px'>"
            f"<div style='font-size:22px;font-weight:700;color:{fade_colour}'>"
            f"{fade_30*1000:.2f} mAh/cy</div>"
            f"<div style='font-size:13px;color:{fade_colour}99'>{fade_signal} degradation</div>"
            f"</div>"
            f"<div style='font-size:13px;color:#a0aec0;line-height:1.8'>{fade_implication}</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:12px'>"
            f"This is qualitative direction only — the carbon figures above are based on "
            f"literature estimates, not measurements from this cell. No percentage saving is "
            f"stated here because the absolute manufacturing CO₂ figure (above) already carries "
            f"wide uncertainty. "
            f"Tier thresholds (slow &lt;2 mAh/cy · moderate 2–5 mAh/cy · accelerating &gt;5 mAh/cy) "
            f"are illustrative — adjust for your cell chemistry."
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _section("Critical Materials Tracker")

    if not is_nasa:
        st.markdown(
            "<div style='background:#2d3748;border-radius:8px;padding:10px 16px;"
            "font-size:12px;color:#718096;margin-bottom:12px'>"
            "Synthetic cells model electrochemical behaviour only — material content "
            "figures below apply to the equivalent real LiCoO₂ 18650 chemistry, not the simulation."
            "</div>",
            unsafe_allow_html=True,
        )

    primary_materials = [m for m in CRITICAL_MATERIALS if m["name"] != "Nickel (Ni)"]
    mat_cols = st.columns(len(primary_materials))
    for col, mat in zip(mat_cols, primary_materials):
        scaled_g = material_content_for_cell(mat["g_per_2ah"], cell_kwh)
        badge_html = make_badge(mat["label"], "#b7791f" if "Cited" in mat["label"] else "#718096")
        rec_html = (
            f"<div style='font-size:12px;color:#68d391;margin-top:4px'>"
            f"~{mat['recovery_pct']}% recovery<br>"
            f"<span style='font-size:11px;color:#4a5568'>{mat['recovery_note']}</span></div>"
            if mat["recovery_pct"] is not None else
            "<div style='font-size:12px;color:#4a5568;margin-top:4px'>Not recovered<br>"
            "<span style='font-size:11px'>Not primary material in LiCoO₂</span></div>"
        )
        eu_dot = (
            "<span style='color:#63b3ed;font-size:10px;margin-left:4px'>EU critical ●</span>"
            if mat["eu_critical"] else ""
        )
        with col:
            st.markdown(
                f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                f"padding:14px 16px'>"
                f"<div style='font-size:12px;color:#a0aec0;font-weight:600'>{mat['name']}{eu_dot}</div>"
                f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>{mat['formula']}</div>"
                f"<div style='font-size:22px;font-weight:700;color:#e2e8f0;margin-top:8px'>{scaled_g:.1f} g</div>"
                f"<div style='font-size:11px;color:#718096'>est. per cell ({mat['g_range']} @ 2 Ah)</div>"
                f"{rec_html}"
                f"<div style='margin-top:10px'>{badge_html}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    ni_mat = next((m for m in CRITICAL_MATERIALS if m["name"] == "Nickel (Ni)"), None)
    if ni_mat:
        ni_g = material_content_for_cell(ni_mat["g_per_2ah"], cell_kwh)
        st.markdown(
            f"<div style='font-size:11px;color:#4a5568;margin-top:10px;padding:8px 14px;"
            f"background:#1a202c;border-radius:6px;border-left:3px solid #2d3748'>"
            f"<strong style='color:#718096'>Nickel (Ni)</strong> — EU critical material, but trace-only in LiCoO₂ chemistry "
            f"(est. {ni_g:.2f} g per cell, {BADGE_ILLUST}). "
            f"EU 2023/1542 nickel recycled-content targets apply to NMC/NCA chemistries where nickel is a primary cathode material, "
            f"not to LiCoO₂. Shown here for completeness only."
            f"</div>",
            unsafe_allow_html=True,
        )

    _section("EU Battery Regulation — Recycled Content Targets (2023/1542, Annex XII)")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px;line-height:1.6'>"
        "Targets apply to industrial batteries and EV batteries by mass of active materials. "
        "Estimated recycled content in <em>current</em> 18650 LiCoO₂ cells is not publicly "
        "certified — the figures below reflect industry-wide estimates, not this specific cell. "
        "This platform cannot make a compliance claim without manufacturer supply chain data."
        "</div>",
        unsafe_allow_html=True,
    )

    for target in EU_RECYCLED_TARGETS:
        est_recycled = target.get("current_industry_range", "—")
        current_note = target.get("current_note", "")
        bar_fill_31  = min(target["target_2031_pct"], 100)
        bar_fill_36  = min(target["target_2036_pct"], 100)
        st.markdown(
            f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
            f"padding:14px 20px;margin-bottom:10px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:8px'>"
            f"<div style='font-size:13px;font-weight:600;color:#e2e8f0'>{target['material']}</div>"
            f"<div style='display:flex;gap:16px;font-size:12px;color:#a0aec0'>"
            f"<span>2031 target: <strong style='color:#63b3ed'>{target['target_2031_pct']}%</strong></span>"
            f"<span>2036 target: <strong style='color:#63b3ed'>{target['target_2036_pct']}%</strong></span>"
            f"<span>Est. current: <strong style='color:#f6ad55'>{est_recycled}</strong> "
            f"{make_badge('Illustrative', '#718096')}</span>"
            f"</div></div>"
            f"<div style='font-size:11px;color:#4a5568;margin-bottom:8px'>{current_note}</div>"
            f"<div style='background:#2d3748;border-radius:4px;height:8px;margin-bottom:4px'>"
            f"<div style='background:#63b3ed33;border-radius:4px;height:8px;width:{bar_fill_31}%;"
            f"position:relative'>"
            f"<div style='background:#63b3ed;border-radius:4px;height:8px;width:{bar_fill_36/bar_fill_31*100:.0f}%'>"
            f"</div></div></div>"
            f"<div style='font-size:10px;color:#4a5568'>Source: {target['source']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _section("EU Green Deal Alignment — Data Coverage")

    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:14px'>"
        "Same three-state system as the Battery Passport page — "
        "Available (pipeline output), Estimated (cited/illustrative), "
        "Not available in demo (genuine gap)."
        "</div>",
        unsafe_allow_html=True,
    )

    for field in EU_GREEN_DEAL_FIELDS:
        state = field["state"]
        badge = make_state_badge(state)
        muted = state == "unavailable"
        val_colour = "#4a5568" if muted else "#a0aec0"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
            f"gap:16px;padding:11px 0;border-bottom:1px solid #2d3748'>"
            f"<div style='flex:1'>"
            f"<div style='font-size:13px;color:{val_colour};font-style:{'italic' if muted else 'normal'}'>"
            f"{field['label']}</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:3px'>{field['note']}</div>"
            f"</div>"
            f"<div style='flex-shrink:0;padding-top:2px'>{badge}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    with st.expander("Assumption sources — CO₂ and material figures", expanded=False):
        sus_keys = ["co2_manufacture", "grid_carbon_intensity", "material_recovery"]
        for key in sus_keys:
            a = ASSUMPTIONS[key]
            badge_colour = "#b7791f" if "Cited" in a["label"] else "#718096"
            st.markdown(
                f"<div style='padding:12px 0;border-bottom:1px solid #2d3748'>"
                f"<div style='font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:6px'>"
                f"{a['unit']} — default {a['value']} &nbsp; {make_badge(a['label'], badge_colour)}"
                f"</div>"
                f"<div style='font-size:12px;color:#718096;line-height:1.6'>{a['source']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
