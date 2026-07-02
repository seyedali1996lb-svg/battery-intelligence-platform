import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from utils import _md_html, base_layout, NASA_CELL_IDS
from design_system import (
    BADGE_VALIDATED, BADGE_ESTIMATE, BADGE_ILLUST,
    make_badge,
)


def page_consequences(
    selected: str,
    df: pd.DataFrame,
    featured_dfs: dict,
    bundles: dict,
    rul_reliable: bool,
):
    from consequences import (
        ASSUMPTIONS, SECOND_LIFE_APPS, CELL_NOMINAL_KWH,
        application_fit, financial_comparison, sustainability_snapshot, breakeven_curve,
    )

    # ── Pull validated model outputs ──
    latest           = df.iloc[-1]
    soh              = float(latest["soh_pct"])
    fade_30          = float(latest.get("fade_rate_30cy", 0.0))
    rul_pred_raw     = latest.get("rul_pred", None)
    rul_pred         = float(rul_pred_raw) if (rul_reliable and rul_pred_raw is not None) else None
    is_nasa          = selected in NASA_CELL_IDS
    source           = "nasa" if is_nasa else "synth"

    peer_fades = [
        float(fdf.iloc[-1].get("fade_30_mah_cy", 0))
        for cid, fdf in featured_dfs.items()
        if (cid in NASA_CELL_IDS) == is_nasa and cid != selected
    ]
    fleet_fade_median = float(pd.Series(peer_fades).median()) if peer_fades else None

    # ── Page header ──
    st.markdown("# EOL Economics")
    st.markdown("##### End-of-life value recovery · second-life fit scores · circular economy")

    _md_html(
        f"""
        <div style="background:rgba(183,121,31,0.07);border:1px solid rgba(183,121,31,0.25);
                    border-radius:10px;padding:14px 20px;margin-bottom:28px;
                    font-size:13px;color:#718096;line-height:1.7">
            <strong style="color:#d69e2e">Assumption transparency.</strong>
            SOH, fade rate, and the RUL reliability flag are {BADGE_VALIDATED} outputs
            from the leave-cell-out validated pipeline.<br>
            All financial and environmental figures carry either an {BADGE_ESTIMATE} badge
            (cited source below) or an {BADGE_ILLUST} badge (engineering judgment only).
            Slider values are yours to adjust — the defaults are mid-points of the cited ranges.
        </div>
        """
    )

    # ── Primary life gate ──
    if soh > 85.0:
        _md_html(
            f"""
            <div style="background:#1e2a38;border:1px dashed #2d3748;border-radius:12px;
                        padding:48px;text-align:center">
                <div style="font-size:18px;font-weight:600;color:#4a5568;margin-bottom:12px">
                    Still in Primary Life
                </div>
                <div style="font-size:14px;color:#4a5568;max-width:480px;margin:0 auto;line-height:1.7">
                    SOH is {soh:.1f}% — above the 85% threshold where second-life assessment
                    becomes relevant. Return here as the cell degrades toward 85% SOH.
                </div>
            </div>
            """
        )
        return

    # ── Validated inputs row (makes the banner concrete) ──
    rul_display = (
        f"{rul_pred:.0f} cy" if rul_pred is not None
        else "not calibrated"
    )
    rul_colour  = "#718096" if rul_pred is None else "#e2e8f0"
    _md_html(
        f"""
        <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:24px">
            <div style="background:#1e2a38;border:1px solid #2d3748;border-radius:8px;
                        padding:10px 18px;min-width:140px">
                <div style="font-size:10px;color:#4a5568;margin-bottom:4px">State of Health</div>
                <div style="font-size:20px;font-weight:700;color:#e2e8f0">{soh:.1f}%</div>
                <div style="margin-top:6px">{BADGE_VALIDATED}</div>
            </div>
            <div style="background:#1e2a38;border:1px solid #2d3748;border-radius:8px;
                        padding:10px 18px;min-width:160px">
                <div style="font-size:10px;color:#4a5568;margin-bottom:4px">Fade rate (30-cy)</div>
                <div style="font-size:20px;font-weight:700;color:#e2e8f0">
                    {fade_30*1000:.2f} <span style="font-size:13px;color:#718096">mAh/cy</span>
                </div>
                <div style="margin-top:6px">{BADGE_VALIDATED}</div>
            </div>
            <div style="background:#1e2a38;border:1px solid #2d3748;border-radius:8px;
                        padding:10px 18px;min-width:140px">
                <div style="font-size:10px;color:#4a5568;margin-bottom:4px">Est. RUL</div>
                <div style="font-size:20px;font-weight:700;color:{rul_colour}">{rul_display}</div>
                <div style="margin-top:6px">{BADGE_VALIDATED}</div>
            </div>
        </div>
        """
    )

    # ────────────────────────────────────────────────────────────────────────
    # Section 1: Second-Life Application Fit
    # ────────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:20px'>Second-Life Application Fit</div>",
        unsafe_allow_html=True,
    )

    fit_results = application_fit(soh, fade_30, fleet_fade_median)

    FIT_STYLE = {
        "fit":      ("#68d391", "#1a2e22", "Fit"),
        "marginal": ("#f6e05e", "#2d2a0a", "Marginal"),
        "not_fit":  ("#fc8181", "#2d0f0f", "Not Fit"),
    }

    fit_cols = st.columns(3)
    for col, (app_key, res) in zip(fit_cols, fit_results.items()):
        fg, bg, label = FIT_STYLE[res["fit"]]
        reasons_html = "".join(
            f"<div style='margin-top:6px;font-size:12px;color:{fg}99;line-height:1.5'>{r}</div>"
            for r in res["reasons"]
        )
        source_html = (
            f"<div style='margin-top:10px;font-size:10px;color:#4a5568;font-style:italic;"
            f"line-height:1.4'>{res['source']}</div>"
        )
        with col:
            _md_html(
                f"""
                <div style="background:{bg};border:1px solid {fg}33;border-radius:10px;
                            padding:20px;height:100%">
                    <div style="font-size:10px;font-weight:700;color:{fg};
                                text-transform:uppercase;letter-spacing:0.08em;
                                margin-bottom:6px">{label}</div>
                    <div style="font-size:15px;font-weight:700;color:{fg};
                                margin-bottom:4px">{res['name']}</div>
                    <div style="font-size:12px;color:{fg}99;margin-bottom:8px">
                        {res['description']}
                    </div>
                    <div style="border-top:1px solid {fg}22;padding-top:8px">
                        {reasons_html}
                    </div>
                    {source_html}
                </div>
                """
            )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # Section 2: Financial Comparison
    # ────────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:20px'>Financial Comparison</div>",
        unsafe_allow_html=True,
    )

    fin_left, fin_right = st.columns([1, 2])

    with fin_left:
        st.markdown(
            "<div style='font-size:12px;color:#4a5568;margin-bottom:12px'>"
            "Adjust assumptions — defaults are mid-points of the cited ranges.</div>",
            unsafe_allow_html=True,
        )
        n_cells = st.number_input(
            "Pack size (number of cells)",
            min_value=1, max_value=10_000, value=1, step=1,
            key="fin_n_cells",
            help="Scale totals to a full pack. Cards show pack total; per-cell shown below each figure.",
        )
        a = ASSUMPTIONS
        recycling_val = st.slider(
            f"Recycling value / cell ({a['recycling_value']['unit']})",
            min_value=float(a["recycling_value"]["slider_range"][0]),
            max_value=float(a["recycling_value"]["slider_range"][1]),
            value=float(a["recycling_value"]["value"]), step=0.25,
            key="fin_recycling",
            help=a["recycling_value"]["source"],
        )
        new_cell_cost = st.slider(
            f"New cell cost ({a['new_cell_cost']['unit']})",
            min_value=float(a["new_cell_cost"]["slider_range"][0]),
            max_value=float(a["new_cell_cost"]["slider_range"][1]),
            value=float(a["new_cell_cost"]["value"]), step=1.0,
            key="fin_new_cell",
            help=a["new_cell_cost"]["source"],
        )
        sl_val_per_kwh = st.slider(
            f"Second-life value ({a['second_life_value_per_kwh']['unit']})",
            min_value=float(a["second_life_value_per_kwh"]["slider_range"][0]),
            max_value=float(a["second_life_value_per_kwh"]["slider_range"][1]),
            value=float(a["second_life_value_per_kwh"]["value"]), step=5.0,
            key="fin_sl_kwh",
            help=a["second_life_value_per_kwh"]["source"],
        )
        repack_cost = st.slider(
            f"Repack cost / cell ({a['repack_cost']['unit']})",
            min_value=float(a["repack_cost"]["slider_range"][0]),
            max_value=float(a["repack_cost"]["slider_range"][1]),
            value=float(a["repack_cost"]["value"]), step=1.0,
            key="fin_repack",
            help=a["repack_cost"]["source"],
        )

    fin = financial_comparison(
        soh=soh, source=source,
        recycling_value=recycling_val,
        new_cell_cost=new_cell_cost,
        sl_value_per_kwh=sl_val_per_kwh,
        repack_cost=repack_cost,
    )

    with fin_right:
        # Three option cards: Reuse, Recycle, Replace new
        sl_net   = fin["sl_net"]
        rec_val  = fin["recycle_value"]
        new_cost = fin["new_cell_cost"]

        best     = max(sl_net, rec_val)
        options  = [
            ("Reuse (second-life)", sl_net,  "#63b3ed", "BADGE_ESTIMATE", a["second_life_value_per_kwh"]["label"]),
            ("Recycle now",         rec_val, "#f6ad55", "BADGE_ESTIMATE", a["recycling_value"]["label"]),
            ("Buy new cell",        -new_cost, "#fc8181", "BADGE_ESTIMATE", a["new_cell_cost"]["label"]),
        ]

        cell_kwh    = fin["cell_kwh"]
        current_kwh = fin["current_kwh"]
        src_label   = "NASA PCoE datasheet, ~2 Ah" if is_nasa else "Oxford dataset spec, 0.74 Ah"
        kwh_note    = (
            f"Cell: {cell_kwh*1000:.1f} Wh nominal ({src_label}) · "
            f"Current: {current_kwh*1000:.1f} Wh at {soh:.1f}% SOH"
        )

        st.markdown(
            f"<div style='font-size:11px;color:#4a5568;margin-bottom:16px'>{kwh_note}</div>",
            unsafe_allow_html=True,
        )

        opt_cols = st.columns(3)
        for col, (name, value, colour, _, badge_label) in zip(opt_cols, options):
            badge_html   = make_badge(badge_label, "#b7791f" if "Cited" in badge_label else "#718096")
            repack_note  = (
                f"<div style='font-size:11px;color:#718096;margin-top:6px'>"
                f"after −${repack_cost:.0f}/cell repack &nbsp;"
                f"{make_badge(a['repack_cost']['label'], '#718096')}</div>"
                if name == "Reuse (second-life)" else
                "<div style='height:0'></div>"
            )
            is_best    = (name != "Buy new cell") and (value == best) and (value > 0)
            border     = f"2px solid {colour}" if is_best else f"1px solid {colour}33"
            bg         = f"{colour}15" if is_best else "#1e2a38"
            best_tag   = (
                f"<div style='font-size:10px;color:{colour};font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px'>Best option</div>"
                if is_best else
                "<div style='height:18px'></div>"
            )
            pack_value = value * n_cells
            pack_sign  = "+" if pack_value > 0 else ""
            cell_note  = (
                f"<div style='font-size:11px;color:{colour}77;margin-top:3px'>"
                f"{'+' if value > 0 else ''}${abs(value):.2f} / cell</div>"
                if n_cells > 1 else
                "<div style='height:0'></div>"
            )
            with col:
                _md_html(
                    f"""
                    <div style="background:{bg};border:{border};border-radius:10px;
                                padding:20px;text-align:center">
                        {best_tag}
                        <div style="font-size:12px;color:#718096;margin-bottom:8px">{name}</div>
                        <div style="font-size:26px;font-weight:700;color:{colour}">
                            {pack_sign}${abs(pack_value):.2f}
                        </div>
                        {cell_note}
                        <div style="margin-top:8px">{badge_html}</div>
                        {repack_note}
                    </div>
                    """
                )

        if not rul_reliable:
            st.markdown(
                "<div style='font-size:12px;color:#718096;margin-top:14px;font-style:italic'>"
                "ℹ RUL is not calibrated for this cell (fold R² below reliability floor). "
                "The break-even chart projects value by SOH only — not by time or cycle count. "
                "A cycle-based timeline would require a reliable RUL estimate.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # Break-even chart
    # ────────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:20px'>Value Crossover — When Does Recycling Win?</div>",
        unsafe_allow_html=True,
    )

    bev = breakeven_curve(
        source=source,
        sl_value_per_kwh=sl_val_per_kwh,
        repack_cost=repack_cost,
        recycling_value=recycling_val,
        soh_current=soh,
    )
    bev_sohs     = bev["sohs"]
    bev_sl       = [v * n_cells for v in bev["sl_nets"]]
    bev_recycle  = bev["recycle_val"] * n_cells
    bev_cross    = bev["crossover_soh"]
    pack_label   = f" (pack of {n_cells})" if n_cells > 1 else " (per cell)"

    bev_fig = go.Figure()

    # Shaded region where reuse > recycle
    bev_fig.add_trace(go.Scatter(
        x=bev_sohs + bev_sohs[::-1],
        y=[max(v, bev_recycle) for v in bev_sl] + [bev_recycle] * len(bev_sohs),
        fill="toself", fillcolor="rgba(99,179,237,0.08)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

    # Reuse net value line
    bev_fig.add_trace(go.Scatter(
        x=bev_sohs, y=bev_sl,
        mode="lines", name=f"Reuse net value{pack_label}",
        line=dict(color="#63b3ed", width=2.5),
        hovertemplate="SOH %{x:.1f}% → $%{y:.2f}<extra>Reuse</extra>",
    ))

    # Recycle flat line
    bev_fig.add_trace(go.Scatter(
        x=[bev_sohs[0], bev_sohs[-1]],
        y=[bev_recycle, bev_recycle],
        mode="lines", name=f"Recycle value{pack_label}",
        line=dict(color="#f6ad55", width=2, dash="dash"),
        hovertemplate=f"Recycle: ${bev_recycle:.2f}<extra></extra>",
    ))

    # Current SOH marker
    bev_fig.add_vline(
        x=soh, line_dash="dot", line_color="#718096", line_width=1.5,
        annotation_text=f"Now ({soh:.1f}%)",
        annotation_position="top left",
        annotation_font_color="#718096", annotation_font_size=11,
    )

    # Crossover annotation
    if bev_cross is not None and bev_cross < soh:
        bev_fig.add_vline(
            x=bev_cross, line_dash="dash", line_color="#fc8181", line_width=1.5,
            annotation_text=f"Recycle wins ({bev_cross:.1f}%)",
            annotation_position="top right",
            annotation_font_color="#fc8181", annotation_font_size=11,
        )
    elif bev_cross is None:
        bev_fig.add_annotation(
            x=bev_sohs[-1], y=bev_sl[-1],
            text="Reuse stays ahead to 62% SOH",
            showarrow=False, font=dict(color="#68d391", size=11),
            xanchor="left", yanchor="bottom",
        )

    bev_fig.update_layout(**base_layout(
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            title="State of Health (%)",
            autorange="reversed",
            gridcolor="#232d3b", linecolor="#2d3748", zeroline=False,
        ),
        yaxis=dict(
            title=f"$ value{pack_label}",
            gridcolor="#232d3b", linecolor="#2d3748", zeroline=False,
            rangemode="tozero",
        ),
    ))
    st.markdown(
        "<div style='font-size:12px;color:#4a5568;margin-bottom:12px'>"
        "Reuse net value = (remaining capacity × $/kWh) − repack cost, projected as SOH declines. "
        "Recycle value is fixed. "
        "All figures are estimates — adjust sliders above to explore scenarios.</div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(bev_fig, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # Section 3: Sustainability
    # ────────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:20px'>Sustainability Snapshot</div>",
        unsafe_allow_html=True,
    )

    sus_left, sus_right = st.columns([1, 2])

    with sus_left:
        co2_val = st.slider(
            f"CO₂ to make one new cell ({ASSUMPTIONS['co2_manufacture']['unit']})",
            min_value=float(ASSUMPTIONS["co2_manufacture"]["slider_range"][0]),
            max_value=float(ASSUMPTIONS["co2_manufacture"]["slider_range"][1]),
            value=float(ASSUMPTIONS["co2_manufacture"]["value"]), step=0.05,
            key="sus_co2",
            help=ASSUMPTIONS["co2_manufacture"]["source"],
        )
        mat_val = st.slider(
            f"Material recovery value ({ASSUMPTIONS['material_recovery']['unit']})",
            min_value=float(ASSUMPTIONS["material_recovery"]["slider_range"][0]),
            max_value=float(ASSUMPTIONS["material_recovery"]["slider_range"][1]),
            value=float(ASSUMPTIONS["material_recovery"]["value"]), step=0.25,
            key="sus_material",
            help=ASSUMPTIONS["material_recovery"]["source"],
        )

    sus = sustainability_snapshot(source=source, co2_per_cell=co2_val, material_recovery=mat_val)

    with sus_right:
        s1, s2 = st.columns(2)
        co2_badge   = make_badge(ASSUMPTIONS["co2_manufacture"]["label"], "#b7791f")
        mat_badge   = make_badge(ASSUMPTIONS["material_recovery"]["label"], "#b7791f")

        with s1:
            _md_html(
                f"""
                <div style="background:#1e2a38;border:1px solid #2d374855;
                            border-radius:10px;padding:20px">
                    <div style="font-size:11px;color:#4a5568;margin-bottom:6px">
                        CO₂ avoided by reuse vs making a new cell
                    </div>
                    <div style="font-size:28px;font-weight:700;color:#68d391">
                        {sus['co2_avoided_by_reuse']:.2f} kg
                    </div>
                    <div style="font-size:11px;color:#4a5568;margin-top:4px">CO₂e avoided</div>
                    <div style="margin-top:10px">{co2_badge}</div>
                    <div style="font-size:11px;color:#4a5568;margin-top:8px;font-style:italic;line-height:1.4">
                        Reusing this cell avoids manufacturing one equivalent new cell.
                        Recycling instead saves only ~{sus['co2_recycling_credit']:.2f} kg
                        &nbsp;{make_badge("Cited estimate", "#b7791f")}&nbsp;
                        (≈15% cathode-material credit, Dunn et al. 2015 — hardcoded, no slider).
                    </div>
                </div>
                """
            )
        with s2:
            _md_html(
                f"""
                <div style="background:#1e2a38;border:1px solid #2d374855;
                            border-radius:10px;padding:20px">
                    <div style="font-size:11px;color:#4a5568;margin-bottom:6px">
                        Recoverable material value if recycled now
                    </div>
                    <div style="font-size:28px;font-weight:700;color:#f6ad55">
                        ${sus['material_recovery_value']:.2f}
                    </div>
                    <div style="font-size:11px;color:#4a5568;margin-top:4px">cobalt + lithium recovery</div>
                    <div style="margin-top:10px">{mat_badge}</div>
                    <div style="font-size:11px;color:#4a5568;margin-top:8px;font-style:italic;line-height:1.4">
                        LiCoO₂ cobalt content is the primary driver. Value tracks cobalt spot price
                        (Sommerville et al. 2020).
                    </div>
                </div>
                """
            )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── Assumption register ──
    with st.expander("All assumptions — sources and labels", expanded=False):
        for key, asmp in ASSUMPTIONS.items():
            badge_colour = "#b7791f" if "Cited" in asmp["label"] else "#718096"
            badge_html   = make_badge(asmp["label"], badge_colour)
            st.markdown(
                f"<div style='padding:12px 0;border-bottom:1px solid #2d3748'>"
                f"<div style='font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:6px'>"
                f"{asmp['unit']} &nbsp;—&nbsp; default {asmp['value']} &nbsp; {badge_html}"
                f"</div>"
                f"<div style='font-size:12px;color:#718096;line-height:1.6'>"
                f"{asmp['source']}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
