"""Page: Consequences (EOL Economics)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import _action_bar, _md_html, _empty_state, base_layout, render_card, metric_tile_html, NASA_CELL_IDS
from design_system import make_badge, BADGE_VALIDATED, BADGE_ESTIMATE, BADGE_ILLUST


# ---------------------------------------------------------------------------
# Solar + Storage Sizing — seasonal consumption shape presets
# ---------------------------------------------------------------------------
# Simple illustrative monthly-weight curves (sum to 1.0), not literature-cited —
# a user picking "Winter-heavy"/"Summer-heavy" is approximating their own bill's
# shape, not relying on this app for a validated load-profile model.
SEASONAL_SHAPES = {
    "Flat":          [1 / 12] * 12,
    "Winter-heavy":  [0.12, 0.11, 0.09, 0.07, 0.06, 0.05, 0.05, 0.05, 0.06, 0.08, 0.10, 0.16],
    "Summer-heavy":  [0.05, 0.05, 0.06, 0.08, 0.10, 0.12, 0.13, 0.13, 0.10, 0.07, 0.06, 0.05],
}

_COMPASS_DIRECTIONS = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315,
}


@st.cache_data(show_spinner=False, ttl=86400)
def _cached_pv_yield(lat: float, lon: float, peakpower_kwp: float, tilt_deg: float, azimuth_deg: float) -> dict:
    """Cached wrapper around pvgis_client.fetch_pv_yield() — PVGIS's climate-average
    output doesn't change hour to hour, and this is the only calculation on this
    page that hits a network API, so results are cached for a day per input combo.
    Passed to deployment_sizing.size_deployment() as its pv_yield_fn — azimuth_deg
    here is already in PVGIS convention (size_deployment converts once from the
    user's compass bearing before calling this for each pv_kwp step)."""
    from pvgis_client import fetch_pv_yield
    return fetch_pv_yield(lat=lat, lon=lon, peakpower_kwp=peakpower_kwp, tilt_deg=tilt_deg, azimuth_deg=azimuth_deg)


# ---------------------------------------------------------------------------
# Page: Consequences
# ---------------------------------------------------------------------------

def page_consequences(
    selected: str,
    df: pd.DataFrame,
    featured_dfs: dict,
    bundles: dict,
    rul_reliable: bool,
):
    _action_bar("consequences")
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
    source           = "nasa" if is_nasa else ("severson" if selected.startswith("S-") else "synth")

    peer_fades = [
        float(fdf.iloc[-1].get("fade_30_mah_cy", 0))
        for cid, fdf in featured_dfs.items()
        if (cid in NASA_CELL_IDS) == is_nasa and cid != selected
    ]
    fleet_fade_median = float(pd.Series(peer_fades).median()) if peer_fades else None

    # ── Page header ──
    st.markdown("# EOL Economics")
    st.markdown(f"##### Second-Life Economics + Sustainability · {selected}")

    _md_html(
        f"""
        <div style="background:rgba(183,121,31,0.07);border:1px solid rgba(183,121,31,0.25);
                    border-radius:10px;padding:14px 20px;margin-bottom:28px;
                    font-size:13px;color:#8896a8;line-height:1.7">
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
                    {fade_30*1000:.2f} <span style="font-size:13px;color:#8896a8">mAh/cy</span>
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
        "fit":      ("#48bb78", "#1a2e22", "Fit"),
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
                f"<div style='font-size:11px;color:#8896a8;margin-top:6px'>"
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
                        <div style="font-size:12px;color:#8896a8;margin-bottom:8px">{name}</div>
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
                "<div style='font-size:12px;color:#8896a8;margin-top:14px;font-style:italic'>"
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
            showarrow=False, font=dict(color="#48bb78", size=11),
            xanchor="left", yanchor="bottom",
        )

    bev_fig.update_layout(**base_layout(
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            title="State of Health (%)",
            autorange="reversed",
            zeroline=False,
        ),
        yaxis=dict(
            title=f"$ value{pack_label}",
            zeroline=False,
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

    # ── H1: NPV / Scenario Planner ──────────────────────────────────────────
    with st.expander("📈 NPV Scenario Planner — Replace / Wait / Repurpose", expanded=False):
        st.markdown(
            "<div style='font-size:12px;color:#8896a8;margin-bottom:12px'>"
            "5-year NPV comparison across three strategies. "
            "Energy value and replacement cost use cited defaults — adjust discount rate only.</div>",
            unsafe_allow_html=True,
        )
        _npv_rate = st.slider("Discount rate (WACC, %/yr)", 3.0, 20.0, 8.0, 0.5, key="npv_rate") / 100

        _energy_usd  = 80.0   # IEA 2024 LCOS range $60–140/kWh·yr — illustrative midpoint
        _repl_cost   = 150.0  # BNEF 2024 range $100–200/cell
        _repack_approx = float(st.session_state.get("sl_repack_cost", 30.0))
        _years = list(range(1, 6))

        def _pv_factor(r, t):
            return 1.0 / ((1 + r) ** t)

        _cap_now = CELL_NOMINAL_KWH.get(source, 0.0057)
        _a_annual = _cap_now * _energy_usd
        _a_npv = sum(_a_annual * _pv_factor(_npv_rate, t) for t in _years) - _repl_cost

        _rul_years = min((rul_pred / 200.0 / 12.0) if rul_pred else 1.5, 5.0)
        _b_cap_degraded = _cap_now * (soh / 100.0)
        _b_annual = _b_cap_degraded * _energy_usd
        _b_npv = (
            sum(_b_annual * _pv_factor(_npv_rate, t) for t in range(1, max(1, int(_rul_years)) + 1))
            + sum(_a_annual * _pv_factor(_npv_rate, t) for t in range(max(1, int(_rul_years)) + 1, 6))
            - _repl_cost * _pv_factor(_npv_rate, _rul_years)
        )

        _sl_annual = _cap_now * (soh / 100.0) * 0.85 * _energy_usd * 0.6
        _c_npv = sum(_sl_annual * _pv_factor(_npv_rate, t) for t in _years) - _repack_approx

        _strategies = [
            ("Replace Now",    _a_npv, "#fc8181",  f"Replace immediately at ${_repl_cost:.0f}/cell. Full capacity from cycle 1."),
            ("Wait to EOL",    _b_npv, "#f6ad55",  f"Run {_rul_years:.1f} yr at {soh:.0f}% SOH, then replace. Defers CAPEX."),
            ("Repurpose (2L)", _c_npv, "#68d391",  f"Second-life at 60% energy rate, ${_repack_approx:.0f} repack. Extends asset life."),
        ]
        _best_npv = max(_strategies, key=lambda x: x[1])

        _md_html(
            "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:8px'>"
            + "".join(
                f"<div style='background:#1e2a38;border:2px solid {('#2d3748' if lbl != _best_npv[0] else col)};border-radius:10px;padding:14px 16px'>"
                f"<div style='font-size:10px;font-weight:700;color:#4a5568;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:6px'>{lbl}</div>"
                f"<div style='font-size:26px;font-weight:800;color:{col}'>${v:,.0f}</div>"
                f"<div style='font-size:10px;color:#718096;margin-top:4px'>5-yr NPV</div>"
                + (f"<div style='font-size:10px;font-weight:700;color:{col};margin-top:6px'>★ Optimal</div>" if lbl == _best_npv[0] else "")
                + f"<div style='font-size:10px;color:#4a5568;margin-top:8px;line-height:1.4'>{desc}</div>"
                + "</div>"
                for lbl, v, col, desc in _strategies
            )
            + "</div>"
        )
        st.caption(
            f"Defaults: $80/kWh·yr energy {make_badge('IEA 2024', '#718096')} · "
            f"${_repl_cost:.0f}/cell replacement {make_badge('BNEF 2024', '#718096')} · "
            f"${_repack_approx:.0f} repack. Illustrative — not financial advice."
        )

    # ── H1b: Solar + Storage Sizing ─────────────────────────────────────────
    with st.expander("☀️ Solar + Storage Sizing", expanded=False):
        from deployment_sizing import SIZING_ASSUMPTIONS, size_deployment

        _md_html(
            f"""
            <div style="font-size:12px;color:#8896a8;margin-bottom:4px;line-height:1.6">
                Size a PV array + this second-life battery against a real consumption
                and tariff profile, using PVGIS (EU Commission public solar data) for
                PV yield — no local irradiance modelling. All figures below are in
                <strong>€</strong>, unlike the rest of this page ($) — this tool is
                grounded in EU tariff/solar data, so the currency switch is deliberate.
            </div>
            <div style="font-size:11px;color:#4a5568;margin-bottom:16px;line-height:1.6">
                Monthly energy-balance estimate, not an hourly dispatch simulation —
                it will tend to overstate real-world savings unless the self-consumption
                derating below is kept realistic. {BADGE_ILLUST} assumptions — expand
                "All assumptions" below the Sustainability Snapshot for details.
            </div>
            """
        )

        siz_c1, siz_c2, siz_c3 = st.columns(3)
        with siz_c1:
            st.markdown("**Consumption**")
            siz_annual_kwh = st.number_input(
                "Annual consumption (kWh)", min_value=0.0, value=4000.0, step=100.0,
                key="siz_annual_kwh",
            )
            siz_shape = st.selectbox(
                "Seasonal shape", list(SEASONAL_SHAPES.keys()), key="siz_shape",
            )
            st.markdown("**Tariff**")
            siz_tariff_high = st.number_input(
                "High tariff (€/kWh)", min_value=0.0, value=0.30, step=0.01,
                key="siz_tariff_high",
            )
            siz_tariff_low = st.number_input(
                "Low tariff (€/kWh)", min_value=0.0, value=0.12, step=0.01,
                key="siz_tariff_low",
            )
        with siz_c2:
            st.markdown("**PV site**")
            siz_lat = st.number_input(
                "Latitude", min_value=-90.0, max_value=90.0, value=45.80, step=0.01,
                key="siz_lat",
            )
            siz_lon = st.number_input(
                "Longitude", min_value=-180.0, max_value=180.0, value=15.98, step=0.01,
                key="siz_lon",
            )
            siz_tilt = st.slider("Tilt (°)", 0, 90, 30, key="siz_tilt")
            siz_orientation = st.selectbox(
                "Orientation", list(_COMPASS_DIRECTIONS.keys()), index=4,  # default S
                key="siz_orientation",
            )
            siz_area = st.number_input(
                "Available area (m²)", min_value=0.0, value=30.0, step=1.0,
                key="siz_area",
            )
        with siz_c3:
            st.markdown("**Financial constraints**")
            siz_max_payoff = st.number_input(
                "Max payoff period (years)", min_value=1.0, value=15.0, step=1.0,
                key="siz_max_payoff",
            )
            siz_max_invest = st.number_input(
                "Max investment (€)", min_value=0.0, value=25_000.0, step=500.0,
                key="siz_max_invest",
            )
            st.markdown(
                "<div style='font-size:11px;color:#4a5568;margin-top:8px;line-height:1.5'>"
                "Battery size is searched automatically (1-20 units of this cell) — "
                "not the pack-size input above, which is for the financial comparison "
                "cards only.</div>",
                unsafe_allow_html=True,
            )

        if st.button("Calculate", key="siz_calculate_btn"):
            monthly_consumption = [siz_annual_kwh * w for w in SEASONAL_SHAPES[siz_shape]]
            azimuth_compass = _COMPASS_DIRECTIONS[siz_orientation]

            with st.spinner("Querying PVGIS and sizing the deployment..."):
                result = size_deployment(
                    lat=siz_lat, lon=siz_lon, tilt_deg=siz_tilt,
                    azimuth_compass_deg=azimuth_compass,
                    available_area_m2=siz_area,
                    cell_kwh_per_cell=fin["current_kwh"],
                    monthly_consumption_kwh=monthly_consumption,
                    tariff_high_eur=siz_tariff_high, tariff_low_eur=siz_tariff_low,
                    max_payoff_years=siz_max_payoff, max_investment_eur=siz_max_invest,
                    pv_yield_fn=_cached_pv_yield,
                )
            st.session_state[f"siz_result_{selected}"] = result

        result = st.session_state.get(f"siz_result_{selected}")
        if result is None:
            _empty_state(
                "No sizing yet",
                "Set the inputs above and click Calculate to size a PV + battery deployment.",
                icon="☀",
            )
        elif result["winner"] is None:
            _empty_state(
                "Solar yield unavailable",
                result.get("constraint_note") or "PVGIS did not return usable data for any size explored. Try again shortly.",
                icon="☀",
            )
        else:
            winner = result["winner"]
            if result["pv_errors"]:
                st.warning(
                    f"PVGIS was unavailable for {len(result['pv_errors'])} of the PV sizes explored — "
                    "results below reflect only the sizes that succeeded."
                )
            if not result["feasible"]:
                st.info(result["constraint_note"])

            w_cols = st.columns(4)
            with w_cols[0]:
                render_card(metric_tile_html("PV size", f"{winner['pv_kwp']:.2f} kWp"))
            with w_cols[1]:
                render_card(metric_tile_html(
                    "Battery size", f"{winner['battery_kwh']*1000:.0f} Wh",
                    sub=f"{winner['n_cells']} cells",
                ))
            with w_cols[2]:
                payback_str = f"{winner['payback_years']:.1f} yr" if winner["payback_years"] is not None else "—"
                render_card(metric_tile_html(
                    "Payback", payback_str,
                    value_color="#48bb78" if result["feasible"] else "#f6ad55",
                ))
            with w_cols[3]:
                render_card(metric_tile_html(
                    "15-yr NPV", f"€{winner['npv_eur']:,.0f}",
                    sub=f"€{winner['investment_eur']:,.0f} investment",
                    value_color="#48bb78" if winner["npv_eur"] > 0 else "#fc8181",
                ))

            monthly = winner["monthly"]
            siz_fig = go.Figure()
            months_lbl = [m["month"] for m in monthly]
            siz_fig.add_trace(go.Bar(x=months_lbl, y=[m["direct_pv_kwh"] for m in monthly], name="Direct PV", marker_color="#f6ad55"))
            siz_fig.add_trace(go.Bar(x=months_lbl, y=[m["pv_via_battery_kwh"] for m in monthly], name="PV via battery", marker_color="#68d391"))
            siz_fig.add_trace(go.Bar(x=months_lbl, y=[m["arbitrage_kwh"] for m in monthly], name="Tariff arbitrage", marker_color="#63b3ed"))
            siz_fig.add_trace(go.Bar(x=months_lbl, y=[m["grid_import_kwh"] for m in monthly], name="Still grid-import", marker_color="#4a5568"))
            siz_fig.update_layout(base_layout(
                barmode="stack",
                xaxis=dict(title="Month", tickmode="linear", zeroline=False),
                yaxis=dict(title="kWh / month", zeroline=False),
            ))
            st.plotly_chart(siz_fig, use_container_width=True)

            try:
                from battery_knowledge import get_document, INDUSTRY_CONTEXT_DOC_IDS
                ctx_text = get_document(INDUSTRY_CONTEXT_DOC_IDS[0])
            except Exception:
                ctx_text = None
            if ctx_text:
                render_card(
                    "<div style='font-size:10px;font-weight:700;color:#718096;text-transform:uppercase;"
                    "letter-spacing:0.08em;margin-bottom:8px'>Industry context</div>"
                    f"<div style='font-size:12px;color:#a0aec0;line-height:1.6'>{ctx_text}</div>",
                    extra_style="margin-top:16px",
                )

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
                    <div style="font-size:28px;font-weight:700;color:#48bb78">
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
                f"<div style='font-size:12px;color:#8896a8;line-height:1.6'>"
                f"{asmp['source']}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

