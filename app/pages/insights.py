import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # app/ dir for utils, sidebar
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from utils import (
    base_layout, LEGEND_H, PLOTLY_CONFIG, _md_html, soh_status,
    friendly, _soh_sparkline_svg, _cell_provenance, _analysis_provenance,
    NASA_CELL_IDS,
)
from design_system import (
    make_badge, make_state_badge, provenance_banner,
    BADGE_VALIDATED, BADGE_ESTIMATE, BADGE_ILLUST, BADGE_UNAVAIL,
    BADGE_MEASURED, BADGE_SIMULATED, BADGE_SYNTHETIC,
    ACTION_META, CONF_META,
)
from model import feature_importance_df, top_drivers


def page_insights(df: pd.DataFrame, bundle: dict, cell_id: str,
                  cell_ids: list | None = None, active_fdfs: dict | None = None):
    st.markdown("# Insights")

    # Warn if this bundle was trained on fewer than 3 cells (uploaded data path)
    if bundle["metrics"].get("lco_limited"):
        n = bundle["metrics"].get("n_cells", "?")
        st.markdown(
            f"<div style='background:rgba(183,121,31,0.10);border:1px solid rgba(183,121,31,0.35);"
            f"border-radius:10px;padding:12px 18px;margin-bottom:16px'>"
            f"<span style='font-weight:700;color:#f6ad55'>⚠ LCO trained on {n} cells</span>"
            f"<span style='font-size:13px;color:#fefcbf;margin-left:10px'>"
            f"Reliability estimates are less stable than with 3+ cells. "
            f"Per-cell R² scores may not reflect true generalisation ability.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    drivers     = top_drivers(bundle, model="soh", top_n=5)
    top_feature = drivers[0]["feature"]
    top_pct     = drivers[0]["importance_pct"]

    is_nasa_cell = cell_id in NASA_CELL_IDS
    model_label = (
        f"NASA real model · {bundle['metrics'].get('n_cells', 4)} cells"
        if is_nasa_cell
        else f"Synthetic model · {bundle['metrics'].get('n_cells', 8)} cells"
    )
    n_cells_trained = bundle["metrics"].get("n_cells", "—")
    _md_html(
        f"""
        <div class="hero-card">
            <div class="hero-label">Why this prediction? · SOH model · {model_label}</div>
            <div class="hero-value hero-blue" style="font-size:32px">
                {friendly(top_feature)}
            </div>
            <div class="hero-sub">
                Top-ranked by <strong style="color:#e2e8f0">split importance ({top_pct:.0f}%)</strong>
                — see the SHAP Attribution tab below for correct feature attribution
                that handles correlated fade-rate features.
                Internal resistance tracks SEI layer growth, the dominant degradation
                mechanism in LiCoO&#x2082; cells.
            </div>
        </div>
        """
    )

    def _importance_bar(fi: "pd.DataFrame", color_hi: str, height: int = 340) -> "go.Figure":
        labels = fi["importance_pct"].apply(
            lambda v: f"{v:.1f}%" if v >= fi["importance_pct"].max() * 0.03 else ""
        )
        fig = go.Figure(go.Bar(
            x=fi["importance_pct"], y=fi["label"],
            orientation="h",
            marker=dict(
                color=fi["importance_pct"],
                colorscale=[[0, "#1e2a38"], [0.4, "#1e2a38"], [1, color_hi]],
                showscale=False,
            ),
            text=labels,
            textposition="inside", insidetextanchor="end",
            textfont=dict(color="#ffffff", size=11),
            customdata=fi["importance_pct"],
            hovertemplate="<b>%{y}</b><br>Importance: %{customdata:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            height=height,
            **base_layout(
                xaxis=dict(title="% importance", gridcolor="#232d3b", linecolor="#2d3748",
                           zeroline=False, range=[0, fi["importance_pct"].max() * 1.12]),
                yaxis=dict(autorange="reversed", gridcolor="#232d3b", linecolor="#2d3748"),
            ),
        )
        return fig

    # ── SHAP feature attribution ─────────────────────────────────────────────
    # SHAP (SHapley Additive exPlanations) correctly handles correlated features,
    # unlike split-based importances which over-credit the first feature in a
    # correlated group (e.g. fade_rate_10cy / _30cy / _50cy all measure the same
    # signal; split importance splits credit unpredictably among them).
    # SHAP uses the Shapley value from cooperative game theory: each feature's
    # contribution is its average marginal contribution across all feature subsets.
    _shap_tab, _imp_tab = st.tabs(["SHAP Attribution (recommended)", "Split Importance (legacy)"])

    with _shap_tab:
        _md_html(provenance_banner(
            "simulated" if cell_id in NASA_CELL_IDS else "synthetic",
            "SHAP values are computed from the trained GBRT model. They describe which features "
            "pushed this cell's SOH prediction higher or lower than the population average, "
            "correctly accounting for correlated features. "
            "<strong>Split-based importances (legacy tab) over-credit the first feature in a "
            "correlated group and are not reliable for feature selection.</strong>"
        ))
        try:
            import shap as _shap
            import numpy as _np_shap
            _scaler = bundle["scaler"]
            _feat_names = bundle["feature_names"]
            _X_test = bundle["test_data"]["X_test"]
            _X_test_sc = _scaler.transform(_X_test)

            # Compute SHAP for SOH model — TreeExplainer is exact for GBRT
            _expl_soh = _shap.TreeExplainer(bundle["soh_model"])
            _shap_soh  = _expl_soh.shap_values(_X_test_sc)
            _mean_abs_soh = _np_shap.abs(_shap_soh).mean(axis=0)
            _shap_df_soh  = pd.DataFrame({
                "feature": _feat_names,
                "mean_abs_shap": _mean_abs_soh,
                "label": [friendly(f) for f in _feat_names],
            }).sort_values("mean_abs_shap", ascending=True).tail(12)

            # Compute SHAP for RUL model
            _expl_rul = _shap.TreeExplainer(bundle["rul_model"])
            _shap_rul  = _expl_rul.shap_values(_X_test_sc)
            _mean_abs_rul = _np_shap.abs(_shap_rul).mean(axis=0)
            _shap_df_rul  = pd.DataFrame({
                "feature": _feat_names,
                "mean_abs_shap": _mean_abs_rul,
                "label": [friendly(f) for f in _feat_names],
            }).sort_values("mean_abs_shap", ascending=True).tail(12)

            _sc1, _sc2 = st.columns(2)
            for _col, _shap_df, _color, _model_lbl in [
                (_sc1, _shap_df_soh, "#63b3ed", "SOH"),
                (_sc2, _shap_df_rul, "#48bb78", "RUL"),
            ]:
                with _col:
                    st.markdown(f"<div class='section-header'>SHAP Attribution — {_model_lbl} Model</div>", unsafe_allow_html=True)
                    _max_shap = _shap_df["mean_abs_shap"].max()
                    _fig_shap = go.Figure(go.Bar(
                        x=_shap_df["mean_abs_shap"],
                        y=_shap_df["label"],
                        orientation="h",
                        marker=dict(
                            color=_shap_df["mean_abs_shap"],
                            colorscale=[[0, "#1e2a38"], [0.4, "#1e2a38"], [1, _color]],
                            showscale=False,
                        ),
                        text=[f"{v:.4f}" for v in _shap_df["mean_abs_shap"]],
                        textposition="inside", insidetextanchor="end",
                        textfont=dict(color="#ffffff", size=10),
                        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
                    ))
                    _fig_shap.update_layout(
                        height=340,
                        **base_layout(
                            xaxis=dict(title="Mean |SHAP value|", gridcolor="#232d3b", linecolor="#2d3748",
                                       zeroline=False, range=[0, _max_shap * 1.12]),
                            yaxis=dict(autorange="reversed", gridcolor="#232d3b", linecolor="#2d3748"),
                        ),
                    )
                    _fig_shap.update_layout(title=dict(
                        text=f"SHAP Mean |φ| — {_model_lbl} · Top 12 features",
                        font=dict(size=12, color="#a0aec0"), x=0,
                    ))
                    st.plotly_chart(_fig_shap, use_container_width=True,
                                   config={**PLOTLY_CONFIG, "toImageButtonOptions": {**PLOTLY_CONFIG["toImageButtonOptions"], "filename": f"shap_{_model_lbl.lower()}"}})

            # Single-cell SHAP waterfall for the selected cell's latest cycle
            _X_this = df[bundle["feature_names"]].dropna().tail(1)
            if len(_X_this) > 0:
                _X_this_sc = _scaler.transform(_X_this)
                _shap_this  = _expl_soh.shap_values(_X_this_sc)[0]
                _base_val   = float(_expl_soh.expected_value)
                _pred_val   = _base_val + float(_shap_this.sum())
                _wf_df = pd.DataFrame({
                    "feature": [friendly(f) for f in _feat_names],
                    "shap": _shap_this,
                }).sort_values("shap", key=abs, ascending=True).tail(10)

                st.markdown("<div class='section-header'>SHAP Waterfall — Why is this cell's SOH predicted at this value?</div>", unsafe_allow_html=True)
                _colors_wf = ["#48bb78" if v > 0 else "#fc8181" for v in _wf_df["shap"]]
                _fig_wf = go.Figure(go.Bar(
                    x=_wf_df["shap"], y=_wf_df["feature"],
                    orientation="h",
                    marker_color=_colors_wf,
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
                    text=[f"{v:+.4f}" for v in _wf_df["shap"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#a0aec0"),
                ))
                _fig_wf.update_layout(
                    height=320,
                    **base_layout(
                        xaxis=dict(title="SHAP value (impact on SOH prediction)", gridcolor="#232d3b",
                                   linecolor="#2d3748", zeroline=True, zerolinecolor="#4a5568"),
                        yaxis=dict(autorange="reversed", gridcolor="#232d3b", linecolor="#2d3748"),
                    ),
                )
                _fig_wf.update_layout(title=dict(
                    text=f"Waterfall: {cell_id} latest cycle · E[f(X)]={_base_val:.1f}% → prediction={_pred_val:.1f}%",
                    font=dict(size=12, color="#a0aec0"), x=0,
                ))
                st.plotly_chart(_fig_wf, use_container_width=True,
                               config={**PLOTLY_CONFIG, "toImageButtonOptions": {**PLOTLY_CONFIG["toImageButtonOptions"], "filename": "shap_waterfall"}})

        except ImportError:
            st.warning("SHAP not installed. Run `pip install shap` to enable SHAP attribution.")
        except Exception as _shap_e:
            st.info(f"SHAP computation unavailable: {_shap_e}")

    with _imp_tab:
        _md_html(provenance_banner(
            "simulated" if cell_id in NASA_CELL_IDS else "synthetic",
            "<strong>⚠ Split-based importances are unreliable for correlated features.</strong> "
            "fade_rate_10cy / _30cy / _50cy are heavily correlated; importance is distributed "
            "unpredictably among them based on which split the tree chooses first. "
            "Use the SHAP tab for correct attribution. "
            "These are shown here for reference only."
        ))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>Split Importance — SOH Model</div>", unsafe_allow_html=True)
            fi_soh = feature_importance_df(bundle, model="soh")
            fi_soh["label"] = fi_soh["feature"].map(friendly)
            st.plotly_chart(_importance_bar(fi_soh, "#63b3ed"), use_container_width=True)
        with col2:
            st.markdown("<div class='section-header'>Split Importance — RUL Model</div>", unsafe_allow_html=True)
            fi_rul = feature_importance_df(bundle, model="rul")
            fi_rul["label"] = fi_rul["feature"].map(friendly)
            st.plotly_chart(_importance_bar(fi_rul, "#48bb78"), use_container_width=True)

    # Explain the profile difference (uses whichever fi_soh/fi_rul were last computed)
    try:
        top_soh_feat = fi_soh.iloc[0]["label"]
        top_rul_feat = fi_rul.iloc[0]["label"]
        if top_soh_feat != top_rul_feat:
            st.markdown(
                f"<div style='font-size:12px;color:#718096;background:#1a202c;border-left:3px solid #2d3748;"
                f"padding:10px 14px;border-radius:4px;margin-bottom:12px'>"
                f"<strong style='color:#a0aec0'>Why the profiles differ:</strong> "
                f"{top_soh_feat} dominates SOH prediction because it tracks cumulative degradation. "
                f"{top_rul_feat} dominates RUL prediction because it determines how fast the remaining "
                f"life gets consumed — a different question.</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # ── Model metrics ──
    st.markdown("<div class='section-header'>Model Performance — Multi-cell Training</div>", unsafe_allow_html=True)

    m = bundle["metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    lco_soh_r2 = m.get("lco_soh_r2", float("nan"))
    lco_rul_r2 = m.get("lco_rul_r2", float("nan"))
    rul_ok     = m.get("rul_reliable", True)

    mc1.metric(
        "SOH R² (LCO)",
        f"{m.get('lco_soh_r2', float('nan')):.3f}" if not (lco_soh_r2 != lco_soh_r2) else "—",
        help="Leave-cell-out R² — model trained on N-1 cells, tested on the held-out cell. "
             "This is the honest generalisation metric, not a row-level split.",
    )
    mc2.metric(
        "SOH MAE",
        f"{m['soh_mae']:.2f}%",
        help="Mean absolute SOH error on held-out test cycles (chronological split within training set).",
    )
    rul_label = f"{lco_rul_r2:.3f}" if rul_ok else f"{lco_rul_r2:.3f}*"
    mc3.metric(
        "RUL (LCO)",
        rul_label,
        help="Leave-cell-out R² for RUL." + ("" if rul_ok else " * below 0.30 floor — shown as 'Not calibrated' in UI."),
    )
    n_cells = m.get("n_cells", "—")
    n_rows  = m.get("n_rows", 0)
    mc4.metric(
        "Training",
        f"{n_cells} cells",
        help=f"{n_cells} cells / {n_rows:,} cycles total in the training set.",
    )

    with st.expander("What does resistance explaining 74% of SOH mean physically?"):
        st.markdown(
            """
            **Internal resistance is a direct proxy for SEI layer thickness.**
            The solid electrolyte interphase (SEI) grows on the anode with each cycle,
            consuming lithium inventory and increasing impedance. Both capacity fade and
            resistance growth are caused by the same underlying mechanism — so resistance
            is highly predictive of SOH, not because the model found a statistical
            shortcut, but because this is the actual physics.

            Real BMS systems (Battery Management Systems) use internal resistance as a
            primary health indicator for exactly this reason.

            **Why temperature explains 30% of RUL but not SOH:**
            Temperature affects how *fast* the remaining life gets consumed, not where
            the cell is right now. A Cell8 (40°C) and Cell7 (20°C) at the same SOH have
            very different futures. The RUL model correctly learns that temperature is
            the key signal for how much life is left.
            """
        )

    # ── Actual vs Predicted ──
    st.markdown("<div class='section-header'>Actual vs Predicted SOH — Test Cycles (all cells)</div>", unsafe_allow_html=True)

    td       = bundle["test_data"]
    actual   = td["y_soh_test"]
    pred     = td["soh_pred"]
    ax_min   = min(actual.min(), pred.min()) - 0.5
    ax_max   = max(actual.max(), pred.max()) + 0.5

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[ax_min, ax_max], y=[ax_min, ax_max],
        mode="lines", line=dict(color="#2d3748", dash="dash", width=1),
        name="Perfect prediction", hoverinfo="skip",
    ))
    fig2.add_trace(go.Scatter(
        x=actual, y=pred, mode="markers",
        marker=dict(color="#63b3ed", size=4, opacity=0.4),
        name="Test cycles",
        hovertemplate="Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>",
    ))
    fig2.update_layout(
        height=300,
        **base_layout(
            xaxis=dict(title="Actual SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                       zeroline=False, range=[ax_min, ax_max]),
            yaxis=dict(title="Predicted SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                       zeroline=False, range=[ax_min, ax_max]),
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 🌡️ Temperature–Degradation Analysis (Arrhenius) ──
    st.markdown("<div class='section-header'>🌡️ Temperature–Degradation Analysis (Arrhenius)</div>", unsafe_allow_html=True)
    _afdfs = active_fdfs if active_fdfs is not None else {}
    if len(_afdfs) <= 1:
        st.info("Select multiple cells to enable cross-cell temperature analysis.")
    else:
        import numpy as _np_arr
        _arr_rows = []
        for _cid, _cdf in _afdfs.items():
            _mean_temp      = float(_cdf["temperature_c"].mean())
            _mean_fade_rate = float(_cdf["fade_rate_50cy"].mean()) if "fade_rate_50cy" in _cdf.columns else float("nan")
            _arr_rows.append({"cell_id": _cid, "mean_temp": _mean_temp, "mean_fade_rate": _mean_fade_rate})
        _arr_df = pd.DataFrame(_arr_rows).dropna()
        if len(_arr_df) >= 2:
            _x = (1000 / (_arr_df["mean_temp"] + 273.15)).values
            _y = _np_arr.log(_arr_df["mean_fade_rate"].clip(lower=1e-9)).values
            _slope, _intercept = _np_arr.polyfit(_x, _y, 1)
            _x_line = _np_arr.linspace(_x.min(), _x.max(), 50)
            _y_line = _slope * _x_line + _intercept
            Ea_J_per_mol = -_slope * 8.314
            Ea_eV        = Ea_J_per_mol / 96485
            _fig_arr = go.Figure()
            _fig_arr.add_trace(go.Scatter(
                x=_x_line.tolist(), y=_y_line.tolist(),
                mode="lines", name="Trendline",
                line=dict(color="#4a5568", width=1.5, dash="dash"),
                hoverinfo="skip",
            ))
            _fig_arr.add_trace(go.Scatter(
                x=_x.tolist(), y=_y.tolist(),
                mode="markers+text",
                text=_arr_df["cell_id"].tolist(),
                textposition="top center",
                textfont=dict(size=10, color="#a0aec0"),
                marker=dict(color="#63b3ed", size=10, line=dict(color="#1a202c", width=1)),
                name="Cells",
                hovertemplate="<b>%{text}</b><br>1000/T=%{x:.3f}<br>ln(fade)=%{y:.3f}<extra></extra>",
            ))
            _fig_arr.update_layout(
                **base_layout(
                    height=320, legend=LEGEND_H,
                    xaxis=dict(title="1000 / T (K⁻¹)", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                    yaxis=dict(title="ln(Fade Rate per Cycle)", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                ),
            )
            _fig_arr.update_layout(title=dict(text="Arrhenius Plot: ln(Fade Rate) vs 1000/T", font=dict(size=12, color="#a0aec0"), x=0))
            st.plotly_chart(_fig_arr, use_container_width=True, config={**PLOTLY_CONFIG, "toImageButtonOptions": {**PLOTLY_CONFIG["toImageButtonOptions"], "filename": "arrhenius_plot"}})

            # Resistance Ea from same cells
            _arr_rows_r = []
            for _cid, _cdf in _afdfs.items():
                _mean_temp  = float(_cdf["temperature_c"].mean())
                _mean_r     = float(_cdf["resistance_ohm"].mean()) if "resistance_ohm" in _cdf.columns else float("nan")
                _arr_rows_r.append({"cell_id": _cid, "mean_temp": _mean_temp, "mean_r": _mean_r})
            _arr_df_r = pd.DataFrame(_arr_rows_r).dropna()
            Ea_eV_r = float("nan")
            if len(_arr_df_r) >= 2:
                _xr = (1000 / (_arr_df_r["mean_temp"] + 273.15)).values
                _yr = _np_arr.log(_arr_df_r["mean_r"].clip(lower=1e-9)).values
                _slope_r, _ = _np_arr.polyfit(_xr, _yr, 1)
                Ea_eV_r = -_slope_r * 8.314 / 96485

            # Ea display boxes
            _ac1, _ac2, _ac3 = st.columns(3)
            _ea_color = "#48bb78" if 0.35 <= Ea_eV <= 0.75 else "#f6ad55"
            _ac1.metric("Ea — Capacity Fade", f"{Ea_eV:.3f} eV", help="Activation energy from ln(fade rate) vs 1/T. Literature: 0.4–0.6 eV for LiCoO₂ SEI.")
            if not _np_arr.isnan(Ea_eV_r):
                _ac2.metric("Ea — Resistance Growth", f"{Ea_eV_r:.3f} eV", help="Activation energy from ln(resistance) vs 1/T.")
            _ac3.metric("Literature ref (LiCoO₂ SEI)", "0.4–0.6 eV", help="Birkl et al. 2017, Waldmann et al. 2014.")
            _md_html(f"""<div style="background:rgba(26,32,44,0.7);border:1px solid #2d3748;border-radius:8px;padding:12px 16px;font-size:13px;margin-top:8px"><span style="color:{_ea_color};font-weight:700">Ea = {Ea_eV:.3f} eV ({Ea_J_per_mol:.0f} J/mol)</span><span style="color:#a0aec0;margin-left:10px">{"Within expected range for thermally-activated SEI growth." if 0.35 <= Ea_eV <= 0.75 else "Outside typical LiCoO₂ range — verify temperature spread across cells."}</span></div>""")

            _arr_c1, _arr_c2 = st.columns(2)
            with _arr_c1:
                _arr_display = _arr_df.copy()
                _arr_display["avg temp (°C)"]   = _arr_display["mean_temp"].map("{:.1f}".format)
                _arr_display["avg fade rate"]    = _arr_display["mean_fade_rate"].map("{:.6f}".format)
                _max_fade = _arr_display["mean_fade_rate"].max()
                _arr_display["relative stress"]  = (_arr_display["mean_fade_rate"] / max(_max_fade, 1e-12)).map("{:.2f}×".format)
                st.dataframe(
                    _arr_display[["cell_id", "avg temp (°C)", "avg fade rate", "relative stress"]].rename(columns={"cell_id": "Cell ID"}),
                    use_container_width=True, hide_index=True,
                )
            with _arr_c2:
                st.caption(
                    "A linear Arrhenius relationship confirms thermally-activated degradation (SEI growth). "
                    "Slope gives activation energy — steeper = more temperature-sensitive chemistry. "
                    "Pre-exponential factor A = exp(intercept) characterises attempt frequency."
                )
        else:
            st.info("Need at least 2 cells with valid data for Arrhenius plot.")
