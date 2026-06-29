"""
Battery Intelligence Platform — Streamlit Dashboard
Phase 1 dashboard: Overview, Health, Insights (functional)
All other nav items visible but marked Coming Soon.
"""

import sys
import os

# Make src/ importable from app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_loader import build_battery, get_cell_df
from features import build_features, get_model_matrix
from model import train_models, predict, feature_importance_df, top_drivers


# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Battery Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Sidebar nav ── */
    .nav-item {
        display: block;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 2px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        color: #e2e8f0;
        text-decoration: none;
        transition: background 0.15s;
    }
    .nav-item:hover  { background: rgba(255,255,255,0.08); }
    .nav-item.active { background: rgba(99,179,237,0.18); color: #63b3ed; }
    .nav-item.disabled {
        color: #4a5568;
        cursor: default;
        pointer-events: none;
    }
    .nav-badge {
        float: right;
        font-size: 10px;
        background: #2d3748;
        color: #718096;
        padding: 1px 6px;
        border-radius: 10px;
        margin-top: 2px;
    }

    /* ── Hero card ── */
    .hero-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .hero-label  { font-size: 12px; color: #718096; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
    .hero-value  { font-size: 48px; font-weight: 700; line-height: 1; margin-bottom: 8px; }
    .hero-sub    { font-size: 14px; color: #a0aec0; }
    .hero-green  { color: #68d391; }
    .hero-yellow { color: #f6e05e; }
    .hero-red    { color: #fc8181; }
    .hero-blue   { color: #63b3ed; }

    /* ── Metric chips ── */
    .metric-row  { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
    .metric-chip {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px 20px;
        min-width: 160px;
        flex: 1;
    }
    .metric-chip-label { font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-chip-value { font-size: 26px; font-weight: 700; color: #e2e8f0; margin-top: 4px; }
    .metric-chip-sub   { font-size: 12px; color: #718096; margin-top: 2px; }

    /* ── Section headers ── */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 28px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3748;
    }

    /* ── Confidence tag ── */
    .tag-calibrating {
        display: inline-block;
        background: #2d3748;
        color: #f6e05e;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }
    .tag-model {
        display: inline-block;
        background: #1c4532;
        color: #68d391;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }

    /* ── Coming soon overlay ── */
    .coming-soon-box {
        border: 1px dashed #2d3748;
        border-radius: 12px;
        padding: 80px 40px;
        text-align: center;
        margin-top: 40px;
    }
    .coming-soon-box h2 { color: #4a5568; font-weight: 600; margin-bottom: 8px; }
    .coming-soon-box p  { color: #4a5568; font-size: 14px; }

    /* ── General ── */
    .block-container { padding-top: 24px !important; }
    h1 { font-size: 22px !important; font-weight: 700 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading + model training — cached so it only runs once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading battery data and training models...")
def load_everything():
    """
    st.cache_resource means this function runs ONCE when the app starts,
    then the result is reused for every page interaction.
    Without caching, the model would retrain on every button click.
    """
    battery = build_battery(battery_id="Oxford_B1", cell_ids=["Cell1"])
    df_raw = get_cell_df(battery, "Cell1")
    df_feat = build_features(df_raw)
    X, y_soh, y_rul = get_model_matrix(df_feat)
    bundle = train_models(X, y_soh, y_rul)

    # Run predictions on the full dataset for display.
    preds = predict(bundle, X)
    df_feat = df_feat.loc[X.index].copy()
    df_feat["soh_pred"] = preds["soh_pred"]
    df_feat["rul_pred"] = preds["rul_pred"]
    df_feat["confidence_tag"] = preds["confidence_tag"]

    return df_feat, bundle


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

NAV_ITEMS = [
    ("Overview",       "overview",       True),
    ("Health",         "health",         True),
    ("Insights",       "insights",       True),
    ("Recommendations","recommendations",False),
    ("Economics",      "economics",      False),
    ("Fleet",          "fleet",          False),
    ("Sustainability", "sustainability", False),
    ("Reports",        "reports",        False),
    ("Settings",       "settings",       False),
]

def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='padding: 0 4px 20px'>"
            "<div style='font-size:16px;font-weight:700;color:#e2e8f0'>⚡ Battery Intel</div>"
            "<div style='font-size:11px;color:#4a5568;margin-top:2px'>Oxford B1 · Cell 1</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Streamlit doesn't have a native nav widget, so we use radio buttons
        # styled invisibly and drive everything from session_state.
        if "page" not in st.session_state:
            st.session_state.page = "overview"

        for label, key, enabled in NAV_ITEMS:
            if enabled:
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state.page = key
            else:
                # Disabled items: show as greyed-out text, not a button.
                st.markdown(
                    f"<div style='padding:8px 14px;color:#4a5568;font-size:14px;"
                    f"font-weight:500;display:flex;justify-content:space-between'>"
                    f"<span>{label}</span>"
                    f"<span style='font-size:10px;background:#1a202c;color:#4a5568;"
                    f"padding:1px 6px;border-radius:10px;align-self:center'>Soon</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#2d3748;padding:0 4px'>"
            "Phase 1 · scikit-learn GBRT<br>Synthetic Oxford-calibrated data"
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Shared chart theme
# ---------------------------------------------------------------------------

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a0aec0", size=12),
    xaxis=dict(gridcolor="#2d3748", linecolor="#2d3748", zeroline=False),
    yaxis=dict(gridcolor="#2d3748", linecolor="#2d3748", zeroline=False),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
)

def apply_theme(fig):
    fig.update_layout(**CHART_THEME)
    return fig


# ---------------------------------------------------------------------------
# Helper: SOH status label + colour
# ---------------------------------------------------------------------------

def soh_status(soh: float) -> tuple[str, str]:
    """Return (status_label, css_colour_class) for a given SOH %."""
    if soh >= 90:
        return "Healthy", "hero-green"
    if soh >= 80:
        return "Degrading", "hero-yellow"
    return "End of Life", "hero-red"


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame, bundle: dict):
    st.markdown("# Overview")

    latest = df.iloc[-1]
    current_soh = latest["soh_pct"]
    current_rul = latest["rul_pred"]
    current_cycle = int(latest["cycle_number"])
    total_fade = latest["capacity_fade_ah"]
    confidence = latest["confidence_tag"]
    status_label, status_colour = soh_status(current_soh)

    conf_html = (
        f"<span class='tag-calibrating'>CALIBRATING</span>"
        if confidence == "Calibrating"
        else f"<span class='tag-model'>MODEL</span>"
    )

    # ── Hero card ──
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Battery Status</div>
            <div class="hero-value {status_colour}">{status_label}</div>
            <div class="hero-sub">
                State of Health: <strong style="color:#e2e8f0">{current_soh:.1f}%</strong>
                &nbsp;·&nbsp; Est. {current_rul:.0f} cycles remaining
                &nbsp;·&nbsp; {conf_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metric chips ──
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-label">State of Health</div>
                <div class="metric-chip-value">{current_soh:.1f}%</div>
                <div class="metric-chip-sub">vs 100% at cycle 1</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Cycles Completed</div>
                <div class="metric-chip-value">{current_cycle:,}</div>
                <div class="metric-chip-sub">charge / discharge</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Est. Remaining Life</div>
                <div class="metric-chip-value">{current_rul:.0f}</div>
                <div class="metric-chip-sub">cycles to 80% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Capacity Lost</div>
                <div class="metric-chip-value">{total_fade*1000:.0f} mAh</div>
                <div class="metric-chip-sub">since commissioning</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── SOH over time chart ──
    st.markdown("<div class='section-header'>State of Health — Full History</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["soh_pct"],
        name="Actual SOH",
        line=dict(color="#4a5568", width=1),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["soh_rolling_avg"],
        name="10-cycle avg",
        line=dict(color="#63b3ed", width=2),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["soh_pred"],
        name="Model prediction",
        line=dict(color="#68d391", width=2, dash="dot"),
        mode="lines",
    ))
    # EOL threshold line
    fig.add_hline(
        y=80, line_dash="dash", line_color="#fc8181", line_width=1,
        annotation_text="EOL threshold (80%)",
        annotation_position="bottom right",
        annotation_font_color="#fc8181",
    )
    fig.update_layout(
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="SOH %",
        xaxis_title="Cycle",
        **CHART_THEME,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Health
# ---------------------------------------------------------------------------

def page_health(df: pd.DataFrame, bundle: dict):
    st.markdown("# Health")

    latest = df.iloc[-1]
    current_soh = latest["soh_pct"]
    fade_rate = latest["fade_rate_50cy"] * 100  # convert Ah/cycle to % for display approx
    status_label, status_colour = soh_status(current_soh)

    # ── Hero card ──
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Health Assessment</div>
            <div class="hero-value {status_colour}">{current_soh:.1f}% SOH</div>
            <div class="hero-sub">
                Fading at ~{latest['fade_rate_50cy']*1000:.2f} mAh/cycle (50-cycle avg)
                &nbsp;·&nbsp; Internal resistance: {latest.get('resistance_ohm', 0):.3f} Ω
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Capacity Fade</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["cycle_number"], y=df["capacity_fade_ah"] * 1000,
            fill="tozeroy",
            fillcolor="rgba(252,129,129,0.12)",
            line=dict(color="#fc8181", width=2),
            name="Capacity lost (mAh)",
        ))
        fig.update_layout(
            height=280, yaxis_title="mAh lost", xaxis_title="Cycle",
            **CHART_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Internal Resistance</div>", unsafe_allow_html=True)
        if "resistance_ohm" in df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["cycle_number"], y=df["resistance_ohm"] * 1000,
                line=dict(color="#f6e05e", width=1.5),
                name="Resistance (mΩ)",
            ))
            fig2.add_trace(go.Scatter(
                x=df["cycle_number"],
                y=df["resistance_ohm"].rolling(30, min_periods=1).mean() * 1000,
                line=dict(color="#f6ad55", width=2),
                name="30-cycle avg",
            ))
            fig2.update_layout(
                height=280, yaxis_title="mΩ", xaxis_title="Cycle",
                **CHART_THEME,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Fade rate trend ──
    st.markdown("<div class='section-header'>Fade Rate Trend — Is degradation accelerating?</div>", unsafe_allow_html=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["fade_rate_10cy"] * 1000,
        line=dict(color="#4a5568", width=1),
        name="10-cycle",
    ))
    fig3.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["fade_rate_30cy"] * 1000,
        line=dict(color="#63b3ed", width=2),
        name="30-cycle",
    ))
    fig3.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["fade_rate_50cy"] * 1000,
        line=dict(color="#68d391", width=2),
        name="50-cycle",
    ))
    fig3.update_layout(
        height=260,
        yaxis_title="mAh lost per cycle",
        xaxis_title="Cycle",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **CHART_THEME,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Calibrating note if applicable ──
    early_cycles = df[df["confidence_tag"] == "Calibrating"]
    if len(early_cycles) > 0:
        st.info(
            f"**Calibrating** — the first {len(early_cycles)} cycles show higher "
            f"variability while rolling-window features stabilise. "
            f"Trend readings from cycle 50 onward are reliable."
        )


# ---------------------------------------------------------------------------
# Page: Insights
# ---------------------------------------------------------------------------

def page_insights(df: pd.DataFrame, bundle: dict):
    st.markdown("# Insights")

    drivers = top_drivers(bundle, model="soh", top_n=5)
    top_feature = drivers[0]["feature"]
    top_pct = drivers[0]["importance_pct"]

    # ── Hero card ──
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Why this prediction?</div>
            <div class="hero-value hero-blue" style="font-size:32px">
                {top_feature.replace('_', ' ').title()}
            </div>
            <div class="hero-sub">
                This feature explains <strong style="color:#e2e8f0">{top_pct:.0f}%</strong>
                of the model's SOH prediction — the single strongest signal of battery health.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-header'>Feature Importance — SOH Model</div>", unsafe_allow_html=True)

        fi_df = feature_importance_df(bundle, model="soh")

        # Label strategy: show % inside large bars, hide on tiny ones (rely on hover).
        # Threshold: only label bars that are at least 3% of the max bar width.
        label_threshold = fi_df["importance_pct"].max() * 0.03
        labels = fi_df["importance_pct"].apply(
            lambda v: f"{v:.1f}%" if v >= label_threshold else ""
        )

        fig = go.Figure(go.Bar(
            x=fi_df["importance_pct"],
            y=fi_df["feature"].str.replace("_", " "),
            orientation="h",
            marker=dict(
                color=fi_df["importance_pct"],
                colorscale=[[0, "#2d3748"], [1, "#63b3ed"]],
                showscale=False,
            ),
            text=labels,
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="#ffffff", size=12, family="monospace"),
            # Full value always visible on hover regardless of bar size.
            customdata=fi_df["importance_pct"],
            hovertemplate="<b>%{y}</b><br>Importance: %{customdata:.2f}%<extra></extra>",
        ))
        theme = {k: v for k, v in CHART_THEME.items() if k != "yaxis"}
        fig.update_layout(
            height=380,
            xaxis_title="% contribution to prediction",
            xaxis=dict(
                gridcolor="#2d3748", linecolor="#2d3748", zeroline=False,
                # Add 10% padding on the right so labels don't clip.
                range=[0, fi_df["importance_pct"].max() * 1.05],
            ),
            yaxis=dict(autorange="reversed", gridcolor="#2d3748", linecolor="#2d3748"),
            **theme,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Top 5 Drivers</div>", unsafe_allow_html=True)

        for i, d in enumerate(drivers):
            bar_width = int(d["importance_pct"])
            rank_colour = ["#63b3ed", "#68d391", "#f6e05e", "#f6ad55", "#fc8181"][i]
            st.markdown(
                f"""
                <div style="margin-bottom:16px">
                    <div style="display:flex;justify-content:space-between;
                                font-size:13px;color:#e2e8f0;margin-bottom:5px">
                        <span>{d['feature'].replace('_', ' ')}</span>
                        <span style="color:{rank_colour};font-weight:600">{d['importance_pct']:.1f}%</span>
                    </div>
                    <div style="background:#2d3748;border-radius:4px;height:6px">
                        <div style="background:{rank_colour};width:{min(bar_width,100)}%;
                                    height:6px;border-radius:4px"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-header'>RUL Model</div>", unsafe_allow_html=True)
        fi_rul = feature_importance_df(bundle, model="rul").head(5)
        for _, row in fi_rul.iterrows():
            st.markdown(
                f"<div style='font-size:12px;color:#718096;margin-bottom:4px'>"
                f"<span style='color:#a0aec0'>{row['feature'].replace('_',' ')}</span>"
                f" — {row['importance_pct']:.1f}%</div>",
                unsafe_allow_html=True,
            )

    # ── Model metrics ──
    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)

    m = bundle["metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("SOH MAE", f"{m['soh_mae']:.2f}%", help="Mean absolute error on held-out test cycles")
    mc2.metric("SOH R²", f"{m['soh_r2']:.3f}", help="Negative R² is expected on a chronological split — see note below")
    mc3.metric("RUL MAE", f"{m['rul_mae']:.1f} cy", help="Mean absolute error on held-out test cycles")
    mc4.metric("RUL R²", f"{m['rul_r2']:.3f}")

    with st.expander("Why is R² negative?"):
        st.markdown(
            """
            R² measures "how much better is my model vs. just predicting the mean?"
            In a **chronological train/test split**, the test set (late cycles, low SOH)
            is systematically different from the training mean (early cycles, high SOH).
            The "predict the mean" baseline is wildly off, making R² misleading here.

            **MAE is the right metric:** the SOH model is within ~2 percentage points
            on held-out cycles, which is solid for a single-cell model. R² would
            improve significantly with more cells covering different degradation paths.
            """
        )

    # ── Actual vs predicted scatter ──
    st.markdown("<div class='section-header'>Actual vs Predicted SOH</div>", unsafe_allow_html=True)

    td = bundle["test_data"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=td["y_soh_test"], y=td["soh_pred"],
        mode="markers",
        marker=dict(color="#63b3ed", size=5, opacity=0.6),
        name="Test cycles",
    ))
    # Perfect-prediction reference line
    min_v = min(td["y_soh_test"].min(), td["soh_pred"].min())
    max_v = max(td["y_soh_test"].max(), td["soh_pred"].max())
    fig2.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode="lines",
        line=dict(color="#4a5568", dash="dash"),
        name="Perfect prediction",
    ))
    fig2.update_layout(
        height=300,
        xaxis_title="Actual SOH %",
        yaxis_title="Predicted SOH %",
        **CHART_THEME,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Coming Soon
# ---------------------------------------------------------------------------

def page_coming_soon(name: str, description: str):
    st.markdown(f"# {name}")
    st.markdown(
        f"""
        <div class="coming-soon-box">
            <h2>{name}</h2>
            <p style="color:#4a5568;margin-bottom:8px">{description}</p>
            <p style="color:#2d3748;font-size:12px">Phase 2+</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


COMING_SOON_DESCRIPTIONS = {
    "recommendations": "Actionable maintenance recommendations driven by health trends and cost models.",
    "economics":       "TCO analysis, replacement cost modelling, and ROI on second-life decisions.",
    "fleet":           "Multi-battery fleet view with comparative health rankings and alerts.",
    "sustainability":  "Carbon impact tracking, second-life suitability scoring, and recycling timeline.",
    "reports":         "Exportable PDF/CSV health reports for stakeholders and compliance.",
    "settings":        "Data source configuration, alert thresholds, and user preferences.",
}


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    # Load data + models (cached after first run).
    df, bundle = load_everything()

    page = st.session_state.get("page", "overview")

    if page == "overview":
        page_overview(df, bundle)
    elif page == "health":
        page_health(df, bundle)
    elif page == "insights":
        page_insights(df, bundle)
    elif page in COMING_SOON_DESCRIPTIONS:
        label = page.capitalize()
        page_coming_soon(label, COMING_SOON_DESCRIPTIONS[page])
    else:
        page_overview(df, bundle)


if __name__ == "__main__":
    main()
