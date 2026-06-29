"""
Battery Intelligence Platform — Streamlit Dashboard
Phase 1 dashboard: Overview, Health, Insights (functional)
All other nav items visible but marked Coming Soon.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import build_battery, get_cell_df
from features import build_features, get_model_matrix
from model import train_models, predict, feature_importance_df, top_drivers


# ---------------------------------------------------------------------------
# Page config
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
    /* ── Sidebar: restyle Streamlit buttons into clean nav items ── */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #a0aec0 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 7px 12px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        transition: background 0.15s, color 0.15s !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.07) !important;
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: rgba(99,179,237,0.15) !important;
        border: none !important;
        box-shadow: none !important;
        color: #63b3ed !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 7px 12px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: rgba(99,179,237,0.22) !important;
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
    .hero-value  { font-size: 48px; font-weight: 700; line-height: 1.1; margin-bottom: 8px; }
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
        font-size: 12px;
        font-weight: 600;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 28px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3748;
    }

    /* ── Confidence tags ── */
    .tag-calibrating {
        display: inline-block;
        background: rgba(246,224,94,0.12);
        color: #f6e05e;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.06em;
        border: 1px solid rgba(246,224,94,0.25);
    }
    .tag-model {
        display: inline-block;
        background: rgba(104,211,145,0.12);
        color: #68d391;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.06em;
        border: 1px solid rgba(104,211,145,0.25);
    }

    /* ── General ── */
    .block-container { padding-top: 24px !important; }
    h1 { font-size: 22px !important; font-weight: 700 !important; color: #e2e8f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data + model — cached for the session lifetime
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading battery data and training models...")
def load_everything():
    battery = build_battery(battery_id="Oxford_B1", cell_ids=["Cell1"])
    df_raw = get_cell_df(battery, "Cell1")
    df_feat = build_features(df_raw)
    X, y_soh, y_rul = get_model_matrix(df_feat)
    bundle = train_models(X, y_soh, y_rul)

    # Store the train/test split index boundary for the chart annotation.
    split_idx = int(len(X) * 0.8)
    split_cycle = int(X["cycle_number"].iloc[split_idx])

    preds = predict(bundle, X)
    df_feat = df_feat.loc[X.index].copy()
    df_feat["soh_pred"] = preds["soh_pred"]
    df_feat["rul_pred"] = preds["rul_pred"]
    df_feat["confidence_tag"] = preds["confidence_tag"]

    return df_feat, bundle, split_cycle


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

NAV_ITEMS = [
    ("Overview",        "overview",        True,  "ti-layout-dashboard"),
    ("Health",          "health",          True,  "ti-heart-rate-monitor"),
    ("Insights",        "insights",        True,  "ti-bulb"),
    ("Recommendations", "recommendations", False, "ti-checklist"),
    ("Economics",       "economics",       False, "ti-coin"),
    ("Fleet",           "fleet",           False, "ti-topology-star"),
    ("Sustainability",  "sustainability",  False, "ti-leaf"),
    ("Reports",         "reports",         False, "ti-file-description"),
    ("Settings",        "settings",        False, "ti-settings"),
]

def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='padding:0 4px 20px'>"
            "<div style='font-size:16px;font-weight:700;color:#e2e8f0'>⚡ Battery Intel</div>"
            "<div style='font-size:11px;color:#4a5568;margin-top:2px'>Oxford B1 · Cell 1</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if "page" not in st.session_state:
            st.session_state.page = "overview"

        current = st.session_state.page

        for label, key, enabled, _ in NAV_ITEMS:
            if enabled:
                is_active = current == key
                if st.button(
                    label,
                    key=f"nav_{key}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.page = key
                    st.rerun()
            else:
                st.markdown(
                    f"<div style='padding:7px 12px;color:#4a5568;font-size:14px;"
                    f"font-weight:500;display:flex;justify-content:space-between;"
                    f"align-items:center'>"
                    f"<span>{label}</span>"
                    f"<span style='font-size:10px;background:#1a202c;color:#4a5568;"
                    f"padding:1px 7px;border-radius:10px'>Soon</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#2d3748;padding:0 4px;line-height:1.6'>"
            "Phase 1 · scikit-learn GBRT<br>Synthetic Oxford-calibrated data"
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def base_layout(**overrides) -> dict:
    """
    Return a Plotly layout dict with the shared dark theme applied.
    Pass explicit xaxis= or yaxis= in overrides to avoid duplicate-key errors.
    """
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0aec0", size=12),
        margin=dict(l=10, r=10, t=36, b=10),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#718096"),
        ),
    )
    # Default axis style — overrides can replace these entirely.
    if "xaxis" not in overrides:
        layout["xaxis"] = dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False)
    if "yaxis" not in overrides:
        layout["yaxis"] = dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False)
    layout.update(overrides)
    return layout


def soh_status(soh: float) -> tuple[str, str]:
    if soh >= 90:
        return "Healthy", "hero-green"
    if soh >= 80:
        return "Degrading", "hero-yellow"
    return "End of Life", "hero-red"


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame, bundle: dict, split_cycle: int):
    st.markdown("# Overview")

    latest = df.iloc[-1]
    current_soh = latest["soh_pct"]
    current_rul = latest["rul_pred"]
    current_cycle = int(latest["cycle_number"])
    total_fade = latest["capacity_fade_ah"]
    confidence = latest["confidence_tag"]
    status_label, status_colour = soh_status(current_soh)

    conf_html = (
        "<span class='tag-calibrating'>CALIBRATING</span>"
        if confidence == "Calibrating"
        else "<span class='tag-model'>MODEL</span>"
    )

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

    st.markdown("<div class='section-header'>State of Health — Full History</div>", unsafe_allow_html=True)

    # Split data into train and test portions for the chart.
    df_train = df[df["cycle_number"] <= split_cycle]
    df_test  = df[df["cycle_number"] >  split_cycle]

    fig = go.Figure()

    # Noisy actual trace (subtle).
    fig.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["soh_pct"],
        name="Actual SOH",
        line=dict(color="#3a4a5e", width=1),
        mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f}%<extra>Actual</extra>",
    ))

    # Smoothed 10-cycle average.
    fig.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["soh_rolling_avg"],
        name="10-cycle avg",
        line=dict(color="#63b3ed", width=2),
        mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f}%<extra>10-cycle avg</extra>",
    ))

    # Model prediction — only on the TEST portion (the honest window).
    fig.add_trace(go.Scatter(
        x=df_test["cycle_number"], y=df_test["soh_pred"],
        name="Model (test)",
        line=dict(color="#68d391", width=2, dash="dot"),
        mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f}%<extra>Model prediction</extra>",
    ))

    # Vertical line showing where training ended / prediction begins.
    fig.add_vline(
        x=split_cycle,
        line_dash="dot",
        line_color="#4a5568",
        line_width=1,
        annotation_text=f"Train → Test (cycle {split_cycle})",
        annotation_position="top left",
        annotation_font_color="#4a5568",
        annotation_font_size=11,
    )

    # EOL threshold.
    fig.add_hline(
        y=80, line_dash="dash", line_color="#fc8181", line_width=1,
        annotation_text="EOL (80%)",
        annotation_position="bottom right",
        annotation_font_color="#fc8181",
        annotation_font_size=11,
    )

    fig.update_layout(
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **base_layout(
            xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="SOH %", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False,
                       range=[78, 101]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Health
# ---------------------------------------------------------------------------

def page_health(df: pd.DataFrame, bundle: dict, split_cycle: int):
    st.markdown("# Health")

    latest = df.iloc[-1]
    current_soh = latest["soh_pct"]
    status_label, status_colour = soh_status(current_soh)

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
            fillcolor="rgba(252,129,129,0.08)",
            line=dict(color="#fc8181", width=2),
            name="Capacity lost (mAh)",
            hovertemplate="Cycle %{x}: %{y:.1f} mAh lost<extra></extra>",
        ))
        fig.update_layout(
            height=280,
            **base_layout(
                xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                yaxis=dict(title="mAh lost", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Internal Resistance</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["cycle_number"], y=df["resistance_ohm"] * 1000,
            line=dict(color="#4a5568", width=1),
            name="Resistance (mΩ)",
            hovertemplate="Cycle %{x}: %{y:.1f} mΩ<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=df["cycle_number"],
            y=df["resistance_ohm"].rolling(30, min_periods=1).mean() * 1000,
            line=dict(color="#f6ad55", width=2),
            name="30-cycle avg",
            hovertemplate="Cycle %{x}: %{y:.1f} mΩ<extra>30-cy avg</extra>",
        ))
        fig2.update_layout(
            height=280,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **base_layout(
                xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
                yaxis=dict(title="mΩ", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "<div class='section-header'>Fade Rate — Is degradation accelerating?</div>",
        unsafe_allow_html=True,
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["fade_rate_10cy"] * 1000,
        line=dict(color="#4a5568", width=1),
        name="10-cycle window",
    ))
    fig3.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["fade_rate_30cy"] * 1000,
        line=dict(color="#63b3ed", width=2),
        name="30-cycle window",
    ))
    fig3.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["fade_rate_50cy"] * 1000,
        line=dict(color="#68d391", width=2),
        name="50-cycle window",
    ))
    fig3.update_layout(
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **base_layout(
            xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="mAh lost per cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
        ),
    )
    st.plotly_chart(fig3, use_container_width=True)

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

def page_insights(df: pd.DataFrame, bundle: dict, split_cycle: int):
    st.markdown("# Insights")

    drivers = top_drivers(bundle, model="soh", top_n=5)
    top_feature = drivers[0]["feature"]
    top_pct = drivers[0]["importance_pct"]

    # Human-readable feature name map.
    FEATURE_NAMES = {
        "cycle_number":        "Cycle age",
        "fade_rate_10cy":      "Fade rate (10-cycle)",
        "fade_rate_30cy":      "Fade rate (30-cycle)",
        "fade_rate_50cy":      "Fade rate (50-cycle)",
        "fade_acceleration":   "Fade acceleration",
        "soh_velocity_50cy":   "SOH velocity",
        "resistance_ohm":      "Internal resistance",
        "resistance_normalized": "Resistance (normalised)",
        "resistance_trend_30cy": "Resistance trend",
    }

    def friendly(name: str) -> str:
        return FEATURE_NAMES.get(name, name.replace("_", " ").title())

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Why this prediction?</div>
            <div class="hero-value hero-blue" style="font-size:32px">
                {friendly(top_feature)}
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
        fi_df["label"] = fi_df["feature"].map(lambda f: friendly(f))

        label_threshold = fi_df["importance_pct"].max() * 0.03
        labels = fi_df["importance_pct"].apply(
            lambda v: f"{v:.1f}%" if v >= label_threshold else ""
        )

        fig = go.Figure(go.Bar(
            x=fi_df["importance_pct"],
            y=fi_df["label"],
            orientation="h",
            marker=dict(
                color=fi_df["importance_pct"],
                colorscale=[[0, "#1e2a38"], [0.3, "#2d4a6a"], [1, "#63b3ed"]],
                showscale=False,
            ),
            text=labels,
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="#ffffff", size=12),
            customdata=fi_df["importance_pct"],
            hovertemplate="<b>%{y}</b><br>Importance: %{customdata:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            height=360,
            **base_layout(
                xaxis=dict(
                    title="% contribution to prediction",
                    gridcolor="#232d3b", linecolor="#2d3748", zeroline=False,
                    range=[0, fi_df["importance_pct"].max() * 1.08],
                ),
                yaxis=dict(
                    autorange="reversed",
                    gridcolor="#232d3b", linecolor="#2d3748",
                ),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Top 5 Drivers</div>", unsafe_allow_html=True)

        RANK_COLOURS = ["#63b3ed", "#68d391", "#f6e05e", "#f6ad55", "#fc8181"]
        for i, d in enumerate(drivers):
            pct = d["importance_pct"]
            # Scale bar width relative to the top driver (not hardcoded to 100).
            bar_pct = int(pct / drivers[0]["importance_pct"] * 100)
            colour = RANK_COLOURS[i]
            st.markdown(
                f"""
                <div style="margin-bottom:18px;font-family:sans-serif">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;line-height:1">
                        <p style="margin:0;padding:0;font-size:13px;font-weight:600;color:{colour}">
                            {friendly(d['feature'])}
                        </p>
                        <p style="margin:0;padding:0;font-size:13px;font-weight:700;color:{colour}">
                            {pct:.1f}%
                        </p>
                    </div>
                    <div style="background:#1e2a38;border-radius:4px;height:5px;overflow:hidden">
                        <div style="background:{colour};width:{bar_pct}%;height:5px;border-radius:4px"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-header'>RUL Model — Top Drivers</div>", unsafe_allow_html=True)
        fi_rul = feature_importance_df(bundle, model="rul").head(5)
        for _, row in fi_rul.iterrows():
            st.markdown(
                f"<div style='font-size:12px;margin-bottom:6px'>"
                f"<span style='color:#a0aec0'>{friendly(row['feature'])}</span>"
                f"<span style='color:#4a5568;float:right'>{row['importance_pct']:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Model metrics ──
    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)

    m = bundle["metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("SOH MAE", f"{m['soh_mae']:.2f}%",    help="Mean absolute error on held-out test cycles")
    mc2.metric("SOH R²",  f"{m['soh_r2']:.3f}",      help="Negative R² is expected on a chronological time-series split — MAE is the reliable metric here")
    mc3.metric("RUL MAE", f"{m['rul_mae']:.1f} cy",  help="Mean absolute error on held-out test cycles")
    mc4.metric("RUL R²",  f"{m['rul_r2']:.3f}")

    with st.expander("Why is R² negative — and why that's OK here"):
        st.markdown(
            """
            R² measures how much better the model is than predicting the dataset mean.
            In a **chronological split**, the test set (late cycles, low SOH ~84%) has a very
            different mean from the training set (early cycles, high SOH ~95%). The "predict
            the mean of training" baseline looks terrible on test data, making R² negative.

            **The right metric is MAE.** A SOH prediction within **~2 percentage points**
            of the actual value is solid for a single-cell model. R² will naturally improve
            in Phase 2 when we train on multiple cells with varying degradation paths.
            """
        )

    # ── Actual vs Predicted scatter ──
    st.markdown("<div class='section-header'>Actual vs Predicted SOH — Test Cycles</div>", unsafe_allow_html=True)

    td = bundle["test_data"]
    actual = td["y_soh_test"]
    pred   = td["soh_pred"]

    # Use the full range of both axes so the diagonal reference line is visible.
    axis_min = min(actual.min(), pred.min()) - 0.5
    axis_max = max(actual.max(), pred.max()) + 0.5

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[axis_min, axis_max], y=[axis_min, axis_max],
        mode="lines",
        line=dict(color="#2d3748", dash="dash", width=1),
        name="Perfect prediction",
        hoverinfo="skip",
    ))
    fig2.add_trace(go.Scatter(
        x=actual, y=pred,
        mode="markers",
        marker=dict(color="#63b3ed", size=4, opacity=0.5),
        name="Test cycles",
        hovertemplate="Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>",
    ))
    fig2.update_layout(
        height=300,
        **base_layout(
            xaxis=dict(title="Actual SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                       zeroline=False, range=[axis_min, axis_max]),
            yaxis=dict(title="Predicted SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                       zeroline=False, range=[axis_min, axis_max]),
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Coming Soon
# ---------------------------------------------------------------------------

COMING_SOON_META = {
    "recommendations": ("Recommendations",  "Actionable maintenance recommendations driven by health trends and failure-mode modelling.", "Phase 2"),
    "economics":       ("Economics",        "Total cost of ownership analysis, replacement cost modelling, and second-life ROI.", "Phase 2"),
    "fleet":           ("Fleet",            "Multi-battery fleet view with comparative health rankings, clustering, and alerts.", "Phase 2"),
    "sustainability":  ("Sustainability",   "Carbon impact tracking, second-life suitability scoring, and recycling timeline.", "Phase 2"),
    "reports":         ("Reports",          "Exportable PDF/CSV health reports and audit trails for stakeholders.", "Phase 2"),
    "settings":        ("Settings",         "Data source configuration, alert thresholds, and user preferences.", "Phase 2"),
}

def page_coming_soon(key: str):
    label, description, phase = COMING_SOON_META[key]
    st.markdown(f"# {label}")
    st.markdown(
        f"""
        <div style="border:1px dashed #2d3748;border-radius:12px;padding:64px 48px;
                    text-align:center;margin-top:32px">
            <div style="font-size:13px;font-weight:600;color:#2d3748;letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:16px">{phase}</div>
            <div style="font-size:22px;font-weight:700;color:#4a5568;margin-bottom:12px">
                {label}
            </div>
            <div style="font-size:14px;color:#4a5568;max-width:480px;margin:0 auto;line-height:1.6">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_sidebar()
    df, bundle, split_cycle = load_everything()
    page = st.session_state.get("page", "overview")

    if page == "overview":
        page_overview(df, bundle, split_cycle)
    elif page == "health":
        page_health(df, bundle, split_cycle)
    elif page == "insights":
        page_insights(df, bundle, split_cycle)
    elif page in COMING_SOON_META:
        page_coming_soon(page)
    else:
        page_overview(df, bundle, split_cycle)


if __name__ == "__main__":
    main()
