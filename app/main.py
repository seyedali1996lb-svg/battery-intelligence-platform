"""
Battery Intelligence Platform — Streamlit Dashboard
Phase 1 dashboard: Overview, Health, Insights (functional)

Data:
  - 8 synthetic cells (Cell1-Cell8): physics-informed, injected cell-to-cell
    stress variation (temperature, C-rate, DoD). Not real measured data.
  - 4 NASA cells (B0005-B0018): real LiCoO2 18650 measurements from NASA PCoE
    Battery Aging dataset (Saha & Goebel, 2007, ~2 Ah, 24 C, 2A discharge).

Model trained on all 12 cells combined. NASA loader (src/nasa_loader.py) must
be run once to populate data/raw/ before the app starts.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import build_battery, get_cell_df, CELL_STRESS_PROFILES, _stress_factor
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
    .metric-row  { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
    .metric-chip {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px 20px;
        min-width: 150px;
        flex: 1;
    }
    .metric-chip-label { font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-chip-value { font-size: 26px; font-weight: 700; color: #e2e8f0; margin-top: 4px; }
    .metric-chip-sub   { font-size: 12px; color: #718096; margin-top: 2px; }
    .section-header {
        font-size: 12px; font-weight: 600; color: #4a5568;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin: 28px 0 12px; padding-bottom: 8px;
        border-bottom: 1px solid #2d3748;
    }
    .tag-calibrating {
        display: inline-block; background: rgba(246,224,94,0.12);
        color: #f6e05e; font-size: 11px; font-weight: 600;
        padding: 2px 8px; border-radius: 4px; letter-spacing: 0.06em;
        border: 1px solid rgba(246,224,94,0.25);
    }
    .tag-model {
        display: inline-block; background: rgba(104,211,145,0.12);
        color: #68d391; font-size: 11px; font-weight: 600;
        padding: 2px 8px; border-radius: 4px; letter-spacing: 0.06em;
        border: 1px solid rgba(104,211,145,0.25);
    }
    .block-container { padding-top: 24px !important; }
    h1 { font-size: 22px !important; font-weight: 700 !important; color: #e2e8f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data + model — cached for the session lifetime
# ---------------------------------------------------------------------------

NASA_CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]


def _nasa_cells_available() -> list[str]:
    """Return which NASA cell CSVs are present in data/raw/."""
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    return [
        cid for cid in NASA_CELL_IDS
        if os.path.exists(os.path.join(data_dir, f"{cid}_summary.csv"))
    ]


@st.cache_resource(show_spinner="Loading cells and training model...")
def load_everything():
    """
    Load synthetic cells (Cell1-Cell8) and NASA cells (B0005-B0018) separately.

    Two separate models are trained — one per data source — because the
    synthetic and NASA resistance measurements are on incompatible scales
    (synthetic: 0.15-0.40 ohm internal resistance; NASA: 0.04-0.07 ohm Re
    from EIS impedance spectroscopy). A combined model confuses the features
    and produces negative R2. Separate models keep each dataset honest.

    The dashboard selects the correct bundle based on which cell is chosen.
    """
    def _train_on_cells(battery_dict: dict) -> tuple[dict, dict, dict]:
        all_X, all_y_soh, all_y_rul = [], [], []
        cell_featured = {}
        for cell_id, cell in battery_dict.items():
            df_feat = build_features(cell["cycles"])
            X, y_soh, y_rul = get_model_matrix(df_feat)
            all_X.append(X); all_y_soh.append(y_soh); all_y_rul.append(y_rul)
            cell_featured[cell_id] = (df_feat, X)
        X_all = pd.concat(all_X)
        y_soh_all = pd.concat(all_y_soh)
        y_rul_all = pd.concat(all_y_rul)
        bndl = train_models(X_all, y_soh_all, y_rul_all)
        bndl["metrics"]["n_cells"] = len(battery_dict)
        bndl["metrics"]["n_rows"]  = len(X_all)
        featured_dfs, split_cycles = {}, {}
        for cell_id, (df_feat, X) in cell_featured.items():
            preds = predict(bndl, X)
            df_out = df_feat.loc[X.index].copy()
            df_out["soh_pred"]       = preds["soh_pred"]
            df_out["rul_pred"]       = preds["rul_pred"]
            df_out["confidence_tag"] = preds["confidence_tag"]
            featured_dfs[cell_id] = df_out
            split_idx = int(len(X) * 0.8)
            split_cycles[cell_id] = int(X["cycle_number"].iloc[split_idx])
        return bndl, featured_dfs, split_cycles

    # ── Synthetic cells ──
    synth_ids = list(CELL_STRESS_PROFILES.keys())
    battery_synth = build_battery(battery_id="Oxford_B1", cell_ids=synth_ids)
    bundle_synth, fdfs_synth, sc_synth = _train_on_cells(battery_synth["cells"])

    # ── NASA real cells (if present) ──
    nasa_ids = _nasa_cells_available()
    bundle_nasa, fdfs_nasa, sc_nasa = None, {}, {}
    if nasa_ids:
        battery_nasa = build_battery(battery_id="NASA_B1", cell_ids=nasa_ids)
        bundle_nasa, fdfs_nasa, sc_nasa = _train_on_cells(battery_nasa["cells"])

    # Merge cell outputs; keep bundles separate
    featured_dfs = {**fdfs_synth, **fdfs_nasa}
    split_cycles = {**sc_synth, **sc_nasa}
    bundles = {"synth": bundle_synth, "nasa": bundle_nasa}

    return featured_dfs, bundles, split_cycles


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

NAV_ITEMS = [
    ("Overview",        "overview",        True),
    ("Health",          "health",          True),
    ("Insights",        "insights",        True),
    ("Recommendations", "recommendations", False),
    ("Economics",       "economics",       False),
    ("Fleet",           "fleet",           False),
    ("Sustainability",  "sustainability",  False),
    ("Reports",         "reports",         False),
    ("Settings",        "settings",        False),
]

LEGEND_H = dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#718096"),
)

def render_sidebar(cell_ids: list[str]):
    nasa_available = [c for c in cell_ids if c in NASA_CELL_IDS]
    n_cells = len(cell_ids)
    subtitle = f"{n_cells} cells ({8} synthetic"
    if nasa_available:
        subtitle += f" + {len(nasa_available)} NASA real)"
    else:
        subtitle += ")"

    with st.sidebar:
        st.markdown(
            f"<div style='padding:0 4px 20px'>"
            f"<div style='font-size:16px;font-weight:700;color:#e2e8f0'>⚡ Battery Intel</div>"
            f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>{subtitle} · multi-cell model</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if "page" not in st.session_state:
            st.session_state.page = "overview"

        current = st.session_state.page
        for label, key, enabled in NAV_ITEMS:
            if enabled:
                if st.button(
                    label, key=f"nav_{key}", use_container_width=True,
                    type="primary" if current == key else "secondary",
                ):
                    st.session_state.page = key
                    st.rerun()
            else:
                st.markdown(
                    f"<div style='padding:7px 12px;color:#4a5568;font-size:14px;"
                    f"font-weight:500;display:flex;justify-content:space-between;align-items:center'>"
                    f"<span>{label}</span>"
                    f"<span style='font-size:10px;background:#1a202c;color:#4a5568;"
                    f"padding:1px 7px;border-radius:10px'>Soon</span></div>",
                    unsafe_allow_html=True,
                )

        # ── Cell selector ──
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            "letter-spacing:0.08em;padding:0 4px 8px'>Cell</div>",
            unsafe_allow_html=True,
        )
        selected = st.selectbox(
            "Select cell",
            options=cell_ids,
            index=0,
            key="selected_cell",
            label_visibility="collapsed",
        )

        # Cell annotation — differ for synthetic vs NASA real cells
        if selected in NASA_CELL_IDS:
            st.markdown(
                "<div style='font-size:11px;color:#4a5568;padding:4px 4px 0;line-height:1.7'>"
                "Source: NASA PCoE Battery Aging Dataset<br>"
                "T=24°C &nbsp; C-rate=2A &nbsp; DoD=100%<br>"
                "<span style='color:#68d391'>Real measured data</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            p  = CELL_STRESS_PROFILES.get(selected, {})
            sf = _stress_factor(p.get("temp_mean", 25), p.get("c_rate", 1), p.get("dod", 1))
            st.markdown(
                f"<div style='font-size:11px;color:#4a5568;padding:4px 4px 0;line-height:1.7'>"
                f"T={p.get('temp_mean',25):.0f}°C &nbsp; "
                f"C-rate={p.get('c_rate',1.0):.1f}C &nbsp; "
                f"DoD={p.get('dod',1.0)*100:.0f}%<br>"
                f"Stress: {sf:.2f}× baseline &nbsp; "
                f"<span style='color:#fc8181'>Synthetic</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#2d3748;padding:0 4px;line-height:1.7'>"
            "Phase 1 · scikit-learn GBRT<br>"
            "8 synthetic + 4 NASA real cells<br>"
            "Cell-to-cell stress variation (T, C-rate, DoD)<br>"
            "<span style='color:#fc8181'>⚠ Synthetic cells: not real measured data</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    return selected


# ---------------------------------------------------------------------------
# Chart base layout helper
# ---------------------------------------------------------------------------

def base_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0aec0", size=12),
        margin=dict(l=10, r=10, t=36, b=10),
        hovermode="x unified",
    )
    if "xaxis" not in overrides:
        layout["xaxis"] = dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False)
    if "yaxis" not in overrides:
        layout["yaxis"] = dict(gridcolor="#232d3b", linecolor="#2d3748", zeroline=False)
    layout.update(overrides)
    return layout


def soh_status(soh: float) -> tuple[str, str]:
    if soh >= 90: return "Healthy",    "hero-green"
    if soh >= 80: return "Degrading",  "hero-yellow"
    return "End of Life", "hero-red"


# ---------------------------------------------------------------------------
# Feature label map
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "cycle_number":        "Cycle age",
    "fade_rate_10cy":      "Fade rate (10-cy)",
    "fade_rate_30cy":      "Fade rate (30-cy)",
    "fade_rate_50cy":      "Fade rate (50-cy)",
    "fade_acceleration":   "Fade acceleration",
    "soh_velocity_50cy":   "SOH velocity",
    "resistance_ohm":      "Internal resistance",
    "resistance_normalized": "Resistance (norm.)",
    "resistance_trend_30cy": "Resistance trend",
    "temp_rolling_30cy":   "Temperature (30-cy avg)",
}

def friendly(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame, split_cycle: int, cell_id: str):
    st.markdown("# Overview")

    latest         = df.iloc[-1]
    current_soh    = latest["soh_pct"]
    current_rul    = latest["rul_pred"]
    current_cycle  = int(latest["cycle_number"])
    total_fade     = latest["capacity_fade_ah"]
    confidence     = latest["confidence_tag"]
    status_label, status_colour = soh_status(current_soh)

    conf_html = (
        "<span class='tag-calibrating'>CALIBRATING</span>"
        if confidence == "Calibrating"
        else "<span class='tag-model'>MODEL</span>"
    )

    is_nasa = cell_id in NASA_CELL_IDS
    if is_nasa:
        source_tag = "NASA real · 24°C · 2A discharge"
    else:
        p  = CELL_STRESS_PROFILES.get(cell_id, {})
        sf = _stress_factor(p.get("temp_mean",25), p.get("c_rate",1.0), p.get("dod",1.0))
        source_tag = f"Synthetic · Stress {sf:.2f}x baseline"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Battery Status · {cell_id}</div>
            <div class="hero-value {status_colour}">{status_label}</div>
            <div class="hero-sub">
                SOH: <strong style="color:#e2e8f0">{current_soh:.1f}%</strong>
                &nbsp;·&nbsp; Est. {current_rul:.0f} cycles remaining
                &nbsp;·&nbsp; {source_tag}
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
            <div class="metric-chip">
                <div class="metric-chip-label">Data Source</div>
                <div class="metric-chip-value" style="font-size:18px">{"NASA" if is_nasa else f"{sf:.2f}×"}</div>
                <div class="metric-chip-sub">{"real measured" if is_nasa else "vs baseline (synthetic)"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        name="Model (test)", line=dict(color="#68d391", width=2, dash="dot"), mode="lines",
        hovertemplate="Cycle %{x}: %{y:.1f}%<extra>Model</extra>",
    ))
    fig.add_vline(
        x=split_cycle, line_dash="dot", line_color="#4a5568", line_width=1,
        annotation_text=f"Train → Test (cy {split_cycle})",
        annotation_position="top left",
        annotation_font_color="#4a5568", annotation_font_size=11,
    )
    fig.add_hline(
        y=80, line_dash="dash", line_color="#fc8181", line_width=1,
        annotation_text="EOL (80%)", annotation_position="bottom right",
        annotation_font_color="#fc8181", annotation_font_size=11,
    )
    y_min = max(df["soh_pct"].min() - 2, 60)
    fig.update_layout(
        height=340, legend=LEGEND_H,
        **base_layout(
            xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                       zeroline=False, range=[y_min, 101]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Health
# ---------------------------------------------------------------------------

def page_health(df: pd.DataFrame, split_cycle: int, cell_id: str):
    st.markdown("# Health")

    latest = df.iloc[-1]
    current_soh = latest["soh_pct"]
    status_label, status_colour = soh_status(current_soh)

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Health Assessment · {cell_id}</div>
            <div class="hero-value {status_colour}">{current_soh:.1f}% SOH</div>
            <div class="hero-sub">
                Fading at ~{latest['fade_rate_50cy']*1000:.2f} mAh/cycle (50-cycle avg)
                &nbsp;·&nbsp; Internal resistance: {latest.get('resistance_ohm',0):.3f} Ω
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
            fill="tozeroy", fillcolor="rgba(252,129,129,0.08)",
            line=dict(color="#fc8181", width=2),
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
            hovertemplate="Cycle %{x}: %{y:.1f} mΩ<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=df["cycle_number"],
            y=df["resistance_ohm"].rolling(30, min_periods=1).mean() * 1000,
            name="30-cycle avg", line=dict(color="#f6ad55", width=2),
            hovertemplate="Cycle %{x}: %{y:.1f} mΩ<extra>30-cy avg</extra>",
        ))
        fig2.update_layout(
            height=280, legend=LEGEND_H,
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
        line=dict(color="#4a5568", width=1), name="10-cycle window",
    ))
    fig3.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["fade_rate_30cy"] * 1000,
        line=dict(color="#63b3ed", width=2), name="30-cycle window",
    ))
    fig3.add_trace(go.Scatter(
        x=df["cycle_number"], y=df["fade_rate_50cy"] * 1000,
        line=dict(color="#68d391", width=2), name="50-cycle window",
    ))
    fig3.update_layout(
        height=260, legend=LEGEND_H,
        **base_layout(
            xaxis=dict(title="Cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="mAh lost per cycle", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
        ),
    )
    st.plotly_chart(fig3, use_container_width=True)

    early = df[df["confidence_tag"] == "Calibrating"]
    if len(early) > 0:
        st.info(
            f"**Calibrating** — the first {len(early)} cycles show higher variability "
            f"while rolling-window features stabilise. Readings from cycle 50 onward are reliable."
        )


# ---------------------------------------------------------------------------
# Page: Insights
# ---------------------------------------------------------------------------

def page_insights(df: pd.DataFrame, bundle: dict, cell_id: str):
    st.markdown("# Insights")

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
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Why this prediction? · SOH model · {model_label}</div>
            <div class="hero-value hero-blue" style="font-size:32px">
                {friendly(top_feature)}
            </div>
            <div class="hero-sub">
                Explains <strong style="color:#e2e8f0">{top_pct:.0f}%</strong>
                of the model's SOH prediction across all cells.
                Internal resistance is a direct proxy for SEI layer growth —
                the dominant degradation mechanism in LiCoO&#x2082; cells.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-header'>Feature Importance — SOH Model</div>", unsafe_allow_html=True)

        fi_df = feature_importance_df(bundle, model="soh")
        fi_df["label"] = fi_df["feature"].map(friendly)

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
                    range=[0, fi_df["importance_pct"].max() * 1.1],
                ),
                yaxis=dict(autorange="reversed", gridcolor="#232d3b", linecolor="#2d3748"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Top 5 Drivers — SOH</div>", unsafe_allow_html=True)

        RANK_COLOURS = ["#63b3ed", "#68d391", "#f6e05e", "#f6ad55", "#fc8181"]
        for i, d in enumerate(drivers):
            pct      = d["importance_pct"]
            bar_pct  = int(pct / drivers[0]["importance_pct"] * 100)
            colour   = RANK_COLOURS[i]
            st.markdown(
                f"""
                <div style="margin-bottom:18px;font-family:sans-serif">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                        <p style="margin:0;font-size:13px;font-weight:600;color:{colour}">
                            {friendly(d['feature'])}
                        </p>
                        <p style="margin:0;font-size:13px;font-weight:700;color:{colour}">
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

        st.markdown("<div class='section-header'>Top 5 Drivers — RUL</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section-header'>Model Performance — Multi-cell Training</div>", unsafe_allow_html=True)

    m = bundle["metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("SOH MAE",  f"{m['soh_mae']:.2f}%",   help="Mean absolute error on held-out test cycles (chronological split)")
    mc2.metric("SOH R²",   f"{m['soh_r2']:.3f}",     help="Fit quality — multi-cell training lets the model generalise across stress conditions and cell chemistry")
    mc3.metric("RUL MAE",  f"{m['rul_mae']:.0f} cy", help="RUL is harder: exact remaining life depends on future usage, which is unknown")
    n_cells = bundle["metrics"].get("n_cells", "—")
    n_rows  = bundle["metrics"].get("n_rows", 0)
    mc4.metric("Training", f"{n_cells} cells / {n_rows:,} cycles")

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


# ---------------------------------------------------------------------------
# Page: Coming Soon
# ---------------------------------------------------------------------------

COMING_SOON_META = {
    "recommendations": ("Recommendations", "Actionable maintenance recommendations driven by health trends and failure-mode modelling.", "Phase 2"),
    "economics":       ("Economics",       "Total cost of ownership analysis, replacement cost modelling, and second-life ROI.", "Phase 2"),
    "fleet":           ("Fleet",           "Multi-battery fleet view with comparative health rankings, clustering, and alerts.\n\nGate: requires real Oxford/NASA data loaded and validated first.", "Phase 2"),
    "sustainability":  ("Sustainability",  "Carbon impact tracking, second-life suitability scoring, and recycling timeline.", "Phase 2"),
    "reports":         ("Reports",         "Exportable PDF/CSV health reports and audit trails for stakeholders.", "Phase 2"),
    "settings":        ("Settings",        "Data source configuration, alert thresholds, and user preferences.", "Phase 2"),
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
            <div style="font-size:22px;font-weight:700;color:#4a5568;margin-bottom:16px">
                {label}
            </div>
            <div style="font-size:14px;color:#4a5568;max-width:520px;margin:0 auto;line-height:1.7;white-space:pre-line">
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
    featured_dfs, bundles, split_cycles = load_everything()

    cell_ids     = list(featured_dfs.keys())
    selected     = render_sidebar(cell_ids)

    df           = featured_dfs[selected]
    split_cycle  = split_cycles[selected]
    bundle       = bundles["nasa"] if selected in NASA_CELL_IDS else bundles["synth"]
    page         = st.session_state.get("page", "overview")

    if page == "overview":
        page_overview(df, split_cycle, selected)
    elif page == "health":
        page_health(df, split_cycle, selected)
    elif page == "insights":
        page_insights(df, bundle, selected)
    elif page in COMING_SOON_META:
        page_coming_soon(page)
    else:
        page_overview(df, split_cycle, selected)


if __name__ == "__main__":
    main()
