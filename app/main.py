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
from lco_eval import run_lco


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

        # LCO validation: only honest generalisation metric for this data structure.
        # Per-cell fold R² is stored individually — the dataset-average can pass
        # while a specific cell's fold fails. RUL display is gated per cell.
        from lco_eval import RUL_RELIABLE_FLOOR
        cell_cycles = {cid: cell["cycles"] for cid, cell in battery_dict.items()}
        lco = run_lco(cell_cycles)
        bndl["metrics"]["lco_soh_r2"]   = lco["soh_r2"]
        bndl["metrics"]["lco_rul_r2"]   = lco["rul_r2"]
        bndl["metrics"]["rul_reliable"] = lco["rul_reliable"]   # dataset average (for Insights display)
        bndl["metrics"]["lco_per_cell"] = lco["per_cell"]

        # Per-cell reliability: each cell checked against the floor independently.
        # A cell whose held-out fold R² is below the floor gets rul_reliable=False
        # even if the dataset-average passes.
        per_cell_rul_ok = {
            cid: (fold["rul_r2"] >= RUL_RELIABLE_FLOOR)
            for cid, fold in lco["per_cell"].items()
        }
        # Cells with only 1 total cell (no LCO possible) inherit the dataset flag.
        bndl["metrics"]["per_cell_rul_reliable"] = per_cell_rul_ok

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
    ("Copilot",         "copilot",         True),
    ("Consequences",    "consequences",    True),
    ("Economics",       "economics",       False),
    ("Fleet",           "fleet",           True),
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

def page_overview(df: pd.DataFrame, split_cycle: int, cell_id: str, rul_reliable: bool = True):
    st.markdown("# Overview")

    latest         = df.iloc[-1]
    current_soh    = latest["soh_pct"]
    current_rul    = latest["rul_pred"]
    current_cycle  = int(latest["cycle_number"])
    total_fade     = latest["capacity_fade_ah"]
    confidence     = latest["confidence_tag"]
    status_label, status_colour = soh_status(current_soh)

    # RUL display: suppress when model doesn't generalise (LCO R² < floor)
    # or when early-cycle features haven't stabilised yet.
    rul_calibrating = (not rul_reliable) or (confidence == "Calibrating")
    rul_display     = "—" if rul_calibrating else f"{current_rul:.0f}"
    rul_sub         = "not calibrated" if not rul_reliable else "cycles to 80% SOH"

    conf_html = (
        "<span class='tag-calibrating'>CALIBRATING</span>"
        if rul_calibrating
        else "<span class='tag-model'>MODEL</span>"
    )
    rul_hero = "Not calibrated" if not rul_reliable else f"Est. {current_rul:.0f} cycles remaining"

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
                &nbsp;·&nbsp; {rul_hero}
                &nbsp;·&nbsp; {source_tag}
                &nbsp;·&nbsp; {conf_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_nasa_src = cell_id in NASA_CELL_IDS
    src_val = "NASA" if is_nasa_src else f"{sf:.2f}x"
    src_sub = "real measured" if is_nasa_src else "vs baseline (synthetic)"

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
                <div class="metric-chip-value">{rul_display}</div>
                <div class="metric-chip-sub">{rul_sub}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Capacity Lost</div>
                <div class="metric-chip-value">{total_fade*1000:.0f} mAh</div>
                <div class="metric-chip-sub">since commissioning</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Data Source</div>
                <div class="metric-chip-value" style="font-size:18px">{src_val}</div>
                <div class="metric-chip-sub">{src_sub}</div>
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
        **base_layout(
            height=340, legend=LEGEND_H,
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
            **base_layout(
                height=280, legend=LEGEND_H,
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
        **base_layout(
            height=260, legend=LEGEND_H,
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
    rul_label = f"{lco_rul_r2:.3f} R²" if rul_ok else f"{lco_rul_r2:.3f} R² — not calibrated"
    mc3.metric(
        "RUL (LCO)",
        rul_label,
        help="Leave-cell-out R² for RUL. Below 0.30 = model shown as 'Not calibrated' in UI.",
    )
    n_cells = m.get("n_cells", "—")
    n_rows  = m.get("n_rows", 0)
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

# ---------------------------------------------------------------------------
# Page: Fleet
# ---------------------------------------------------------------------------

def page_fleet(featured_dfs: dict, bundles: dict):
    st.markdown("# Fleet")

    # ── Build fleet summary row per cell ──
    rows = []
    # Per-cell reliability lookup — must check the individual fold R², not the group average.
    synth_per_cell = bundles["synth"]["metrics"].get("per_cell_rul_reliable", {})
    nasa_per_cell  = (bundles["nasa"]["metrics"].get("per_cell_rul_reliable", {})
                      if bundles["nasa"] else {})

    for cell_id, df in featured_dfs.items():
        is_nasa   = cell_id in NASA_CELL_IDS
        per_cell  = nasa_per_cell if is_nasa else synth_per_cell
        # Fall back to dataset-level flag only when this cell had no LCO fold (shouldn't happen).
        rul_ok    = per_cell.get(
            cell_id,
            bundles["nasa"]["metrics"].get("rul_reliable", False) if is_nasa
            else bundles["synth"]["metrics"].get("rul_reliable", False),
        )
        latest    = df.iloc[-1]
        soh       = latest["soh_pct"]
        cycle     = int(latest["cycle_number"])
        fade_30   = latest.get("fade_rate_30cy", float("nan")) * 1000  # mSOH/cy
        rul       = latest["rul_pred"] if rul_ok else None
        eol_row   = df[df["is_eol"]]
        eol_at    = int(eol_row["cycle_number"].iloc[0]) if len(eol_row) else None
        cycles_to_eol = max(0, eol_at - cycle) if eol_at else None

        status_label, _ = soh_status(soh)
        rows.append({
            "cell_id":      cell_id,
            "source":       "NASA" if is_nasa else "Synthetic",
            "soh":          soh,
            "status":       status_label,
            "cycle":        cycle,
            "fade_30":      fade_30,
            "rul":          rul,
            "rul_ok":       rul_ok,
            "eol_at":       eol_at,
            "cycles_to_eol": cycles_to_eol,
        })

    # Sort: worst SOH first (most urgent)
    rows.sort(key=lambda r: r["soh"])

    # ── Header metrics ──
    n_eol       = sum(1 for r in rows if r["status"] == "End of Life")
    n_degrading = sum(1 for r in rows if r["status"] == "Degrading")
    n_healthy   = sum(1 for r in rows if r["status"] == "Healthy")
    worst_soh   = rows[0]["soh"]
    best_soh    = rows[-1]["soh"]

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-label">Total Cells</div>
                <div class="metric-chip-value">{len(rows)}</div>
                <div class="metric-chip-sub">8 synthetic · {sum(1 for r in rows if r['source']=='NASA')} NASA real</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">End of Life</div>
                <div class="metric-chip-value" style="color:#fc8181">{n_eol}</div>
                <div class="metric-chip-sub">below 80% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Degrading</div>
                <div class="metric-chip-value" style="color:#f6e05e">{n_degrading}</div>
                <div class="metric-chip-sub">80–90% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Healthy</div>
                <div class="metric-chip-value" style="color:#68d391">{n_healthy}</div>
                <div class="metric-chip-sub">above 90% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Fleet SOH Range</div>
                <div class="metric-chip-value" style="font-size:20px">{worst_soh:.0f}–{best_soh:.0f}%</div>
                <div class="metric-chip-sub">worst to best</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Ranking table ──
    st.markdown("<div class='section-header'>Health Ranking — Worst First</div>", unsafe_allow_html=True)

    STATUS_COLOUR = {"Healthy": "#68d391", "Degrading": "#f6e05e", "End of Life": "#fc8181"}
    SOURCE_STYLE  = {
        "NASA":      "background:rgba(104,211,145,0.12);color:#68d391;border:1px solid rgba(104,211,145,0.25)",
        "Synthetic": "background:rgba(74,85,104,0.3);color:#718096;border:1px solid #2d3748",
    }

    table_rows_html = ""
    for rank, r in enumerate(rows, 1):
        sc = STATUS_COLOUR[r["status"]]
        ss = SOURCE_STYLE[r["source"]]
        soh_bar = int(max(0, min(100, r["soh"])))
        bar_colour = sc

        rul_cell = (
            f"{r['rul']:.0f} cy" if (r["rul"] is not None and r["rul_ok"])
            else "<span style='color:#4a5568'>—</span>"
        )
        eol_cell = (
            f"<span style='color:#fc8181'>Reached at {r['eol_at']}</span>"
            if r["eol_at"] and r["cycles_to_eol"] == 0
            else (f"{r['cycles_to_eol']} cy" if r["cycles_to_eol"] is not None else "—")
        )

        table_rows_html += f"""
        <tr style="border-bottom:1px solid #1a202c">
            <td style="padding:14px 12px;color:#4a5568;font-size:13px">{rank}</td>
            <td style="padding:14px 12px">
                <span style="font-weight:600;color:#e2e8f0;font-size:14px">{r['cell_id']}</span>
                <span style="margin-left:8px;font-size:10px;padding:2px 6px;border-radius:8px;{ss}">{r['source']}</span>
            </td>
            <td style="padding:14px 12px">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="color:{sc};font-weight:700;font-size:15px;min-width:46px">{r['soh']:.1f}%</span>
                    <div style="flex:1;background:#1a202c;border-radius:3px;height:6px;min-width:80px">
                        <div style="background:{bar_colour};width:{soh_bar}%;height:6px;border-radius:3px"></div>
                    </div>
                </div>
            </td>
            <td style="padding:14px 12px">
                <span style="font-size:12px;font-weight:600;padding:2px 8px;border-radius:4px;
                             background:{sc}22;color:{sc};border:1px solid {sc}44">{r['status']}</span>
            </td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{r['cycle']:,}</td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{r['fade_30']:.2f} mSOH/cy</td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{rul_cell}</td>
            <td style="padding:14px 12px;font-size:13px">{eol_cell}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-family:sans-serif">
            <thead>
                <tr style="border-bottom:2px solid #2d3748">
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">#</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Cell</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">SOH</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Status</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Cycles</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Fade Rate</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Est. RUL</th>
                    <th style="padding:10px 12px;text-align:left;font-size:11px;color:#4a5568;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">EOL Proximity</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # ── SOH distribution chart ──
    st.markdown("<div class='section-header'>SOH Distribution Across Fleet</div>", unsafe_allow_html=True)

    sorted_ids  = [r["cell_id"] for r in rows]
    sorted_sohs = [r["soh"] for r in rows]
    bar_colours = [STATUS_COLOUR[r["status"]] for r in rows]

    fig = go.Figure(go.Bar(
        x=sorted_ids, y=sorted_sohs,
        marker_color=bar_colours,
        hovertemplate="<b>%{x}</b><br>SOH: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="#fc8181", line_width=1,
                  annotation_text="EOL (80%)", annotation_position="top right",
                  annotation_font_color="#fc8181", annotation_font_size=11)
    fig.add_hline(y=90, line_dash="dot", line_color="#f6e05e", line_width=1,
                  annotation_text="Degrading (90%)", annotation_position="top right",
                  annotation_font_color="#f6e05e", annotation_font_size=11)
    fig.update_layout(
        height=280,
        **base_layout(
            xaxis=dict(title="Cell", gridcolor="#232d3b", linecolor="#2d3748", zeroline=False),
            yaxis=dict(title="SOH %", gridcolor="#232d3b", linecolor="#2d3748",
                       zeroline=False, range=[50, 102]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Second-life screening ──
    st.markdown("<div class='section-header'>Second-Life Readiness Screening</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#718096;margin-bottom:20px;line-height:1.6'>"
        "Conventional second-life assessment window: <strong style='color:#e2e8f0'>SOH 70–85%</strong>. "
        "Cells above 85% are still in primary life. Below 70% is below most application floors. "
        "Click a cell in the sidebar to open the Consequences page for detailed economics.</div>",
        unsafe_allow_html=True,
    )

    SL_BUCKETS = {
        "primary":    ("Primary Life",          "SOH > 85%",    "#4a5568", "#1a202c"),
        "candidate":  ("Second-Life Candidate", "SOH 70–85%",   "#68d391", "#1a2e22"),
        "below_floor":("Below Floor",           "SOH < 70%",    "#fc8181", "#2d0f0f"),
    }

    def _sl_bucket(r_soh):
        if r_soh > 85.0:  return "primary"
        if r_soh >= 70.0: return "candidate"
        return "below_floor"

    bucketed = {"primary": [], "candidate": [], "below_floor": []}
    for r in rows:
        bucketed[_sl_bucket(r["soh"])].append(r)

    sl_cols = st.columns(3)
    for col, (bkey, (blabel, brange, bfg, bbg)) in zip(sl_cols, SL_BUCKETS.items()):
        cells_in_bucket = bucketed[bkey]
        count = len(cells_in_bucket)
        pills = "".join(
            f"<div style='display:inline-block;margin:4px;padding:5px 12px;"
            f"background:{bfg}22;border:1px solid {bfg}44;border-radius:20px;"
            f"font-size:12px;font-weight:600;color:{bfg}'>"
            f"{r['cell_id']} <span style='font-weight:400;color:{bfg}88'>{r['soh']:.0f}%</span>"
            f"</div>"
            for r in cells_in_bucket
        ) if cells_in_bucket else (
            f"<div style='font-size:12px;color:#4a5568;font-style:italic;padding:8px 0'>None</div>"
        )
        with col:
            st.markdown(
                f"""
                <div style="background:{bbg};border:1px solid {bfg}33;border-radius:10px;padding:18px;min-height:120px">
                    <div style="font-size:10px;font-weight:700;color:{bfg};text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:4px">{blabel}</div>
                    <div style="font-size:12px;color:{bfg}88;margin-bottom:12px">{brange} · {count} cell{'s' if count != 1 else ''}</div>
                    <div style="line-height:2">{pills}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Honest copy on cross-type RUL ──
    st.markdown("<div class='section-header'>About This Ranking</div>", unsafe_allow_html=True)
    st.markdown(
        """
        **Ranking method:** Cells are ranked by current SOH %, fade rate (30-cycle rolling average),
        and EOL proximity. SOH % is directly comparable across all cells — it's always relative
        to each cell's own initial capacity, so 75% SOH means the same thing for a NASA cell
        as for a synthetic cell.

        **Why RUL is not ranked across cell types:**
        Remaining Useful Life (cycles) is predicted by separate models for synthetic vs NASA cells,
        because the resistance features used for prediction are on incompatible measurement scales
        (synthesised internal resistance vs EIS electrolyte resistance from impedance spectroscopy).
        Comparing "300 cycles remaining" from one model to "40 cycles remaining" from another
        would be a meaningless number — the models aren't measuring the same thing.

        Individual RUL estimates are shown per cell on the Overview and Health pages (drill in
        via the cell selector in the sidebar).
        """,
    )

    with st.expander("Roadmap: Unified Ranking — what would unlock cross-type RUL comparison?"):
        st.markdown(
            """
            **Gate: 8+ real cells with diverse usage histories.**

            The current 4 NASA cells were all tested at identical lab conditions (24°C, 2A discharge).
            That means the only variation between them is cell-to-cell manufacturing spread —
            not the operating temperature, C-rate, and DoD variation that would make a combined
            resistance signal meaningful.

            Once 8 or more real cells are available with varied operating conditions, two changes
            become worthwhile:

            1. **Replace `resistance_ohm` with `resistance_normalized`** (ratio to each cell's
               own initial resistance) as the only resistance feature. Both synthetic and real
               cells start at 1.0 and rise — this is comparable across measurement methods.

            2. **Train one unified model** on the combined dataset. Validate with leave-cell-out
               to confirm it generalises across both real and synthetic cells.

            Until then, ranking by SOH and fade rate is the honest choice.
            """
        )


# ---------------------------------------------------------------------------
# Page: Copilot
# ---------------------------------------------------------------------------

def page_copilot(
    cell_ids: list[str],
    featured_dfs: dict,
    bundles: dict,
    selected: str,
):
    from copilot import (
        build_cell_context,
        build_fleet_stats,
        context_summary,
        answer_health,
        answer_prediction_drivers,
        answer_rul,
        answer_compare,
        answer_anomaly,
        answer_recent_trajectory,
        answer_fleet_compare,
        answer_alerts,
        QUERY_LABELS,
        FOLLOW_UP_MAP,
    )

    st.markdown("# Copilot")

    # ── Disclosure banner ──
    st.markdown(
        """
        <div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.18);
                    border-radius:10px;padding:14px 20px;margin-bottom:24px;
                    font-size:13px;color:#718096;line-height:1.6">
            <strong style="color:#63b3ed">Grounded narration only.</strong>
            Every sentence is derived from values already computed by the model pipeline —
            SOH, feature importances, per-cell RUL reliability, fade rates.
            The Copilot never calculates, estimates, or infers a value not already in the bundle.
            If a number is not there, it says so.
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.session_state.get("copilot_query", None)

    def _qbtn(key: str, col):
        with col:
            if st.button(
                QUERY_LABELS[key], key=f"cpq_{key}", use_container_width=True,
                type="primary" if query == key else "secondary",
            ):
                st.session_state.copilot_query = key
                st.rerun()

    # ── Row 1: cell-level queries ──
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    _qbtn("health",  r1c1)
    _qbtn("drivers", r1c2)
    _qbtn("rul",     r1c3)
    _qbtn("compare", r1c4)

    # ── Row 2: deeper queries + fleet ──
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    _qbtn("recent",        r2c1)
    _qbtn("anomaly",       r2c2)
    _qbtn("fleet_compare", r2c3)
    _qbtn("alerts",        r2c4)

    if not query:
        st.markdown(
            "<div style='text-align:center;padding:56px 24px;color:#4a5568;font-size:14px'>"
            "Select a question above — the Copilot will explain using only the numbers "
            "already in the model bundle for <strong style='color:#718096'>"
            + selected + "</strong>.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Pre-compute fleet stats (cheap: just iterates already-computed DataFrames) ──
    fleet_stats = build_fleet_stats(featured_dfs, bundles)

    # ── Build context for the selected cell (not needed for fleet-only queries) ──
    fleet_only = (query == "alerts")
    ctx        = None if fleet_only else build_cell_context(selected, featured_dfs, bundles)
    contexts   = []

    # ── Second cell selector for compare ──
    compare_with = None
    if query == "compare":
        other_ids = [c for c in cell_ids if c != selected]
        if not other_ids:
            st.warning("At least two cells are needed for comparison.")
            return
        compare_with = st.selectbox(
            "Compare with:", options=other_ids, key="copilot_compare_cell",
        )

    # ── Generate response ──
    if query == "health":
        response = answer_health(ctx)
        contexts = [ctx]
    elif query == "drivers":
        response = answer_prediction_drivers(ctx)
        contexts = [ctx]
    elif query == "rul":
        response = answer_rul(ctx)
        contexts = [ctx]
    elif query == "compare":
        ctx_b    = build_cell_context(compare_with, featured_dfs, bundles)
        response = answer_compare(ctx, ctx_b)
        contexts = [ctx, ctx_b]
    elif query == "recent":
        response = answer_recent_trajectory(ctx, featured_dfs[selected])
        contexts = [ctx]
    elif query == "anomaly":
        response = answer_anomaly(ctx, fleet_stats)
        contexts = [ctx]
    elif query == "fleet_compare":
        response = answer_fleet_compare(ctx, fleet_stats)
        contexts = [ctx]
    elif query == "alerts":
        response = answer_alerts(fleet_stats)
        contexts = []
    else:
        response = f"Unknown query: {query}"
        contexts = []

    # ── Response header ──
    cell_label = f" &nbsp;·&nbsp; {selected}" if not fleet_only else " &nbsp;·&nbsp; all cells"
    st.markdown(
        f"<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:28px 0 16px;padding-bottom:8px;border-bottom:1px solid #2d3748'>"
        f"{QUERY_LABELS.get(query, '')}{cell_label}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(response)

    # ── Copy / export button ──
    st.download_button(
        label="Export as text",
        data=response,
        file_name=f"copilot_{query}_{selected}.txt",
        mime="text/plain",
        key="copilot_export",
    )

    # ── Follow-up suggestions ──
    follow_ups = FOLLOW_UP_MAP.get(query, [])
    if follow_ups:
        st.markdown(
            "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            "letter-spacing:0.08em;margin:32px 0 10px'>Ask next</div>",
            unsafe_allow_html=True,
        )
        fu_cols = st.columns(len(follow_ups))
        for fu_key, col in zip(follow_ups, fu_cols):
            with col:
                if st.button(
                    QUERY_LABELS[fu_key],
                    key=f"fu_{fu_key}_{query}",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state.copilot_query = fu_key
                    st.rerun()

    # ── Transparency footer ──
    if contexts:
        with st.expander("Context used — what data drove this response", expanded=False):
            for c in contexts:
                st.markdown(f"**{c['cell_id']}**")
                st.code(context_summary(c), language=None)
    elif fleet_only:
        with st.expander("Context used — fleet aggregates", expanded=False):
            lines = [
                f"Cells monitored: {fleet_stats['n_cells']}",
                f"SOH range:       {fleet_stats['soh_min']:.1f}% – {fleet_stats['soh_max']:.1f}%",
                f"SOH median:      {fleet_stats['soh_median']:.1f}%",
                f"EOL cells:       {', '.join(fleet_stats['eol_cells']) or 'none'}",
                f"Degrading cells: {', '.join(fleet_stats['degrading_cells']) or 'none'}",
                f"Uncalibrated RUL: {', '.join(fleet_stats['unreliable_rul']) or 'none'}",
            ]
            st.code("\n".join(lines), language=None)


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
    from consequences import (
        ASSUMPTIONS, SECOND_LIFE_APPS, CELL_NOMINAL_KWH,
        application_fit, financial_comparison, sustainability_snapshot, breakeven_curve,
    )

    # ── Pull validated model outputs ──
    latest           = df.iloc[-1]
    soh              = float(latest["soh_pct"])
    fade_30          = float(latest.get("fade_30_mah_cy", 0.0))
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
    st.markdown("# Consequences")
    st.markdown(f"##### Second-Life Economics + Sustainability · {selected}")

    # ── Assumption transparency banner ──
    def _badge(label: str, colour: str = "#b7791f") -> str:
        return (
            f"<span style='background:{colour}22;border:1px solid {colour}55;"
            f"color:{colour};font-size:10px;font-weight:700;padding:1px 7px;"
            f"border-radius:10px;letter-spacing:0.06em'>{label}</span>"
        )

    BADGE_ESTIMATE  = _badge("Estimate", "#b7791f")
    BADGE_ILLUST    = _badge("Illustrative — not sourced", "#718096")
    BADGE_VALIDATED = _badge("Validated model output", "#2f855a")

    st.markdown(
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
        """,
        unsafe_allow_html=True,
    )

    # ── Primary life gate ──
    if soh > 85.0:
        st.markdown(
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
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Validated inputs row (makes the banner concrete) ──
    rul_display = (
        f"{rul_pred:.0f} cy" if rul_pred is not None
        else "not calibrated"
    )
    rul_colour  = "#718096" if rul_pred is None else "#e2e8f0"
    st.markdown(
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
        """,
        unsafe_allow_html=True,
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
            st.markdown(
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
                """,
                unsafe_allow_html=True,
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
            badge_html   = _badge(badge_label, "#b7791f" if "Cited" in badge_label else "#718096")
            repack_note  = (
                f"<div style='font-size:11px;color:#718096;margin-top:6px'>"
                f"after −${repack_cost:.0f}/cell repack &nbsp;"
                f"{_badge(a['repack_cost']['label'], '#718096')}</div>"
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
                st.markdown(
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
                    """,
                    unsafe_allow_html=True,
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
        co2_badge   = _badge(ASSUMPTIONS["co2_manufacture"]["label"], "#b7791f")
        mat_badge   = _badge(ASSUMPTIONS["material_recovery"]["label"], "#b7791f")

        with s1:
            st.markdown(
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
                        &nbsp;{_badge("Cited estimate", "#b7791f")}&nbsp;
                        (≈15% cathode-material credit, Dunn et al. 2015 — hardcoded, no slider).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
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
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── Assumption register ──
    with st.expander("All assumptions — sources and labels", expanded=False):
        for key, asmp in ASSUMPTIONS.items():
            badge_colour = "#b7791f" if "Cited" in asmp["label"] else "#718096"
            badge_html   = _badge(asmp["label"], badge_colour)
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


COMING_SOON_META = {
    "recommendations": ("Recommendations", "Actionable maintenance recommendations driven by health trends and failure-mode modelling.", "Phase 2"),
    "economics":       ("Economics",       "Total cost of ownership analysis, replacement cost modelling, and second-life ROI.", "Phase 2"),
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

    # Per-cell reliability: use the specific fold R² for this cell, not the group average.
    per_cell_ok  = bundle["metrics"].get("per_cell_rul_reliable", {})
    rul_reliable = per_cell_ok.get(selected, bundle["metrics"].get("rul_reliable", True))

    if page == "overview":
        page_overview(df, split_cycle, selected, rul_reliable=rul_reliable)
    elif page == "health":
        page_health(df, split_cycle, selected)
    elif page == "insights":
        page_insights(df, bundle, selected)
    elif page == "copilot":
        page_copilot(cell_ids, featured_dfs, bundles, selected)
    elif page == "consequences":
        page_consequences(selected, df, featured_dfs, bundles, rul_reliable)
    elif page == "fleet":
        page_fleet(featured_dfs, bundles)
    elif page in COMING_SOON_META:
        page_coming_soon(page)
    else:
        page_overview(df, split_cycle, selected)


if __name__ == "__main__":
    main()
