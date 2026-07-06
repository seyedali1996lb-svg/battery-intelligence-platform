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
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import build_battery, get_cell_df, CELL_STRESS_PROFILES, _stress_factor
from features import build_features, get_model_matrix
from model import train_models, predict, feature_importance_df, top_drivers
from lco_eval import run_lco, RUL_RELIABLE_FLOOR
from design_system import (
    make_badge, make_state_badge, section_header_html,
    BADGE_VALIDATED, BADGE_ESTIMATE, BADGE_ILLUST, BADGE_UNAVAIL,
    BADGE_MEASURED, BADGE_SIMULATED, BADGE_SYNTHETIC,
    BADGE_CALIBRATING, C_CALIBRATING,
    provenance_banner,
    ACTION_META, CONF_META,
)
from trajectory_memory import TrajectoryMemory
from utils import (
    NASA_CELL_IDS, LEGEND_H, PLOTLY_CONFIG, FEATURE_LABELS,
    _md_html, _empty_state, _action_bar,
    _cell_provenance, _analysis_provenance,
    base_layout, soh_status, friendly, _soh_sparkline_svg,
    render_pack_builder, _resample_df,
)
from import_adapter import adapt_upload_to_pipeline
from bundle_cache import (load_cached, save_cached, clear_cache as clear_bundle_cache,
                          load_features_cached, save_features_cached)
from knee_detection import detect_knee, degradation_phases


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
    .hero-label  { font-size: 12px; color: #8896a8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
    .hero-value  { font-size: 72px; font-weight: 700; line-height: 1.0; margin-bottom: 8px; }
    .hero-sub    { font-size: 14px; color: #a0aec0; }
    .hero-green  { color: #48bb78; }
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
    .metric-chip-label { font-size: 11px; color: #8896a8; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-chip-value { font-size: 26px; font-weight: 700; color: #e2e8f0; margin-top: 4px; }
    .metric-chip-sub   { font-size: 12px; color: #8896a8; margin-top: 2px; }
    div[data-testid="stMetricValue"] {
        font-size: 18px;
        white-space: normal;
        overflow-wrap: break-word;
        line-height: 1.3;
    }
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
        color: #48bb78; font-size: 11px; font-weight: 600;
        padding: 2px 8px; border-radius: 4px; letter-spacing: 0.06em;
        border: 1px solid rgba(104,211,145,0.25);
    }
    /* ── Spacing tokens (4-step scale) ── */
    :root {
        --sp-1: 4px;  --sp-2: 8px;  --sp-3: 16px;  --sp-4: 24px;
        --r-chip: 4px;  --r-card: 8px;  --r-section: 12px;
        --c-border: #2d3748;  --c-surface: #1e2a38;  --c-muted: #8896a8;
    }
    /* ── Type ramp (C1) ── */
    .block-container { padding-top: 56px !important; }
    h1 { font-size: 28px !important; font-weight: 800 !important; color: #e2e8f0 !important; margin-bottom: 4px !important; }
    h2 { font-size: 22px !important; font-weight: 700 !important; color: #e2e8f0 !important; margin-bottom: 2px !important; }
    h3 { font-size: 16px !important; font-weight: 600 !important; color: #cbd5e0 !important; }
    h4 { font-size: 13px !important; font-weight: 600 !important; color: #a0aec0 !important; }
    /* section-header sits clearly below h3 in the hierarchy */
    .section-header {
        font-size: 11px !important; font-weight: 700 !important; color: #4a5568 !important;
        text-transform: uppercase !important; letter-spacing: 0.12em !important;
        margin: 28px 0 12px !important; padding-bottom: 6px !important;
        border-bottom: 1px solid #2d3748 !important;
    }
    /* ── Accessibility: contrast fix — #8896a8 on #1a202c = 5.2:1 (WCAG AA pass) ── */
    .metric-chip-sub, .hero-sub { color: #8896a8 !important; }
    /* ── Accessibility: focus styles — WCAG 2.4.7 (L3) ── */
    button:focus-visible, [role="button"]:focus-visible,
    input:focus-visible, select:focus-visible, a:focus-visible,
    [data-testid="stExpander"]:focus-visible {
        outline: 2px solid #63b3ed !important;
        outline-offset: 3px !important;
        border-radius: 4px !important;
    }
    section[data-testid="stSidebar"] button:focus-visible {
        outline: 2px solid #63b3ed !important;
        outline-offset: 2px !important;
        border-radius: 6px !important;
    }
    /* ── Accessibility: status badge text supplement (color + shape + text) ── */
    .status-good::before    { content: "✓ "; }
    .status-warning::before { content: "⚠ "; }
    .status-critical::before{ content: "✕ "; }
    /* ── Empty state (C3) ── */
    .empty-state {
        text-align: center; padding: 48px 32px; border: 1px dashed #2d3748;
        border-radius: var(--r-card); margin: 16px 0;
    }
    .empty-state-icon  { font-size: 32px; margin-bottom: 12px; opacity: 0.4; }
    .empty-state-title { font-size: 16px; font-weight: 600; color: #a0aec0; margin-bottom: 6px; }
    .empty-state-body  { font-size: 13px; color: #4a5568; line-height: 1.6; max-width: 420px; margin: 0 auto; }
    .empty-state-action{ margin-top: 16px; font-size: 12px; color: #63b3ed; }
    /* ── Type utility classes (B1) ── */
    .t-metric  { font-size: 32px; font-weight: 700; color: #e2e8f0; line-height: 1.1; }
    .t-heading { font-size: 16px; font-weight: 600; color: #e2e8f0; }
    .t-body    { font-size: 13px; font-weight: 400; color: #a0aec0; line-height: 1.5; }
    .t-caption { font-size: 11px; font-weight: 400; color: #8896a8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Light mode override (toggle in sidebar). Default stays dark until the
# HTML-card inline-style pass (trajectory-match banner, provenance badges,
# etc.) is done in a follow-up — flipping the default now would show those
# dark boxes against a light background app-wide.
if st.session_state.get("light_mode", False):
    st.markdown(
        """
        <style>
        /* ���═ LIGHT MODE — full-coverage enterprise theme ══════════════════ */

        /* ── Page & sidebar backgrounds ── */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"],
        .main .block-container { background: #f7f8fa !important; color: #1a202c !important; }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div:first-child {
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }

        /* ── Typography ── */
        .stApp p, .stApp li, .stApp label, .stApp span,
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stMarkdownContainer"] * { color: #1a202c !important; }
        h1, h2, h3, h4 { color: #1a202c !important; }
        .hero-label  { color: #4a5568 !important; }
        .hero-sub    { color: #4a5568 !important; }
        .t-body, .t-caption { color: #4a5568 !important; }
        .metric-chip-label, .metric-chip-sub { color: #4a5568 !important; }

        /* ── Sidebar nav ── */
        section[data-testid="stSidebar"] button[kind="secondary"] {
            color: #4a5568 !important; background: transparent !important;
        }
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(0,0,0,0.05) !important; color: #1a202c !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"] {
            background: rgba(49,130,206,0.10) !important; color: #2b6cb0 !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"]:hover {
            background: rgba(49,130,206,0.18) !important;
        }

        /* ── Cards, chips, surfaces ── */
        .hero-card {
            background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%) !important;
            border-color: #cbd5e0 !important;
        }
        .metric-chip {
            background: #ffffff !important;
            border-color: #e2e8f0 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.07) !important;
        }
        .metric-chip-value { color: #1a202c !important; }
        .empty-state {
            border-color: #cbd5e0 !important;
            background: #f7f8fa !important;
        }
        .empty-state-title { color: #2d3748 !important; }
        .empty-state-body  { color: #4a5568 !important; }

        /* ── Section headers ── */
        .section-header {
            color: #4a5568 !important;
            border-color: #e2e8f0 !important;
        }

        /* ── Streamlit native widgets ── */
        div[data-testid="stMetricValue"]  { color: #1a202c !important; }
        div[data-testid="stMetricLabel"]  { color: #4a5568 !important; }
        div[data-testid="stMetricDelta"]  { color: #2d3748 !important; }
        .stSelectbox label,
        .stSlider label,
        .stRadio label,
        .stCheckbox label,
        .stTextInput label,
        .stNumberInput label { color: #4a5568 !important; }
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] {
            background: #ffffff !important;
            border-color: #cbd5e0 !important;
            color: #1a202c !important;
        }
        div[data-testid="stExpander"] {
            border-color: #e2e8f0 !important;
            background: #ffffff !important;
        }
        div[data-testid="stExpander"] summary {
            background: #f7f8fa !important;
            color: #2d3748 !important;
        }
        /* st.info / st.warning / st.error boxes */
        div[data-testid="stInfo"]    { background: #ebf8ff !important; border-color: #90cdf4 !important; color: #2c5282 !important; }
        div[data-testid="stWarning"] { background: #fffaf0 !important; border-color: #f6ad55 !important; color: #744210 !important; }
        div[data-testid="stError"]   { background: #fff5f5 !important; border-color: #fc8181 !important; color: #742a2a !important; }

        /* ── Plotly chart backgrounds ── */
        .js-plotly-plot .plotly,
        .js-plotly-plot .bg { fill: #f7f8fa !important; }
        .js-plotly-plot text { fill: #4a5568 !important; }
        .gridlayer path { stroke: #e2e8f0 !important; }

        /* ── Sidebar inline divs with dark hard-coded colours ── */
        section[data-testid="stSidebar"] div[style*="color:#8896a8"],
        section[data-testid="stSidebar"] div[style*="color:#4a5568"],
        section[data-testid="stSidebar"] div[style*="color:#2d3748"] { color: #4a5568 !important; }
        section[data-testid="stSidebar"] div[style*="color:#e2e8f0"],
        section[data-testid="stSidebar"] div[style*="color:#a0aec0"] { color: #2d3748 !important; }
        section[data-testid="stSidebar"] div[style*="background:#1e2a38"],
        section[data-testid="stSidebar"] div[style*="background:#1a202c"] {
            background: #f0f4f8 !important;
            border-color: #e2e8f0 !important;
        }
        /* ══ END LIGHT MODE ════════════════════════════════════════════════ */
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data + model — cached for the session lifetime
# ---------------------------------------------------------------------------

def _nasa_cells_available() -> list[str]:
    """Return which NASA cell CSVs are present in data/raw/."""
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    return [
        cid for cid in NASA_CELL_IDS
        if os.path.exists(os.path.join(data_dir, f"{cid}_summary.csv"))
    ]


def _compute_features_only(battery_dict: dict) -> tuple[dict, dict]:
    """Feature engineering pass — no model training.

    Returns (raw_fdfs, model_inputs) where:
      raw_fdfs:     {cell_id: df_feat}         — build_features() output, no predictions
      model_inputs: {cell_id: (X, y_soh, y_rul)} — ready for train_models()

    Separated from model training so the output can be cached independently,
    allowing the model to be retrained without re-running feature engineering.
    """
    raw_fdfs     = {}
    model_inputs = {}
    for cell_id, cell in battery_dict.items():
        df_feat = build_features(cell["cycles"])
        X, y_soh, y_rul = get_model_matrix(df_feat)
        raw_fdfs[cell_id]     = df_feat
        model_inputs[cell_id] = (X, y_soh, y_rul)
    return raw_fdfs, model_inputs


def _train_and_predict(battery_dict: dict, raw_fdfs: dict,
                       model_inputs: dict) -> tuple[dict, dict, dict]:
    """Train SOH+RUL models on pre-computed features and apply predictions.

    Separated from feature engineering so load_everything() can use a features
    cache hit to skip build_features() while still running LCO + model training.
    """
    X_all     = pd.concat([m[0] for m in model_inputs.values()])
    y_soh_all = pd.concat([m[1] for m in model_inputs.values()])
    y_rul_all = pd.concat([m[2] for m in model_inputs.values()])

    bndl = train_models(X_all, y_soh_all, y_rul_all)
    bndl["metrics"]["n_cells"] = len(battery_dict)
    bndl["metrics"]["n_rows"]  = len(X_all)

    cell_cycles = {cid: cell["cycles"] for cid, cell in battery_dict.items()}
    lco = run_lco(cell_cycles)
    bndl["metrics"]["lco_soh_r2"]   = lco["soh_r2"]
    bndl["metrics"]["lco_rul_r2"]   = lco["rul_r2"]
    bndl["metrics"]["rul_reliable"] = lco["rul_reliable"]
    bndl["metrics"]["lco_per_cell"] = lco["per_cell"]
    per_cell_rul_ok = {
        cid: (fold["rul_r2"] >= RUL_RELIABLE_FLOOR)
        for cid, fold in lco["per_cell"].items()
    }
    bndl["metrics"]["per_cell_rul_reliable"] = per_cell_rul_ok

    featured_dfs, split_cycles = {}, {}
    for cell_id, (X, y_soh, y_rul) in model_inputs.items():
        df_feat = raw_fdfs[cell_id]
        preds   = predict(bndl, X)
        df_out  = df_feat.loc[X.index].copy()
        df_out["soh_pred"]       = preds["soh_pred"]
        df_out["rul_pred"]       = preds["rul_pred"]
        df_out["rul_q10"]        = preds.get("rul_q10", preds["rul_pred"])
        df_out["rul_q90"]        = preds.get("rul_q90", preds["rul_pred"])
        df_out["confidence_tag"] = preds["confidence_tag"]
        featured_dfs[cell_id]  = df_out
        split_idx = int(len(X) * 0.8)
        split_cycles[cell_id]  = int(X["cycle_number"].iloc[split_idx])

    return bndl, featured_dfs, split_cycles


def _train_on_cells(battery_dict: dict) -> tuple[dict, dict, dict]:
    """Full pipeline: feature engineering + model training + predictions.

    Called by page_import() for user-uploaded data. For built-in data,
    load_everything() uses _compute_features_only + _train_and_predict
    so each stage can be cached independently.
    """
    raw_fdfs, model_inputs = _compute_features_only(battery_dict)
    return _train_and_predict(battery_dict, raw_fdfs, model_inputs)


@st.cache_resource(show_spinner=False)
def _get_shap_values(bundle_id: str, bundle: dict):
    """Cache SHAP TreeExplainer results keyed by bundle_id (avoids rebuilding on every render)."""
    try:
        import shap as _shap
        import numpy as _np_shap
        scaler = bundle["scaler"]
        X_test_sc = scaler.transform(bundle["test_data"]["X_test"])
        expl_soh = _shap.TreeExplainer(bundle["soh_model"])
        expl_rul = _shap.TreeExplainer(bundle["rul_model"])
        shap_soh = _np_shap.abs(expl_soh.shap_values(X_test_sc)).mean(axis=0)
        shap_rul = _np_shap.abs(expl_rul.shap_values(X_test_sc)).mean(axis=0)
        return shap_soh, shap_rul
    except Exception:
        return None, None


@st.cache_resource(show_spinner=False)
def load_everything():
    """
    Load synthetic, NASA, and Severson cells in parallel (independent pipelines).

    Two separate models are trained — one per data source — because the
    synthetic and NASA resistance measurements are on incompatible scales
    (synthetic: 0.15-0.40 ohm internal resistance; NASA: 0.04-0.07 ohm Re
    from EIS impedance spectroscopy). A combined model confuses the features
    and produces negative R2. Separate models keep each dataset honest.

    Parallelised with ThreadPoolExecutor: the three pipelines share no state
    so they can run concurrently. st.write/st.progress only called from the
    main thread (after threads complete) to avoid Streamlit threading issues.
    """
    import concurrent.futures as _cf

    def _load_or_train_bg(key: str, cell_dict: dict) -> tuple[dict, dict, dict]:
        """3-tier cache: full bundle → features-only → full pipeline. Thread-safe — no st.* calls."""
        cached = load_cached(key, cell_dict)
        if cached is not None:
            return cached
        feat_cached = load_features_cached(key, cell_dict)
        if feat_cached is not None:
            raw_fdfs, model_inputs = feat_cached
            result = _train_and_predict(cell_dict, raw_fdfs, model_inputs)
            save_cached(key, cell_dict, result)
            return result
        raw_fdfs, model_inputs = _compute_features_only(cell_dict)
        save_features_cached(key, cell_dict, raw_fdfs, model_inputs)
        result = _train_and_predict(cell_dict, raw_fdfs, model_inputs)
        save_cached(key, cell_dict, result)
        return result

    with st.status("Initialising platform…", expanded=False) as _status:
        _prog = st.progress(0, text="Loading data sources in parallel…")

        # ── Build cell dicts (main thread — fast) ──────────────────────────
        synth_ids     = list(CELL_STRESS_PROFILES.keys())
        battery_synth = build_battery(battery_id="Oxford_B1", cell_ids=synth_ids)

        nasa_ids = _nasa_cells_available()
        battery_nasa = build_battery(battery_id="NASA_B1", cell_ids=nasa_ids) if nasa_ids else None

        sev_cell_dicts = {}
        try:
            from severson_loader import load_severson_cells, any_cached as _sev_any_cached
            if _sev_any_cached():
                sev_cells = load_severson_cells(status_fn=lambda msg: None)
                if sev_cells:
                    sev_cell_dicts = {cid: {"cycles": c["cycles"]} for cid, c in sev_cells.items()}
        except Exception:
            pass

        # ── Run three pipelines concurrently ───────────────────────────────
        _prog.progress(10, text="Training models…")
        futures = {}
        with _cf.ThreadPoolExecutor(max_workers=3) as _pool:
            futures["synth"] = _pool.submit(_load_or_train_bg, "synth", battery_synth["cells"])
            if battery_nasa:
                futures["nasa"] = _pool.submit(_load_or_train_bg, "nasa", battery_nasa["cells"])
            if sev_cell_dicts:
                futures["severson"] = _pool.submit(_load_or_train_bg, "severson", sev_cell_dicts)

        _prog.progress(90, text="Merging results…")

        bundle_synth, fdfs_synth, sc_synth = futures["synth"].result()
        bundle_nasa,  fdfs_nasa,  sc_nasa  = futures["nasa"].result()  if "nasa"     in futures else (None, {}, {})
        bundle_sev,   fdfs_sev,   sc_sev   = futures["severson"].result() if "severson" in futures else (None, {}, {})

        _prog.progress(100, text="Platform ready ✓")
        # st.status()/st.progress() return None when called outside a real
        # Streamlit script run (e.g. imported by src/api.py under a bare
        # uvicorn process) — guard so load_everything() works from both.
        if _status is not None:
            _status.update(label="Platform ready ✓", state="complete", expanded=False)

    featured_dfs = {**fdfs_synth, **fdfs_nasa, **fdfs_sev}
    split_cycles = {**sc_synth, **sc_nasa, **sc_sev}
    bundles = {"synth": bundle_synth, "nasa": bundle_nasa, "severson": bundle_sev}

    return featured_dfs, bundles, split_cycles


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

# Grouped nav — each entry is (group_label, [(page_label, page_key), ...])
# Passport and Reports are merged into "Compliance" (tabbed view).
# "Consequences" is renamed "EOL Economics" — routing key unchanged.
NAV_GROUPS = [
    ("Analyse", [
        ("Overview",   "overview"),
        ("Health",     "health"),
        ("Explore",    "compare"),
    ]),
    ("EU Passport", [          # A6: promoted from 3rd to 2nd — 2027 deadline
        ("Compliance", "compliance"),
    ]),
    ("Operate", [
        ("Fleet",        "fleet"),
        ("Decide & Ask", "decision"),
        ("Live Monitor", "live_monitor"),
    ]),
    ("Configure", [
        ("Configure", "configure"),
    ]),
]

# Flat alias kept for any code that still iterates NAV_ITEMS
NAV_ITEMS = [(label, key, True) for _, items in NAV_GROUPS for label, key in items]

# Contextual action bar — 3 quick-jump suggestions per page
# Each entry: (button_label, target_page_key, tooltip)
PAGE_ACTIONS: dict[str, list[tuple[str, str, str]]] = {
    "overview":        [("Health →",          "health",          "Deep-dive degradation curves"),
                        ("Copilot →",         "copilot",         "Plain-English explanation"),
                        ("Recommendations →", "recommendations", "Recommended action for this cell")],
    "health":          [("Compare →",         "compare",         "Side-by-side with another cell"),
                        ("Insights →",        "insights",        "What is driving degradation"),
                        ("EOL Economics →",   "consequences",    "Model end-of-life economics")],
    "compare":         [("Health →",          "health",          "Single-cell deep dive"),
                        ("Fleet →",           "fleet",           "Full fleet ranking"),
                        ("Insights →",        "insights",        "SHAP feature attribution")],
    "insights":        [("Health →",          "health",          "Visualise the degradation curves"),
                        ("Copilot →",         "copilot",         "Get a narrative explanation"),
                        ("Recommendations →", "recommendations", "Recommended action")],
    "copilot":         [("Overview →",        "overview",        "Back to key metrics"),
                        ("Health →",          "health",          "Visualise curves"),
                        ("Recommendations →", "recommendations", "See recommended action")],
    "fleet":           [("Health →",          "health",          "Inspect selected cell"),
                        ("Recommendations →", "recommendations", "Action for selected cell"),
                        ("EOL Economics →",   "consequences",    "Economics for selected cell")],
    "recommendations": [("EOL Economics →",   "consequences",    "Model the economics in detail"),
                        ("Health →",          "health",          "Review degradation curves"),
                        ("Compliance →",      "compliance",      "Generate EU battery passport")],
    "consequences":    [("Recommendations →", "recommendations", "See the recommended action"),
                        ("Compliance →",      "compliance",      "EU passport and reports"),
                        ("Sustainability →",  "sustainability",  "Lifecycle CO₂ and materials")],
    "sustainability":  [("Compliance →",      "compliance",      "EU battery passport"),
                        ("EOL Economics →",   "consequences",    "End-of-life economics"),
                        ("Fleet →",           "fleet",           "Fleet-level overview")],
    "compliance":      [("Sustainability →",  "sustainability",  "Lifecycle CO₂ analysis"),
                        ("EOL Economics →",   "consequences",    "End-of-life economics"),
                        ("Overview →",        "overview",        "Back to key metrics")],
    "grading":         [("Fleet →",           "fleet",           "Fleet ranking"),
                        ("Health →",          "health",          "Deep-dive this cell"),
                        ("Insights →",        "insights",        "Feature attribution")],
    "import":          [("Overview →",        "overview",        "Analyse imported data"),
                        ("Fleet →",           "fleet",           "Fleet ranking"),
                        ("Settings →",        "settings",        "Configure thresholds")],
    "settings":        [("Overview →",        "overview",        "Back to analysis"),
                        ("Fleet →",           "fleet",           "Fleet view"),
                        ("Import →",          "import",          "Import new data")],
    "live_monitor":    [("Health →",          "health",          "Deep-dive selected cell"),
                        ("Fleet →",           "fleet",           "Fleet ranking"),
                        ("Recommendations →", "recommendations", "Action for this cell")],
}

def _upload_status_line(meta: dict) -> str:
    """One-line status string for the My Data mode row."""
    n = meta["n_cells"]
    k = meta.get("calibrating_count", 0)
    parts = [f"{n} cells"]
    if k > 0:
        parts.append(f"{k} Calibrating")
    parts.append("uploaded today")
    if meta.get("lco_limited"):
        parts.append("limited LCO")
    return " · ".join(parts)


def render_mode_switcher(nasa_n: int, synth_n: int, up_meta: dict | None,
                         sev_n: int = 0):
    """Persistent data-source selector rendered inside the sidebar."""
    current = st.session_state.get("data_mode", "nasa")

    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding:0 4px 6px'>Data Source</div>",
        unsafe_allow_html=True,
    )

    modes = [
        {
            "key":       "severson",
            "label":     "Severson 2019 (LFP)",
            "status":    f"{sev_n} cells · real measured · Nature Energy 2019",
            "available": sev_n > 0,
        },
        {
            "key":       "nasa",
            "label":     "NASA PCoE",
            "status":    f"{nasa_n} cells · real measured · LiCoO2 chemistry",
            "available": nasa_n > 0,
        },
        {
            "key":       "synthetic",
            "label":     "Synthetic Fleet",
            "status":    f"{synth_n} cells · physics-informed",
            "available": True,
        },
        {
            "key":       "uploaded",
            "label":     "My Data",
            "status":    _upload_status_line(up_meta) if up_meta else "Not yet uploaded",
            "available": up_meta is not None,
        },
    ]

    for m in modes:
        is_active    = current == m["key"]
        is_available = m["available"]

        if is_active:
            # Active row: styled div — ● bold label + muted status
            st.markdown(
                f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:8px;"
                f"padding:9px 12px;margin-bottom:5px'>"
                f"<div style='font-size:13px;font-weight:700;color:#e2e8f0'>"
                f"● {m['label']}</div>"
                f"<div style='font-size:11px;color:#8896a8;margin-top:2px'>{m['status']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif is_available:
            # Inactive available: button triggers mode switch
            if st.button(
                f"○  {m['label']}",
                key=f"mode_btn_{m['key']}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state["data_mode"] = m["key"]
                st.rerun()
            st.markdown(
                f"<div style='font-size:11px;color:#4a5568;margin:-8px 0 5px 4px'>"
                f"{m['status']}</div>",
                unsafe_allow_html=True,
            )
        else:
            # Unavailable (My Data, no upload yet) — grayed out, not clickable
            st.markdown(
                f"<div style='padding:8px 12px;margin-bottom:5px;opacity:0.45'>"
                f"<div style='font-size:13px;color:#8896a8'>○  {m['label']}</div>"
                f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>{m['status']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# D4: Command palette dialog — ⌘K / Ctrl+K from any page
@st.dialog("Command Palette")
def _command_palette_dialog():
    _PALETTE_ROUTES = {
        "fleet": ("fleet",    ["fleet", "cells", "monitor", "attention", "ranking", "overview", "all cells"]),
        "overview": ("overview", ["health", "cell", "soh", "status", "check"]),
        "compliance": ("compliance", ["passport", "eu", "compliance", "regulation", "eol"]),
        "decision": ("decision",  ["decision", "replace", "repurpose", "second life", "what should"]),
        "copilot": ("copilot",   ["copilot", "ask", "budget", "cost", "risk", "question"]),
        "health": ("health",    ["degrading", "mechanism", "lli", "lam", "fade", "resistance"]),
    }
    st.markdown("### ⌘ Command Palette")
    _pal_input = st.text_input(
        "What do you want to do?",
        placeholder="e.g. 'show cells with high fade', 'replacement budget', 'EU passport'…",
        key="_palette_input",
    )
    st.caption("Navigates to the best matching page. Press Enter or click Go.")
    if st.button("Go", key="_palette_go", use_container_width=True, type="primary") and _pal_input:
        _lower = _pal_input.lower()
        _dest = "copilot"  # default
        _best_score = 0
        for _page, (_route, _keywords) in _PALETTE_ROUTES.items():
            _score = sum(1 for kw in _keywords if kw in _lower)
            if _score > _best_score:
                _best_score = _score
                _dest = _route
        st.session_state.page = _dest
        if _dest == "copilot":
            st.session_state["copilot_free_text"] = _pal_input
            st.session_state.pop("copilot_query", None)
        elif _dest == "fleet" and any(kw in _lower for kw in ["fade", "soh below", "worst"]):
            st.session_state["fleet_filter_query"] = _pal_input
        st.rerun()


# Guided tour — 4-step onboarding modal shown once per session for first-time
# visitors. Uses the same @st.dialog pattern as the Command Palette above.
_TOUR_STEPS = [
    (
        "Real measured data, not a toy demo",
        "This fleet leads with <strong>NASA PCoE</strong> cells — real LiCoO₂ 18650 "
        "measurements, not synthetic curves. Watch for the "
        "<strong>Failure trajectory match</strong> chip in the sidebar: it flags cells "
        "whose degradation pattern closely resembles a cell that already failed.",
    ),
    (
        "Fleet — see every cell at a glance",
        "The Fleet page ranks every cell by health and flags which ones need attention "
        "first, with a grade (A/B/C) and proactive alerts for anything trending toward "
        "end-of-life.",
    ),
    (
        "Decide & Ask — turn health into a decision",
        "Decide & Ask recommends Continue / Inspect / Second-Life / Recycle for each cell, "
        "backed by an NPV comparison. Every decision you log is kept in an auditable trail.",
    ),
    (
        "Compliance — EU Battery Passport",
        "The Compliance page tracks EU Battery Passport completeness field-by-field, so you "
        "can see exactly what regulatory data is available, estimated, or still missing.",
    ),
]


# ── First-run onboarding overlay sequencing ─────────────────────────────────
# Rule, not a per-instance patch: at most one first-run overlay (role picker,
# guided tour, any future one) renders per session, in this fixed order.
# Two first-run modals stacked on top of each other on a fresh login was a
# real, confusing bug found in review — this makes it structurally
# impossible for a *future* onboarding overlay to reintroduce the same
# collision, since adding one here is the only thing a new overlay needs to
# do to be sequenced correctly (no need to hand-write an "and the other one
# is already done" condition again, the way the tour's fix originally did).
_FIRST_RUN_OVERLAYS = ["role_chosen", "tour_seen"]  # session_state "done" flags, in show-order


def _active_first_run_overlay() -> "str | None":
    """Return the session_state key of the next not-yet-completed first-run
    overlay, or None once all of them are done."""
    for key in _FIRST_RUN_OVERLAYS:
        if not st.session_state.get(key, False):
            return key
    return None


@st.dialog("Welcome — Guided Tour")
def _guided_tour_dialog():
    step = st.session_state.get("tour_step", 0)
    title, body = _TOUR_STEPS[step]

    st.progress((step + 1) / len(_TOUR_STEPS))
    st.markdown(f"#### {title}")
    _md_html(f"<div style='font-size:14px;color:#a0aec0;line-height:1.6'>{body}</div>")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _col_skip, _col_next = st.columns([1, 2])
    with _col_skip:
        if st.button("Skip tour", key="tour_skip", use_container_width=True):
            st.session_state["tour_seen"] = True
            st.rerun()
    with _col_next:
        _is_last = step == len(_TOUR_STEPS) - 1
        if st.button(
            "Finish tour" if _is_last else "Next →",
            key="tour_next", type="primary", use_container_width=True,
        ):
            if _is_last:
                st.session_state["tour_seen"] = True
                st.session_state["page"] = "compliance"
            else:
                st.session_state["tour_step"] = step + 1
            st.rerun()


def render_sidebar(cell_ids: list[str], mode: str, nasa_n: int, synth_n: int,
                   up_meta: dict | None, sev_n: int = 0,
                   active_fdfs: dict | None = None,
                   traj_flag: int = 0) -> str:  # A4: number of flagged cells
    with st.sidebar:
        # Dynamic subtitle based on active mode
        n_cells = len(cell_ids)
        if mode == "severson":
            subtitle = f"{n_cells} Severson 2019 LFP cells · real measured"
        elif mode == "nasa":
            subtitle = f"{n_cells} NASA real cells · leave-cell-out model"
        elif mode == "synthetic":
            subtitle = f"{n_cells} synthetic cells · leave-cell-out model"
        elif mode == "uploaded":
            cell_label = up_meta.get("cell_ids", cell_ids) if up_meta else cell_ids
            subtitle = f"{n_cells} uploaded cell{'s' if n_cells != 1 else ''} · your data"
        else:
            subtitle = f"{nasa_n + synth_n} cells ({synth_n} synthetic + {nasa_n} NASA real) · multi-cell model"

        st.markdown(
            f"<div style='padding:0 4px 16px'>"
            f"<div style='font-size:17px;font-weight:800;color:#ffffff;letter-spacing:0.01em'>⚡ Battery Intel</div>"
            f"<div style='font-size:11px;color:#8896a8;margin-top:3px'>{subtitle}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # A1: Inline page search (replaces 3-column task picker)
        _SIDEBAR_SEARCH_ROUTES = {
            "fleet":      ["fleet", "cells", "monitor", "ranking", "attention", "all cells", "worst"],
            "overview":   ["overview", "cell health", "soh", "status", "check cell"],
            "compliance": ["passport", "eu", "compliance", "regulation", "eol", "battery passport"],
            "decision":   ["decision", "replace", "repurpose", "second life", "what should", "copilot", "ask", "cost", "budget", "risk"],
            "health":     ["health", "degrading", "mechanism", "lli", "lam", "fade", "resistance"],
            "compare":    ["compare", "cluster", "cohort", "side by side", "explore"],
            "configure":  ["import", "settings", "upload", "configure", "data source", "threshold"],
            "live_monitor": ["live", "mqtt", "streaming", "bms", "anomaly"],
        }
        _sb_query = st.text_input(
            "search", placeholder="Search pages…",
            key="_sidebar_search", label_visibility="collapsed",
        )
        if _sb_query and _sb_query != st.session_state.get("_last_sb_search", ""):
            st.session_state["_last_sb_search"] = _sb_query
            _sq = _sb_query.lower().strip()
            _best_dest, _best_score = "fleet", 0
            for _dest, _kws in _SIDEBAR_SEARCH_ROUTES.items():
                _sc = sum(1 for k in _kws if k in _sq)
                if _sc > _best_score:
                    _best_score, _best_dest = _sc, _dest
            if _best_score == 0:
                # fall through to copilot with free-text
                _best_dest = "decision"
                st.session_state["copilot_free_text"] = _sb_query
            st.session_state.page = _best_dest
            st.rerun()
        # Quick-access strip (single row, compact)
        _qa1, _qa2, _qa3 = st.columns(3)
        with _qa1:
            if st.button("Fleet", key="qa_fleet", use_container_width=True):
                st.session_state.page = "fleet"; st.rerun()
        with _qa2:
            if st.button("Cell", key="qa_cell", use_container_width=True):
                st.session_state.page = "overview"; st.rerun()
        with _qa3:
            if st.button("Passport", key="qa_passport", use_container_width=True):
                st.session_state.page = "compliance"; st.rerun()
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Data source (collapsed by default) ──
        with st.expander("Data source", expanded=False):
            render_mode_switcher(nasa_n, synth_n, up_meta, sev_n=sev_n)

        # ── Role selector (A2: compact inline — no nested expander) ──
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _cur_role = st.session_state.get("user_role", "Engineer")
        _role_icons = {"Engineer": "⚙", "Fleet Manager": "🗂", "Executive": "📊"}
        _role_icon = _role_icons.get(_cur_role, "⚙")
        _rc1, _rc2 = st.columns([4, 1])
        with _rc1:
            st.markdown(
                f"<div style='font-size:11px;color:#8896a8;padding:4px 0 2px'>"
                f"{_role_icon} <span style='color:#e2e8f0;font-weight:600'>{_cur_role}</span></div>",
                unsafe_allow_html=True,
            )
        with _rc2:
            if st.button("↺", key="change_role_btn", use_container_width=True,
                         help="Switch role (Engineer / Fleet Manager / Executive)"):
                st.session_state["role_chosen"] = False
                st.rerun()

        # ── Nav (grouped) ──
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if "page" not in st.session_state:
            st.session_state.page = "overview"
        current_page = st.session_state.page

        # A4: Persistent trajectory match chip — visible from any page
        if traj_flag > 0:
            _chip_col = "#ef4444" if traj_flag >= 2 else "#f59e0b"
            if st.button(
                f"⚠ {traj_flag} cell{'s' if traj_flag != 1 else ''} flagged",
                key="traj_chip_sidebar",
                use_container_width=True,
                help="Failure trajectory matches detected — click to view in Fleet",
            ):
                st.session_state.page = "fleet"
                st.rerun()
            st.markdown(
                f"<div style='font-size:10px;color:{_chip_col};margin:-6px 0 6px 4px;"
                f"padding:0 2px'>Failure trajectory match</div>",
                unsafe_allow_html=True,
            )

        for group_label, group_items in NAV_GROUPS:
            st.markdown(
                f"<div style='font-size:10px;font-weight:700;color:#4a5568;"
                f"text-transform:uppercase;letter-spacing:0.1em;"
                f"padding:10px 4px 4px;margin-top:2px'>{group_label}</div>",
                unsafe_allow_html=True,
            )
            for label, key in group_items:
                if st.button(
                    label, key=f"nav_{key}", use_container_width=True,
                    type="primary" if current_page == key else "secondary",
                ):
                    st.session_state.page = key
                    st.rerun()

        # Derive chemistry from active mode — stored for page-level display
        _mode_chem = st.session_state.get("data_mode", "synthetic")
        _chem_label = {
            "severson":  "LFP (Severson 2019)",
            "nasa":      "LiCoO₂ (NASA PCoE)",
            "synthetic": "LiCoO₂ (synthetic)",
            "uploaded":  "User-defined",
        }.get(_mode_chem, "LiCoO₂")
        st.session_state["active_chemistry"] = _chem_label

        # ── Cell selector ──
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            "letter-spacing:0.08em;padding:0 4px 8px'>Cell</div>",
            unsafe_allow_html=True,
        )
        # Apply any pending Prev/Next navigation BEFORE the selectbox renders.
        # Newer Streamlit raises StreamlitAPIException if you set a widget-keyed
        # session state value after that widget has already been instantiated in
        # the current run, so we use a proxy key "_nav_cell" and consume it here.
        if "_nav_cell" in st.session_state:
            _nav_target = st.session_state.pop("_nav_cell")
            if _nav_target in cell_ids:
                st.session_state["selected_cell"] = _nav_target
        # Determine default index — preserve current selection when cell list changes.
        # If the previously-selected cell isn't in the new list (e.g. after switching
        # data source), the widget's *stored* session_state value must be corrected
        # here, before the widget is instantiated below — passing index= alone only
        # sets the default the first time this key is ever created, so a stale value
        # from a prior data source otherwise keeps rendering even though it's invalid.
        _cur_sel = st.session_state.get("selected_cell")
        if _cur_sel not in cell_ids:
            st.session_state["selected_cell"] = cell_ids[0]
        _sel_idx = cell_ids.index(st.session_state["selected_cell"])
        selected = st.selectbox(
            "Select cell",
            options=cell_ids,
            index=_sel_idx,
            key="selected_cell",
            label_visibility="collapsed",
        )

        # Prev / Next quick navigation — write to proxy key, not widget key
        _cur_idx = cell_ids.index(selected)
        _nav_prev, _nav_next = st.columns(2)
        with _nav_prev:
            if st.button("← Prev", key="cell_prev", use_container_width=True,
                         disabled=(_cur_idx == 0)):
                st.session_state["_nav_cell"] = cell_ids[_cur_idx - 1]
                st.rerun()
        with _nav_next:
            if st.button("Next →", key="cell_next", use_container_width=True,
                         disabled=(_cur_idx == len(cell_ids) - 1)):
                st.session_state["_nav_cell"] = cell_ids[_cur_idx + 1]
                st.rerun()

        # ── Fleet alerts (directly after Prev/Next) ──
        if active_fdfs:
            _soh_thresh  = float(st.session_state.get("soh_alert_pct", 85))
            _res_mult    = float(st.session_state.get("resistance_alert_mult", 1.8))
            _spread_thresh = float(st.session_state.get("spread_alert_pct", 5.0))
            _alert_msgs  = []
            for _cid, _fdf in active_fdfs.items():
                _latest = _fdf.iloc[-1]
                if float(_latest.get("soh_pct", 100)) < _soh_thresh:
                    _alert_msgs.append(("warn", f"**{_cid}** SOH {float(_latest['soh_pct']):.1f}% — below {_soh_thresh:.0f}%"))
                if "resistance_ohm" in _fdf.columns and len(_fdf) > 1:
                    _r_init = float(_fdf["resistance_ohm"].iloc[0])
                    _r_now  = float(_latest.get("resistance_ohm", 0))
                    if _r_init > 0 and _r_now > _r_init * _res_mult:
                        _alert_msgs.append(("error", f"**{_cid}** R = {_r_now/_r_init:.2f}× initial"))
            if len(active_fdfs) > 1:
                _soh_vals = [float(fdf["soh_pct"].iloc[-1]) for fdf in active_fdfs.values()]
                _spread   = max(_soh_vals) - min(_soh_vals)
                if _spread > _spread_thresh:
                    _alert_msgs.append(("warn", f"**Fleet spread** {_spread:.1f}% SOH range"))
            if _alert_msgs:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                _label = f"🔔 {len(_alert_msgs)} Alert{'s' if len(_alert_msgs) > 1 else ''}"
                with st.expander(_label, expanded=False):
                    for _kind, _msg in _alert_msgs:
                        if _kind == "error":
                            st.error(_msg)
                        else:
                            st.warning(_msg)

        # ── Cell annotation — compact one-liner ──
        if mode == "uploaded":
            temp_assumed_cells = (up_meta or {}).get("temperature_assumed_cells", [])
            temp_assumed = selected in temp_assumed_cells
            _ann_color, _ann_text = ("#63b3ed", "Uploaded · T " + ("assumed" if temp_assumed else "measured"))
        elif mode == "nasa":
            _ann_color, _ann_text = "#48bb78", "NASA PCoE · real · T=24°C · 2A"
        elif mode == "severson":
            _ann_color, _ann_text = "#48bb78", "Severson 2019 · real · LFP"
        else:
            p = CELL_STRESS_PROFILES.get(selected, {})
            sf = _stress_factor(p.get("temp_mean", 25), p.get("c_rate", 1), p.get("dod", 1))
            _ann_color, _ann_text = "#fc8181", f"Synthetic · {sf:.2f}× stress · T={p.get('temp_mean',25):.0f}°C"
        st.markdown(
            f"<div style='font-size:10px;color:{_ann_color};padding:4px 4px 0'>{_ann_text}</div>",
            unsafe_allow_html=True,
        )

        # D4: Command palette — button only, no fragile JS keyboard injection
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Search pages…", key="cmd_palette_btn", use_container_width=True,
                     help="Search pages and navigate quickly"):
            _command_palette_dialog()

        # Theme toggle — separate widget key from the state key it drives,
        # since Streamlit forbids setting a widget-keyed session_state value
        # after the widget has been instantiated this run.
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _lm_current = st.session_state.get("light_mode", False)
        _lm_toggle = st.toggle("Light mode", value=_lm_current, key="light_mode_toggle")
        if _lm_toggle != _lm_current:
            st.session_state["light_mode"] = _lm_toggle
            st.rerun()

    return selected












# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Authentication gate ───────────────────────────────────────────────────
    from _pages.login import render_login
    from _pages.explore import page_compare
    from _pages.live_monitor import page_live_monitor
    from _pages.import_page import page_import
    from _pages.settings import page_settings
    from _pages.overview import page_overview
    from _pages.health import page_health
    from _pages.grading import page_grading
    from _pages.fleet import page_fleet
    from _pages.copilot import page_copilot
    from _pages.decision import page_decision
    from _pages.compliance import page_compliance
    if not render_login():
        return   # login form rendered; stop until credentials provided

    _train_placeholder = st.empty()
    _train_placeholder.markdown(
        "<div style='text-align:center;padding:40px;color:#4a5568;font-size:14px'>"
        "Initialising models… (first run only — cached on subsequent loads)</div>",
        unsafe_allow_html=True,
    )
    featured_dfs_all, bundles, split_cycles_all = load_everything()
    _train_placeholder.empty()

    # ── Guided tour (once per session, first-time visitors) ───────────────────
    # Sequenced via _active_first_run_overlay() so it can never stack with the
    # role-onboarding interstitial (below) or any future first-run overlay.
    if "tour_seen" not in st.session_state:
        st.session_state["tour_seen"] = False
    if "tour_step" not in st.session_state:
        st.session_state["tour_step"] = 0
    if _active_first_run_overlay() == "tour_seen":
        _guided_tour_dialog()

    # ── Failure trajectory memory (built once per session) ────────────────────
    # Uses ALL cells across all sources so the signature library is as large
    # as possible, regardless of which data source is currently active.
    # Signatures persist across sessions (src/db.py): load what's already
    # known, merge in freshly-built signatures from this session's cells
    # (freshest wins per cell_id), then save the merged library back.
    if "trajectory_memory" not in st.session_state:
        import db as _db_tm
        _db_tm.init_db()
        _tm = TrajectoryMemory()
        _persisted_sigs = _db_tm.load_failure_signatures(st.session_state["auth_org_id"])
        _tm.build(featured_dfs_all)
        _tm.merge_dedupe_by_cell_id(_persisted_sigs)
        _tm.save(st.session_state["auth_org_id"])
        st.session_state["trajectory_memory"] = _tm
    trajectory_memory: TrajectoryMemory = st.session_state["trajectory_memory"]

    # ── Separate built-in cells by type ──────────────────────────────────────
    nasa_fdfs  = {k: v for k, v in featured_dfs_all.items() if k in NASA_CELL_IDS}
    sev_fdfs   = {k: v for k, v in featured_dfs_all.items() if k.startswith("S-")}
    synth_fdfs = {k: v for k, v in featured_dfs_all.items()
                  if k not in NASA_CELL_IDS and not k.startswith("S-")}
    nasa_sc    = {k: v for k, v in split_cycles_all.items() if k in NASA_CELL_IDS}
    sev_sc     = {k: v for k, v in split_cycles_all.items() if k.startswith("S-")}
    synth_sc   = {k: v for k, v in split_cycles_all.items()
                  if k not in NASA_CELL_IDS and not k.startswith("S-")}

    # ── Session state init (first run per session only) ───────────────────────
    if "data_mode" not in st.session_state:
        default_mode = "severson" if sev_fdfs else ("nasa" if nasa_fdfs else "synthetic")
        st.session_state["data_mode"] = default_mode
    if "uploaded_mode_meta" not in st.session_state:
        st.session_state["uploaded_mode_meta"] = None

    # ── Persistence hydration (SQLite-backed, first run per session only) ─────
    import db as _db_main
    _db_main.init_db()
    if "decision_log" not in st.session_state:
        st.session_state["decision_log"] = _db_main.load_decisions(st.session_state["auth_org_id"])
    if "pinned_cell" not in st.session_state:
        st.session_state["pinned_cell"] = _db_main.get_setting(st.session_state["auth_org_id"], "pinned_cell")
    if "app_profile" not in st.session_state:
        _persisted_profile = _db_main.get_setting(st.session_state["auth_org_id"], "app_profile")
        if _persisted_profile is not None:
            st.session_state["app_profile"] = _persisted_profile
    if "cost_of_delay_mult" not in st.session_state:
        _persisted_cod = _db_main.get_setting(st.session_state["auth_org_id"], "cost_of_delay_mult")
        if _persisted_cod is not None:
            st.session_state["cost_of_delay_mult"] = _persisted_cod
    for _wh_key in ("webhook_url", "webhook_secret", "webhook_events"):
        if _wh_key not in st.session_state:
            _persisted_wh = _db_main.get_setting(st.session_state["auth_org_id"], _wh_key)
            if _persisted_wh is not None:
                st.session_state[_wh_key] = _persisted_wh

    # Application EOL threshold — user-configurable in Settings.
    # Changing this does NOT retrain the model; it rescales the displayed RUL
    # using the current fade rate to project to the user-defined threshold.
    if "eol_threshold_pct" not in st.session_state:
        st.session_state["eol_threshold_pct"] = _db_main.get_setting(st.session_state["auth_org_id"], "eol_threshold_pct", 80.0)

    # ── Resolve active data from current mode ─────────────────────────────────
    mode      = st.session_state["data_mode"]
    up_fdfs   = st.session_state.get("uploaded_featured_dfs", {})
    up_sc     = st.session_state.get("uploaded_split_cycles", {})
    up_bundle = st.session_state.get("uploaded_bundle")
    up_meta   = st.session_state["uploaded_mode_meta"]

    # If "My Data" mode is selected but this session hasn't uploaded anything
    # yet (e.g. a returning user in a fresh browser session), try to reload
    # this org's previously-uploaded bundle from disk before falling back to
    # the empty state below — an org's uploaded fleet now survives a refresh.
    if mode == "uploaded" and (not up_fdfs or up_bundle is None):
        from bundle_cache import load_tenant_bundle
        _persisted_tenant = load_tenant_bundle(st.session_state["auth_org_id"])
        if _persisted_tenant is not None:
            up_fdfs, up_bundle, up_sc = _persisted_tenant
            st.session_state["uploaded_featured_dfs"] = up_fdfs
            st.session_state["uploaded_bundle"]       = up_bundle
            st.session_state["uploaded_split_cycles"] = up_sc

    # Guard: if mode is "uploaded" but the bundle was cleared (e.g. after reset),
    # fall back silently to NASA mode.
    if mode == "uploaded" and (not up_fdfs or up_bundle is None):
        st.session_state["data_mode"] = "nasa"
        mode = "nasa"

    # Guard: unknown mode value (stale session state after server restart)
    if mode not in ("nasa", "synthetic", "severson", "uploaded"):
        mode = "severson" if sev_fdfs else ("nasa" if nasa_fdfs else "synthetic")
        st.session_state["data_mode"] = mode

    if mode == "severson":
        active_fdfs   = sev_fdfs
        active_sc     = sev_sc
        active_bundle = bundles.get("severson") or bundles.get("nasa")
    elif mode == "nasa":
        active_fdfs   = nasa_fdfs
        active_sc     = nasa_sc
        active_bundle = bundles["nasa"]
    elif mode == "synthetic":
        active_fdfs   = synth_fdfs
        active_sc     = synth_sc
        active_bundle = bundles["synth"]
    else:  # uploaded
        active_fdfs   = up_fdfs
        active_sc     = up_sc
        active_bundle = up_bundle

    cell_ids = list(active_fdfs.keys())

    # A2: Role onboarding interstitial — shown once per session. Sequenced via
    # _active_first_run_overlay() so it can never stack with the guided tour
    # or any future first-run overlay.
    if _active_first_run_overlay() == "role_chosen":
        st.markdown(
            "<div style='max-width:680px;margin:80px auto 0;text-align:center'>"
            "<div style='font-size:28px;font-weight:800;color:#e2e8f0;margin-bottom:8px'>"
            "Welcome to Battery Intelligence</div>"
            "<div style='font-size:14px;color:#718096;margin-bottom:32px'>"
            "I'll personalise the dashboard for your role. You can change this any time in Settings.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        _r1, _r2, _r3 = st.columns(3)
        _role_picked = None
        with _r1:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                "<div style='font-size:28px;margin-bottom:8px'>🔧</div>"
                "<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>Engineer</div>"
                "<div style='font-size:12px;color:#718096'>Diagnose cells · Deep analytics · Root-cause tools</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select Engineer", key="onboard_eng", use_container_width=True):
                _role_picked = "Engineer"
        with _r2:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                "<div style='font-size:28px;margin-bottom:8px'>🚗</div>"
                "<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>Fleet Manager</div>"
                "<div style='font-size:12px;color:#718096'>Monitor fleet · Prioritise replacements · Alerts</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select Fleet Manager", key="onboard_fleet", use_container_width=True):
                _role_picked = "Fleet Manager"
        with _r3:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                "<div style='font-size:28px;margin-bottom:8px'>📊</div>"
                "<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>Executive</div>"
                "<div style='font-size:12px;color:#718096'>Fleet KPIs · CAPEX forecast · ESG compliance</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select Executive", key="onboard_exec", use_container_width=True):
                _role_picked = "Executive"
        if _role_picked:
            st.session_state["user_role"] = _role_picked
            st.session_state["role_chosen"] = True
            # Don't clobber the guided tour's "Finish tour" destination.
            if not (st.session_state.get("tour_seen") and st.session_state.get("page") == "compliance"):
                if _role_picked == "Executive":
                    st.session_state.page = "exec_summary"
                elif _role_picked == "Fleet Manager":
                    st.session_state.page = "fleet"
                else:
                    st.session_state.page = "overview"
            st.rerun()
        st.stop()

    # A4: count flagged cells for trajectory chip in sidebar
    try:
        _traj_flag = len(trajectory_memory.match_fleet(active_fdfs))
    except Exception:
        _traj_flag = 0

    selected = render_sidebar(
        cell_ids    = cell_ids,
        mode        = mode,
        nasa_n      = len(nasa_fdfs),
        synth_n     = len(synth_fdfs),
        up_meta     = up_meta,
        sev_n       = len(sev_fdfs),
        active_fdfs = active_fdfs,
        traj_flag   = _traj_flag,
    )

    df          = active_fdfs[selected]
    split_cycle = active_sc[selected]
    bundle      = active_bundle
    page        = st.session_state.get("page", "fleet")

    # ── Audit logging ─────────────────────────────────────────────────────────
    import sys as _sys
    _src = os.path.join(os.path.dirname(__file__), "..", "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from audit import log_page_view
    _last_audited = st.session_state.get("_audit_last", "")
    if f"{page}:{selected}" != _last_audited:
        log_page_view(page, selected)
        st.session_state["_audit_last"] = f"{page}:{selected}"

    # ── Demo mode notice (footer-style, low visual weight) ────────────────────
    st.markdown(
        "<div style='text-align:right;margin-bottom:4px'>"
        "<span title='No auth · session-scoped uploads · data not persisted — see README → Production Roadmap' "
        "style='font-size:10px;color:#4a5568;cursor:default'>demo mode</span></div>",
        unsafe_allow_html=True,
    )

    # Per-cell reliability: use the specific fold R² for this cell, not the group average.
    per_cell_ok  = bundle["metrics"].get("per_cell_rul_reliable", {})
    rul_reliable = per_cell_ok.get(selected, bundle["metrics"].get("rul_reliable", True))

    if page == "overview":
        page_overview(df, split_cycle, selected, rul_reliable=rul_reliable, bundle=bundle,
                      trajectory_memory=trajectory_memory)
    elif page == "health":
        page_health(df, split_cycle, selected, bundle=bundle, rul_reliable=rul_reliable)
    elif page == "compare":
        page_compare(cell_ids, active_fdfs, bundles)
    elif page in ("copilot", "insights"):
        page_copilot(cell_ids, active_fdfs, bundles, selected)
    elif page in ("decision", "consequences", "recommendations"):
        page_decision(selected, df, active_fdfs, bundles, rul_reliable)
    elif page in ("compliance", "sustainability", "passport", "reports"):
        page_compliance(selected, df, bundle, rul_reliable, active_fdfs, bundles)
    elif page in ("fleet", "exec_summary"):
        page_fleet(active_fdfs, bundles, trajectory_memory=trajectory_memory)
    elif page == "grading":
        page_grading(cell_ids, active_fdfs, bundles, selected)
    elif page == "live_monitor":
        page_live_monitor(cell_ids, active_fdfs)
    elif page in ("settings", "import", "configure"):
        # Merged Configure page — Import and Settings as tabs
        st.markdown("# Configure")
        _cfg_tab_import, _cfg_tab_settings = st.tabs(["Import Data", "Settings"])
        with _cfg_tab_import:
            page_import()
        with _cfg_tab_settings:
            page_settings(
                featured_dfs_all,
                {"nasa": bundles["nasa"], "synth": bundles["synth"], "uploaded": up_bundle},
            )
    elif page in COMING_SOON_META:
        page_coming_soon(page)
    else:
        page_overview(df, split_cycle, selected)


if __name__ == "__main__":
    main()

