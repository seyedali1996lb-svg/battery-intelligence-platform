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
# Page: Copilot
# ---------------------------------------------------------------------------

def page_copilot(
    cell_ids: list[str],
    featured_dfs: dict,
    bundles: dict,
    selected: str,
):
    _action_bar("copilot")
    from battery_copilot import (
        build_cell_context,
        build_fleet_stats,
        context_summary,
        answer_query,
        answer_health,
        answer_prediction_drivers,
        answer_rul,
        answer_compare,
        answer_anomaly,
        answer_recent_trajectory,
        answer_fleet_compare,
        answer_alerts,
        answer_replacement_budget,
        answer_fleet_risk,
        answer_business_case,
        QUERY_LABELS,
        FOLLOW_UP_MAP,
    )
    try:
        from battery_copilot import llm_answer as _llm_answer_fn
    except ImportError:
        def _llm_answer_fn(query, template_answer, api_key):  # type: ignore[misc]
            return template_answer
    llm_answer = _llm_answer_fn

    st.markdown("# Copilot")

    # ── Disclosure banner ──
    _md_html(
        """<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.18);border-radius:10px;padding:14px 20px;margin-bottom:24px;font-size:13px;color:#8896a8;line-height:1.6"><strong style="color:#63b3ed">Grounded narration only.</strong> Every sentence is derived from values already computed by the model pipeline — SOH, feature importances, per-cell RUL reliability, fade rates. The Copilot never calculates, estimates, or infers a value not already in the bundle. If a number is not there, it says so.</div>"""
    )

    query = st.session_state.get("copilot_query", None)
    # D3: track free-text query separately from chip selection
    _free_text_query = st.session_state.get("copilot_free_text", None)

    # ── Ask bar ────────────────────────────────────────────────────────────
    _ask_input = st.text_input(
        "Ask a question",
        value="",
        placeholder="Ask anything — e.g. 'Why is RUL uncertainty so wide?' or 'Which cells need attention?'",
        label_visibility="collapsed",
        key="copilot_ask_bar",
    )
    if _ask_input and _ask_input != st.session_state.get("_last_ask", ""):
        st.session_state["_last_ask"] = _ask_input
        _typed_lower = _ask_input.lower()
        _matched = next(
            (k for k, v in QUERY_LABELS.items() if _typed_lower in v.lower()),
            None,
        )
        if _matched:
            st.session_state.copilot_query = _matched
            st.session_state.pop("copilot_free_text", None)
        else:
            # D3: unmatched → free-text, route to LLM or template fallback
            st.session_state["copilot_free_text"] = _ask_input
            st.session_state.pop("copilot_query", None)
            query = None
        st.rerun()

    _free_text_query = st.session_state.get("copilot_free_text", None)

    # ── Chip grid — suggestions beneath the text input ─────────────────────
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.1em;padding:10px 0 4px'>Cell questions</div>",
        unsafe_allow_html=True,
    )
    _tech_keys = ["health", "drivers", "rul", "compare", "recent", "anomaly", "fleet_compare"]
    _trows = [_tech_keys[:4], _tech_keys[4:]]
    for _row_keys in _trows:
        _cols = st.columns(len(_row_keys))
        for _key, _col in zip(_row_keys, _cols):
            with _col:
                if st.button(
                    QUERY_LABELS[_key], key=f"cpq_{_key}",
                    use_container_width=True,
                    type="primary" if query == _key else "secondary",
                ):
                    st.session_state.copilot_query = _key
                    st.rerun()

    # Fleet & business questions
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.1em;padding:10px 0 4px'>Fleet &amp; business</div>",
        unsafe_allow_html=True,
    )
    _biz_keys = ["alerts", "replacement_budget", "fleet_risk", "business_case"]
    _bcols = st.columns(len(_biz_keys))
    for _key, _col in zip(_biz_keys, _bcols):
        with _col:
            if st.button(
                QUERY_LABELS[_key], key=f"cpq_{_key}",
                use_container_width=True,
                type="primary" if query == _key else "secondary",
            ):
                st.session_state.copilot_query = _key
                st.rerun()

    if not query and not _free_text_query:
        st.markdown(
            "<div style='text-align:center;padding:40px 24px;color:#4a5568;font-size:14px'>"
            "Type a question above or choose a topic. The Copilot answers using only "
            "values already computed by the model pipeline for "
            "<strong style='color:#8896a8'>" + selected + "</strong>.</div>",
            unsafe_allow_html=True,
        )
        return

    # D3: Free-text path — route to LLM (or template summary as fallback)
    if _free_text_query and not query:
        _api_key_ft = st.session_state.get("anthropic_api_key", "")
        _ctx_ft = build_cell_context(selected, featured_dfs, bundles)
        _template_ft = answer_query(_free_text_query, _ctx_ft)
        _llm_key_ft = "claude-haiku-4-5-20251001"
        st.markdown(
            f"<div style='font-size:11px;color:#4a5568;margin-bottom:8px'>"
            f"Answering: <em style='color:#8896a8'>{_free_text_query}</em>"
            + (" · Claude Haiku" if _api_key_ft else " · Template fallback")
            + "</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Thinking…"):
            try:
                _ft_answer = llm_answer(_free_text_query, _template_ft, _api_key_ft)
            except Exception as _e:
                _ft_answer = _template_ft
        _md_html(
            f"<div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;"
            f"padding:18px 22px;font-size:14px;color:#a0aec0;line-height:1.7'>{_ft_answer}</div>"
        )
        if st.button("Clear", key="cpft_clear"):
            st.session_state.pop("copilot_free_text", None)
            st.session_state.pop("_last_ask", None)
            st.rerun()
        return

    # ── Pre-compute fleet stats (cheap: just iterates already-computed DataFrames) ──
    fleet_stats = build_fleet_stats(featured_dfs, bundles)

    # ── Build context for the selected cell (not needed for fleet-only queries) ──
    fleet_only = query in ("alerts", "replacement_budget", "fleet_risk")
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
    elif query == "replacement_budget":
        response = answer_replacement_budget(fleet_stats)
        contexts = []
    elif query == "fleet_risk":
        response = answer_fleet_risk(fleet_stats)
        contexts = []
    elif query == "business_case":
        response = answer_business_case(ctx, fleet_stats)
        contexts = [ctx]
    else:
        response = f"Unknown query: {query}"
        contexts = []

    # ── LLM pass (if API key set) ────────────────────────────────────────────
    _llm_key_cp = st.session_state.get("anthropic_api_key", "")
    if _llm_key_cp:
        with st.spinner("Claude Haiku thinking…"):
            response = llm_answer(query, response, _llm_key_cp)

    # ── Response header ──
    cell_label = f" &nbsp;·&nbsp; {selected}" if not fleet_only else " &nbsp;·&nbsp; all cells"
    _llm_badge = (
        make_badge("Claude Haiku", "#9f7aea") + "&nbsp;" if _llm_key_cp else ""
    )
    st.markdown(
        f"<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:28px 0 16px;padding-bottom:8px;border-bottom:1px solid #2d3748'>"
        f"{QUERY_LABELS.get(query, '')}{cell_label}&nbsp;&nbsp;{_llm_badge}</div>",
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

    # ── Feature Attribution (Insights merged in) ────────────────────────────
    if not fleet_only and ctx is not None:
        with st.expander("Feature attribution — why does the model predict this?", expanded=False):
            try:
                _shap_bundle = bundles.get(
                    "nasa" if selected in NASA_CELL_IDS else
                    ("severson" if selected.startswith("S-") else "synth"),
                    list(bundles.values())[0],
                )
                _drivers = top_drivers(_shap_bundle, model="soh", top_n=8)
                _fi = feature_importance_df(_shap_bundle, model="soh", top_n=8)
                _top_feat = _drivers[0]["feature"] if _drivers else "—"
                _top_pct  = _drivers[0]["importance_pct"] if _drivers else 0
                st.markdown(
                    f"**Top predictor: {friendly(_top_feat)}** ({_top_pct:.0f}% split importance)"
                )
                _bar_fig = go.Figure(go.Bar(
                    x=_fi["importance_pct"], y=_fi["label"],
                    orientation="h",
                    marker=dict(
                        color=_fi["importance_pct"],
                        colorscale=[[0, "#1e2a38"], [0.4, "#1e2a38"], [1, "#63b3ed"]],
                        showscale=False,
                    ),
                    text=_fi["importance_pct"].apply(lambda v: f"{v:.1f}%"),
                    textposition="inside", insidetextanchor="end",
                    textfont=dict(color="#ffffff", size=11),
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
                ))
                _bar_fig.update_layout(
                    height=300,
                    **base_layout(
                        xaxis=dict(title="% importance", zeroline=False),
                        yaxis=dict(autorange="reversed"),
                    ),
                )
                st.plotly_chart(_bar_fig, use_container_width=True)
                st.caption(
                    "Split importance from GradientBoostingRegressor. For correlated features "
                    "(fade_rate_10/30/50cy), SHAP values give a more accurate attribution — "
                    "available when the model bundle includes SHAP data."
                )
            except Exception as _si_e:
                st.info(f"Feature attribution unavailable: {_si_e}")

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
# Page: Decision  (merged Recommendations + EOL Economics)
# ---------------------------------------------------------------------------

def page_decision(
    selected: str,
    df: pd.DataFrame,
    featured_dfs: dict,
    bundles: dict,
    rul_reliable: bool,
):
    """Merged Recommendations + EOL Economics page.

    Layout:
      1. Hero recommendation card (action + confidence)
      2. NPV 3-column decision table (Replace / Wait / Repurpose)
      3. Maintenance calendar
      4. Application fit scores
      5. Log decision + SoC window
      Full economics in expander below.
    """
    _action_bar("decision")
    from recommendations import classify, SOH_PRIMARY_FLOOR, SOH_INSPECT_FLOOR, SOH_SECONDLIFE_FLOOR
    from consequences import (
        ASSUMPTIONS, SECOND_LIFE_APPS, CELL_NOMINAL_KWH,
        application_fit, financial_comparison, sustainability_snapshot, breakeven_curve,
    )

    latest          = df.iloc[-1]
    soh             = float(latest["soh_pct"])
    fade_30         = float(latest.get("fade_rate_30cy", 0.0))
    fade_50         = float(latest.get("fade_rate_50cy", 0.0))
    is_nasa         = selected in NASA_CELL_IDS
    source          = "nasa" if is_nasa else "synth"
    rul_pred_raw    = latest.get("rul_pred", None)
    rul_pred        = float(rul_pred_raw) if (rul_reliable and rul_pred_raw is not None) else None

    peer_fades = [
        float(fdf.iloc[-1].get("fade_rate_30cy", 0))
        for cid, fdf in featured_dfs.items()
        if (cid in NASA_CELL_IDS) == is_nasa and cid != selected
    ]
    fleet_fade_median = float(pd.Series(peer_fades).median()) if peer_fades else None
    fit_scores = application_fit(soh, fade_30, fleet_fade_median)
    result     = classify(soh, fade_30, fade_50, rul_reliable, rul_pred, fit_scores)

    action          = result["action"]
    action_label, action_colour, action_bg = ACTION_META[action]
    conf_colour, conf_label = CONF_META[result["confidence"]]

    st.markdown(f"# What should I do with {selected}?")

    # ── 1. Hero recommendation ──────────────────────────────────────────────
    reason_html = "".join(
        f"<div style='margin-top:6px;font-size:13px;color:{action_colour}cc'>· {r}</div>"
        for r in result["action_reasons"]
    )
    _md_html(
        f"<div style='background:{action_bg};border:2px solid {action_colour}55;"
        f"border-radius:14px;padding:24px 28px;margin-bottom:20px'>"
        f"<div style='font-size:10px;font-weight:700;color:{action_colour}99;"
        f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Recommended Action</div>"
        f"<div style='font-size:32px;font-weight:800;color:{action_colour}'>{action_label}</div>"
        f"<div style='margin-top:8px'>"
        f"<span style='background:{conf_colour}22;border:1px solid {conf_colour}55;color:{conf_colour};"
        f"font-size:11px;font-weight:700;padding:3px 10px;border-radius:10px'>{conf_label}</span>"
        f"</div>{reason_html}</div>"
    )

    # ── Compact mechanism verdict (U3: merged from Health page's LLI/LAM classifier) ──
    try:
        from recommendations import diagnose_mechanism
        _dec_mech = diagnose_mechanism(df)
        st.markdown(
            f"<div style='background:#111827;border-left:3px solid {_dec_mech['verdict_color']};"
            f"border-radius:6px;padding:8px 14px;margin:-12px 0 20px;font-size:12px;color:#a0aec0'>"
            f"⚗️ <strong style='color:{_dec_mech['verdict_color']}'>{_dec_mech['verdict']}</strong>"
            f" <span style='color:#4a5568'>({_dec_mech['confidence_label']} confidence)</span>"
            f" — {_dec_mech['verdict_body'].split('.')[0]}."
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # ── 2. NPV Decision Table (3 options, no chart) ─────────────────────────
    st.markdown("<div class='section-header'>Financial Decision</div>", unsafe_allow_html=True)
    _npv_rate   = 0.08
    _energy_usd = 80.0
    _repl_cost  = 150.0
    _years      = list(range(1, 6))
    def _pv(r, t): return 1.0 / ((1 + r) ** t)
    _rul_yrs = min((rul_pred / 200.0 / 12.0) if rul_pred else 1.5, 5.0)

    _cap_now  = CELL_NOMINAL_KWH.get(source, 0.0057)
    _a_npv = sum(_cap_now * _energy_usd * _pv(_npv_rate, t) for t in _years) - _repl_cost

    _b_cap = _cap_now * (soh / 100.0)
    _b_npv = (
        sum(_b_cap * _energy_usd * _pv(_npv_rate, t) for t in range(1, max(1, int(_rul_yrs)) + 1))
        + sum(_cap_now * _energy_usd * _pv(_npv_rate, t) for t in range(max(1, int(_rul_yrs)) + 1, 6))
        - _repl_cost * _pv(_npv_rate, _rul_yrs)
    )

    _sl_cap   = _cap_now * (soh / 100.0) * 0.85
    _repack   = 30.0
    _c_npv    = sum(_sl_cap * _energy_usd * 0.6 * _pv(_npv_rate, t) for t in _years) - _repack

    _opts = [
        ("Replace Now",          _a_npv, "#fc8181",
         "Immediate outlay. New cell, full life ahead.", f"${_repl_cost:.0f} upfront"),
        ("Wait to EOL",          _b_npv, "#f6ad55",
         f"~{_rul_yrs:.1f} yr remaining at current fade rate.", "Risk of degraded performance"),
        ("Repurpose (2nd life)", _c_npv, "#68d391",
         "Lower-demand application (stationary storage).", f"${_repack:.0f} repack cost"),
    ]
    _best_npv = max(_opts, key=lambda x: x[1])
    _best_lbl, _best_npv_v, _best_col_v, _best_desc, _best_cost_note = _best_npv
    _md_html(
        f"<div style='background:#1e2a38;border:2px solid {_best_col_v};"
        f"border-radius:12px;padding:20px 24px;margin-bottom:10px'>"
        f"<div style='font-size:10px;font-weight:700;color:{_best_col_v};"
        f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px'>★ Recommended</div>"
        f"<div style='font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:8px'>{_best_lbl}</div>"
        f"<div style='font-size:40px;font-weight:800;color:{_best_col_v}'>${_best_npv_v:,.0f}</div>"
        f"<div style='font-size:10px;color:#718096;margin-top:2px'>5-yr NPV at 8% discount rate</div>"
        f"<div style='font-size:12px;color:#a0aec0;margin-top:10px;line-height:1.5'>{_best_desc}</div>"
        f"<div style='font-size:11px;color:#4a5568;margin-top:4px'>{_best_cost_note}</div>"
        f"</div>"
    )
    with st.expander("Compare alternatives"):
        _alts = [o for o in _opts if o[0] != _best_lbl]
        _nd_cols = st.columns(len(_alts))
        for _col, (_lbl, _npv_v, _col_v, _desc, _cost_note) in zip(_nd_cols, _alts):
            with _col:
                _md_html(
                    f"<div style='background:#1e2a38;border:1px solid #2d3748;"
                    f"border-radius:10px;padding:16px 18px;height:100%'>"
                    f"<div style='font-size:10px;font-weight:700;color:#4a5568;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin-bottom:6px'>{_lbl}</div>"
                    f"<div style='font-size:26px;font-weight:800;color:{_col_v}'>${_npv_v:,.0f}</div>"
                    f"<div style='font-size:10px;color:#718096;margin-top:2px'>5-yr NPV</div>"
                    f"<div style='font-size:11px;color:#8896a8;margin-top:8px;line-height:1.5'>{_desc}</div>"
                    f"<div style='font-size:10px;color:#4a5568;margin-top:4px'>{_cost_note}</div>"
                    f"</div>"
                )

    with st.popover("ⓘ Assumptions", use_container_width=False):
        st.markdown(
            f"<div style='font-size:12px;color:#a0aec0;line-height:1.8'>"
            f"<strong style='color:#e2e8f0'>5-yr NPV · {_npv_rate*100:.0f}% discount rate</strong><br>"
            f"WACC assumption — adjust in the NPV Scenario Planner below.<br><br>"
            f"<strong style='color:#e2e8f0'>$80/kWh·yr energy value</strong> "
            f"{make_badge('Illustrative', '#718096')}<br>"
            f"IEA 2024 LCOS range $60–140/kWh·yr.<br><br>"
            f"<strong style='color:#e2e8f0'>${_repl_cost:.0f}/cell replacement</strong> "
            f"{make_badge('Illustrative', '#718096')}<br>"
            f"BNEF 2024 range $100–200/cell.<br><br>"
            f"<strong style='color:#e2e8f0'>${_repack:.0f} repack cost</strong> "
            f"{make_badge('Illustrative', '#718096')}<br>"
            f"Engineering estimate.<br><br>"
            f"<em style='color:#4a5568'>Not financial advice.</em>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── 📋 Supporting details (U2 density reduction): maintenance calendar, ──
    # application fit, second-life marketplace, confidence-reason callouts ──
    with st.expander("📋 Supporting details", expanded=False):
        # ── 3. Maintenance Calendar ─────────────────────────────────────────
        st.markdown("<div class='section-header'>Maintenance Calendar</div>", unsafe_allow_html=True)
        _eol_thr    = float(st.session_state.get("eol_threshold_pct", 80.0))
        _cur_soh    = float(df["soh_pct"].iloc[-1])
        _fp_cy      = float(df["fade_rate_50cy"].iloc[-1]) * 100 / (float(df["capacity_ah"].iloc[0]) + 1e-9)
        _rul_cy     = max(0, (_cur_soh - _eol_thr) / _fp_cy) if _fp_cy > 1e-6 else None

        if "test_date" in df.columns and df["test_date"].notna().any():
            _dates = pd.to_datetime(df["test_date"].dropna())
            _cpd   = len(df) / max((_dates.iloc[-1] - _dates.iloc[0]).days, 1)
        else:
            _cpd   = 1.0

        if _rul_cy is not None and _rul_cy > 0:
            from datetime import date as _date, timedelta as _td
            _days    = _rul_cy / _cpd
            _rep_dt  = _date.today() + _td(days=_days)
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Recommended Replacement", _rep_dt.strftime("%B %Y"))
            _c2.metric("Cycles Remaining", f"{_rul_cy:.0f}")
            _c3.metric("Days Remaining",   f"{_days:.0f}")
        else:
            _empty_state(
                "Replacement timeline unavailable",
                "Insufficient cycle history to compute a fade-based schedule. "
                "At least 50 cycles with a measurable fade trend are required.",
                "", "📅",
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── 4. Application Fit ──────────────────────────────────────────────
        if soh <= 90:
            st.markdown("<div class='section-header'>Application Fit Scores</div>", unsafe_allow_html=True)
            _af_cols = st.columns(min(len(fit_scores), 4))
            _af_colour = {"fit": "#48bb78", "marginal": "#f6ad55", "not_fit": "#fc8181"}
            for _i, (_app_key, _app) in enumerate(fit_scores.items()):
                with _af_cols[_i % len(_af_cols)]:
                    st.markdown(
                        f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                        f"padding:14px 16px'>"
                        f"<div style='font-size:11px;color:#4a5568'>{_app['short']}</div>"
                        f"<div style='font-size:16px;font-weight:700;color:{_af_colour[_app['fit']]};margin-top:4px'>"
                        f"{_app['fit'].replace('_', ' ').title()}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── 4b. Second-life marketplace ──────────────────────────────────────
        if action in ("second_life", "recycle") or soh <= 85:
            st.markdown("<div class='section-header'>Second-Life Marketplace</div>", unsafe_allow_html=True)
            _sl_eligible = soh >= 70.0
            _sl_col = "#68d391" if _sl_eligible else "#718096"
            _md_html(
                f"<div style='background:#1e2a38;border:1px solid {_sl_col}44;border-radius:10px;"
                f"padding:18px 22px;margin-bottom:12px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
                f"<div>"
                f"<div style='font-size:14px;font-weight:700;color:{_sl_col}'>"
                f"{'Eligible for second-life listing' if _sl_eligible else 'Below second-life floor (SOH < 70%)'}</div>"
                f"<div style='font-size:12px;color:#8896a8;margin-top:6px;line-height:1.6'>"
                f"Cell {selected} · SOH {soh:.1f}% · Est. residual capacity {soh * _cap_now / 100 * 1000:.0f} mAh<br>"
                f"Suitable for: stationary storage, UPS backup, low-power IoT applications"
                f"</div>"
                f"</div>"
                f"<div style='font-size:22px;color:{_sl_col};margin-left:16px'>♻</div>"
                f"</div>"
                f"</div>"
            )
            if _sl_eligible:
                _mk_col1, _mk_col2 = st.columns(2)
                if _mk_col1.button(
                    "List on Circunomics →", key="sl_list_circ",
                    help="Opens second-life battery exchange (demo — not a live API call)",
                    use_container_width=True,
                ):
                    _circ_chemistry = "LFP" if selected.startswith("S-") else "LiCoO2"
                    _circ_capacity_ah = round(_cap_now * (soh / 100) * 1000 / 3.7, 2)
                    _circ_asking_usd = round(_c_npv * 0.4, 2)
                    _circ_api_key = st.session_state.get("circunomics_api_key", "")
                    _circ_result = None
                    if _circ_api_key:
                        from circunomics_adapter import list_cell_on_circunomics
                        _circ_result = list_cell_on_circunomics(
                            selected, soh, _circ_chemistry, _circ_capacity_ah,
                            _circ_asking_usd, _circ_api_key,
                        )
                    if _circ_result is not None and "error" not in _circ_result:
                        st.success(
                            f"Listing submitted to Circunomics for {selected} at "
                            f"${_circ_asking_usd:.2f}."
                        )
                    else:
                        _listing = {
                            "cell_id":        selected,
                            "soh_pct":        round(soh, 1),
                            "chemistry":      _circ_chemistry,
                            "capacity_ah":    _circ_capacity_ah,
                            "asking_usd":     _circ_asking_usd,
                            "listed_at":      datetime.datetime.now().isoformat(),
                            "platform":       "battery-intelligence-platform",
                            "note":           "Demo listing — not submitted to a live exchange",
                        }
                        if "sl_listings" not in st.session_state:
                            st.session_state["sl_listings"] = []
                        st.session_state["sl_listings"].append(_listing)
                        if _circ_result is not None:
                            st.error(f"Circunomics submission failed ({_circ_result['error']}) — saved as a local demo listing instead.")
                        else:
                            st.success(
                                f"Listing created for {selected} at ${_listing['asking_usd']:.2f} "
                                f"(demo — no real API call made). Configure a Circunomics API key in "
                                f"Settings to submit real listings."
                            )
                if _mk_col2.button(
                    "List on Battery-Lifecycle.com →", key="sl_list_blc",
                    help="Second-life exchange for industrial battery packs (demo)",
                    use_container_width=True,
                ):
                    st.info(
                        "Battery Lifecycle Company integration requires an API key. "
                        "In production: POST /api/v1/listings with cell ID, SOH, chemistry, "
                        "capacity, and asking price. See Settings to configure the endpoint."
                    )
            _sl_listings = st.session_state.get("sl_listings", [])
            if _sl_listings:
                with st.expander(f"Active listings ({len(_sl_listings)})"):
                    import pandas as _pd_sl
                    st.dataframe(
                        _pd_sl.DataFrame(_sl_listings),
                        use_container_width=True, hide_index=True,
                    )
                    st.download_button(
                        "Export listings",
                        data=_pd_sl.DataFrame(_sl_listings).to_csv(index=False).encode(),
                        file_name="secondlife_listings.csv",
                        mime="text/csv",
                        key="sl_listings_export",
                    )

        # ── Confidence-reason callouts ────────────────────────────────────────
        if result["confidence_reasons"]:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for note in result["confidence_reasons"]:
                note_colour = "#b7791f" if "fit scores" in note else "#718096"
                st.markdown(
                    f"<div style='background:{note_colour}11;border:1px solid {note_colour}33;"
                    f"border-radius:8px;padding:10px 16px;margin-bottom:8px;"
                    f"font-size:12px;color:#a0aec0'>{note}</div>",
                    unsafe_allow_html=True,
                )

    # ── 5. Log Decision ─────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _log_col, _cmms_col, _ = st.columns([1, 1, 2])
    if _log_col.button("Log Decision", key="dec_log_btn", use_container_width=True):
        import db as _db
        _entry = {
            "id":         f"{selected}_{datetime.datetime.now().strftime('%H%M%S')}",
            "cell_id":    selected,
            "action":     action_label,
            "confidence": conf_label,
            "soh_pct":    round(soh, 1),
            "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status":     "Pending",
            "outcome_soh": None,
        }
        if "decision_log" not in st.session_state:
            st.session_state["decision_log"] = []
        st.session_state["decision_log"].append(_entry)
        _db.save_decision(st.session_state["auth_org_id"], _entry)
        st.success(f"Logged: {action_label} for {selected}")

    # ── U4: CMMS/ERP write-back ──────────────────────────────────────────────
    _cmms_api_key = st.session_state.get("cmms_api_key", "")
    if _cmms_col.button(
        "Create CMMS ticket", key="dec_cmms_btn", use_container_width=True,
        disabled=not _cmms_api_key,
        help="Create a maintenance ticket in your CMMS/ERP system for this recommendation. "
             "Configure a CMMS API key in Settings to enable this."
             if not _cmms_api_key else
             "Creates a maintenance ticket in your configured CMMS/ERP system.",
    ):
        from cmms_adapter import create_maintenance_ticket
        _cmms_base_url = st.session_state.get("cmms_api_base_url", "") or "https://api.example-cmms.com/v1"
        _cmms_title = f"{action_label} — {selected}"
        _cmms_desc = (
            f"Recommended action: {action_label} (confidence: {conf_label}). "
            f"SOH: {soh:.1f}%. " + " ".join(result["action_reasons"])
        )
        _cmms_priority = "high" if action in ("recycle",) else "medium" if action == "second_life" else "low"
        _cmms_result = create_maintenance_ticket(
            selected, _cmms_title, _cmms_desc, _cmms_priority, _cmms_api_key,
            api_base_url=_cmms_base_url,
        )
        if _cmms_result is None:
            st.warning("No CMMS API key configured — set one up in Settings first.")
        elif "error" in _cmms_result:
            st.error(f"CMMS ticket creation failed: {_cmms_result['error']}")
        else:
            st.success(f"CMMS ticket created for {selected}.")

    # E4: Audit trail with status chips and outcome tracking
    _dlog = st.session_state.get("decision_log", [])
    if _dlog:
        with st.expander(f"Decision Audit Trail ({len(_dlog)} entries)", expanded=False):
            _STATUS_COLOURS = {
                "Pending": "#718096", "Approved": "#63b3ed",
                "Completed": "#48bb78", "Verified": "#9f7aea",
            }
            for _i, _dl in enumerate(_dlog):
                _sc   = _STATUS_COLOURS.get(_dl.get("status", "Pending"), "#718096")
                _sc22 = _sc + "22"
                _cols_e4 = st.columns([3, 2, 2, 2, 1])
                _cols_e4[0].markdown(
                    f"<div style='padding:6px 0'>"
                    f"<div style='font-size:13px;font-weight:600;color:#e2e8f0'>"
                    f"{_dl['cell_id']} — {_dl['action']}</div>"
                    f"<div style='font-size:11px;color:#4a5568'>{_dl['timestamp']} · SOH {_dl['soh_pct']}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                _new_status = _cols_e4[1].selectbox(
                    "Status", options=["Pending", "Approved", "Completed", "Verified"],
                    index=["Pending", "Approved", "Completed", "Verified"].index(
                        _dl.get("status", "Pending")),
                    key=f"e4_status_{_i}",
                    label_visibility="collapsed",
                )
                if _new_status != _dl.get("status"):
                    st.session_state["decision_log"][_i]["status"] = _new_status
                    import db as _db
                    _db.update_decision(st.session_state["auth_org_id"], _dl["id"], status=_new_status)
                    st.rerun()
                _cols_e4[2].markdown(
                    f"<div style='padding:8px 0'>"
                    f"<span style='background:{_sc22};border:1px solid {_sc}44;color:{_sc};"
                    f"font-size:11px;font-weight:700;padding:3px 10px;border-radius:10px'>"
                    f"{_dl.get('status', 'Pending')}</span></div>",
                    unsafe_allow_html=True,
                )
                # Outcome tracking: prompt for SOH when marked Completed
                if _dl.get("status") == "Completed" and _dl.get("outcome_soh") is None:
                    _new_soh = _cols_e4[3].number_input(
                        "Actual SOH after action (%)", min_value=0.0, max_value=100.0,
                        value=float(_dl["soh_pct"]), step=0.5,
                        key=f"e4_soh_{_i}", label_visibility="collapsed",
                    )
                    if _cols_e4[4].button("Save", key=f"e4_save_{_i}"):
                        st.session_state["decision_log"][_i]["outcome_soh"] = round(_new_soh, 1)
                        st.session_state["decision_log"][_i]["status"] = "Verified"
                        import db as _db
                        _db.update_decision(st.session_state["auth_org_id"], _dl["id"], outcome_soh=round(_new_soh, 1), status="Verified")
                        st.rerun()
                elif _dl.get("outcome_soh") is not None:
                    _delta_soh = _dl["outcome_soh"] - _dl["soh_pct"]
                    _delta_col = "#48bb78" if _delta_soh >= 0 else "#fc8181"
                    _cols_e4[3].markdown(
                        f"<div style='padding:8px 0;font-size:12px;color:{_delta_col}'>"
                        f"Outcome: {_dl['outcome_soh']}% "
                        f"({_delta_soh:+.1f}%)</div>",
                        unsafe_allow_html=True,
                    )
            import pandas as _pd_log
            st.download_button(
                "Export log as CSV",
                data=_pd_log.DataFrame(_dlog).to_csv(index=False).encode(),
                file_name="battery_decision_log.csv",
                mime="text/csv",
                key="dec_export_log",
            )

    # ── Full Economics (expander) ────────────────────────────────────────────
    with st.expander("Full economics & breakeven analysis", expanded=False):
        page_consequences(selected, df, featured_dfs, bundles, rul_reliable)

    # ── Inline Copilot panel (merged Decision + Copilot) ─────────────────────
    st.markdown("<div class='section-header'>Ask about this cell</div>", unsafe_allow_html=True)
    if selected in NASA_CELL_IDS:
        _dc_bundle = bundles.get("nasa")
    elif selected.startswith("S-"):
        _dc_bundle = bundles.get("severson")
    elif selected in CELL_STRESS_PROFILES:
        _dc_bundle = bundles.get("synth")
    else:
        _dc_bundle = bundles.get("upload")
    if _dc_bundle:
        try:
            from battery_copilot import build_cell_context, answer_query
            from copilot_llm import llm_answer
            _dc_ctx = build_cell_context(selected, featured_dfs, bundles)
            _dc_input = st.text_input(
                "Question", placeholder="e.g. 'Why is this cell degrading faster than others?'",
                key="decision_copilot_input", label_visibility="collapsed",
            )
            _dc_chips_row = st.columns(4)
            _dc_presets = [
                ("What should I do?",   "what_to_do"),
                ("Replacement budget?", "budget"),
                ("What's the risk?",    "risk"),
                ("Repurpose options?",  "repurpose"),
            ]
            for _dci, (_dclbl, _dckey) in enumerate(_dc_presets):
                if _dc_chips_row[_dci].button(_dclbl, key=f"dc_chip_{_dckey}", use_container_width=True):
                    st.session_state["decision_copilot_input"] = _dclbl
                    st.rerun()
            if _dc_input:
                _api_key_dc = st.session_state.get("anthropic_api_key", "")
                _template_dc = answer_query(_dc_input, _dc_ctx)
                _dc_answer = llm_answer(_dc_input, _template_dc, _api_key_dc) if _api_key_dc else _template_dc
                _badge_dc = make_badge("Claude Haiku", "#667eea") if _api_key_dc else make_badge("Template", "#718096")
                _md_html(
                    f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                    f"padding:16px 20px;margin-top:8px'>"
                    f"<div style='font-size:10px;color:#4a5568;margin-bottom:8px'>{_badge_dc} · {selected}</div>"
                    f"<div style='font-size:13px;color:#e2e8f0;line-height:1.7'>{_dc_answer}</div>"
                    f"</div>"
                )
        except Exception as _dc_e:
            st.caption(f"Copilot unavailable: {_dc_e}")


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


def _passport_field_row(f: dict) -> str:
    muted = f["state"] == "unavailable"
    value_colour = "#4a5568" if muted else "#e2e8f0"
    note_html = (
        f"<div style='font-size:11px;color:#4a5568;margin-top:3px;line-height:1.5'>{f['note']}</div>"
        if f.get("note") else "<div style='height:0'></div>"
    )
    font_style = "italic" if muted else "normal"
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        f"gap:16px;padding:12px 0;border-bottom:1px solid #2d3748'>"
        f"<div style='flex:1;min-width:0'>"
        f"<div style='font-size:12px;color:#8896a8'>{f['label']}</div>"
        f"<div style='font-size:14px;font-weight:600;color:{value_colour};margin-top:2px;"
        f"font-style:{font_style}'>{f['value']}</div>"
        f"{note_html}"
        f"</div>"
        f"<div style='flex-shrink:0;padding-top:2px'>{make_state_badge(f['state'])}</div>"
        f"</div>"
    )


def page_passport(selected: str, df: pd.DataFrame, bundle: dict, rul_reliable: bool):
    from passport import build_passport

    is_nasa = selected in NASA_CELL_IDS
    p = build_passport(selected, df, bundle, rul_reliable, is_nasa)
    summ = p["summary"]

    st.markdown("# Battery Passport")
    st.markdown(f"##### Battery Passport Interface · {selected}")

    # ── E8: Passport completeness score ───────────────────────────────────────
    _e8_n_avail = summ.get("n_available", 0)
    _e8_n_est   = summ.get("n_estimated", 0)
    _e8_n_unavail = summ.get("n_unavailable", 0)
    _e8_total   = max(1, summ.get("n_total", _e8_n_avail + _e8_n_est + _e8_n_unavail))
    _e8_score   = round((_e8_n_avail + _e8_n_est * 0.5) / _e8_total * 100)

    # Proactive webhook push — once per cell per session, only below threshold.
    _wh_url_pg  = st.session_state.get("webhook_url", "")
    _wh_evts_pg = st.session_state.get("webhook_events", [])
    _PASSPORT_GAP_THRESHOLD = 60
    if _wh_url_pg and "PASSPORT_GAP" in _wh_evts_pg and _e8_score < _PASSPORT_GAP_THRESHOLD:
        if "_alerted_passport_gap_cells" not in st.session_state:
            st.session_state["_alerted_passport_gap_cells"] = set()
        if selected not in st.session_state["_alerted_passport_gap_cells"]:
            from notifications import send_webhook
            send_webhook(
                "PASSPORT_GAP",
                {
                    "cell_id": selected, "completeness_pct": _e8_score,
                    "n_available": _e8_n_avail, "n_estimated": _e8_n_est,
                    "n_unavailable": _e8_n_unavail,
                },
                _wh_url_pg, st.session_state.get("webhook_secret", ""),
            )
            st.session_state["_alerted_passport_gap_cells"].add(selected)

    _e8_col1, _e8_col2 = st.columns([3, 1])
    with _e8_col1:
        _e8_bar_colour = "#48bb78" if _e8_score >= 70 else ("#f6ad55" if _e8_score >= 40 else "#fc8181")
        st.markdown(
            f"<div style='margin-bottom:4px;font-size:12px;color:#a0aec0'>"
            f"Passport completeness: <strong style='color:{_e8_bar_colour}'>{_e8_score}%</strong> "
            f"&nbsp;·&nbsp; {_e8_n_avail} available · {_e8_n_est} estimated · {_e8_n_unavail} unavailable</div>",
            unsafe_allow_html=True,
        )
        st.progress(_e8_score / 100)
    with _e8_col2:
        if _e8_n_unavail > 0:
            with st.expander(f"Fill {_e8_n_unavail} gaps", expanded=False):
                _E8_HOW_TO: dict[str, str] = {
                    "manufacturer": "Set in sidebar → Cell Metadata → Manufacturer",
                    "chemistry":    "Detected automatically once enough cycle data is loaded",
                    "carbon_footprint_kg_co2": "Upload manufacturer LCA document in Import page",
                    "recycled_content_pct":    "Enter on Import page → Advanced → Recycled content",
                    "supply_chain_due_diligence": "Link supplier declaration PDF in Import page",
                    "hazardous_substances":    "Set on Import page → Advanced → Hazardous substances",
                    "end_of_life_instructions": "Auto-populated once chemistry is confirmed",
                    "second_life_status":       "Set on Decision page after repurpose decision is logged",
                }
                _all_fields: list[dict] = []
                for _grp_key in ("identity", "soh", "lifecycle", "carbon"):
                    _all_fields.extend(p.get(_grp_key, []))
                _missing = [f for f in _all_fields if f.get("state") == "unavailable"]
                for _mf in _missing:
                    _fname  = _mf.get("label", _mf.get("field", "Unknown field"))
                    _tip    = _E8_HOW_TO.get(str(_mf.get("field", "")), "Provide this data on the Import page")
                    st.markdown(
                        f"<div style='font-size:11px;padding:4px 0;border-bottom:1px solid #2d3748;'>"
                        f"<span style='color:#fc8181'>●</span> <strong style='color:#e2e8f0'>{_fname}</strong>"
                        f"<br><span style='color:#718096;font-size:10px'>How to fill: {_tip}</span></div>",
                        unsafe_allow_html=True,
                    )
    st.markdown("")

    _md_html(
        f"""
        <div style="background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.25);
                    border-radius:10px;padding:14px 20px;margin-bottom:28px;
                    font-size:13px;color:#8896a8;line-height:1.7">
            <strong style="color:#63b3ed">Battery Passport Interface</strong> — demonstrating the
            EU Battery Regulation (EU) 2023/1542 data structure. This is <strong>not</strong> a
            compliance claim: every field below is marked {make_state_badge("available")},
            {make_state_badge("estimated")}, or {make_state_badge("unavailable")} based on what this
            demonstration actually has. Nothing is hidden or faked to look complete.
        </div>
        """
    )

    _show_unavail = st.checkbox(
        "Show unavailable fields", key="passport_show_unavail", value=False
    )

    groups = [
        ("identity",  "1 · Battery Identity"),
        ("soh",       "2 · State of Health"),
        ("lifecycle", "3 · Lifecycle History"),
        ("carbon",    "4 · Carbon Footprint"),
    ]
    for key, title in groups:
        _fields = p[key] if _show_unavail else [f for f in p[key] if f.get("state") != "unavailable"]
        if not _fields:
            continue
        st.markdown(
            f"<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
            f"letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
            f"margin-bottom:4px;margin-top:20px'>{title}</div>",
            unsafe_allow_html=True,
        )
        rows_html = "".join(_passport_field_row(f) for f in _fields)
        st.markdown(f"<div>{rows_html}</div>", unsafe_allow_html=True)

    # ── 5: Critical Raw Materials (EU Battery Regulation Art. 13) ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:4px;margin-top:20px'>5 · Critical Raw Materials (EU Reg. 2023/1542 Art. 13)</div>",
        unsafe_allow_html=True,
    )
    _crm_settings = {k: st.session_state.get(k) for k in [
        "crm_lfp_li_pct", "crm_nca_co_pct", "crm_nca_ni_pct", "crm_nca_li_pct",
        "crm_nca_recycled_co_pct", "crm_nca_recycled_ni_pct",
        "crm_synth_co_pct", "crm_synth_ni_pct", "crm_synth_li_pct",
        "crm_user_co_pct", "crm_user_ni_pct", "crm_user_li_pct",
        "crm_user_recycled_co_pct", "crm_user_recycled_ni_pct",
    ]}
    _crm_settings_clean = {k: v for k, v in _crm_settings.items() if v is not None}
    from chemistry_profiles import ChemistryProfile as _ChemProfile  # noqa: E402
    _crm_fields = _ChemProfile.for_cell(selected).get_crm_fields(_crm_settings_clean)
    _crm_visible = _crm_fields if _show_unavail else [f for f in _crm_fields if f.get("state") != "unavailable"]
    _crm_html = "".join(_passport_field_row(f) for f in _crm_visible)
    st.markdown(f"<div>{_crm_html}</div>", unsafe_allow_html=True)

    # ── 6: End-of-Life R-code Taxonomy ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:4px;margin-top:20px'>6 · End-of-Life R-Code Taxonomy (IEC 62902 / EU Art. 70)</div>",
        unsafe_allow_html=True,
    )
    _soh_now = float(df.iloc[-1]["soh_pct"]) if "soh_pct" in df.columns else 85.0
    _eol_r_code = (
        "R3 — Remanufacture / Second-life application" if _soh_now >= 80
        else "R4 — Recycle (hydrometallurgical / direct)" if _soh_now >= 60
        else "R5 — Recover (energy or material)"
    )
    _eol_color  = "#48bb78" if _soh_now >= 80 else "#f6ad55" if _soh_now >= 60 else "#fc8181"
    _r_fields = [
        {"label": "Recommended R-code", "value": _eol_r_code, "state": "estimated",
         "note": f"Based on current SOH = {_soh_now:.1f}%. IEC 62902 R0–R9 taxonomy."},
        {"label": "R0 — Reuse", "value": "SOH ≥ 90% required", "state": "estimated" if _soh_now >= 90 else "unavailable"},
        {"label": "R3 — Second-life", "value": "SOH 80–90% (stationary storage)", "state": "available" if 80 <= _soh_now < 90 else "estimated"},
        {"label": "R4 — Recycle", "value": "Hydromet / direct recycling pathway", "state": "estimated"},
        {"label": "Recycled content declaration", "value": "IEC 63338 audit required", "state": "unavailable",
         "note": "Carbon footprint per kWh must be declared for market access (IEC 63338, from 2025)."},
    ]
    _r_visible = _r_fields if _show_unavail else [f for f in _r_fields if f.get("state") != "unavailable"]
    _r_html = "".join(_passport_field_row(f) for f in _r_visible)
    st.markdown(f"<div style='margin-bottom:4px'><span style='color:{_eol_color};font-weight:700;font-size:13px'>Recommended: {_eol_r_code}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div>{_r_html}</div>", unsafe_allow_html=True)

    # ── 7: Compliance Status (prose, no badge) ──
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:12px;margin-top:20px'>7 · Compliance Status</div>",
        unsafe_allow_html=True,
    )
    _md_html(
        f"""
        <div style="background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:20px 24px;
                    font-size:13px;color:#a0aec0;line-height:1.8">
            <strong style="color:#e2e8f0">This is a data-structure demonstration, not a regulatory
            submission.</strong><br><br>
            Of {summ['n_total']} fields modelled on the EU Battery Regulation's data requirements:
            <strong style="color:#48bb78">{summ['n_available']} are available</strong> from this
            platform's validated pipeline, <strong style="color:#d69e2e">{summ['n_estimated']} are
            cited estimates</strong> from the Consequences module, and
            <strong style="color:#8896a8">{summ['n_unavailable']} are not available</strong> in
            this demonstration.<br><br>
            An actual regulatory submission under (EU) 2023/1542 would additionally require:
            manufacturer-submitted identity and supply-chain records, a third-party accredited
            carbon footprint audit, repair/refurbishment history tracking, and notified-body
            sign-off — none of which a portfolio project can provide. No field on this page should
            be read as a compliance claim.
        </div>
        """
    )


    # ── E3: QR Code generator ─────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:12px;margin-top:24px'>8 · Physical Deployment — QR Code</div>",
        unsafe_allow_html=True,
    )
    _qr_api_base = st.text_input(
        "API base URL (for QR code)",
        value="http://localhost:8000",
        key="passport_qr_base",
        help="Points to your FastAPI deployment. The QR encodes /cells/{cell_id}.",
    )
    if st.button("Generate QR Code", key="passport_qr_btn"):
        try:
            import qrcode as _qrlib
            import io as _io
            _qr_url = f"{_qr_api_base.rstrip('/')}/cells/{selected}"
            _qr = _qrlib.QRCode(error_correction=_qrlib.constants.ERROR_CORRECT_M, box_size=6, border=3)
            _qr.add_data(_qr_url)
            _qr.make(fit=True)
            _qr_img = _qr.make_image(fill_color="white", back_color="#0e1117")
            _qr_buf = _io.BytesIO()
            _qr_img.save(_qr_buf, format="PNG")
            _qr_buf.seek(0)
            st.image(_qr_buf, caption=f"Scan to open live passport for {selected}", width=200)
            st.download_button(
                "Download QR PNG", data=_qr_buf.getvalue(),
                file_name=f"passport_qr_{selected}.png", mime="image/png",
                key="passport_qr_dl",
            )
            st.caption(f"Encoded URL: {_qr_url} · Print and affix to battery for on-site scanning.")
        except ImportError:
            st.warning(
                "qrcode library not installed. Add `qrcode[pil]` to requirements.txt "
                "and restart the app."
            )

    # ── Machine-readable export (JSON-LD) ─────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:0.08em;padding-bottom:8px;border-bottom:1px solid #2d3748;"
        "margin-bottom:12px;margin-top:24px'>9 · Machine-Readable Export</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "JSON-LD export of this passport's field data, preserving each field's "
        "available/estimated/unavailable provenance tag."
    )
    from passport_export import to_json_ld
    import json as _json_dpp
    _dpp_jsonld = to_json_ld(p, selected)
    st.download_button(
        "Download machine-readable passport (JSON-LD)",
        data=_json_dpp.dumps(_dpp_jsonld, indent=2).encode(),
        file_name=f"{selected}_passport.jsonld",
        mime="application/ld+json",
        key="passport_jsonld_dl",
    )


def page_reports(selected: str, df: pd.DataFrame, bundle: dict, rul_reliable: bool):
    from passport import build_passport
    from consequences import ASSUMPTIONS, application_fit, financial_comparison

    is_nasa = selected in NASA_CELL_IDS
    source  = "nasa" if is_nasa else "synth"
    p       = build_passport(selected, df, bundle, rul_reliable, is_nasa)

    latest  = df.iloc[-1]
    soh     = float(latest["soh_pct"])

    st.markdown("# Reports")
    st.markdown(f"##### Demonstration report export · {selected}")

    _md_html("""<div style="background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.25);border-radius:10px;padding:14px 20px;margin-bottom:28px;font-size:13px;color:#8896a8;line-height:1.7"><strong style="color:#63b3ed">Demonstration report</strong> — not a regulatory document. Exports the current battery's identity, SOH/RUL with reliability flags, second-life recommendation (if applicable), and the assumption register, with the same Available / Estimate / Not-available-in-demo labelling used throughout this platform.</div>""")

    second_life = None
    if soh <= 85.0:
        fade_30 = float(latest.get("fade_rate_30cy", 0.0))
        fit     = application_fit(soh, fade_30, fleet_fade_median=None)
        best_key, best = max(fit.items(), key=lambda kv: {"fit": 2, "marginal": 1, "not_fit": 0}[kv[1]["fit"]])

        a   = {k: v["value"] for k, v in ASSUMPTIONS.items()}
        fc  = financial_comparison(
            soh=soh, source=source,
            recycling_value=a["recycling_value"], new_cell_cost=a["new_cell_cost"],
            sl_value_per_kwh=a["second_life_value_per_kwh"], repack_cost=a["repack_cost"],
        )
        second_life = {
            "best_app": best["name"],
            "best_fit": best["fit"],
            "financials": {
                "Reuse (second-life)": fc["sl_net"],
                "Recycle now":         fc["recycle_value"],
                "Buy new cell":        -fc["new_cell_cost"],
            },
        }

    st.markdown("##### Preview")
    st.markdown(f"**Cell:** {selected} &nbsp;·&nbsp; **SOH:** {soh:.1f}%")
    if second_life:
        st.markdown(
            f"**Second-life fit:** {second_life['best_app']} ({second_life['best_fit']}) — "
            f"figures are cited estimates, see Consequences page for full sliders."
        )
    else:
        st.markdown("**Second-life fit:** still in primary life — no recommendation yet.")
    summ = p["summary"]
    st.markdown(
        f"**Field coverage:** {summ['n_available']} available · {summ['n_estimated']} estimated · "
        f"{summ['n_unavailable']} not available in demo"
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    from report_pdf import build_report_pdf
    from passport_export import to_json_ld
    import json as _json_report

    pdf_bytes, _doc_id = build_report_pdf(p, second_life, ASSUMPTIONS)
    _jsonld = to_json_ld(p, selected, doc_id=_doc_id)

    st.caption(f"Document ID: {_doc_id} — both files below share this ID, since they're the same export.")
    _rep_col1, _rep_col2 = st.columns(2)
    with _rep_col1:
        st.download_button(
            label="Download demonstration report (PDF)",
            data=pdf_bytes,
            file_name=f"battery_passport_{selected}_{_doc_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    with _rep_col2:
        st.download_button(
            label="Download machine-readable passport (JSON-LD)",
            data=_json_report.dumps(_jsonld, indent=2).encode(),
            file_name=f"battery_passport_{selected}_{_doc_id}.jsonld",
            mime="application/ld+json",
            use_container_width=True,
        )


def page_sustainability(selected: str, df: pd.DataFrame):
    _action_bar("sustainability")
    from consequences import ASSUMPTIONS, sustainability_snapshot, CELL_NOMINAL_KWH
    from sustainability import (
        CRITICAL_MATERIALS, EU_RECYCLED_TARGETS, EU_GREEN_DEAL_FIELDS,
        material_content_for_cell,
    )

    is_nasa = selected in NASA_CELL_IDS
    source  = "nasa" if is_nasa else "synth"
    latest  = df.iloc[-1]
    soh     = float(latest["soh_pct"])
    cycles  = int(latest["cycle_number"])
    cell_kwh = CELL_NOMINAL_KWH[source]

    # ── Badge helpers ──
    def _section(title: str):
        st.markdown(section_header_html(title), unsafe_allow_html=True)

    # ── Page header ──
    st.markdown("# Sustainability")
    st.markdown(f"##### Lifecycle carbon + circularity · {selected}")

    st.markdown(
        f"<div style='background:rgba(183,121,31,0.07);border:1px solid rgba(183,121,31,0.25);"
        f"border-radius:10px;padding:14px 20px;margin-bottom:28px;"
        f"font-size:13px;color:#8896a8;line-height:1.7'>"
        f"<strong style='color:#d69e2e'>Figure transparency.</strong> "
        f"All CO₂ and material figures are estimates from literature sources — "
        f"not measurements from this specific cell. Each figure is labeled "
        f"{BADGE_ESTIMATE} or {BADGE_ILLUST} at the point of display. "
        f"No aggregated sustainability score is shown: individual labeled figures "
        f"are more honest than any index that mixes sources with different confidence levels."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Sliders ──
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

    # ────────────────────────────────────────────────────────────────────────
    # Section 1: Hero CO₂ comparison
    # ────────────────────────────────────────────────────────────────────────
    _section("CO₂ Impact — Reuse vs Recycle vs New Cell")

    h1, h2, h3, h4 = st.columns(4)
    hero_tiles = [
        (h1, "Manufacturing CO₂\n(one new cell)",   f"{co2_val:.2f} kg", "#f6ad55", BADGE_ESTIMATE,
         "IVL 2019 · range 0.30–1.00 kg/cell · adjust slider"),
        (h2, "Use phase CO₂\n(to date, this cell)", f"{use_phase_co2:.2f} kg", "#718096", BADGE_ILLUST,
         f"Grid: {grid_val:.2f} kg CO₂/kWh (IEA 2023) · {cycles} cycles"),
        (h3, "Reuse saves\n(vs making a new cell)", f"{sus['co2_avoided_by_reuse']:.2f} kg", "#48bb78", BADGE_ESTIMATE,
         "Harper et al. 2019 · Dunn 2015 · vs manufacturing a new cell"),
        (h4, "Recycle credit\n(15% cathode, Dunn 2015)", f"{sus['co2_recycling_credit']:.2f} kg", "#f6e05e", BADGE_ESTIMATE,
         "Dunn et al. 2015 · 15% cathode material recovery credit"),
    ]
    for col, label, val, colour, badge, src_note in hero_tiles:
        label_lines = label.split("\n")
        label_html = "<br>".join(label_lines)
        with col:
            st.markdown(
                f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                f"padding:16px 20px;height:100%'>"
                f"<div style='font-size:11px;color:#4a5568;line-height:1.5'>{label_html}</div>"
                f"<div style='font-size:26px;font-weight:700;color:{colour};margin-top:6px'>{val}</div>"
                f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>CO₂e</div>"
                f"<div style='margin-top:8px'>{badge}</div>"
                f"<div style='font-size:10px;color:#2d3748;margin-top:4px;line-height:1.4'>{src_note}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # Section 2: Lifecycle carbon chart
    # ────────────────────────────────────────────────────────────────────────
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
        index=0,
        horizontal=True,
        key="sus7_lc_norm",
        help="Per kWh delivered divides all bars by total energy throughput for each scenario, making scenarios with different lifetimes comparable.",
    )

    reuse_cycles       = cycles * extension_val
    new_cell_cycles    = cycles   # same history length as counterfactual baseline
    reuse_use_co2      = cell_kwh * reuse_cycles * grid_val
    new_cell_use_co2   = cell_kwh * new_cell_cycles * grid_val

    scenarios = ["Recycle now", f"Reuse (×{extension_val:.1f} cycles)", "New cell (counterfactual)"]
    mfg_bars       = [co2_val, co2_val, co2_val]
    use_bars        = [use_phase_co2, reuse_use_co2, new_cell_use_co2]
    eol_credit_bars = [
        -sus["co2_recycling_credit"],
        -(sus["co2_recycling_credit"] + sus["co2_avoided_by_reuse"]),
        -sus["co2_recycling_credit"],
    ]

    if lc_norm == "Per kWh delivered":
        kwh_denominators = [
            cell_kwh * cycles,
            cell_kwh * reuse_cycles,
            cell_kwh * new_cell_cycles,
        ]
        mfg_bars       = [v / d for v, d in zip(mfg_bars, kwh_denominators)]
        use_bars       = [v / d for v, d in zip(use_bars, kwh_denominators)]
        eol_credit_bars = [v / d for v, d in zip(eol_credit_bars, kwh_denominators)]
        yaxis_label    = "kg CO₂e per kWh delivered"
        bar_suffix     = " kg/kWh"
    else:
        yaxis_label = "kg CO₂e per cell"
        bar_suffix  = " kg"

    fig_lc = go.Figure()
    fig_lc.add_trace(go.Bar(
        name="Manufacturing CO₂",
        x=scenarios, y=mfg_bars,
        marker_color="#f6ad55",
        text=[f"{v:.3f}{bar_suffix}" for v in mfg_bars],
        textposition="inside", textfont=dict(size=10, color="#1a202c"),
    ))
    fig_lc.add_trace(go.Bar(
        name="Use phase CO₂ (illustrative)",
        x=scenarios, y=use_bars,
        marker_color="#718096",
        text=[f"{v:.3f}{bar_suffix}" for v in use_bars],
        textposition="inside", textfont=dict(size=10, color="#e2e8f0"),
    ))
    fig_lc.add_trace(go.Bar(
        name="EOL credit (negative = saving)",
        x=scenarios, y=eol_credit_bars,
        marker_color="#48bb78",
        text=[f"{v:.3f}{bar_suffix}" for v in eol_credit_bars],
        textposition="inside", textfont=dict(size=10, color="#1a202c"),
    ))
    fig_lc.update_layout(
        **base_layout(
            barmode="relative",
            xaxis=dict(zeroline=False),
            yaxis=dict(
                zeroline=True,
                zerolinecolor="#4a5568",
                title=dict(text=yaxis_label, font=dict(size=11)),
            ),
            height=360,
        )
    )
    fig_lc.update_layout(
        legend=LEGEND_H,
        title=dict(
            text="Lifecycle CO₂ — three end-of-life scenarios",
            font=dict(size=13, color="#a0aec0"),
            x=0,
        ),
    )
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

    # ────────────────────────────────────────────────────────────────────────
    # Section 2b: Per-cell carbon context (qualitative — no percentages)
    # ────────────────────────────────────────────────────────────────────────
    _section("Degradation Rate & Carbon — What the Trend Means")

    fade_30 = float(latest.get("fade_rate_30cy", float("nan")))
    if not (fade_30 != fade_30):  # not NaN
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
            fade_colour = "#48bb78"

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
            f"<div style='font-size:13px;color:#a0aec0;line-height:1.8'>"
            f"{fade_implication}"
            f"</div>"
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

    # ────────────────────────────────────────────────────────────────────────
    # Section 3: Critical materials tracker
    # ────────────────────────────────────────────────────────────────────────
    _section("Critical Materials Tracker")

    if not is_nasa:
        st.markdown(
            "<div style='background:#2d3748;border-radius:8px;padding:10px 16px;"
            "font-size:12px;color:#8896a8;margin-bottom:12px'>"
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
            f"<div style='font-size:12px;color:#48bb78;margin-top:4px'>"
            f"~{mat['recovery_pct']}% recovery<br>"
            f"<span style='font-size:11px;color:#4a5568'>{mat['recovery_note']}</span></div>"
            if mat["recovery_pct"] is not None else
            "<div style='font-size:12px;color:#4a5568;margin-top:4px'>Not recovered<br>"
            "<span style='font-size:11px'>Not primary material in LiCoO₂</span></div>"
        )
        eu_dot = (
            "<span style='color:#63b3ed;font-size:10px;margin-left:4px'>"
            "EU critical ●</span>"
            if mat["eu_critical"] else ""
        )
        with col:
            st.markdown(
                f"<div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;"
                f"padding:14px 16px'>"
                f"<div style='font-size:12px;color:#a0aec0;font-weight:600'>"
                f"{mat['name']}{eu_dot}</div>"
                f"<div style='font-size:11px;color:#4a5568;margin-top:2px'>{mat['formula']}</div>"
                f"<div style='font-size:22px;font-weight:700;color:#e2e8f0;margin-top:8px'>"
                f"{scaled_g:.1f} g</div>"
                f"<div style='font-size:11px;color:#8896a8'>est. per cell ({mat['g_range']} @ 2 Ah)</div>"
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
            f"<strong style='color:#8896a8'>Nickel (Ni)</strong> — EU critical material, but trace-only in LiCoO₂ chemistry "
            f"(est. {ni_g:.2f} g per cell, {BADGE_ILLUST}). "
            f"EU 2023/1542 nickel recycled-content targets apply to NMC/NCA chemistries where nickel is a primary cathode material, "
            f"not to LiCoO₂. Shown here for completeness only."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Section 4: EU regulation recycled content targets
    # ────────────────────────────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────────────────
    # Section 5: EU Green Deal alignment (three-state)
    # ────────────────────────────────────────────────────────────────────────
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

    # ── Assumption register ──
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
                f"<div style='font-size:12px;color:#8896a8;line-height:1.6'>{a['source']}</div>"
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
    _md_html(
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
        """
    )


# ---------------------------------------------------------------------------
# Regulatory Alert Service
# ---------------------------------------------------------------------------

def _page_regulatory_alerts(
    selected: str,
    df: pd.DataFrame,
    featured_dfs: dict,
    bundles: dict,
):
    """EU Battery Regulation Article 14(4) compliance deadline tracker.

    Shows which cells will be non-compliant by key regulation dates,
    and generates a draft submission text.
    """
    import datetime as _dt_reg

    st.markdown("### Regulatory Alert Service")
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:18px;line-height:1.6'>"
        "EU Battery Regulation 2023/1542 requires a battery state report under "
        "<strong style='color:#e2e8f0'>Article 14(4)</strong> when SOH drops below defined "
        "thresholds. This tracker identifies cells at risk of non-compliance before key deadlines."
        "</div>"
    )

    # Key regulatory deadlines
    _deadlines = [
        {
            "label":       "Art. 14(4) — SOH report obligation",
            "date":        _dt_reg.date(2026, 2, 18),
            "soh_floor":   80.0,
            "description": "Batteries placed on EU market must have SOH ≥ 80% or carry a non-compliance notice.",
        },
        {
            "label":       "Art. 70 — End-of-life transparency",
            "date":        _dt_reg.date(2027, 8, 18),
            "soh_floor":   None,
            "description": "Battery Passport must include full end-of-life R-code and recycled content declaration.",
        },
        {
            "label":       "Annex XII — Recycled content targets (Phase 1)",
            "date":        _dt_reg.date(2031, 1, 1),
            "soh_floor":   None,
            "description": "12% Co, 4% Li, 4% Ni recycled content required in active materials.",
        },
    ]

    _today = _dt_reg.date.today()
    _rows_reg = []
    for _cid, _fdf in featured_dfs.items():
        if _fdf.empty or "soh_pct" not in _fdf.columns:
            continue
        _soh_now = float(_fdf.iloc[-1]["soh_pct"])
        _fade    = float(_fdf.iloc[-1].get("fade_rate_50cy", 0))
        _eol_cy  = None
        if _fade > 1e-6:
            _eol_cy = int(max(0, (_soh_now - 80.0) / _fade * 50))
        _rows_reg.append({
            "cell_id": _cid,
            "soh_now": _soh_now,
            "fade":    _fade,
            "cycles_to_eol": _eol_cy,
        })

    # Alert summary
    _art14_floor = 80.0
    _art14_date  = _deadlines[0]["date"]
    _days_to_deadline = (_art14_date - _today).days

    _at_risk = [r for r in _rows_reg if r["soh_now"] < _art14_floor]
    _approaching = [
        r for r in _rows_reg
        if r["soh_now"] >= _art14_floor and r["cycles_to_eol"] is not None
        and r["cycles_to_eol"] < max(0, _days_to_deadline) * 1.5
    ]

    _al1, _al2, _al3 = st.columns(3)
    _al_col = "#fc8181" if _at_risk else ("#f6ad55" if _approaching else "#48bb78")
    _al1.metric("Cells non-compliant now", len(_at_risk), help="SOH already below 80%")
    _al2.metric("Cells approaching threshold", len(_approaching),
                help=f"Will breach 80% before {_art14_date.strftime('%B %Y')}")
    _al3.metric("Days to Art. 14(4) deadline", _days_to_deadline)

    # Per-deadline table
    for _dl in _deadlines:
        _dl_days = (_dl["date"] - _today).days
        _dl_col  = "#fc8181" if _dl_days < 180 else ("#f6ad55" if _dl_days < 365 else "#48bb78")
        _md_html(
            f"<div style='background:#1e2a38;border:1px solid {_dl_col}44;"
            f"border-radius:10px;padding:14px 20px;margin-bottom:10px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div style='font-size:13px;font-weight:600;color:#e2e8f0'>{_dl['label']}</div>"
            f"<div style='font-size:12px;font-weight:700;color:{_dl_col}'>"
            f"{_dl['date'].strftime('%d %b %Y')} · {_dl_days} days</div>"
            f"</div>"
            f"<div style='font-size:12px;color:#8896a8;margin-top:6px'>{_dl['description']}</div>"
            f"</div>"
        )

    # Affected cells
    if _at_risk or _approaching:
        st.markdown("<div class='section-header'>Cells Requiring Action</div>", unsafe_allow_html=True)
        for _r in sorted(_at_risk + _approaching, key=lambda x: x["soh_now"]):
            _is_breach = _r["soh_now"] < _art14_floor
            _status_c  = "#fc8181" if _is_breach else "#f6ad55"
            _status_l  = "Non-compliant" if _is_breach else "Approaching threshold"
            _md_html(
                f"<div style='display:flex;justify-content:space-between;padding:10px 16px;"
                f"border-bottom:1px solid #1a202c;align-items:center'>"
                f"<div style='font-size:13px;font-weight:600;color:#e2e8f0'>{_r['cell_id']}</div>"
                f"<div style='font-size:12px;color:#a0aec0'>SOH {_r['soh_now']:.1f}%</div>"
                f"<div style='font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;"
                f"background:{_status_c}22;color:{_status_c};border:1px solid {_status_c}44'>"
                f"{_status_l}</div>"
                f"</div>"
            )

        # Draft submission
        st.markdown("<div class='section-header'>Draft Compliance Notice</div>", unsafe_allow_html=True)
        _draft_cells = "\n".join(
            f"  - Cell {r['cell_id']}: SOH {r['soh_now']:.1f}% "
            f"({'below' if r['soh_now'] < _art14_floor else 'approaching'} 80% threshold)"
            for r in sorted(_at_risk + _approaching, key=lambda x: x["soh_now"])
        )
        _draft_text = (
            f"BATTERY STATE COMPLIANCE NOTICE\n"
            f"EU Battery Regulation (EU) 2023/1542 — Article 14(4)\n"
            f"Generated: {_today.strftime('%d %B %Y')}\n"
            f"Compliance deadline: {_art14_date.strftime('%d %B %Y')}\n\n"
            f"The following battery cells have been identified as non-compliant or approaching\n"
            f"the mandatory State of Health threshold under Art. 14(4):\n\n"
            f"{_draft_cells}\n\n"
            f"Recommended actions:\n"
            f"  1. Commission independent SOH verification for cells marked non-compliant.\n"
            f"  2. Update Battery Passport records with current SOH and R-code classification.\n"
            f"  3. Schedule replacement or repurposing before the compliance deadline.\n\n"
            f"This notice was generated automatically by the Battery Intelligence Platform\n"
            f"using GBRT-derived SOH estimates. Values should be confirmed by certified testing\n"
            f"before submission to a regulatory authority.\n\n"
            f"Platform: Battery Intelligence System\n"
            f"Data pipeline: GBRT with leave-cell-out cross-validation\n"
            f"Regulation reference: (EU) 2023/1542, OJ L 2023/1542\n"
        )
        st.text_area("Draft notice (edit before submitting)", value=_draft_text, height=280,
                     key="reg_draft_text")
        st.download_button(
            "Download notice as .txt",
            data=_draft_text,
            file_name=f"compliance_notice_{_today.isoformat()}.txt",
            mime="text/plain",
            key="reg_draft_download",
        )
    else:
        st.success(
            f"All {len(_rows_reg)} cells are compliant with the 80% SOH threshold. "
            f"Next scheduled review: {_art14_date.strftime('%B %Y')}."
        )


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
        _action_bar("compliance")
        st.markdown(f"# Is {selected} EU passport-ready?")
        st.markdown("##### EU Battery Regulation 2023/1542 · Passport · Reports · Sustainability")
        _tab_passport, _tab_reports, _tab_sus, _tab_reg = st.tabs(
            ["EU Battery Passport", "Reports & Export", "Sustainability", "Regulatory Alerts"]
        )
        with _tab_passport:
            page_passport(selected, df, bundle, rul_reliable)
        with _tab_reports:
            page_reports(selected, df, bundle, rul_reliable)
        with _tab_sus:
            page_sustainability(selected, df)
        with _tab_reg:
            _page_regulatory_alerts(selected, df, active_fdfs, bundles)
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

