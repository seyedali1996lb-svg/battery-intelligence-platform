"""
Battery Intelligence Platform — Streamlit Dashboard
Phase 1 dashboard: Overview, Health, Insights (functional)

Data:
  - 8 synthetic cells (Cell1-Cell8): physics-informed, injected cell-to-cell
    stress variation (temperature, C-rate, DoD). Not real measured data.
  - 4 NASA cells (B0005-B0018): real LiCoO2 18650 measurements from NASA PCoE
    Battery Aging dataset (Saha & Goebel, 2007, ~2 Ah, 24 C, 2A discharge).

Model trained on all 12 cells combined. NASA loader (batlab.datasets.nasa,
`python -m batlab.datasets.nasa`) must be run once to populate data/raw/
before the app starts.
"""

import sys
import os
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import build_battery, CELL_STRESS_PROFILES, _stress_factor
from batlab.features.engineering import build_features, get_model_matrix
from batlab.models.gbrt import train_models, predict
from batlab.validation.lco import run_lco, RUL_RELIABLE_FLOOR
from trajectory_memory import TrajectoryMemory
from utils import NASA_CELL_IDS, _md_html, render_card, load_tenant_bundle_cached, cached_match_fleet
from bundle_cache import load_cached, save_cached, load_features_cached, save_features_cached
import cell_store
from chemistry_profiles import ChemistryProfile


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

# Base dark-theme CSS is a static asset (app/static/theme.css), fetched once
# by the browser's normal HTTP cache instead of being re-sent as part of the
# app's element stream on every single rerun (any widget interaction
# anywhere in the app) — see that file's own header comment for why, and
# .streamlit/config.toml for the required server.enableStaticServing=true.
# The conditional light-mode override block below stays inline (unchanged)
# since it's only sent on the sessions that actually toggle it on.
st.markdown(
    '<link rel="stylesheet" href="app/static/theme.css">',
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
        .hero-label  { color:#a0aec0 !important; }
        .hero-sub    { color:#a0aec0 !important; }
        .t-body, .t-caption { color:#a0aec0 !important; }
        .metric-chip-label, .metric-chip-sub { color:#a0aec0 !important; }

        /* ── Sidebar nav ── */
        section[data-testid="stSidebar"] button[kind="secondary"] {
            color:#a0aec0 !important; background: transparent !important;
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
        .empty-state-body  { color:#a0aec0 !important; }

        /* ── Section headers ── */
        .section-header {
            color:#a0aec0 !important;
            border-color: #e2e8f0 !important;
        }

        /* ── Streamlit native widgets ── */
        div[data-testid="stMetricValue"]  { color: #1a202c !important; }
        div[data-testid="stMetricLabel"]  { color:#a0aec0 !important; }
        div[data-testid="stMetricDelta"]  { color: #2d3748 !important; }
        .stSelectbox label,
        .stSlider label,
        .stRadio label,
        .stCheckbox label,
        .stTextInput label,
        .stNumberInput label { color:#a0aec0 !important; }
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
        section[data-testid="stSidebar"] div[style*="color:#a0aec0"],
        section[data-testid="stSidebar"] div[style*="color:#2d3748"] { color:#a0aec0 !important; }
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
        df_feat = build_features(cell["cycles"], cell_id=cell_id)
        X, y_soh, y_rul = get_model_matrix(df_feat)
        raw_fdfs[cell_id]     = df_feat
        model_inputs[cell_id] = (X, y_soh, y_rul)
    return raw_fdfs, model_inputs


def _train_and_predict(battery_dict: dict, raw_fdfs: dict,
                       model_inputs: dict, dataset: "str | None" = None,
                       org_id: "int | None" = None) -> tuple[dict, dict, dict]:
    """Train SOH+RUL models on pre-computed features and apply predictions.

    Separated from feature engineering so load_everything() can use a features
    cache hit to skip build_features() while still running LCO + model training.

    dataset/org_id: when both are given, this fit is logged to the
    experiment registry (src/experiment_registry.py) — the single place a
    real train_models()+run_lco() call happens for the built-in reference
    fleets, so every load_everything() pipeline gets logged automatically
    with no separate call needed. Left as None for callers (tests, dead
    code paths) that don't have a dataset/org_id to attribute the run to.
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

    if dataset is not None and org_id is not None:
        import experiment_registry as _reg
        from batlab.features.engineering import FEATURE_VERSION as _FV
        from batlab.models.gbrt import GBRT_PARAMS as _GBRT_PARAMS
        from chemistry_profiles import ChemistryProfile as _CP
        _sample_cell = next(iter(battery_dict))
        bndl["metrics"]["experiment_run_id"] = _reg.log_run(
            org_id=org_id,
            dataset=dataset,
            chemistry=_CP.for_cell(_sample_cell).short_name,
            feature_set=list(X_all.columns),
            feature_version=_FV,
            hyperparams=dict(_GBRT_PARAMS),
            seed=_GBRT_PARAMS["random_state"],
            cell_ids=list(battery_dict.keys()),
            n_rows=len(X_all),
            lco_metrics=lco,
        )

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

    def _persist_cell_data(featured_dfs: dict) -> None:
        """Write each cell's full per-cycle DataFrame to cell_store's
        Parquet store + a precomputed CellSummary row -- called whenever
        featured_dfs is freshly computed (a bundle-cache miss). A cache HIT
        means this already happened on the run that originally populated
        the cache, so it's not repeated. Only 3 threads ever call this
        concurrently (synth/nasa/severson below), each writing a disjoint
        set of cell_ids, so no explicit locking is added -- cell_store's
        LRU dict operations are individually atomic under the GIL and never
        touch the same key from two threads at once here."""
        import db as _db_persist
        from experiment_registry import PLATFORM_ORG_ID as _PLATFORM_ORG_ID

        for cell_id, df in featured_dfs.items():
            cell_store.save_cell_df(cell_id, df)
            _db_persist.upsert_cell_summary(
                _PLATFORM_ORG_ID, cell_id, cell_store.build_summary(cell_id, df),
            )

    def _load_or_train_bg(key: str, cell_dict: dict) -> tuple[dict, dict]:
        """2-tier cache: full bundle → features-only → full pipeline.
        Returns (bundle, split_cycles) -- per-cell DataFrames are persisted
        to cell_store (see _persist_cell_data above) instead of being kept
        in this return value, so load_everything() never has to hold every
        cell's full data resident just to satisfy this function's return
        contract. Thread-safe — no st.* calls."""
        import experiment_registry as _reg

        cached = load_cached(key, cell_dict)
        if cached is not None:
            return cached
        feat_cached = load_features_cached(key, cell_dict)
        if feat_cached is not None:
            raw_fdfs, model_inputs = feat_cached
        else:
            raw_fdfs, model_inputs = _compute_features_only(cell_dict)
            save_features_cached(key, cell_dict, raw_fdfs, model_inputs)

        bundle, featured_dfs, split_cycles = _train_and_predict(
            cell_dict, raw_fdfs, model_inputs, dataset=key, org_id=_reg.PLATFORM_ORG_ID,
        )
        _persist_cell_data(featured_dfs)
        result = (bundle, split_cycles)
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
            from batlab.datasets.severson import load_severson_cells, any_cached as _sev_any_cached
            if _sev_any_cached():
                sev_cells = load_severson_cells(status_fn=lambda msg: None)
                if sev_cells:
                    sev_cell_dicts = {cid: {"cycles": df} for cid, df in sev_cells.items()}
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

        bundle_synth, sc_synth = futures["synth"].result()
        bundle_nasa,  sc_nasa  = futures["nasa"].result()  if "nasa"     in futures else (None, {})
        bundle_sev,   sc_sev   = futures["severson"].result() if "severson" in futures else (None, {})

        _prog.progress(100, text="Platform ready ✓")
        # st.status()/st.progress() return None when called outside a real
        # Streamlit script run (e.g. imported by src/api.py under a bare
        # uvicorn process) — guard so load_everything() works from both.
        if _status is not None:
            _status.update(label="Platform ready ✓", state="complete", expanded=False)

    sev_ids = list(sev_cell_dicts.keys())
    split_cycles = {**sc_synth, **sc_nasa, **sc_sev}
    bundles = {"synth": bundle_synth, "nasa": bundle_nasa, "severson": bundle_sev}
    # Lazily-loaded, LRU-bounded view over cell_store's Parquet files instead
    # of a dict holding every cell's full DataFrame resident for the whole
    # process lifetime -- see cell_store.py's module docstring. Every cell
    # listed here was persisted by _persist_cell_data() above (on a cache
    # miss) or on the earlier run that populated bundle_cache (on a hit).
    featured_dfs = cell_store.LazyCellFrameMap(synth_ids + nasa_ids + sev_ids)

    return featured_dfs, bundles, split_cycles


@st.cache_resource(show_spinner=False)
def get_platform_graph(_featured_dfs_all: dict, _bundles: dict):
    """Build (once per process, same caching pattern as load_everything()
    itself) the Battery Digital Knowledge Graph (src/knowledge_graph.py) for
    the platform's shared reference fleets. Leading-underscore params are
    excluded from Streamlit's cache-key hashing (the same documented
    convention app/utils.py's cached_match_fleet() uses) — featured_dfs_all/
    bundles are themselves already the output of a cache_resource call, so
    this only actually runs once per process too.

    This is the ONE graph instance Health / the Cell Workbench / the
    Copilot / Explore's "cells like this" panel all read from — see
    knowledge_graph.get_or_compute_mechanism()'s docstring for why sharing
    one graph instance (not just one function) closes the mechanism-
    verdict-computed-independently bug class for good.
    """
    import knowledge_graph as kg
    import experiment_registry as reg

    graph = kg.build_platform_graph(_featured_dfs_all, _bundles, tenant_org_id=None)
    try:
        kg.save_graph(reg.PLATFORM_ORG_ID, graph)
    except Exception:
        pass  # persistence is a nice-to-have snapshot, not required for the graph to work in-session
    return graph


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

# Grouped nav — each entry is (group_label, [(page_label, page_key), ...]).
#
# IA rule (review finding: "a new user can't predict which pattern applies
# where" — this makes the rule explicit and applies it uniformly): a
# top-level group/item is a distinct, addressable workflow surface.
# Multiple lenses on the *same* underlying cell/fleet data are never split
# across separate nav items — they live as an in-page radio or tab within
# one item instead. This is now applied consistently everywhere in the
# app: Fleet (Fleet Overview / Cell Grading tabs), Explore (Compare /
# Cluster / Cohort / Pack Builder / Reference Datasets tabs), Compliance
# (Passport / Reports / Sustainability / Regulatory Alerts tabs), and
# Health + Decide & Ask (merged into one "Diagnose & Decide" workbench,
# below — see app/_pages/workbench.py; this used to be two separate nav
# items requiring two page navigations for what is really one workflow on
# one cell, a real UX cost flagged in review).
#
# "EU Passport" is the one deliberate exception to "distinct workflow
# only": it is a single-item group promoted to peer status with the
# broader functional groups for a business reason (EU 2027 passport
# deadline visibility for the Compliance Officer role), not an oversight.
NAV_GROUPS = [
    ("Analyse", [
        ("Overview",   "overview"),
        ("Explore",    "compare"),
        ("Benchmark",  "benchmark"),
    ]),
    ("EU Passport", [          # A6: promoted from 3rd to 2nd — 2027 deadline
        ("Compliance", "compliance"),
    ]),
    ("Operate", [
        ("Fleet",             "fleet"),
        ("Diagnose & Decide", "decision"),
        ("Live Monitor",      "live_monitor"),
    ]),
    ("Configure", [
        ("Configure", "configure"),
    ]),
]

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
        "<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
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
            # Active row: styled card — ● bold label + muted status
            render_card(
                f"<div style='font-size:13px;font-weight:700;color:#e2e8f0'>"
                f"● {m['label']}</div>"
                f"<div style='font-size:11px;color:#8896a8;margin-top:2px'>{m['status']}</div>",
                padding="9px 12px",
                extra_style="margin-bottom:5px",
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
                f"<div style='font-size:11px;color:#a0aec0;margin:-8px 0 5px 4px'>"
                f"{m['status']}</div>",
                unsafe_allow_html=True,
            )
        else:
            # Unavailable (My Data, no upload yet) — grayed out, not clickable
            st.markdown(
                f"<div style='padding:8px 12px;margin-bottom:5px;opacity:0.45'>"
                f"<div style='font-size:13px;color:#8896a8'>○  {m['label']}</div>"
                f"<div style='font-size:11px;color:#a0aec0;margin-top:2px'>{m['status']}</div>"
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
        "benchmark": ("benchmark", ["benchmark", "leaderboard", "experiment", "runs", "model performance"]),
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
#
# Step 0's intro line used to hardcode "This fleet leads with NASA PCoE
# cells", regardless of which data source was actually active — false
# whenever a session started in Severson/synthetic/uploaded mode. It's now a
# template filled in per-mode by _tour_data_mode_line() at render time,
# mirroring the same mode -> description mapping render_sidebar() already
# uses for its subtitle, so the two can't drift out of sync independently.
_TOUR_STEPS = [
    (
        "Real measured data, not a toy demo",
        "This fleet is running on {data_mode_desc}. Watch for the "
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
_FIRST_RUN_OVERLAYS = ["role_chosen", "mode_chosen", "tour_seen"]  # session_state "done" flags, in show-order


def _active_first_run_overlay() -> "str | None":
    """Return the session_state key of the next not-yet-completed first-run
    overlay, or None once all of them are done."""
    for key in _FIRST_RUN_OVERLAYS:
        if not st.session_state.get(key, False):
            return key
    return None


def _tour_data_mode_line(mode: str) -> str:
    """Mode-appropriate intro line for the guided tour's first step — mirrors
    render_sidebar()'s subtitle mapping so both surfaces describe the
    actually-active data source, never a hardcoded one."""
    return {
        "severson":  "real, measured <strong>Severson 2019 LFP</strong> cells — not synthetic curves",
        "nasa":      "real, measured <strong>NASA PCoE</strong> cells — real LiCoO₂ 18650 measurements, not synthetic curves",
        "synthetic": "<strong>physics-informed synthetic</strong> cells — modelled degradation, not measured data",
        "uploaded":  "<strong>your own uploaded</strong> cell data",
    }.get(mode, "real, measured <strong>NASA PCoE</strong> cells — real LiCoO₂ 18650 measurements, not synthetic curves")


@st.dialog("Welcome — Guided Tour")
def _guided_tour_dialog():
    step = st.session_state.get("tour_step", 0)
    title, body = _TOUR_STEPS[step]
    if step == 0:
        body = body.format(data_mode_desc=_tour_data_mode_line(st.session_state.get("data_mode", "nasa")))

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
        _role_icons = {"Engineer": "⚙", "Fleet Manager": "🗂", "Executive": "📊", "Compliance Officer": "📋"}
        _role_icon = _role_icons.get(_cur_role, "⚙")
        _rc1, _rc2 = st.columns([4, 1])
        with _rc1:
            st.markdown(
                f"<div style='font-size:11px;color:#8896a8;padding:4px 0 2px'>"
                f"{_role_icon} <span style='color:#e2e8f0;font-weight:600'>{_cur_role}</span></div>",
                unsafe_allow_html=True,
            )
        with _rc2:
            if st.button("↺ Change role", key="change_role_btn", use_container_width=True,
                         help="Switch role (Engineer / Fleet Manager / Executive)"):
                st.session_state["role_chosen"] = False
                st.rerun()

        if st.button("↺ Change focus", key="change_mode_btn", use_container_width=True,
                     help="Re-pick Diagnose / Monitor / Plan — doesn't change your role"):
            st.session_state["mode_chosen"] = False
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

        # A7: front-load 1-2 relevant nav groups for non-technical roles —
        # other groups stay reachable, just collapsed, not hidden.
        _PRIORITY_GROUPS = {
            "Executive": {"Operate", "EU Passport"},
            "Compliance Officer": {"EU Passport"},
        }
        _priority = _PRIORITY_GROUPS.get(_cur_role)

        for group_label, group_items in NAV_GROUPS:
            _has_current = any(key == current_page for _, key in group_items)
            _expanded = True if _priority is None else (group_label in _priority or _has_current)
            with st.expander(group_label.upper(), expanded=_expanded):
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
            "<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
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
        # Runs on every page render, uncached -- the highest-frequency
        # full-fleet touch point in the app, so reference-fleet modes read
        # precomputed CellSummary rows instead of every cell's full
        # per-cycle DataFrame (see cell_store.py's module docstring).
        # "uploaded" mode's fleet is a real, session-scoped dict, out of
        # scope for this migration -- unchanged.
        if mode == "uploaded":
            _cell_records = [
                (
                    _cid,
                    _fdf.iloc[-1].get("soh_pct"),
                    float(_fdf["resistance_ohm"].iloc[0]) if "resistance_ohm" in _fdf.columns and len(_fdf) > 1 else None,
                    _fdf.iloc[-1].get("resistance_ohm") if "resistance_ohm" in _fdf.columns and len(_fdf) > 1 else None,
                )
                for _cid, _fdf in (active_fdfs or {}).items()
            ]
        else:
            import db as _db_alerts
            _org_id = st.session_state.get("auth_org_id")
            _summaries_by_id = {r["cell_id"]: r for r in _db_alerts.get_cell_summaries(_org_id)} if _org_id else {}
            _cell_records = [
                (
                    _cid,
                    _summaries_by_id[_cid].get("soh_pct"),
                    _summaries_by_id[_cid].get("resistance_ohm_initial"),
                    _summaries_by_id[_cid].get("resistance_ohm"),
                )
                for _cid in cell_ids if _cid in _summaries_by_id
            ]

        if _cell_records:
            _soh_thresh  = float(st.session_state.get("soh_alert_pct", 85))
            _res_mult    = float(st.session_state.get("resistance_alert_mult", 1.8))
            _spread_thresh = float(st.session_state.get("spread_alert_pct", 5.0))
            _alert_msgs  = []
            _soh_vals = []
            for _cid, _soh, _r_init, _r_now in _cell_records:
                if _soh is None:
                    continue
                _soh_vals.append(float(_soh))
                if float(_soh) < _soh_thresh:
                    _alert_msgs.append(("warn", f"**{_cid}** SOH {float(_soh):.1f}% — below {_soh_thresh:.0f}%"))
                if _r_init and _r_now and float(_r_init) > 0 and float(_r_now) > float(_r_init) * _res_mult:
                    _alert_msgs.append(("error", f"**{_cid}** R = {float(_r_now)/float(_r_init):.2f}× initial"))
            if len(_soh_vals) > 1:
                _spread = max(_soh_vals) - min(_soh_vals)
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

        # Identity + logout — a login session otherwise persists indefinitely
        # (no app-level timeout beyond the browser tab/WebSocket staying
        # open), and there was no UI path at all to end it early. Real gap
        # on a shared/kiosk machine, found during an Enterprise Readiness
        # audit alongside login.py's missing lockout (see db.py).
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:11px;color:#a0aec0;padding:0 4px 4px'>"
            f"{st.session_state.get('auth_name') or st.session_state.get('auth_user', '')} · "
            f"{st.session_state.get('auth_org_name', '')}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Log out", key="sidebar_logout_btn", use_container_width=True):
            for _k in ("authenticated", "auth_user", "auth_role", "auth_name",
                       "auth_org_id", "auth_org_name"):
                st.session_state.pop(_k, None)
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
    from _pages.grading import page_grading
    from _pages.fleet import page_fleet
    from _pages.copilot import page_copilot
    from _pages.workbench import page_cell_workbench
    from _pages.compliance import page_compliance
    from _pages.benchmark import page_benchmark
    if not render_login():
        return   # login form rendered; stop until credentials provided

    _train_placeholder = st.empty()
    _train_placeholder.markdown(
        "<div style='text-align:center;padding:40px;color:#a0aec0;font-size:14px'>"
        "Initialising models… (first run only — cached on subsequent loads)</div>",
        unsafe_allow_html=True,
    )
    featured_dfs_all, bundles, split_cycles_all = load_everything()
    _train_placeholder.empty()
    graph = get_platform_graph(featured_dfs_all, bundles)

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
    # load_everything() only ever merges synth+NASA+Severson into
    # featured_dfs_all/split_cycles_all (Oxford and uploaded data are kept
    # out of this pipeline entirely) so ChemistryProfile.for_cell().source_kind
    # only ever resolves to one of these 3 buckets here.
    # Filtered by cell_id only (never touches a cell's full DataFrame just to
    # bucket it by source) -- iterating a LazyCellFrameMap yields keys, same
    # as a real dict, and each filtered bucket stays lazy too.
    nasa_fdfs  = cell_store.LazyCellFrameMap([k for k in featured_dfs_all if ChemistryProfile.for_cell(k).source_kind == "nasa"])
    sev_fdfs   = cell_store.LazyCellFrameMap([k for k in featured_dfs_all if ChemistryProfile.for_cell(k).source_kind == "severson"])
    synth_fdfs = cell_store.LazyCellFrameMap([k for k in featured_dfs_all if ChemistryProfile.for_cell(k).source_kind == "synth"])
    nasa_sc    = {k: v for k, v in split_cycles_all.items() if ChemistryProfile.for_cell(k).source_kind == "nasa"}
    sev_sc     = {k: v for k, v in split_cycles_all.items() if ChemistryProfile.for_cell(k).source_kind == "severson"}
    synth_sc   = {k: v for k, v in split_cycles_all.items() if ChemistryProfile.for_cell(k).source_kind == "synth"}

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
    # Batched: one query for all 6 settings instead of one round-trip each.
    _settings = _db_main.get_settings(
        st.session_state["auth_org_id"],
        keys=["pinned_cell", "app_profile", "cost_of_delay_mult",
              "webhook_url", "webhook_secret", "webhook_events", "eol_threshold_pct"],
    )
    if "pinned_cell" not in st.session_state:
        st.session_state["pinned_cell"] = _settings.get("pinned_cell")
    if "app_profile" not in st.session_state:
        _persisted_profile = _settings.get("app_profile")
        if _persisted_profile is not None:
            st.session_state["app_profile"] = _persisted_profile
    if "cost_of_delay_mult" not in st.session_state:
        _persisted_cod = _settings.get("cost_of_delay_mult")
        if _persisted_cod is not None:
            st.session_state["cost_of_delay_mult"] = _persisted_cod
    for _wh_key in ("webhook_url", "webhook_secret", "webhook_events"):
        if _wh_key not in st.session_state:
            _persisted_wh = _settings.get(_wh_key)
            if _persisted_wh is not None:
                st.session_state[_wh_key] = _persisted_wh

    # Application EOL threshold — user-configurable in Settings.
    # Changing this does NOT retrain the model; it rescales the displayed RUL
    # using the current fade rate to project to the user-defined threshold.
    if "eol_threshold_pct" not in st.session_state:
        _persisted_eol = _settings.get("eol_threshold_pct")
        st.session_state["eol_threshold_pct"] = _persisted_eol if _persisted_eol is not None else 80.0

    # ── Resolve active data from current mode ─────────────────────────────────
    mode    = st.session_state["data_mode"]
    up_meta = st.session_state["uploaded_mode_meta"]

    # An org's uploaded ("My Data") fleet is never stored in session_state —
    # load_tenant_bundle_cached() (app/utils.py) is a process-wide
    # st.cache_resource, shared across every session of this org rather than
    # duplicated per-session, and transparently reloads from disk on a cache
    # miss (first access after a server restart, or after a fresh upload
    # cleared the cache — see app/_pages/import_page.py).
    _tenant = load_tenant_bundle_cached(st.session_state["auth_org_id"])
    if _tenant is not None:
        up_fdfs, up_bundle, up_sc = _tenant
    else:
        up_fdfs, up_sc, up_bundle = {}, {}, None

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
            "<div style='font-size:14px;color:#a0aec0;margin-bottom:32px'>"
            "I'll personalise the dashboard for your role. You can change this any time in Settings.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        _r1, _r2, _r3, _r4 = st.columns(4)
        _role_picked = None
        with _r1:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                "<div style='font-size:28px;margin-bottom:8px'>🔧</div>"
                "<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>Engineer</div>"
                "<div style='font-size:12px;color:#a0aec0'>Diagnose cells · Deep analytics · Root-cause tools</div>"
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
                "<div style='font-size:12px;color:#a0aec0'>Monitor fleet · Prioritise replacements · Alerts</div>"
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
                "<div style='font-size:12px;color:#a0aec0'>Fleet KPIs · CAPEX forecast · ESG compliance</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select Executive", key="onboard_exec", use_container_width=True):
                _role_picked = "Executive"
        with _r4:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                "<div style='font-size:28px;margin-bottom:8px'>📋</div>"
                "<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>Compliance Officer</div>"
                "<div style='font-size:12px;color:#a0aec0'>EU 2023/1542 passport · Audit trail · Regulatory alerts</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select Compliance Officer", key="onboard_compliance", use_container_width=True):
                _role_picked = "Compliance Officer"
        if _role_picked:
            st.session_state["user_role"] = _role_picked
            st.session_state["role_chosen"] = True
            # Don't clobber the guided tour's "Finish tour" destination.
            if not (st.session_state.get("tour_seen") and st.session_state.get("page") == "compliance"):
                if _role_picked == "Executive":
                    st.session_state.page = "exec_summary"
                elif _role_picked == "Fleet Manager":
                    st.session_state.page = "fleet"
                elif _role_picked == "Compliance Officer":
                    st.session_state.page = "compliance"
                else:
                    st.session_state.page = "overview"
            st.rerun()
        st.stop()

    # Use-case landing interstitial — shown once per session, right after the
    # role picker. Deliberately outcome-oriented labels ("Diagnose"/"Monitor"/
    # "Plan"), not "BMS"/"ESS" mode names: this app has no real hardware
    # connection today (src/bms_connectors.py's adapters are documented-shape-
    # only, never validated against a live account; Live Monitor's MQTT feed
    # is an honestly-labelled simulation), so badging this as a live
    # management mode would oversell what's real. This only decides where you
    # land and which already-existing expanders start open — no new
    # persistent state that changes how any page renders afterward.
    if _active_first_run_overlay() == "mode_chosen":
        st.markdown(
            "<div style='max-width:680px;margin:80px auto 0;text-align:center'>"
            "<div style='font-size:28px;font-weight:800;color:#e2e8f0;margin-bottom:8px'>"
            "What are you here to do?</div>"
            "<div style='font-size:14px;color:#a0aec0;margin-bottom:32px'>"
            "This just picks where you land — everything stays reachable from the sidebar either way.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        _m1, _m2, _m3 = st.columns(3)
        _mode_picked = None
        with _m1:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                "<div style='font-size:28px;margin-bottom:8px'>🔋</div>"
                "<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>Diagnose a battery</div>"
                "<div style='font-size:12px;color:#a0aec0'>SOH/RUL, degradation mechanism, recommendations</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select", key="onboard_mode_diagnose", use_container_width=True):
                _mode_picked = "diagnose"
        with _m2:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                "<div style='font-size:28px;margin-bottom:8px'>📡</div>"
                "<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>Monitor live telemetry</div>"
                "<div style='font-size:12px;color:#a0aec0'>Streaming SOH/anomaly view (demo mode simulates the feed)</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select", key="onboard_mode_monitor", use_container_width=True):
                _mode_picked = "monitor"
        with _m3:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                "<div style='font-size:28px;margin-bottom:8px'>☀️</div>"
                "<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>Plan a storage deployment</div>"
                "<div style='font-size:12px;color:#a0aec0'>Size a second-life battery + solar, payback/NPV</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select", key="onboard_mode_plan", use_container_width=True):
                _mode_picked = "plan"
        if _mode_picked:
            st.session_state["mode_chosen"] = True
            if _mode_picked == "monitor":
                st.session_state.page = "live_monitor"
            elif _mode_picked == "plan":
                st.session_state.page = "decision"
                st.session_state["mode_landing_ess"] = True
            else:
                st.session_state.page = "overview"
            st.rerun()
        st.stop()

    # A4: count flagged cells for trajectory chip in sidebar (cached — see
    # utils.cached_match_fleet; on the Fleet page itself this is the same
    # active_fdfs Fleet's own match_fleet() call uses, so it's a cache hit
    # there too, not a second computation)
    try:
        _traj_flag = len(cached_match_fleet(trajectory_memory, active_fdfs))
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
        "style='font-size:10px;color:#a0aec0;cursor:default'>demo mode</span></div>",
        unsafe_allow_html=True,
    )

    # Per-cell reliability: use the specific fold R² for this cell, not the group average.
    per_cell_ok  = bundle["metrics"].get("per_cell_rul_reliable", {})
    rul_reliable = per_cell_ok.get(selected, bundle["metrics"].get("rul_reliable", True))

    if page == "overview":
        page_overview(df, split_cycle, selected, rul_reliable=rul_reliable, bundle=bundle,
                      trajectory_memory=trajectory_memory)
    elif page == "health":
        page_cell_workbench("health", selected, df, split_cycle, active_fdfs, bundles,
                             rul_reliable, bundle, graph=graph)
    elif page == "compare":
        page_compare(cell_ids, active_fdfs, bundles, graph=graph)
    elif page == "benchmark":
        page_benchmark(st.session_state["auth_org_id"])
    elif page in ("copilot", "insights"):
        page_copilot(cell_ids, active_fdfs, bundles, selected, graph=graph)
    elif page in ("decision", "consequences", "recommendations"):
        page_cell_workbench("decision", selected, df, split_cycle, active_fdfs, bundles,
                             rul_reliable, bundle, graph=graph)
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

