"""
Battery Intelligence Platform — Streamlit Dashboard

Thin orchestrator that wires together data loading (_data.py),
sidebar rendering (_sidebar.py), and page routing (_router.py).

Data:
  - 8 synthetic cells (Cell1-Cell8): physics-informed, injected cell-to-cell
    stress variation (temperature, C-rate, DoD). Not real measured data.
  - 4 NASA cells (B0005-B0018): real LiCoO2 18650 measurements from NASA PCoE
    Battery Aging dataset (Saha & Goebel, 2007, ~2 Ah, 24 C, 2A discharge).

Model trained on all 12 cells combined. NASA loader (batlab.datasets.nasa,
`python -m batlab.datasets.nasa`) must be run once to populate data/raw/
before the app starts.
"""

from __future__ import annotations

import _paths  # noqa: F401 — ensures src/ and app/ are on sys.path

import streamlit as st

from _data import (
    load_everything,
    ensure_cell_summaries_synced,
    get_platform_graph,
)
from _sidebar import (
    render_sidebar,
    _active_first_run_overlay,
    _guided_tour_dialog,
    NAV_GROUPS,
)
from _router import route, _render_role_onboarding, _render_mode_onboarding
from _session import hydrate_persistence, partition_cells, resolve_data_mode
from utils import _md_html, cached_match_fleet, load_tenant_bundle_cached
from trajectory_memory import TrajectoryMemory


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
# Global styles — light-mode override
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

if st.session_state.get("light_mode", False):
    st.markdown(
        """
        <style>
        /* ══ LIGHT MODE — full-coverage enterprise theme ══════════════════ */

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Authentication gate ───────────────────────────────────────────────────
    from _pages.login import render_login
    if not render_login():
        return   # login form rendered; stop until credentials provided

    import db as _db_init
    _db_init.init_db()

    # ── Load data + train models ──────────────────────────────────────────────
    _train_placeholder = st.empty()
    _train_placeholder.markdown(
        "<div style='text-align:center;padding:40px;color:#a0aec0;font-size:14px'>"
        "Initialising models… (first run only — cached on subsequent loads)</div>",
        unsafe_allow_html=True,
    )
    featured_dfs_all, bundles, split_cycles_all = load_everything()
    _train_placeholder.empty()
    ensure_cell_summaries_synced(list(featured_dfs_all.keys()))
    graph = get_platform_graph(featured_dfs_all, bundles)

    # ── Guided tour (once per session, first-time visitors) ───────────────────
    if "tour_seen" not in st.session_state:
        st.session_state["tour_seen"] = False
    if "tour_step" not in st.session_state:
        st.session_state["tour_step"] = 0
    if _active_first_run_overlay() == "tour_seen":
        _guided_tour_dialog()

    # ── Failure trajectory memory (built once per session) ────────────────────
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

    # ── Separate built-in cells + hydrate session state ──────────────────────
    parts = partition_cells(featured_dfs_all, split_cycles_all)
    nasa_fdfs, sev_fdfs, synth_fdfs = parts["nasa_fdfs"], parts["sev_fdfs"], parts["synth_fdfs"]
    nasa_sc, sev_sc, synth_sc = parts["nasa_sc"], parts["sev_sc"], parts["synth_sc"]

    hydrate_persistence(st.session_state["auth_org_id"])

    # ── Resolve active data from current mode ─────────────────────────────────
    _tenant = load_tenant_bundle_cached(st.session_state["auth_org_id"])
    if _tenant is not None:
        up_fdfs, up_bundle, up_sc = _tenant
    else:
        up_fdfs, up_sc, up_bundle = {}, {}, None
    up_meta = st.session_state.get("uploaded_mode_meta")

    mode, active_fdfs, active_sc, active_bundle = resolve_data_mode(
        nasa_fdfs=nasa_fdfs, sev_fdfs=sev_fdfs, synth_fdfs=synth_fdfs,
        nasa_sc=nasa_sc, sev_sc=sev_sc, synth_sc=synth_sc,
        bundles=bundles, up_fdfs=up_fdfs, up_bundle=up_bundle, up_sc=up_sc,
    )
    cell_ids = list(active_fdfs.keys())

    # ── Onboarding interstitials ──────────────────────────────────────────────
    if _active_first_run_overlay() == "role_chosen":
        _render_role_onboarding()

    if _active_first_run_overlay() == "mode_chosen":
        _render_mode_onboarding()

    # ── Sidebar ───────────────────────────────────────────────────────────────
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

    # ── Route to the selected page ────────────────────────────────────────────
    route(
        selected          = selected,
        df                = df,
        split_cycle       = split_cycle,
        bundle            = bundle,
        active_fdfs       = active_fdfs,
        bundles           = bundles,
        cell_ids          = cell_ids,
        rul_reliable      = bundle["metrics"].get("per_cell_rul_reliable", {}).get(
            selected, bundle["metrics"].get("rul_reliable", True)
        ),
        graph             = graph,
        up_bundle         = up_bundle,
        trajectory_memory = trajectory_memory,
    )


if __name__ == "__main__":
    main()
