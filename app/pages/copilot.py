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
    _md_html(
        """<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.18);border-radius:10px;padding:14px 20px;margin-bottom:24px;font-size:13px;color:#718096;line-height:1.6"><strong style="color:#63b3ed">Grounded narration only.</strong> Every sentence is derived from values already computed by the model pipeline — SOH, feature importances, per-cell RUL reliability, fade rates. The Copilot never calculates, estimates, or infers a value not already in the bundle. If a number is not there, it says so.</div>"""
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
