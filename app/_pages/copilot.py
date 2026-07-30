"""Page: Copilot."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go

from utils import _action_bar, _md_html, PLOTLY_CONFIG, friendly
from batlab.models.gbrt import top_drivers, feature_importance_df
from design_system import make_badge
from chemistry_profiles import ChemistryProfile


# ---------------------------------------------------------------------------
# Page: Copilot
# ---------------------------------------------------------------------------

def page_copilot(
    cell_ids: list[str],
    featured_dfs: dict,
    bundles: dict,
    selected: str,
    graph=None,
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
        answer_mechanism,
        answer_alerts,
        answer_replacement_budget,
        answer_fleet_risk,
        answer_business_case,
        QUERY_LABELS,
        FOLLOW_UP_MAP,
    )
    from battery_copilot import llm_answer

    st.markdown("# Copilot")

    # ── Disclosure banner ──
    _md_html(
        """<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.18);border-radius:10px;padding:14px 20px;margin-bottom:24px;font-size:13px;color:#8896a8;line-height:1.6"><strong style="color:#63b3ed">Grounded, and now tool-calling on free-text questions.</strong> Topic buttons always use fixed templates over values already computed by the model pipeline — SOH, feature importances, per-cell RUL reliability, fade rates. Typed questions that don't match a topic use real Claude tool-calling (when a personal API key is configured, see Settings): the model decides which of a handful of real data-fetching tools to call — including across multiple cells for compositional questions — but every tool returns only already-computed values, never a number the model invents itself. Without a key, typed questions fall back to the same grounded template shown for topic buttons. Either way, the Copilot never calculates, estimates, or infers a value not returned by a tool or already in the bundle.</div>"""
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
        "<div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;"
        "letter-spacing:0.1em;padding:10px 0 4px'>Cell questions</div>",
        unsafe_allow_html=True,
    )
    _tech_keys = ["health", "drivers", "rul", "compare", "recent", "anomaly", "fleet_compare", "mechanism"]
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
        "<div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;"
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
            "<div style='text-align:center;padding:40px 24px;color:#a0aec0;font-size:14px'>"
            "Type a question above or choose a topic. The Copilot answers using only "
            "values already computed by the model pipeline for "
            "<strong style='color:#8896a8'>" + selected + "</strong>.</div>",
            unsafe_allow_html=True,
        )
        return

    # D3: Free-text path — real Claude tool-use (Sonnet 5) when a personal
    # API key is configured, since the model needs to decide for itself
    # which real data-fetching tools a compositional question needs; falls
    # back to today's template+retrieval path with no key, or on any
    # failure -- answer_with_tools() never raises, it returns (None, []).
    if _free_text_query and not query:
        _api_key_ft = st.session_state.get("anthropic_api_key", "")
        _ft_answer = None
        _tools_used_ft: list = []
        if _api_key_ft:
            from copilot_agent import answer_with_tools
            with st.spinner("Thinking…"):
                _ft_answer, _tools_used_ft = answer_with_tools(
                    _free_text_query, selected, st.session_state["auth_org_id"],
                    featured_dfs, bundles, graph, _api_key_ft,
                )
        _used_agent = _ft_answer is not None
        if not _used_agent:
            _ctx_ft = build_cell_context(selected, featured_dfs, bundles)
            _ft_answer = answer_query(_free_text_query, _ctx_ft)
        st.markdown(
            f"<div style='font-size:11px;color:#a0aec0;margin-bottom:8px'>"
            f"Answering: <em style='color:#8896a8'>{_free_text_query}</em>"
            + (" · Claude Sonnet 5, tool-calling" if _used_agent else " · Template fallback")
            + "</div>",
            unsafe_allow_html=True,
        )
        _md_html(
            f"<div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;"
            f"padding:18px 22px;font-size:14px;color:#a0aec0;line-height:1.7'>{_ft_answer}</div>"
        )
        if _tools_used_ft:
            with st.expander(f"🔧 Tools used ({len(_tools_used_ft)})"):
                st.caption(", ".join(_tools_used_ft))
        if st.button("Clear", key="cpft_clear"):
            st.session_state.pop("copilot_free_text", None)
            st.session_state.pop("_last_ask", None)
            st.rerun()
        return

    # ── Pre-compute fleet stats (reads precomputed CellSummary rows) ──
    fleet_stats = build_fleet_stats(st.session_state["auth_org_id"], featured_dfs, bundles)

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
    elif query == "mechanism":
        _mech_edge, _mech_lit = None, []
        if graph is not None:
            try:
                from knowledge_graph import get_or_compute_mechanism, literature_for_mechanism, MECHANISM_VERDICT_TO_KEY
                _mech_edge = get_or_compute_mechanism(graph, selected, featured_dfs[selected])
                _mech_key = MECHANISM_VERDICT_TO_KEY.get(_mech_edge.get("verdict"), "insufficient_data")
                _mech_lit = literature_for_mechanism(graph, _mech_key)
            except Exception:
                _mech_edge = None
        response = answer_mechanism(ctx, _mech_edge, _mech_lit)
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
        # Grounded in the same classify()/financial_comparison() results
        # Decide & Ask itself uses, not an independent SOH/cost ladder --
        # see answer_business_case()'s docstring for the bug this replaced.
        from consequences import application_fit, financial_comparison, ASSUMPTIONS, CELL_NOMINAL_KWH
        from recommendations import classify
        _bc_latest = featured_dfs[selected].iloc[-1]
        _bc_soh = float(_bc_latest["soh_pct"])
        _bc_fade_30 = float(_bc_latest.get("fade_rate_30cy", 0.0))
        _bc_fade_50 = float(_bc_latest.get("fade_rate_50cy", 0.0))
        _bc_fit = application_fit(_bc_soh, _bc_fade_30, fleet_fade_median=None)
        _bc_result = classify(
            _bc_soh, _bc_fade_30, _bc_fade_50,
            ctx["rul_reliable"], ctx["rul_pred"], _bc_fit,
        )
        _bc_assumptions = {k: v["value"] for k, v in ASSUMPTIONS.items()}
        _bc_financials = None
        if ctx["source_kind"] in CELL_NOMINAL_KWH:
            _bc_financials = financial_comparison(
                soh=_bc_soh, source=ctx["source_kind"],
                recycling_value=_bc_assumptions["recycling_value"],
                new_cell_cost=_bc_assumptions["new_cell_cost"],
                sl_value_per_kwh=_bc_assumptions["second_life_value_per_kwh"],
                repack_cost=_bc_assumptions["repack_cost"],
            )
        response = answer_business_case(ctx, _bc_result, _bc_financials, _bc_assumptions)
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
        f"<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
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
            "<div style='font-size:11px;font-weight:600;color:#a0aec0;text-transform:uppercase;"
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
                    ChemistryProfile.for_cell(selected).source_kind,
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

