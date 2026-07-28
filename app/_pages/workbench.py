"""Page: Cell Workbench — merged Health (mechanism) + Decide & Ask (decision).

Review finding (UX Design / Workflow Design, High severity): the core
workflow loop -- see cell -> understand mechanism -> get recommendation ->
log decision -> create ticket -- required visiting Health and Decide & Ask
as two separate page navigations, even though they operate on the exact
same cell and the same underlying data. Industry decision-support tools
(Palantir AIP, ServiceNow) build one workbench per decision type: a
detail view with tabs, not two independently-styled pages requiring
back-and-forth sidebar navigation.

This module does not rewrite page_health()/page_decision() -- both are
substantial, already tested (AppTest smoke coverage + the full pytest
suite), and their internal logic is unrelated to this fix. Instead it's a
thin, view-selecting wrapper, exactly mirroring the existing convention
already used on the Fleet page for its own Grading sub-view
(st.radio("View", ["Fleet Overview", "Cell Grading"])) and on Explore's
Compare/Cluster/Cohort tabs -- one nav item, one or more in-page views,
never two nav items for what is really one workflow on one cell.
"""

import streamlit as st
import pandas as pd

from utils import render_regenerate_report_button

_VIEW_MECHANISM = "Mechanism (Health)"
_VIEW_DECISION  = "Decision (Decide & Ask)"


def page_cell_workbench(
    default_view: str,
    selected: str,
    df: pd.DataFrame,
    split_cycle: int,
    featured_dfs: dict,
    bundles: dict,
    rul_reliable: bool,
    bundle: dict,
    graph=None,
) -> None:
    """
    default_view: "health" or "decision" -- which page-key the router used
    to arrive here (nav button, quick-jump, trajectory-match card, etc.).
    Forces the radio to the matching view whenever the *router's* page key
    changes since the last render, so every existing "Go to Decide & Ask"/
    "Health ->" call site (they only ever set st.session_state.page, never
    touch this module) keeps landing on the correct tab. A view switch made
    via the radio itself, without changing the router's page key, is left
    alone -- that's the user actively browsing between the two lenses on
    the same cell, not a fresh navigation.
    """
    from _pages.health import page_health
    from _pages.decision import page_decision

    default_label = _VIEW_MECHANISM if default_view == "health" else _VIEW_DECISION
    _last_key = st.session_state.get("_workbench_last_default_view")
    if _last_key != default_view:
        st.session_state["workbench_view_radio"] = default_label
        st.session_state["_workbench_last_default_view"] = default_view

    st.markdown(f"# {selected}: Diagnose & Decide")
    view = st.radio(
        "View", [_VIEW_MECHANISM, _VIEW_DECISION],
        horizontal=True, key="workbench_view_radio", label_visibility="collapsed",
    )

    if view == _VIEW_MECHANISM:
        page_health(df, split_cycle, selected, bundle=bundle, rul_reliable=rul_reliable, graph=graph)
    else:
        page_decision(selected, df, featured_dfs, bundles, rul_reliable, graph=graph)

    render_regenerate_report_button(
        bundle, st.session_state.get("auth_org_id"), key_suffix=f"workbench_{selected}"
    )
