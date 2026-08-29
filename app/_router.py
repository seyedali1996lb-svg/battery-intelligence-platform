"""
Page routing, onboarding interstitials, and session-state hydration.

Extracted from app/main.py to keep the orchestrator thin.  The route()
function is called once per Streamlit script run after data is loaded
and the sidebar has been rendered.
"""

from __future__ import annotations

import _paths  # noqa: F401 — ensures src/ and app/ are on sys.path

from typing import Any

import streamlit as st

from _sidebar import (
    _active_first_run_overlay,
    _guided_tour_dialog,
    NAV_GROUPS,
)


# ---------------------------------------------------------------------------
# Page imports (lazy — only imported when needed)
# ---------------------------------------------------------------------------

def _import_pages():
    """Import all page renderers.  Called once from route()."""
    from _pages.overview import page_overview
    from _pages.fleet import page_fleet
    from _pages.copilot import page_copilot
    from _pages.workbench import page_cell_workbench
    from _pages.compliance import page_compliance
    from _pages.benchmark import page_benchmark
    from _pages.grading import page_grading
    from _pages.live_monitor import page_live_monitor
    from _pages.explore import page_compare
    from _pages.import_page import page_import
    from _pages.settings import page_settings
    from _pages.operations import page_operations
    from _pages.compliance import COMING_SOON_META, page_coming_soon

    return {
        "page_overview":      page_overview,
        "page_fleet":         page_fleet,
        "page_copilot":       page_copilot,
        "page_cell_workbench": page_cell_workbench,
        "page_compliance":    page_compliance,
        "page_benchmark":     page_benchmark,
        "page_grading":       page_grading,
        "page_live_monitor":  page_live_monitor,
        "page_compare":       page_compare,
        "page_import":        page_import,
        "page_settings":      page_settings,
        "page_operations":    page_operations,
        "COMING_SOON_META":   COMING_SOON_META,
        "page_coming_soon":   page_coming_soon,
    }


# ---------------------------------------------------------------------------
# Role onboarding interstitial
# ---------------------------------------------------------------------------

def _render_role_onboarding() -> None:
    """Render the role picker interstitial (shown once per session)."""
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


# ---------------------------------------------------------------------------
# Use-case landing interstitial
# ---------------------------------------------------------------------------

def _render_mode_onboarding() -> None:
    """Render the use-case picker interstitial (shown once per session)."""
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


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def route(
    selected: str,
    df: Any,
    split_cycle: int,
    bundle: dict,
    active_fdfs: Any,
    bundles: dict,
    cell_ids: list[str],
    rul_reliable: bool,
    graph: Any,
    up_bundle: Any,
    trajectory_memory: Any,
) -> None:
    """Render the currently-selected page.  Called once per Streamlit script run."""
    pages = _import_pages()

    # Audit logging
    import os
    import audit as _audit
    _last_audited = st.session_state.get("_audit_last", "")
    page = st.session_state.get("page", "fleet")
    if f"{page}:{selected}" != _last_audited:
        _audit.log_page_view(page, selected)
        st.session_state["_audit_last"] = f"{page}:{selected}"

    # Demo mode notice
    st.markdown(
        "<div style='text-align:right;margin-bottom:4px'>"
        "<span title='No auth · session-scoped uploads · data not persisted — see README → Production Roadmap' "
        "style='font-size:10px;color:#a0aec0;cursor:default'>demo mode</span>"
        "<div style='font-size:9px;color:#a0aec0;margin-top:1px'>"
        "No auth · session-scoped uploads · data not persisted</div></div>",
        unsafe_allow_html=True,
    )

    if page == "overview":
        pages["page_overview"](df, split_cycle, selected, rul_reliable=rul_reliable, bundle=bundle,
                               trajectory_memory=trajectory_memory)
    elif page == "health":
        pages["page_cell_workbench"]("health", selected, df, split_cycle, active_fdfs, bundles,
                                     rul_reliable, bundle, graph=graph)
    elif page == "compare":
        pages["page_compare"](cell_ids, active_fdfs, bundles, graph=graph)
    elif page == "benchmark":
        pages["page_benchmark"](st.session_state["auth_org_id"])
    elif page in ("copilot", "insights"):
        pages["page_copilot"](cell_ids, active_fdfs, bundles, selected, graph=graph)
    elif page in ("decision", "consequences", "recommendations"):
        pages["page_cell_workbench"]("decision", selected, df, split_cycle, active_fdfs, bundles,
                                     rul_reliable, bundle, graph=graph)
    elif page in ("compliance", "sustainability", "passport", "reports"):
        pages["page_compliance"](selected, df, bundle, rul_reliable, active_fdfs, bundles)
    elif page in ("fleet", "exec_summary"):
        pages["page_fleet"](active_fdfs, bundles, trajectory_memory=trajectory_memory)
    elif page == "grading":
        pages["page_grading"](cell_ids, active_fdfs, bundles, selected)
    elif page == "live_monitor":
        pages["page_live_monitor"](cell_ids, active_fdfs)
    elif page == "operations":
        pages["page_operations"](cell_ids, active_fdfs)
    elif page in ("settings", "import", "configure"):
        st.markdown("# Configure")
        _cfg_tab_import, _cfg_tab_settings = st.tabs(["Import Data", "Settings"])
        with _cfg_tab_import:
            pages["page_import"]()
        with _cfg_tab_settings:
            pages["page_settings"](
                active_fdfs,
                {"nasa": bundles["nasa"], "synth": bundles["synth"], "uploaded": up_bundle},
            )
    elif page in pages["COMING_SOON_META"]:
        pages["page_coming_soon"](page)
    else:
        pages["page_overview"](df, split_cycle, selected)
