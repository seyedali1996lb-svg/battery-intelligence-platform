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

from _onboarding import ONBOARDING_KEY
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
    from _pages.model_validation import page_model_validation
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
        "page_model_validation": page_model_validation,
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
# First-run interstitial — one screen, not three
# ---------------------------------------------------------------------------

# Each card answers "what are you here to do?" with a landing page *and* the
# role the dashboard should speak to. The role used to be a separate screen
# with its own four cards, which asked the user to know the platform's
# internal persona vocabulary before they had seen a single number.
_ONBOARDING_INTENTS = [
    {
        "key":   "onboard_mode_diagnose",
        "icon":  "🔋",
        "title": "Diagnose a battery",
        "blurb": "SOH/RUL, degradation mechanism, recommendations",
        "role":  "Engineer",
        "page":  "overview",
    },
    {
        "key":   "onboard_mode_monitor",
        "icon":  "📡",
        "title": "Monitor live telemetry",
        "blurb": "Streaming SOH/anomaly view (demo mode simulates the feed)",
        "role":  "Engineer",
        "page":  "live_monitor",
    },
    {
        "key":   "onboard_mode_plan",
        "icon":  "☀️",
        "title": "Plan a storage deployment",
        "blurb": "Size a second-life battery + solar, payback/NPV",
        "role":  "Executive",
        "page":  "decision",
    },
    {
        "key":   "onboard_mode_compliance",
        "icon":  "📋",
        "title": "Prove EU compliance",
        "blurb": "EU 2023/1542 passport · audit trail · regulatory alerts",
        "role":  "Compliance Officer",
        "page":  "compliance",
    },
]


def _apply_onboarding_intent(intent: dict) -> None:
    """Set the landing page and role for the chosen intent, then re-run.

    The legacy ``role_chosen`` / ``mode_chosen`` keys are still written so a
    session that later reads them (and the tests that assert on them) sees the
    same state the three-screen flow used to produce.
    """
    st.session_state["user_role"] = intent["role"]
    st.session_state["role_chosen"] = True
    st.session_state["mode_chosen"] = True
    st.session_state[ONBOARDING_KEY] = True
    st.session_state["page"] = intent["page"]
    if intent["page"] == "decision":
        # Opens the nested economics expanders on arrival (pop-once; see
        # tests/test_app_state_combinations.py's regression guard).
        st.session_state["mode_landing_ess"] = True
    st.rerun()


def _render_onboarding() -> None:
    """Render the single first-run interstitial (shown once per session)."""
    st.markdown(
        "<div style='max-width:680px;margin:80px auto 0;text-align:center'>"
        "<div style='font-size:28px;font-weight:800;color:#e2e8f0;margin-bottom:8px'>"
        "What are you here to do?</div>"
        "<div style='font-size:14px;color:#a0aec0;margin-bottom:32px'>"
        "This picks where you land and the role the dashboard speaks to — everything "
        "stays reachable from the sidebar either way.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _cols = st.columns(len(_ONBOARDING_INTENTS))
    _picked = None
    for _col, _intent in zip(_cols, _ONBOARDING_INTENTS):
        with _col:
            st.markdown(
                "<div style='border:1px solid #2d3748;border-radius:8px;padding:20px;text-align:center'>"
                f"<div style='font-size:28px;margin-bottom:8px'>{_intent['icon']}</div>"
                f"<div style='font-weight:700;color:#e2e8f0;margin-bottom:6px'>{_intent['title']}</div>"
                f"<div style='font-size:12px;color:#a0aec0'>{_intent['blurb']}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Select", key=_intent["key"], use_container_width=True):
                _picked = _intent
    if _picked is not None:
        _apply_onboarding_intent(_picked)

    _skip_col, _ = st.columns([1, 2])
    with _skip_col:
        if st.button("Skip — show me the dashboard", key="onboard_skip",
                     use_container_width=True):
            _apply_onboarding_intent({
                "role": "Engineer", "page": "overview",
            })
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

    # Deployment notice. This line used to read "No auth · session-scoped
    # uploads · data not persisted", which was false on every page that could
    # render it: the only way to reach here is through the login gate, against
    # bcrypt users with per-org scoping and server-side RBAC (src/db.py,
    # src/rbac.py), and uploads are persisted per tenant. A notice that
    # understates the platform's own security on every screen is the same
    # class of error as one that overstates it.
    st.markdown(
        "<div style='text-align:right;margin-bottom:4px'>"
        "<span title='Demo deployment on public reference datasets — see README → Limitations' "
        "style='font-size:10px;color:#a0aec0;cursor:default'>demo deployment</span>"
        "<div style='font-size:9px;color:#a0aec0;margin-top:1px'>"
        "signed in · org-scoped · public reference datasets</div></div>",
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
    elif page == "model_validation":
        # Fleet-wide, not cell-specific: it grades a model against a reference
        # fleet it reloads itself, so it needs no cell or bundle from the
        # router (same shape as the Benchmark page).
        pages["page_model_validation"]()
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
