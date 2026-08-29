"""
First-run onboarding overlays: guided tour, role/mode pickers, command palette.

Extracted from _sidebar.py to separate UI chrome (onboarding dialogs) from
the persistent sidebar navigation and role switching.
"""

from __future__ import annotations

import streamlit as st

import _paths  # noqa: F401

from utils import _md_html


# ---------------------------------------------------------------------------
# Command palette dialog
# ---------------------------------------------------------------------------

@st.dialog("Command Palette")
def command_palette_dialog() -> None:
    _PALETTE_ROUTES = {
        "fleet":      ("fleet",      ["fleet", "cells", "monitor", "attention", "ranking", "overview", "all cells"]),
        "overview":   ("overview",   ["health", "cell", "soh", "status", "check"]),
        "compliance": ("compliance", ["passport", "eu", "compliance", "regulation", "eol"]),
        "decision":   ("decision",   ["decision", "replace", "repurpose", "second life", "what should"]),
        "copilot":    ("copilot",    ["copilot", "ask", "budget", "cost", "risk", "question"]),
        "health":     ("health",     ["degrading", "mechanism", "lli", "lam", "fade", "resistance"]),
        "benchmark":  ("benchmark",  ["benchmark", "leaderboard", "experiment", "runs", "model performance"]),
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


# ---------------------------------------------------------------------------
# Guided tour
# ---------------------------------------------------------------------------

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
        "Physics-informed diagnostics, not just a fitted model",
        "Open <strong>Engineering details</strong> on Health for a real PyBaMM electrochemical "
        "re-fit (SEI vs. active-material-loss decomposition), cross-checked against the ML "
        "verdict — the two are flagged explicitly whenever they disagree. Explore → "
        "<strong>Related Cells</strong> uses the Battery Knowledge Graph to find cells sharing "
        "this one's degradation mechanism, each edge traceable to the literature that supports it.",
    ),
    (
        "Compliance — EU Battery Passport",
        "The Compliance page tracks EU Battery Passport completeness field-by-field, so you "
        "can see exactly what regulatory data is available, estimated, or still missing.",
    ),
]

# Session-state keys that mark first-run overlays as completed, in show-order.
_FIRST_RUN_OVERLAYS = ["role_chosen", "mode_chosen", "tour_seen"]


def active_first_run_overlay() -> str | None:
    """Return the session_state key of the next not-yet-completed first-run
    overlay, or None once all of them are done."""
    for key in _FIRST_RUN_OVERLAYS:
        if not st.session_state.get(key, False):
            return key
    return None


def _tour_data_mode_line(mode: str) -> str:
    """Mode-appropriate intro line for the guided tour's first step."""
    return {
        "severson":  "real, measured <strong>Severson 2019 LFP</strong> cells — not synthetic curves",
        "nasa":      "real, measured <strong>NASA PCoE</strong> cells — real LiCoO₂ 18650 measurements, not synthetic curves",
        "synthetic": "<strong>physics-informed synthetic</strong> cells — modelled degradation, not measured data",
        "uploaded":  "<strong>your own uploaded</strong> cell data",
    }.get(mode, "real, measured <strong>NASA PCoE</strong> cells — real LiCoO₂ 18650 measurements, not synthetic curves")


@st.dialog("Welcome — Guided Tour")
def guided_tour_dialog() -> None:
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
