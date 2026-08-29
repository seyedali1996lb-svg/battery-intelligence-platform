"""
Sidebar, navigation, mode switcher, and onboarding dialogs.

Extracted from app/main.py to keep the orchestrator thin.  Every function
here renders Streamlit widgets inside the sidebar or a dialog modal.
"""

from __future__ import annotations

import _paths  # noqa: F401 — ensures src/ and app/ are on sys.path

import streamlit as st

from utils import _md_html, render_card
from data_loader import CELL_STRESS_PROFILES, _stress_factor
from chemistry_profiles import ChemistryProfile
import rbac


# ---------------------------------------------------------------------------
# Navigation groups
# ---------------------------------------------------------------------------

NAV_GROUPS = [
    ("Analyse", [
        ("Overview",   "overview"),
        ("Explore",    "compare"),
        ("Benchmark",  "benchmark"),
    ]),
    ("EU Passport", [
        ("Compliance", "compliance"),
    ]),
    ("Operate", [
        ("Fleet",             "fleet"),
        ("Diagnose & Decide", "decision"),
        ("Live Monitor",      "live_monitor"),
        ("Grid Services",     "operations"),
    ]),
    ("Configure", [
        ("Configure", "configure"),
    ]),
]


# ---------------------------------------------------------------------------
# Upload status helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mode switcher
# ---------------------------------------------------------------------------

def render_mode_switcher(nasa_n: int, synth_n: int, up_meta: dict | None,
                         sev_n: int = 0) -> None:
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
            render_card(
                f"<div style='font-size:13px;font-weight:700;color:#e2e8f0'>"
                f"● {m['label']}</div>"
                f"<div style='font-size:11px;color:#8896a8;margin-top:2px'>{m['status']}</div>",
                padding="9px 12px",
                extra_style="margin-bottom:5px",
            )
        elif is_available:
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
            st.markdown(
                f"<div style='padding:8px 12px;margin-bottom:5px;opacity:0.45'>"
                f"<div style='font-size:13px;color:#8896a8'>○  {m['label']}</div>"
                f"<div style='font-size:11px;color:#a0aec0;margin-top:2px'>{m['status']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# Onboarding dialogs — imported from _onboarding.py for backward compat.
from _onboarding import (
    command_palette_dialog as _command_palette_dialog,
    active_first_run_overlay as _active_first_run_overlay,
    guided_tour_dialog as _guided_tour_dialog,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(
    cell_ids: list[str],
    mode: str,
    nasa_n: int,
    synth_n: int,
    up_meta: dict | None,
    sev_n: int = 0,
    active_fdfs=None,
    traj_flag: int = 0,
) -> str:
    """Render the full sidebar and return the selected cell ID."""
    with st.sidebar:
        # Dynamic subtitle
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

        # ── Inline page search ──
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
                _sc = sum(1 for kw in _kws if kw in _sq)
                if _sc > _best_score:
                    _best_score, _best_dest = _sc, _dest
            if _best_score == 0:
                _best_dest = "decision"
                st.session_state["copilot_free_text"] = _sb_query
            st.session_state.page = _best_dest
            st.rerun()

        # Quick-access strip
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

        # ── Role selector ──
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

        # Trajectory match chip
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

        _priority = rbac.front_loaded_nav(_cur_role)

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

        # Chemistry label
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
        if "_nav_cell" in st.session_state:
            _nav_target = st.session_state.pop("_nav_cell")
            if _nav_target in cell_ids:
                st.session_state["selected_cell"] = _nav_target
        _cur_sel = st.session_state.get("selected_cell")
        if _cur_sel not in cell_ids:
            st.session_state["selected_cell"] = cell_ids[0]
        _sel_idx = cell_ids.index(st.session_state["selected_cell"])
        selected = st.selectbox(
            "Cell",
            options=cell_ids,
            index=_sel_idx,
            key="selected_cell",
            label_visibility="collapsed",
        )

        # Prev / Next
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

        # ── Fleet alerts ──
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
            _soh_thresh    = float(st.session_state.get("soh_alert_pct", 85))
            _res_mult      = float(st.session_state.get("resistance_alert_mult", 1.8))
            _spread_thresh = float(st.session_state.get("spread_alert_pct", 5.0))
            _alert_msgs: list[tuple[str, str]] = []
            _soh_vals: list[float] = []
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
                _label = f"🔔 {len(_alert_msgs)} Alert{'s' if len(_alert_msgs) != 1 else ''}"
                with st.expander(_label, expanded=False):
                    for _kind, _msg in _alert_msgs:
                        if _kind == "error":
                            st.error(_msg)
                        else:
                            st.warning(_msg)

        # ── Cell annotation ──
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

        # D4: Command palette — button only
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Search pages…", key="cmd_palette_btn", use_container_width=True,
                     help="Search pages and navigate quickly"):
            _command_palette_dialog()

        # Theme toggle
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _lm_current = st.session_state.get("light_mode", False)
        _lm_toggle = st.toggle("Light mode", value=_lm_current, key="light_mode_toggle")
        if _lm_toggle != _lm_current:
            st.session_state["light_mode"] = _lm_toggle
            st.rerun()

        # Identity + logout
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
