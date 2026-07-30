"""Fleet page — diagnostics tier (everything below the exec summary bar).

Extracted from fleet.py's page_fleet() (was 1,210 lines — the 2nd largest
file in app/_pages/, after health.py). page_fleet() keeps the always-
visible hero (ask-the-fleet, daily digest webhook, per-cell `rows`
computation, header metrics, exec summary bar) and calls
render_fleet_diagnostics() once, threading through the unfiltered `rows`
list plus featured_dfs/bundles/trajectory_memory.

Split boundary rationale: `rows` is filtered partway through (the Filter
Fleet UI) and every section from that point on must see the filtered
list, not the original -- render_filter_and_view_selector() returns the
filtered rows (or None if the user picked the Cell Grading view, which
fully renders its own page and needs nothing further from here) and the
orchestrator threads that onward. Everything else recomputes cheap
per-call state (STATUS_COLOUR, _traj_matches) locally rather than take
more parameters, same pattern as _health_diagnostics.py.
"""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    _md_html, _empty_state, base_layout, LEGEND_H,
    render_pack_builder, cached_match_fleet,
)
from data_loader import CELL_STRESS_PROFILES
from _pages.grading import page_grading


# ---------------------------------------------------------------------------
# T1: Executive Fleet Dashboard
# ---------------------------------------------------------------------------

def render_fleet_summary_expander(rows: list) -> None:
    with st.expander("📊 Fleet Summary", expanded=False):
        import numpy as _np_exec
        # CAPEX forecast: cells at risk of EOL within 3/6/12 months
        # Assume 200 cycles/month (typical daily EV cycling)
        _CYCLES_PER_MONTH = 200
        _REPLACEMENT_COST_USD = 150   # $/cell — configurable assumption
        _CO2_KG_PER_KWH = 0.85       # kg CO₂e per kWh capacity (manufacturing)
        _CELL_KWH = 0.0057            # kWh per cell (e.g. 2Ah × 2.85V nominal)

        _eol_3m = [r for r in rows if r["cycles_to_eol"] is not None and r["cycles_to_eol"] <= 3 * _CYCLES_PER_MONTH]
        _eol_6m = [r for r in rows if r["cycles_to_eol"] is not None and r["cycles_to_eol"] <= 6 * _CYCLES_PER_MONTH]
        _eol_12m = [r for r in rows if r["cycles_to_eol"] is not None and r["cycles_to_eol"] <= 12 * _CYCLES_PER_MONTH]

        _capex_3m  = len(_eol_3m)  * _REPLACEMENT_COST_USD
        _capex_6m  = len(_eol_6m)  * _REPLACEMENT_COST_USD
        _capex_12m = len(_eol_12m) * _REPLACEMENT_COST_USD
        _co2_12m   = len(_eol_12m) * _CELL_KWH * _CO2_KG_PER_KWH

        # Fleet health index (0–100) — weighted average SOH
        _sohs = [r["soh"] for r in rows]
        _fleet_hi = float(_np_exec.mean(_sohs)) if _sohs else 0.0
        _hi_color = "#48bb78" if _fleet_hi >= 90 else ("#f6ad55" if _fleet_hi >= 80 else "#fc8181")

        # Knee-alert count
        _knee_past = sum(1 for r in rows if r["knee"] and r["knee"].get("status") == "past_knee")
        _knee_near = sum(1 for r in rows if r["knee"] and r["knee"].get("status") == "approaching")

        _md_html(f"""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px'>
          <div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:14px 16px'>
            <div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px'>Fleet Health Index</div>
            <div style='font-size:28px;font-weight:800;color:{_hi_color}'>{_fleet_hi:.1f}%</div>
            <div style='font-size:11px;color:#a0aec0;margin-top:4px'>Avg SOH · {len(rows)} cells monitored</div>
          </div>
          <div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:14px 16px'>
            <div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px'>Accelerated Degradation</div>
            <div style='font-size:28px;font-weight:800;color:{"#fc8181" if _knee_past > 0 else "#f6ad55"}'>{_knee_past + _knee_near}</div>
            <div style='font-size:11px;color:#a0aec0;margin-top:4px'>{_knee_past} past knee · {_knee_near} approaching</div>
          </div>
          <div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:14px 16px'>
            <div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px'>CAPEX Outlook</div>
            <div style='font-size:28px;font-weight:800;color:#63b3ed'>${_capex_12m:,}</div>
            <div style='font-size:11px;color:#a0aec0;margin-top:4px'>{len(_eol_12m)} replacements forecast · 12-month horizon</div>
          </div>
        </div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px'>
          <div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:14px 16px'>
            <div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px'>Urgent — 3 Months</div>
            <div style='font-size:22px;font-weight:800;color:{"#fc8181" if _eol_3m else "#48bb78"}'>{len(_eol_3m)} cells</div>
            <div style='font-size:11px;color:#a0aec0;margin-top:4px'>CAPEX: <strong style='color:#fc8181'>${_capex_3m:,}</strong></div>
          </div>
          <div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:14px 16px'>
            <div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px'>Plan — 6 Months</div>
            <div style='font-size:22px;font-weight:800;color:{"#f6ad55" if _eol_6m else "#48bb78"}'>{len(_eol_6m)} cells</div>
            <div style='font-size:11px;color:#a0aec0;margin-top:4px'>CAPEX: <strong style='color:#f6ad55'>${_capex_6m:,}</strong></div>
          </div>
          <div style='background:#1e2a38;border:1px solid #2d3748;border-radius:10px;padding:14px 16px'>
            <div style='font-size:10px;font-weight:700;color:#a0aec0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px'>CO₂ Liability — 12M</div>
            <div style='font-size:22px;font-weight:800;color:#b794f4'>{_co2_12m:.1f} kg</div>
            <div style='font-size:11px;color:#a0aec0;margin-top:4px'>Manufacturing CO₂e · {len(_eol_12m)} new cells</div>
          </div>
        </div>
        <div style='font-size:10px;color:#a0aec0;margin-top:10px'>
        Assumptions: {_CYCLES_PER_MONTH} cycles/month · ${_REPLACEMENT_COST_USD}/cell replacement · {_CO2_KG_PER_KWH} kg CO₂e/kWh · {_CELL_KWH*1000:.0f} Wh/cell.
        Adjust in Configure → Application Profile.
        </div>
        """)


# ---------------------------------------------------------------------------
# Trajectory match alerts (fleet-wide)
# ---------------------------------------------------------------------------

def get_trajectory_matches_and_render_banner(featured_dfs: dict, trajectory_memory) -> dict:
    _traj_matches: dict = {}
    if trajectory_memory is not None:
        try:
            _traj_matches = cached_match_fleet(trajectory_memory, featured_dfs)
        except Exception:
            _traj_matches = {}

    # Proactive webhook push — once per cell_id per session, not on every rerun.
    _wh_url_tm  = st.session_state.get("webhook_url", "")
    _wh_evts_tm = st.session_state.get("webhook_events", [])
    if _traj_matches and _wh_url_tm and "TRAJECTORY_MATCH" in _wh_evts_tm:
        if "_alerted_trajectory_cells" not in st.session_state:
            st.session_state["_alerted_trajectory_cells"] = set()
        from notifications import send_webhook
        for _tcid, _tm in _traj_matches.items():
            if _tcid in st.session_state["_alerted_trajectory_cells"]:
                continue
            send_webhook(
                "TRAJECTORY_MATCH",
                {
                    "cell_id": _tcid, "warning_level": _tm.warning_level,
                    "best_similarity": _tm.best_similarity, "best_cell_id": _tm.best_cell_id,
                    "failure_mode": _tm.failure_mode,
                    "cycles_remaining_min": _tm.cycles_remaining_min,
                    "cycles_remaining_max": _tm.cycles_remaining_max,
                },
                _wh_url_tm, st.session_state.get("webhook_secret", ""),
            )
            st.session_state["_alerted_trajectory_cells"].add(_tcid)

    if _traj_matches:
        _tm_crit  = {c: m for c, m in _traj_matches.items() if m.warning_level == "critical"}
        _tm_high  = {c: m for c, m in _traj_matches.items() if m.warning_level == "high"}
        _tm_watch = {c: m for c, m in _traj_matches.items() if m.warning_level == "watch"}
        _tm_total = len(_traj_matches)
        _tm_header_col = "#ef4444" if _tm_crit else ("#f59e0b" if _tm_high else "#3b82f6")
        st.markdown(
            f"<div style='background:#1a1a2e;border:1px solid {_tm_header_col};"
            f"border-radius:8px;padding:14px 18px;margin-bottom:16px'>"
            f"<div style='font-size:11px;font-weight:700;letter-spacing:0.08em;"
            f"color:{_tm_header_col};margin-bottom:10px'>"
            f"⚠ FAILURE TRAJECTORY MATCHES — {_tm_total} "
            f"{'CELL' if _tm_total == 1 else 'CELLS'} FLAGGED</div>",
            unsafe_allow_html=True,
        )
        # Literal tier text alongside each icon, not color/icon alone -- see
        # overview.py's equivalent trajectory-match card (already correct)
        # and the accessibility audit that ported the pattern here.
        for _priority_set, _icon, _tier, _col in [
            (_tm_crit,  "🔴", "CRITICAL", "#fca5a5"),
            (_tm_high,  "🟠", "HIGH",     "#fcd34d"),
            (_tm_watch, "🔵", "WATCH",    "#93c5fd"),
        ]:
            for _cid, _m in _priority_set.items():
                _sim_pct = int(_m.best_similarity * 100)
                st.markdown(
                    f"<div style='display:flex;gap:10px;align-items:baseline;padding:6px 0;"
                    f"border-bottom:1px solid #1e293b'>"
                    f"<span>{_icon} {_tier}</span>"
                    f"<span style='font-weight:700;color:#e2e8f0;min-width:80px'>{_cid}</span>"
                    f"<span style='color:{_col}'>"
                    f"Matched {_m.best_cell_id} ({_sim_pct}% sim) · {_m.failure_mode}"
                    f"</span>"
                    f"<span style='color:#94a3b8;font-size:11px;margin-left:auto'>"
                    f"Est. EOL cycle {_m.predicted_eol_min}–{_m.predicted_eol_max} "
                    f"· {_m.cycles_remaining_min}–{_m.cycles_remaining_max} cycles remain"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    return _traj_matches


# ---------------------------------------------------------------------------
# T2: Proactive Alert Inbox
# ---------------------------------------------------------------------------

def render_alert_inbox(rows: list, _traj_matches: dict) -> None:
    _CYCLES_PER_MONTH = 200
    _alerts = []
    # Inject trajectory match warnings into the alert inbox
    for _cid, _m in _traj_matches.items():
        _sev = "critical" if _m.warning_level == "critical" else "high" if _m.warning_level == "high" else "medium"
        _alerts.append({
            "severity": _sev,
            "cell": _cid,
            "title": "Failure Trajectory Match",
            "body": (
                f"Degradation pattern matches {_m.n_matches} historical failure"
                f"{'s' if _m.n_matches > 1 else ''} "
                f"(best: {_m.best_cell_id}, {int(_m.best_similarity*100)}% similarity). "
                f"{_m.failure_mode}. "
                f"Est. {_m.cycles_remaining_min}–{_m.cycles_remaining_max} cycles remaining "
                f"before EOL (cycle {_m.predicted_eol_min}–{_m.predicted_eol_max})."
            ),
        })
    for r in rows:
        _cid = r["cell_id"]
        # Critical: at/past EOL
        if r["status"] == "End of Life":
            _alerts.append({"severity": "critical", "cell": _cid,
                "title": "End of Life", "body": f"SOH {r['soh']:.1f}% — replace or reassign to second-life application immediately."})
        # High: approaching EOL within 3 months
        elif r["cycles_to_eol"] is not None and r["cycles_to_eol"] <= 3 * _CYCLES_PER_MONTH:
            _alerts.append({"severity": "high", "cell": _cid,
                "title": "Replacement Due Within 3 Months", "body": f"{r['cycles_to_eol']:.0f} cycles remaining · SOH {r['soh']:.1f}%."})
        # High: past knee (accelerating fade)
        if r["knee"] and r["knee"].get("status") == "past_knee":
            _alerts.append({"severity": "high", "cell": _cid,
                "title": "Accelerated Degradation Phase", "body": f"Cell has passed its knee point — fade rate now accelerating. SOH {r['soh']:.1f}%."})
        # Medium: approaching knee
        elif r["knee"] and r["knee"].get("status") == "approaching":
            _alerts.append({"severity": "medium", "cell": _cid,
                "title": "Approaching Knee Point", "body": f"Fade acceleration detected — schedule inspection. SOH {r['soh']:.1f}%."})
        # Medium: accelerating trend without knee
        elif r["trend"] == "Accelerating" and r["status"] != "End of Life":
            _alerts.append({"severity": "medium", "cell": _cid,
                "title": "Fade Rate Accelerating", "body": f"30-cycle fade rate increased >20% vs prior window. SOH {r['soh']:.1f}%."})

    if _alerts:
        _SEV_ORDER = {"critical": 0, "high": 1, "medium": 2}
        _SEV_COLOR = {"critical": "#fc8181", "high": "#f6ad55", "medium": "#f6e05e"}
        _SEV_ICON  = {"critical": "🔴", "high": "🟠", "medium": "🟡"}
        _alerts.sort(key=lambda a: _SEV_ORDER[a["severity"]])
        st.markdown(f"<h4 class='section-header'>Alert Inbox — {len(_alerts)} Active</h4>", unsafe_allow_html=True)
        _alert_html = ""
        for _al in _alerts:
            _sc = _SEV_COLOR[_al["severity"]]
            _si = _SEV_ICON[_al["severity"]]
            _alert_html += (
                f"<div style='display:flex;align-items:flex-start;gap:12px;padding:10px 14px;"
                f"background:#1e2a38;border-left:3px solid {_sc};border-radius:0 8px 8px 0;margin-bottom:6px'>"
                f"<span style='font-size:14px;margin-top:1px'>{_si}</span>"
                f"<div><div style='font-size:12px;font-weight:700;color:{_sc}'>"
                f"{_al['severity'].upper()} · {_al['title']}"
                f"<span style='font-weight:400;color:#a0aec0;margin-left:8px'>· {_al['cell']}</span></div>"
                f"<div style='font-size:11px;color:#8896a8;margin-top:2px'>{_al['body']}</div></div></div>"
            )
        _md_html(_alert_html)


# ---------------------------------------------------------------------------
# Filter Fleet UI + Fleet Overview / Cell Grading view selector
# ---------------------------------------------------------------------------

def render_filter_and_view_selector(rows: list, featured_dfs: dict, bundles: dict) -> "list | None":
    """Returns the filtered rows list, or None if the user picked "Cell
    Grading" (already fully rendered here) or the filter left zero rows --
    both cases mean the orchestrator should stop, nothing more to render."""
    st.markdown("<h4 class='section-header'>Filter Fleet</h4>", unsafe_allow_html=True)
    _fq1, _fq2, _fq3, _fq4 = st.columns([2, 2, 2, 2])
    _fq_soh_max = _fq1.number_input(
        "SOH below (%)", min_value=50.0, max_value=100.0,
        value=float(st.session_state.get("fq_soh_max", 100.0)),
        step=1.0, key="fq_soh_max",
        help="Show only cells with SOH below this value"
    )
    _all_fades = [r["fade_30"] for r in rows if r["fade_30"] is not None]
    _fq_fade_min = _fq2.number_input(
        "Fade rate above (mSOH/cy)", min_value=0.0,
        max_value=float(max(_all_fades) * 2 if _all_fades else 1.0),
        value=float(st.session_state.get("fq_fade_min", 0.0)),
        step=0.01, format="%.3f", key="fq_fade_min",
        help="Show only cells with fade rate above this value"
    )
    _fq_status = _fq3.multiselect(
        "Status", options=["Healthy", "Degrading", "End of Life"],
        default=st.session_state.get("fq_status", ["Healthy", "Degrading", "End of Life"]),
        key="fq_status",
    )
    _fq_source = _fq4.multiselect(
        # Default to the real-measured sources (NASA + Severson), matching this
        # filter's original "real curves on first visit" intent — Severson was
        # previously missing its own bucket entirely and fell through to
        # "Uploaded", so a NASA-only default silently hid every Severson cell
        # whenever Severson was the active data source.
        "Source", options=["NASA", "Severson", "Synthetic", "Uploaded"],
        default=st.session_state.get("fq_source", ["NASA", "Severson"]),
        key="fq_source",
    )
    _fq_active = (
        _fq_soh_max < 100.0 or _fq_fade_min > 0.0
        or len(_fq_status) < 3 or len(_fq_source) < 4
    )
    _rows_before = len(rows)
    rows = [
        r for r in rows
        if r["soh"] <= _fq_soh_max
        and r["fade_30"] >= _fq_fade_min
        and r["status"] in (_fq_status or ["Healthy", "Degrading", "End of Life"])
        and r["source"] in (_fq_source or ["NASA", "Severson", "Synthetic", "Uploaded"])
    ]
    if _fq_active:
        _n_filtered = _rows_before - len(rows)
        st.caption(f"Filter active — showing {len(rows)} of {_rows_before} cells ({_n_filtered} hidden).")
    if not rows:
        _empty_state("No cells match your filter", "Adjust the filter criteria above to see results.", "", "🔍")
        return None

    _fleet_tab_sel = st.radio(
        "View",
        ["Fleet Overview", "Cell Grading"],
        horizontal=True,
        key="fleet_tab_radio",
        label_visibility="collapsed",
    )
    if _fleet_tab_sel == "Cell Grading":
        page_grading(list(featured_dfs.keys()), featured_dfs, bundles, list(featured_dfs.keys())[0])
        return None
    return rows


# ---------------------------------------------------------------------------
# Weekly digest + Health Ranking table + jump buttons + CSV export
# ---------------------------------------------------------------------------

def render_health_ranking_and_export(rows: list, _traj_matches: dict) -> None:
    # ── E6: Weekly fleet health summary card ──────────────────────────────────
    import datetime as _dt_e6
    _e6_today    = _dt_e6.date.today().strftime("%d %b %Y")
    _e6_eol_cnt  = sum(1 for r in rows if r["status"] == "End of Life")
    _e6_deg_cnt  = sum(1 for r in rows if r["status"] == "Degrading")
    _e6_accel    = sum(1 for r in rows if r["trend"] == "Accelerating")
    # Reuses _traj_matches (built above via cached_match_fleet()) instead of a
    # third per-row trajectory_memory.match() call — same scoping as the
    # original (only counts cells present in `rows`), just not recomputed.
    _e6_traj_cnt = sum(1 for r in rows if r["cell_id"] in _traj_matches)
    _REPL_USD    = 150
    _DELAY_30D   = 30  # days
    _CPD         = float(st.session_state.get("cycles_per_day", 1.0))
    _e6_at_risk  = [r for r in rows if r.get("cycles_to_eol") is not None and r["cycles_to_eol"] < _CPD * _DELAY_30D]
    _e6_capex30  = len(_e6_at_risk) * _REPL_USD
    _e6_col      = "#fc8181" if (_e6_eol_cnt > 0 or _e6_accel > 1) else ("#f6ad55" if _e6_deg_cnt > 0 else "#48bb78")
    with st.expander(f"This Week — Fleet Digest · {_e6_today}", expanded=True):
        _e6_bullets = []
        if _e6_eol_cnt:
            _e6_bullets.append(f"<strong style='color:#fc8181'>{_e6_eol_cnt} cell{'s' if _e6_eol_cnt!=1 else ''}</strong> at or past EOL — replace immediately.")
        if _e6_deg_cnt:
            _e6_bullets.append(f"<strong style='color:#f6ad55'>{_e6_deg_cnt} cell{'s' if _e6_deg_cnt!=1 else ''}</strong> degrading (80–90% SOH).")
        if _e6_accel:
            _e6_bullets.append(f"<strong style='color:#f6ad55'>{_e6_accel} cell{'s' if _e6_accel!=1 else ''}</strong> showing accelerating fade — inspect next cycle.")
        if _e6_traj_cnt:
            _e6_bullets.append(f"<strong style='color:#ef4444'>{_e6_traj_cnt} trajectory match{'es' if _e6_traj_cnt!=1 else ''}</strong> detected against known failure patterns.")
        if _e6_capex30:
            _e6_bullets.append(f"Estimated replacement CAPEX if delayed 30 days: <strong style='color:#e2e8f0'>${_e6_capex30:,}</strong> ({len(_e6_at_risk)} cell{'s' if len(_e6_at_risk)!=1 else ''} reaching EOL).")
        if not _e6_bullets:
            _e6_bullets.append("<strong style='color:#48bb78'>All cells healthy</strong> — no action required this week.")
        _bullets_html = "".join(f"<li style='margin-bottom:6px'>{b}</li>" for b in _e6_bullets)
        _md_html(
            f"<ul style='margin:0;padding-left:20px;font-size:13px;color:#a0aec0;line-height:1.7'>"
            f"{_bullets_html}</ul>"
        )

    st.markdown("<h4 class='section-header'>Health Ranking — Worst First</h4>", unsafe_allow_html=True)
    st.caption(
        "Est. RUL comes from leave-cell-out cross-validation on each cell's data source "
        "(NASA n=4, Severson n=12, synthetic n=8) — a thin population for any fleet-scale "
        "reliability claim. Treat as directional; see Overview for per-cell confidence."
    )

    STATUS_COLOUR = {"Healthy": "#48bb78", "Degrading": "#f6e05e", "End of Life": "#fc8181"}
    SOURCE_STYLE  = {
        "NASA":      "background:rgba(104,211,145,0.12);color:#48bb78;border:1px solid rgba(104,211,145,0.25)",
        "Severson":  "background:rgba(104,211,145,0.12);color:#48bb78;border:1px solid rgba(104,211,145,0.25)",
        "Synthetic": "background:rgba(74,85,104,0.3);color:#8896a8;border:1px solid #2d3748",
        "Uploaded":  "background:rgba(99,179,237,0.12);color:#63b3ed;border:1px solid rgba(99,179,237,0.25)",
    }

    table_rows_html = ""
    for rank, r in enumerate(rows, 1):
        sc = STATUS_COLOUR[r["status"]]
        ss = SOURCE_STYLE[r["source"]]
        soh_bar = int(max(0, min(100, r["soh"])))
        bar_colour = sc

        rul_cell = (
            f"{r['rul']:.0f} cy" if (r["rul"] is not None and r["rul_ok"])
            else "<span style='color:#a0aec0'>—</span>"
        )
        eol_cell = (
            f"<span style='color:#fc8181'>Reached at {r['eol_at']}</span>"
            if r["eol_at"] and r["cycles_to_eol"] == 0
            else (f"{r['cycles_to_eol']} cy" if r["cycles_to_eol"] is not None else "—")
        )

        TREND_STYLE = {
            "Accelerating": ("⚡", "#fc8181"),
            "Stable":       ("→",  "#a0aec0"),
            "Decelerating": ("↘",  "#48bb78"),
        }
        trend_icon, trend_colour = TREND_STYLE.get(r["trend"], ("→", "#a0aec0"))
        GRADE_COLOUR = {"A": "#48bb78", "B": "#ed8936", "C": "#fc8181", "—": "#4a5568"}
        gc = GRADE_COLOUR.get(r.get("grade", "—"), "#4a5568")
        grade_html = (
            f"<span style='font-size:13px;font-weight:700;color:{gc}'>{r.get('grade','—')}</span>"
        )

        table_rows_html += f"""
        <tr style="border-bottom:1px solid #1a202c">
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{rank}</td>
            <td style="padding:14px 12px">
                <span style="font-weight:600;color:#e2e8f0;font-size:14px">{r['cell_id']}</span>
                <span style="margin-left:8px;font-size:10px;padding:2px 6px;border-radius:8px;{ss}">{r['source']}</span>
            </td>
            <td style="padding:14px 12px">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="color:{sc};font-weight:700;font-size:15px;min-width:46px">{r['soh']:.1f}%</span>
                    <div style="flex:1;background:#1a202c;border-radius:3px;height:6px;min-width:80px">
                        <div style="background:{bar_colour};width:{soh_bar}%;height:6px;border-radius:3px"></div>
                    </div>
                </div>
            </td>
            <td style="padding:14px 12px">
                <span style="font-size:12px;font-weight:600;padding:2px 8px;border-radius:4px;
                             background:{sc}22;color:{sc};border:1px solid {sc}44">{r['status']}</span>
            </td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{r['cycle']:,}</td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{r['fade_30']:.2f} mSOH/cy</td>
            <td style="padding:14px 12px;color:#a0aec0;font-size:13px">{rul_cell}</td>
            <td style="padding:14px 12px;font-size:13px">{eol_cell}</td>
            <td style="padding:14px 12px;font-size:12px;color:{trend_colour}">{trend_icon} {r['trend']}</td>
            <td style="padding:14px 12px;font-size:12px;color:{'#9f7aea' if r['knee']['detected'] else '#2d3748'}">
                {'⬡ cy ' + str(r['knee']['cycle']) if r['knee']['detected'] else '—'}
            </td>
            <td style="padding:14px 12px;text-align:center">{grade_html}</td>
        </tr>
        """

    _md_html(
        f"""
        <table style="width:100%;border-collapse:collapse;font-family:sans-serif">
            <thead>
                <tr style="border-bottom:2px solid #2d3748">
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">#</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Cell</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">SOH</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Status</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Cycles</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Fade Rate</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Est. RUL</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">EOL Proximity</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Trend</th>
                    <th scope="col" style="padding:10px 12px;text-align:left;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Knee</th>
                    <th scope="col" style="padding:10px 12px;text-align:center;font-size:11px;color:#a0aec0;
                               text-transform:uppercase;letter-spacing:0.08em;font-weight:600">Grade</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
        """
    )

    # ── Click-to-navigate: jump to any cell's Health view ───────────────────
    st.markdown(
        "<div style='font-size:11px;color:#a0aec0;text-transform:uppercase;"
        "letter-spacing:0.07em;margin:12px 0 6px'>Open in Health view →</div>",
        unsafe_allow_html=True,
    )
    _jump_cols = st.columns(min(len(rows), 8))
    for _ji, _jr in enumerate(rows):
        _sc = STATUS_COLOUR[_jr["status"]]
        if _jump_cols[_ji % len(_jump_cols)].button(
            _jr["cell_id"],
            key=f"fleet_jump_{_jr['cell_id']}",
            help=f"{_jr['status']} · SOH {_jr['soh']:.1f}% · Click to open Health page",
        ):
            st.session_state["selected_cell"] = _jr["cell_id"]
            st.session_state["page"] = "health"
            st.rerun()

    # ── CSV Export ──────────────────────────────────────────────────────────
    _csv_rows = []
    for r in rows:
        _csv_rows.append({
            "cell_id": r["cell_id"], "source": r["source"],
            "soh_pct": round(r["soh"], 2), "status": r["status"],
            "cycle": r["cycle"], "fade_rate_mSOH_cy": round(r["fade_30"], 3),
            "rul_cycles": round(r["rul"], 0) if r["rul"] is not None else "",
            "eol_at_cycle": r["eol_at"] or "",
            "cycles_to_eol": r["cycles_to_eol"] or "",
            "trend": r["trend"],
            "knee_cycle": r["knee"]["cycle"] if r["knee"]["detected"] else "",
        })
    _csv_bytes = pd.DataFrame(_csv_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export fleet table as CSV",
        data=_csv_bytes,
        file_name="fleet_health_summary.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# SOH distribution bar chart
# ---------------------------------------------------------------------------

def render_soh_distribution_chart(rows: list) -> None:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    STATUS_COLOUR = {"Healthy": "#48bb78", "Degrading": "#f6e05e", "End of Life": "#fc8181"}
    sorted_ids  = [r["cell_id"] for r in rows]
    sorted_sohs = [r["soh"] for r in rows]
    bar_colours = [STATUS_COLOUR[r["status"]] for r in rows]

    fig = go.Figure(go.Bar(
        x=sorted_ids, y=sorted_sohs,
        marker_color=bar_colours,
        hovertemplate="<b>%{x}</b><br>SOH: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="#fc8181", line_width=1,
                  annotation_text="EOL (80%)", annotation_position="top right",
                  annotation_font_color="#fc8181", annotation_font_size=11)
    fig.add_hline(y=90, line_dash="dot", line_color="#f6e05e", line_width=1,
                  annotation_text="Degrading (90%)", annotation_position="top right",
                  annotation_font_color="#f6e05e", annotation_font_size=11)
    fig.update_layout(
        height=280,
        **base_layout(
            xaxis=dict(title="Cell", zeroline=False),
            yaxis=dict(title="SOH %",
                       zeroline=False, range=[50, 102]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Risk matrix: SOH vs RUL
# ---------------------------------------------------------------------------

def render_risk_matrix(rows: list) -> None:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    cal_rows   = [r for r in rows if r["rul_ok"] and r["rul"] is not None]
    uncal_rows = [r for r in rows if not (r["rul_ok"] and r["rul"] is not None)]

    if cal_rows:
        import numpy as _np
        rul_vals    = [r["rul"] for r in cal_rows]
        rul_med     = float(_np.median(rul_vals))
        soh_thresh  = 80.0   # EOL threshold (consistent with rest of platform)

        def _quadrant(soh_v, rul_v):
            h_soh = soh_v >= soh_thresh
            h_rul = rul_v >= rul_med
            if h_soh and h_rul:     return "Continue", "#48bb78"
            if h_soh and not h_rul: return "Watch",    "#d69e2e"
            if not h_soh and h_rul: return "Act",      "#f6ad55"
            return "Critical", "#fc8181"

        fig_risk = go.Figure()

        # Calibrated cells — colored by quadrant
        fig_risk.add_trace(go.Scatter(
            x=[r["soh"] for r in cal_rows],
            y=[r["rul"] for r in cal_rows],
            mode="markers+text",
            text=[r["cell_id"] for r in cal_rows],
            textposition="top center",
            textfont=dict(size=11, color="#a0aec0"),
            marker=dict(
                size=16,
                color=[_quadrant(r["soh"], r["rul"])[1] for r in cal_rows],
                line=dict(color="#1a202c", width=1),
            ),
            customdata=[[r["cell_id"], r["soh"], r["rul"], _quadrant(r["soh"], r["rul"])[0]]
                        for r in cal_rows],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "SOH: %{customdata[1]:.1f}%<br>"
                "RUL: %{customdata[2]:.0f} cycles<br>"
                "Quadrant: %{customdata[3]}<extra></extra>"
            ),
            name="Calibrated RUL",
            showlegend=False,
        ))

        # Uncalibrated cells — plotted at y=0 with X markers
        if uncal_rows:
            fig_risk.add_trace(go.Scatter(
                x=[r["soh"] for r in uncal_rows],
                y=[0] * len(uncal_rows),
                mode="markers+text",
                text=[r["cell_id"] for r in uncal_rows],
                textposition="top center",
                textfont=dict(size=11, color="#4a5568"),
                marker=dict(size=14, color="#4a5568", symbol="x", line=dict(color="#4a5568", width=2)),
                customdata=[[r["cell_id"], r["soh"]] for r in uncal_rows],
                hovertemplate="<b>%{customdata[0]}</b><br>SOH: %{customdata[1]:.1f}%<br>RUL: not calibrated<extra></extra>",
                name="RUL not calibrated",
                showlegend=bool(uncal_rows),
            ))

        # Quadrant dividers
        fig_risk.add_vline(x=soh_thresh, line_dash="dash", line_color="#fc8181", line_width=1)
        fig_risk.add_hline(y=rul_med,    line_dash="dash", line_color="#4a5568",  line_width=1)

        # Quadrant labels
        x_lo, x_hi = 55, max(r["soh"] for r in cal_rows) + 5
        y_hi        = max(rul_vals) * 1.05
        for (sx, sy, label, c) in [
            (soh_thresh - 1, rul_med + y_hi * 0.02, "INSPECT / REPLACE", "#f6ad55"),
            (soh_thresh + 1, rul_med + y_hi * 0.02, "CONTINUE",          "#48bb78"),
            (soh_thresh - 1, y_hi * 0.04,           "CRITICAL",          "#fc8181"),
            (soh_thresh + 1, y_hi * 0.04,           "WATCH",             "#d69e2e"),
        ]:
            fig_risk.add_annotation(
                x=sx, y=sy, text=label, showarrow=False,
                font=dict(size=9, color=c, family="monospace"),
                xanchor="right" if sx < soh_thresh else "left",
            )

        fig_risk.update_layout(
            height=320,
            **base_layout(
                xaxis=dict(title="SOH %",
                           zeroline=False, range=[x_lo, x_hi]),
                yaxis=dict(title="Est. RUL (cycles)",
                           zeroline=False, range=[-y_hi * 0.08, y_hi]),
            ),
        )
        fig_risk.update_layout(legend=dict(font=dict(size=11, color="#718096")))
        st.plotly_chart(fig_risk, use_container_width=True)

        if uncal_rows:
            st.markdown(
                f"<div style='font-size:11px;color:#a0aec0;margin-top:-8px'>"
                f"✕ = RUL not calibrated (LCO fold R² below 0.30) — plotted at y=0. "
                f"Quadrant split at SOH 80% and median RUL of calibrated cells ({rul_med:.0f} cy).</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Risk matrix requires at least one cell with a calibrated RUL estimate.")


# ---------------------------------------------------------------------------
# Cell-to-Cell Spread Trending
# ---------------------------------------------------------------------------

def render_spread_trending(featured_dfs: dict) -> None:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    _md_html("""<div style="font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6">A <strong style="color:#e2e8f0">rising σ(SOH)</strong> means one cell is falling behind the fleet — the earliest warning of a cell that will force a pack-level service event. When spread exceeds ~3%, investigation is warranted.</div>""")
    try:
        import numpy as _np_sp
        _nonempty = [df for df in featured_dfs.values() if len(df) > 0]
        if not _nonempty:
            raise ValueError("No cycle data available")
        _cy_min = int(max(df.iloc[0]["cycle_number"] for df in _nonempty))
        _cy_max = int(min(df.iloc[-1]["cycle_number"] for df in _nonempty))
        if _cy_max > _cy_min + 20 and len(featured_dfs) >= 2:
            _check_cycles = list(range(_cy_min, _cy_max + 1, max(1, (_cy_max - _cy_min) // 80)))
            _sigma_data = []
            for _cy in _check_cycles:
                _sohs = [
                    float(_np_sp.interp(_cy, fdf["cycle_number"].values, fdf["soh_pct"].values))
                    for fdf in featured_dfs.values()
                    if _cy >= fdf["cycle_number"].min() and _cy <= fdf["cycle_number"].max()
                ]
                if len(_sohs) >= 2:
                    _sigma_data.append({"cycle": _cy, "sigma": float(_np_sp.std(_sohs)), "n": len(_sohs)})
            if _sigma_data:
                _sd_df = pd.DataFrame(_sigma_data)
                _sigma_smoothed = pd.Series(_sd_df["sigma"]).rolling(5, min_periods=1).mean()
                _fig_spread = go.Figure()
                _fig_spread.add_trace(go.Scatter(
                    x=_sd_df["cycle"].tolist(), y=_sd_df["sigma"].tolist(),
                    name="σ(SOH) raw", mode="lines",
                    line=dict(color="#4a5568", width=1),
                    hovertemplate="Cycle %{x}: σ=%{y:.2f}%<extra>Raw spread</extra>",
                ))
                _fig_spread.add_trace(go.Scatter(
                    x=_sd_df["cycle"].tolist(), y=_sigma_smoothed.tolist(),
                    name="σ(SOH) smoothed", mode="lines",
                    line=dict(color="#63b3ed", width=2.5),
                    hovertemplate="Cycle %{x}: σ=%{y:.2f}% (smoothed)<extra></extra>",
                ))
                _fig_spread.add_hline(y=3.0, line=dict(color="#f6ad55", width=1, dash="dot"),
                                      annotation_text="3% — investigation threshold",
                                      annotation=dict(font=dict(size=9, color="#f6ad55")))
                _fig_spread.add_hline(y=5.0, line=dict(color="#fc8181", width=1, dash="dot"),
                                      annotation_text="5% — service threshold",
                                      annotation=dict(font=dict(size=9, color="#fc8181")))
                _fig_spread.update_layout(
                    **base_layout(
                        height=280, legend=LEGEND_H,
                        xaxis=dict(title="Cycle", zeroline=False),
                        yaxis=dict(title="σ(SOH) %", zeroline=False),
                    ),
                )
                _fig_spread.update_layout(title=dict(text=f"Fleet SOH Spread ({len(featured_dfs)} cells)", font=dict(size=12, color="#a0aec0"), x=0))
                st.plotly_chart(_fig_spread, use_container_width=True)
                _peak_sigma = float(_sd_df["sigma"].max())
                _trend_dir = "rising" if _sigma_smoothed.iloc[-1] > _sigma_smoothed.iloc[max(0, len(_sigma_smoothed)//2)] else "falling or stable"
                st.caption(f"Peak spread: {_peak_sigma:.2f}% SOH — current trend: {_trend_dir}.")
        else:
            _empty_state(
                "Fleet spread needs multiple cells",
                "Spread analysis requires ≥ 2 cells with overlapping cycle ranges.",
                icon="📊",
            )
    except Exception as _sp_e:
        st.info(f"Spread trend unavailable: {_sp_e}")


# ---------------------------------------------------------------------------
# SOH Distribution Shift Over Time (histogram animation)
# ---------------------------------------------------------------------------

def render_distribution_shift_histogram(featured_dfs: dict) -> None:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "The <strong style='color:#e2e8f0'>histogram shifting left</strong> is the primary signal a "
        "fleet manager watches — individual cell lines are noise; the distribution drift is the trend."
        "</div>"
    )
    try:
        import numpy as _np_hist
        _nonempty_h = [fdf for fdf in featured_dfs.values() if len(fdf) > 10 and "soh_pct" in fdf.columns and "cycle_number" in fdf.columns]
        if len(_nonempty_h) >= 2:
            _cy_min_h = int(max(df["cycle_number"].min() for df in _nonempty_h))
            _cy_max_h = int(min(df["cycle_number"].max() for df in _nonempty_h))
            _n_snapshots = 5
            _snap_cycles = [
                int(_cy_min_h + (_cy_max_h - _cy_min_h) * i / (_n_snapshots - 1))
                for i in range(_n_snapshots)
            ]
            _snap_colours = ["#4a5568", "#718096", "#63b3ed", "#f6ad55", "#fc8181"]
            # Plotly's fillcolor rejects 8-digit (CSS4-style) hex -- it only
            # accepts rgba()/rgb()/6-digit hex/named colors (confirmed via
            # its own error message) -- so translucency needs real rgba(),
            # not a hex+alpha-suffix concatenation. Same alpha (0x44/255)
            # the old string was reaching for.
            _snap_rgba = [
                "rgba(74,85,104,0.27)", "rgba(113,128,150,0.27)", "rgba(99,179,237,0.27)",
                "rgba(246,173,85,0.27)", "rgba(252,129,129,0.27)",
            ]
            _fig_hist = go.Figure()
            for _si, _snap_cy in enumerate(_snap_cycles):
                _snap_sohs = []
                for _fdf in _nonempty_h:
                    _cy_arr = _fdf["cycle_number"].values
                    _soh_arr = _fdf["soh_pct"].values
                    if _snap_cy >= _cy_arr.min() and _snap_cy <= _cy_arr.max():
                        _snap_sohs.append(float(_np_hist.interp(_snap_cy, _cy_arr, _soh_arr)))
                if len(_snap_sohs) >= 2:
                    _fig_hist.add_trace(go.Violin(
                        x=_snap_sohs,
                        name=f"Cycle {_snap_cy:,}",
                        orientation="h",
                        side="positive",
                        meanline_visible=True,
                        line_color=_snap_colours[_si],
                        fillcolor=_snap_rgba[_si],
                        opacity=0.7,
                        width=0.8,
                        points=False,
                        showlegend=True,
                    ))
            _fig_hist.update_layout(
                height=300,
                violingap=0.2,
                violinmode="overlay",
                **base_layout(
                    xaxis=dict(title="SOH %", range=[50, 105], zeroline=False),
                    yaxis=dict(visible=False),
                    legend=LEGEND_H,
                ),
            )
            _fig_hist.add_vline(x=80, line_dash="dash", line_color="#fc8181", line_width=1,
                                annotation_text="EOL 80%", annotation_font_color="#fc8181", annotation_font_size=10)
            _fig_hist.add_vline(x=90, line_dash="dot", line_color="#f6e05e", line_width=1,
                                annotation_text="Degrading 90%", annotation_font_color="#f6e05e", annotation_font_size=10)
            st.plotly_chart(_fig_hist, use_container_width=True)
            st.caption(f"Violin distribution of SOH across {len(_nonempty_h)} cells at {_n_snapshots} cycle snapshots. Left shift = fleet aging.")
        else:
            _empty_state(
                "Distribution shift needs multiple cells",
                "SOH distribution analysis requires ≥ 2 cells with cycle history.",
                icon="📊",
            )
    except Exception as _hist_e:
        st.info(f"SOH trend histogram unavailable: {_hist_e}")


# ---------------------------------------------------------------------------
# E2: Fleet what-if scenario planner
# ---------------------------------------------------------------------------

def render_whatif_scenario_planner(rows: list) -> None:
    st.markdown("<h4 class='section-header'>Fleet What-If Scenario Planner</h4>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Adjust operating conditions to see the projected impact on fleet-average RUL "
        "and replacement CAPEX. Uses the Arrhenius stress model applied to each cell's "
        "current fade rate (◐ SIMULATED — indicative only)."
        "</div>",
        unsafe_allow_html=True,
    )
    _e2c1, _e2c2 = st.columns(2)
    with _e2c1:
        _e2_crate_delta = st.slider(
            "C-rate adjustment (all cells)",
            min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
            format="%.1fC",
            key="e2_crate_delta",
            help="Positive = higher load → faster degradation. Negative = reduced load → slower.",
        )
    with _e2c2:
        _e2_temp_delta = st.slider(
            "Temperature adjustment (°C)",
            min_value=-10, max_value=10, value=0, step=1,
            key="e2_temp_delta",
            help="Arrhenius model: every +10°C roughly doubles degradation rate.",
        )
    import numpy as _np_e2, math as _math_e2
    _e2_stress_rows = []
    _REPL_COST_E2   = 150
    _CYCLES_24M_E2  = int(730 * float(st.session_state.get("cycles_per_day", 1.0)))
    for _r in rows:
        _cid       = _r["cell_id"]
        _prof      = CELL_STRESS_PROFILES.get(_cid, {})
        _t_base    = _prof.get("temp_mean", 25.0)
        _c_base    = _prof.get("c_rate", 1.0)
        _t_new     = _t_base + _e2_temp_delta
        _c_new     = max(0.1, _c_base + _e2_crate_delta)
        # Arrhenius ratio: exp(-Ea/R * (1/T_new - 1/T_base)), Ea=50 kJ/mol, R=8.314
        _e2_k_temp  = _math_e2.exp(-50000 / 8.314 * (1/(_t_new+273.15) - 1/(_t_base+273.15)))
        _e2_k_crate = (_c_new / _c_base) ** 0.7 if _c_base > 0.05 else 1.0
        _e2_stress  = _e2_k_temp * _e2_k_crate
        _fade_base  = abs(_r["fade_30"]) / 1000   # mSOH/cy → fraction
        _fade_new   = _fade_base * _e2_stress
        _soh_now    = _r["soh"]
        _eol        = float(st.session_state.get("eol_threshold_pct", 80.0))
        _rul_base   = (_soh_now - _eol) / max(_fade_base, 1e-9) if _fade_base > 0 else None
        _rul_new    = (_soh_now - _eol) / max(_fade_new, 1e-9) if _fade_new > 0 else None
        _e2_stress_rows.append({
            "cell_id": _cid, "rul_base": _rul_base, "rul_new": _rul_new,
            "fade_base": _fade_base, "fade_new": _fade_new,
        })
    _e2_rul_base  = [r["rul_base"] for r in _e2_stress_rows if r["rul_base"] is not None]
    _e2_rul_new   = [r["rul_new"]  for r in _e2_stress_rows if r["rul_new"]  is not None]
    if _e2_rul_base and _e2_rul_new:
        _e2_avg_base = float(_np_e2.mean(_e2_rul_base))
        _e2_avg_new  = float(_np_e2.mean(_e2_rul_new))
        _e2_delta    = _e2_avg_new - _e2_avg_base
        _e2_repl_base = sum(1 for r in _e2_stress_rows if r["rul_base"] is not None and r["rul_base"] < _CYCLES_24M_E2)
        _e2_repl_new  = sum(1 for r in _e2_stress_rows if r["rul_new"]  is not None and r["rul_new"]  < _CYCLES_24M_E2)
        _e2_capex_saved = (_e2_repl_base - _e2_repl_new) * _REPL_COST_E2
        _e2_col = "#48bb78" if _e2_delta >= 0 else "#fc8181"
        _e2_dir = "increase" if _e2_delta >= 0 else "decrease"
        _e2c3, _e2c4, _e2c5 = st.columns(3)
        _e2c3.metric("Fleet-avg RUL change", f"{_e2_delta:+.0f} cy",
                     help="Projected change in average fleet remaining life")
        _e2c4.metric("Replacements avoided (24 mo)", f"{_e2_repl_base - _e2_repl_new:+d}",
                     help="Cells no longer reaching EOL within 24 months under new conditions")
        _e2c5.metric("CAPEX impact (24 mo)", f"${_e2_capex_saved:+,.0f}",
                     help="Replacement cost saved (or added) at $150/cell over 24 months")
        _e2_txt = (
            f"Under the adjusted conditions (C-rate {_e2_crate_delta:+.1f}C, temp {_e2_temp_delta:+d}°C), "
            f"fleet-average RUL would <strong style='color:{_e2_col}'>{_e2_dir} by {abs(_e2_delta):.0f} cycles</strong>. "
            + (f"Estimated CAPEX saving over 24 months: <strong style='color:#48bb78'>${_e2_capex_saved:,.0f}</strong>."
               if _e2_capex_saved > 0 else
               f"Estimated additional CAPEX over 24 months: <strong style='color:#fc8181'>${abs(_e2_capex_saved):,.0f}</strong>.")
            if _e2_crate_delta != 0 or _e2_temp_delta != 0 else
            "Adjust the sliders above to model a different operating scenario."
        )
        _md_html(f"<div style='font-size:12px;color:#a0aec0;padding:8px 0 16px;line-height:1.6'>{_e2_txt}</div>")
    else:
        _empty_state(
            "RUL projection unavailable",
            "At least one cell needs a calibrated fade rate to generate fleet-wide RUL projections.",
            "Import data with ≥ 30 cycles to calibrate the fade model.",
            icon="⏱",
        )


# ---------------------------------------------------------------------------
# Second-Life Readiness Screening
# ---------------------------------------------------------------------------

def render_second_life_screening(rows: list) -> None:
    from consequences import application_fit

    st.markdown("<h4 class='section-header'>Second-Life Readiness Screening</h4>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:20px;line-height:1.6'>"
        "Cells above 85% SOH are still in primary life. Below that, each cell's fit for "
        "second-life reuse is checked against real application thresholds (SOH band + fade "
        "rate vs. the fleet), not SOH alone -- see the EOL Economics page for the full "
        "per-application breakdown. Click a cell in the sidebar to open it.</div>",
        unsafe_allow_html=True,
    )

    SL_BUCKETS = {
        "primary":    ("Primary Life",          "SOH > 85%",                 "#4a5568", "#1a202c"),
        "candidate":  ("Second-Life Fit",        "Fits at least one application", "#48bb78", "#1a2e22"),
        "below_floor":("Recycle Recommended",   "No application fit found", "#fc8181", "#2d0f0f"),
    }

    # fleet_fade_median computed once across the already-loaded rows (no
    # extra DataFrame/DB access -- consistent with this project's
    # CellSummary-based fleet-view performance discipline).
    _fleet_fades = [r["fade_30"] for r in rows if r.get("fade_30") is not None]
    _fleet_fade_median = float(pd.Series(_fleet_fades).median()) if _fleet_fades else None

    def _sl_bucket(r):
        if r["soh"] > 85.0:
            return "primary"
        # Circular Economy Coverage fix: bucketed on real application_fit()
        # results, not a flat SOH band -- a cell in the old 70-85% "always
        # candidate" window can still have every application score
        # not_fit if its fade rate is too fast relative to the fleet.
        fit = application_fit(r["soh"], r.get("fade_30") or 0.0, _fleet_fade_median)
        _, best_app = max(fit.items(), key=lambda kv: {"fit": 2, "marginal": 1, "not_fit": 0}[kv[1]["fit"]])
        return "candidate" if best_app["fit"] in ("fit", "marginal") else "below_floor"

    bucketed = {"primary": [], "candidate": [], "below_floor": []}
    for r in rows:
        bucketed[_sl_bucket(r)].append(r)

    sl_cols = st.columns(3)
    for col, (bkey, (blabel, brange, bfg, bbg)) in zip(sl_cols, SL_BUCKETS.items()):
        cells_in_bucket = bucketed[bkey]
        count = len(cells_in_bucket)
        pills = "".join(
            f"<div style='display:inline-block;margin:4px;padding:5px 12px;"
            f"background:{bfg}22;border:1px solid {bfg}44;border-radius:20px;"
            f"font-size:12px;font-weight:600;color:{bfg}'>"
            f"{r['cell_id']} <span style='font-weight:400;color:{bfg}88'>{r['soh']:.0f}%</span>"
            f"</div>"
            for r in cells_in_bucket
        ) if cells_in_bucket else (
            f"<div style='font-size:12px;color:#a0aec0;font-style:italic;padding:8px 0'>None</div>"
        )
        with col:
            _md_html(
                f"""
                <div style="background:{bbg};border:1px solid {bfg}33;border-radius:10px;padding:18px;min-height:120px">
                    <div style="font-size:10px;font-weight:700;color:{bfg};text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:4px">{blabel}</div>
                    <div style="font-size:12px;color:{bfg}88;margin-bottom:12px">{brange} · {count} cell{'s' if count != 1 else ''}</div>
                    <div style="line-height:2">{pills}</div>
                </div>
                """
            )


# ---------------------------------------------------------------------------
# "About This Ranking" methodology note + roadmap expander (static content)
# ---------------------------------------------------------------------------

def render_ranking_methodology_note() -> None:
    st.markdown("<h4 class='section-header'>About This Ranking</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        **Ranking method:** Cells are ranked by current SOH %, fade rate (30-cycle rolling average),
        and EOL proximity. SOH % is directly comparable across all cells — it's always relative
        to each cell's own initial capacity, so 75% SOH means the same thing for a NASA cell
        as for a synthetic cell.

        **Why RUL is not ranked across cell types:**
        Remaining Useful Life (cycles) is predicted by separate models for synthetic vs NASA cells,
        because the resistance features used for prediction are on incompatible measurement scales
        (synthesised internal resistance vs EIS electrolyte resistance from impedance spectroscopy).
        Comparing "300 cycles remaining" from one model to "40 cycles remaining" from another
        would be a meaningless number — the models aren't measuring the same thing.

        Individual RUL estimates are shown per cell on the Overview and Health pages (drill in
        via the cell selector in the sidebar).
        """,
    )

    with st.expander("Roadmap: Unified Ranking — what would unlock cross-type RUL comparison?"):
        st.markdown(
            """
            **Gate: 8+ real cells with diverse usage histories.**

            The current 4 NASA cells were all tested at identical lab conditions (24°C, 2A discharge).
            That means the only variation between them is cell-to-cell manufacturing spread —
            not the operating temperature, C-rate, and DoD variation that would make a combined
            resistance signal meaningful.

            Once 8 or more real cells are available with varied operating conditions, two changes
            become worthwhile:

            1. **Replace `resistance_ohm` with `resistance_normalized`** (ratio to each cell's
               own initial resistance) as the only resistance feature. Both synthetic and real
               cells start at 1.0 and rise — this is comparable across measurement methods.

            2. **Train one unified model** on the combined dataset. Validate with leave-cell-out
               to confirm it generalises across both real and synthetic cells.

            Until then, ranking by SOH and fade rate is the honest choice.
            """
        )


# ---------------------------------------------------------------------------
# Anomaly Alert History Log
# ---------------------------------------------------------------------------

def render_anomaly_alert_history(featured_dfs: dict) -> None:
    st.markdown("<h4 class='section-header'>Anomaly Alert History</h4>", unsafe_allow_html=True)
    _anom_log = []
    for _cid, _fdf in featured_dfs.items():
        if "capacity_anomaly" in _fdf.columns:
            _cap_anom = _fdf[_fdf["capacity_anomaly"] == True]
            if len(_cap_anom) > 0:
                _last10 = _fdf.iloc[-min(10, len(_fdf)):]
                _recent = int(_last10["capacity_anomaly"].sum())
                _anom_log.append({
                    "cell_id": _cid,
                    "type": "Capacity",
                    "total_flags": len(_cap_anom),
                    "last_10_cycles": _recent,
                    "last_flagged_cycle": int(_cap_anom["cycle_number"].iloc[-1]),
                    "severity": "High" if _recent >= 3 else ("Moderate" if _recent >= 1 else "Low"),
                })
        if "resistance_anomaly" in _fdf.columns:
            _res_anom = _fdf[_fdf["resistance_anomaly"] == True]
            if len(_res_anom) > 0:
                _last10r = _fdf.iloc[-min(10, len(_fdf)):]
                _recent_r = int(_last10r["resistance_anomaly"].sum())
                _anom_log.append({
                    "cell_id": _cid,
                    "type": "Resistance",
                    "total_flags": len(_res_anom),
                    "last_10_cycles": _recent_r,
                    "last_flagged_cycle": int(_res_anom["cycle_number"].iloc[-1]),
                    "severity": "High" if _recent_r >= 3 else ("Moderate" if _recent_r >= 1 else "Low"),
                })
    if _anom_log:
        _anom_df = pd.DataFrame(_anom_log).sort_values(["severity", "last_10_cycles"], ascending=[True, False])
        _SEV_COLOUR = {"High": "#fc8181", "Moderate": "#f6ad55", "Low": "#718096"}
        for _, _al in _anom_df.iterrows():
            _sc = _SEV_COLOUR[_al["severity"]]
            _md_html(
                f"<div style='display:flex;align-items:center;gap:16px;padding:10px 14px;"
                f"margin-bottom:6px;background:#1e2a38;border-radius:8px;"
                f"border-left:3px solid {_sc}'>"
                f"<span style='font-size:13px;font-weight:700;color:#e2e8f0;min-width:80px'>{_al['cell_id']}</span>"
                f"<span style='font-size:11px;padding:2px 8px;border-radius:4px;"
                f"background:{_sc}22;color:{_sc};border:1px solid {_sc}44'>{_al['type']}</span>"
                f"<span style='font-size:12px;color:#a0aec0'>Total: <strong style='color:#e2e8f0'>{_al['total_flags']}</strong> flags</span>"
                f"<span style='font-size:12px;color:#a0aec0'>Last 10 cycles: <strong style='color:{_sc}'>{_al['last_10_cycles']}</strong></span>"
                f"<span style='font-size:12px;color:#a0aec0'>Last at cycle {_al['last_flagged_cycle']}</span>"
                f"<span style='margin-left:auto;font-size:11px;font-weight:700;color:{_sc}'>{_al['severity']}</span>"
                f"</div>"
            )
    else:
        st.success("No anomaly flags detected across the fleet.")


# ---------------------------------------------------------------------------
# Orchestrator — called once from page_fleet() after the exec summary bar
# ---------------------------------------------------------------------------

def render_fleet_diagnostics(rows: list, featured_dfs: dict, bundles: dict, trajectory_memory) -> None:
    render_fleet_summary_expander(rows)
    _traj_matches = get_trajectory_matches_and_render_banner(featured_dfs, trajectory_memory)
    render_alert_inbox(rows, _traj_matches)

    rows = render_filter_and_view_selector(rows, featured_dfs, bundles)
    if rows is None:
        return

    render_health_ranking_and_export(rows, _traj_matches)
    render_soh_distribution_chart(rows)
    render_risk_matrix(rows)
    render_spread_trending(featured_dfs)
    render_distribution_shift_histogram(featured_dfs)
    render_whatif_scenario_planner(rows)
    render_second_life_screening(rows)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    render_pack_builder(featured_dfs, bundles, key_prefix="fleet")
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    render_ranking_methodology_note()
    render_anomaly_alert_history(featured_dfs)
