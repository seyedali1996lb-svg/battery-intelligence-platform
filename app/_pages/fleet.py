"""Page: Fleet.

Everything below the exec summary bar (Fleet Summary, alerts, ranking
table, charts, what-if planner, second-life screening, pack builder,
anomaly log) lives in _fleet_diagnostics.py -- this file was 1,210 lines,
the 2nd largest in app/_pages/. page_fleet() keeps the always-visible
hero (ask-the-fleet, daily digest webhook, per-cell `rows` computation --
the single source of truth every summary on this page reads from --
header metrics, exec summary bar) and delegates the rest.
"""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

from utils import (
    _action_bar, _md_html, _empty_state,
    soh_status, render_card, cached_detect_knee,
)
from design_system import make_badge
from chemistry_profiles import ChemistryProfile

from _pages._fleet_diagnostics import render_fleet_diagnostics


# ---------------------------------------------------------------------------
# Page: Fleet
# ---------------------------------------------------------------------------

def page_fleet(featured_dfs: dict, bundles: dict, trajectory_memory: "TrajectoryMemory | None" = None):
    _action_bar("fleet")
    st.markdown("# Which cells need attention this week?")

    # ── U5: "Ask the fleet" natural-language front door ──────────────────────
    _af_input = st.text_input(
        "Ask the fleet", placeholder="e.g. 'What are the current fleet alerts?' or "
        "'What will replacement cost over the next 12 months?'",
        key="fleet_ask_input",
    )
    st.caption("Routes to grounded fleet-level answers — not open-ended reasoning. If your question doesn't match a known topic, it says so rather than guessing.")
    if _af_input:
        from battery_copilot import build_fleet_stats, answer_fleet_query
        _af_stats = build_fleet_stats(featured_dfs, bundles)
        _af_answer = answer_fleet_query(_af_input, _af_stats)
        render_card(
            f"<div style='font-size:10px;color:#4a5568;margin-bottom:8px'>{make_badge('Template', '#718096')} · Fleet</div>"
            f"<div style='font-size:13px;color:#e2e8f0;line-height:1.7'>{_af_answer}</div>",
            padding="16px 20px",
            extra_style="margin-bottom:16px",
        )

    # ── Best-effort daily fleet digest (session/page-load-triggered, not a
    # real background cron — this app has no daemon process available) ──────
    import db as _db_fleet
    _wh_url_f  = st.session_state.get("webhook_url", "")
    _wh_evts_f = st.session_state.get("webhook_events", [])
    if _wh_url_f and "FLEET_DIGEST" in _wh_evts_f:
        _today = datetime.date.today().isoformat()
        if _db_fleet.get_setting(st.session_state["auth_org_id"], "last_digest_sent") != _today:
            from notifications import send_webhook
            _n_cells_f = len(featured_dfs)
            _n_flagged_f = sum(
                1 for _df in featured_dfs.values()
                if len(_df) and "soh_pct" in _df.columns
                and float(_df["soh_pct"].iloc[-1]) < st.session_state.get("eol_threshold_pct", 80.0)
            )
            send_webhook(
                "FLEET_DIGEST",
                {"n_cells": _n_cells_f, "n_flagged_below_eol": _n_flagged_f, "trigger": "fleet_page_load"},
                _wh_url_f, st.session_state.get("webhook_secret", ""),
            )
            _db_fleet.set_setting(st.session_state["auth_org_id"], "last_digest_sent", _today)

    # ── Build fleet summary row per cell ──
    # Bundle and per-cell reliability lookup is source-aware; uploaded cells use
    # the "upload" bundle, NASA cells the "nasa" bundle, synthetic the "synth" bundle.
    # This is computed once, before any rendering, and every summary on this
    # page (exec bar below, header metric chips, ranking table) reads from
    # this same `rows` list — previously the exec bar independently recomputed
    # its own stats straight from featured_dfs, which could silently diverge
    # from the ranking table whenever a cell's bundle lookup failed (that
    # exact divergence was a real bug found in review: the exec bar showed
    # populated stats while the ranking table said "No cells loaded").
    import numpy as _np_fleet
    rows = []

    def _bundle_for_cell(cid: str) -> dict | None:
        return bundles.get(ChemistryProfile.for_cell(cid).source_kind)

    for cell_id, df in featured_dfs.items():
        bndl      = _bundle_for_cell(cell_id)
        if bndl is None:
            continue
        per_cell  = bndl["metrics"].get("per_cell_rul_reliable", {})
        rul_ok    = per_cell.get(cell_id, bndl["metrics"].get("rul_reliable", False))
        _profile  = ChemistryProfile.for_cell(cell_id)
        latest    = df.iloc[-1]
        soh       = latest["soh_pct"]
        cycle     = int(latest["cycle_number"])
        fade_30   = latest.get("fade_rate_30cy", float("nan")) * 1000  # mSOH/cy
        rul       = latest["rul_pred"] if rul_ok else None
        eol_row   = df[df["is_eol"]]
        eol_at    = int(eol_row["cycle_number"].iloc[0]) if len(eol_row) else None
        cycles_to_eol = max(0, eol_at - cycle) if eol_at else None

        status_label, _ = soh_status(soh)

        # Knee-point detection per cell (cached — see utils.cached_detect_knee)
        knee_result = cached_detect_knee(df["soh_pct"], df["cycle_number"])

        # Degradation trend: compare current 30-cy fade rate vs 30 cycles earlier
        trend = "Stable"
        if "fade_rate_30cy" in df.columns and len(df) >= 31:
            fade_now  = df["fade_rate_30cy"].iloc[-1]
            fade_prev = df["fade_rate_30cy"].iloc[-31]
            delta_pct = (fade_now - fade_prev) / (abs(fade_prev) + 1e-9) * 100
            if delta_pct > 20:
                trend = "Accelerating"
            elif delta_pct < -20:
                trend = "Decelerating"

        # A5: early-cycle grade (Severson method, first 100 cycles)
        _early = df[df["cycle_number"] <= 100]
        if len(_early) >= 20 and "capacity_ah" in _early.columns:
            _cap0  = float(_early["capacity_ah"].iloc[0])
            _res0  = max(float(_early["resistance_ohm"].iloc[0]), 1e-6) if "resistance_ohm" in _early.columns else 1e-6
            _gfade = (float(_early["capacity_ah"].iloc[0]) - float(_early["capacity_ah"].iloc[-1])) / len(_early)
            _gvar  = float(_early["capacity_ah"].var())
            _gslope = float(_np_fleet.polyfit(_early["cycle_number"], _early["resistance_ohm"], 1)[0]) if "resistance_ohm" in _early.columns else 0.0
            _gfp   = _gfade / _cap0 * 100
            _gcv2  = _gvar / (_cap0 ** 2) * 1e4
            _grsp  = abs(_gslope) / _res0 * 100
            _gscore = float(_np_fleet.clip(100 - _gfp * 400 - _gcv2 * 8 - _grsp * 150, 0, 100))
            _grade  = "A" if _gscore >= 75 else ("B" if _gscore >= 50 else "C")
        else:
            _grade  = "—"

        rows.append({
            "cell_id":      cell_id,
            "source":       _profile.source_label,
            "soh":          soh,
            "status":       status_label,
            "cycle":        cycle,
            "fade_30":      fade_30,
            "rul":          rul,
            "rul_ok":       rul_ok,
            "eol_at":       eol_at,
            "cycles_to_eol": cycles_to_eol,
            "trend":        trend,
            "knee":         knee_result,
            "grade":        _grade,
        })

    # Sort: worst SOH first (most urgent)
    rows.sort(key=lambda r: r["soh"])

    if not rows:
        _empty_state(
            "No cells loaded",
            "The fleet view requires at least one cell to be loaded. "
            "Switch data source in the sidebar or import a cell on the Import page.",
            "→ Go to Import",
            "○",
        )
        return

    # ── Header metrics ──
    n_eol       = sum(1 for r in rows if r["status"] == "End of Life")
    n_degrading = sum(1 for r in rows if r["status"] == "Degrading")
    n_healthy   = sum(1 for r in rows if r["status"] == "Healthy")
    worst_soh   = rows[0]["soh"]
    best_soh    = rows[-1]["soh"]
    n_nasa      = sum(1 for r in rows if r["source"] == "NASA")
    n_synth     = sum(1 for r in rows if r["source"] == "Synthetic")
    n_severson  = sum(1 for r in rows if r["source"] == "Severson")
    n_upload    = sum(1 for r in rows if r["source"] == "Uploaded")
    src_parts   = []
    if n_synth:    src_parts.append(f"{n_synth} synthetic")
    if n_nasa:     src_parts.append(f"{n_nasa} NASA real")
    if n_severson: src_parts.append(f"{n_severson} Severson real")
    if n_upload:   src_parts.append(f"{n_upload} uploaded")
    src_sub = " · ".join(src_parts) or "—"

    # ── Executive summary bar (always visible) ──────────────────────────────
    # Derived from the same `rows`/n_eol/n_degrading/n_healthy computed above
    # — not a second, independent pass over featured_dfs — so this bar and
    # the ranking table below can never show a different cell count again.
    _fscore    = sum(r["soh"] for r in rows) / len(rows)
    _score_col = "#48bb78" if _fscore >= 90 else ("#ed8936" if _fscore >= 80 else "#fc8181")
    _REPL_COST = 150
    _capex12   = (n_eol + n_degrading) * _REPL_COST

    _fe1, _fe2, _fe3, _fe4, _fe5 = st.columns(5)
    with _fe1:
        _md_html(
            f"<div style='text-align:center;padding:8px 0'>"
            f"<div style='font-size:10px;color:#4a5568;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin-bottom:4px'>Fleet Health</div>"
            f"<div style='font-size:36px;font-weight:900;color:{_score_col};line-height:1'>"
            f"{_fscore:.1f}%</div>"
            f"<div style='font-size:11px;color:#718096;margin-top:2px'>"
            f"{n_healthy} healthy · {n_degrading} degrading · {n_eol} EOL"
            f"</div></div>"
        )
    with _fe2:
        st.metric("Cells at Risk", f"{n_eol + n_degrading}", f"of {len(rows)}")
    with _fe3:
        st.metric("CAPEX (12 mo)", f"${_capex12:,.0f}", f"{n_eol + n_degrading} replacements")
    with _fe4:
        if n_eol:
            st.error(f"{n_eol} cell{'s' if n_eol != 1 else ''} need immediate replacement")
        elif n_degrading:
            st.warning(f"{n_degrading} cell{'s' if n_degrading != 1 else ''} degrading")
        else:
            st.success("All cells healthy")
    with _fe5:
        if st.button("Business analysis →", key="fleet_to_copilot_biz", use_container_width=True):
            st.session_state.page = "copilot"
            st.session_state.copilot_query = "fleet_risk"
            st.rerun()

    st.markdown(
        "<div style='height:1px;background:#2d3748;margin:16px 0'></div>",
        unsafe_allow_html=True,
    )

    _md_html(
        f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-label">Total Cells</div>
                <div class="metric-chip-value">{len(rows)}</div>
                <div class="metric-chip-sub">{src_sub}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">End of Life</div>
                <div class="metric-chip-value" style="color:#fc8181">{n_eol}</div>
                <div class="metric-chip-sub">below 80% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Degrading</div>
                <div class="metric-chip-value" style="color:#f6e05e">{n_degrading}</div>
                <div class="metric-chip-sub">80–90% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Healthy</div>
                <div class="metric-chip-value" style="color:#48bb78">{n_healthy}</div>
                <div class="metric-chip-sub">above 90% SOH</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Fleet SOH Range</div>
                <div class="metric-chip-value" style="font-size:20px">{worst_soh:.0f}–{best_soh:.0f}%</div>
                <div class="metric-chip-sub">worst to best</div>
            </div>
        </div>
        """
    )

    render_fleet_diagnostics(rows, featured_dfs, bundles, trajectory_memory)
