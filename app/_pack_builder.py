"""
Virtual Pack Builder — cell selection, topology, pack metrics, and trajectory divergence.

Extracted from _ui_helpers.py as a self-contained widget used by both the Fleet
page and the Explore page's Pack Builder tab.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import _paths  # noqa: F401

from _design_tokens import PACK_BUNDLE_KEY
from _ui_helpers import _md_html, _empty_state, _cell_source, base_layout


def render_pack_builder(featured_dfs: dict, bundles: dict, key_prefix: str) -> None:
    """Shared Virtual Pack Builder — cell selection, series/parallel topology,
    pack metrics, and pairwise cell-matching scores.

    key_prefix namespaces session_state keys (e.g. "fleet"/"explore") so
    both call sites keep independent selections without colliding.
    """
    from pack_builder import compute_pack_metrics, compute_matching_scores, compute_trajectory_divergence

    st.markdown("<h4 class='section-header'>Virtual Pack Builder</h4>", unsafe_allow_html=True)
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "Select cells to model as a series or parallel pack. Capacity and resistance scale "
        "differently across chemistries and sources, so selections must come from a single "
        "data source (NASA, Severson, synthetic, or uploaded)."
        "</div>"
    )

    cell_ids  = list(featured_dfs.keys())
    cells_key = f"{key_prefix}_pack_cells"
    topo_key  = f"{key_prefix}_pack_topology"

    if cells_key in st.session_state:
        st.session_state[cells_key] = [c for c in st.session_state[cells_key] if c in cell_ids]

    selected = st.multiselect(
        "Select cells for virtual pack", options=cell_ids,
        default=cell_ids[:min(4, len(cell_ids))],
        key=cells_key,
    )

    if len(selected) < 2:
        _empty_state(
            "Select at least 2 cells",
            "Choose 2 or more cells from the same data source to build a virtual pack.",
            icon="🔋",
        )
        return

    sources = {_cell_source(c) for c in selected}
    if len(sources) > 1:
        _empty_state(
            "Mixed data sources selected",
            f"Selected cells span {', '.join(sorted(sources))} — capacity and resistance "
            "scales are not comparable across chemistries/sources. Choose cells from a "
            "single source.",
            "→ Narrow your selection to one source and try again.",
            "⚠",
        )
        return

    topology = st.radio("Configuration", ["Series", "Parallel"], horizontal=True, key=topo_key)

    _bundle = bundles.get(PACK_BUNDLE_KEY.get(next(iter(sources)), "synth"))
    _per_cell_ok = (_bundle or {}).get("metrics", {}).get("per_cell_rul_reliable", {})
    _default_ok  = (_bundle or {}).get("metrics", {}).get("rul_reliable", False)

    cell_stats = []
    for cid in selected:
        df = featured_dfs.get(cid)
        if df is None or len(df) == 0:
            continue
        latest = df.iloc[-1]
        soh = float(latest["soh_pct"]) if "soh_pct" in latest.index else float("nan")
        if soh != soh:  # NaN
            continue
        cell_stats.append({
            "cell_id":        cid,
            "soh_pct":        soh,
            "capacity_ah":    float(latest["capacity_ah"]) if "capacity_ah" in latest.index else float("nan"),
            "resistance_ohm": float(latest["resistance_ohm"]) if "resistance_ohm" in latest.index else float("nan"),
            "rul_pred":       float(latest["rul_pred"]) if "rul_pred" in latest.index else None,
            "rul_reliable":   _per_cell_ok.get(cid, _default_ok),
        })

    if len(cell_stats) < 2:
        _empty_state(
            "Insufficient data",
            "Selected cells are missing capacity or SOH data at the latest cycle.",
            icon="⚠",
        )
        return

    metrics = compute_pack_metrics(cell_stats, topology)

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric(metrics["pack_soh_label"], f"{metrics['pack_soh']:.1f}%")
    _m2.metric("Pack RUL", f"{metrics['pack_rul']:.0f} cy" if metrics["pack_rul"] is not None else "—")
    _m3.metric("Pack Capacity", f"{metrics['pack_capacity_ah'] * 1000:.0f} mAh")
    _pack_res = metrics["pack_resistance_ohm"]
    _m4.metric("Pack Resistance", f"{_pack_res * 1000:.1f} mΩ" if _pack_res == _pack_res else "—")

    if metrics["spread_level"] == "Imbalanced":
        st.error(
            f"⚠️ **{metrics['bottleneck_cell_id']}** is the pack bottleneck "
            f"(SOH spread σ={metrics['soh_stdev']:.1f}%, range {metrics['soh_spread']:.1f}%). "
            f"Consider replacing or rebalancing."
        )
    elif metrics["spread_level"] == "Watch":
        st.warning(
            f"⚡ SOH spread is σ={metrics['soh_stdev']:.1f}% (range {metrics['soh_spread']:.1f}%). "
            f"Monitor {metrics['bottleneck_cell_id']} closely."
        )
    else:
        st.success(
            f"✅ Pack is well-balanced (SOH spread σ={metrics['soh_stdev']:.1f}%, "
            f"range {metrics['soh_spread']:.1f}%)"
        )
    if metrics["n_uncalibrated"]:
        st.caption(f"{metrics['n_uncalibrated']} cell(s) excluded from Pack RUL — not calibrated.")

    _traj = compute_trajectory_divergence({cid: featured_dfs.get(cid) for cid in selected})
    if _traj["widening"] and _traj["fastest_diverging_cell"]:
        _fd_cell = _traj["fastest_diverging_cell"]
        _fd_fade = _traj["fastest_diverging_fade"] * 1000
        _fd_med  = _traj["pack_median_fade"]
        _fd_ratio = f" ({_traj['fastest_diverging_fade'] / _fd_med:.1f}× pack median)" if _fd_med else ""
        st.warning(
            f"📈 SOH spread across this pack is **widening** over its shared cycling history — "
            f"**{_fd_cell}** is currently fading fastest at {_fd_fade:.2f} mAh/cycle{_fd_ratio}. "
            f"It may not be today's bottleneck yet, but it's on track to become one."
        )
    elif _traj["widening"] is False and metrics["spread_level"] != "Imbalanced":
        st.caption("Pack SOH spread has stayed stable across the cells' shared cycling history — no widening trend detected.")

    st.caption(
        f"Pack SOH is reported as {metrics['pack_soh_label'].lower()} — bottleneck-cell SOH is "
        "meaningful for series packs (usable capacity is gated by the weakest cell); "
        "capacity-weighted average is meaningful for parallel packs (capacity sums across cells)."
    )

    _soh_values = [c["soh_pct"] for c in cell_stats]
    _bar_colors = []
    for _sv in _soh_values:
        _dist = abs(_sv - metrics["pack_soh"])
        _bar_colors.append("#48bb78" if _dist <= 2 else ("#f6ad55" if _dist <= 5 else "#fc8181"))
    _fig_pack = go.Figure(go.Bar(
        x=[c["cell_id"] for c in cell_stats], y=_soh_values,
        marker_color=_bar_colors,
        hovertemplate="<b>%{x}</b><br>SOH: %{y:.1f}%<extra></extra>",
    ))
    _fig_pack.add_hline(
        y=metrics["pack_soh"], line_dash="dash", line_color="#63b3ed", line_width=1,
        annotation_text=f"Pack SOH {metrics['pack_soh']:.1f}%", annotation_font_color="#63b3ed",
    )
    _fig_pack.update_layout(**base_layout(height=250, yaxis=dict(title="SOH %", range=[50, 102])))
    st.plotly_chart(_fig_pack, use_container_width=True)

    with st.expander("Cell matching & per-cell breakdown", expanded=False):
        st.caption(
            "Cells with similar degradation trajectories are better matched for pack "
            "assembly (minimises balancing losses)."
        )
        match_rows = compute_matching_scores(cell_stats)
        if match_rows:
            st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(cell_stats).set_index("cell_id"), use_container_width=True)
