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
from _pack_layout_3d import render_pack_layout_3d, soh_band_color
from _ui_helpers import _md_html, _empty_state, _cell_source, base_layout


def render_pack_builder(featured_dfs: dict, bundles: dict, key_prefix: str) -> None:
    """Shared Virtual Pack Builder — cell selection, series/parallel topology,
    pack metrics, and pairwise cell-matching scores.

    key_prefix namespaces session_state keys (e.g. "fleet"/"explore") so
    both call sites keep independent selections without colliding.
    """
    from pack_builder import (
        build_parallel_groups, compute_matching_scores, compute_pack_metrics,
        compute_trajectory_divergence, compute_xpys_metrics,
    )

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

    topology = st.radio(
        "Configuration", ["Series", "Parallel", "XpYs"], horizontal=True, key=topo_key,
        help="Series: one string. Parallel: one block sharing a node pair. XpYs: Y parallel "
             "groups of up to X cells each, strung in series — the only configuration that "
             "raises the question of how current divides inside a group.",
    )

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

    # ── XpYs: how many cells sit in parallel per group ─────────────────────
    groups = None
    cells_per_group = None
    if topology == "XpYs":
        _per_group_key = f"{key_prefix}_pack_cells_per_group"
        _max_per_group = len(cell_stats)
        _stored = st.session_state.get(_per_group_key)
        if not isinstance(_stored, int) or not 1 <= _stored <= _max_per_group:
            st.session_state[_per_group_key] = min(4, _max_per_group)
        cells_per_group = st.slider(
            "Max cells in parallel per group (X)",
            min_value=1, max_value=_max_per_group, key=_per_group_key,
            help="Up to X cells in parallel, Y = ceil(N / X) groups in series. The packer "
                 "balances capacity within that cap, so groups can come out smaller and equal "
                 "(4 cells at 3p pack as 2p2s rather than 3p1s, which would be gated by the "
                 "one-cell group). X=1 is the plain series configuration and X=N the plain "
                 "parallel one — a real XpYs build is in between.",
        )
        groups = build_parallel_groups(cell_stats, cells_per_group)
        metrics = compute_xpys_metrics(cell_stats, groups)
        _sizes = [len(g) for g in groups]
        _config = (
            f"{_sizes[0]}p{len(groups)}s" if len(set(_sizes)) == 1
            else f"{min(_sizes)}-{max(_sizes)}p{len(groups)}s"
        )
        _ragged = (
            f" ⚠ {_max_per_group} cells at {cells_per_group}p does not divide evenly — the "
            f"smaller group(s) {_sizes} gate the string; pick an X that divides "
            f"{_max_per_group}."
            if len(set(_sizes)) > 1 else ""
        )
        st.caption(f"**{_config}** — {_max_per_group} cells in {len(groups)} parallel group(s) "
                   f"of up to {cells_per_group}, strung in series.{_ragged}")
    else:
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

    if groups:
        st.caption(
            f"Pack SOH is reported as {metrics['pack_soh_label'].lower()} — a series string's "
            "usable capacity is gated by its weakest group, and each group's SOH is the "
            "capacity-weighted average of the cells that share it (the same framing the "
            "pack builder has always used for a plain parallel block). Pack RUL is the weakest "
            "calibrated cell's."
        )
    else:
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

    # ── XpYs: how the packer grouped the cells, and how current divides ────
    if groups:
        _render_group_sharing(cell_stats, metrics)

    # ── What that load ratio does over time ────────────────────────────────
    _aging_groups = groups if groups else (
        [list(range(len(cell_stats)))] if topology == "Parallel"
        else [[i] for i in range(len(cell_stats))]
    )
    _render_load_aging(
        cell_stats, _aging_groups, featured_dfs, key_prefix=key_prefix, topology=topology,
    )

    # ── 3D layout: the same cells as cylinders, with the wiring drawn ──
    render_pack_layout_3d(
        cell_stats, topology, metrics, key_prefix=key_prefix,
        groups=groups, cells_per_group=cells_per_group,
    )

    with st.expander("Cell matching & per-cell breakdown", expanded=False):
        st.caption(
            "Cells with similar degradation trajectories are better matched for pack "
            "assembly (minimises balancing losses)."
        )
        match_rows = compute_matching_scores(cell_stats)
        if match_rows:
            st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(cell_stats).set_index("cell_id"), use_container_width=True)


def _fmt_metric(value, scale: float = 1.0, unit: str = "", decimals: int = 1) -> str:
    """Format one metric for a table cell; an em dash when it isn't available."""
    try:
        num = float(value) * scale
    except (TypeError, ValueError):
        return "—"
    if num != num:
        return "—"
    rendered = f"{num:,.{decimals}f}"
    if not unit:
        return rendered
    return f"{rendered}%" if unit == "%" else f"{rendered} {unit}"


def _render_load_aging(cell_stats: list, groups: list, featured_dfs: dict, *,
                       key_prefix: str, topology: str) -> None:
    """Turn the current-sharing load ratio into an aging consequence.

    The load ratio is a instantaneous number — "this cell carries 1.10× its
    share right now". This section projects what that does to the pack: each
    cell stepped forward over a scenario horizon twice, once with the loading
    feedback and once without it, using two measured inputs from each cell's
    own record (its fade slope and its resistance growth per SOH point) and one
    stated assumption (a current→fade exponent, defaulted to the same sub-linear
    0.7 the platform's own ``stress_index`` uses).

    Renders nothing but a scope note when no two selected cells share a node —
    a series string loads every cell identically, so there is no load ratio to
    turn into anything.
    """
    from pack_aging import (
        DEFAULT_CURRENT_AGING_EXPONENT, DEFAULT_HORIZON_CYCLES,
        measure_cell_aging_inputs, simulate_load_aging,
    )

    if not any(len(g) > 1 for g in groups):
        st.caption(
            "Load-driven divergence is not reported for a series configuration: every cell "
            "carries the same current, so there is no load ratio to turn into an aging "
            "consequence. It applies to cells sharing a node — the Parallel topology, or a "
            "parallel group of an XpYs build."
        )
        return

    st.markdown("<h4 class='section-header'>Load-driven divergence</h4>", unsafe_allow_html=True)
    _md_html(
        "<div style='font-size:13px;color:#8896a8;margin-bottom:14px;line-height:1.6'>"
        "The load ratio above is an instant, not a consequence. This projects each selected "
        "cell forward over a scenario horizon twice — once with the loading feedback (a cell "
        "carrying more current fades faster, so its resistance separates from its group-mates, "
        "which moves the shares again) and once without it — and reports the difference."
        "</div>"
    )

    # ── Measured inputs, one per selected cell, from its own record ────────
    enriched = []
    for stat in cell_stats:
        cid = stat["cell_id"]
        measured = measure_cell_aging_inputs(featured_dfs.get(cid), cid)
        entry = dict(stat)
        latest_soh = measured["soh_now"]
        if latest_soh == latest_soh:
            entry["soh_pct"] = latest_soh
        latest_resistance = measured["resistance_ohm"]
        if latest_resistance == latest_resistance:
            entry["resistance_ohm"] = latest_resistance
        entry.update({
            "fade_pct_per_cycle": measured["fade_pct_per_cycle"],
            "fade_r2": measured["fade_r2"],
            "fade_cycles": measured["fade_cycles"],
            "fade_recent_pct_per_cycle": measured["fade_recent_pct_per_cycle"],
            "resistance_gain_ohm_per_soh_pct": measured["resistance_gain_ohm_per_soh_pct"],
            "resistance_gain_corr": measured["resistance_gain_corr"],
            "resistance_gain_measured": measured["resistance_gain_measured"],
            "unavailable": measured["unavailable"],
        })
        enriched.append(entry)

    _h_key = f"{key_prefix}_pack_aging_horizon"
    _a_key = f"{key_prefix}_pack_aging_exponent"
    _c1, _c2 = st.columns(2)
    horizon = _c1.slider(
        "Scenario horizon (cycles)", min_value=25, max_value=500, step=25,
        value=int(DEFAULT_HORIZON_CYCLES), key=_h_key,
        help="How far forward to step the pack. An arbitrary, disclosed scenario length — "
             "not a lifetime forecast, and nothing here is fed into Pack RUL.",
    )
    exponent = _c2.slider(
        "Current→fade exponent (α)", min_value=0.0, max_value=1.5, step=0.1,
        value=float(DEFAULT_CURRENT_AGING_EXPONENT), key=_a_key,
        help="Fade ∝ (load ratio)^α. The default 0.7 is the sub-linear C-rate term the "
             "platform's own stress_index feature already applies; it is an assumption "
             "borrowed for consistency, not a parameter fitted to this fleet. α=0 turns "
             "the loading feedback off.",
    )

    result = simulate_load_aging(enriched, groups, horizon_cycles=float(horizon),
                                 exponent=float(exponent))
    if not result["available"]:
        st.info(
            "Load-driven divergence is not reported for this build: "
            f"{result.get('unavailable_reason') or 'no two selected cells share a node'}."
        )
        if result["excluded"]:
            st.caption(
                "Cells that could not be projected: "
                + "; ".join(f"{e['cell_id']} ({e['reason']})" for e in result["excluded"])
            )
        return

    if not any(g["sharing_modelled"] for g in result["groups"]):
        st.info(
            "No two selected cells with usable resistance share a node in this configuration, "
            "so every cell carries the same current and there is no load-driven divergence to "
            "estimate."
        )
        if result["excluded"]:
            st.caption(
                "Cells that could not be projected: "
                + "; ".join(f"{e['cell_id']} ({e['reason']})" for e in result["excluded"])
            )
        return

    loaded_id = result["loaded_cell_id"]
    ratio = result["loaded_load_ratio"]
    loaded = next(c for c in result["cells"] if c["cell_id"] == loaded_id)

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Most-loaded cell", f"{loaded_id}")
    _m2.metric("Its load / fair share", f"{ratio:.2f}×")
    _accel = result["loaded_fade_acceleration_pct"]
    _m3.metric("Its fade acceleration", f"{_accel:+.1f}%" if _accel == _accel else "—")
    _m4.metric(
        f"Extra SOH lost by cycle {horizon}",
        f"{result['loaded_extra_loss_pct']:.2f} pts",
        help="With the loading feedback minus without it, for the most-loaded cell.",
    )

    colors = {c["cell_id"]: soh_band_color(c["soh_start"]) for c in result["cells"]}
    fig = go.Figure()
    for cell in result["cells"]:
        color = colors[cell["cell_id"]]
        label = cell["cell_id"] + (f" · G{cell['group']}" if len(result["groups"]) > 1 else "")
        fig.add_trace(go.Scatter(
            x=cell["cycles"], y=cell["soh_with"], mode="lines",
            name=f"{label} — with loading",
            line=dict(color=color, width=2.2),
            hovertemplate=(f"<b>{label}</b> with loading<br>cycle %{{x:.0f}}<br>"
                           "SOH %{y:.2f}%<extra></extra>"),
        ))
        fig.add_trace(go.Scatter(
            x=cell["cycles"], y=cell["soh_without"], mode="lines",
            name=f"{label} — no loading feedback",
            line=dict(color=color, width=1.2, dash="dot"),
            hovertemplate=(f"<b>{label}</b> no feedback<br>cycle %{{x:.0f}}<br>"
                           "SOH %{y:.2f}%<extra></extra>"),
        ))
    fig.add_hline(
        y=result["eol_soh_pct"], line_dash="dash", line_color="#fc8181", line_width=1,
        annotation_text=f"EOL {result['eol_soh_pct']:.0f}%", annotation_font_color="#fc8181",
    )
    fig.update_layout(**base_layout(
        height=340, hovermode="closest",
        xaxis=dict(title="Cycles from now (scenario)"),
        yaxis=dict(title="SOH (%)"),
        legend=dict(orientation="v", y=1.0, x=1.02, xanchor="left",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#a0aec0")),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ── The finding, in one paragraph ─────────────────────────────────────
    _direction = "rises" if loaded["load_relief_pct"] > 0 else "falls"
    st.caption(
        f"**{loaded_id}** carries **{ratio:.2f}×** its fair share of its group's current, "
        f"which by this model accelerates its fade by **{_accel:+.1f}%** — **"
        f"{result['loaded_extra_loss_pct']:.2f} SOH points** over {horizon} cycles, or about "
        f"**{result['loaded_equivalent_cycles']:.1f} cycles** of its own measured baseline fade "
        f"({loaded['fade_pct_per_cycle']:.3f} %/cycle) spent early. Its share then "
        f"{_direction} to {loaded['load_ratio_end']:.2f}× as the cells' resistances separate — "
        f"that drift is read from each cell's own measured dR/dSOH, not assumed: a cell whose "
        f"*absolute* resistance grows faster sheds load, and having a low resistance is not on "
        f"its own enough to guarantee it does."
    )

    _delta = result["spread_delta_pct"]
    _median_soh = sorted(c["soh_start"] for c in result["cells"])[len(result["cells"]) // 2]
    if _delta == _delta and abs(_delta) >= 0.005:
        _verb = "narrows" if _delta < 0 else "widens"
        _toward = "above" if loaded["soh_start"] >= _median_soh else "below"
        st.caption(
            f"Across the selection the SOH spread goes from **{result['spread_without_pct']:.2f}%** "
            f"without the feedback to **{result['spread_with_pct']:.2f}%** with it — the feedback "
            f"**{_verb}** the pack's spread by {abs(_delta):.2f} points, and "
            f"**{result['bottleneck_without']}** is the bottleneck cell in both runs. The loaded "
            f"cell started {_toward} the selection's median SOH, so the extra aging it takes "
            f"{'pulls it toward' if _delta < 0 else 'pushes it away from'} the rest. Current "
            f"sharing redistributes aging rather than simply adding it: the hardest-worked cell "
            f"is often the strong one, because low resistance goes with high SOH."
        )
    else:
        st.caption(
            f"Across the selection the SOH spread is essentially unchanged by the feedback "
            f"({result['spread_without_pct']:.2f}% → {result['spread_with_pct']:.2f}%) — with "
            f"these cells the load imbalance is too small for the redistribution to move it."
        )

    # ── Assumptions, the evidence behind them, and the per-cell numbers ───
    with st.expander("Scenario assumptions, measured inputs & per-cell projection"):
        _spread_fig = go.Figure()
        _spread_fig.add_trace(go.Scatter(
            x=result["spread_curve_cycles"], y=result["spread_curve_without"],
            mode="lines", name="No loading feedback",
            line=dict(color="#63b3ed", width=1.6, dash="dot"),
            hovertemplate="No feedback<br>cycle %{x:.0f}<br>σ %{y:.2f}%<extra></extra>",
        ))
        _spread_fig.add_trace(go.Scatter(
            x=result["spread_curve_cycles"], y=result["spread_curve_with"],
            mode="lines", name="With loading feedback",
            line=dict(color="#f6ad55", width=1.8),
            hovertemplate="With feedback<br>cycle %{x:.0f}<br>σ %{y:.2f}%<extra></extra>",
        ))
        _spread_fig.update_layout(**base_layout(
            height=280,
            xaxis=dict(title="Cycles from now (scenario)"),
            yaxis=dict(title="SOH spread across the selection (σ, %)"),
            legend=dict(orientation="h", y=1.15, x=1, xanchor="right",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#a0aec0")),
        ))
        st.plotly_chart(_spread_fig, use_container_width=True)

        st.dataframe(pd.DataFrame([
            {
                "Cell": c["cell_id"],
                "Group": f"G{c['group']}",
                "Load start": f"{c['load_ratio_start']:.2f}×",
                "Load at horizon": f"{c['load_ratio_end']:.2f}×",
                "Fade accel": f"{c['fade_acceleration_pct']:+.1f}%",
                "Baseline fade": f"{c['fade_pct_per_cycle']:.3f} %/cy",
                "Extra SOH lost": f"{c['extra_loss_pct']:.2f} pts",
                "Equiv. cycles": f"{c['equivalent_cycles_lost']:.1f}",
                "dR/dSOH": (
                    f"{c['resistance_gain_ohm_per_soh_pct'] * 1000:.2f} mΩ/pt"
                    if c["resistance_gain_ohm_per_soh_pct"] is not None else "— (held constant)"
                ),
                "SOH at horizon (with / without)": (
                    f"{c['soh_with_end']:.2f}% / {c['soh_without_end']:.2f}%"
                ),
            }
            for c in result["cells"]
        ]), use_container_width=True, hide_index=True)

        st.markdown(
            """
**Where the inputs come from (measured).** Each cell's fade slope is an ordinary
least-squares fit of its own `soh_pct` against `cycle_number` over the cycles it
recorded — *not* the rolling `fade_rate_30cy` column, which is a 30-cycle mean of
|Δcapacity| and on these cells reads roughly twice the true long-run slope because
per-cycle measurement noise dominates it. Each cell's dR/dSOH is an OLS fit of its own
`resistance_ohm` against `soh_pct`. Both refuse rather than guess: a flat or rising SOH
trend, fewer than 10 positive-resistance points, or a resistance/SOH correlation weaker
than −0.5 means the input is reported as unavailable and named here, never substituted.

**What is assumed, and why.** The one number the data cannot supply is α, the
current→fade exponent: this fleet is cycled at one duty cycle, so it holds no variation
in load to identify a rate exponent from. The default 0.7 is the same sub-linear
`(C/1C)^0.7` term the platform's own `stress_index` feature already applies
(`batlab/features/engineering.py`, with its stated Doyle-Fuller-Newman rationale for a
non-uniform current distribution in a porous electrode), borrowed so the app has one
convention rather than two. Move the slider to see the whole conclusion scale with it. **Fair share = the cells' historical duty**: a load
ratio of 1.0 means the pack is driven so an evenly-split group reproduces the current the
cell was actually cycled at; a pack driven harder adds an acceleration this comparison
does not show, because it would shift both runs by the same factor. One consequence of
α < 1 is worth knowing before reading the numbers: the relationship is concave, so
concentrating current in one cell costs the *group* marginally less total fade than
splitting it evenly — the redistribution is not zero-sum. (For α > 1 the same model says
the opposite; both signs are pinned by tests.)

**Why the difference is more trustworthy than the absolute numbers.** Both runs share the
same fade model, so the gap between them is attributable to loading rather than to the
model's imperfections. Fade is held constant per cycle — no knee acceleration is
extrapolated — and real fade accelerates with age, so the extra loss here is a **lower
bound**. Conversely, a cell whose resistance growth could not be measured is projected
with a *constant* resistance, so its load never drifts: the **upper bound** for that cell.
See the `dR/dSOH` column for which is which.

**Not modelled.** Sharing is the first-order, resistance-only, steady-state model from
the section above: no SOC-dependent OCV differences, no thermal feedback, no
contact/wiring resistance. Thermal feedback in particular runs *the other way* — a loaded
cell warms and its resistance falls, which widens the imbalance, so the self-limiting
drift this projection sometimes shows is optimistic on a real pack. Nothing here is a
lifetime prediction, and nothing here feeds Pack RUL or any published accuracy number.
"""
        )
        if result["excluded"]:
            st.caption(
                "Cells excluded from the projection (no measured fade slope or no usable "
                "resistance): "
                + "; ".join(f"{e['cell_id']} — {e['reason']}" for e in result["excluded"])
            )
        _notes = []
        for cell in enriched:
            for what, why in cell.get("unavailable") or []:
                _notes.append(f"{cell['cell_id']}: {what} — {why}")
        if _notes:
            st.caption("Measured inputs that were refused: " + "; ".join(_notes))


def _render_group_sharing(cell_stats: list, metrics: dict) -> None:
    """XpYs: the group composition table and the current-sharing chart.

    The chart is the honest core of the XpYs view: it plots each cell's share of
    its group's current, computed from resistance alone (see
    src/pack_builder.compute_group_current_sharing), against the fair 1/X split.
    Any bar above the line is a cell doing more than its share of the work.
    """
    rows = metrics.get("groups") or []
    soh_by_cell = {str(c.get("cell_id")): c.get("soh_pct", float("nan")) for c in cell_stats}

    st.markdown("<h4 class='section-header'>Parallel groups & current sharing</h4>",
                unsafe_allow_html=True)

    if metrics.get("current_sharing_available"):
        labels, shares, colors, fair = [], [], [], []
        for row in rows:
            for cell_id, share in (row.get("shares_pct") or {}).items():
                labels.append(f"G{row.get('group')} · {cell_id}")
                shares.append(float(share))
                colors.append(soh_band_color(soh_by_cell.get(cell_id, float("nan"))))
                fair.append(float(row.get("equal_share_pct", float("nan"))))

        fig_share = go.Figure()
        fig_share.add_trace(go.Bar(
            x=labels, y=shares, marker_color=colors, name="Share of group current",
            text=[f"{s:.1f}%" for s in shares], textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:.1f}% of its group's current<extra></extra>",
        ))
        fig_share.add_trace(go.Scatter(
            x=labels, y=fair, mode="lines+markers", name="Fair share (1/X)",
            line=dict(color="#63b3ed", width=1, dash="dash"), marker=dict(size=6),
            hovertemplate="Fair share: %{y:.1f}%<extra></extra>",
        ))
        fig_share.update_layout(**base_layout(
            height=280,
            hovermode="closest",
            xaxis=dict(title="", tickangle=-30),
            yaxis=dict(title="Share of group current (%)"),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#a0aec0")),
        ))
        st.plotly_chart(fig_share, use_container_width=True)

        loaded = metrics.get("most_loaded_cell_id")
        ratio = metrics.get("max_overload_ratio")
        if loaded and ratio == ratio:
            st.caption(
                f"**{loaded}** sits furthest above the line: at any given group current it "
                f"carries **{ratio:.2f}× its fair share**, because it has the lowest resistance "
                f"in its group. A group current that would be 1C per cell in a perfectly "
                f"balanced group therefore loads that cell at {ratio:.2f}C."
            )
        st.caption(
            "First-order, resistance-only, steady-state model: cells in parallel sit at one "
            "voltage, so current divides inversely with DC resistance. Not modelled: "
            "SOC-dependent OCV differences, temperature feedback (the loaded cell warms and its "
            "resistance falls, which is destabilizing), contact/wiring resistance, or the "
            "differential aging this loading causes. The direction it does capture is "
            "self-limiting — a cell whose resistance has risen with age draws *less* current."
        )
    else:
        st.info(
            "Current sharing is not reported for this build: "
            f"{metrics.get('current_sharing_reason') or 'a parallel group needs at least two cells with resistance data'}."
        )

    with st.expander("Group composition & per-group metrics", expanded=False):
        st.dataframe(pd.DataFrame([
            {
                "Group": f"G{row.get('group')}" + (" — bottleneck" if row.get("is_bottleneck") else ""),
                "Cells": ", ".join(row.get("cell_ids") or []),
                "Capacity": _fmt_metric(row.get("capacity_ah"), 1000.0, "mAh", 0),
                "Group SOH": _fmt_metric(row.get("soh_pct"), 1.0, "%", 1),
                "Resistance": _fmt_metric(row.get("resistance_ohm"), 1000.0, "mΩ", 2),
                "Weakest cell": row.get("weakest_cell_id") or "—",
                "Most loaded": (
                    f"{row.get('max_share_cell_id')} "
                    f"({_fmt_metric(row.get('max_share_pct'), 1.0, '%', 1)})"
                    if row.get("max_share_cell_id") and row.get("n_cells", 0) > 1 else "—"
                ),
                "Load ratio": (
                    f"{row['overload_ratio']:.2f}×"
                    if row.get("n_cells", 0) > 1 and row.get("overload_ratio") == row.get("overload_ratio")
                    else "—"
                ),
                "Member capacity spread": _fmt_metric(row.get("capacity_imbalance_pct"), 1.0, "%", 1),
            }
            for row in rows
        ]), use_container_width=True, hide_index=True)
        st.caption(
            "Groups are packed to balance capacity — the determinant of a series string's usable "
            "energy, which is why cells are binned at all. The packing shown is this tool's own "
            "greedy result (largest cells first, each into the lightest group), not an optimal "
            "solution, and it does not attempt to equalize group resistance or impedance."
        )
