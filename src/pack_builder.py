"""
Shared Virtual Pack Builder calculations — used by both the Fleet page and
the Explore page's Pack Builder tab, which previously had two (Fleet
actually had two internally) independent, organically-diverged
implementations of this same math. Pure logic, no Streamlit dependency,
so it's independently testable and the UI (app/utils.py's
render_pack_builder()) stays a thin rendering layer.
"""

import statistics


def compute_pack_metrics(cell_stats: list, topology: str) -> dict:
    """
    cell_stats: list of {"cell_id", "soh_pct", "capacity_ah", "resistance_ohm",
                          "rul_pred", "rul_reliable"} — one entry per selected cell.
    topology: "Series" or "Parallel".

    Series: pack SOH = weakest cell's SOH (bottleneck framing — usable capacity
    is gated by the weakest cell). Parallel: pack SOH = capacity-weighted average
    (capacity sums across cells, so a strong cell's larger share of the pack's
    energy dominates). Both framings are physically meaningful; only the one
    matching the selected topology is surfaced as "pack_soh".
    """
    soh_values = [c["soh_pct"] for c in cell_stats]
    cap_values = [c["capacity_ah"] for c in cell_stats]
    res_values = [c["resistance_ohm"] for c in cell_stats]
    has_resistance = all(r == r and r > 0 for r in res_values)  # no NaN, no zero

    bottleneck_idx = soh_values.index(min(soh_values))
    bottleneck_cell_id = cell_stats[bottleneck_idx]["cell_id"]

    if topology == "Series":
        pack_soh = min(soh_values)
        pack_soh_label = "Bottleneck-cell SOH"
        pack_capacity_ah = min(cap_values)
        pack_resistance_ohm = sum(res_values) if has_resistance else float("nan")
    else:
        total_cap = sum(cap_values)
        pack_soh = (
            sum(s * c for s, c in zip(soh_values, cap_values)) / total_cap
            if total_cap else float("nan")
        )
        pack_soh_label = "Capacity-weighted avg SOH"
        pack_capacity_ah = total_cap
        pack_resistance_ohm = (
            1.0 / sum(1.0 / r for r in res_values) if has_resistance else float("nan")
        )

    rul_values = [
        c["rul_pred"] for c in cell_stats
        if c.get("rul_reliable") and c.get("rul_pred") is not None
    ]
    pack_rul = min(rul_values) if rul_values else None
    n_uncalibrated = len(cell_stats) - len(rul_values)

    soh_spread = max(soh_values) - min(soh_values)
    soh_stdev = statistics.stdev(soh_values) if len(soh_values) > 1 else 0.0
    if soh_stdev < 2:
        spread_level, spread_color = "Balanced", "#48bb78"
    elif soh_stdev < 5:
        spread_level, spread_color = "Watch", "#f6ad55"
    else:
        spread_level, spread_color = "Imbalanced", "#fc8181"

    return {
        "pack_soh": pack_soh,
        "pack_soh_label": pack_soh_label,
        "pack_capacity_ah": pack_capacity_ah,
        "pack_resistance_ohm": pack_resistance_ohm,
        "pack_rul": pack_rul,
        "n_uncalibrated": n_uncalibrated,
        "bottleneck_cell_id": bottleneck_cell_id,
        "soh_spread": soh_spread,
        "soh_stdev": soh_stdev,
        "spread_level": spread_level,
        "spread_color": spread_color,
    }


def compute_trajectory_divergence(cell_frames: dict) -> dict:
    """
    cell_frames: {cell_id: DataFrame} — each cell's FULL featured history
    (cycle_number, soh_pct, fade_rate_30cy columns), not just the latest
    snapshot compute_pack_metrics() uses. compute_pack_metrics()'s
    soh_spread/soh_stdev are a single cross-sectional snapshot — two cells
    can show an identical spread today while one arrived there by a slow,
    stable fade and the other by a fade rate that's actively accelerating
    away from the pack. This detects that difference: whether the pack's
    SOH spread is *widening* over the cells' shared cycling history, and
    which cell is fading fastest right now even if it isn't today's
    bottleneck yet.

    Comparison is restricted to the cycle range every selected cell has
    actually reached (common_min_cycle..common_max_cycle) — comparing a
    cell's cycle-900 state against another cell's cycle-200 state would
    conflate "further into life" with "genuinely diverging faster".

    Returns:
      widening:               True/False, or None if there isn't enough
                               shared history (fewer than 2 cells with
                               overlapping cycle ranges) to judge a trend.
      spread_trend:            soh_stdev at each checkpoint (may contain NaN
                               where a checkpoint had fewer than 2 cells).
      checkpoint_cycles:       cycle numbers the checkpoints were taken at
                               (25/50/75/100% of the shared range).
      fastest_diverging_cell:  cell_id with the highest current fade_rate_30cy
                               among the selected cells, or None.
      fastest_diverging_fade:  that cell's fade_rate_30cy value.
      pack_median_fade:        median fade_rate_30cy across selected cells,
                               for comparison against fastest_diverging_fade.
    """
    empty = {
        "widening": None, "spread_trend": [], "checkpoint_cycles": [],
        "fastest_diverging_cell": None, "fastest_diverging_fade": None,
        "pack_median_fade": None,
    }
    valid = {
        cid: df for cid, df in cell_frames.items()
        if df is not None and len(df) > 0
        and "cycle_number" in df.columns and "soh_pct" in df.columns
    }
    if len(valid) < 2:
        return empty

    common_max_cycle = min(int(df["cycle_number"].max()) for df in valid.values())
    common_min_cycle = max(int(df["cycle_number"].min()) for df in valid.values())
    if common_max_cycle <= common_min_cycle:
        return empty

    checkpoint_cycles = sorted(set(
        int(common_min_cycle + f * (common_max_cycle - common_min_cycle))
        for f in (0.25, 0.5, 0.75, 1.0)
    ))

    spread_trend = []
    for cy in checkpoint_cycles:
        vals = []
        for df in valid.values():
            sub = df[df["cycle_number"] <= cy]
            if len(sub) > 0:
                vals.append(float(sub["soh_pct"].iloc[-1]))
        spread_trend.append(statistics.stdev(vals) if len(vals) >= 2 else float("nan"))

    valid_trend = [v for v in spread_trend if v == v]  # drop NaN
    widening = None
    if len(valid_trend) >= 2:
        if valid_trend[-1] <= 1e-9:
            widening = False  # still ~0 spread at the end — clearly not widening
        elif valid_trend[0] <= 1e-9:
            widening = True   # went from ~0 spread to a real one
        else:
            # 15% growth threshold — small enough to catch a real trend, large
            # enough to not flag ordinary rolling-window noise as "widening".
            widening = valid_trend[-1] > valid_trend[0] * 1.15

    fades = {}
    for cid, df in valid.items():
        if "fade_rate_30cy" in df.columns:
            recent = df[df["cycle_number"] <= common_max_cycle]
            if len(recent) > 0 and recent["fade_rate_30cy"].notna().any():
                fades[cid] = float(recent["fade_rate_30cy"].dropna().iloc[-1])

    fastest_diverging_cell = fastest_diverging_fade = pack_median_fade = None
    if fades:
        pack_median_fade = statistics.median(fades.values())
        fastest_diverging_cell = max(fades, key=fades.get)
        fastest_diverging_fade = fades[fastest_diverging_cell]

    return {
        "widening": widening,
        "spread_trend": spread_trend,
        "checkpoint_cycles": checkpoint_cycles,
        "fastest_diverging_cell": fastest_diverging_cell,
        "fastest_diverging_fade": fastest_diverging_fade,
        "pack_median_fade": pack_median_fade,
    }


def compute_matching_scores(cell_stats: list) -> list:
    """
    Pairwise 0-100 "how well-matched are these two cells for pack assembly"
    score — penalizes SOH/capacity/resistance mismatch (mismatched cells
    force a BMS to derate the whole pack to protect the weakest one).
    Returns one row per unique pair, each with a plain-English recommendation.
    """
    rows = []
    n = len(cell_stats)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cell_stats[i], cell_stats[j]
            soh_diff = abs(a["soh_pct"] - b["soh_pct"])
            cap_diff = abs(a["capacity_ah"] - b["capacity_ah"]) / (a["capacity_ah"] + 1e-9) * 100
            res_diff = abs(a["resistance_ohm"] - b["resistance_ohm"]) / (a["resistance_ohm"] + 1e-9) * 100
            score = max(0, min(100, 100 - (soh_diff * 2 + cap_diff * 1.5 + res_diff * 0.5)))
            recommendation = (
                "Excellent match" if score > 80 else
                "Good match" if score > 60 else
                "Acceptable" if score > 40 else
                "Poor — avoid pairing"
            )
            rows.append({
                "Cell A": a["cell_id"], "Cell B": b["cell_id"],
                "Match Score": f"{score:.0f}", "Recommendation": recommendation,
            })
    return rows
