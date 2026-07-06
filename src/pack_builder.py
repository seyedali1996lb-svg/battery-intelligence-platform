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
