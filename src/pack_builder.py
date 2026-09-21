"""
Shared Virtual Pack Builder calculations — used by both the Fleet page and
the Explore page's Pack Builder tab, which previously had two (Fleet
actually had two internally) independent, organically-diverged
implementations of this same math. Pure logic, no Streamlit dependency,
so it's independently testable and the UI (app/utils.py's
render_pack_builder()) stays a thin rendering layer.

Three topologies are supported. ``compute_pack_metrics()`` covers the two flat
ones (all-series, all-parallel). ``build_parallel_groups()`` +
``compute_xpys_metrics()`` cover a real **XpYs** build — X cells in parallel per
group, Y groups in series — and are *strict supersets* of the flat two:
``build_parallel_groups(cells, 1)`` is the all-series configuration and
``build_parallel_groups(cells, len(cells))`` is the all-parallel one, with
identical pack SOH/capacity/resistance to ``compute_pack_metrics()`` (pinned by
tests). The XpYs path is the one that adds the two things a flat topology cannot
say: which parallel group gates the string, and how current actually divides
between the cells that share a node.
"""

import math
import statistics

# Grouping-quality thresholds, as % of the mean group capacity. Deliberately the
# same 2/5 split the SOH spread verdict uses (see compute_pack_metrics), so the UI
# can label both with one convention instead of inventing a second scale.
BINNING_SPREAD_BALANCED_PCT = 2.0
BINNING_SPREAD_WATCH_PCT = 5.0


def _finite(value, default: float = float("nan")) -> float:
    """``float(value)`` when it is a real number, else ``default``."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return num if num == num else default


def _is_finite(value) -> bool:
    """True when ``value`` is a real number (not NaN, None, or non-numeric)."""
    num = _finite(value, float("nan"))
    return num == num


def _usable_resistance(value) -> "float | None":
    """A resistance that can carry current sharing: finite and strictly positive.

    Non-positive readings are sentinel/parse artefacts rather than measurements
    (Severson's raw r1 log parses to 0.0 on some cycles), so they are treated as
    "not available" rather than as a short circuit.
    """
    num = _finite(value, float("nan"))
    if num != num or num <= 0.0:
        return None
    return num


def build_parallel_groups(cell_stats: list, cells_per_group: int) -> list[list[int]]:
    """Partition cells into parallel groups of at most ``cells_per_group``.

    Objective, stated rather than implied: **balance group capacity**. A series
    string is gated by its weakest group and a parallel group's capacity is the
    sum of its members', so equalizing group capacity maximizes usable pack
    capacity — the reason cell manufacturers bin cells into groups at all. This
    is what the binning does; it does *not* attempt to equalize group
    resistance or impedance, which is a different (and partly conflicting)
    objective.

    Algorithm: greedy longest-processing-time — cells in descending capacity
    order, each placed into the currently lightest eligible group (ties broken
    by fewest cells, then by group index, so the same selection always packs the
    same way). Not an optimal makespan solver; a deterministic bin-packing
    heuristic whose result is shown to the user rather than hidden.

    Returns a list of groups of *indices* into ``cell_stats``, in series order
    (group 0 is the negative end). ``cells_per_group=1`` gives the all-series
    configuration, ``>= len(cells)`` the all-parallel one.
    """
    n_cells = len(cell_stats)
    if n_cells == 0:
        return []
    per_group = max(1, int(cells_per_group))
    n_groups = math.ceil(n_cells / per_group)
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    group_capacity = [0.0] * n_groups

    order = sorted(
        range(n_cells),
        key=lambda i: (-_finite(cell_stats[i].get("capacity_ah"), 0.0),
                       str(cell_stats[i].get("cell_id", ""))),
    )
    for idx in order:
        eligible = [g for g in range(n_groups) if len(groups[g]) < per_group]
        if not eligible:  # defensive: cannot happen while n_groups = ceil(n/per)
            eligible = list(range(n_groups))
        target = min(eligible, key=lambda g: (group_capacity[g], len(groups[g]), g))
        groups[target].append(idx)
        group_capacity[target] += _finite(cell_stats[idx].get("capacity_ah"), 0.0)
    return groups


def compute_group_current_sharing(member_cells: list) -> dict:
    """How DC current divides between the cells of one parallel group.

    Cells in parallel sit at the same terminal voltage, so at a given instant the
    current divides inversely with each cell's resistance:

        share_c = (1 / R_c) / Σ_g (1 / R_g)

    Scope, stated because it bounds every number this returns: this is a
    first-order, resistance-only, steady-state model. It does not model
    SOC-dependent OCV differences, temperature feedback (the loaded cell warms
    and its resistance falls, which is destabilizing), contact/wiring
    resistance, or the differential aging this loading causes. Note the
    direction it *does* capture: a cell whose resistance has risen with age
    draws **less** current here, which is self-limiting — the opposite of a
    runaway.

    Returns shares as percentages, plus the load ratio of the hardest-worked
    cell against a fair (equal) split. ``unavailable_reason`` is set — and
    ``shares`` left empty — when any member's resistance is missing or
    non-positive, rather than substituting a value.
    """
    n_members = len(member_cells)
    result = {
        "shares_pct": {}, "equal_share_pct": (100.0 / n_members) if n_members else float("nan"),
        "max_share_cell_id": None, "max_share_pct": float("nan"),
        "min_share_pct": float("nan"), "overload_ratio": float("nan"),
        "unavailable_reason": None,
    }
    if n_members == 0:
        result["unavailable_reason"] = "no cells in this group"
        return result

    resistances: list[float] = []
    missing = []
    for cell in member_cells:
        value = _usable_resistance(cell.get("resistance_ohm"))
        if value is None:
            missing.append(str(cell.get("cell_id", "?")))
            resistances.append(float("nan"))
        else:
            resistances.append(value)
    if missing:
        result["unavailable_reason"] = (
            f"resistance unavailable for {len(missing)} of {n_members} cell(s) "
            f"({', '.join(missing[:4])}{'…' if len(missing) > 4 else ''})"
        )
        return result
    if n_members == 1:
        # A group of one has nothing to share out.
        only = member_cells[0]
        result["shares_pct"] = {str(only.get("cell_id", "?")): 100.0}
        result["max_share_cell_id"] = str(only.get("cell_id", "?"))
        result["max_share_pct"] = result["min_share_pct"] = 100.0
        result["overload_ratio"] = 1.0
        return result

    conductance = [1.0 / r for r in resistances]
    total = sum(conductance)
    if total <= 0.0:  # unreachable given the positivity check, but keep it total
        result["unavailable_reason"] = "no usable resistance in this group"
        return result

    shares = {str(cell.get("cell_id", "?")): (cond / total) * 100.0
              for cell, cond in zip(member_cells, conductance)}
    hardest = max(shares, key=lambda cid: shares[cid])  # pyright: ignore[reportArgumentType, reportCallIssue]
    result["shares_pct"] = shares
    result["max_share_cell_id"] = hardest
    result["max_share_pct"] = shares[hardest]
    result["min_share_pct"] = min(shares.values())
    result["overload_ratio"] = shares[hardest] / (100.0 / n_members)
    return result


def compute_xpys_metrics(cell_stats: list, groups: list[list[int]]) -> dict:
    """Pack metrics for an XpYs build (X parallel per group, Y groups in series).

    The generalization this encodes, made explicit because it is what makes the
    two flat topologies special cases rather than duplicates:

    - Parallel group (X cells): capacity **adds** (charge shares), SOH is the
      capacity-weighted average of its members, resistance is the parallel
      combination 1/Σ(1/R). A group's weakest member is flagged for reference.
    - Series string (Y groups): capacity is the **minimum** group capacity (the
      string is gated by its weakest group), SOH is the minimum group SOH, and
      resistance is the sum.

    With X=1 this reduces exactly to compute_pack_metrics(..., "Series") and with
    Y=1 to compute_pack_metrics(..., "Parallel") — both pinned by tests, so the
    new topology cannot silently disagree with the two the app already served.

    Returns compute_pack_metrics()'s key set (so the existing metric tiles read
    it unchanged) plus the group breakdown, the binning-quality verdict, and the
    current-sharing summary.
    """
    if not cell_stats or not groups:
        return {
            "pack_soh": float("nan"), "pack_soh_label": "Bottleneck-group SOH",
            "pack_capacity_ah": float("nan"), "pack_resistance_ohm": float("nan"),
            "pack_rul": None, "n_uncalibrated": 0, "bottleneck_cell_id": None,
            "bottleneck_group": None, "soh_spread": float("nan"),
            "soh_stdev": float("nan"), "spread_level": "Balanced",
            "spread_color": "#48bb78", "groups": [], "n_groups": 0,
            "group_capacity_spread_ah": float("nan"),
            "group_capacity_spread_pct": float("nan"),
            "binning_level": "Balanced", "binning_color": "#48bb78",
            "most_loaded_cell_id": None, "most_loaded_group": None,
            "max_overload_ratio": float("nan"),
            "current_sharing_available": False,
            "current_sharing_reason": "no cells selected",
        }

    group_rows = []
    for group_index, members in enumerate(groups):
        member_cells = [cell_stats[i] for i in members]
        capacities = [_finite(c.get("capacity_ah"), float("nan")) for c in member_cells]
        sohs = [_finite(c.get("soh_pct"), float("nan")) for c in member_cells]
        valid_cap = [(s, q) for s, q in zip(sohs, capacities) if s == s and q == q]
        group_capacity = sum(q for _s, q in valid_cap) if valid_cap else float("nan")
        group_soh = (
            sum(s * q for s, q in valid_cap) / group_capacity
            if valid_cap and group_capacity > 0 else float("nan")
        )
        resistances = [_usable_resistance(c.get("resistance_ohm")) for c in member_cells]
        complete = [r for r in resistances if r is not None]
        group_resistance = (
            1.0 / sum(1.0 / r for r in complete)
            if complete and len(complete) == len(resistances) and complete else float("nan")
        )
        weak_candidates = [c for c in member_cells if _is_finite(c.get("soh_pct"))]
        weakest = (
            min(weak_candidates, key=lambda c: _finite(c.get("soh_pct"), float("inf")))
            if weak_candidates else (member_cells[0] if member_cells else None)
        )
        sharing = compute_group_current_sharing(member_cells)
        finite_caps = [q for q in capacities if q == q]
        mean_cap = (sum(finite_caps) / len(finite_caps)) if finite_caps else float("nan")
        capacity_imbalance_pct = (
            (max(finite_caps) - min(finite_caps)) / mean_cap * 100.0
            if finite_caps and mean_cap else float("nan")
        )
        group_rows.append({
            "group": group_index + 1,
            "cell_ids": [str(c.get("cell_id", "?")) for c in member_cells],
            "n_cells": len(member_cells),
            "capacity_ah": group_capacity,
            "soh_pct": group_soh,
            "resistance_ohm": group_resistance,
            "weakest_cell_id": str(weakest.get("cell_id", "?")) if weakest else None,
            "capacity_imbalance_pct": capacity_imbalance_pct,
            "shares_pct": sharing["shares_pct"],
            "equal_share_pct": sharing["equal_share_pct"],
            "max_share_cell_id": sharing["max_share_cell_id"],
            "max_share_pct": sharing["max_share_pct"],
            "min_share_pct": sharing["min_share_pct"],
            "overload_ratio": sharing["overload_ratio"],
            "sharing_unavailable_reason": sharing["unavailable_reason"],
            "is_bottleneck": False,
        })

    usable_groups = [g for g in group_rows if _is_finite(g["capacity_ah"])] or group_rows
    bottleneck = min(usable_groups, key=lambda g: _finite(g["capacity_ah"], float("inf")))
    bottleneck["is_bottleneck"] = True

    group_capacities = [_finite(g["capacity_ah"], float("nan")) for g in group_rows]
    finite_group_caps = [q for q in group_capacities if q == q]
    group_spread = (max(finite_group_caps) - min(finite_group_caps)) if finite_group_caps else float("nan")
    mean_group_cap = (sum(finite_group_caps) / len(finite_group_caps)) if finite_group_caps else 0.0
    spread_pct = (group_spread / mean_group_cap * 100.0) if mean_group_cap else 0.0
    if spread_pct <= BINNING_SPREAD_BALANCED_PCT:
        binning_level, binning_color = "Balanced", "#48bb78"
    elif spread_pct <= BINNING_SPREAD_WATCH_PCT:
        binning_level, binning_color = "Watch", "#f6ad55"
    else:
        binning_level, binning_color = "Imbalanced", "#fc8181"

    soh_values = [_finite(c.get("soh_pct"), float("nan")) for c in cell_stats]
    finite_sohs = [s for s in soh_values if s == s]
    soh_spread = (max(finite_sohs) - min(finite_sohs)) if finite_sohs else float("nan")
    soh_stdev = statistics.stdev(finite_sohs) if len(finite_sohs) > 1 else 0.0
    if soh_stdev < 2:
        spread_level, spread_color = "Balanced", "#48bb78"
    elif soh_stdev < 5:
        spread_level, spread_color = "Watch", "#f6ad55"
    else:
        spread_level, spread_color = "Imbalanced", "#fc8181"

    rul_values = [
        c["rul_pred"] for c in cell_stats
        if c.get("rul_reliable") and c.get("rul_pred") is not None
    ]
    group_sohs = [_finite(g["soh_pct"], float("nan")) for g in group_rows]
    finite_group_sohs = [s for s in group_sohs if s == s]
    group_resistances = [_finite(g["resistance_ohm"], float("nan")) for g in group_rows]

    sharing_groups = [g for g in group_rows if g["shares_pct"]]
    loaded = [g for g in sharing_groups if g["max_share_cell_id"] is not None and g["n_cells"] > 1]
    hardest = max(loaded, key=lambda g: _finite(g["overload_ratio"], 0.0), default=None)

    return {
        "pack_soh": min(finite_group_sohs) if finite_group_sohs else float("nan"),
        "pack_soh_label": "Bottleneck-group SOH",
        "pack_capacity_ah": _finite(bottleneck["capacity_ah"], float("nan")),
        "pack_resistance_ohm": (
            sum(group_resistances) if all(r == r for r in group_resistances) else float("nan")
        ),
        "pack_rul": min(rul_values) if rul_values else None,
        "n_uncalibrated": len(cell_stats) - len(rul_values),
        "bottleneck_cell_id": bottleneck["weakest_cell_id"],
        "bottleneck_group": bottleneck["group"],
        "soh_spread": soh_spread,
        "soh_stdev": soh_stdev,
        "spread_level": spread_level,
        "spread_color": spread_color,
        "groups": group_rows,
        "n_groups": len(group_rows),
        "group_capacity_spread_ah": group_spread,
        "group_capacity_spread_pct": spread_pct,
        "binning_level": binning_level,
        "binning_color": binning_color,
        "most_loaded_cell_id": hardest["max_share_cell_id"] if hardest else None,
        "most_loaded_group": hardest["group"] if hardest else None,
        "max_overload_ratio": _finite(hardest["overload_ratio"], float("nan")) if hardest else float("nan"),
        "current_sharing_available": bool(sharing_groups),
        "current_sharing_reason": (
            None if sharing_groups
            else (group_rows[0].get("sharing_unavailable_reason") if group_rows
                  else "no cells selected")
        ),
    }


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
        fastest_diverging_cell = max(fades, key=fades.get)  # pyright: ignore[reportArgumentType, reportCallIssue]
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
