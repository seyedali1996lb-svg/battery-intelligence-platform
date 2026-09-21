"""Unit tests for src/pack_builder.py's pure calculation logic."""

import itertools
import math

import pytest

from conftest import make_cycles_df
from batlab.features.engineering import build_features
from pack_builder import (
    build_parallel_groups,
    compute_group_current_sharing,
    compute_matching_scores,
    compute_pack_metrics,
    compute_trajectory_divergence,
    compute_xpys_metrics,
)


def _stats(cell_id, soh, cap, res, rul=None, rul_ok=False):
    return {
        "cell_id": cell_id, "soh_pct": soh, "capacity_ah": cap,
        "resistance_ohm": res, "rul_pred": rul, "rul_reliable": rul_ok,
    }


def test_series_pack_soh_is_bottleneck_cell():
    cells = [_stats("A", 90.0, 2.0, 0.05), _stats("B", 70.0, 2.0, 0.05)]
    m = compute_pack_metrics(cells, "Series")
    assert m["pack_soh"] == 70.0
    assert m["pack_soh_label"] == "Bottleneck-cell SOH"
    assert m["bottleneck_cell_id"] == "B"


def test_series_pack_capacity_is_minimum():
    cells = [_stats("A", 90.0, 2.5, 0.05), _stats("B", 90.0, 1.8, 0.05)]
    m = compute_pack_metrics(cells, "Series")
    assert m["pack_capacity_ah"] == 1.8


def test_series_pack_resistance_is_sum():
    cells = [_stats("A", 90.0, 2.0, 0.05), _stats("B", 90.0, 2.0, 0.03)]
    m = compute_pack_metrics(cells, "Series")
    assert abs(m["pack_resistance_ohm"] - 0.08) < 1e-9


def test_parallel_pack_soh_is_capacity_weighted_average():
    cells = [_stats("A", 100.0, 3.0, 0.05), _stats("B", 80.0, 1.0, 0.05)]
    m = compute_pack_metrics(cells, "Parallel")
    # (100*3 + 80*1) / 4 = 95.0
    assert abs(m["pack_soh"] - 95.0) < 1e-9
    assert m["pack_soh_label"] == "Capacity-weighted avg SOH"


def test_parallel_pack_capacity_is_sum():
    cells = [_stats("A", 90.0, 2.0, 0.05), _stats("B", 90.0, 1.5, 0.05)]
    m = compute_pack_metrics(cells, "Parallel")
    assert abs(m["pack_capacity_ah"] - 3.5) < 1e-9


def test_parallel_pack_resistance_is_harmonic_sum():
    cells = [_stats("A", 90.0, 2.0, 0.1), _stats("B", 90.0, 2.0, 0.1)]
    m = compute_pack_metrics(cells, "Parallel")
    # 1 / (1/0.1 + 1/0.1) = 0.05
    assert abs(m["pack_resistance_ohm"] - 0.05) < 1e-9


def test_missing_resistance_on_any_cell_yields_nan_pack_resistance():
    cells = [_stats("A", 90.0, 2.0, float("nan")), _stats("B", 90.0, 2.0, 0.05)]
    m = compute_pack_metrics(cells, "Series")
    assert m["pack_resistance_ohm"] != m["pack_resistance_ohm"]  # NaN


def test_pack_rul_excludes_uncalibrated_cells():
    cells = [
        _stats("A", 90.0, 2.0, 0.05, rul=300, rul_ok=True),
        _stats("B", 85.0, 2.0, 0.05, rul=50, rul_ok=False),
    ]
    m = compute_pack_metrics(cells, "Series")
    assert m["pack_rul"] == 300
    assert m["n_uncalibrated"] == 1


def test_pack_rul_is_none_when_no_cell_calibrated():
    cells = [_stats("A", 90.0, 2.0, 0.05, rul=300, rul_ok=False)]
    m = compute_pack_metrics(cells, "Series")
    assert m["pack_rul"] is None


def test_spread_level_thresholds():
    balanced = [_stats("A", 90.0, 2.0, 0.05), _stats("B", 89.0, 2.0, 0.05)]
    watch    = [_stats("A", 90.0, 2.0, 0.05), _stats("B", 85.0, 2.0, 0.05)]
    bad      = [_stats("A", 95.0, 2.0, 0.05), _stats("B", 60.0, 2.0, 0.05)]
    assert compute_pack_metrics(balanced, "Series")["spread_level"] == "Balanced"
    assert compute_pack_metrics(watch, "Series")["spread_level"] == "Watch"
    assert compute_pack_metrics(bad, "Series")["spread_level"] == "Imbalanced"


def test_matching_scores_identical_cells_score_100():
    cells = [_stats("A", 90.0, 2.0, 0.05), _stats("B", 90.0, 2.0, 0.05)]
    rows = compute_matching_scores(cells)
    assert len(rows) == 1
    assert rows[0]["Match Score"] == "100"
    assert rows[0]["Recommendation"] == "Excellent match"


def test_matching_scores_very_different_cells_score_low():
    cells = [_stats("A", 95.0, 2.0, 0.03), _stats("B", 50.0, 0.5, 0.20)]
    rows = compute_matching_scores(cells)
    assert float(rows[0]["Match Score"]) < 40
    assert rows[0]["Recommendation"] == "Poor — avoid pairing"


def test_matching_scores_returns_one_row_per_unique_pair():
    cells = [_stats(c, 90.0, 2.0, 0.05) for c in ("A", "B", "C")]
    rows = compute_matching_scores(cells)
    assert len(rows) == 3
    pairs = {(r["Cell A"], r["Cell B"]) for r in rows}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}


# ---------------------------------------------------------------------------
# compute_trajectory_divergence() — cell-to-cell fade divergence over shared
# cycling history, distinct from compute_pack_metrics()'s latest-snapshot-only
# soh_spread/soh_stdev.
# ---------------------------------------------------------------------------

def test_identical_fade_rates_are_not_widening():
    frames = {
        "A": build_features(make_cycles_df(n_cycles=300, fade_per_cycle=0.0006)),
        "B": build_features(make_cycles_df(n_cycles=300, fade_per_cycle=0.0006)),
    }
    result = compute_trajectory_divergence(frames)
    assert result["widening"] is False


def test_diverging_fade_rates_flagged_as_widening_with_fastest_cell_named():
    frames = {
        "slow": build_features(make_cycles_df(n_cycles=300, fade_per_cycle=0.0004)),
        "fast": build_features(make_cycles_df(n_cycles=300, fade_per_cycle=0.0025)),
    }
    result = compute_trajectory_divergence(frames)
    assert result["widening"] is True
    assert result["fastest_diverging_cell"] == "fast"
    assert result["fastest_diverging_fade"] > result["pack_median_fade"]


def test_fewer_than_two_cells_yields_none_widening():
    frames = {"A": build_features(make_cycles_df(n_cycles=100))}
    result = compute_trajectory_divergence(frames)
    assert result["widening"] is None
    assert result["fastest_diverging_cell"] is None


def test_no_overlapping_cycle_range_yields_none_widening():
    frames = {
        "A": build_features(make_cycles_df(n_cycles=50)),
        "B": make_cycles_df(n_cycles=100).assign(cycle_number=lambda d: d["cycle_number"] + 1000).pipe(build_features),
    }
    result = compute_trajectory_divergence(frames)
    assert result["widening"] is None


# ---------------------------------------------------------------------------
# XpYs: group binning
# ---------------------------------------------------------------------------

def _best_achievable_min_group_capacity(capacities, per_group):
    """Brute-force the max-min group capacity over every packing, so the greedy
    result can be compared against the actual optimum rather than asserted to
    be good."""
    n = len(capacities)
    n_groups = math.ceil(n / per_group)
    best = -1.0
    for assignment in itertools.product(range(n_groups), repeat=n):
        blocks = [[i for i in range(n) if assignment[i] == g] for g in range(n_groups)]
        if any(len(b) > per_group for b in blocks) or any(len(b) == 0 for b in blocks):
            continue
        best = max(best, min(sum(capacities[i] for i in b) for b in blocks))
    return best


def test_build_parallel_groups_respects_the_size_cap():
    cells = [_stats(f"C{i}", 90.0, 2.0, 0.05) for i in range(7)]
    groups = build_parallel_groups(cells, 3)
    assert len(groups) == 3, "7 cells at 3p is ceil(7/3) = 3 groups"
    assert all(len(g) <= 3 for g in groups)
    assert sorted(i for g in groups for i in g) == list(range(7)), (
        "every cell is placed exactly once"
    )


def test_build_parallel_groups_balances_capacity_optimally_for_a_small_fleet():
    """The packing objective is the one that matters for a series string:
    maximize the weakest group's capacity. Assert the greedy binner actually
    reaches the optimum on a case small enough to brute-force."""
    capacities = [2.05, 2.00, 1.90, 1.70]
    cells = [_stats(cid, 90.0, cap, 0.05) for cid, cap in zip("DABC", capacities)]
    groups = build_parallel_groups(cells, 2)
    greedy_min = min(sum(capacities[i] for i in g) for g in groups)
    assert greedy_min == pytest.approx(_best_achievable_min_group_capacity(capacities, 2))
    assert greedy_min == pytest.approx(3.75)


def test_build_parallel_groups_is_deterministic_and_tie_breaks_by_cell_id():
    """Equal capacities give the greedy pass nothing but its own tie-breaks, so
    this pins them: cells are ordered by (-capacity, cell_id), and each goes to
    the lightest group, then the least-full one, then the lowest index."""
    cells = [_stats(cid, 90.0, 2.0, 0.05) for cid in ("B", "D", "A", "C")]
    first = build_parallel_groups(cells, 2)
    second = build_parallel_groups(cells, 2)
    assert first == second
    groups_as_ids = [[cells[i]["cell_id"] for i in g] for g in first]
    assert groups_as_ids == [["A", "C"], ["B", "D"]]


def test_build_parallel_groups_of_one_is_the_all_series_configuration():
    cells = [_stats(cid, 90.0, 2.0, 0.05) for cid in "ABC"]
    groups = build_parallel_groups(cells, 1)
    assert groups == [[0], [1], [2]]


def test_build_parallel_groups_wider_than_the_fleet_is_one_parallel_block():
    cells = [_stats(cid, 90.0, 2.0, 0.05) for cid in "ABC"]
    assert build_parallel_groups(cells, 99) == [[0, 1, 2]]


def test_build_parallel_groups_handles_empty_input():
    assert build_parallel_groups([], 4) == []


# ---------------------------------------------------------------------------
# XpYs: the two flat topologies must remain exact special cases
# ---------------------------------------------------------------------------

_XPYS_CELLS = [
    _stats("A", 92.0, 2.00, 0.050),
    _stats("B", 88.0, 1.90, 0.055),
    _stats("C", 80.0, 1.70, 0.070),
    _stats("D", 95.0, 2.05, 0.048),
]


def test_xpys_with_one_cell_per_group_equals_the_series_topology():
    flat = compute_pack_metrics(_XPYS_CELLS, "Series")
    xpys = compute_xpys_metrics(_XPYS_CELLS, build_parallel_groups(_XPYS_CELLS, 1))
    assert xpys["pack_soh"] == pytest.approx(flat["pack_soh"])
    assert xpys["pack_capacity_ah"] == pytest.approx(flat["pack_capacity_ah"])
    assert xpys["pack_resistance_ohm"] == pytest.approx(flat["pack_resistance_ohm"])
    assert xpys["bottleneck_cell_id"] == flat["bottleneck_cell_id"]


def test_xpys_with_one_group_equals_the_parallel_topology():
    flat = compute_pack_metrics(_XPYS_CELLS, "Parallel")
    xpys = compute_xpys_metrics(_XPYS_CELLS, build_parallel_groups(_XPYS_CELLS, len(_XPYS_CELLS)))
    assert xpys["pack_soh"] == pytest.approx(flat["pack_soh"])
    assert xpys["pack_capacity_ah"] == pytest.approx(flat["pack_capacity_ah"])
    assert xpys["pack_resistance_ohm"] == pytest.approx(flat["pack_resistance_ohm"])


def test_xpys_returns_every_key_the_flat_metrics_return():
    """The UI reads one metric interface for all three topologies — a missing
    key on the XpYs path would be a crash in the metric tiles, not a nicety."""
    flat = compute_pack_metrics(_XPYS_CELLS, "Series")
    xpys = compute_xpys_metrics(_XPYS_CELLS, build_parallel_groups(_XPYS_CELLS, 2))
    assert set(flat) <= set(xpys)


# ---------------------------------------------------------------------------
# XpYs: pack metrics
# ---------------------------------------------------------------------------

def test_xpys_capacity_adds_within_a_group_and_is_gated_between_groups():
    cells = [_stats("A", 90.0, 2.0, 0.05), _stats("B", 90.0, 1.0, 0.05),
             _stats("C", 90.0, 1.5, 0.05), _stats("D", 90.0, 1.5, 0.05)]
    groups = [[0, 1], [2, 3]]  # (A,B) and (C,D)
    m = compute_xpys_metrics(cells, groups)
    assert m["groups"][0]["capacity_ah"] == pytest.approx(3.0)
    assert m["pack_capacity_ah"] == pytest.approx(3.0), "the weakest group gates the string"
    assert m["bottleneck_group"] == 1


def test_xpys_group_soh_is_capacity_weighted_and_pack_soh_is_the_weakest_group():
    cells = [_stats("A", 100.0, 3.0, 0.05), _stats("B", 80.0, 1.0, 0.05),
             _stats("C", 60.0, 2.0, 0.05), _stats("D", 60.0, 2.0, 0.05)]
    groups = [[0, 1], [2, 3]]
    m = compute_xpys_metrics(cells, groups)
    assert m["groups"][0]["soh_pct"] == pytest.approx(95.0)   # (100*3 + 80*1) / 4
    assert m["groups"][1]["soh_pct"] == pytest.approx(60.0)
    assert m["pack_soh"] == pytest.approx(60.0)
    assert m["pack_soh_label"] == "Bottleneck-group SOH"


def test_xpys_resistance_is_the_parallel_combination_inside_and_a_sum_between():
    cells = [_stats("A", 90.0, 2.0, 0.10), _stats("B", 90.0, 2.0, 0.10),
             _stats("C", 90.0, 2.0, 0.20)]
    groups = [[0, 1], [2]]
    m = compute_xpys_metrics(cells, groups)
    assert m["groups"][0]["resistance_ohm"] == pytest.approx(0.05)   # 1/(1/.1 + 1/.1)
    assert m["groups"][1]["resistance_ohm"] == pytest.approx(0.20)
    assert m["pack_resistance_ohm"] == pytest.approx(0.25)           # series adds


def test_xpys_bottleneck_group_is_the_lowest_capacity_one_and_names_its_weakest_cell():
    cells = [_stats("A", 95.0, 2.0, 0.05), _stats("B", 70.0, 2.0, 0.05),
             _stats("C", 90.0, 1.0, 0.05), _stats("D", 90.0, 1.0, 0.05)]
    groups = [[0, 1], [2, 3]]
    m = compute_xpys_metrics(cells, groups)
    bottleneck = [g for g in m["groups"] if g["is_bottleneck"]]
    assert len(bottleneck) == 1
    assert bottleneck[0]["group"] == 2
    assert m["bottleneck_cell_id"] == "C" or m["bottleneck_cell_id"] == "D"
    assert m["bottleneck_cell_id"] in ("C", "D")


def test_xpys_flags_a_ragged_small_group_as_the_bottleneck():
    """Cells that do not divide evenly leave a short group; that group's
    capacity is smaller by construction and must be seen to gate the string."""
    cells = [_stats(cid, 90.0, 2.0, 0.05) for cid in "ABCDE"]
    groups = build_parallel_groups(cells, 2)   # 5 cells -> sizes 1/2/2
    assert sorted(len(g) for g in groups) == [1, 2, 2]
    m = compute_xpys_metrics(cells, groups)
    short = next(g for g in m["groups"] if g["n_cells"] == 1)
    assert short["is_bottleneck"]
    assert m["bottleneck_group"] == short["group"]


def test_xpys_binning_verdict_is_balanced_for_an_even_split():
    cells = [_stats(cid, 90.0, 2.0, 0.05) for cid in "ABCD"]
    m = compute_xpys_metrics(cells, build_parallel_groups(cells, 2))
    assert m["binning_level"] == "Balanced"
    assert m["group_capacity_spread_ah"] == pytest.approx(0.0)


def test_xpys_group_member_imbalance_is_reported_even_inside_one_group():
    """A mismatched parallel group is worth flagging on its own terms — the two
    cells share a node, so the mismatch shows up as current sharing, not as
    group-count spread. With one group there is nothing to compare between
    groups, so the binning verdict is correctly Balanced while the intra-group
    figure is not."""
    cells = [_stats("A", 90.0, 4.0, 0.05), _stats("B", 90.0, 1.0, 0.05)]
    m = compute_xpys_metrics(cells, [[0, 1]])
    assert m["groups"][0]["capacity_imbalance_pct"] == pytest.approx(120.0)
    assert m["binning_level"] == "Balanced"


def test_xpys_binning_verdict_flags_a_ragged_build():
    """3 cells at 2p leaves a group of one; its capacity can never match a
    two-cell group's, so the build is Imbalanced by construction."""
    cells = [_stats(cid, 90.0, 2.0, 0.05) for cid in "ABC"]
    groups = build_parallel_groups(cells, 2)
    assert sorted(len(g) for g in groups) == [1, 2]
    m = compute_xpys_metrics(cells, groups)
    assert m["binning_level"] == "Imbalanced"
    assert m["group_capacity_spread_pct"] == pytest.approx(2.0 / 3.0 * 100.0)


def test_xpys_pack_rul_is_the_weakest_calibrated_cell_and_counts_the_rest():
    cells = [
        _stats("A", 90.0, 2.0, 0.05, rul=300, rul_ok=True),
        _stats("B", 90.0, 2.0, 0.05, rul=50, rul_ok=False),
        _stats("C", 90.0, 2.0, 0.05, rul=120, rul_ok=True),
    ]
    m = compute_xpys_metrics(cells, [[0, 1], [2]])
    assert m["pack_rul"] == 120
    assert m["n_uncalibrated"] == 1


def test_xpys_without_usable_resistance_still_reports_capacity_and_soh():
    cells = [_stats("A", 90.0, 2.0, 0.0), _stats("B", 80.0, 2.0, float("nan"))]
    m = compute_xpys_metrics(cells, [[0, 1]])
    assert m["current_sharing_available"] is False
    assert m["most_loaded_cell_id"] is None
    assert m["pack_resistance_ohm"] != m["pack_resistance_ohm"]   # NaN, not invented
    assert m["pack_capacity_ah"] == pytest.approx(4.0)
    assert m["pack_soh"] == pytest.approx(85.0)


def test_xpys_handles_empty_input_without_raising():
    m = compute_xpys_metrics([], [])
    assert m["groups"] == []
    assert m["pack_rul"] is None
    assert m["soh_stdev"] != m["soh_stdev"]   # NaN, nothing to summarise


# ---------------------------------------------------------------------------
# Current sharing inside a parallel group
# ---------------------------------------------------------------------------

def test_current_sharing_divides_inversely_with_resistance():
    cells = [_stats("A", 90.0, 2.0, 0.05), _stats("B", 90.0, 2.0, 0.10)]
    sharing = compute_group_current_sharing(cells)
    # 1/0.05 : 1/0.10 = 2:1
    assert sharing["shares_pct"]["A"] == pytest.approx(66.666, abs=1e-3)
    assert sharing["shares_pct"]["B"] == pytest.approx(33.333, abs=1e-3)
    assert sharing["max_share_cell_id"] == "A"
    assert sharing["overload_ratio"] == pytest.approx(4.0 / 3.0)
    assert sum(sharing["shares_pct"].values()) == pytest.approx(100.0)


def test_current_sharing_is_even_for_identical_cells():
    cells = [_stats(cid, 90.0, 2.0, 0.05) for cid in "ABC"]
    sharing = compute_group_current_sharing(cells)
    assert all(abs(v - 100.0 / 3.0) < 1e-9 for v in sharing["shares_pct"].values())
    assert sharing["overload_ratio"] == pytest.approx(1.0)


def test_current_sharing_reports_unavailable_rather_than_inventing_a_share():
    cells = [_stats("A", 90.0, 2.0, float("nan")), _stats("B", 90.0, 2.0, 0.05)]
    sharing = compute_group_current_sharing(cells)
    assert sharing["shares_pct"] == {}
    assert sharing["unavailable_reason"] is not None
    assert "A" in sharing["unavailable_reason"]
    assert sharing["max_share_cell_id"] is None


def test_current_sharing_treats_a_zero_resistance_as_missing_not_as_a_short():
    cells = [_stats("A", 90.0, 2.0, 0.0), _stats("B", 90.0, 2.0, 0.05)]
    sharing = compute_group_current_sharing(cells)
    assert sharing["shares_pct"] == {}, "0 Ω is a sentinel, not a cell that takes all the current"
    assert sharing["unavailable_reason"] is not None


def test_current_sharing_of_a_single_cell_group_is_all_of_it():
    sharing = compute_group_current_sharing([_stats("A", 90.0, 2.0, 0.05)])
    assert sharing["shares_pct"] == {"A": 100.0}
    assert sharing["overload_ratio"] == pytest.approx(1.0)


def test_current_sharing_of_an_empty_group_is_unavailable():
    sharing = compute_group_current_sharing([])
    assert sharing["shares_pct"] == {}
    assert sharing["unavailable_reason"] is not None


def test_the_most_loaded_cell_is_the_lowest_resistance_cell_in_its_group():
    """The physical claim the UI makes — sharing is inverse to resistance, so the
    hardest-worked cell is the one with the least resistance."""
    cells = [_stats("A", 90.0, 2.0, 0.070), _stats("B", 90.0, 2.0, 0.048),
             _stats("C", 90.0, 2.0, 0.060)]
    m = compute_xpys_metrics(cells, [[0, 1, 2]])
    assert m["most_loaded_cell_id"] == "B"
    assert m["max_overload_ratio"] > 1.0
    shares = m["groups"][0]["shares_pct"]
    assert shares["B"] == max(shares.values())


def test_xpys_reports_the_most_loaded_cell_across_groups():
    cells = [
        _stats("A", 90.0, 2.0, 0.050), _stats("B", 90.0, 2.0, 0.052),   # close pair
        _stats("C", 90.0, 2.0, 0.040), _stats("D", 90.0, 2.0, 0.090),   # badly mismatched pair
    ]
    m = compute_xpys_metrics(cells, [[0, 1], [2, 3]])
    assert m["most_loaded_cell_id"] == "C"
    assert m["most_loaded_group"] == 2
    assert m["max_overload_ratio"] > m["groups"][0]["overload_ratio"]
