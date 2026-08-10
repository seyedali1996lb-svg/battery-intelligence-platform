"""Unit tests for src/pack_builder.py's pure calculation logic."""

from conftest import make_cycles_df
from batlab.features.engineering import build_features
from pack_builder import compute_pack_metrics, compute_matching_scores, compute_trajectory_divergence


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
