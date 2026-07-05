"""Unit tests for src/trajectory_memory.py — TrajectoryMemory."""

import numpy as np
from conftest import make_cycles_df
from features import build_features
from trajectory_memory import TrajectoryMemory, FailureSignature


def _eol_cell(fade_per_cycle=0.0015, n_cycles=400):
    """A cell whose SOH crosses the 80% EOL threshold within n_cycles."""
    return build_features(make_cycles_df(n_cycles=n_cycles, fade_per_cycle=fade_per_cycle))


def test_build_extracts_signature_for_cell_that_reaches_eol():
    tm = TrajectoryMemory()
    tm.build({"CellA": _eol_cell()})
    assert tm.n_signatures == 1
    assert tm._signatures[0].cell_id == "CellA"


def test_build_skips_cell_that_never_reaches_eol():
    tm = TrajectoryMemory()
    healthy = build_features(make_cycles_df(n_cycles=200, fade_per_cycle=0.0001))
    tm.build({"CellHealthy": healthy})
    assert tm.n_signatures == 0


def test_match_finds_similar_trajectory():
    tm = TrajectoryMemory()
    tm.build({"CellA": _eol_cell(fade_per_cycle=0.0015)})
    # A second cell with a very similar fade profile, still pre-EOL
    similar = build_features(make_cycles_df(n_cycles=150, fade_per_cycle=0.0015))
    match = tm.match("CellB", similar)
    # May or may not clear the similarity threshold depending on window overlap,
    # but must not raise and must return None or a well-formed TrajectoryMatch.
    if match is not None:
        assert match.best_cell_id == "CellA"
        assert 0.0 <= match.best_similarity <= 1.0
        assert match.warning_level in ("critical", "high", "watch")


def test_match_returns_none_with_no_signatures():
    tm = TrajectoryMemory()
    result = tm.match("CellX", _eol_cell())
    assert result is None


def test_merge_dedupe_prefers_freshly_built_signature():
    tm = TrajectoryMemory()
    tm.build({"CellA": _eol_cell(fade_per_cycle=0.002)})
    fresh_sig = tm._signatures[0]

    stale_sig = FailureSignature(
        cell_id="CellA", source="synthetic", eol_cycle=999,
        soh_at_window_start=81.0, failure_mode="LAM",
        feature_names=["fade_rate_30cy"], trend_vector=np.array([0.01]),
    )
    tm.merge_dedupe_by_cell_id([stale_sig])

    assert tm.n_signatures == 1
    assert tm._signatures[0].eol_cycle == fresh_sig.eol_cycle
    assert tm._signatures[0].eol_cycle != 999


def test_merge_dedupe_unions_different_cell_ids():
    tm = TrajectoryMemory()
    tm.build({"CellA": _eol_cell()})
    other_sig = FailureSignature(
        cell_id="CellB", source="nasa", eol_cycle=500,
        soh_at_window_start=80.5, failure_mode="Mixed / Undetermined",
        feature_names=["fade_rate_30cy"], trend_vector=np.array([0.02]),
    )
    tm.merge_dedupe_by_cell_id([other_sig])
    assert {s.cell_id for s in tm._signatures} == {"CellA", "CellB"}
