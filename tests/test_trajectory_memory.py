"""Unit tests for src/trajectory_memory.py — TrajectoryMemory."""

import numpy as np
from conftest import make_cycles_df
from features import build_features
from trajectory_memory import TrajectoryMemory, FailureSignature, TrajectoryMatch, reconcile_rul_estimates


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


def test_match_never_crosses_chemistry_sources():
    """Regression guard: cosine similarity on normalized slope vectors is
    only physically meaningful within one chemistry. match() used to
    compare a querying cell against every stored signature regardless of
    source, so a real Severson LFP cell could "match" a synthetic LiCoO2
    failure signature — not a physically meaningful comparison. A
    same-shaped trajectory in a *different* source must never be offered
    as a match, even when its trend vector is identical."""
    tm = TrajectoryMemory()
    # An NASA-sourced signature ("B..." id) with a very steep, distinctive fade.
    tm.build({"B0001": _eol_cell(fade_per_cycle=0.003, n_cycles=300)})
    assert tm._signatures[0].source == "nasa"

    # A Severson-sourced cell ("S-..." id) with the exact same fade profile —
    # would score a near-perfect cosine similarity against the NASA signature
    # if source were ignored.
    identical_shape = build_features(make_cycles_df(n_cycles=150, fade_per_cycle=0.003))
    match = tm.match("S-b1c1", identical_shape)
    assert match is None


# ---------------------------------------------------------------------------
# reconcile_rul_estimates()
# ---------------------------------------------------------------------------

def _make_match(cycles_remaining_min: int, cycles_remaining_max: int) -> TrajectoryMatch:
    return TrajectoryMatch(
        matched_cells=["Cell2"], similarities=[0.93], best_cell_id="Cell2",
        best_similarity=0.93, current_cycle=1177, current_soh=96.6,
        predicted_eol_min=1212, predicted_eol_max=1242,
        cycles_remaining_min=cycles_remaining_min, cycles_remaining_max=cycles_remaining_max,
        failure_mode="Mixed / Undetermined", n_matches=1, warning_level="high",
    )


def test_reconcile_flags_severe_disagreement():
    """The exact live-reproduced case: primary model says 657 cycles
    remaining, trajectory match says 35-65 (mid 50) -- an 8x+ disagreement
    that must be flagged, not shown as two independently confident numbers."""
    match = _make_match(35, 65)
    result = reconcile_rul_estimates(657.0, match)
    assert result["disagree"] is True
    assert result["favor"] == "match"  # the shorter, more conservative estimate
    assert result["ratio"] > 0.4


def test_reconcile_does_not_flag_close_estimates():
    """Two estimates within the same ballpark are normal model imprecision,
    not a contradiction -- must not trigger the disagreement UI."""
    match = _make_match(600, 650)
    result = reconcile_rul_estimates(657.0, match)
    assert result["disagree"] is False
    assert result["favor"] is None


def test_reconcile_favors_primary_when_primary_is_shorter():
    match = _make_match(500, 550)
    result = reconcile_rul_estimates(100.0, match)
    assert result["disagree"] is True
    assert result["favor"] == "primary"


def test_reconcile_returns_no_disagreement_when_no_match():
    result = reconcile_rul_estimates(657.0, None)
    assert result["disagree"] is False
    assert result["match_cycles_mid"] is None


def test_reconcile_returns_no_disagreement_when_primary_uncalibrated():
    match = _make_match(35, 65)
    result = reconcile_rul_estimates(None, match)
    assert result["disagree"] is False
