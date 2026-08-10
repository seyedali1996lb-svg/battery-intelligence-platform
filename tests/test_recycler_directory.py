"""Unit tests for src/recycler_directory.py's recommend_recyclers()."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from recycler_directory import recommend_recyclers, RECYCLER_DIRECTORY


def test_directory_excludes_bankrupt_li_cycle():
    """Li-Cycle filed for bankruptcy in 2025 -- must not appear anywhere
    in the directory as a viable routing target."""
    names = [r["name"] for r in RECYCLER_DIRECTORY]
    assert not any("li-cycle" in n.lower() for n in names)


def test_lfp_only_returns_lfp_compatible_recyclers():
    results = recommend_recyclers("LFP")
    assert len(results) > 0
    for r in results:
        assert "LFP" in r["chemistries"]


def test_chemistry_filter_is_hard_not_soft():
    """A recycler that doesn't process the given chemistry must never
    appear, regardless of region preference."""
    results = recommend_recyclers("LFP", user_region="Europe")
    for r in results:
        assert "LFP" in r["chemistries"]


def test_same_region_ranks_first():
    results = recommend_recyclers("LiCoO2", user_region="Europe", top_n=len(RECYCLER_DIRECTORY))
    same_region_flags = [r["same_region"] for r in results]
    # Once a False appears, no True should follow (same-region entries sorted first)
    seen_false = False
    for flag in same_region_flags:
        if not flag:
            seen_false = True
        elif seen_false:
            assert False, "same-region entry appeared after a non-same-region one"


def test_no_region_preserves_directory_order_among_compatible():
    results = recommend_recyclers("NCA", user_region=None)
    compatible_in_order = [r["name"] for r in RECYCLER_DIRECTORY if "NCA" in r["chemistries"]]
    assert [r["name"] for r in results] == compatible_in_order[:len(results)]


def test_top_n_respected():
    results = recommend_recyclers("LiCoO2", top_n=2)
    assert len(results) <= 2


def test_unknown_chemistry_falls_back_gracefully_no_crash():
    results = recommend_recyclers("SomeUnknownChemistry")
    assert isinstance(results, list)
    assert len(results) > 0  # falls back to LiCoO2-compatible entries
