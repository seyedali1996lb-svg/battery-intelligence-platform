"""Unit tests for src/cell_store.py — per-cell Parquet persistence, the
bounded in-process LRU, build_summary()'s reduction, and LazyCellFrameMap's
dict-like behavior."""

import numpy as np
import pandas as pd
import pytest

import cell_store as cs


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CELL_STORE_DIR", tmp_path / "cell_store")
    cs.clear_lru()
    yield cs
    cs.clear_lru()


def _fade_rate(soh: np.ndarray, window: int) -> np.ndarray:
    """SOH-drop-per-cycle over a trailing window, zero-filled for the first
    `window` cycles (mirrors batlab's own rolling-window feature columns) --
    always returns an array the same length as soh, even when n_cycles is
    smaller than the window itself."""
    n = len(soh)
    w = min(window, n)
    rate = np.zeros(n)
    if n > w:
        rate[w:] = -(soh[w:] - soh[:-w]) / w
    return rate


def _make_df(n_cycles: int = 120, soh_start: float = 100.0, soh_end: float = 70.0,
             capacity_ah: float = 2.0, resistance_ohm: float = 0.05,
             eol_cycle: "int | None" = None) -> pd.DataFrame:
    """A minimal but realistic per-cycle DataFrame with the columns
    build_summary()/fleet.py/grading.py's formulas actually read."""
    cycle = np.arange(1, n_cycles + 1)
    soh = np.linspace(soh_start, soh_end, n_cycles)
    capacity = capacity_ah * (soh / 100.0)
    resistance = resistance_ohm * np.linspace(1.0, 1.3, n_cycles)
    fade_30 = _fade_rate(soh, 30)
    fade_50 = _fade_rate(soh, 50)
    is_eol = cycle >= (eol_cycle or n_cycles + 1)
    return pd.DataFrame({
        "cycle_number": cycle,
        "soh_pct": soh,
        "capacity_ah": capacity,
        "resistance_ohm": resistance,
        "fade_rate_30cy": fade_30,
        "fade_rate_50cy": fade_50,
        "is_eol": is_eol,
        "rul_pred": np.maximum(0, n_cycles - cycle),
        "rul_q10": np.maximum(0, n_cycles - cycle - 20),
        "rul_q90": np.maximum(0, n_cycles - cycle + 20),
    })


def test_get_cell_df_returns_none_when_never_saved(isolated_store):
    assert isolated_store.get_cell_df("no-such-cell") is None


def test_save_then_get_round_trip(isolated_store):
    df = _make_df()
    isolated_store.save_cell_df("CellA", df)
    loaded = isolated_store.get_cell_df("CellA")
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True))


def test_get_cell_df_survives_lru_eviction_by_reloading_from_disk(isolated_store, monkeypatch):
    monkeypatch.setattr(isolated_store, "MAX_CACHE_CELLS", 5)
    dfs = {f"Cell{i}": _make_df(n_cycles=30 + i) for i in range(8)}
    for cid, df in dfs.items():
        isolated_store.save_cell_df(cid, df)

    # The LRU is bounded even after writing more cells than the cap.
    assert len(isolated_store._lru) <= 5

    # An evicted cell (Cell0, the earliest written) is still correctly
    # reloadable from disk -- eviction drops RAM residency, not data.
    reloaded = isolated_store.get_cell_df("Cell0")
    pd.testing.assert_frame_equal(reloaded.reset_index(drop=True), dfs["Cell0"].reset_index(drop=True))


def test_lru_never_exceeds_max_cache_cells_across_many_saves(isolated_store, monkeypatch):
    monkeypatch.setattr(isolated_store, "MAX_CACHE_CELLS", 10)
    for i in range(60):
        isolated_store.save_cell_df(f"Cell{i}", _make_df(n_cycles=25))
        assert len(isolated_store._lru) <= 10


def test_build_summary_basic_fields(isolated_store):
    df = _make_df(n_cycles=120, soh_start=100.0, soh_end=70.0, eol_cycle=110)
    summary = isolated_store.build_summary("CellA", df)
    assert summary["soh_pct"] == pytest.approx(df["soh_pct"].iloc[-1])
    assert summary["cycle_number"] == 120
    assert summary["capacity_ah"] == pytest.approx(df["capacity_ah"].iloc[-1])
    assert summary["resistance_ohm"] == pytest.approx(df["resistance_ohm"].iloc[-1])
    assert summary["resistance_ohm_initial"] == pytest.approx(df["resistance_ohm"].iloc[0])
    assert summary["eol_at_cycle"] == 110
    assert summary["rul_pred"] == pytest.approx(df["rul_pred"].iloc[-1])
    assert summary["rul_q10"] == pytest.approx(df["rul_q10"].iloc[-1])
    assert summary["rul_q90"] == pytest.approx(df["rul_q90"].iloc[-1])


def test_build_summary_eol_at_cycle_is_none_when_never_reached(isolated_store):
    df = _make_df(n_cycles=80, eol_cycle=None)
    summary = isolated_store.build_summary("CellA", df)
    assert summary["eol_at_cycle"] is None


def test_build_summary_knee_matches_direct_detect_knee_call(isolated_store):
    """build_summary() must reuse batlab's detect_knee(), not reimplement
    knee detection -- assert its output matches calling detect_knee()
    directly on the identical series."""
    from batlab.features.knee_detection import detect_knee

    df = _make_df(n_cycles=200, soh_start=100.0, soh_end=60.0)
    expected = detect_knee(df["soh_pct"], df["cycle_number"])
    summary = isolated_store.build_summary("CellA", df)
    assert summary["knee_detected"] == expected["detected"]
    assert summary["knee_cycle"] == expected["cycle"]
    assert summary["knee_soh"] == expected["soh_at_knee"]
    assert summary["knee_confidence"] == expected["confidence"]
    assert summary["knee_phase"] == expected["phase"]


def test_build_summary_fade_trend_accelerating(isolated_store):
    n = 120
    cycle = np.arange(1, n + 1)
    soh = np.concatenate([
        np.linspace(100.0, 95.0, 60),   # slow fade for the first 60 cycles
        np.linspace(95.0, 60.0, 60),    # then fast fade
    ])
    df = pd.DataFrame({
        "cycle_number": cycle,
        "soh_pct": soh,
        "capacity_ah": 2.0 * (soh / 100.0),
        "resistance_ohm": 0.05 * np.linspace(1.0, 1.3, n),
        "fade_rate_30cy": np.concatenate([np.full(90, 0.02), np.full(30, 0.5)]),
        "is_eol": np.zeros(n, dtype=bool),
        "rul_pred": np.maximum(0, n - cycle),
    })
    summary = isolated_store.build_summary("CellA", df)
    assert summary["fade_trend"] == "Accelerating"


def test_build_summary_grade_matches_hand_computed_score(isolated_store):
    """The A/B/C grade formula, ported from fleet.py/grading.py's
    duplicated logic -- verify against a hand-computed score for a
    deliberately near-flat (high-grade) early-life curve."""
    n = 150
    cycle = np.arange(1, n + 1)
    early_n = 100
    cap0 = 2.0
    capacity = np.concatenate([
        np.full(early_n, cap0),  # zero fade / zero variance in the first 100 cycles
        np.linspace(cap0, cap0 * 0.7, n - early_n),
    ])
    resistance = np.full(n, 0.05)  # flat resistance -> zero slope
    df = pd.DataFrame({
        "cycle_number": cycle,
        "soh_pct": capacity / cap0 * 100,
        "capacity_ah": capacity,
        "resistance_ohm": resistance,
        "fade_rate_30cy": np.zeros(n),
        "is_eol": np.zeros(n, dtype=bool),
        "rul_pred": np.maximum(0, n - cycle),
    })
    summary = isolated_store.build_summary("CellA", df)
    # Zero fade, zero variance, zero resistance slope -> the formula's
    # perfect score, therefore Grade A.
    assert summary["grade"] == "A"
    assert summary["grade_score"] == pytest.approx(100.0)
    assert summary["grade_fade"] == pytest.approx(0.0)
    assert summary["grade_var"] == pytest.approx(0.0)
    assert summary["grade_slope"] == pytest.approx(0.0)


def test_build_summary_grade_is_dash_with_insufficient_early_life_data(isolated_store):
    df = _make_df(n_cycles=10)  # fewer than the 20-cycle early-life minimum
    summary = isolated_store.build_summary("CellA", df)
    assert summary["grade"] == "—"
    assert summary["grade_score"] is None


# ---------------------------------------------------------------------------
# LazyCellFrameMap
# ---------------------------------------------------------------------------

def test_lazy_map_getitem_matches_real_dict(isolated_store):
    dfs = {"CellA": _make_df(n_cycles=40), "CellB": _make_df(n_cycles=60)}
    for cid, df in dfs.items():
        isolated_store.save_cell_df(cid, df)

    lazy = cs.LazyCellFrameMap(["CellA", "CellB"])
    pd.testing.assert_frame_equal(lazy["CellA"].reset_index(drop=True), dfs["CellA"].reset_index(drop=True))
    pd.testing.assert_frame_equal(lazy["CellB"].reset_index(drop=True), dfs["CellB"].reset_index(drop=True))


def test_lazy_map_missing_key_raises_key_error(isolated_store):
    lazy = cs.LazyCellFrameMap(["CellA"])
    with pytest.raises(KeyError):
        lazy["CellA"]  # never saved


def test_lazy_map_len_and_contains(isolated_store):
    isolated_store.save_cell_df("CellA", _make_df())
    isolated_store.save_cell_df("CellB", _make_df())
    lazy = cs.LazyCellFrameMap(["CellA", "CellB"])
    assert len(lazy) == 2
    assert "CellA" in lazy
    assert "CellC" not in lazy


def test_lazy_map_excludes_cells_that_exist_in_the_store_but_not_in_this_map(isolated_store):
    """Regression test: get_cell_df() (the default loader) will happily load
    ANY cell that exists anywhere in the global Parquet store -- membership
    in a particular LazyCellFrameMap instance must be checked against that
    instance's own cell_ids, not just "does the loader return non-None."
    Found via app/_pages/grading.py's CellSummary migration: a NASA-only
    LazyCellFrameMap incorrectly reported a Severson cell (saved earlier in
    the same store) as a member, because `in` fell through to the loader."""
    isolated_store.save_cell_df("NasaCell1", _make_df())
    isolated_store.save_cell_df("SeversonCell1", _make_df())

    nasa_only = cs.LazyCellFrameMap(["NasaCell1"])
    assert "NasaCell1" in nasa_only
    assert "SeversonCell1" not in nasa_only, (
        "A cell that exists in the global store under a different map's "
        "scope must not be reported as a member of this one"
    )
    with pytest.raises(KeyError):
        nasa_only["SeversonCell1"]


def test_lazy_map_items_and_values_behave_like_a_real_dict(isolated_store):
    dfs = {"CellA": _make_df(n_cycles=40), "CellB": _make_df(n_cycles=60)}
    for cid, df in dfs.items():
        isolated_store.save_cell_df(cid, df)

    lazy = cs.LazyCellFrameMap(list(dfs.keys()))
    items = dict(lazy.items())
    assert set(items.keys()) == set(dfs.keys())
    for cid in dfs:
        pd.testing.assert_frame_equal(items[cid].reset_index(drop=True), dfs[cid].reset_index(drop=True))

    values = list(lazy.values())
    assert len(values) == 2


def test_lazy_map_star_unpacking_materializes_a_real_dict(isolated_store):
    isolated_store.save_cell_df("CellA", _make_df())
    lazy = cs.LazyCellFrameMap(["CellA"])
    merged = {**lazy, "CellB": "placeholder"}
    assert set(merged.keys()) == {"CellA", "CellB"}
