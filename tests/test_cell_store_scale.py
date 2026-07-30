"""Proof that cell_store.py's architecture actually bounds RAM by cache
size, not fleet size -- the concrete claim behind this whole fix (see
cell_store.py's module docstring). Today's real fleet is ~24 cells;
this generates a few hundred (meaningfully more than that, still fast
enough for CI) to prove the mechanism at a scale beyond what the app
naturally exercises today.

Deliberately NOT measuring real process RSS via psutil -- that measures
the OS's memory allocator behavior, not this code, and would be slow/
flaky for no added confidence. The LRU dict's own length is the right
level of proof: it's the actual data structure enforcing the bound.
"""

import numpy as np
import pandas as pd
import pytest

import cell_store as cs
import db as db_module


N_SYNTHETIC_CELLS = 300
MAX_CACHE_CELLS_FOR_TEST = 16


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CELL_STORE_DIR", tmp_path / "cell_store")
    monkeypatch.setattr(cs, "MAX_CACHE_CELLS", MAX_CACHE_CELLS_FOR_TEST)
    cs.clear_lru()
    yield cs
    cs.clear_lru()


@pytest.fixture
def db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


def _tiny_synthetic_df(cell_id: str, n_cycles: int = 40) -> pd.DataFrame:
    """A minimal per-cycle DataFrame with the columns build_summary() reads
    -- deliberately small/cheap (not routed through the real GBRT training
    pipeline, which would test training performance, not this fix)."""
    rng = np.random.default_rng(abs(hash(cell_id)) % (2**32))
    cycle = np.arange(1, n_cycles + 1)
    soh_end = float(rng.uniform(60.0, 95.0))
    soh = np.linspace(100.0, soh_end, n_cycles)
    capacity = 2.0 * (soh / 100.0)
    resistance = 0.05 * np.linspace(1.0, 1.3, n_cycles)
    return pd.DataFrame({
        "cycle_number": cycle,
        "soh_pct": soh,
        "capacity_ah": capacity,
        "resistance_ohm": resistance,
        "fade_rate_30cy": np.zeros(n_cycles),
        "fade_rate_50cy": np.zeros(n_cycles),
        "is_eol": soh < 80.0,
        "rul_pred": np.maximum(0, n_cycles - cycle),
        "rul_q10": np.maximum(0, n_cycles - cycle - 5),
        "rul_q90": np.maximum(0, n_cycles - cycle + 5),
    })


def test_lru_stays_bounded_while_saving_many_more_cells_than_the_cache_size(isolated_store, db):
    """Saving N_SYNTHETIC_CELLS (>> MAX_CACHE_CELLS_FOR_TEST) cells' full
    DataFrames must never let the in-process LRU grow past its configured
    bound -- this is the actual mechanism that keeps RAM O(cache size),
    not O(fleet size), regardless of how many cells exist in total."""
    from experiment_registry import PLATFORM_ORG_ID

    for i in range(N_SYNTHETIC_CELLS):
        cell_id = f"SynthScale{i:04d}"
        df = _tiny_synthetic_df(cell_id)
        isolated_store.save_cell_df(cell_id, df)
        db.upsert_cell_summary(PLATFORM_ORG_ID, cell_id, isolated_store.build_summary(cell_id, df))
        assert len(isolated_store._lru) <= MAX_CACHE_CELLS_FOR_TEST, (
            f"LRU grew to {len(isolated_store._lru)} after saving cell {i} "
            f"-- exceeds the configured bound of {MAX_CACHE_CELLS_FOR_TEST}"
        )

    # The bound holds at the end too, long after every cell has been
    # written -- not just transiently during the save loop itself.
    assert len(isolated_store._lru) <= MAX_CACHE_CELLS_FOR_TEST


def test_fleet_wide_summary_render_touches_zero_parquet_reads(isolated_store, db, monkeypatch):
    """The concrete proof that fleet-wide rendering (Fleet's ranking table,
    Grading, etc.) no longer touches full per-cycle data at all: populate
    the store, clear the in-process LRU (simulating a cold cache -- no
    cell happens to already be resident), then confirm a CellSummary-based
    "fleet render" returns every cell correctly while get_cell_df() is
    never called."""
    from experiment_registry import PLATFORM_ORG_ID

    cell_ids = [f"SynthScale{i:04d}" for i in range(N_SYNTHETIC_CELLS)]
    for cell_id in cell_ids:
        df = _tiny_synthetic_df(cell_id)
        isolated_store.save_cell_df(cell_id, df)
        db.upsert_cell_summary(PLATFORM_ORG_ID, cell_id, isolated_store.build_summary(cell_id, df))

    isolated_store.clear_lru()  # cold cache -- nothing resident in-process

    call_count = {"n": 0}
    real_get_cell_df = isolated_store.get_cell_df

    def _spy_get_cell_df(cell_id):
        call_count["n"] += 1
        return real_get_cell_df(cell_id)

    monkeypatch.setattr(isolated_store, "get_cell_df", _spy_get_cell_df)

    rows = db.get_cell_summaries(PLATFORM_ORG_ID, include_platform=False)

    assert len(rows) == N_SYNTHETIC_CELLS
    assert {r["cell_id"] for r in rows} == set(cell_ids)
    for r in rows:
        assert r["soh_pct"] is not None
        assert r["grade"] in ("A", "B", "C", "—")

    assert call_count["n"] == 0, (
        f"Fleet-wide summary render called get_cell_df() {call_count['n']} times -- "
        "it should read exclusively from precomputed CellSummary rows, never "
        "touching a cell's full per-cycle DataFrame"
    )
