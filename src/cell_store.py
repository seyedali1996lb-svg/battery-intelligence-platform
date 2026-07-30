"""
Per-cell lazy data store — the RAM-bound half of the scalability fix
documented in docs/history.md's Production Readiness Roadmap ("Per-cell
lazy load from Parquet/SQLite; summary metrics pre-computed and cached").

Before this module, app/main.py's load_everything() built a dict holding
every reference-fleet cell's full per-cycle DataFrame and returned it from
a parameterless @st.cache_resource function — the whole dict stayed
RAM-resident for the process's entire lifetime regardless of how many
cells were actually being looked at. Today's real fleet (~24 cells) never
felt this; the documented ceiling ("low thousands of cells") is what this
fixes.

Two pieces:

- save_cell_df()/get_cell_df(): each cell's full per-cycle DataFrame is
  persisted to its own Parquet file, lazily loaded on demand, and kept in
  a bounded in-process LRU (MAX_CACHE_CELLS) rather than a dict growing
  without bound. RAM ceiling for this cache is roughly
  MAX_CACHE_CELLS x (bytes per cell's DataFrame) -- deliberately picked
  well above today's real fleet size (24) and comfortably below the
  documented "low thousands" ceiling, trading a bounded RAM footprint for
  a Parquet re-read on a cache miss.

- build_summary(): the one real "full history -> scalar summary" reduction,
  computed once per cell (when its data is (re)trained/ingested), not
  recomputed on every fleet-wide page render. Fleet-wide consumers (Fleet,
  Grading, Compliance's regulatory alerts, etc.) read these precomputed
  summaries (via src/db.py's CellSummary table) instead of touching full
  per-cycle DataFrames at all -- this is what actually removes the
  standing per-render cost of iterating every cell's full history.

LazyCellFrameMap makes this transparent to every single-cell/bounded-subset
consumer (Health, Copilot, Decide & Ask, trajectory matching, etc.): it's a
collections.abc.Mapping wrapping get_cell_df(), so `featured_dfs[cell_id]`,
`.items()`, `.values()`, and `**`-unpacking all behave exactly like a real
dict -- those call sites need zero code changes.
"""

import collections
import pathlib

import pandas as pd

from batlab.features.knee_detection import detect_knee

CELL_STORE_DIR = pathlib.Path(__file__).parent.parent / "data" / "cell_store"

# Comfortably above today's real fleet (~24 cells across NASA/Severson/
# synthetic), well below the documented "low thousands" ceiling. RAM
# ceiling for the in-process cache is roughly this many cells' worth of
# full per-cycle DataFrames (a few hundred KB each at today's row/column
# counts) -- trades that bound against a Parquet re-read on a cache miss.
MAX_CACHE_CELLS = 200

# cell_id -> DataFrame, ordered by most-recently-used (OrderedDict.move_to_end).
_lru: "collections.OrderedDict[str, pd.DataFrame]" = collections.OrderedDict()


def save_cell_df(cell_id: str, df: pd.DataFrame) -> None:
    """Persist a cell's full per-cycle DataFrame to its own Parquet file."""
    CELL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CELL_STORE_DIR / f"{cell_id}.parquet", engine="pyarrow")
    # Keep the LRU consistent with what was just written, same as a fresh read would.
    _lru[cell_id] = df
    _lru.move_to_end(cell_id)
    _evict_if_needed()


def get_cell_df(cell_id: str) -> "pd.DataFrame | None":
    """Lazily load a cell's full per-cycle DataFrame, LRU-cached in-process.

    Returns None if the cell was never saved (mirrors bundle_cache's
    None-on-miss convention).
    """
    if cell_id in _lru:
        _lru.move_to_end(cell_id)
        return _lru[cell_id]
    path = CELL_STORE_DIR / f"{cell_id}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    _lru[cell_id] = df
    _evict_if_needed()
    return df


def _evict_if_needed() -> None:
    while len(_lru) > MAX_CACHE_CELLS:
        _lru.popitem(last=False)  # drop least-recently-used


def clear_lru() -> None:
    """Test/debug helper -- drop the in-process cache without touching disk."""
    _lru.clear()


def build_summary(cell_id: str, df: pd.DataFrame) -> dict:
    """Reduce a cell's full per-cycle DataFrame to the scalar/summary fields
    fleet-wide views need, computed once so they never have to touch the
    full DataFrame themselves. Ported from the byte-for-byte-duplicated
    grade formula in app/_pages/fleet.py and app/_pages/grading.py, and the
    knee/trend/EOL logic in app/_pages/fleet.py -- centralizing both here
    fixes that duplication as a byproduct.
    """
    latest = df.iloc[-1]
    first = df.iloc[0]

    eol_rows = df[df["is_eol"]] if "is_eol" in df.columns else df.iloc[0:0]
    eol_at_cycle = int(eol_rows["cycle_number"].iloc[0]) if len(eol_rows) else None

    fade_trend = "Stable"
    if "fade_rate_30cy" in df.columns and len(df) >= 31:
        fade_now = df["fade_rate_30cy"].iloc[-1]
        fade_prev = df["fade_rate_30cy"].iloc[-31]
        delta_pct = (fade_now - fade_prev) / (abs(fade_prev) + 1e-9) * 100
        if delta_pct > 20:
            fade_trend = "Accelerating"
        elif delta_pct < -20:
            fade_trend = "Decelerating"

    knee = detect_knee(df["soh_pct"], df["cycle_number"])

    grade, grade_score, grade_fade, grade_var, grade_slope = _early_life_grade(df)

    return {
        "soh_pct": float(latest["soh_pct"]),
        "cycle_number": int(latest["cycle_number"]),
        "capacity_ah": float(latest["capacity_ah"]) if "capacity_ah" in df.columns else None,
        "resistance_ohm": float(latest["resistance_ohm"]) if "resistance_ohm" in df.columns else None,
        "resistance_ohm_initial": float(first["resistance_ohm"]) if "resistance_ohm" in df.columns else None,
        "fade_rate_30cy": float(latest["fade_rate_30cy"]) if "fade_rate_30cy" in df.columns else None,
        "fade_rate_50cy": float(latest["fade_rate_50cy"]) if "fade_rate_50cy" in df.columns else None,
        "fade_trend": fade_trend,
        "eol_at_cycle": eol_at_cycle,
        "rul_pred": float(latest["rul_pred"]) if "rul_pred" in df.columns else None,
        "rul_q10": float(latest["rul_q10"]) if "rul_q10" in df.columns else None,
        "rul_q90": float(latest["rul_q90"]) if "rul_q90" in df.columns else None,
        "knee_detected": bool(knee["detected"]),
        "knee_cycle": knee["cycle"],
        "knee_soh": knee["soh_at_knee"],
        "knee_confidence": knee["confidence"],
        "knee_phase": knee["phase"],
        "grade": grade,
        "grade_score": grade_score,
        "grade_fade": grade_fade,
        "grade_var": grade_var,
        "grade_slope": grade_slope,
    }


def _early_life_grade(df: pd.DataFrame):
    """A/B/C grade from the first 100 cycles -- Severson et al. (2019)
    method, identical formula previously duplicated in fleet.py and
    grading.py. Returns (grade, score, fade, var, slope); all None except
    grade ("--") when there isn't enough early-life data."""
    import numpy as np

    early = df[df["cycle_number"] <= 100]
    if len(early) < 20 or "capacity_ah" not in early.columns:
        return "—", None, None, None, None

    cap0 = float(early["capacity_ah"].iloc[0])
    res0 = max(float(early["resistance_ohm"].iloc[0]), 1e-6) if "resistance_ohm" in early.columns else 1e-6
    fade = (float(early["capacity_ah"].iloc[0]) - float(early["capacity_ah"].iloc[-1])) / len(early)
    var = float(early["capacity_ah"].var())
    slope = (
        float(np.polyfit(early["cycle_number"], early["resistance_ohm"], 1)[0])
        if "resistance_ohm" in early.columns else 0.0
    )
    fade_pct_per_cy = fade / cap0 * 100
    cv2 = var / (cap0 ** 2) * 1e4
    r_slope_pct = abs(slope) / res0 * 100
    score = float(np.clip(100 - fade_pct_per_cy * 400 - cv2 * 8 - r_slope_pct * 150, 0, 100))
    grade = "A" if score >= 75 else ("B" if score >= 50 else "C")
    return grade, score, fade, var, slope


class LazyCellFrameMap(collections.abc.Mapping):
    """A dict-like {cell_id: DataFrame} view backed by get_cell_df(), so
    every existing single-cell/bounded-subset consumer (featured_dfs[cid],
    .items(), .values(), **-unpacking) keeps working unchanged while the
    full per-cycle data itself is lazily loaded and LRU-bounded rather than
    held resident for the whole process lifetime."""

    def __init__(self, cell_ids, loader=get_cell_df):
        self._cell_ids = list(cell_ids)
        self._cell_id_set = set(self._cell_ids)
        self._loader = loader

    def __getitem__(self, cell_id):
        # Membership must be checked against this map's own cell_ids first --
        # get_cell_df() looks up any cell that exists anywhere in the global
        # Parquet store, not just the ones this particular filtered view
        # (e.g. app/main.py's nasa_fdfs/sev_fdfs/synth_fdfs) is scoped to.
        # Without this check, `"B0005" in nasa_only_map` would incorrectly
        # return True for a cell that exists in the store under a different
        # source's map, since Mapping's default __contains__ is built on
        # __getitem__.
        if cell_id not in self._cell_id_set:
            raise KeyError(cell_id)
        df = self._loader(cell_id)
        if df is None:
            raise KeyError(cell_id)
        return df

    def __iter__(self):
        return iter(self._cell_ids)

    def __len__(self):
        return len(self._cell_ids)
