"""Structural guard against reintroducing full-fleet DataFrame iteration.

Before src/cell_store.py, app/main.py's load_everything() held every
reference-fleet cell's full per-cycle DataFrame in one process-wide dict
for the app's entire lifetime -- see docs/history.md's Production
Readiness Roadmap. The fix has two halves, both real: (1) per-cell lazy
Parquet loading, LRU-bounded, via cell_store.get_cell_df()/LazyCellFrameMap
-- transparent to single-cell/bounded-subset consumers, no code changes
needed; (2) precomputed CellSummary rows (src/db.py) for the genuinely
full-fleet consumers (Fleet's ranking table, Grading, Compliance's
regulatory alerts, Decide & Ask's/EOL Economics' peer-fade comparisons,
the sidebar's fleet alerts, the fleet-wide Copilot/webhook-digest stats,
src/api.py's fleet endpoints), migrated off touching every cell's full
DataFrame on every render.

Unlike test_source_classification_guard.py (one canonical function to
redirect to), there are two legitimate replacements here -- a CellSummary
query, or staying on the lazy map when a consumer genuinely needs full or
windowed per-cycle series (knee/spread/histogram/anomaly-log/clustering/
trajectory-matching -- pre-summarizing those into CellSummary would either
lose real information or couple CellSummary's schema to one algorithm's
internals, see cell_store.py's own module docstring). So this guard can't
say "call X instead" -- it asserts an explicit allowlist of file:line
sites already reviewed as legitimate, and fails on any *new* full-fleet
`.items()`/`.values()` iteration outside it, forcing deliberate review
instead of a silent regression back to touching every cell's full data on
every render.
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).parent.parent
APP_DIR = REPO_ROOT / "app"
SRC_DIR = REPO_ROOT / "src"

# Variable names this codebase actually uses for a {cell_id: DataFrame}
# collection covering more than one or two cells (see e.g. app/main.py's
# featured_dfs/active_fdfs/nasa_fdfs/sev_fdfs/synth_fdfs, src/api.py's
# org_fdfs, src/trajectory_memory.py's all_featured_dfs).
_FDFS_VAR_RE = r"(?:active_fdfs|featured_dfs|nasa_fdfs|sev_fdfs|synth_fdfs|all_featured_dfs|org_fdfs|up_fdfs|_featured_dfs_all)"
_ITER_RE = re.compile(rf"\b{_FDFS_VAR_RE}\.(?:items|values)\(\)")

# file:line -> why this specific full-fleet iteration is legitimate and
# doesn't need to (or can't cleanly) migrate to a CellSummary query.
_ALLOWLIST = {
    "app/main.py:335": (
        "Inside _persist_cell_data() -- the one place allowed to touch "
        "every cell's full DataFrame: it's the write path that populates "
        "cell_store's Parquet files + CellSummary in the first place, "
        "right after _train_and_predict() freshly computes them."
    ),
    "app/_pages/explore.py:289": (
        "Explore's Cluster tab -- opt-in secondary view, needs "
        "ce_rolling_30cy (not in CellSummary's schema, and not worth "
        "growing it for one tab; see cell_store.py's module docstring)."
    ),
    "app/_pages/_fleet_diagnostics.py:673": (
        "Spread trending -- needs every cell's full soh_pct/cycle_number "
        "series to interpolate cross-fleet SOH spread over cycle number, "
        "not reducible to a last-value scalar."
    ),
    "app/_pages/_fleet_diagnostics.py:684": (
        "Spread trending (same function as :673)."
    ),
    "app/_pages/_fleet_diagnostics.py:747": (
        "SOH distribution-shift histogram -- needs every cell's full "
        "soh_pct/cycle_number series to build historical snapshots."
    ),
    "app/_pages/_fleet_diagnostics.py:1034": (
        "Anomaly Alert History -- needs each cell's full "
        "capacity_anomaly/resistance_anomaly boolean columns to count "
        "total flags and recent-window flags, not just the latest value."
    ),
    "src/api.py:358": (
        "Inside _cell_stat_rows() -- iterates only the requesting org's "
        "own small, session-bounded uploaded fleet (org_fdfs from "
        "load_tenant_bundle()), which is explicitly out of scope for "
        "CellSummary (see cell_store.py's module docstring)."
    ),
    "src/fleet_clustering.py:86": (
        "cluster_fleet() has zero callers anywhere in app/ or src/ -- "
        "confirmed dead code, so this never actually executes. Left "
        "allowlisted rather than modified since removing/migrating dead "
        "code is a separate cleanup, not part of this fix."
    ),
    "src/knowledge_graph.py:478": (
        "populate_reference_fleet(), called once per process via "
        "get_platform_graph()'s @st.cache_resource -- a one-time "
        "graph-build cost, not a per-render one."
    ),
    "src/knowledge_graph.py:501": (
        "Same one-time graph-build pass as :478 (edge-building loop)."
    ),
    "src/trajectory_memory.py:211": (
        "TrajectoryMemory.build() needs each cell's tail(WINDOW_CYCLES) "
        "recent raw values for its trend-vector signature -- genuinely "
        "windowed, not summarizable into CellSummary without coupling "
        "that table's schema to this one algorithm's internals."
    ),
    "src/trajectory_memory.py:385": (
        "match_fleet() -- same windowed-series need as build() (:211), "
        "and it's called via app/utils.py's cached_match_fleet(), itself "
        "st.cache_data-memoized so this only re-runs when the active "
        "fleet's cell-id list actually changes."
    ),
}


def _checked_files():
    for base in (APP_DIR, SRC_DIR):
        yield from sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_new_full_fleet_dataframe_iteration_outside_allowlist():
    offenders = []
    for path in _checked_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if _ITER_RE.search(line):
                key = f"{rel}:{lineno}"
                if key not in _ALLOWLIST:
                    offenders.append(f"{key}: {line.strip()}")

    assert not offenders, (
        "New full-fleet DataFrame .items()/.values() iteration found outside "
        "the reviewed allowlist in this test's _ALLOWLIST -- either migrate "
        "it to a db.get_cell_summaries() query (see src/cell_store.py's "
        "module docstring for the pattern used throughout Fleet/Grading/"
        "Compliance/Decide & Ask/etc.), or if it genuinely needs full/"
        "windowed per-cycle series, add it to _ALLOWLIST with a one-line "
        "justification like the existing entries:\n" + "\n".join(offenders)
    )


def test_allowlist_entries_still_exist_at_their_recorded_location():
    """The inverse check: an allowlist entry whose line no longer matches
    (moved, refactored away, or the iteration itself was removed) is stale
    and should be cleaned up or updated, not silently ignored forever."""
    stale = []
    for key in _ALLOWLIST:
        rel, lineno = key.rsplit(":", 1)
        path = REPO_ROOT / rel
        lines = path.read_text(encoding="utf-8").splitlines()
        idx = int(lineno) - 1
        if idx < 0 or idx >= len(lines) or not _ITER_RE.search(lines[idx]):
            stale.append(key)

    assert not stale, (
        "Allowlist entries that no longer match a real full-fleet iteration "
        "at their recorded file:line (moved or removed -- update this "
        "test's _ALLOWLIST):\n" + "\n".join(stale)
    )
