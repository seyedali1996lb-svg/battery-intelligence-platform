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
say "call X instead" -- it asserts an explicit allowlist of sites already
reviewed as legitimate, and fails on any *new* full-fleet
`.items()`/`.values()` iteration outside it, forcing deliberate review
instead of a silent regression back to touching every cell's full data on
every render.

Sites are keyed by ``file::enclosing.function`` (resolved via AST), not by
line number: the original file:line keys went stale every time an
unrelated edit landed above one of them (2026-09-13: six entries drifted
at once after a feature commit and the guard failed on code that had not
changed). A function key survives that; each entry also carries the
number of iterations reviewed inside that function, so a second site
pasted into an already-allowlisted function still trips the guard.
"""

import ast
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

# "file::enclosing.function" -> (reviewed iteration count in that function,
# why this full-fleet iteration is legitimate and doesn't need to (or can't
# cleanly) migrate to a CellSummary query). Nested functions use dotted
# names (load_everything._persist_cell_data); module-level code is "<module>".
_ALLOWLIST: dict[str, tuple[int, str]] = {
    # _persist_cell_data() was extracted out of load_everything() to module
    # level (it is now called from two write paths there); the key follows the
    # function's actual scope, per this module's file::enclosing.function rule.
    "app/_data.py::_persist_cell_data": (1,
        "The write path that populates cell_store's Parquet files + CellSummary, "
        "called from load_everything() -- a persistence pass, not a render pass."
    ),
    "app/_pages/explore.py::_page_explore_cluster": (1,
        "Explore's Cluster tab -- opt-in secondary view, needs "
        "ce_rolling_30cy (not in CellSummary's schema, and not worth "
        "growing it for one tab; see cell_store.py's module docstring)."
    ),
    "app/_pages/_fleet_diagnostics.py::render_spread_trending": (2,
        "Spread trending -- one pre-filter pass over every cell's full "
        "soh_pct to drop empty DataFrames, then the interpolation pass "
        "over each cell's soh_pct/cycle_number series to build cross-fleet "
        "SOH spread over cycle number; not reducible to a last-value scalar."
    ),
    "app/_pages/_fleet_diagnostics.py::render_distribution_shift_histogram": (1,
        "SOH distribution-shift histogram -- needs every cell's full "
        "soh_pct/cycle_number series to build historical snapshots."
    ),
    "app/_pages/_fleet_diagnostics.py::render_anomaly_alert_history": (1,
        "Anomaly Alert History -- needs each cell's full "
        "capacity_anomaly/resistance_anomaly boolean columns to count "
        "total flags and recent-window flags, not just the latest value."
    ),
    "src/api.py::_cell_stat_rows": (1,
        "Iterates only the requesting org's own small, session-bounded "
        "uploaded fleet (org_fdfs from load_tenant_bundle()), which is "
        "explicitly out of scope for CellSummary (see cell_store.py's "
        "module docstring)."
    ),
    "src/fleet_clustering.py::cluster_fleet": (1,
        "cluster_fleet() has zero callers anywhere in app/ or src/ -- "
        "confirmed dead code, so this never actually executes. Left "
        "allowlisted rather than modified since removing/migrating dead "
        "code is a separate cleanup, not part of this fix."
    ),
    "src/knowledge_graph.py::populate_reference_fleet": (2,
        "Called once per process via get_platform_graph()'s "
        "@st.cache_resource -- a one-time graph-build cost (node pass + "
        "edge-building loop), not a per-render one."
    ),
    "src/trajectory_memory.py::TrajectoryMemory.build": (1,
        "build() needs each cell's tail(WINDOW_CYCLES) recent raw values "
        "for its trend-vector signature -- genuinely windowed, not "
        "summarizable into CellSummary without coupling that table's "
        "schema to this one algorithm's internals."
    ),
    "src/trajectory_memory.py::TrajectoryMemory.match_fleet": (1,
        "match_fleet() -- same windowed-series need as build(), and it's "
        "called via app/utils.py's cached_match_fleet(), itself "
        "st.cache_data-memoized so this only re-runs when the active "
        "fleet's cell-id list actually changes."
    ),
}


def _checked_files():
    for base in (APP_DIR, SRC_DIR):
        yield from sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _enclosing_scopes(source: str) -> dict[int, str]:
    """line number -> dotted name of the innermost def/class containing it."""
    scopes: dict[int, str] = {}

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = stack + [child.name]
                for ln in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                    scopes[ln] = ".".join(names)
                visit(child, names)
            else:
                visit(child, stack)

    visit(ast.parse(source), [])
    return scopes


def _iteration_sites() -> dict[str, list[str]]:
    """Every full-fleet iteration in app/ + src/, grouped by allowlist key,
    each entry rendered as 'file:line: source' for failure messages."""
    sites: dict[str, list[str]] = {}
    for path in _checked_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        scopes = _enclosing_scopes(source)
        for lineno, line in enumerate(source.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if _ITER_RE.search(line):
                key = f"{rel}::{scopes.get(lineno, '<module>')}"
                sites.setdefault(key, []).append(f"{rel}:{lineno}: {line.strip()}")
    return sites


def test_no_new_full_fleet_dataframe_iteration_outside_allowlist():
    offenders = []
    for key, found in _iteration_sites().items():
        allowed = _ALLOWLIST.get(key)
        if allowed is None:
            offenders.extend(f"{s}  [{key}: not allowlisted]" for s in found)
        elif len(found) > allowed[0]:
            offenders.extend(
                f"{s}  [{key}: {len(found)} iterations, {allowed[0]} reviewed]"
                for s in found
            )
    assert not offenders, (
        "New full-fleet DataFrame .items()/.values() iteration found outside "
        "the reviewed allowlist in this test's _ALLOWLIST -- either migrate "
        "it to a db.get_cell_summaries() query (see src/cell_store.py's "
        "module docstring for the pattern used throughout Fleet/Grading/"
        "Compliance/Decide & Ask/etc.), or if it genuinely needs full/"
        "windowed per-cycle series, add it to _ALLOWLIST (key = "
        "file::enclosing.function, value = (count, justification)) like "
        "the existing entries:\n" + "\n".join(offenders)
    )


def test_allowlist_entries_still_exist_at_their_recorded_location():
    """The inverse check: an allowlist entry whose function no longer
    contains that many full-fleet iterations (moved, refactored away, or
    the iteration itself was removed) is stale and should be cleaned up or
    updated, not silently ignored forever."""
    sites = _iteration_sites()
    stale = []
    for key, (count, _why) in _ALLOWLIST.items():
        found = len(sites.get(key, []))
        if found != count:
            stale.append(f"{key}: {count} reviewed, {found} found")
    assert not stale, (
        "Allowlist entries that no longer match the recorded number of "
        "full-fleet iterations in that function (moved or removed -- update "
        "this test's _ALLOWLIST):\n" + "\n".join(stale)
    )
