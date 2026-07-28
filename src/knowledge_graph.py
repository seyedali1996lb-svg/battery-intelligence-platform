"""
Battery Digital Knowledge Graph — Phase 5.

Why a graph, and why in-process NetworkX
-----------------------------------------
Every other data surface in this platform is flat: one row per cell, one
table per feature. There is no queryable structure for "this cell's
mechanism verdict corroborates this literature claim" or "show me cells
like this one" — those questions currently require re-deriving the answer
by hand in every page that asks them (see the mechanism-verdict-computed-
independently-in-three-places bug class this project has already found
and fixed twice — app/_pages/health.py and app/_pages/decision.py used to
each carry their own inline re-implementation of the LLI/LAM classifier
before being consolidated onto one diagnose_mechanism() call each; this
module goes one step further and makes them share one *edge*, not just
one function). The corpus here is thousands of nodes (cells, literature
docs, experiment runs), not billions — an external graph database would
be unjustified infrastructure for this scale. NetworkX's in-process
MultiDiGraph is enough, and it stays honest about what it is: a
process-local structure, snapshotted to SQLite (src/db.py's kg_nodes /
kg_edges tables) for persistence across restarts, not a live distributed
store.

Schema
------
Node types (NODE_TYPES):
  cell                — one battery cell (id = cell_id, e.g. "B0005", "S-12")
  chemistry           — a registered chemistry (id = ChemistryProfile.short_name,
                         e.g. "LFP", "LiCoO2", "NCA")
  dataset             — a data source (id = "nasa" | "synth" | "severson" |
                         "oxford" | "uploaded")
  mechanism           — a degradation-mechanism verdict type. Exactly four
                         nodes exist, always, seeded by new_graph(): "lli",
                         "lam", "mixed", "insufficient_data" (mirroring
                         recommendations.diagnose_mechanism()'s four possible
                         verdicts).
  literature           — one curated corpus entry (id = doc id from
                         battery_knowledge.DOCUMENTS)
  experiment_run       — one logged GBRT run (id = run_id from
                         experiment_registry.py)

Edge types (EDGE_TYPES) — deliberately only these four, reused across node
pairs rather than growing a new edge type per relationship:
  exhibits      cell -> mechanism        (this cell's diagnosed mechanism)
  derived_from  cell -> dataset          (which raw source this cell's data comes from)
                cell -> chemistry        (its chemistry classification is derived
                                          from ChemistryProfile.for_cell())
                cell -> experiment_run   (its predictions come from this logged
                                          training run)
  corroborates  mechanism -> literature  (curated: this literature entry supports
                                          this mechanism's definition/diagnostic signal)
  contradicts   cell -> mechanism        (this cell's SOH-based recommended action
                                          is in tension with its own diagnosed LAM
                                          mechanism's implied urgency — reuses
                                          recommendations.mechanism_corroboration_note(),
                                          the same arbitration logic already live on
                                          the Decide & Ask page. NOT a claim that the
                                          mechanism diagnosis itself is wrong.)

Provenance is not optional
---------------------------
add_edge() refuses to add any edge without a source_fn (a "module.function"
or "module.Class.method" string naming the real code that computed it) or a
doi. scripts/audit_knowledge_graph_provenance.py (wired into CI) rebuilds
the static parts of this graph and additionally verifies every source_fn
actually resolves to a real, importable attribute — catching the exact
"renamed a function, forgot the string that pointed at it" drift this
project's history has hit before (abandoned-refactor dead code, stale
signatures), applied to graph edges instead of Python call sites.

Ownership split (same pattern as trajectory_memory.py / experiment_registry.py)
---------------------------------------------------------------------------
This module owns the graph data structure and all domain logic (schema,
population, queries, provenance). src/db.py owns storage only — two plain
edge-list tables (kg_nodes, kg_edges), scoped by org_id exactly like every
other multi-tenant table in that file. PLATFORM_ORG_ID (reused directly
from experiment_registry.py, not redefined here) is used for the graph
built from the platform's shared reference fleets (NASA/synthetic/
Severson) — the same sentinel already used for their experiment runs.
"""

from __future__ import annotations

import datetime
import importlib
import json
import re

import networkx as nx

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

NODE_CELL           = "cell"
NODE_CHEMISTRY      = "chemistry"
NODE_DATASET        = "dataset"
NODE_MECHANISM      = "mechanism"
NODE_LITERATURE     = "literature"
NODE_EXPERIMENT_RUN = "experiment_run"

NODE_TYPES = frozenset({
    NODE_CELL, NODE_CHEMISTRY, NODE_DATASET,
    NODE_MECHANISM, NODE_LITERATURE, NODE_EXPERIMENT_RUN,
})

EDGE_EXHIBITS     = "exhibits"
EDGE_DERIVED_FROM = "derived_from"
EDGE_CORROBORATES = "corroborates"
EDGE_CONTRADICTS  = "contradicts"

EDGE_TYPES = frozenset({EDGE_EXHIBITS, EDGE_DERIVED_FROM, EDGE_CORROBORATES, EDGE_CONTRADICTS})

# recommendations.diagnose_mechanism()'s four possible verdict strings, mapped
# to the four fixed mechanism node ids this graph always seeds.
MECHANISM_VERDICT_TO_KEY = {
    "LLI — Loss of Lithium Inventory": "lli",
    "LAM — Loss of Active Material":   "lam",
    "Mixed LLI + LAM":                 "mixed",
    "Insufficient data":               "insufficient_data",
}
MECHANISM_KEYS = ("lli", "lam", "mixed", "insufficient_data")
MECHANISM_LABELS = {
    "lli":               "LLI — Loss of Lithium Inventory",
    "lam":                "LAM — Loss of Active Material",
    "mixed":              "Mixed LLI + LAM",
    "insufficient_data":  "Insufficient data",
}

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def node_key(node_type: str, node_id: str) -> str:
    """The networkx node key — globally unique across node types."""
    return f"{node_type}:{node_id}"


# ---------------------------------------------------------------------------
# Graph construction primitives
# ---------------------------------------------------------------------------

def new_graph() -> nx.MultiDiGraph:
    """A fresh graph, pre-seeded with the four fixed mechanism nodes (always
    present so exhibits/corroborates/contradicts edges always have a valid
    target, even before any cell has been added)."""
    g = nx.MultiDiGraph()
    for key in MECHANISM_KEYS:
        _add_node(g, NODE_MECHANISM, key, label=MECHANISM_LABELS[key])
    return g


def _add_node(g: nx.MultiDiGraph, node_type: str, node_id: str, **attrs) -> str:
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unknown node type {node_type!r} — must be one of {sorted(NODE_TYPES)}")
    key = node_key(node_type, node_id)
    if key in g.nodes:
        g.nodes[key].update(attrs)
    else:
        g.add_node(key, type=node_type, id=node_id, **attrs)
    return key


def add_cell(g: nx.MultiDiGraph, cell_id: str, **attrs) -> str:
    return _add_node(g, NODE_CELL, cell_id, **attrs)


def add_chemistry(g: nx.MultiDiGraph, short_name: str, **attrs) -> str:
    return _add_node(g, NODE_CHEMISTRY, short_name, **attrs)


def add_dataset(g: nx.MultiDiGraph, dataset_key: str, **attrs) -> str:
    return _add_node(g, NODE_DATASET, dataset_key, **attrs)


def add_literature(g: nx.MultiDiGraph, doc_id: str, **attrs) -> str:
    return _add_node(g, NODE_LITERATURE, doc_id, **attrs)


def add_experiment_run(g: nx.MultiDiGraph, run_id: str, **attrs) -> str:
    return _add_node(g, NODE_EXPERIMENT_RUN, run_id, **attrs)


def add_edge(
    g: nx.MultiDiGraph,
    src_type: str, src_id: str,
    dst_type: str, dst_id: str,
    edge_type: str,
    source_fn: "str | None" = None,
    doi: "str | None" = None,
    **attrs,
) -> None:
    """Add (or update, if the same (src, dst, edge_type) triple already
    exists — population is idempotent by design so rebuilding the graph
    never duplicates edges) one provenanced edge.

    Every edge requires source_fn and/or doi — see this module's docstring
    on why provenance is enforced here rather than left as an optional
    convention. Both endpoints must already exist as nodes.
    """
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Unknown edge type {edge_type!r} — must be one of {sorted(EDGE_TYPES)}")
    if not source_fn and not doi:
        raise ValueError(
            f"Refusing to add a {edge_type} edge ({src_type}:{src_id} -> "
            f"{dst_type}:{dst_id}) with no source_fn and no doi — every edge "
            "in this graph must be traceable to the real code or literature "
            "that produced it."
        )
    src_key = node_key(src_type, src_id)
    dst_key = node_key(dst_type, dst_id)
    if src_key not in g.nodes:
        raise ValueError(f"Source node {src_key!r} does not exist — add it before adding an edge from it.")
    if dst_key not in g.nodes:
        raise ValueError(f"Destination node {dst_key!r} does not exist — add it before adding an edge to it.")
    g.add_edge(
        src_key, dst_key, key=edge_type,
        type=edge_type, source_fn=source_fn, doi=doi,
        computed_at=_now_iso(), **attrs,
    )


def _edges_of_type(g: nx.MultiDiGraph, node_key_: str, edge_type: str, direction: str = "out"):
    it = g.out_edges(node_key_, data=True) if direction == "out" else g.in_edges(node_key_, data=True)
    for u, v, data in it:
        if data.get("type") == edge_type:
            yield u, v, data


def _first_neighbor_of_type(g: nx.MultiDiGraph, node_key_: str, edge_type: str, node_type: str) -> "str | None":
    """The id (not the composite node key) of the first out-neighbor reached
    via an edge of edge_type whose node type is node_type."""
    for _, v, _ in _edges_of_type(g, node_key_, edge_type, "out"):
        data = g.nodes[v]
        if data.get("type") == node_type:
            return data.get("id")
    return None


# ---------------------------------------------------------------------------
# Population — cell -> dataset / chemistry
# ---------------------------------------------------------------------------

def dataset_key_for_cell(cell_id: str) -> str:
    """Same source-classification rules used throughout the app (NASA_CELL_IDS
    list, 'S-' Severson prefix, 'OX-' Oxford prefix, else synthetic/uploaded —
    see app/utils.py's NASA_CELL_IDS / SEVERSON_CELL_PREFIX and
    chemistry_profiles.ChemistryProfile.for_cell() for the same convention).
    Returns "uploaded" for anything not matching a known built-in pattern —
    callers with better information (e.g. import_page.py, which knows a cell
    came from an upload for certain) should pass dataset_key explicitly
    instead of relying on this guess.
    """
    from data_loader import CELL_STRESS_PROFILES

    nasa_ids = {"B0005", "B0006", "B0007", "B0018"}
    if cell_id in nasa_ids:
        return "nasa"
    if cell_id.startswith("S-"):
        return "severson"
    if cell_id.startswith("OX-"):
        return "oxford"
    if cell_id in CELL_STRESS_PROFILES or cell_id.startswith(("Cell", "OxBat")):
        return "synth"
    return "uploaded"


def link_cell_to_dataset(
    g: nx.MultiDiGraph, cell_id: str, dataset_key: "str | None" = None,
    source_fn: str = "knowledge_graph.dataset_key_for_cell",
) -> str:
    dataset_key = dataset_key or dataset_key_for_cell(cell_id)
    add_dataset(g, dataset_key)
    add_edge(g, NODE_CELL, cell_id, NODE_DATASET, dataset_key, EDGE_DERIVED_FROM, source_fn=source_fn)
    return dataset_key


def link_cell_to_chemistry(
    g: nx.MultiDiGraph, cell_id: str,
    source_fn: str = "chemistry_profiles.ChemistryProfile.for_cell",
) -> str:
    from chemistry_profiles import ChemistryProfile

    profile = ChemistryProfile.for_cell(cell_id)
    add_chemistry(
        g, profile.short_name,
        display_name=profile.display_name,
        provenance=profile.provenance,
        dataset_citation=profile.dataset_citation,
    )
    add_edge(g, NODE_CELL, cell_id, NODE_CHEMISTRY, profile.short_name, EDGE_DERIVED_FROM, source_fn=source_fn)
    return profile.short_name


# ---------------------------------------------------------------------------
# Population — cell -> mechanism (the one shared "exhibits" edge)
# ---------------------------------------------------------------------------

def mechanism_edge(g: nx.MultiDiGraph, cell_id: str) -> "dict | None":
    """Read the cell's already-computed exhibits edge, if one exists.
    This is the single read path Health / the Cell Workbench / the Copilot
    all use — see get_or_compute_mechanism() for the write side."""
    cell_key = node_key(NODE_CELL, cell_id)
    if cell_key not in g.nodes:
        return None
    for _, _, data in _edges_of_type(g, cell_key, EDGE_EXHIBITS, "out"):
        return data
    return None


def link_cell_to_mechanism(
    g: nx.MultiDiGraph, cell_id: str, df,
    source_fn: str = "recommendations.diagnose_mechanism",
) -> dict:
    """Compute this cell's mechanism verdict and add/overwrite its exhibits
    edge. Always recomputes — callers that want the "compute once, read
    everywhere" behavior should call get_or_compute_mechanism() instead.

    Stores diagnose_mechanism()'s full return dict (verdict/colors/icon/
    body/confidence/scores/signals) as edge attrs, not a trimmed subset —
    Health's compact card and deep expander both render verdict_color/
    verdict_icon/confidence_color/signals directly, so a partial edge would
    either KeyError or silently show the wrong color for a cell's actual
    verdict. signals' values are cast to plain floats (diagnose_mechanism()
    computes them via numpy.polyfit, which returns numpy.float64 — not
    JSON-serializable as-is, and this edge must round-trip through
    src/db.py's JSON-encoded kg_edges.attrs column).
    """
    from recommendations import diagnose_mechanism

    add_cell(g, cell_id)
    mech = diagnose_mechanism(df)
    key = MECHANISM_VERDICT_TO_KEY.get(mech["verdict"], "insufficient_data")
    add_edge(
        g, NODE_CELL, cell_id, NODE_MECHANISM, key, EDGE_EXHIBITS, source_fn=source_fn,
        verdict=mech["verdict"], verdict_body=mech["verdict_body"],
        verdict_color=mech["verdict_color"], verdict_icon=mech["verdict_icon"],
        confidence_label=mech["confidence_label"], confidence_color=mech["confidence_color"],
        lli_score=mech["lli_score"], lam_score=mech["lam_score"],
        confidence_notes=list(mech["confidence_notes"]),
        signals={k: float(v) for k, v in mech["signals"].items()},
    )
    return mechanism_edge(g, cell_id)


def get_or_compute_mechanism(
    g: nx.MultiDiGraph, cell_id: str, df,
    source_fn: str = "recommendations.diagnose_mechanism",
) -> dict:
    """The one function Health / the Cell Workbench (Decide & Ask) / the
    Copilot should all call for a cell's mechanism verdict. If the shared
    graph already has this cell's exhibits edge (e.g. precomputed once by
    build_platform_graph() for every reference-fleet cell), that edge is
    returned as-is — no second, independent diagnose_mechanism() call, so
    the three surfaces cannot silently diverge on the same cell the way
    they have twice before in this project's history. If the cell isn't in
    the graph yet (e.g. a freshly-uploaded cell not yet part of the platform
    graph), it's computed once here and added.
    """
    existing = mechanism_edge(g, cell_id)
    if existing is not None:
        return existing
    return link_cell_to_mechanism(g, cell_id, df, source_fn=source_fn)


def link_cell_action_mechanism_conflict(
    g: nx.MultiDiGraph, cell_id: str, action: str, mechanism: dict,
    source_fn: str = "recommendations.mechanism_corroboration_note",
) -> bool:
    """Add a contradicts edge (cell -> its own diagnosed LAM mechanism) when
    recommendations.mechanism_corroboration_note() detects the SOH-based
    recommended action is in tension with the mechanism's implied urgency —
    reusing that function's exact logic rather than re-deriving the check.
    Returns True if a contradicts edge was added, False if nothing disagreed
    (the common case — see that function's own docstring). This is NOT a
    claim the mechanism diagnosis is wrong; it flags that two independent
    analytical surfaces (the action classifier and the mechanism classifier)
    disagree, exactly the "no arbitration between analytical surfaces"
    finding already fixed once on the Decide & Ask page (see
    recommendations.mechanism_corroboration_note()'s own docstring).
    """
    from recommendations import mechanism_corroboration_note

    note = mechanism_corroboration_note(action, mechanism)
    if note is None:
        return False
    key = MECHANISM_VERDICT_TO_KEY.get(mechanism.get("verdict", ""), "insufficient_data")
    add_edge(
        g, NODE_CELL, cell_id, NODE_MECHANISM, key, EDGE_CONTRADICTS,
        source_fn=source_fn, action=action, note=note,
    )
    return True


# ---------------------------------------------------------------------------
# Population — cell -> experiment_run
# ---------------------------------------------------------------------------

def link_cells_to_experiment_run(
    g: nx.MultiDiGraph, run: dict,
    source_fn: str = "experiment_registry.leaderboard",
) -> int:
    """Add an ExperimentRun node for one logged run (dict shape from
    experiment_registry.leaderboard()/get_run()) and a derived_from edge
    from every cell in run['cell_ids'] that already exists as a node in g
    (cells not yet present — e.g. a cross-chemistry transfer run's
    off-domain population — are skipped rather than silently creating a
    bare cell node with no other data). Returns the number of edges added.
    """
    add_experiment_run(
        g, run["run_id"],
        dataset=run.get("dataset"), chemistry=run.get("chemistry"),
        soh_mae=run.get("soh_mae"), rul_mae=run.get("rul_mae"),
        timestamp=run.get("timestamp"), git_commit=run.get("git_commit"),
    )
    n = 0
    for cell_id in run.get("cell_ids", []):
        if node_key(NODE_CELL, cell_id) not in g.nodes:
            continue
        add_edge(g, NODE_CELL, cell_id, NODE_EXPERIMENT_RUN, run["run_id"], EDGE_DERIVED_FROM, source_fn=source_fn)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Population — literature ingestion + mechanism corroboration
# ---------------------------------------------------------------------------

# Curated, hand-checked mapping: which battery_knowledge.py corpus entries
# actually support each mechanism node's definition or one of
# diagnose_mechanism()'s three real signals (CE trend, fade-shape
# nonlinearity, resistance-rise rate). Deliberately conservative — only docs
# whose text can be pointed to directly are listed; "insufficient_data" has
# no corroborating literature by design (there is no claim to support).
MECHANISM_LITERATURE_CORROBORATION = {
    "lli":   ("lli-vs-lam", "coulombic-efficiency"),
    "lam":   ("lli-vs-lam", "knee-point-degradation"),
    "mixed": ("lli-vs-lam",),
}


def ingest_literature(
    g: nx.MultiDiGraph,
    source_fn: str = "knowledge_graph.ingest_literature",
) -> int:
    """Add every battery_knowledge.py DOCUMENTS entry as a LiteratureSource
    node, plus the curated corroborates edges from MECHANISM_LITERATURE_CORROBORATION
    above. Returns the number of literature nodes added."""
    from battery_knowledge import DOCUMENTS

    for doc in DOCUMENTS:
        add_literature(g, doc["id"], text=doc["text"])

    for mech_key, doc_ids in MECHANISM_LITERATURE_CORROBORATION.items():
        for doc_id in doc_ids:
            add_edge(g, NODE_MECHANISM, mech_key, NODE_LITERATURE, doc_id, EDGE_CORROBORATES, source_fn=source_fn)

    return len(DOCUMENTS)


# ---------------------------------------------------------------------------
# Orchestration — build the platform-level graph from already-computed
# pipeline data (no new heavy computation: reuses diagnose_mechanism(),
# ChemistryProfile.for_cell(), application_fit()/classify(), and the
# experiment registry exactly as the pages already call them)
# ---------------------------------------------------------------------------

def populate_reference_fleet(g: nx.MultiDiGraph, featured_dfs: dict, bundles: "dict | None" = None) -> None:
    """Add every cell in featured_dfs plus its dataset/chemistry/mechanism
    edges, and (when bundles is supplied) its contradicts edge if the
    default-threshold recommendation and its mechanism verdict disagree.

    Uses recommendations.classify()'s module-level default thresholds, not
    any per-user-configured EOL threshold (Settings' 70-95% slider) — this
    is a structural, platform-level signal, not a live per-viewer
    recommendation. Documented here rather than silently assumed.
    """
    from consequences import application_fit
    from recommendations import classify

    for cell_id, df in featured_dfs.items():
        if len(df) == 0:
            continue
        add_cell(g, cell_id, soh_pct=float(df.iloc[-1]["soh_pct"]), cycle_number=int(df.iloc[-1]["cycle_number"]))
        dataset_key = link_cell_to_dataset(g, cell_id)
        link_cell_to_chemistry(g, cell_id)
        mech = link_cell_to_mechanism(g, cell_id, df)

        if bundles is None:
            continue
        bundle = bundles.get(dataset_key) or bundles.get("synth")
        if bundle is None:
            continue
        latest = df.iloc[-1]
        soh     = float(latest["soh_pct"])
        fade_30 = float(latest.get("fade_rate_30cy", 0.0))
        fade_50 = float(latest.get("fade_rate_50cy", 0.0))
        per_cell_ok  = bundle.get("metrics", {}).get("per_cell_rul_reliable", {})
        rul_reliable = per_cell_ok.get(cell_id, bundle.get("metrics", {}).get("rul_reliable", False))
        rul_pred     = float(latest["rul_pred"]) if (rul_reliable and "rul_pred" in latest.index) else None

        peer_fades = [
            float(other.iloc[-1].get("fade_rate_30cy", 0.0))
            for other_id, other in featured_dfs.items()
            if other_id != cell_id and dataset_key_for_cell(other_id) == dataset_key and len(other) > 0
        ]
        fleet_fade_median = float(sorted(peer_fades)[len(peer_fades) // 2]) if peer_fades else None

        fit_scores = application_fit(soh, fade_30, fleet_fade_median)
        result = classify(soh, fade_30, fade_50, rul_reliable, rul_pred, fit_scores)
        link_cell_action_mechanism_conflict(g, cell_id, result["action"], mech)


def populate_experiment_runs(g: nx.MultiDiGraph, tenant_org_id: "int | None" = None) -> int:
    """Add ExperimentRun nodes + derived_from edges for every logged run
    visible to tenant_org_id (platform runs + that tenant's own, or just
    platform runs if tenant_org_id is None) — see
    experiment_registry.leaderboard(). Returns the number of runs added."""
    import experiment_registry as reg

    runs = reg.leaderboard(tenant_org_id=tenant_org_id)
    for run in runs:
        link_cells_to_experiment_run(g, run)
    return len(runs)


def build_platform_graph(
    featured_dfs: dict, bundles: "dict | None" = None, tenant_org_id: "int | None" = None,
) -> nx.MultiDiGraph:
    """Orchestrates a full graph build from data the app has already
    computed: literature corpus -> reference-fleet cells (dataset/
    chemistry/mechanism/contradicts) -> experiment runs. This is the
    function app/utils.py's cached wrapper calls once per process (mirrors
    app/main.py's load_everything() caching pattern) to produce the one
    graph instance Health / the Cell Workbench / the Copilot / Explore's
    "cells like this" panel all share.
    """
    g = new_graph()
    ingest_literature(g)
    populate_reference_fleet(g, featured_dfs, bundles)
    try:
        populate_experiment_runs(g, tenant_org_id)
    except Exception:
        # Registry unavailable (e.g. no db.py access in this context) —
        # the graph is still useful without experiment_run edges.
        pass
    return g


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def cells_like(g: nx.MultiDiGraph, cell_id: str, top_k: int = 5) -> list[dict]:
    """"Cells like this one" — same chemistry (required, since cross-
    chemistry comparison isn't physically meaningful — see
    battery_knowledge.py's why-resistance-scales-differ entry), ranked by
    mechanism-verdict match first, then closeness in SOH. Returns up to
    top_k dicts: {cell_id, dataset, chemistry, mechanism, soh_pct,
    same_mechanism, same_dataset, soh_diff}.
    """
    cell_key = node_key(NODE_CELL, cell_id)
    if cell_key not in g.nodes:
        return []

    my_chem    = _first_neighbor_of_type(g, cell_key, EDGE_DERIVED_FROM, NODE_CHEMISTRY)
    my_dataset = _first_neighbor_of_type(g, cell_key, EDGE_DERIVED_FROM, NODE_DATASET)
    my_mech    = _first_neighbor_of_type(g, cell_key, EDGE_EXHIBITS, NODE_MECHANISM)
    my_soh     = g.nodes[cell_key].get("soh_pct")

    if my_chem is None:
        return []

    candidates = []
    for node, data in g.nodes(data=True):
        if data.get("type") != NODE_CELL or node == cell_key:
            continue
        their_chem = _first_neighbor_of_type(g, node, EDGE_DERIVED_FROM, NODE_CHEMISTRY)
        if their_chem != my_chem:
            continue
        their_dataset = _first_neighbor_of_type(g, node, EDGE_DERIVED_FROM, NODE_DATASET)
        their_mech    = _first_neighbor_of_type(g, node, EDGE_EXHIBITS, NODE_MECHANISM)
        their_soh     = data.get("soh_pct")

        same_mechanism = my_mech is not None and their_mech == my_mech
        soh_diff = abs(my_soh - their_soh) if (my_soh is not None and their_soh is not None) else float("inf")

        candidates.append({
            "cell_id":        data["id"],
            "dataset":        their_dataset,
            "chemistry":      their_chem,
            "mechanism":      their_mech,
            "soh_pct":        their_soh,
            "same_mechanism": same_mechanism,
            "same_dataset":   their_dataset == my_dataset,
            "soh_diff":       soh_diff,
        })

    candidates.sort(key=lambda c: (0 if c["same_mechanism"] else 1, c["soh_diff"]))
    return candidates[:top_k]


def literature_for_mechanism(g: nx.MultiDiGraph, mechanism_key: str) -> list[dict]:
    """Literature nodes corroborating a mechanism (via its corroborates
    edges) — used to show citations backing a cell's mechanism verdict."""
    mech_node = node_key(NODE_MECHANISM, mechanism_key)
    if mech_node not in g.nodes:
        return []
    out = []
    for _, v, _ in _edges_of_type(g, mech_node, EDGE_CORROBORATES, "out"):
        data = g.nodes[v]
        out.append({"id": data["id"], "text": data.get("text", "")})
    return out


def provenance_audit(g: nx.MultiDiGraph) -> list[dict]:
    """Every edge failing provenance: missing both source_fn and doi, a
    source_fn that doesn't resolve to a real importable attribute, or a doi
    that doesn't match DOI syntax. Returns a list of violation dicts (empty
    = clean). Used by scripts/audit_knowledge_graph_provenance.py (CI) and
    directly in tests.
    """
    violations = []
    for u, v, data in g.edges(data=True):
        edge_type = data.get("type", "?")
        source_fn = data.get("source_fn")
        doi       = data.get("doi")

        if not source_fn and not doi:
            violations.append({"edge": (u, v, edge_type), "reason": "no source_fn and no doi"})
            continue

        if source_fn:
            if not _resolves(source_fn):
                violations.append({
                    "edge": (u, v, edge_type),
                    "reason": f"source_fn {source_fn!r} does not resolve to a real importable attribute",
                })

        if doi and not _DOI_RE.match(doi):
            violations.append({"edge": (u, v, edge_type), "reason": f"doi {doi!r} is not a well-formed DOI"})

    return violations


def _resolves(source_fn: str) -> bool:
    """Best-effort check that a 'module.attr' or 'module.Class.attr' string
    names something real and importable. Returns False (not a raised
    exception) for any resolution failure so provenance_audit() can collect
    every violation in one pass instead of stopping at the first."""
    parts = source_fn.split(".")
    if len(parts) < 2:
        return False
    module_name = parts[0]
    try:
        obj = importlib.import_module(module_name)
    except ImportError:
        return False
    for attr in parts[1:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return False
    return True


# ---------------------------------------------------------------------------
# Serialization — plain rows for src/db.py's kg_nodes / kg_edges tables
# ---------------------------------------------------------------------------

_RESERVED_NODE_ATTRS = ("type", "id")
_RESERVED_EDGE_ATTRS = ("type", "source_fn", "doi")


def graph_to_rows(g: nx.MultiDiGraph) -> "tuple[list[dict], list[dict]]":
    """Flatten a graph into (node_rows, edge_rows) — plain dicts matching
    src/db.py's KGNode / KGEdge columns exactly (attrs JSON-encoded)."""
    node_rows = []
    for node, data in g.nodes(data=True):
        extra = {k: v for k, v in data.items() if k not in _RESERVED_NODE_ATTRS}
        node_rows.append({
            "node_type": data["type"], "node_id": data["id"],
            "attrs": json.dumps(extra),
        })

    edge_rows = []
    for u, v, k, data in g.edges(keys=True, data=True):
        extra = {kk: vv for kk, vv in data.items() if kk not in _RESERVED_EDGE_ATTRS}
        edge_rows.append({
            "edge_id":   f"{u}|{v}|{k}",
            "edge_type": data["type"],
            "src_key":   u,
            "dst_key":   v,
            "source_fn": data.get("source_fn"),
            "doi":       data.get("doi"),
            "attrs":     json.dumps(extra),
        })

    return node_rows, edge_rows


def graph_from_rows(node_rows: list, edge_rows: list) -> nx.MultiDiGraph:
    """Reconstruct a graph from src/db.py's saved rows (inverse of
    graph_to_rows())."""
    g = nx.MultiDiGraph()
    for r in node_rows:
        attrs = json.loads(r["attrs"]) if r["attrs"] else {}
        g.add_node(node_key(r["node_type"], r["node_id"]), type=r["node_type"], id=r["node_id"], **attrs)
    for r in edge_rows:
        attrs = json.loads(r["attrs"]) if r["attrs"] else {}
        g.add_edge(
            r["src_key"], r["dst_key"], key=r["edge_type"],
            type=r["edge_type"], source_fn=r["source_fn"], doi=r["doi"], **attrs,
        )
    return g


def save_graph(org_id: int, g: nx.MultiDiGraph) -> None:
    """Persist a full snapshot of g to src/db.py's kg_nodes/kg_edges tables,
    replacing any previously saved snapshot for org_id."""
    import db

    node_rows, edge_rows = graph_to_rows(g)
    db.save_knowledge_graph(org_id, node_rows, edge_rows)


def load_graph(org_id: int) -> nx.MultiDiGraph:
    """Reconstruct a graph from its last-saved snapshot for org_id. Returns
    an empty (but mechanism-seeded) graph if nothing has been saved yet."""
    import db

    node_rows, edge_rows = db.load_knowledge_graph_rows(org_id)
    if not node_rows:
        return new_graph()
    return graph_from_rows(node_rows, edge_rows)
