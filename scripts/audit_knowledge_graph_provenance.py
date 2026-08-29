"""
CI gate: fail if any Battery Knowledge Graph edge (src/knowledge_graph.py)
lacks traceable provenance.

Scope — what this script builds and why
----------------------------------------
A full production graph (populate_reference_fleet() + populate_experiment_runs())
needs trained GBRT bundles and, for Severson, several GB of raw data this CI
job doesn't have cached. That's the wrong thing to gate a fast CI job on
anyway: knowledge_graph.add_edge() already REFUSES to add any edge without a
source_fn or doi at construction time (see its docstring), so a full
production graph can only ever contain edges built through this module's own
API — the interesting failure mode isn't "an edge slipped through
unprovenanced" (structurally prevented), it's "a source_fn string still
LOOKS valid but the function it names was renamed or deleted" — the exact
class of drift this project's history has hit repeatedly elsewhere
(abandoned refactors, stale signatures). That check needs no trained models
at all — every source_fn used anywhere in this module is exercised by
building the parts of the graph that only need cheap, deterministic inputs:
literature ingestion (corroborates edges) and dataset/chemistry/mechanism
edges for a small set of representative cells using synthetic cycle data
generated on the fly (no NASA/Severson download required), plus a minimal
stand-in bundle so the contradicts-edge path is exercised too if the
sample cells happen to trigger it.

Run locally:  python scripts/audit_knowledge_graph_provenance.py
"""

import sys as _sys
import os as _os
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
import _paths  # noqa: F401

import numpy as np
import pandas as pd


def _make_cycles_df(n_cycles: int, fade_per_cycle: float, resistance_rise_per_cycle: float) -> pd.DataFrame:
    """Same shape as tests/conftest.py's make_cycles_df() — duplicated
    (not imported) so this script has no dependency on the tests/ package
    and can run standalone in the same job that runs pytest."""
    cycles = np.arange(1, n_cycles + 1)
    capacity = 2.0 - fade_per_cycle * cycles
    resistance = 0.05 + resistance_rise_per_cycle * cycles
    df = pd.DataFrame({
        "cycle_number": cycles,
        "capacity_ah": capacity,
        "resistance_ohm": resistance,
        "temperature_c": np.full(n_cycles, 25.0),
    })
    df["soh_pct"] = (df["capacity_ah"] / 2.0) * 100.0
    return df


def build_auditable_graph():
    from batlab.features.engineering import build_features
    import knowledge_graph as kg

    g = kg.new_graph()
    kg.ingest_literature(g)

    # A couple of representative synthetic cell ids (chemistry_profiles.py's
    # for_cell() recognizes "Cell*"/"OxBat*" prefixes as synthetic — no real
    # dataset file needed) with different fade/resistance profiles so both
    # LLI-leaning and LAM-leaning verdicts get exercised.
    cells = {
        "Cell1": _make_cycles_df(220, fade_per_cycle=0.0005, resistance_rise_per_cycle=0.00003),
        "Cell2": _make_cycles_df(220, fade_per_cycle=0.0016, resistance_rise_per_cycle=0.00035),
    }
    featured_dfs = {cid: build_features(df, cell_id=cid) for cid, df in cells.items()}

    # A minimal stand-in bundle (no real trained model) — only its metrics
    # dict is read by populate_reference_fleet()'s contradicts-edge check,
    # so this is enough to exercise that source_fn too without training
    # anything.
    fake_bundle = {"metrics": {"per_cell_rul_reliable": {}, "rul_reliable": False}}
    kg.populate_reference_fleet(g, featured_dfs, bundles={"synth": fake_bundle})

    return g


def main() -> int:
    import knowledge_graph as kg

    g = build_auditable_graph()
    violations = kg.provenance_audit(g)

    print(f"Audited {g.number_of_edges()} edges across {g.number_of_nodes()} nodes.")
    if not violations:
        print("Provenance audit passed — every edge has a traceable source_fn or doi.")
        return 0

    print(f"Provenance audit FAILED — {len(violations)} violation(s):")
    for v in violations:
        u, dst, edge_type = v["edge"]
        print(f"  [{edge_type}] {u} -> {dst}: {v['reason']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
