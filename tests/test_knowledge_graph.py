"""
Unit tests for src/knowledge_graph.py — the Battery Digital Knowledge Graph.

Same isolated-SQLite-file fixture pattern as tests/test_db.py /
tests/test_experiment_registry.py for the persistence tests. Graph-building
tests use tests/conftest.py's make_cycles_df() + batlab's real
build_features()/diagnose_mechanism() — no mocking of the domain logic
itself, since the whole point of this module is to wrap real, already-
tested functions with provenance, not to reimplement them.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest

import knowledge_graph as kg
from conftest import make_cycles_df
from batlab.features.engineering import build_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point src/db.py at a throwaway SQLite file for the duration of one test."""
    import db as db_module

    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    db_module.init_db()
    return db_module


@pytest.fixture
def featured_dfs():
    """Two synthetic cells with deliberately different fade/resistance
    profiles — one mild (should not trigger a strong LAM verdict), one
    aggressive (fast fade + fast resistance rise, should trigger LAM)."""
    mild = build_features(make_cycles_df(n_cycles=220, fade_per_cycle=0.0004, resistance_rise_per_cycle=0.00002))
    aggressive = build_features(make_cycles_df(n_cycles=220, fade_per_cycle=0.0018, resistance_rise_per_cycle=0.0004))
    return {"Cell1": mild, "Cell2": aggressive}


# ---------------------------------------------------------------------------
# Schema / construction primitives
# ---------------------------------------------------------------------------

def test_new_graph_seeds_four_mechanism_nodes():
    g = kg.new_graph()
    assert g.number_of_nodes() == 4
    for key in kg.MECHANISM_KEYS:
        assert kg.node_key(kg.NODE_MECHANISM, key) in g.nodes


def test_add_edge_refuses_edge_without_provenance():
    g = kg.new_graph()
    kg.add_cell(g, "CellX")
    with pytest.raises(ValueError, match="source_fn"):
        kg.add_edge(g, kg.NODE_CELL, "CellX", kg.NODE_MECHANISM, "lli", kg.EDGE_EXHIBITS)


def test_add_edge_requires_both_endpoints_to_exist():
    g = kg.new_graph()
    with pytest.raises(ValueError, match="does not exist"):
        kg.add_edge(g, kg.NODE_CELL, "Ghost", kg.NODE_MECHANISM, "lli", kg.EDGE_EXHIBITS, source_fn="test.fn")


def test_add_edge_rejects_unknown_edge_type():
    g = kg.new_graph()
    kg.add_cell(g, "CellX")
    with pytest.raises(ValueError, match="Unknown edge type"):
        kg.add_edge(g, kg.NODE_CELL, "CellX", kg.NODE_MECHANISM, "lli", "invented_type", source_fn="test.fn")


def test_add_edge_accepts_doi_only_provenance():
    g = kg.new_graph()
    kg.add_literature(g, "doc1")
    kg.add_edge(g, kg.NODE_MECHANISM, "lli", kg.NODE_LITERATURE, "doc1", kg.EDGE_CORROBORATES,
                doi="10.1016/j.jpowsour.2016.12.011")
    assert g.number_of_edges() == 1


def test_repeated_add_edge_is_idempotent_not_duplicating():
    g = kg.new_graph()
    kg.add_cell(g, "CellX")
    kg.add_edge(g, kg.NODE_CELL, "CellX", kg.NODE_MECHANISM, "lli", kg.EDGE_EXHIBITS, source_fn="test.fn")
    kg.add_edge(g, kg.NODE_CELL, "CellX", kg.NODE_MECHANISM, "lli", kg.EDGE_EXHIBITS, source_fn="test.fn", extra=1)
    assert g.number_of_edges() == 1
    data = kg.mechanism_edge(g, "CellX")
    assert data["extra"] == 1  # second call updated the same edge, didn't add a parallel one


# ---------------------------------------------------------------------------
# Cell -> dataset / chemistry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell_id,expected", [
    ("B0005", "nasa"), ("S-12", "severson"), ("OX-3", "oxford"),
    ("Cell1", "synth"), ("UploadedCellA", "uploaded"),
])
def test_dataset_key_for_cell(cell_id, expected):
    assert kg.dataset_key_for_cell(cell_id) == expected


def test_link_cell_to_dataset_and_chemistry():
    g = kg.new_graph()
    kg.add_cell(g, "B0005")
    dataset_key = kg.link_cell_to_dataset(g, "B0005")
    chem_key = kg.link_cell_to_chemistry(g, "B0005")
    assert dataset_key == "nasa"
    assert chem_key == "LiCoO2"
    assert kg.node_key(kg.NODE_DATASET, "nasa") in g.nodes
    assert kg.node_key(kg.NODE_CHEMISTRY, "LiCoO2") in g.nodes


# ---------------------------------------------------------------------------
# Cell -> mechanism (the shared exhibits edge)
# ---------------------------------------------------------------------------

def test_link_cell_to_mechanism_stores_full_verdict(featured_dfs):
    g = kg.new_graph()
    mech = kg.link_cell_to_mechanism(g, "Cell2", featured_dfs["Cell2"])
    assert mech["verdict"] in kg.MECHANISM_LABELS.values()
    assert mech["source_fn"] == "recommendations.diagnose_mechanism"
    assert "verdict_color" in mech and "verdict_icon" in mech
    assert isinstance(mech["signals"], dict)
    for v in mech["signals"].values():
        assert isinstance(v, float)  # not numpy.float64 — must survive JSON round-trip


def test_get_or_compute_mechanism_reuses_existing_edge_not_recomputing(featured_dfs, monkeypatch):
    g = kg.new_graph()
    first = kg.get_or_compute_mechanism(g, "Cell2", featured_dfs["Cell2"])
    assert first is not None

    calls = {"n": 0}
    import recommendations

    def _boom(df):
        calls["n"] += 1
        raise AssertionError("diagnose_mechanism() should not be called again for an already-graphed cell")

    monkeypatch.setattr(recommendations, "diagnose_mechanism", _boom)
    second = kg.get_or_compute_mechanism(g, "Cell2", featured_dfs["Cell2"])
    assert second == first
    assert calls["n"] == 0


def test_get_or_compute_mechanism_computes_for_new_cell(featured_dfs):
    g = kg.new_graph()
    assert kg.mechanism_edge(g, "Cell1") is None
    result = kg.get_or_compute_mechanism(g, "Cell1", featured_dfs["Cell1"])
    assert result is not None
    assert kg.mechanism_edge(g, "Cell1") == result


def test_mechanism_edge_none_for_unknown_cell():
    g = kg.new_graph()
    assert kg.mechanism_edge(g, "NoSuchCell") is None


# ---------------------------------------------------------------------------
# Contradicts — action vs mechanism arbitration
# ---------------------------------------------------------------------------

def test_link_cell_action_mechanism_conflict_adds_edge_on_real_disagreement():
    g = kg.new_graph()
    kg.add_cell(g, "CellX")
    lam_mechanism = {
        "verdict": "LAM — Loss of Active Material",
        "confidence_label": "Medium",
    }
    added = kg.link_cell_action_mechanism_conflict(g, "CellX", "continue", lam_mechanism)
    assert added is True
    edges = list(kg._edges_of_type(g, kg.node_key(kg.NODE_CELL, "CellX"), kg.EDGE_CONTRADICTS, "out"))
    assert len(edges) == 1
    assert edges[0][2]["source_fn"] == "recommendations.mechanism_corroboration_note"


def test_link_cell_action_mechanism_conflict_no_edge_when_nothing_disagrees():
    g = kg.new_graph()
    kg.add_cell(g, "CellX")
    healthy_mechanism = {"verdict": "Insufficient data", "confidence_label": "No data"}
    added = kg.link_cell_action_mechanism_conflict(g, "CellX", "continue", healthy_mechanism)
    assert added is False
    assert g.number_of_edges() == 0


# ---------------------------------------------------------------------------
# Literature ingestion + corroboration
# ---------------------------------------------------------------------------

def test_ingest_literature_adds_real_documents_and_corroboration_edges():
    from battery_knowledge import DOCUMENTS

    g = kg.new_graph()
    n = kg.ingest_literature(g)
    assert n == len(DOCUMENTS)
    for doc in DOCUMENTS:
        assert kg.node_key(kg.NODE_LITERATURE, doc["id"]) in g.nodes

    # Every curated mapping entry must point at a real document id — this
    # would catch a typo'd or renamed doc id in MECHANISM_LITERATURE_CORROBORATION.
    doc_ids = {d["id"] for d in DOCUMENTS}
    for mech_key, ids in kg.MECHANISM_LITERATURE_CORROBORATION.items():
        for doc_id in ids:
            assert doc_id in doc_ids, f"{doc_id!r} referenced by mechanism {mech_key!r} is not a real document id"

    lli_lit = kg.literature_for_mechanism(g, "lli")
    assert any(d["id"] == "lli-vs-lam" for d in lli_lit)

    # "insufficient_data" has no corroborating literature by design.
    assert kg.literature_for_mechanism(g, "insufficient_data") == []


def test_literature_for_mechanism_unknown_key_returns_empty():
    g = kg.new_graph()
    kg.ingest_literature(g)
    assert kg.literature_for_mechanism(g, "not_a_real_mechanism") == []


# ---------------------------------------------------------------------------
# populate_reference_fleet / build_platform_graph orchestration
# ---------------------------------------------------------------------------

def test_populate_reference_fleet_adds_every_cell_with_dataset_chemistry_mechanism(featured_dfs):
    g = kg.new_graph()
    kg.populate_reference_fleet(g, featured_dfs)

    for cell_id in featured_dfs:
        assert kg.node_key(kg.NODE_CELL, cell_id) in g.nodes
        assert kg.mechanism_edge(g, cell_id) is not None
        chem = kg._first_neighbor_of_type(g, kg.node_key(kg.NODE_CELL, cell_id), kg.EDGE_DERIVED_FROM, kg.NODE_CHEMISTRY)
        dataset = kg._first_neighbor_of_type(g, kg.node_key(kg.NODE_CELL, cell_id), kg.EDGE_DERIVED_FROM, kg.NODE_DATASET)
        assert chem == "LiCoO2"
        assert dataset == "synth"


def test_build_platform_graph_end_to_end_has_no_provenance_violations(featured_dfs):
    fake_bundle = {"metrics": {"per_cell_rul_reliable": {}, "rul_reliable": False}}
    g = kg.build_platform_graph(featured_dfs, bundles={"synth": fake_bundle}, tenant_org_id=None)
    assert g.number_of_nodes() > 4  # more than just the seeded mechanism nodes
    violations = kg.provenance_audit(g)
    assert violations == []


# ---------------------------------------------------------------------------
# cells_like query
# ---------------------------------------------------------------------------

def test_cells_like_requires_same_chemistry_and_ranks_by_mechanism_then_soh(featured_dfs):
    g = kg.new_graph()
    kg.populate_reference_fleet(g, featured_dfs)
    # An NCA Oxford cell should never appear as a match for a synthetic LiCoO2 cell.
    kg.add_cell(g, "OX-99", soh_pct=90.0, cycle_number=5)
    kg.link_cell_to_dataset(g, "OX-99")
    kg.link_cell_to_chemistry(g, "OX-99")

    matches = kg.cells_like(g, "Cell1", top_k=5)
    match_ids = {m["cell_id"] for m in matches}
    assert "OX-99" not in match_ids
    assert "Cell2" in match_ids


def test_cells_like_unknown_cell_returns_empty():
    g = kg.new_graph()
    assert kg.cells_like(g, "Ghost") == []


# ---------------------------------------------------------------------------
# provenance_audit
# ---------------------------------------------------------------------------

def test_provenance_audit_clean_graph_has_no_violations():
    g = kg.new_graph()
    kg.ingest_literature(g)
    assert kg.provenance_audit(g) == []


def test_provenance_audit_catches_stale_source_fn():
    g = kg.new_graph()
    kg.add_cell(g, "CellX")
    # Bypass add_edge()'s provenance gate directly (simulating a source_fn
    # string that pointed at something real once but no longer resolves —
    # e.g. after a function rename) to verify the audit actually catches it.
    g.add_edge(
        kg.node_key(kg.NODE_CELL, "CellX"), kg.node_key(kg.NODE_MECHANISM, "lli"),
        key=kg.EDGE_EXHIBITS, type=kg.EDGE_EXHIBITS,
        source_fn="recommendations.this_function_was_renamed_away", doi=None,
    )
    violations = kg.provenance_audit(g)
    assert len(violations) == 1
    assert "does not resolve" in violations[0]["reason"]


def test_provenance_audit_catches_malformed_doi():
    g = kg.new_graph()
    kg.add_literature(g, "doc1")
    g.add_edge(
        kg.node_key(kg.NODE_MECHANISM, "lli"), kg.node_key(kg.NODE_LITERATURE, "doc1"),
        key=kg.EDGE_CORROBORATES, type=kg.EDGE_CORROBORATES,
        source_fn=None, doi="not-a-real-doi",
    )
    violations = kg.provenance_audit(g)
    assert len(violations) == 1
    assert "not a well-formed DOI" in violations[0]["reason"]


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_graph_to_rows_and_back_preserves_structure(featured_dfs):
    g = kg.new_graph()
    kg.ingest_literature(g)
    kg.populate_reference_fleet(g, featured_dfs)

    node_rows, edge_rows = kg.graph_to_rows(g)
    g2 = kg.graph_from_rows(node_rows, edge_rows)

    assert g2.number_of_nodes() == g.number_of_nodes()
    assert g2.number_of_edges() == g.number_of_edges()
    assert kg.mechanism_edge(g2, "Cell1")["verdict"] == kg.mechanism_edge(g, "Cell1")["verdict"]


# ---------------------------------------------------------------------------
# Persistence via src/db.py
# ---------------------------------------------------------------------------

def test_save_and_load_graph_round_trip(db, featured_dfs):
    g = kg.new_graph()
    kg.ingest_literature(g)
    kg.populate_reference_fleet(g, featured_dfs)

    kg.save_graph(0, g)
    loaded = kg.load_graph(0)

    assert loaded.number_of_nodes() == g.number_of_nodes()
    assert loaded.number_of_edges() == g.number_of_edges()
    assert kg.mechanism_edge(loaded, "Cell2")["verdict"] == kg.mechanism_edge(g, "Cell2")["verdict"]


def test_save_graph_replaces_previous_snapshot_for_same_org(db, featured_dfs):
    g1 = kg.new_graph()
    kg.populate_reference_fleet(g1, {"Cell1": featured_dfs["Cell1"]})
    kg.save_graph(0, g1)

    g2 = kg.new_graph()
    kg.populate_reference_fleet(g2, featured_dfs)  # both cells this time
    kg.save_graph(0, g2)

    loaded = kg.load_graph(0)
    assert loaded.number_of_nodes() == g2.number_of_nodes()
    assert kg.node_key(kg.NODE_CELL, "Cell2") in loaded.nodes


def test_load_graph_with_no_snapshot_returns_seeded_empty_graph(db):
    loaded = kg.load_graph(999)
    assert loaded.number_of_nodes() == 4  # just the seeded mechanism nodes
    assert loaded.number_of_edges() == 0


def test_org_scoped_graphs_do_not_leak(db, featured_dfs):
    g1 = kg.new_graph()
    kg.populate_reference_fleet(g1, {"Cell1": featured_dfs["Cell1"]})
    kg.save_graph(1, g1)

    g2 = kg.new_graph()
    kg.populate_reference_fleet(g2, {"Cell2": featured_dfs["Cell2"]})
    kg.save_graph(2, g2)

    loaded1 = kg.load_graph(1)
    loaded2 = kg.load_graph(2)
    assert kg.node_key(kg.NODE_CELL, "Cell1") in loaded1.nodes
    assert kg.node_key(kg.NODE_CELL, "Cell1") not in loaded2.nodes
    assert kg.node_key(kg.NODE_CELL, "Cell2") in loaded2.nodes
    assert kg.node_key(kg.NODE_CELL, "Cell2") not in loaded1.nodes


# ---------------------------------------------------------------------------
# populate_experiment_runs
# ---------------------------------------------------------------------------

def test_populate_experiment_runs_links_only_cells_present_in_graph(db, featured_dfs):
    import experiment_registry as reg

    reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset="synth", chemistry="LiCoO2",
        feature_set=["cycle_number"], feature_version="v1",
        hyperparams={"n_estimators": 10}, seed=42,
        cell_ids=["Cell1", "Cell2", "CellNotInGraph"],
        n_rows=100,
        lco_metrics={"soh_mae": 1.0, "soh_r2": 0.8, "rul_mae": 20.0, "rul_r2": 0.5,
                     "rul_reliable": True, "per_cell": {}},
    )

    g = kg.new_graph()
    kg.populate_reference_fleet(g, featured_dfs)  # Cell1, Cell2 only
    n = kg.populate_experiment_runs(g, tenant_org_id=None)

    assert n == 1  # one run visible
    run_node = [n for n, d in g.nodes(data=True) if d.get("type") == kg.NODE_EXPERIMENT_RUN]
    assert len(run_node) == 1
    run_key = run_node[0]

    # Only Cell1/Cell2 (present in the graph) got derived_from edges; the
    # third cell_id on the logged run was never added as a bare node.
    linked_cells = [u for u, v, d in g.in_edges(run_key, data=True) if d.get("type") == kg.EDGE_DERIVED_FROM]
    assert set(linked_cells) == {kg.node_key(kg.NODE_CELL, "Cell1"), kg.node_key(kg.NODE_CELL, "Cell2")}
    assert kg.node_key(kg.NODE_CELL, "CellNotInGraph") not in g.nodes
