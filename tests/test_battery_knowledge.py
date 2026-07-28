"""Unit tests for src/battery_knowledge.py's corpus and id-lookup helper,
and a regression check that copilot_retrieval.retrieve() still surfaces the
original (non-IEA) documents correctly after the IEA-sourced entries were
appended for the Solar + Storage Sizing "Industry context" callout — TF-IDF
retrieval is corpus-wide, so a badly-chosen new entry could in principle
crowd out an old one for a query that used to hit it cleanly."""

from battery_knowledge import (
    DOCUMENTS,
    FEATURE_CITATIONS,
    INDUSTRY_CONTEXT_DOCS_BY_SIGNAL,
    get_document,
    get_feature_citation,
    industry_context_doc_ids,
)
from copilot_retrieval import retrieve


def test_all_documents_have_unique_ids():
    ids = [d["id"] for d in DOCUMENTS]
    assert len(ids) == len(set(ids))


def test_get_document_returns_text_for_known_id():
    text = get_document("iec62619-thermal-runaway")
    assert text is not None
    assert "thermal runaway" in text.lower() or "THERMAL_RUNAWAY" in text


def test_get_document_returns_none_for_unknown_id():
    assert get_document("this-id-does-not-exist") is None


def test_industry_context_signal_docs_resolve_to_real_documents():
    for doc_id in INDUSTRY_CONTEXT_DOCS_BY_SIGNAL.values():
        assert get_document(doc_id) is not None


def test_industry_context_signal_docs_cite_a_real_iea_source_inline():
    for doc_id in INDUSTRY_CONTEXT_DOCS_BY_SIGNAL.values():
        text = get_document(doc_id)
        assert "IEA" in text


def test_industry_context_doc_ids_always_includes_default():
    ids = industry_context_doc_ids(payback_years=5.0, battery_kwh=0.001)
    assert ids == [INDUSTRY_CONTEXT_DOCS_BY_SIGNAL["default"]]


def test_industry_context_doc_ids_long_payback_adds_cost_context():
    ids = industry_context_doc_ids(payback_years=15.0, battery_kwh=0.001)
    assert INDUSTRY_CONTEXT_DOCS_BY_SIGNAL["long_payback"] in ids
    assert len(ids) == 2


def test_industry_context_doc_ids_large_deployment_adds_demand_context():
    ids = industry_context_doc_ids(payback_years=5.0, battery_kwh=1.0)
    assert INDUSTRY_CONTEXT_DOCS_BY_SIGNAL["large_deployment"] in ids
    assert len(ids) == 2


def test_industry_context_doc_ids_none_payback_never_crashes():
    ids = industry_context_doc_ids(payback_years=None, battery_kwh=0.001)
    assert ids == [INDUSTRY_CONTEXT_DOCS_BY_SIGNAL["default"]]


def test_retrieve_still_surfaces_original_thermal_runaway_doc():
    # Pre-existing spot-check query, unrelated to the new IEA entries —
    # regression guard that the corpus append didn't crowd it out.
    results = retrieve("thermal runaway precursor temperature rise", top_k=3)
    assert any("THERMAL_RUNAWAY_PRECURSOR" in r for r in results)


def test_retrieve_still_surfaces_original_lli_lam_doc():
    results = retrieve("loss of lithium inventory versus loss of active material", top_k=3)
    assert any("Loss of Lithium Inventory" in r for r in results)


def test_retrieve_surfaces_new_iea_stationary_storage_doc_for_its_own_topic():
    results = retrieve("second-life battery reuse stationary storage economics", top_k=3)
    assert any("stationary storage" in r.lower() for r in results)


# ---------------------------------------------------------------------------
# FEATURE_CITATIONS — structured citation objects for battery_copilot's
# FEATURE_PHYSICS entries
# ---------------------------------------------------------------------------

def test_feature_citations_covers_every_battery_copilot_physics_feature():
    """FEATURE_CITATIONS must have an entry for every key in
    battery_copilot.FEATURE_PHYSICS (the static prose these citations back)
    -- catches a future FEATURE_PHYSICS addition silently going uncited."""
    from battery_copilot import FEATURE_PHYSICS
    assert set(FEATURE_CITATIONS.keys()) == set(FEATURE_PHYSICS.keys())


def test_every_feature_citation_has_doi_title_and_relevance():
    for feature, citation in FEATURE_CITATIONS.items():
        assert citation.get("doi"), f"{feature} citation missing a doi"
        assert citation.get("title"), f"{feature} citation missing a title"
        assert citation.get("relevance"), f"{feature} citation missing a relevance line"
        # A real DOI always starts with "10." (the DOI registrant prefix) --
        # guards against an empty or placeholder string sneaking past review.
        assert citation["doi"].startswith("10."), f"{feature} doi doesn't look like a real DOI: {citation['doi']!r}"


def test_get_feature_citation_returns_none_for_uncited_feature():
    # stress_index/dod_proxy are derived/dimensionless features, not
    # themselves named diagnostics in either cited paper -- no citation on
    # file for them is the honest answer, not a bug.
    assert get_feature_citation("stress_index") is None
    assert get_feature_citation("this-is-not-a-real-feature") is None


def test_get_feature_citation_returns_the_recorded_object():
    citation = get_feature_citation("resistance_ohm")
    assert citation is not None
    assert citation["doi"] == "10.1016/j.jpowsour.2005.01.006"


def test_answer_prediction_drivers_cites_a_source_for_a_cited_feature():
    """battery_copilot.answer_prediction_drivers() must inline a citation
    line for any top driver that has one on file -- the actual consumer of
    FEATURE_CITATIONS, not just the lookup helper in isolation."""
    from battery_copilot import answer_prediction_drivers

    ctx = {
        "cell_id": "TestCell",
        "top_features": [{"feature": "resistance_ohm", "importance_pct": 42.0}],
        "soh_fold_r2": 0.85,
        "lco_soh_r2": 0.8,
    }
    answer = answer_prediction_drivers(ctx)
    assert "doi:10.1016/j.jpowsour.2005.01.006" in answer
    assert "Vetter" in answer
