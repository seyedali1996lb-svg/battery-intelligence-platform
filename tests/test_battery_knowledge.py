"""Unit tests for src/battery_knowledge.py's corpus and id-lookup helper,
and a regression check that copilot_retrieval.retrieve() still surfaces the
original (non-IEA) documents correctly after the IEA-sourced entries were
appended for the Solar + Storage Sizing "Industry context" callout — TF-IDF
retrieval is corpus-wide, so a badly-chosen new entry could in principle
crowd out an old one for a query that used to hit it cleanly."""

from battery_knowledge import DOCUMENTS, INDUSTRY_CONTEXT_DOC_IDS, get_document
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


def test_industry_context_doc_ids_resolve_to_real_documents():
    assert INDUSTRY_CONTEXT_DOC_IDS  # non-empty
    for doc_id in INDUSTRY_CONTEXT_DOC_IDS:
        assert get_document(doc_id) is not None


def test_industry_context_docs_cite_a_real_iea_source_inline():
    for doc_id in INDUSTRY_CONTEXT_DOC_IDS:
        text = get_document(doc_id)
        assert "IEA" in text


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
