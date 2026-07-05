"""
TF-IDF/BM25-style retrieval over the battery-knowledge corpus
(src/battery_knowledge.py) for the Copilot's free-text query path.

Deliberately NOT neural embeddings: no new paid API key, no large model
download (sentence-transformers + torch would be 500MB-1GB+, a real risk
for this app's Streamlit Cloud free-tier deploy, which has already OOM-
crashed once before on a much smaller download — see README's debugging
story). TF-IDF is genuine retrieval (not the old if/elif keyword routing)
and works well for this domain precisely because battery terminology is
exact-term-heavy (IEC 62619, LLI, LAM, dQ/dV, SOH, SoP) rather than
conversational — TF-IDF's weakness (no semantic understanding of
paraphrase) matters less here than it would for open-domain Q&A.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from battery_knowledge import DOCUMENTS

_vectorizer: TfidfVectorizer | None = None
_doc_matrix = None


def _ensure_fitted() -> None:
    """Lazily fit the vectorizer once per process — cheap for ~15 short
    documents, no need for Streamlit's cache_resource machinery here since
    this module has no Streamlit dependency and can be reused standalone."""
    global _vectorizer, _doc_matrix
    if _vectorizer is not None:
        return
    _vectorizer = TfidfVectorizer(stop_words="english")
    _doc_matrix = _vectorizer.fit_transform([d["text"] for d in DOCUMENTS])


def retrieve(query: str, top_k: int = 3, min_score: float = 0.05) -> list[str]:
    """
    Return the top_k most relevant document texts for a free-text query,
    ranked by TF-IDF cosine similarity. Documents scoring below min_score
    are dropped entirely rather than padding the result with irrelevant
    matches — an empty list is a valid, expected return for an off-topic
    query.
    """
    if not query or not query.strip():
        return []
    _ensure_fitted()
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _doc_matrix)[0]
    ranked = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
    return [DOCUMENTS[i]["text"] for i, score in ranked[:top_k] if score >= min_score]
