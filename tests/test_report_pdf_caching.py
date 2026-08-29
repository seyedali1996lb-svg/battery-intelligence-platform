"""
Regression tests for app/_pages/compliance.py's _cached_build_report_pdf().

Previously page_reports() called report_pdf.build_report_pdf() unconditionally
on every rerun (any tab switch, checkbox, or slider on the Compliance page) --
rebuilding the full ReportLab document from scratch even when the selected
cell and its data hadn't changed. It was also non-deterministic: document_id()
embeds datetime.now() in its hash, so the "Document ID" shown to the user (and
the "Generated {time}" text inside the PDF body) changed on every rerun even
for byte-identical passport data.

Same "call the page function directly, outside AppTest" pattern as
tests/test_compliance_source_labeling.py -- bare st.* calls outside a script
run context are safe no-ops.
"""

import sys
import pathlib
from unittest.mock import patch

import pandas as pd

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from _pages import compliance
import report_pdf


def _sample_df(last_soh: float = 97.0):
    return pd.DataFrame({
        "soh_pct":        [100, 99, 98, last_soh],
        "cycle_number":   [1, 2, 3, 4],
        "resistance_ohm": [0.05, 0.051, 0.052, 0.053],
        "fade_rate_30cy": [0.001] * 4,
        "capacity_ah":    [2.0, 1.98, 1.96, 1.94],
        "rul_pred":       [None, None, None, 600.0],
    })


def test_cached_build_report_pdf_returns_stable_doc_id_and_bytes():
    """The actual bug: repeat calls with identical content used to produce a
    different Document ID and PDF every time."""
    compliance._cached_build_report_pdf.clear()
    from passport import build_passport

    bundle = {"metrics": {"lco_soh_r2": 0.9, "per_cell_rul_reliable": {"B0005": True}}}
    df = _sample_df()
    passport = build_passport("B0005", df, bundle, rul_reliable=True)
    from consequences import ASSUMPTIONS
    import hashlib, json

    key = hashlib.sha256(
        json.dumps({"passport": passport, "second_life": None, "assumptions": ASSUMPTIONS},
                    sort_keys=True, default=str).encode()
    ).hexdigest()[:20]

    pdf1, doc1 = compliance._cached_build_report_pdf(key, passport, None, ASSUMPTIONS)
    pdf2, doc2 = compliance._cached_build_report_pdf(key, passport, None, ASSUMPTIONS)
    assert doc1 == doc2, "Document ID must be stable across identical calls"
    assert pdf1 == pdf2, "PDF bytes must be identical across identical calls"


def test_cached_build_report_pdf_differs_for_different_content():
    compliance._cached_build_report_pdf.clear()
    from passport import build_passport
    from consequences import ASSUMPTIONS
    import hashlib, json

    bundle_a = {"metrics": {"lco_soh_r2": 0.9, "per_cell_rul_reliable": {"B0005": True}}}
    bundle_b = {"metrics": {"lco_soh_r2": 0.9, "per_cell_rul_reliable": {"B0006": True}}}
    p_a = build_passport("B0005", _sample_df(last_soh=95.0), bundle_a, rul_reliable=True)
    p_b = build_passport("B0006", _sample_df(last_soh=60.0), bundle_b, rul_reliable=True)
    assert p_a != p_b  # sanity check the two inputs are genuinely different

    def key_of(p):
        return hashlib.sha256(
            json.dumps({"passport": p, "second_life": None, "assumptions": ASSUMPTIONS},
                        sort_keys=True, default=str).encode()
        ).hexdigest()[:20]

    pdf_a, doc_a = compliance._cached_build_report_pdf(key_of(p_a), p_a, None, ASSUMPTIONS)
    pdf_b, doc_b = compliance._cached_build_report_pdf(key_of(p_b), p_b, None, ASSUMPTIONS)
    assert doc_a != doc_b, "different passport content must yield a different Document ID"
    assert pdf_a != pdf_b, "different passport content must yield different PDF bytes"


def test_page_reports_only_calls_real_build_report_pdf_once_for_repeat_renders():
    """End-to-end through page_reports() itself (not just the cached wrapper
    in isolation): rendering the same cell twice must hit report_pdf.build_report_pdf
    exactly once, not twice."""
    compliance._cached_build_report_pdf.clear()
    real_build_report_pdf = report_pdf.build_report_pdf
    calls = []

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_build_report_pdf(*args, **kwargs)

    bundle = {"metrics": {"lco_soh_r2": 0.9, "per_cell_rul_reliable": {"B0005": True}}}
    df = _sample_df()

    with patch("report_pdf.build_report_pdf", side_effect=_spy), \
         patch("passport_export.to_json_ld", return_value={}):
        compliance.page_reports("B0005", df, bundle, rul_reliable=True)
        compliance.page_reports("B0005", df, bundle, rul_reliable=True)

    assert calls == [1], f"expected exactly 1 real PDF build across 2 identical renders, got {len(calls)}"
