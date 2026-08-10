"""Unit tests for src/report_pdf.py's build_bankability_pdf()."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from bankability_report import build_bankability_report
from report_pdf import build_bankability_pdf


def _sample_report():
    return build_bankability_report(
        cell_id="B0005", source="nasa", chemistry="LiCoO2", soh=80.0,
        fade_30_mah_cy=0.001, fleet_fade_median=0.0009, cycle_count=100,
        n_lco_cells=4, lco_soh_r2=0.8, rul_reliable=True,
        rul_q10=150.0, rul_pred=200.0, rul_q90=250.0,
        mechanism={"verdict": "LLI", "confidence": "Medium"},
        experiment_run_id="run-123", git_commit="abc1234",
    )


def test_returns_valid_pdf_bytes():
    pdf_bytes = build_bankability_pdf(_sample_report())
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_handles_missing_optional_fields_gracefully():
    """Missing mechanism/RUL/provenance (build_bankability_report's honest-
    unavailable paths) still yield a valid, non-empty PDF -- not a crash."""
    report = build_bankability_report(
        cell_id="B0006", source="nasa", chemistry="LiCoO2", soh=60.0,
        fade_30_mah_cy=0.002, fleet_fade_median=None, cycle_count=50,
        n_lco_cells=None, lco_soh_r2=None, rul_reliable=False,
        rul_q10=None, rul_pred=None, rul_q90=None,
        mechanism=None,
    )
    pdf_bytes = build_bankability_pdf(report)
    assert pdf_bytes.startswith(b"%PDF")
