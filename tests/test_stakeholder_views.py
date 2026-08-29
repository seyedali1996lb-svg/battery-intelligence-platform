"""Unit tests for src/stakeholder_views.py's 3 stakeholder-specific slices."""

import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from stakeholder_views import build_oem_view, build_operator_view, build_recycler_view, STAKEHOLDER_BUILDERS


def test_stakeholder_builders_registry_has_all_three():
    assert set(STAKEHOLDER_BUILDERS.keys()) == {"oem", "operator", "recycler"}


# ---------------------------------------------------------------------------
# OEM view
# ---------------------------------------------------------------------------

def test_oem_view_includes_mechanism_and_warranty_not_financials():
    fields = build_oem_view(
        "B0005", "LiCoO2", "nasa", 80.0, 100, 0.001,
        mechanism={"verdict": "LLI", "confidence": "Medium"},
        rul_reliable=True, rul_pred=200.0, rul_q10=150.0, rul_q90=250.0,
    )
    labels = [f["label"] for f in fields]
    assert any("Degradation mechanism" in l for l in labels)
    assert any("Warranty risk" in l for l in labels)
    assert not any("reuse value" in l.lower() for l in labels)  # operator-only concept


def test_oem_view_handles_missing_mechanism_gracefully():
    fields = build_oem_view(
        "B0005", "LiCoO2", "nasa", 80.0, 100, 0.001,
        mechanism=None, rul_reliable=False, rul_pred=None, rul_q10=None, rul_q90=None,
    )
    assert not any("Degradation mechanism" in f["label"] for f in fields)


def test_oem_view_flags_already_breached_warranty():
    fields = build_oem_view(
        "B0005", "LiCoO2", "nasa", 60.0, 500, 0.005,
        mechanism=None, rul_reliable=True, rul_pred=50.0, rul_q10=30.0, rul_q90=70.0,
        warranty_floor_soh_pct=70.0,
    )
    warranty_field = next(f for f in fields if "Warranty risk" in f["label"])
    assert warranty_field["value"] == "Already breached"


# ---------------------------------------------------------------------------
# Operator view
# ---------------------------------------------------------------------------

def test_operator_view_includes_action_and_financials_not_mechanism():
    fields = build_operator_view(
        "B0005", "LiCoO2", "nasa", 80.0, 100, 0.001, 0.0009, None,
        rul_reliable=True, rul_pred=200.0, rul_q10=150.0, rul_q90=250.0,
    )
    labels = [f["label"] for f in fields]
    assert any("Recommended action" in l for l in labels)
    assert any("reuse value" in l.lower() for l in labels)
    assert not any("mechanism" in l.lower() for l in labels)  # OEM-only concept


def test_operator_view_unreliable_rul_shown_honestly():
    fields = build_operator_view(
        "B0005", "LiCoO2", "nasa", 80.0, 10, 0.001, 0.0009, None,
        rul_reliable=False, rul_pred=None, rul_q10=None, rul_q90=None,
    )
    rul_field = next(f for f in fields if "Remaining Useful Life" in f["label"])
    assert rul_field["state"] == "unavailable"


# ---------------------------------------------------------------------------
# Recycler view
# ---------------------------------------------------------------------------

def test_recycler_view_includes_r_code_not_financials_or_mechanism():
    fields = build_recycler_view("B0006", "LiCoO2", 58.3, 0.002)
    labels = [f["label"] for f in fields]
    assert any("End-of-life recommendation" in l for l in labels)
    assert not any("reuse value" in l.lower() for l in labels)
    assert not any("mechanism" in l.lower() for l in labels)


def test_recycler_view_shows_recyclers_when_r4_or_r5():
    """A low-SOH cell should land on R4/R5 and get real recycler names."""
    fields = build_recycler_view("B0006", "LiCoO2", 58.3, 0.002)
    recycler_field = next(f for f in fields if "Recommended recyclers" in f["label"])
    assert recycler_field["state"] == "estimated"
    assert recycler_field["value"] != "Not applicable"


def test_recycler_view_not_applicable_when_still_primary_life():
    fields = build_recycler_view("B0005", "LiCoO2", 95.0, 0.0005)
    recycler_field = next(f for f in fields if "Recommended recyclers" in f["label"])
    assert recycler_field["value"] == "Not applicable"
    assert recycler_field["state"] == "unavailable"
