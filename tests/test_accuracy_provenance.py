"""Unit tests for src/accuracy_provenance.py.

The point of this module is that a per-cell accuracy number cannot be read as
a generic claim: its fold count, its own fold R², and its chemistry source
travel with it, and the per-cell reliability gate is never silently replaced
by the dataset average. These tests pin that behaviour down, plus the
never-raises guarantee (a page must not break because provenance resolution
failed).
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import _paths  # noqa: F401
from accuracy_provenance import (
    cell_model_provenance,
    provenance_label,
    per_cell_reliability_detail,
)


def _bundle(**over):
    metrics = {
        "lco_per_cell": {
            "B0005": {"soh_mae": 1.2, "soh_r2": 0.78, "rul_mae": 8.0, "rul_r2": 0.63},
            "B0006": {"soh_mae": 1.6, "soh_r2": 0.71, "rul_mae": 11.0, "rul_r2": 0.46},
            "B0007": {"soh_mae": 1.9, "soh_r2": 0.69, "rul_mae": 13.0, "rul_r2": 0.52},
            "B0018": {"soh_mae": 2.1, "soh_r2": 0.66, "rul_mae": 15.0, "rul_r2": 0.92},
        },
        "per_cell_rul_reliable": {
            "B0005": True, "B0006": True, "B0007": True, "B0018": True,
        },
        "baseline_lco_per_cell": {
            "B0005": {"baseline_soh_r2": 0.61},
        },
        "rul_reliable": True,
    }
    metrics.update(over)
    return {"metrics": metrics}


def test_resolves_this_cells_own_fold_and_the_population_size():
    prov = cell_model_provenance(_bundle(), "B0006")
    assert prov["has_lco"] is True
    assert prov["n_cells"] == 4                     # the held-out population
    assert prov["fold_rul_r2"] == 0.46              # this cell's own fold, not an average
    assert prov["has_own_fold"] is True


def test_fold_without_observed_labels_is_named_not_silent():
    """Tier-0 (v12): rul_r2=None means the fold's RUL labels are all formula
    extrapolations. The label must SAY that — never render as a number, and
    never silently drop the R² token (which would read as a formatting bug
    rather than a statement about the data)."""
    bundle = _bundle()
    bundle["metrics"]["lco_per_cell"]["B0005"] = {
        "soh_mae": 1.2, "soh_r2": 0.78, "rul_mae": None, "rul_r2": None,
    }
    bundle["metrics"]["rul_label_coverage"] = 0.0
    prov = cell_model_provenance(bundle, "B0005")
    assert prov["has_own_fold"] is True
    assert prov["fold_rul_r2"] is None
    label = provenance_label(prov)
    assert "RUL not evaluable" in label
    detail = per_cell_reliability_detail(prov, floor=0.3)
    assert "no observed-EOL rows" in detail and "withheld" in detail


def test_baseline_is_this_cells_fold_not_the_dataset_mean():
    assert cell_model_provenance(_bundle(), "B0005")["baseline_fold_r2"] == 0.61
    # B0006 has no baseline fold of its own -> None, never the B0005 value.
    assert cell_model_provenance(_bundle(), "B0006")["baseline_fold_r2"] is None


def test_chemistry_and_source_resolve_per_cell():
    prov = cell_model_provenance(_bundle(), "B0006")
    assert prov["fold_soh_r2"] == 0.71
    assert prov["fold_rul_mae"] == 11.0
    assert prov["chemistry"] == "LiCoO2"
    assert prov["source"] == "NASA"


def test_baseline_is_this_cells_fold_not_the_dataset_mean():
    assert cell_model_provenance(_bundle(), "B0005")["baseline_fold_r2"] == 0.61
    # B0006 has no baseline fold of its own -> None, never the B0005 value.
    assert cell_model_provenance(_bundle(), "B0006")["baseline_fold_r2"] is None


def test_per_cell_reliability_beats_the_dataset_average():
    """The exact bug class the per-cell gate exists for: the dataset average
    says reliable, but this cell's own fold says otherwise."""
    bundle = _bundle(per_cell_rul_reliable={
        "B0005": True, "B0006": True, "B0007": True, "B0018": False,
    }, rul_reliable=True)
    assert cell_model_provenance(bundle, "B0018")["rul_reliable"] is False
    assert cell_model_provenance(bundle, "B0005")["rul_reliable"] is True


def test_dataset_average_is_only_a_fallback_for_a_cell_absent_from_the_map():
    bundle = _bundle(per_cell_rul_reliable={}, rul_reliable=True)
    assert cell_model_provenance(bundle, "B0005")["rul_reliable"] is True


def test_none_and_malformed_bundles_never_raise():
    for bundle in (None, {}, {"metrics": None}, {"metrics": {"lco_per_cell": None}}, "nonsense"):
        prov = cell_model_provenance(bundle, "B0005")  # type: ignore[arg-type]
        assert prov["has_lco"] is False
        assert prov["n_cells"] is None
        assert prov["fold_rul_r2"] is None
        assert provenance_label(prov) == ""            # no implied validation


def test_label_carries_population_fold_and_chemistry():
    prov = cell_model_provenance(_bundle(), "B0005")
    assert provenance_label(prov) == "n=4 · R²=0.63 · NASA LiCoO2"
    assert provenance_label(prov, include_baseline=True) == (
        "n=4 · R²=0.63 · baseline R²=0.61 · NASA LiCoO2"
    )
    assert provenance_label(prov, include_chemistry=False) == "n=4 · R²=0.63"


def test_label_for_a_severson_cell_reports_lfp():
    bundle = {
        "metrics": {
            "lco_per_cell": {"S-b1c2": {"rul_r2": 0.99}},
            "per_cell_rul_reliable": {"S-b1c2": True},
        }
    }
    prov = cell_model_provenance(bundle, "S-b1c2")
    assert prov["chemistry"] == "LFP"
    assert provenance_label(prov) == "n=1 · R²=0.99 · Severson LFP"


def test_reliability_detail_states_the_thin_population():
    prov = cell_model_provenance(_bundle(), "B0005")
    detail = per_cell_reliability_detail(prov, 0.3)
    assert "4 held-out cell(s)" in detail
    assert "0.63" in detail
    assert "0.30" in detail
    assert "thin population" in detail


def test_reliability_detail_handles_no_population_and_missing_fold():
    assert "No leave-cell-out population" in per_cell_reliability_detail(
        cell_model_provenance(None, "B0005"), 0.3
    )
    bundle = {"metrics": {"lco_per_cell": {"S-other": {"rul_r2": 0.5}}}}
    detail = per_cell_reliability_detail(cell_model_provenance(bundle, "B0005"), 0.3)
    assert "no fold" in detail
