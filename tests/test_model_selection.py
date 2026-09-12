"""Unit tests for src/model_selection.py — chemistry-keyed model selection.

The behaviour under test is the fix for a real, silent defect class: several
call sites resolved a cell's model with `bundles.get(kind) or
bundles.get("synth")`, so a cell of an unrecognised or missing source could be
scored by a model of a *different chemistry* with nothing in the output saying
so. These tests pin down the three tiers (native -> same-chemistry -> refuse),
that the best model is chosen on genuine skill rather than raw R², and that
per-chemistry accuracy is reported separately.
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import _paths  # noqa: F401
import pytest
from model_selection import (
    chemistry_accuracy_line,
    chemistry_accuracy_report,
    describe_bundle,
    describe_bundles,
    describe_bundle_for_cell,
    per_chemistry_accuracy,
    per_chemistry_accuracy_line,
    select_model_for_cell,
)

# Cell ids that resolve to distinct ChemistryProfile buckets:
NASA_CELL = "B0005"            # LiCoO2 / NASA
SYNTH_CELL = "Cell7"           # LiCoO2 / synthetic
SEVERSON_CELL = "S-b1c2"       # LFP / Severson
UPLOAD_CELL = "MyUploaded1"    # Custom / upload


def _bundle(chemistry, source_kind, *, soh_r2=0.8, baseline=None, rul_r2=0.5,
            rul_reliable=True):
    """A bundle whose cell ids make ChemistryProfile resolve the desired
    chemistry/source, with the metrics keys app/_data.py actually writes.
    The population sizes mirror the real reference fleets (NASA 4, Severson 3,
    synthetic 3, upload 2)."""
    ids = {
        "nasa": ("B0005", "B0006", "B0007", "B0018"),
        "severson": ("S-b1c2", "S-b2c1", "S-b3c0"),
        "synth": ("Cell1", "Cell2", "Cell3"),
        "upload": ("MyUploaded1", "MyUploaded2"),
    }[source_kind]
    per_cell = {cid: {"soh_r2": soh_r2, "rul_r2": rul_r2, "rul_mae": 10.0} for cid in ids}
    return {"metrics": {
        "lco_per_cell": per_cell,
        "per_cell_rul_reliable": {cid: rul_reliable for cid in ids},
        "lco_soh_r2": soh_r2,
        "lco_rul_r2": rul_r2,
        "baseline_soh_r2": baseline,
        "rul_reliable": rul_reliable,
        "n_cells": len(ids),
    }}


# ---------------------------------------------------------------------------
# describe_bundle
# ---------------------------------------------------------------------------

def test_describe_bundle_resolves_chemistry_source_and_real_advantage():
    d = describe_bundle("nasa", _bundle("LiCoO2", "nasa", soh_r2=0.759, baseline=0.603))
    assert d["chemistry"] == "LiCoO2"
    assert d["source"] == "NASA"
    assert d["model_label"] == "NASA LiCoO2"
    assert d["soh_r2"] == 0.759
    assert d["advantage"] == pytest.approx(0.156)
    assert d["n_folds"] == 4


def test_describe_bundle_advantage_is_none_without_a_baseline():
    """A run predating the baseline metric can't be ranked on skill — it must
    not be credited with a fabricated advantage."""
    d = describe_bundle("nasa", _bundle("LiCoO2", "nasa"))
    assert d["baseline_soh_r2"] is None
    assert d["advantage"] is None


def test_describe_bundles_skips_empty_entries():
    bundles = {"nasa": _bundle("LiCoO2", "nasa"), "severson": None}
    assert [d["key"] for d in describe_bundles(bundles)] == ["nasa"]


# ---------------------------------------------------------------------------
# select_model_for_cell — the three tiers
# ---------------------------------------------------------------------------

def test_native_model_wins_when_the_cells_own_source_has_one():
    bundles = {
        "nasa": _bundle("LiCoO2", "nasa", soh_r2=0.75, baseline=0.60),
        "synth": _bundle("LiCoO2", "synth", soh_r2=0.998, baseline=-1.9),
    }
    sel = select_model_for_cell(NASA_CELL, bundles)
    assert sel["found"] is True
    assert sel["key"] == "nasa"
    assert sel["native"] is True
    assert sel["selection"] == "native"
    assert sel["model_label"] == "NASA LiCoO2"


def test_chemistry_match_is_used_when_no_native_model_exists_and_is_flagged():
    """A severson LFP cell with no severson bundle is answered by the synthetic
    fleet ONLY if that fleet were LFP — it isn't, so the real answer here is a
    refusal. The positive case uses a chemistry that genuinely matches."""
    bundles = {"nasa": _bundle("LiCoO2", "nasa", soh_r2=0.75, baseline=0.60)}
    sel = select_model_for_cell(SYNTH_CELL, bundles)
    assert sel["found"] is True
    assert sel["native"] is False                    # a *different* source
    assert sel["selection"] == "chemistry_match"
    assert sel["model_label"] == "NASA LiCoO2"
    assert "NOT a validation on this cell" in sel["reason"]


def test_never_answers_a_cell_with_a_different_chemistry():
    """The whole point: an LFP cell must NOT be scored by a LiCoO2 model."""
    bundles = {"nasa": _bundle("LiCoO2", "nasa"), "synth": _bundle("LiCoO2", "synth")}
    sel = select_model_for_cell(SEVERSON_CELL, bundles)
    assert sel["found"] is False
    assert sel["selection"] == "none"
    assert sel["bundle"] == {}
    assert "withheld" in sel["reason"]
    assert "cross-chemistry" in sel["reason"]


def test_unknown_chemistry_upload_cell_is_refused_rather_than_answered_by_a_reference_model():
    """Regression: this used to be silently answered by whichever bundle came
    first (`bundles.get(kind) or bundles.get("synth")`). An uploaded cell whose
    own bundle isn't loaded must not inherit a reference model of a chemistry
    nobody verified."""
    bundles = {"nasa": _bundle("LiCoO2", "nasa"), "severson": _bundle("LFP", "severson")}
    sel = select_model_for_cell(UPLOAD_CELL, bundles)
    assert sel["found"] is False
    assert sel["selection"] == "none"


def test_uploaded_cell_uses_its_own_bundle_when_present():
    bundles = {
        "nasa": _bundle("LiCoO2", "nasa"),
        "upload": _bundle("Custom", "upload", soh_r2=0.9, baseline=0.1),
    }
    sel = select_model_for_cell(UPLOAD_CELL, bundles)
    assert sel["found"] is True
    assert sel["key"] == "upload"
    assert sel["native"] is True


def test_best_same_chemistry_model_is_chosen_on_genuine_skill_not_raw_r2():
    """The synthetic LiCoO2 fleet scores a higher RAW R² (0.998) but most of
    that is the shape of an aging curve (baseline -1.967 => advantage +2.965;
    actually that is genuine too) — so construct the discriminating case
    directly: model A has the higher raw R², model B the higher advantage."""
    high_raw_low_skill = _bundle("LiCoO2", "nasa", soh_r2=0.95, baseline=0.94)    # +0.01
    lower_raw_more_skill = _bundle("LiCoO2", "synth", soh_r2=0.90, baseline=0.10)  # +0.80

    # Both are chemistry candidates (neither is the cell's own source), so the
    # ranking alone decides: the higher raw R² must NOT win.
    sel = select_model_for_cell(SYNTH_CELL, {"a": high_raw_low_skill, "b": lower_raw_more_skill})
    assert sel["native"] is False
    assert sel["key"] == "b", "the larger genuine advantage must win, not the larger raw R²"
    assert sel["accuracy"]["advantage"] == pytest.approx(0.80)


def test_selection_never_raises_on_junk():
    for bundles in (None, {}, {"x": {}}, {"x": "nonsense"}, {"x": None}):
        sel = select_model_for_cell("B0005", bundles)  # type: ignore[arg-type]
        assert isinstance(sel, dict)
        assert "found" in sel and "reason" in sel


# ---------------------------------------------------------------------------
# describe_bundle_for_cell — pages handed a single bundle
# ---------------------------------------------------------------------------

def test_describe_bundle_for_cell_recognises_a_native_bundle():
    sel = describe_bundle_for_cell(_bundle("LiCoO2", "nasa"), NASA_CELL)
    assert sel["native"] is True
    assert sel["compatible"] is True
    assert sel["selection"] == "native"


def test_describe_bundle_for_cell_flags_an_incompatible_chemistry():
    """Health/Overview receive one bundle from the router; if that bundle's
    chemistry is simply not the cell's, the record must say so instead of
    letting the caller present the number as if it applied."""
    sel = describe_bundle_for_cell(_bundle("LiCoO2", "synth"), SEVERSON_CELL)
    assert sel["compatible"] is False
    assert sel["selection"] == "incompatible"
    assert "must not be presented" in sel["reason"]


def test_describe_bundle_for_cell_flags_a_same_chemistry_stand_in():
    sel = describe_bundle_for_cell(_bundle("LiCoO2", "synth"), NASA_CELL)
    assert sel["compatible"] is True
    assert sel["native"] is False
    assert sel["selection"] == "chemistry_match"
    assert "NOT a validation" in sel["reason"]


# ---------------------------------------------------------------------------
# Per-chemistry accuracy reporting
# ---------------------------------------------------------------------------

def test_chemistry_accuracy_report_groups_by_chemistry_and_picks_best_skill():
    bundles = {
        "nasa":  _bundle("LiCoO2", "nasa", soh_r2=0.759, baseline=0.603),   # +0.156
        "synth": _bundle("LiCoO2", "synth", soh_r2=0.998, baseline=-1.967),  # +2.965
        "severson": _bundle("LFP", "severson", soh_r2=0.986, baseline=-0.758),
    }
    rows = chemistry_accuracy_report(bundles)
    by_chem = {r["chemistry"]: r for r in rows}

    # Two LiCoO2 models collapse into ONE row for that chemistry, with the
    # larger-skill model reported as the one selection would pick.
    assert len(rows) == 2
    assert by_chem["LiCoO2"]["n_models"] == 2
    assert by_chem["LiCoO2"]["selected_key"] == "synth"
    assert by_chem["LiCoO2"]["sources"] == ["NASA", "Synthetic"]
    assert by_chem["LiCoO2"]["advantage"] == pytest.approx(2.965)
    assert by_chem["LFP"]["n_models"] == 1

    # Sorted by genuine advantage: LFP's +1.744 sits between the two... no --
    # only one row per chemistry, so LiCoO2 (+2.965) leads LFP (+1.744).
    assert [r["chemistry"] for r in rows] == ["LiCoO2", "LFP"]


def test_chemistry_accuracy_line_reports_population_and_advantage():
    rows = chemistry_accuracy_report({
        "severson": _bundle("LFP", "severson", soh_r2=0.986, baseline=-0.758),
    })
    line = chemistry_accuracy_line(rows[0])
    assert "LFP" in line
    assert "Severson" in line
    assert "n=" in line
    assert "vs trivial baseline +1.744" in line


def test_chemistry_accuracy_report_empty_for_no_bundles():
    assert chemistry_accuracy_report({}) == []
    assert chemistry_accuracy_report(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# per_chemistry_accuracy() — the registry-backed half
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    """Throwaway SQLite file for the experiment registry, same pattern as
    tests/test_experiment_registry.py."""
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


def _log_run(dataset, chemistry, soh_r2, baseline, n_cells, rul_r2=0.5):
    import experiment_registry as reg
    from batlab.features.engineering import FEATURE_VERSION
    return reg.log_run(
        org_id=reg.PLATFORM_ORG_ID, dataset=dataset, chemistry=chemistry,
        feature_set=["cycle_number"], feature_version=FEATURE_VERSION,
        hyperparams={"random_state": 42}, seed=42,
        cell_ids=[f"C{i}" for i in range(n_cells)], n_rows=n_cells * 100,
        lco_metrics={
            "soh_mae": 1.0, "soh_r2": soh_r2, "baseline_soh_r2": baseline,
            "rul_mae": 20.0, "rul_r2": rul_r2, "rul_reliable": True, "per_cell": {},
        },
    )


def test_per_chemistry_accuracy_aggregates_the_registry_by_chemistry(db):
    _log_run("nasa", "LiCoO2", 0.759, 0.603, 4)
    _log_run("synth", "LiCoO2", 0.998, -1.967, 8)
    _log_run("severson", "LFP", 0.986, -0.758, 12)

    rows = per_chemistry_accuracy()
    by_chem = {r["chemistry"]: r for r in rows}
    assert set(by_chem) == {"LiCoO2", "LFP"}

    assert by_chem["LiCoO2"]["n_models"] == 2
    assert by_chem["LiCoO2"]["n_cells_total"] == 12
    assert by_chem["LiCoO2"]["selected_key"] == "synth"     # larger genuine skill
    assert by_chem["LiCoO2"]["advantage"] == pytest.approx(2.965)
    assert by_chem["LFP"]["n_folds"] == 12

    # The whole point of "reported separately": the LiCoO2 row must NOT be a
    # single blended number over its two models.
    assert by_chem["LiCoO2"]["soh_r2"] == 0.998


def test_per_chemistry_accuracy_can_be_filtered_to_one_chemistry(db):
    _log_run("nasa", "LiCoO2", 0.759, 0.603, 4)
    _log_run("severson", "LFP", 0.986, -0.758, 12)

    rows = per_chemistry_accuracy(chemistry="LFP")
    assert [r["chemistry"] for r in rows] == ["LFP"]

    line = per_chemistry_accuracy_line("LFP")
    assert "LFP" in line and "LiCoO2" not in line


def test_per_chemistry_accuracy_is_empty_on_a_fresh_registry(db):
    """No logged runs yet is not an error and not a fabricated number."""
    assert per_chemistry_accuracy() == []
    assert per_chemistry_accuracy_line() == ""


def test_per_chemistry_accuracy_ignores_runs_without_a_usable_chemistry(db):
    _log_run("nasa_to_severson", "LiCoO2 -> LFP", -34.5, None, 16)
    assert per_chemistry_accuracy() == []
