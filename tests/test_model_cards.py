"""
Tests for src.model_cards — auto-generated, honest model cards from logged
experiment-registry runs.
"""

import json

import pytest

from src.model_cards import build_model_card, model_card_markdown, dataset_license


def _run(**overrides):
    run = {
        "run_id": "nasa_20260825_000000_abc123",
        "org_id": 0,
        "dataset": "nasa",
        "chemistry": "LiCoO2",
        "feature_set": ["fade_rate_30cy", "sop_pct", "resistance_normalized"],
        "feature_version": "v11-usage-profile",
        "hyperparams": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                        "subsample": 0.8, "random_state": 42},
        "seed": 42,
        "cell_ids": ["B0005", "B0006"],
        "n_cells": 2,
        "n_rows": 336,
        "soh_mae": 1.2, "soh_r2": 0.806,
        "rul_mae": 50.0, "rul_r2": 0.75,
        "rul_reliable": True,
        "fold_metrics": {"B0005": {"soh_r2": 0.9, "rul_r2": 0.7}},
        "git_commit": "abc1234",
        "timestamp": "2026-08-25T12:00:00.000",
        "notes": None,
    }
    run.update(overrides)
    return run


# ---------------------------------------------------------------------------
# dataset_license
# ---------------------------------------------------------------------------

def test_license_resolves_public_dataset():
    info = dataset_license("nasa")
    assert info["license_key"] == "nasa"
    assert "Public" in info["license"] or "government" in info["license"].lower()


def test_license_cross_chemistry_uses_training_domain():
    assert dataset_license("nasa_to_severson")["license_key"] == "nasa"


def test_license_synthetic_disclosed_as_generated():
    info = dataset_license("synth")
    assert info["license_key"] is None
    assert "Internally generated" in info["license"]


def test_license_upload_honest_unknown():
    info = dataset_license("upload")
    assert "Unknown" in info["license"]


# ---------------------------------------------------------------------------
# build_model_card
# ---------------------------------------------------------------------------

def test_card_structure_and_values():
    card = build_model_card(_run())
    assert card["model"]["run_id"] == "nasa_20260825_000000_abc123"
    assert card["dataset"]["license_key"] == "nasa"
    assert card["validation"]["rul_reliable"] is True
    assert card["validation"]["n_folds"] == 1
    assert card["hyperparameters"]["n_estimators"] == 200
    assert card["limitations"]
    # All four honest limitation categories present.
    joined = " ".join(card["limitations"])
    assert "public laboratory datasets" in joined
    assert "fold populations" in joined
    assert "80% SOH" in joined


def test_card_cross_chemistry_flagged_not_lco():
    card = build_model_card(_run(dataset="nasa_to_severson", rul_reliable=False))
    assert any("NOT leave-cell-out" in lim or "not leave-cell-out" in lim for lim in card["limitations"])


def test_card_synthetic_limitation():
    card = build_model_card(_run(dataset="synth"))
    assert any("synthetic fleet" in lim for lim in card["limitations"])


def test_card_reproducibility_hyperparams_match():
    card = build_model_card(_run())
    assert card["reproducibility"]["hyperparams_match_current_gbrt_params"] is True
    assert card["reproducibility"]["hyperparams_diff_vs_current"] == {}


def test_card_reproducibility_surfaces_divergence():
    # A run logged when GBRT_PARAMS had different constants.
    card = build_model_card(_run(hyperparams={"n_estimators": 200, "random_state": 42}))
    assert card["reproducibility"]["hyperparams_match_current_gbrt_params"] is False
    assert card["reproducibility"]["hyperparams_diff_vs_current"]  # non-empty


def test_card_is_json_serializable():
    card = build_model_card(_run())
    payload = json.dumps(card, indent=2, default=str)
    assert json.loads(payload)["dataset"]["key"] == "nasa"


# ---------------------------------------------------------------------------
# model_card_markdown
# ---------------------------------------------------------------------------

def test_markdown_renders_sections_and_metrics():
    md = model_card_markdown(build_model_card(_run()))
    for section in ("### Model card", "#### Data", "#### Validation (Leave-Cell-Out)",
                    "#### Hyperparameters", "#### Reproducibility", "#### Limitations"):
        assert section in md
    assert "R²: 0.806" in md
    assert "**SOH** — MAE: 1.200%" in md


def test_markdown_notes_included():
    md = model_card_markdown(build_model_card(_run(notes="custom note here")))
    assert "custom note here" in md
