"""Tests for batlab.validation.holdout — the sealed-holdout manifest.

Sealing makes a holdout *verifiable*: the cells are content-hashed at seal
time, later evaluation verifies byte-identity, and training pools are
mechanically checked to exclude the sealed cells.
"""

import pytest

from conftest import make_cycles_df
from batlab.validation.holdout import (
    seal_holdout,
    load_holdout,
    verify_holdout,
    assert_clean_training_pool,
)


def test_seal_and_verify_roundtrip():
    holdout = {f"H{i}": make_cycles_df(n_cycles=100 + i) for i in range(2)}
    manifest = seal_holdout(holdout, training_cell_ids=["T1", "T2"], label="unit test")
    assert manifest["holdout_cells"]["H0"]["n_cycles"] == 100
    result = verify_holdout(manifest, holdout)
    assert result["intact"]
    assert result["missing"] == [] and result["mutated"] == {}


def test_verify_detects_a_mutated_holdout_cell():
    holdout = {"H1": make_cycles_df(n_cycles=100)}
    manifest = seal_holdout(holdout)
    # Append cycles — the most plausible mutation (someone "extends" the
    # holdout cell's history before evaluating on it).
    tampered = {"H1": make_cycles_df(n_cycles=120)}
    result = verify_holdout(manifest, tampered)
    assert not result["intact"]
    assert "H1" in result["mutated"]


def test_verify_detects_a_missing_holdout_cell():
    holdout = {"H1": make_cycles_df(), "H2": make_cycles_df()}
    manifest = seal_holdout(holdout)
    result = verify_holdout(manifest, {"H1": holdout["H1"]})
    assert not result["intact"]
    assert result["missing"] == ["H2"]


def test_digest_is_deterministic_across_row_order():
    """The digest normalizes by cycle_number, so re-sorting the same data
    must not produce a false mutation alarm."""
    df = make_cycles_df(n_cycles=90)
    manifest = seal_holdout({"H1": df})
    shuffled = df.iloc[::-1].reset_index(drop=True)
    result = verify_holdout(manifest, {"H1": shuffled})
    assert result["intact"]


def test_training_pool_containing_a_sealed_cell_raises():
    holdout = {"H1": make_cycles_df()}
    manifest = seal_holdout(holdout, training_cell_ids=["T1"])
    with pytest.raises(ValueError, match="SEALED-HOLDOUT VIOLATION"):
        assert_clean_training_pool(manifest, {"H1": make_cycles_df(), "T1": make_cycles_df()})


def test_clean_training_pool_passes():
    holdout = {"H1": make_cycles_df()}
    manifest = seal_holdout(holdout, training_cell_ids=["T1"])
    assert_clean_training_pool(manifest, {"T1": make_cycles_df(), "T2": make_cycles_df()})


def test_cannot_seal_a_cell_that_is_also_trainable():
    with pytest.raises(ValueError, match="BOTH"):
        seal_holdout({"H1": make_cycles_df()}, training_cell_ids=["H1"])


def test_seal_rejects_an_empty_holdout():
    with pytest.raises(ValueError, match="empty"):
        seal_holdout({})


def test_manifest_roundtrips_through_disk(tmp_path):
    holdout = {"H1": make_cycles_df()}
    path = tmp_path / "holdout.json"
    seal_holdout(holdout, path=path, label="final eval")
    loaded = load_holdout(path)
    assert loaded["label"] == "final eval"
    assert loaded["feature_version"]
    assert verify_holdout(loaded, holdout)["intact"]


def test_manifest_records_feature_version():
    from batlab.features.engineering import FEATURE_VERSION
    manifest = seal_holdout({"H1": make_cycles_df()})
    assert manifest["feature_version"] == FEATURE_VERSION
