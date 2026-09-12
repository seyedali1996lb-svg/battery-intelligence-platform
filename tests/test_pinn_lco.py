"""Unit tests for batlab.validation.pinn_lco — the PINN leave-cell-out harness.

The behaviour under test is comparability: the PINN must be evaluated on the
SAME folds, the SAME targets, and the SAME metrics as the GBRT
(batlab.validation.lco.run_lco), so that putting the two numbers side by side
is a fair comparison — including when the PINN loses.
"""

from conftest import make_cycles_df
from batlab.validation.pinn_lco import (
    run_pinn_lco,
    pinn_hyperparams,
    MODEL_KIND,
    LAMBDA_PHYSICS_DEFAULT,
)
from batlab.validation.lco import run_lco, RUL_RELIABLE_FLOOR


def _two_cells():
    # Fast fade so both cells cross EOL in-window → observed RUL labels exist
    # and the headline RUL metrics are populated for both harnesses.
    return {
        "CellA": make_cycles_df(n_cycles=300, fade_per_cycle=0.003),
        "CellB": make_cycles_df(n_cycles=300, fade_per_cycle=0.0035,
                                initial_resistance_ohm=0.06),
    }


def test_single_cell_returns_nan_and_no_folds():
    """LCO needs >=2 cells, exactly like run_lco — same precondition, same
    honest NaN rather than a fabricated number."""
    result = run_pinn_lco({"CellA": make_cycles_df()})
    assert result["soh_r2"] != result["soh_r2"]  # NaN != NaN
    assert result["rul_reliable"] is False
    assert result["per_cell"] == {}
    assert result["model_kind"] == MODEL_KIND


def test_fold_structure_matches_the_gbrt_harness():
    """Same held-out cells, same per-fold keys — so the two models are
    genuinely evaluated on identical folds."""
    cell_data = _two_cells()
    pinn = run_pinn_lco(cell_data)
    gbrt = run_lco(cell_data)

    assert set(pinn["per_cell"].keys()) == set(gbrt["per_cell"].keys())
    assert set(pinn["per_cell"].keys()) == {"CellA", "CellB"}
    for fold in pinn["per_cell"].values():
        assert {"soh_mae", "soh_r2", "rul_mae", "rul_r2"} <= set(fold.keys())


def test_metrics_are_finite_and_reliability_flag_is_consistent():
    result = run_pinn_lco(_two_cells())
    assert result["soh_r2"] == result["soh_r2"]  # not NaN
    assert result["rul_r2"] == result["rul_r2"]
    assert result["rul_reliable"] == (result["rul_r2"] >= RUL_RELIABLE_FLOOR)


def test_pinn_rul_scored_on_observed_rows_only():
    """Same Tier-0 rule as the GBRT harness: with no observed-EOL rows the
    PINN's headline RUL is not evaluable, never a formula-recovery score."""
    slow = {
        f"Cell{i}": make_cycles_df(n_cycles=300, fade_per_cycle=0.0002 + i * 1e-5)
        for i in range(3)
    }
    result = run_pinn_lco(slow)
    assert result["rul_r2"] != result["rul_r2"]  # NaN
    assert result["rul_reliable"] is False
    assert result["rul_label_coverage"] == 0.0


def test_predictions_track_a_clean_synthetic_fade_curve():
    """On a self-consistent fading population the physics fit should at least
    capture the direction and order of magnitude — a smoke test that the
    harness computes a real prediction, not a constant."""
    result = run_pinn_lco({
        "CellA": make_cycles_df(n_cycles=300, fade_per_cycle=0.0005),
        "CellB": make_cycles_df(n_cycles=300, fade_per_cycle=0.0006,
                                initial_resistance_ohm=0.055),
    })
    # A constant predictor would score <= 0 R^2; the fit must beat that.
    assert result["soh_r2"] > 0.0
    assert result["soh_mae"] > 0.0


def test_extrapolation_flags_are_reported_per_cell():
    """The EOL projection can fall outside the observed window; that is
    disclosed per cell rather than silently producing an RUL number."""
    result = run_pinn_lco(_two_cells())
    assert set(result["pinn_extrapolation_flags"].keys()) == {"CellA", "CellB"}
    for flag in result["pinn_extrapolation_flags"].values():
        assert isinstance(flag["eol_reached"], bool)


def test_hyperparams_record_the_physics_loss_weight():
    """The physics-loss weight is part of the experiment record, mirroring
    GBRT_PARAMS, so a replay can reproduce the fit."""
    hp = pinn_hyperparams(0.5)
    assert hp["model_kind"] == MODEL_KIND
    assert hp["lambda_physics"] == 0.5
    assert "degradation_law" in hp

    result = run_pinn_lco(_two_cells(), lambda_physics=0.75)
    assert result["hyperparams"]["lambda_physics"] == 0.75
    assert pinn_hyperparams()["lambda_physics"] == LAMBDA_PHYSICS_DEFAULT


def test_featured_frames_are_reused_without_rebuilding():
    """Passing pre-built feature frames must give the same answer as building
    them internally — the reuse path must not change the number."""
    from batlab.features.engineering import build_features

    cell_data = _two_cells()
    plain = run_pinn_lco(cell_data)
    featured = {cid: build_features(df, cell_id=cid) for cid, df in cell_data.items()}
    reused = run_pinn_lco(cell_data, featured=featured)

    assert reused["soh_r2"] == plain["soh_r2"]
    assert reused["per_cell"].keys() == plain["per_cell"].keys()
