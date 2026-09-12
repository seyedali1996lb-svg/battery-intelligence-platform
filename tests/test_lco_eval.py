"""Unit tests for batlab.validation.lco — run_lco() and the Tier-0 RUL
label-provenance semantics (FEATURE_VERSION v12)."""

from conftest import make_cycles_df
from batlab.validation.lco import run_lco, RUL_RELIABLE_FLOOR


def test_run_lco_returns_nan_for_single_cell():
    """LCO needs >=2 cells (train on N-1, test on the held-out one)."""
    result = run_lco({"CellA": make_cycles_df()})
    assert result["soh_r2"] != result["soh_r2"]  # NaN != NaN
    assert result["rul_reliable"] is False
    assert result["per_cell"] == {}


def test_run_lco_two_cells_structure():
    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008, initial_resistance_ohm=0.06),
    }
    result = run_lco(cell_data)
    assert set(result["per_cell"].keys()) == {"CellA", "CellB"}
    for fold in result["per_cell"].values():
        assert set(fold.keys()) == {
            "soh_mae", "soh_r2", "rul_mae", "rul_r2",
            "rul_mae_extrapolated", "rul_r2_extrapolated", "rul_label_kinds",
        }
    assert result["rul_reliable"] == (result["rul_r2"] >= RUL_RELIABLE_FLOOR)


def test_run_lco_reports_label_kinds():
    """Every fold must record how many of its RUL labels are observed vs
    formula-extrapolated — the denominator the headline RUL R² rests on."""
    cell_data = {
        "CellA": make_cycles_df(n_cycles=200, fade_per_cycle=0.0006),
        "CellB": make_cycles_df(n_cycles=200, fade_per_cycle=0.0008, initial_resistance_ohm=0.06),
    }
    result = run_lco(cell_data)
    assert set(result["per_cell"].keys()) == {"CellA", "CellB"}
    for fold in result["per_cell"].values():
        kinds = fold["rul_label_kinds"]
        assert kinds["observed"] + kinds["extrapolated"] + kinds["unknown"] > 0
    assert 0.0 <= result["rul_label_coverage"] <= 1.0


def test_extrapolated_labels_never_count_toward_reliability():
    """The decisive Tier-0 regression test: a fleet where NO cell reaches EOL
    in-window has a 100% formula-generated RUL target. run_lco() must report
    rul_r2 as not-evaluable and rul_reliable=False — even though the old
    mixed-pool metric could score ≈1.0 there by recovering the formula."""
    # Long, slow fade: capacity never reaches 80% of initial within the window.
    cell_data = {
        f"Cell{i}": make_cycles_df(n_cycles=300, fade_per_cycle=0.0002 + i * 1e-5)
        for i in range(3)
    }
    result = run_lco(cell_data)
    assert result["rul_r2"] != result["rul_r2"]  # NaN — nothing observed to score
    assert result["rul_reliable"] is False
    assert result["rul_label_coverage"] == 0.0
    assert result["n_rul_observed_rows"] == 0
    # The formula-recovery diagnostic pool is reported separately:
    for fold in result["per_cell"].values():
        assert fold["rul_r2"] is None
        assert fold["rul_r2_extrapolated"] is not None or fold["rul_mae_extrapolated"] is None


def test_observed_eol_fleet_scores_honestly():
    """A fleet whose cells DO reach EOL in-window: RUL labels are measured,
    the headline RUL number is populated, and coverage is 100%."""
    # Fast fade: crosses 80% SOH inside the window (2.0 Ah initial, 0.0006/cycle
    # → 80% at cycle ~667... use a faster fade so it crosses within 300 cycles).
    cell_data = {
        f"Cell{i}": make_cycles_df(n_cycles=300, fade_per_cycle=0.003 + i * 0.0005)
        for i in range(3)
    }
    result = run_lco(cell_data)
    assert result["n_rul_observed_rows"] > 0
    assert result["rul_label_coverage"] == 1.0
    assert result["rul_r2"] == result["rul_r2"]  # finite
    for fold in result["per_cell"].values():
        assert fold["rul_r2"] is not None
