"""Unit tests for batlab.validation.lco — run_lco()."""

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
        assert set(fold.keys()) == {"soh_mae", "soh_r2", "rul_mae", "rul_r2"}
    assert result["rul_reliable"] == (result["rul_r2"] >= RUL_RELIABLE_FLOOR)
