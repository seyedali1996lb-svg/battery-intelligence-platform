"""Unit tests for batlab.datasets.nasa — protocol-condition attrs.

CSVs for all 4 NASA cells are already committed at data/raw/*.csv, so
load_nasa_cells() hits its cache path and never attempts a network
download here — safe and fast to call directly.
"""

from batlab.datasets.nasa import load_nasa_cells
from batlab.datasets.schema import condition_completeness, validate_schema


def test_nasa_cells_carry_confirmed_protocol_attrs():
    cells = load_nasa_cells()
    assert set(cells) == {"B0005", "B0006", "B0007", "B0018"}

    for cell_id, df in cells.items():
        assert df.attrs["voltage_charge_cutoff_v"] == 4.2
        assert df.attrs["test_temperature_c"] == 24.0
        validate_schema(df, kind="cycle")  # exercises the charge>discharge sanity check


def test_nasa_b0018_has_different_discharge_cutoff_than_the_others():
    cells = load_nasa_cells()
    assert cells["B0018"].attrs["voltage_discharge_cutoff_v"] == 2.5
    for cell_id in ("B0005", "B0006", "B0007"):
        assert cells[cell_id].attrs["voltage_discharge_cutoff_v"] == 2.7


def test_nasa_condition_completeness_is_fully_known():
    cells = load_nasa_cells()
    result = condition_completeness(cells["B0005"])
    assert result["score"] == 1.0
    assert result["caveats"] == []
