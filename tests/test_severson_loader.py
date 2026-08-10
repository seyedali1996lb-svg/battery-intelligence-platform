"""Unit tests for batlab.datasets.severson — protocol-condition attrs.

CSVs for all 12 Severson cells are already committed at
data/raw/severson/*.csv, so load_severson_cells() hits its cache path and
never attempts a network download here — safe and fast to call directly.
"""

from batlab.datasets.severson import load_severson_cells
from batlab.datasets.schema import condition_completeness, validate_schema


def test_severson_cells_carry_confirmed_conditions_and_omit_unknown_charge_cutoff():
    cells = load_severson_cells()
    assert len(cells) == 12

    for cell_id, df in cells.items():
        assert df.attrs["voltage_discharge_cutoff_v"] == 2.0
        assert df.attrs["test_temperature_c"] == 30.0
        # Charge cutoff deliberately NOT set — 72 distinct multi-step
        # SOC-based fast-charge policies, no single voltage applies.
        assert "voltage_charge_cutoff_v" not in df.attrs
        validate_schema(df, kind="cycle")


def test_severson_condition_completeness_surfaces_the_charge_cutoff_caveat():
    cells = load_severson_cells()
    df = next(iter(cells.values()))
    result = condition_completeness(df)
    assert result["known"]["voltage_charge_cutoff_v"] is False
    assert result["score"] < 1.0
    assert any("multi-step" in c for c in result["caveats"])
