"""Unit tests for src/import_adapter.py — adapt_upload_to_pipeline()."""

import pandas as pd
from import_adapter import adapt_upload_to_pipeline


def _raw_upload_df(with_temperature=True, missing_temp_cell=None):
    rows = []
    for cell_id in ["CellA", "CellB"]:
        for cy in range(1, 51):
            row = {
                "cell_id": cell_id, "cycle_number": cy,
                "capacity_ah": 2.0 - cy * 0.001, "resistance_ohm": 0.05 + cy * 0.0001,
            }
            if with_temperature:
                row["temperature_c"] = None if (cell_id == missing_temp_cell and cy == 1) else 25.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_adapt_upload_produces_expected_schema():
    battery = adapt_upload_to_pipeline(_raw_upload_df())
    assert battery["source"] == "uploaded"
    assert set(battery["cells"].keys()) == {"CellA", "CellB"}
    for cell in battery["cells"].values():
        df = cell["cycles"]
        for col in ["cycle_number", "capacity_ah", "resistance_ohm", "temperature_c", "soh_pct"]:
            assert col in df.columns


def test_missing_temperature_column_entirely_flags_all_cells():
    battery = adapt_upload_to_pipeline(_raw_upload_df(with_temperature=False))
    assert set(battery["temperature_assumed_cells"]) == {"CellA", "CellB"}
    for cell in battery["cells"].values():
        assert cell["temperature_assumed"] is True
        assert (cell["cycles"]["temperature_c"] == 25.0).all()


def test_partial_missing_temperature_flags_only_that_cell():
    df = _raw_upload_df(missing_temp_cell="CellA")
    battery = adapt_upload_to_pipeline(df)
    assert battery["temperature_assumed_cells"] == ["CellA"]
    assert battery["cells"]["CellA"]["temperature_assumed"] == True
    assert battery["cells"]["CellB"]["temperature_assumed"] == False


def test_cycles_sorted_by_cycle_number():
    df = _raw_upload_df()
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    battery = adapt_upload_to_pipeline(shuffled)
    for cell in battery["cells"].values():
        cycles = cell["cycles"]["cycle_number"].tolist()
        assert cycles == sorted(cycles)
