"""Unit tests for batlab.datasets.calce — CALCE CS2 loader.

No real CALCE data is committed here. tests/fixtures/calce/CS2_TEST/
contains two small synthetic workbooks (~20 rows each) mimicking the raw
Arbin BT2000 export shape (Cycle_Index, Discharge_Capacity(Ah),
Charge_Capacity(Ah), Aux_Temperature(C)_1) across two test sessions for
one synthetic cell, to exercise both the per-workbook parsing and the
cross-file cycle-renumbering logic without any real downloaded data.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from batlab.datasets.calce import (
    CalceDataNotFoundError,
    _cycle_summary_from_raw,
    load_calce_cells,
)
from batlab.datasets.schema import validate_schema

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "calce"


def test_cycle_summary_from_raw_reduces_to_one_row_per_cycle():
    df = pd.DataFrame({
        "Cycle_Index": [1, 1, 1, 2, 2, 2],
        "Discharge_Capacity(Ah)": [0.0, 0.5, 1.1, 0.0, 0.5, 1.09],
        "Charge_Capacity(Ah)": [0.0, 0.51, 1.11, 0.0, 0.51, 1.10],
    })
    out = _cycle_summary_from_raw(df)
    assert list(out["cycle_number"]) == [1, 2]
    assert out["capacity_ah"].iloc[0] == pytest.approx(1.1)
    assert out["capacity_ah"].iloc[1] == pytest.approx(1.09)
    # Coulombic efficiency ~ discharge / charge, close to but under 1.0
    assert 0.9 < out["coulombic_efficiency"].iloc[0] < 1.0


def test_cycle_summary_from_raw_picks_up_temperature_by_hint():
    df = pd.DataFrame({
        "Cycle_Index": [1, 1],
        "Discharge_Capacity(Ah)": [0.0, 1.0],
        "Aux_Temperature(C)_1": [24.8, 25.2],
    })
    out = _cycle_summary_from_raw(df)
    assert "temperature_c" in out.columns
    assert out["temperature_c"].iloc[0] == pytest.approx(25.0)


def test_cycle_summary_from_raw_missing_required_column_raises_instructive_error():
    df = pd.DataFrame({"Cycle_Index": [1, 2], "Voltage(V)": [3.7, 3.6]})
    with pytest.raises(CalceDataNotFoundError, match="Discharge_Capacity"):
        _cycle_summary_from_raw(df)


def test_load_calce_cells_no_local_data_raises_instructive_error(tmp_path):
    with pytest.raises(CalceDataNotFoundError, match="calce.umd.edu"):
        load_calce_cells(data_dir=tmp_path)


def test_load_calce_cells_reads_fixture_and_renumbers_across_sessions():
    cells = load_calce_cells(cell_ids=["CS2_TEST"], data_dir=FIXTURE_DIR)
    assert set(cells) == {"CS2_TEST"}

    df = cells["CS2_TEST"]
    # Two sessions x 2 cycles each = 4 cycles total, renumbered 1-4 (not
    # reset to 1-2 twice, which is the real Arbin/CALCE quirk this loader
    # exists specifically to handle).
    assert list(df["cycle_number"]) == [1, 2, 3, 4]
    assert df["cycle_number"].is_monotonic_increasing

    # Standardized schema contract
    validate_schema(df, kind="cycle")
    for attr in ("cell_id", "source", "chemistry", "citation", "license"):
        assert attr in df.attrs
    assert df.attrs["source"] == "calce"
    assert df.attrs["chemistry"] == "LiCoO2"
    assert df.attrs["citation"] == "calce"

    # Capacity fades monotonically across the synthetic fixture's 4 cycles
    assert df["capacity_ah"].iloc[0] > df["capacity_ah"].iloc[-1]
    assert np.isclose(df["soh_pct"].iloc[0], 100.0)


def test_load_calce_cells_missing_cell_id_yields_instructive_error(tmp_path):
    (tmp_path / "CS2_EMPTY").mkdir()
    with pytest.raises(CalceDataNotFoundError, match="no readable"):
        load_calce_cells(cell_ids=["CS2_EMPTY"], data_dir=tmp_path)
