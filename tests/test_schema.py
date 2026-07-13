"""Unit tests for batlab.datasets.schema — validate_schema(), compute_soh_pct(), concat_cells().

Previously only exercised indirectly through the dataset loaders' own tests
— these test the schema module's own logic directly.
"""

import numpy as np
import pandas as pd
import pytest

from batlab.datasets.schema import SchemaError, compute_soh_pct, concat_cells, validate_schema


def _cell_df(cell_id: str, source: str, chemistry: str, n: int = 5) -> pd.DataFrame:
    cap = np.linspace(2.0, 1.9, n)
    df = pd.DataFrame({
        "cycle_number": np.arange(1, n + 1),
        "capacity_ah": cap,
        "soh_pct": compute_soh_pct(pd.Series(cap)),
    })
    df.attrs["cell_id"] = cell_id
    df.attrs["source"] = source
    df.attrs["chemistry"] = chemistry
    df.attrs["citation"] = source
    df.attrs["license"] = "test"
    return df


def test_validate_schema_passes_on_well_formed_cycle_df():
    validate_schema(_cell_df("A", "nasa", "LiCoO2"), kind="cycle")  # must not raise


def test_validate_schema_rejects_missing_required_column():
    df = _cell_df("A", "nasa", "LiCoO2").drop(columns=["capacity_ah"])
    with pytest.raises(SchemaError, match="capacity_ah"):
        validate_schema(df, kind="cycle")


def test_validate_schema_rejects_missing_attrs():
    df = _cell_df("A", "nasa", "LiCoO2")
    del df.attrs["citation"]
    with pytest.raises(SchemaError, match="citation"):
        validate_schema(df, kind="cycle")


def test_validate_schema_rejects_non_monotonic_cycle_number():
    df = _cell_df("A", "nasa", "LiCoO2")
    df["cycle_number"] = df["cycle_number"].iloc[::-1].to_numpy()
    with pytest.raises(SchemaError, match="cycle_number"):
        validate_schema(df, kind="cycle")


def test_validate_schema_rejects_implausible_soh():
    df = _cell_df("A", "nasa", "LiCoO2")
    df.loc[0, "soh_pct"] = 500.0
    with pytest.raises(SchemaError, match="soh_pct"):
        validate_schema(df, kind="cycle")


def test_compute_soh_pct_first_row_is_100():
    cap = pd.Series([2.0, 1.9, 1.8])
    soh = compute_soh_pct(cap)
    assert soh.iloc[0] == pytest.approx(100.0)
    assert soh.iloc[-1] == pytest.approx(90.0)


def test_compute_soh_pct_rejects_nonpositive_first_value():
    with pytest.raises(SchemaError, match="First capacity_ah"):
        compute_soh_pct(pd.Series([0.0, 1.9]))


# ---------------------------------------------------------------------------
# concat_cells() — regression coverage for pandas silently dropping .attrs
# on a bare pd.concat() call (verified empirically: pd.concat([...]).attrs
# is always {}, even when every input DataFrame carries real attrs).
# ---------------------------------------------------------------------------

def test_concat_cells_preserves_provenance_as_real_columns():
    cells = {
        "B0005": _cell_df("B0005", "nasa", "LiCoO2", n=3),
        "S-b1c2": _cell_df("S-b1c2", "severson2019", "LFP", n=4),
    }
    out = concat_cells(cells)

    assert len(out) == 3 + 4
    assert set(out["cell_id"]) == {"B0005", "S-b1c2"}
    assert set(out.loc[out["cell_id"] == "B0005", "source"]) == {"nasa"}
    assert set(out.loc[out["cell_id"] == "S-b1c2", "chemistry"]) == {"LFP"}


def test_bare_pd_concat_drops_attrs_demonstrating_why_concat_cells_exists():
    """Not testing batlab code — documents the actual pandas behavior
    concat_cells() exists to work around, so this fails loudly (not
    silently) if a future pandas version changes it."""
    df1 = _cell_df("A", "nasa", "LiCoO2")
    df2 = _cell_df("B", "nasa", "LiCoO2")
    assert pd.concat([df1, df2]).attrs == {}


def test_concat_cells_falls_back_to_dict_key_when_attrs_missing_cell_id():
    df = pd.DataFrame({"cycle_number": [1, 2], "capacity_ah": [2.0, 1.9], "soh_pct": [100.0, 95.0]})
    out = concat_cells({"fallback-id": df})
    assert list(out["cell_id"]) == ["fallback-id", "fallback-id"]
