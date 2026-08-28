"""Unit tests for batlab.datasets.zhu2022 — the Zhu 2022 voltage-relaxation loader.

Fixture-based per CONTRIBUTING.md: no real downloaded data is committed;
the fixture CSV in tests/fixtures/zhu2022/ mimics the raw source format
(same column names, a handful of rows) and includes a deliberately
"special" cycle with two discharge runs, mirroring the exact structure that
forced the run-delta derivation in the real data.
"""

import pathlib

import pytest

from batlab.datasets.schema import SchemaError, condition_completeness, validate_schema
from batlab.datasets.zhu2022 import _finalize_cell, derive_cell_summary, load_zhu2022_cells

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "zhu2022" / "CY25-05_1-#1.csv"


def test_derive_cell_summary_uses_largest_discharge_run():
    """Cycle 3 of the fixture has TWO discharge runs (0 -> 0.5 Ah, then
    0.5 -> 2.375 Ah): its derived capacity must be the largest run's delta
    (1.875 Ah), NOT the naive per-cycle maximum (2.375 Ah) — the exact
    special-cycle inflation bug found in the real Zenodo data."""
    summary = derive_cell_summary(FIXTURE)
    assert summary is not None
    caps = dict(zip(summary["cycle_number"], summary["capacity_ah"]))
    assert caps[1] == pytest.approx(2.500, abs=1e-3)
    assert caps[2] == pytest.approx(2.475, abs=1e-3)
    assert caps[3] == pytest.approx(1.875, abs=1e-3)
    assert caps[4] == pytest.approx(2.400, abs=1e-3)
    assert caps[5] == pytest.approx(2.350, abs=1e-3)
    assert caps[11] == pytest.approx(2.050, abs=1e-3)
    # Cycle 12 is charge-only (no discharge run at all) and must be dropped.
    assert 12 not in caps
    # Cycle numbers stay ascending from 1.
    assert list(summary["cycle_number"]) == sorted(summary["cycle_number"])


def test_derived_summary_passes_schema_with_attrs():
    summary = derive_cell_summary(FIXTURE)
    df = _finalize_cell(summary, "CY25-05_1-#1")
    validate_schema(df, kind="cycle")
    assert df.attrs["source"] == "zhu2022"
    assert df.attrs["chemistry"] == "NCM+NCA"
    assert df.attrs["citation"] == "zhu2022"
    assert df.attrs["test_temperature_c"] == 25.0
    assert df["soh_pct"].iloc[0] == pytest.approx(100.0)
    assert df["soh_pct"].max() <= 100.0 + 1e-9


def test_condition_completeness_discloses_unknown_conditions():
    summary = derive_cell_summary(FIXTURE)
    df = _finalize_cell(summary, "CY25-05_1-#1")
    cc = condition_completeness(df)
    assert cc["known"]["test_temperature_c"] is True
    assert cc["known"]["voltage_charge_cutoff_v"] is False
    assert cc["known"]["voltage_discharge_cutoff_v"] is False
    # Undisclosed voltage cutoffs must surface as an honest caveat.
    assert any("voltage" in c or "cutoff" in c for c in cc["caveats"])


def test_load_with_custom_raw_dir_never_downloads(tmp_path):
    """A non-default raw_dir with no summaries must return {} WITHOUT a
    network download (that's how the fixture-based tests stay offline)."""
    assert load_zhu2022_cells(raw_dir=tmp_path) == {}


def test_loader_rejects_nonconformant_summary():
    summary = derive_cell_summary(FIXTURE)
    df = _finalize_cell(summary, "CY25-05_1-#1")
    with pytest.raises(SchemaError):
        validate_schema(df.drop(columns=["soh_pct"]), kind="cycle")
