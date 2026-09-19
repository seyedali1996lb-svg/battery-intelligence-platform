"""Unit tests for batlab.datasets.severson — protocol-condition attrs.

CSVs for all 46 batch-1 Severson records are already committed at
data/raw/severson/*.csv, so load_severson_cells() hits its cache path and
never attempts a network download here — safe and fast to call directly.
"""

from batlab.datasets.severson import load_severson_cells
from batlab.datasets.schema import condition_completeness, validate_schema


def test_severson_cells_carry_confirmed_conditions_and_omit_unknown_charge_cutoff():
    cells = load_severson_cells()
    # All 46 batch-1 records (was 12 representative cells): the 5 cells the
    # original authors exclude (b1c8/10/12/13/22 — never reach 80%) are kept
    # as right-censored observations for the survival readout rather than
    # discarded, and the batch carries no corrupt records (content-verified).
    assert len(cells) == 46

    for cell_id, df in cells.items():
        assert cell_id.startswith("S-b1c")
        assert df.attrs["voltage_discharge_cutoff_v"] == 2.0
        assert df.attrs["test_temperature_c"] == 30.0
        # Charge cutoff deliberately NOT set — 72 distinct multi-step
        # SOC-based fast-charge policies, no single voltage applies.
        assert "voltage_charge_cutoff_v" not in df.attrs
        validate_schema(df, kind="cycle")


def test_severson_naming_is_batch_position_with_legacy_shift_disclosed():
    """The loader names records by 0-based batch-array position; the legacy
    loader used 1-based names, so legacy S-b1c2 (1177 cycles, final SOH
    96.55%) is now S-b1c1 — a one-position shift, disclosed in the module
    docstring. Pinned here so the naming convention cannot drift silently
    again."""
    cells = load_severson_cells()
    legacy_b1c2 = cells["S-b1c1"]
    assert len(legacy_b1c2) == 1177
    assert round(float(legacy_b1c2["soh_pct"].iloc[-1]), 2) == 96.55


def test_severson_condition_completeness_surfaces_the_charge_cutoff_caveat():
    cells = load_severson_cells()
    df = next(iter(cells.values()))
    result = condition_completeness(df)
    assert result["known"]["voltage_charge_cutoff_v"] is False
    assert result["score"] < 1.0
    assert any("multi-step" in c for c in result["caveats"])
