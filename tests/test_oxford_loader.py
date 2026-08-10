"""Unit tests for batlab.datasets.oxford — pure-logic pieces only.

No network calls or real .mat parsing here (that needs the ~3.1 GB source
zips across all 4 groups, downloaded once locally per the module docstring)
— these test the pieces that don't require it: the max-Amphr-delta-across-
steps capacity extraction rule, the SOH-relative-to-checkpoint-0
computation, and the group-aware zip filename matching.
"""

import numpy as np
import pandas as pd
import batlab.datasets.oxford as oxford_mod
from batlab.datasets.oxford import (
    _capacity_ah_from_table, _add_soh_column, _find_zip_entry,
    _CELL_GROUP, _PROCEDURE_LABELS, OXFORD_CELL_IDS, _GROUP_C_RATE,
)


def _make_maccor_table(step_specs: list[tuple]) -> pd.DataFrame:
    """Build a synthetic DataFrame mimicking the real Maccor column shape
    (Cyc, Step, Amphr, Amps, ...). step_specs: list of (step_number,
    n_rows, amphr_start, amphr_end) — Amphr ramps linearly within a step,
    matching how the real per-timestamp Amphr column behaves."""
    rows = []
    for step, n, amphr_start, amphr_end in step_specs:
        amphr = np.linspace(amphr_start, amphr_end, n)
        for a in amphr:
            rows.append({"Cyc": 0.0, "Step": float(step), "Amphr": a, "Amps": -1.5, "Volts": 3.7})
    return pd.DataFrame(rows)


def test_capacity_extraction_picks_largest_amphr_swing_step():
    # Step 1: rest (no swing). Step 2: small partial discharge (0.5Ah swing).
    # Step 3: full reference discharge (3.05Ah swing) — this is the one that
    # should be picked, matching the real dataset's pattern where the full
    # discharge step has by far the largest Amphr delta.
    df = _make_maccor_table([
        (1, 50, 0.0, 0.0),
        (2, 50, 0.0, 0.5),
        (3, 200, 0.0, 3.05),
    ])
    cap = _capacity_ah_from_table(df)
    assert abs(cap - 3.05) < 1e-9


def test_capacity_extraction_robust_to_step_numbering_shift():
    """Real dataset quirk: step numbers shifted between files for the same
    cell (BoL had steps up to 41, RPT1 up to 36) — the extraction rule must
    not depend on a hardcoded step number, only on which step has the
    largest swing."""
    df_a = _make_maccor_table([(8, 100, 0.0, 3.1), (1, 20, 0.0, 0.0)])
    df_b = _make_maccor_table([(19, 100, 0.0, 2.9), (1, 20, 0.0, 0.0), (2, 20, 0.0, 0.1)])
    assert abs(_capacity_ah_from_table(df_a) - 3.1) < 1e-9
    assert abs(_capacity_ah_from_table(df_b) - 2.9) < 1e-9


def test_capacity_extraction_all_rest_returns_zero():
    df = _make_maccor_table([(1, 10, 0.0, 0.0), (2, 10, 0.0, 0.0)])
    assert _capacity_ah_from_table(df) == 0.0


def test_soh_relative_to_first_checkpoint():
    df = pd.DataFrame({
        "checkpoint_index": [0, 1, 2, 3],
        "checkpoint_label": ["BoL", "RPT1", "RPT2", "RPT3"],
        "capacity_ah": [3.0, 2.85, 2.7, 2.55],
    })
    out = _add_soh_column(df)
    assert out["soh_pct"].iloc[0] == 100.0
    assert abs(out["soh_pct"].iloc[1] - 95.0) < 1e-9
    assert abs(out["soh_pct"].iloc[3] - 85.0) < 1e-9


def test_soh_column_monotonic_for_monotonic_capacity():
    df = pd.DataFrame({
        "checkpoint_index": [0, 1, 2],
        "checkpoint_label": ["BoL", "RPT1", "RPT2"],
        "capacity_ah": [3.0, 2.9, 2.8],
    })
    out = _add_soh_column(df)
    assert out["soh_pct"].is_monotonic_decreasing


def test_find_zip_entry_matches_correct_group_prefix():
    """Cells in different groups share the same TPG index numbering, so the
    group prefix in the filename regex must actually be used — a Group 3
    filename must not match when searching for a Group 1 entry."""
    names = ["Group 1/TPG1.4-Cell 9.mat", "Group 3/TPG3.4-Cell10.mat"]
    assert _find_zip_entry(names, 1, 9, 4) == "Group 1/TPG1.4-Cell 9.mat"
    assert _find_zip_entry(names, 3, 10, 4) == "Group 3/TPG3.4-Cell10.mat"
    assert _find_zip_entry(names, 1, 10, 4) is None  # right cell, wrong group


def test_find_zip_entry_handles_inconsistent_spacing():
    names = ["TPG2 - Cell 3.mat", "TPG2.5-Cell3.mat", "TPG2.20 - Cell 3.mat"]
    assert _find_zip_entry(names, 2, 3, 1) == "TPG2 - Cell 3.mat"
    assert _find_zip_entry(names, 2, 3, 5) == "TPG2.5-Cell3.mat"
    assert _find_zip_entry(names, 2, 3, 20) == "TPG2.20 - Cell 3.mat"


def test_all_12_cells_across_4_groups_have_procedure_labels():
    assert set(_CELL_GROUP) == set(_PROCEDURE_LABELS)
    assert len(_CELL_GROUP) == 12
    assert sorted(_CELL_GROUP.values()) == sorted([1] * 3 + [2] * 3 + [3] * 3 + [4] * 3)
    assert set(OXFORD_CELL_IDS) == {f"OX-{c}" for c in _CELL_GROUP}


def test_load_cached_backfills_c_rate_for_a_legacy_csv_that_predates_it(tmp_path, monkeypatch):
    """Every already-committed data/raw/oxford/*.csv predates the c_rate
    column being added to _extract_group()'s row output — this exercises
    the same backfill path _load_cached() applies to those real files,
    without needing the 3GB+ source zips to regenerate them."""
    monkeypatch.setattr(oxford_mod, "_RAW_DIR", tmp_path)
    legacy_df = pd.DataFrame({
        "checkpoint_index": [0, 1, 2],
        "checkpoint_label": ["BoL", "RPT1", "RPT2"],
        "capacity_ah": [3.0, 2.9, 2.8],
        "soh_pct": [100.0, 96.7, 93.3],
    })
    legacy_df.to_csv(tmp_path / "cell9_checkpoints.csv", index=False)  # cell 9 -> Group 1

    df = oxford_mod._load_cached(9)
    assert "c_rate" in df.columns
    assert (df["c_rate"] == _GROUP_C_RATE[1]).all()


def test_group_c_rate_matches_module_docstring_c2_c4_split():
    # Groups 1 & 3 cycle @ C/2, Groups 2 & 4 @ C/4 (see module docstring).
    assert _GROUP_C_RATE[1] == _GROUP_C_RATE[3] == 0.5
    assert _GROUP_C_RATE[2] == _GROUP_C_RATE[4] == 0.25
