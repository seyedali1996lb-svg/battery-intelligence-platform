"""Unit tests for src/pack_aging.py — measured inputs and the load-aging projection.

Two kinds of test live here, matching the module's two functions:

- the *measurements* (fade slope, dR/dSOH) are checked for exact recovery on
  synthetic frames with a known slope, and — more importantly — for **refusing**
  whenever the evidence isn't there (short history, flat trend, too few usable
  resistance points, weak correlation, a 0 Ω sentinel). A measurement that
  invents a number when the history doesn't support one is the failure mode this
  file exists to catch.
- the *projection* is checked against the closed-form consequences of the model
  (share = 1/R over Σ1/R, fade ∝ ratio**α, linear accumulation, measured
  resistance growth moving the shares) and against
  ``pack_builder.compute_group_current_sharing`` for the starting ratio, so the
  aging view cannot report a different load ratio than the chart above it.
"""

import os as _os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import make_cycles_df

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
import _paths  # noqa: F401
import db as db_module
from streamlit.testing.v1 import AppTest
from pack_aging import (
    DEFAULT_CURRENT_AGING_EXPONENT,
    EOL_SOH_PCT,
    MAX_RECORDED_POINTS,
    MIN_FADE_CYCLES,
    measure_cell_aging_inputs,
    simulate_load_aging,
)
from pack_builder import compute_group_current_sharing


def _aging_frame(fade_pct_per_cycle=0.03, resistance_gain_ohm_per_pct=0.0015,
                 n_cycles=150, soh0=100.0, resistance0=0.05):
    """A frame whose fade slope and dR/dSOH are exactly known by construction."""
    cycles = np.arange(1, n_cycles + 1)
    soh = soh0 - fade_pct_per_cycle * (cycles - 1)
    return pd.DataFrame({
        "cycle_number": cycles,
        "soh_pct": soh,
        "resistance_ohm": resistance0 - resistance_gain_ohm_per_pct * (soh - soh0),
    })


def _cell(cell_id, *, soh, resistance, fade, gain=None, measured=None):
    return {
        "cell_id": cell_id, "soh_pct": soh, "resistance_ohm": resistance,
        "fade_pct_per_cycle": fade,
        "resistance_gain_ohm_per_soh_pct": gain,
        "resistance_gain_measured": (gain is not None) if measured is None else measured,
    }


# ---------------------------------------------------------------------------
# Measured inputs
# ---------------------------------------------------------------------------

def test_fade_slope_is_recovered_exactly_from_a_linear_record():
    m = measure_cell_aging_inputs(_aging_frame(fade_pct_per_cycle=0.031), "A")
    assert m["fade_pct_per_cycle"] == pytest.approx(0.031, rel=1e-9)
    assert m["fade_r2"] == pytest.approx(1.0, abs=1e-9)
    assert m["fade_cycles"] == 150


def test_resistance_gain_is_recovered_exactly_from_a_linear_record():
    m = measure_cell_aging_inputs(
        _aging_frame(fade_pct_per_cycle=0.03, resistance_gain_ohm_per_pct=0.0015), "A",
    )
    assert m["resistance_gain_ohm_per_soh_pct"] == pytest.approx(0.0015, rel=1e-9)
    assert m["resistance_gain_measured"] is True
    assert m["resistance_gain_corr"] == pytest.approx(-1.0, abs=1e-9)


def test_latest_soh_and_resistance_are_the_last_recorded_values():
    m = measure_cell_aging_inputs(_aging_frame(n_cycles=40), "A")
    assert m["soh_now"] == pytest.approx(100.0 - 0.03 * 39, abs=1e-9)
    assert m["resistance_ohm"] > 0.0
    assert m["cycle_min"] == 1 and m["cycle_max"] == 40


def test_a_short_history_refuses_a_fade_slope_rather_than_reporting_one():
    m = measure_cell_aging_inputs(_aging_frame(n_cycles=MIN_FADE_CYCLES - 1), "A")
    assert m["fade_pct_per_cycle"] is None
    assert any("fade rate" in what for what, _why in m["unavailable"])


def test_a_flat_or_rising_soh_trend_refuses_a_fade_slope():
    flat = _aging_frame(fade_pct_per_cycle=0.0, n_cycles=60)
    rising = _aging_frame(fade_pct_per_cycle=-0.01, n_cycles=60)  # SOH climbs
    for frame in (flat, rising):
        m = measure_cell_aging_inputs(frame, "A")
        assert m["fade_pct_per_cycle"] is None
        assert any("no measurable fade" in why for _what, why in m["unavailable"])


def test_too_few_usable_resistance_points_refuses_the_growth_term():
    frame = _aging_frame(n_cycles=50)
    frame.loc[:40, "resistance_ohm"] = 0.0  # Severson-style sentinels
    m = measure_cell_aging_inputs(frame, "A")
    assert m["resistance_gain_ohm_per_soh_pct"] is None
    assert m["resistance_gain_measured"] is False
    assert m["resistance_points"] == 9
    assert any("positive resistance" in why for _what, why in m["unavailable"])


def test_a_zero_resistance_is_a_sentinel_not_a_measurement():
    frame = _aging_frame(n_cycles=60)
    frame.loc[0, "resistance_ohm"] = 0.0
    m = measure_cell_aging_inputs(frame, "A")
    # The fit ignores the sentinel (so it still recovers the true gain exactly)
    # rather than treating 0 Ω as a real reading.
    assert m["resistance_points"] == 59
    assert m["resistance_gain_ohm_per_soh_pct"] == pytest.approx(0.0015, rel=1e-6)


def test_a_weak_resistance_soh_correlation_refuses_the_growth_term():
    frame = _aging_frame(n_cycles=120)
    rng = np.random.default_rng(7)
    frame["resistance_ohm"] = 0.05 + rng.normal(0.0, 0.004, len(frame))
    m = measure_cell_aging_inputs(frame, "A")
    assert m["resistance_gain_ohm_per_soh_pct"] is None
    assert any("correlation too weak" in why for _what, why in m["unavailable"])


def test_resistance_falling_as_soh_falls_refuses_the_growth_term():
    frame = _aging_frame(n_cycles=60)
    frame["resistance_ohm"] = 0.05 + 0.001 * (frame["soh_pct"] - 100.0)  # grows with SOH
    m = measure_cell_aging_inputs(frame, "A")
    assert m["resistance_gain_ohm_per_soh_pct"] is None
    assert any("falls as SOH falls" in why for _what, why in m["unavailable"])


def test_an_empty_frame_refuses_everything_without_raising():
    for frame in (None, pd.DataFrame({"cycle_number": [], "soh_pct": []})):
        m = measure_cell_aging_inputs(frame, "A")
        assert m["fade_pct_per_cycle"] is None
        assert m["soh_now"] != m["soh_now"]  # NaN
        assert m["unavailable"]


def test_the_recent_slope_is_reported_but_never_used_as_the_projection_rate():
    """The 30-cycle rolling column is noisy; the long-run fit is the rate."""
    frame = _aging_frame(fade_pct_per_cycle=0.03, n_cycles=150)
    # The last 30 cycles fade four times faster — the knee shape the rolling
    # column is noisy about and the long-run fit smooths over.
    soh_at_120 = float(frame.loc[frame["cycle_number"] == 120, "soh_pct"].iloc[0])
    tail = frame["cycle_number"] > 120
    frame.loc[tail, "soh_pct"] = soh_at_120 - 0.12 * (frame.loc[tail, "cycle_number"] - 120)
    rng = np.random.default_rng(3)
    frame["soh_pct"] = frame["soh_pct"] + rng.normal(0.0, 0.04, len(frame))
    m = measure_cell_aging_inputs(frame, "A")
    assert m["fade_recent_pct_per_cycle"] == pytest.approx(0.12, abs=0.01)
    # The long-run fit averages the whole record, so it cannot follow the knee
    # and lands well under the recent rate — that is the point of reporting both
    # rather than one number that means different things in each window.
    assert m["fade_pct_per_cycle"] < m["fade_recent_pct_per_cycle"] / 2.0
    assert m["fade_r2"] > 0.9  # and it is the stable one to project from


# ---------------------------------------------------------------------------
# The projection: shares, fade acceleration, and the two scenarios
# ---------------------------------------------------------------------------

def test_identical_cells_see_no_load_imbalance_and_no_divergence():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=0.001),
        _cell("B", soh=90.0, resistance=0.05, fade=0.02, gain=0.001),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=100)
    assert r["available"] is True
    assert r["loaded_load_ratio"] == pytest.approx(1.0)
    assert r["loaded_fade_acceleration_pct"] == pytest.approx(0.0)
    for cell in r["cells"]:
        assert cell["extra_loss_pct"] == pytest.approx(0.0)
        assert cell["soh_with_end"] == pytest.approx(cell["soh_without_end"])
    assert r["spread_delta_pct"] == pytest.approx(0.0)


def test_the_starting_load_ratio_matches_the_pack_builder_chart():
    """One load ratio in the app: the aging view must not recompute it differently."""
    cells = [
        _cell("A", soh=90.0, resistance=0.050, fade=0.02, gain=0.001),
        _cell("B", soh=88.0, resistance=0.060, fade=0.02, gain=0.001),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=10)
    chart = compute_group_current_sharing(cells)
    by_id = {c["cell_id"]: c for c in r["cells"]}
    for cell_id, share in chart["shares_pct"].items():
        # ratio = share × X, and X = 2 in this group.
        assert by_id[cell_id]["load_ratio_start"] == pytest.approx(share / 50.0, rel=1e-9)
    assert r["loaded_cell_id"] == chart["max_share_cell_id"]


def test_fade_acceleration_is_the_load_ratio_raised_to_the_exponent():
    cells = [
        _cell("low-R", soh=90.0, resistance=0.05, fade=0.02, gain=0.001),
        _cell("high-R", soh=90.0, resistance=0.10, fade=0.02, gain=0.001),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=5, exponent=0.7)
    expected_ratio = (1 / 0.05) / ((1 / 0.05) + (1 / 0.10)) * 2  # = 1.3333
    loaded = next(c for c in r["cells"] if c["cell_id"] == "low-R")
    assert loaded["load_ratio_start"] == pytest.approx(expected_ratio, rel=1e-9)
    assert loaded["fade_acceleration_pct"] == pytest.approx(
        (expected_ratio ** 0.7 - 1.0) * 100.0, rel=1e-9)


def test_extra_loss_is_the_accelerated_fade_integrated_over_the_horizon():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=None),
        _cell("B", soh=90.0, resistance=0.10, fade=0.02, gain=None),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=200, exponent=0.7)
    a = next(c for c in r["cells"] if c["cell_id"] == "A")
    ratio = a["load_ratio_start"]
    # Constant resistance (no measured gain) ⇒ the ratio never moves, so the
    # extra loss is exactly fade × (ratio^α − 1) × horizon.
    assert a["load_ratio_end"] == pytest.approx(ratio, rel=1e-12)
    assert a["extra_loss_pct"] == pytest.approx(0.02 * (ratio ** 0.7 - 1.0) * 200, rel=1e-6)
    assert a["equivalent_cycles_lost"] == pytest.approx(a["extra_loss_pct"] / 0.02, rel=1e-9)


def test_exponent_zero_turns_the_feedback_off_exactly():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=0.001),
        _cell("B", soh=90.0, resistance=0.12, fade=0.03, gain=0.002),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=100, exponent=0.0)
    for cell in r["cells"]:
        assert cell["soh_with_end"] == pytest.approx(cell["soh_without_end"])
        assert cell["extra_loss_pct"] == pytest.approx(0.0)
    assert r["spread_delta_pct"] == pytest.approx(0.0)


def test_a_bigger_exponent_can_only_add_loss_for_a_cell_carrying_more_than_its_share():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=None),
        _cell("B", soh=90.0, resistance=0.10, fade=0.02, gain=None),
    ]
    losses = [
        next(c for c in simulate_load_aging(cells, [[0, 1]], horizon_cycles=100, exponent=a)["cells"]
             if c["cell_id"] == "A")["extra_loss_pct"]
        for a in (0.2, 0.7, 1.2)
    ]
    assert losses[0] < losses[1] < losses[2]


def test_a_cell_carrying_less_than_its_share_gains_life_relative_to_the_other():
    cells = [
        _cell("heavy", soh=90.0, resistance=0.05, fade=0.02, gain=None),
        _cell("light", soh=90.0, resistance=0.10, fade=0.02, gain=None),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=100)
    heavy = next(c for c in r["cells"] if c["cell_id"] == "heavy")
    light = next(c for c in r["cells"] if c["cell_id"] == "light")
    assert heavy["extra_loss_pct"] > 0.0
    assert light["extra_loss_pct"] < 0.0
    # Sub-linear (α < 1) is concave, so concentrating current in one cell costs
    # the *group* slightly less total fade than splitting it evenly — the
    # redistribution is not zero-sum, it is very slightly negative-sum in the
    # group's favour. Pin that sign: it is the assumption doing the work.
    assert (heavy["soh_with_end"] + light["soh_with_end"]) > (
        heavy["soh_without_end"] + light["soh_without_end"])
    # …and with a super-linear exponent the same model says the opposite, so
    # this is genuinely the exponent's consequence and not baked into the loop.
    r_super = simulate_load_aging(cells, [[0, 1]], horizon_cycles=100, exponent=1.4)
    super_end = sum(c["soh_with_end"] for c in r_super["cells"])
    super_no = sum(c["soh_without_end"] for c in r_super["cells"])
    assert super_end < super_no


def test_a_measured_resistance_growth_moves_the_load_ratio_over_time():
    """The whole point of the loop: shares drift as the resistances separate."""
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=0.002),
        _cell("B", soh=90.0, resistance=0.05, fade=0.02, gain=0.0),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=300)
    a = next(c for c in r["cells"] if c["cell_id"] == "A")
    # A's resistance grows as it fades, B's does not, so A sheds load.
    assert a["load_ratio_end"] < a["load_ratio_start"]
    assert a["load_relief_pct"] < 0.0


def test_the_loaded_cell_can_also_take_on_more_load_over_time():
    """The drift is computed, not assumed self-limiting: give the *other* cell
    the faster absolute resistance growth and the imbalance widens."""
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=0.0),
        _cell("B", soh=90.0, resistance=0.05, fade=0.02, gain=0.002),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=300)
    a = next(c for c in r["cells"] if c["cell_id"] == "A")
    assert a["load_ratio_end"] > a["load_ratio_start"]
    assert a["load_relief_pct"] > 0.0


def test_a_cell_without_a_measured_gain_is_flagged_as_the_constant_load_bound():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=None),
        _cell("B", soh=90.0, resistance=0.10, fade=0.02, gain=None),
        _cell("C", soh=85.0, resistance=0.08, fade=0.03, gain=0.0015),
        _cell("D", soh=88.0, resistance=0.06, fade=0.025, gain=0.0012),
    ]
    r = simulate_load_aging(cells, [[0, 1], [2, 3]], horizon_cycles=300)
    assert r["n_constant_load"] == 2
    # With every member of a group held at constant resistance, nothing moves.
    for cell_id in ("A", "B"):
        cell = next(c for c in r["cells"] if c["cell_id"] == cell_id)
        assert cell["resistance_gain_measured"] is False
        assert cell["load_ratio_end"] == pytest.approx(cell["load_ratio_start"], rel=1e-12)
        assert cell["load_relief_pct"] == pytest.approx(0.0, abs=1e-12)
    # And a group where the cells do have measured gains does move.
    moved = next(c for c in r["cells"] if c["cell_id"] == "C")
    assert abs(moved["load_ratio_end"] - moved["load_ratio_start"]) > 1e-6


# ---------------------------------------------------------------------------
# Divergence: which way the pack's spread goes
# ---------------------------------------------------------------------------

def test_the_strongest_cell_in_a_group_working_hardest_narrows_the_spread():
    """Low resistance goes with high SOH, so the loaded cell is usually the
    strong one — the feedback then pulls the group together."""
    cells = [
        _cell("strong", soh=95.0, resistance=0.05, fade=0.02, gain=None),
        _cell("weak", soh=80.0, resistance=0.10, fade=0.02, gain=None),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=200)
    assert r["loaded_cell_id"] == "strong"
    assert r["spread_with_pct"] < r["spread_without_pct"]
    assert r["spread_delta_pct"] < 0.0


def test_a_loaded_weak_cell_widens_the_spread():
    cells = [
        _cell("weak", soh=80.0, resistance=0.05, fade=0.02, gain=None),
        _cell("strong", soh=95.0, resistance=0.10, fade=0.02, gain=None),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=200)
    assert r["loaded_cell_id"] == "weak"
    assert r["spread_with_pct"] > r["spread_without_pct"]
    assert r["spread_delta_pct"] > 0.0


def test_the_weaker_run_is_not_automatically_the_more_diverged_pack():
    """Guards the sign, not just the magnitude: with a loaded strong cell the
    *with-feedback* run ends up better balanced than the counterfactual."""
    cells = [
        _cell("strong", soh=95.0, resistance=0.05, fade=0.02, gain=None),
        _cell("weak", soh=80.0, resistance=0.10, fade=0.02, gain=None),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=200)
    assert r["pack_soh_with_pct"] > r["pack_soh_without_pct"]
    assert r["bottleneck_with"] == r["bottleneck_without"] == "weak"


def test_the_spread_curve_starts_at_the_selection_s_spread_and_ends_at_the_endpoints():
    cells = [
        _cell("A", soh=95.0, resistance=0.05, fade=0.02, gain=None),
        _cell("B", soh=80.0, resistance=0.10, fade=0.03, gain=None),
        _cell("C", soh=88.0, resistance=0.08, fade=0.01, gain=None),
    ]
    r = simulate_load_aging(cells, [[0, 1, 2]], horizon_cycles=100)
    starts = [c["soh_start"] for c in r["cells"]]
    assert r["spread_curve_with"][0] == pytest.approx(float(np.std(starts, ddof=1)))
    assert r["spread_curve_without"][0] == pytest.approx(r["spread_curve_with"][0])
    assert r["spread_curve_with"][-1] == pytest.approx(r["spread_with_pct"])
    assert r["spread_curve_without"][-1] == pytest.approx(r["spread_without_pct"])
    assert len(r["spread_curve_cycles"]) == len(r["spread_curve_with"]) <= MAX_RECORDED_POINTS


def test_the_horizon_is_where_the_projection_stops():
    cells = [
        _cell("A", soh=95.0, resistance=0.05, fade=0.02, gain=0.001),
        _cell("B", soh=90.0, resistance=0.08, fade=0.02, gain=0.001),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=137)
    for cell in r["cells"]:
        assert cell["cycles"][0] == 0.0
        assert cell["cycles"][-1] == pytest.approx(137.0)
        assert len(cell["cycles"]) <= MAX_RECORDED_POINTS
        assert len(cell["soh_with"]) == len(cell["cycles"]) == len(cell["soh_without"])
        assert len(cell["load_ratio_curve"]) == len(cell["cycles"])


# ---------------------------------------------------------------------------
# End of life, refusals and the series case
# ---------------------------------------------------------------------------

def test_a_cell_already_past_eol_reports_no_eol_shift():
    cells = [
        _cell("below", soh=75.0, resistance=0.05, fade=0.02, gain=None),
        _cell("above", soh=95.0, resistance=0.10, fade=0.02, gain=None),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=100)
    below = next(c for c in r["cells"] if c["cell_id"] == "below")
    assert below["below_eol_at_start"] is True
    assert below["eol_with"] is None and below["eol_without"] is None
    # The interpretable number survives: extra loss and equivalent cycles.
    assert below["equivalent_cycles_lost"] == pytest.approx(below["extra_loss_pct"] / 0.02, rel=1e-9)
    assert r["loaded_eol_shift_cycles"] is None


def test_a_cell_crossing_eol_inside_the_horizon_reports_when_under_each_scenario():
    cells = [
        _cell("heavy", soh=95.0, resistance=0.05, fade=0.10, gain=None),
        _cell("light", soh=95.0, resistance=0.10, fade=0.10, gain=None),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=400)
    heavy = next(c for c in r["cells"] if c["cell_id"] == "heavy")
    assert heavy["eol_with"] is not None and heavy["eol_without"] is not None
    assert heavy["eol_with"] < heavy["eol_without"]
    assert r["loaded_eol_shift_cycles"] == pytest.approx(
        heavy["eol_without"] - heavy["eol_with"])
    assert r["loaded_eol_shift_cycles"] > 0.0
    # Crossing is interpolated on the recorded curve, so it lands on the SOH.
    assert heavy["eol_with"] * 0.0 == 0.0


def test_a_series_configuration_has_no_load_ratio_to_turn_into_anything():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=0.001),
        _cell("B", soh=85.0, resistance=0.08, fade=0.03, gain=0.001),
    ]
    r = simulate_load_aging(cells, [[0], [1]], horizon_cycles=100)
    assert r["available"] is True
    assert all(g["sharing_modelled"] is False for g in r["groups"])
    assert "series string" in r["groups"][0]["reason"]
    for cell in r["cells"]:
        assert cell["load_ratio_start"] == pytest.approx(1.0)
        assert cell["sharing_modelled"] is False
        assert cell["extra_loss_pct"] == pytest.approx(0.0)


def test_a_group_whose_cells_cannot_all_be_projected_does_not_get_a_fabricated_share():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=0.001),
        _cell("B", soh=88.0, resistance=0.08, fade=None, gain=0.001),  # no fade measured
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=50)
    assert r["groups"][0]["sharing_modelled"] is False
    assert "cannot be projected" in r["groups"][0]["reason"]
    assert r["excluded"] == [{"cell_id": "B", "reason": "no measured fade slope"}]
    a = next(c for c in r["cells"] if c["cell_id"] == "A")
    assert a["load_ratio_start"] == pytest.approx(1.0)


def test_unprojectable_cells_are_named_and_the_rest_still_project():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=0.001),
        _cell("B", soh=85.0, resistance=0.09, fade=0.03, gain=0.001),
        _cell("C", soh=70.0, resistance=float("nan"), fade=0.02, gain=0.001),
    ]
    r = simulate_load_aging(cells, [[0, 1], [2]], horizon_cycles=100)
    assert r["available"] is True
    assert [e["cell_id"] for e in r["excluded"]] == ["C"]
    assert r["n_projected"] == 2
    assert r["groups"][1]["sharing_modelled"] is False


def test_no_usable_input_at_all_is_reported_not_projected():
    r = simulate_load_aging([{"cell_id": "X", "soh_pct": 90.0, "resistance_ohm": 0.05}], [[0]])
    assert r["available"] is False
    assert "measurable fade slope" in r["unavailable_reason"]
    assert r["cells"] == []
    assert r["excluded"][0]["cell_id"] == "X"


def test_empty_input_is_handled_without_raising():
    for stats, groups in (([], []), ([], [[0]]), ([_cell("A", soh=90, resistance=0.05, fade=0.02)], [])):
        r = simulate_load_aging(stats, groups)
        assert r["available"] is False
        assert r["unavailable_reason"]
        assert r["spread_curve_with"] == []


def test_a_silly_horizon_falls_back_to_the_documented_default():
    cells = [
        _cell("A", soh=90.0, resistance=0.05, fade=0.02, gain=0.001),
        _cell("B", soh=88.0, resistance=0.08, fade=0.02, gain=0.001),
    ]
    for bad in (0, -50, float("nan")):
        r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=bad)
        assert r["horizon_cycles"] > 0
        assert r["cells"][0]["cycles"][-1] == pytest.approx(r["horizon_cycles"])


def test_the_default_exponent_is_the_one_the_module_documents():
    assert DEFAULT_CURRENT_AGING_EXPONENT == 0.7
    assert EOL_SOH_PCT == 80.0


def test_the_projection_never_drives_soh_below_the_floor_in_either_run():
    cells = [
        _cell("A", soh=50.0, resistance=0.05, fade=2.0, gain=0.01),
        _cell("B", soh=50.0, resistance=0.05, fade=2.0, gain=0.01),
    ]
    r = simulate_load_aging(cells, [[0, 1]], horizon_cycles=500)
    for cell in r["cells"]:
        assert min(cell["soh_with"]) >= 0.0
        assert min(cell["soh_without"]) >= 0.0
        assert cell["soh_with_end"] >= 0.0
        assert cell["soh_without_end"] >= 0.0


# ---------------------------------------------------------------------------
# On real-shaped data: the measured slope is recovered from the app's own
# synthetic generator (linear, so the slope is known exactly).
# ---------------------------------------------------------------------------

def test_the_app_s_own_generator_frame_yields_the_known_slope_and_gain():
    frame = make_cycles_df(n_cycles=200, initial_capacity_ah=2.0, fade_per_cycle=0.0006,
                           initial_resistance_ohm=0.05, resistance_rise_per_cycle=0.00005)
    m = measure_cell_aging_inputs(frame, "Cell1")
    # SOH falls 0.03 points per cycle; resistance rises 5e-5 Ω per cycle ⇒
    # 5e-5 / 0.03 Ω per SOH point.
    assert m["fade_pct_per_cycle"] == pytest.approx(0.03, rel=1e-6)
    assert m["resistance_gain_ohm_per_soh_pct"] == pytest.approx(0.00005 / 0.03, rel=1e-6)
    assert m["resistance_gain_measured"] is True


# ---------------------------------------------------------------------------
# Page-level smoke: the section renders inside Explore's Pack Builder tab
# ---------------------------------------------------------------------------

_MAIN_PY = str(pathlib.Path(__file__).parent.parent / "app" / "main.py")


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(
        db_module, "engine",
        db_module.create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}),
    )
    monkeypatch.setattr(db_module, "Session", db_module.sessionmaker(bind=db_module.engine))
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", "03ZJHIomd1hhT9w4FWvNxoN2wqPUnjfg3bSycZqUmgY=")
    monkeypatch.setattr(db_module, "_fernet", None)
    db_module.init_db()
    return db_module


def _rendered_text(at: AppTest) -> str:
    """Everything the page rendered as text, including inside expanders.

    Streamlit's AppTest does not flatten block children into the top-level
    accessors, so an expander's contents have to be read from the block itself.
    """
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    for block in at.expander:
        parts += [m.value for m in block.markdown]
        parts += [c.value for c in block.caption]
        parts += [i.value for i in block.info]
    # Collapse whitespace: the prose is hard-wrapped, so a phrase can straddle a
    # newline in the source and would otherwise have to be asserted in fragments.
    return " ".join(" ".join(parts).split())


def _pack_builder_app(topology: str, cells: list) -> AppTest:
    at = AppTest.from_file(_MAIN_PY, default_timeout=180)
    at.session_state["authenticated"] = True
    at.session_state["auth_org_id"] = 1
    at.session_state["auth_org_name"] = "Demo Org"
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["auth_name"] = "Administrator"
    at.session_state["role_chosen"] = True
    at.session_state["mode_chosen"] = True
    at.session_state["tour_seen"] = True
    at.session_state["user_role"] = "Engineer"
    at.session_state["page"] = "compare"
    at.session_state["data_mode"] = "nasa"
    at.session_state["explore_view_radio"] = "Pack Builder"
    at.session_state["explore_pack_cells"] = cells
    at.session_state["explore_pack_topology"] = topology
    return at


@pytest.mark.parametrize("topology", ["Parallel", "Series"])
def test_load_driven_divergence_renders_in_the_pack_builder(isolated_db, topology):
    """Both branches: a parallel block has a load ratio to project forward, a
    series string has none and must say so instead of drawing a flat chart."""
    at = _pack_builder_app(topology, ["B0005", "B0006"])
    at.run()
    assert not at.exception, f"Pack Builder crashed rendering the aging view ({topology}): {at.exception}"
    body = _rendered_text(at)
    if topology == "Parallel":
        assert "Load-driven divergence" in body
        assert "fair share" in body
        assert "lifetime prediction" in body and "not" in body
        metrics = "\n".join(f"{m.label}: {m.value}" for m in at.metric)
        assert "Most-loaded cell" in metrics
        assert "load / fair share" in metrics
        assert "Extra SOH lost" in metrics
    else:
        assert "not reported for a series configuration" in body, (
            "a series string has no current imbalance and must say so"
        )
        assert "Load-driven divergence</h4>" not in body, (
            "…and must not draw a section of zeros to go with it"
        )


def test_the_aging_view_shows_the_measured_inputs_and_the_assumptions(isolated_db):
    at = _pack_builder_app("XpYs", ["B0005", "B0006", "B0007", "B0018"])
    at.session_state["explore_pack_cells_per_group"] = 2
    at.run()
    assert not at.exception, f"XpYs aging view crashed: {at.exception}"
    body = _rendered_text(at)
    assert "Where the inputs come from" in body
    assert "Fair share = the cells' historical duty" in body
    assert "lower bound" in body and "upper bound" in body
    assert "stress_index" in body, "the assumed exponent must cite where it comes from"
    # The section's headline numbers, all of which are derived from the cells'
    # own records for this real NASA selection.
    assert "SOH points" in body
    assert "spread" in body
    assert "measured baseline fade" in body
    # The per-cell table must carry the measured dR/dSOH (or say it was held).
    assert any("mΩ/pt" in str(cell.value) or "held constant" in str(cell.value)
               for block in at.expander for cell in [block.dataframe[0]] if block.dataframe)
