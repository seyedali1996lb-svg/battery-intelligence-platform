"""Unit tests for src/warranty.py's warranty-breach risk scoring."""

import sys
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from warranty import (
    cycles_to_soh_floor_linear, warranty_breach_estimate,
    probability_of_breach_by, estimated_breach_date,
)


def test_cycles_to_soh_floor_linear_basic():
    # 90% -> 70% floor at 0.02%/cycle fade = 1000 cycles
    assert cycles_to_soh_floor_linear(90.0, 0.02, 70.0) == 1000.0


def test_cycles_to_soh_floor_linear_already_breached():
    assert cycles_to_soh_floor_linear(65.0, 0.02, 70.0) == 0.0


def test_cycles_to_soh_floor_linear_zero_fade_rate_is_none():
    assert cycles_to_soh_floor_linear(90.0, 0.0, 70.0) is None


def test_warranty_breach_estimate_flags_already_breached():
    result = warranty_breach_estimate(65.0, 0.02, 70.0, rul_pred=100, rul_q10=80, rul_q90=120, rul_reliable=True)
    assert result["breached"] is True
    assert result["linear_estimate"] == 0.0
    assert result["model_scaled_estimate"] == 0.0


def test_warranty_breach_estimate_scales_model_rul_toward_stricter_floor():
    """Floor (75%) closer to current SOH (90%) than eol_threshold (80%) is
    -- current-90, eol-80 => remaining_to_eol=10; current-90, floor-75 =>
    remaining_to_floor=15 -- so the floor is actually LESS strict here
    (75 < 80), meaning it takes longer to reach than 80%. Scale should be
    1.5x rul_pred."""
    result = warranty_breach_estimate(
        current_soh_pct=90.0, fade_rate_pct_per_cycle=0.02, warranty_floor_soh_pct=75.0,
        rul_pred=200.0, rul_q10=150.0, rul_q90=250.0, eol_threshold_pct=80.0, rul_reliable=True,
    )
    assert result["confidence"] == "model"
    assert abs(result["model_scaled_estimate"] - 300.0) < 1e-6  # 200 * (15/10)
    assert abs(result["model_scaled_q10"] - 225.0) < 1e-6
    assert abs(result["model_scaled_q90"] - 375.0) < 1e-6


def test_warranty_breach_estimate_falls_back_to_linear_only_when_rul_unreliable():
    result = warranty_breach_estimate(
        current_soh_pct=90.0, fade_rate_pct_per_cycle=0.02, warranty_floor_soh_pct=75.0,
        rul_pred=200.0, rul_q10=150.0, rul_q90=250.0, rul_reliable=False,
    )
    assert result["confidence"] == "linear_only"
    assert result["model_scaled_estimate"] is None
    assert result["linear_estimate"] is not None  # linear estimate always computable


def test_warranty_breach_estimate_falls_back_when_current_soh_already_at_or_below_eol_threshold():
    """current_soh_pct <= eol_threshold_pct makes remaining_to_eol <= 0 --
    scaling would be undefined, so this must fall back to linear_only
    rather than dividing by a non-positive number."""
    result = warranty_breach_estimate(
        current_soh_pct=79.0, fade_rate_pct_per_cycle=0.02, warranty_floor_soh_pct=70.0,
        rul_pred=50.0, rul_q10=30.0, rul_q90=70.0, eol_threshold_pct=80.0, rul_reliable=True,
    )
    assert result["confidence"] == "linear_only"
    assert result["model_scaled_estimate"] is None


def test_probability_of_breach_by_midpoint_is_50pct():
    assert abs(probability_of_breach_by(150.0, q10=100.0, q90=200.0) - 0.5) < 1e-9


def test_probability_of_breach_by_before_q10_is_zero():
    assert probability_of_breach_by(50.0, q10=100.0, q90=200.0) == 0.0


def test_probability_of_breach_by_after_q90_is_one():
    assert probability_of_breach_by(300.0, q10=100.0, q90=200.0) == 1.0


def test_probability_of_breach_by_none_without_quantiles():
    assert probability_of_breach_by(150.0, q10=None, q90=200.0) is None


def test_estimated_breach_date_none_without_cycling_rate():
    assert estimated_breach_date(500.0, cycles_per_year=None) is None
    assert estimated_breach_date(None, cycles_per_year=365.0) is None


def test_estimated_breach_date_returns_future_date():
    import datetime
    result = estimated_breach_date(365.0, cycles_per_year=365.0)
    assert result is not None
    assert result > datetime.date.today()
