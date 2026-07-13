"""Unit tests for batlab.features.knee_detection — detect_knee() / degradation_phases()."""

import numpy as np
import pandas as pd
from batlab.features.knee_detection import detect_knee, degradation_phases


def test_too_short_series_not_detected():
    soh = pd.Series(np.linspace(100, 95, 20))
    cyc = pd.Series(np.arange(1, 21))
    result = detect_knee(soh, cyc, min_cycles=50)
    assert result["detected"] is False
    assert result["cycle"] is None


def test_clear_knee_is_detected():
    """A long plateau followed by a sharp drop should produce a high-confidence
    knee roughly where the plateau ends."""
    n = 300
    knee_at = 220
    cyc = np.arange(1, n + 1)
    soh = np.where(cyc < knee_at, 100 - (cyc / knee_at) * 5, 95 - (cyc - knee_at) * 0.15)
    result = detect_knee(pd.Series(soh), pd.Series(cyc))
    assert result["detected"] == True
    assert result["confidence"] > 0.15
    # Knee should land somewhere in the back half of the curve, near the bend
    assert knee_at - 40 <= result["cycle"] <= knee_at + 40


def test_degradation_phases_labels_early_cycles():
    n = 300
    cyc = pd.Series(np.arange(1, n + 1))
    soh = pd.Series(np.linspace(100, 80, n))
    phases = degradation_phases(soh, cyc)
    assert (phases[cyc <= 50] == "Early").all()
    assert set(phases.unique()).issubset({"Early", "Plateau", "Accelerating"})
