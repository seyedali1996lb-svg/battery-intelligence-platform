"""
Knee-point detection for battery SOH degradation curves.

A knee point is the cycle where capacity fade transitions from a long, slow
plateau into rapid acceleration — the "cliff" that precedes end of life.
Detecting it early is the most actionable signal in battery fleet management.

Algorithm (maximum-curvature / L-method):
  1. Smooth SOH with a rolling window to remove cycle noise.
  2. Normalise both axes to [0, 1] so curvature is scale-independent.
  3. Fit a line from the first to the last point of the smoothed curve.
  4. The knee = point of maximum perpendicular distance from that line.
  5. A confidence score measures how pronounced the bend is vs. a linear model.

Why not the second-derivative peak?
  The second derivative is sensitive to noise even after smoothing, and
  produces false positives on cells with multi-phase degradation. The
  L-method (Satopää et al. 2011, adapted) is more robust on small datasets.
"""

import numpy as np
import pandas as pd


def detect_knee(
    soh_series: pd.Series,
    cycle_series: pd.Series,
    smooth_window: int = 15,
    min_cycles: int = 50,
) -> dict:
    """
    Detect the knee point in a SOH vs cycle curve.

    Args:
        soh_series:   SOH % values (may have noise).
        cycle_series: Corresponding cycle numbers.
        smooth_window: Rolling mean window for denoising.
        min_cycles:   Minimum cycles before knee detection is attempted.

    Returns:
        {
            "detected":    bool   — True if a credible knee was found,
            "cycle":       int    — cycle number of the knee (or None),
            "soh_at_knee": float  — SOH % at the knee (or None),
            "confidence":  float  — 0–1; how pronounced the bend is,
            "phase":       str    — "Early", "Plateau", "Accelerating", or "Post-knee",
        }
    """
    soh = soh_series.values.astype(float)
    cyc = cycle_series.values.astype(float)

    if len(soh) < min_cycles:
        return _no_knee(soh, cyc)

    # Smooth
    s = pd.Series(soh).rolling(smooth_window, center=True, min_periods=max(3, smooth_window // 3)).mean()
    valid = ~s.isna()
    s_vals = s[valid].values
    c_vals = cyc[valid]

    if len(s_vals) < 20:
        return _no_knee(soh, cyc)

    # Normalise to [0, 1]
    c_norm = (c_vals - c_vals[0]) / (c_vals[-1] - c_vals[0] + 1e-9)
    s_norm = (s_vals - s_vals[-1]) / (s_vals[0] - s_vals[-1] + 1e-9)  # flip so 1=fresh

    # Line from first to last point in normalised space
    # Perpendicular distance from point (x, y) to line through (0,1)→(1,0):
    #   d = |x + y - 1| / sqrt(2)
    distances = np.abs(c_norm + s_norm - 1.0) / np.sqrt(2)

    # Knee = maximum distance, but only in the second half of the curve
    # (first-quarter fluctuations are early-cycle noise, not a knee)
    quarter = max(1, len(distances) // 4)
    knee_idx_local = int(np.argmax(distances[quarter:])) + quarter
    knee_cycle = int(c_vals[knee_idx_local])
    knee_soh   = float(s_vals[knee_idx_local])
    max_dist   = float(distances[knee_idx_local])

    # Confidence: ratio of max distance to what a perfectly linear curve would give (0)
    # A perfectly linear curve has zero distance everywhere; we scale by the theoretical
    # maximum possible distance (0.5 * sqrt(2) / sqrt(2) = 0.5 for a right-angle step).
    confidence = float(np.clip(max_dist / 0.5, 0.0, 1.0))

    # Only flag as "detected" if bend is substantial
    detected = confidence > 0.15 and knee_cycle > c_vals[quarter]

    # Degradation phase of the CURRENT (last) point
    phase = _degradation_phase(soh, cyc, knee_cycle if detected else None)

    return {
        "detected":    detected,
        "cycle":       knee_cycle if detected else None,
        "soh_at_knee": round(knee_soh, 1) if detected else None,
        "confidence":  round(confidence, 3),
        "phase":       phase,
    }


def degradation_phases(
    soh_series: pd.Series,
    cycle_series: pd.Series,
    smooth_window: int = 15,
) -> pd.Series:
    """
    Label every cycle with its degradation phase: Early / Plateau / Accelerating.

    Used for the per-cycle phase colouring on the Health page.
    """
    result = detect_knee(soh_series, cycle_series, smooth_window)
    knee = result["cycle"]

    labels = pd.Series("Plateau", index=soh_series.index)

    # Early: first 50 cycles (rolling features haven't stabilised)
    early_mask = cycle_series <= 50
    labels[early_mask] = "Early"

    if knee is not None:
        labels[cycle_series >= knee] = "Accelerating"

    return labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_knee(soh, cyc):
    phase = _degradation_phase(soh, cyc, None)
    return {"detected": False, "cycle": None, "soh_at_knee": None,
            "confidence": 0.0, "phase": phase}


def _degradation_phase(soh, cyc, knee_cycle):
    """Classify the most recent data point's degradation phase."""
    if len(cyc) == 0:
        return "Unknown"
    last_cycle = float(cyc[-1])
    if last_cycle <= 50:
        return "Early"
    if knee_cycle is not None and last_cycle >= knee_cycle:
        return "Accelerating"
    return "Plateau"
