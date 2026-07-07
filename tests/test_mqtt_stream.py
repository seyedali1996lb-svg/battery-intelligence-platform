"""Regression tests for src/mqtt_stream.py telemetry schema and anomaly detection.

Guards against the SOH/SOC mislabeling bug: the replay publisher reads a cell's
soh_pct column but used to republish it under the wire key "soc_pct" (State of
Charge), which is a different metric than State of Health. Live Monitor's
metrics strip and charts then labeled that value "SOC" to the user. The fix
renames the wire key (and the AnomalyDetector's internal variable names /
flag text) to soh_pct/SOH throughout, matching what the data actually is.
"""

import inspect

import numpy as np

import mqtt_stream
from mqtt_stream import AnomalyDetector, _ZSCORE_WINDOW


def test_no_soc_pct_key_anywhere_in_module():
    """soc_pct must not reappear in the module — the wire schema only ever
    carries soh_pct; a real point-in-time SOC field doesn't exist in this
    telemetry stream, so reintroducing "soc_pct" would recreate the mislabel."""
    source = inspect.getsource(mqtt_stream)
    assert "soc_pct" not in source


def test_capacity_plunge_reads_soh_pct_key():
    detector = AnomalyDetector("B0005", "default")
    detector.check({"voltage_v": 3.7, "soh_pct": 80.0})
    detector.check({"voltage_v": 3.7, "soh_pct": 80.0})
    flags = detector.check({"voltage_v": 3.7, "soh_pct": 70.0})

    kinds = [f["kind"] for f in flags]
    assert "CAPACITY_PLUNGE" in kinds


def test_capacity_plunge_message_says_soh_not_soc():
    detector = AnomalyDetector("B0005", "default")
    detector.check({"voltage_v": 3.7, "soh_pct": 80.0})
    detector.check({"voltage_v": 3.7, "soh_pct": 80.0})
    flags = detector.check({"voltage_v": 3.7, "soh_pct": 70.0})

    plunge = next(f for f in flags if f["kind"] == "CAPACITY_PLUNGE")
    assert "SOH dropped" in plunge["detail"]
    assert "SOC" not in plunge["detail"]


def test_soc_pct_key_no_longer_triggers_plunge():
    """Sending the old (wrong) key should not be picked up at all."""
    detector = AnomalyDetector("B0005", "default")
    detector.check({"voltage_v": 3.7, "soc_pct": 80.0})
    detector.check({"voltage_v": 3.7, "soc_pct": 80.0})
    flags = detector.check({"voltage_v": 3.7, "soc_pct": 10.0})

    assert all(f["kind"] != "CAPACITY_PLUNGE" for f in flags)


# ---------------------------------------------------------------------------
# MULTI_SIGNAL_ANOMALY -- AI Integration review finding: the review's
# recommendation was to invest real anomaly-detection effort beyond static
# thresholds, not just rephrase the same threshold checks with an LLM. The
# rolling per-channel Z-score already existed; this adds a genuine
# multivariate step -- a combined (Euclidean-norm) score across channels
# that can flag a real anomaly even when no single channel alone crosses
# its own threshold, since correlated moderate drift across multiple
# sensors is a materially stronger fault signal than one channel alone.
# ---------------------------------------------------------------------------

def _seed_baseline(detector, voltages, temps):
    for v, t in zip(voltages, temps):
        detector.check({"voltage_v": float(v), "temperature_c": float(t)})


def _baseline_series():
    rng = np.random.default_rng(42)
    voltages = 3.70 + rng.normal(0, 0.01, _ZSCORE_WINDOW)
    temps    = 24.0 + rng.normal(0, 0.2, _ZSCORE_WINDOW)
    return voltages, temps


def test_multi_signal_anomaly_fires_when_no_single_channel_would():
    """
    Two channels each drifting only moderately (individually below the
    single-channel Z-score alert threshold) at the same time must still
    raise MULTI_SIGNAL_ANOMALY -- this is exactly the case a per-channel-
    only check structurally cannot see, and the reason this feature exists.
    """
    voltages, temps = _baseline_series()
    v_mu, v_sigma = voltages.mean(), voltages.std()
    t_mu, t_sigma = temps.mean(), temps.std()

    detector = AnomalyDetector("B0005", "default")
    _seed_baseline(detector, voltages, temps)

    zfactor = 3.2  # empirically: below single-channel threshold, above combined
    v_test = v_mu - zfactor * v_sigma
    t_test = t_mu + zfactor * t_sigma
    flags = detector.check({"voltage_v": float(v_test), "temperature_c": float(t_test)})
    kinds = [f["kind"] for f in flags]

    assert "MULTI_SIGNAL_ANOMALY" in kinds
    assert "ZSCORE_VOLTAGE" not in kinds
    assert "ZSCORE_TEMPERATURE" not in kinds


def test_multi_signal_anomaly_does_not_fire_from_one_channel_alone():
    """A large deviation confined to a single channel is a single-channel
    story (ZSCORE_* / OVER-/UNDERVOLTAGE), not a correlated multi-signal
    one -- MULTI_SIGNAL_ANOMALY requires 2+ channels elevated together."""
    voltages, temps = _baseline_series()
    v_mu, v_sigma = voltages.mean(), voltages.std()

    detector = AnomalyDetector("B0005", "default")
    _seed_baseline(detector, voltages, temps)

    v_test = v_mu - 3.2 * v_sigma
    t_test = float(temps[-1])  # temperature stays at baseline -- only voltage moves
    flags = detector.check({"voltage_v": float(v_test), "temperature_c": t_test})
    kinds = [f["kind"] for f in flags]

    assert "MULTI_SIGNAL_ANOMALY" not in kinds


def test_multi_signal_anomaly_does_not_fire_on_quiet_baseline():
    """No perturbation at all -- must not false-positive on stable data."""
    voltages, temps = _baseline_series()
    detector = AnomalyDetector("B0005", "default")
    _seed_baseline(detector, voltages, temps)

    flags = detector.check({"voltage_v": float(voltages[-1]), "temperature_c": float(temps[-1])})
    kinds = [f["kind"] for f in flags]

    assert "MULTI_SIGNAL_ANOMALY" not in kinds
