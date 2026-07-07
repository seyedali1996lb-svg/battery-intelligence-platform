"""Regression tests for src/mqtt_stream.py telemetry schema and anomaly detection.

Guards against the SOH/SOC mislabeling bug: the replay publisher reads a cell's
soh_pct column but used to republish it under the wire key "soc_pct" (State of
Charge), which is a different metric than State of Health. Live Monitor's
metrics strip and charts then labeled that value "SOC" to the user. The fix
renames the wire key (and the AnomalyDetector's internal variable names /
flag text) to soh_pct/SOH throughout, matching what the data actually is.
"""

import inspect

import mqtt_stream
from mqtt_stream import AnomalyDetector


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
