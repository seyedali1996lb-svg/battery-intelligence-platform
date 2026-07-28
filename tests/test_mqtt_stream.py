"""Regression tests for src/mqtt_stream.py telemetry schema and anomaly detection.

Guards against the SOH/SOC mislabeling bug: the replay publisher reads a cell's
soh_pct column but used to republish it under the wire key "soc_pct" (State of
Charge), which is a different metric than State of Health. Live Monitor's
metrics strip and charts then labeled that value "SOC" to the user. The fix
renames the wire key (and the AnomalyDetector's internal variable names /
flag text) to soh_pct/SOH throughout, matching what the data actually is.
"""

import inspect
import json
from types import SimpleNamespace

import numpy as np

import mqtt_stream
from mqtt_stream import AnomalyDetector, _ZSCORE_WINDOW, validate_telemetry


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


# ---------------------------------------------------------------------------
# validate_telemetry() -- ingestion fault detection (malformed/corrupted
# data), a structurally different concern from AnomalyDetector.check()
# above (which only ever judges an already-well-formed reading's value).
# ---------------------------------------------------------------------------

def _clean_msg(**overrides):
    msg = {
        "cell_id": "B0005", "cycle": 1, "seq": 0,
        "ts": "2024-01-15T12:00:00Z",
        "voltage_v": 3.7, "current_a": -2.0, "temperature_c": 24.0,
        "capacity_ah": 1.8, "soh_pct": 90.0,
    }
    msg.update(overrides)
    return msg


def test_clean_message_produces_no_faults():
    detector = AnomalyDetector("B0005", "default")
    assert validate_telemetry(_clean_msg(), detector) == []


def test_missing_cell_id_flagged():
    detector = AnomalyDetector("B0005", "default")
    msg = _clean_msg()
    del msg["cell_id"]
    faults = validate_telemetry(msg, detector)
    assert any(f["kind"] == "MISSING_CELL_ID" for f in faults)


def test_missing_timestamp_flagged():
    detector = AnomalyDetector("B0005", "default")
    msg = _clean_msg()
    del msg["ts"]
    faults = validate_telemetry(msg, detector)
    assert any(f["kind"] == "MISSING_TIMESTAMP" for f in faults)


def test_unparseable_timestamp_flagged():
    detector = AnomalyDetector("B0005", "default")
    faults = validate_telemetry(_clean_msg(ts="not-a-timestamp"), detector)
    assert any(f["kind"] == "UNPARSEABLE_TIMESTAMP" for f in faults)


def test_out_of_order_timestamp_flagged():
    detector = AnomalyDetector("B0005", "default")
    validate_telemetry(_clean_msg(ts="2024-01-15T12:00:10Z"), detector)
    faults = validate_telemetry(_clean_msg(ts="2024-01-15T12:00:05Z"), detector)
    assert any(f["kind"] == "OUT_OF_ORDER_TIMESTAMP" for f in faults)


def test_duplicate_timestamp_flagged():
    detector = AnomalyDetector("B0005", "default")
    validate_telemetry(_clean_msg(ts="2024-01-15T12:00:10Z"), detector)
    faults = validate_telemetry(_clean_msg(ts="2024-01-15T12:00:10Z"), detector)
    assert any(f["kind"] == "DUPLICATE_TIMESTAMP" for f in faults)


def test_monotonic_timestamps_do_not_fault():
    detector = AnomalyDetector("B0005", "default")
    validate_telemetry(_clean_msg(ts="2024-01-15T12:00:00Z"), detector)
    faults = validate_telemetry(_clean_msg(ts="2024-01-15T12:00:05Z"), detector)
    assert faults == []


def test_dropped_packet_gap_flagged():
    detector = AnomalyDetector("B0005", "default")
    validate_telemetry(_clean_msg(seq=10), detector)
    faults = validate_telemetry(_clean_msg(seq=15), detector)
    gap_faults = [f for f in faults if f["kind"] == "DROPPED_PACKET_GAP"]
    assert len(gap_faults) == 1
    assert "4 message" in gap_faults[0]["detail"]


def test_consecutive_seq_does_not_fault():
    detector = AnomalyDetector("B0005", "default")
    validate_telemetry(_clean_msg(seq=10), detector)
    faults = validate_telemetry(_clean_msg(seq=11), detector)
    assert not any(f["kind"] == "DROPPED_PACKET_GAP" for f in faults)


def test_implausible_capacity_flagged_for_unit_mixup():
    """A capacity_ah of 1823 (an Ah field actually holding a millamp-hour
    value -- 1.823 Ah reported as 1823) must be flagged -- this is a real
    unit-mixup case the existing AnomalyDetector cannot catch at all, since
    it never looks at capacity_ah."""
    detector = AnomalyDetector("B0005", "default")
    faults = validate_telemetry(_clean_msg(capacity_ah=1823.0), detector)
    assert any(f["kind"] == "IMPLAUSIBLE_CAPACITY" for f in faults)


def test_negative_capacity_flagged():
    detector = AnomalyDetector("B0005", "default")
    faults = validate_telemetry(_clean_msg(capacity_ah=-1.0), detector)
    assert any(f["kind"] == "IMPLAUSIBLE_CAPACITY" for f in faults)


def test_plausible_capacity_does_not_fault():
    detector = AnomalyDetector("B0005", "default")
    faults = validate_telemetry(_clean_msg(capacity_ah=1.8), detector)
    assert faults == []


def test_non_numeric_capacity_flagged():
    detector = AnomalyDetector("B0005", "default")
    faults = validate_telemetry(_clean_msg(capacity_ah="oops"), detector)
    assert any(f["kind"] == "NON_NUMERIC_CAPACITY" for f in faults)


# ---------------------------------------------------------------------------
# _on_message() wiring -- faults are quarantined (logged + queued), not
# silently dropped, and don't block telemetry/anomaly processing.
# ---------------------------------------------------------------------------

def _fake_msg(payload: dict, topic: str = "battery-intelligence/B0005/telemetry"):
    return SimpleNamespace(payload=json.dumps(payload).encode(), topic=topic)


def test_on_message_pushes_fault_to_queue():
    mqtt_stream._detectors.pop("FAULT-TEST-1", None)
    mqtt_stream.drain_faults(1000)  # empty any leftovers from other tests

    mqtt_stream._on_message(None, {"chemistry_map": {}}, _fake_msg(_clean_msg(cell_id="FAULT-TEST-1", capacity_ah=9999.0)))

    faults = mqtt_stream.drain_faults(1000)
    assert any(f["kind"] == "IMPLAUSIBLE_CAPACITY" and f["cell_id"] == "FAULT-TEST-1" for f in faults)


def test_on_message_malformed_json_logged_not_raised(caplog):
    bad_msg = SimpleNamespace(payload=b"{not valid json", topic="battery-intelligence/B0005/telemetry")
    with caplog.at_level("WARNING", logger="mqtt_stream"):
        mqtt_stream._on_message(None, {"chemistry_map": {}}, bad_msg)  # must not raise
    assert any("unparseable" in r.getMessage().lower() for r in caplog.records)


def test_on_message_still_pushes_telemetry_and_anomalies_alongside_faults():
    mqtt_stream._detectors.pop("FAULT-TEST-2", None)
    mqtt_stream.drain_telemetry(1000)
    mqtt_stream.drain_anomalies(1000)
    mqtt_stream.drain_faults(1000)

    mqtt_stream._on_message(
        None, {"chemistry_map": {}},
        _fake_msg(_clean_msg(cell_id="FAULT-TEST-2", capacity_ah=9999.0, voltage_v=10.0)),
    )

    telemetry = mqtt_stream.drain_telemetry(1000)
    anomalies = mqtt_stream.drain_anomalies(1000)
    faults    = mqtt_stream.drain_faults(1000)
    assert len(telemetry) == 1
    assert any(a["kind"] == "OVERVOLTAGE" for a in anomalies)
    assert any(f["kind"] == "IMPLAUSIBLE_CAPACITY" for f in faults)
