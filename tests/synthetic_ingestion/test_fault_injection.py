"""
Fault-injection assertion suite: replays real, already-cached NASA cycling
data (see corruptions.py) through the actual MQTT ingestion function
(mqtt_stream._on_message()) with each SYNTHETIC corruption applied, and
asserts that src/mqtt_stream.py's validate_telemetry() (added for this
Phase 7 ingestion-fault-detection work) actually catches it -- exercising
the ingestion layer's error handling before any real partner data exists
to exercise it for real.

No live MQTT broker is used anywhere in this file -- _on_message() is
called directly with a fake msg object, exactly like
tests/test_mqtt_stream.py's own _on_message() tests.
"""

import json
from types import SimpleNamespace

import mqtt_stream

from synthetic_ingestion.corruptions import (
    corrupt_timestamps, corrupt_units, drop_packets,
    load_source_cell_rows, rows_to_clean_messages,
)


def _fake_msg(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        payload=json.dumps(payload).encode(),
        topic=f"battery-intelligence/{payload.get('cell_id', '?')}/telemetry",
    )


def _replay(messages: "list[dict]", cell_id: str) -> "list[dict]":
    """Drains any leftover state for cell_id, replays every message through
    the real _on_message() ingestion function, and returns every fault it
    produced (drains the full queue, not just the last batch)."""
    mqtt_stream._detectors.pop(cell_id, None)
    mqtt_stream.drain_faults(10_000)
    mqtt_stream.drain_anomalies(10_000)
    mqtt_stream.drain_telemetry(10_000)

    for m in messages:
        mqtt_stream._on_message(None, {"chemistry_map": {}}, _fake_msg(m))

    return mqtt_stream.drain_faults(10_000)


_SOURCE_ROWS = load_source_cell_rows("B0005", n=30)


def test_clean_replay_of_real_nasa_data_produces_no_faults():
    """Sanity/no-false-positive baseline: replaying real NASA B0005 data
    through the ingestion pipeline with no corruption applied must not
    trip any fault check."""
    messages = rows_to_clean_messages(_SOURCE_ROWS, cell_id="SYN-CLEAN")
    faults = _replay(messages, "SYN-CLEAN")
    assert faults == []


def test_shuffled_timestamps_trigger_out_of_order_fault():
    """SYNTHETIC: reordered messages (network reordering / clock skew)."""
    clean = rows_to_clean_messages(_SOURCE_ROWS, cell_id="SYN-TS-SHUFFLE")
    corrupted = corrupt_timestamps(clean, mode="shuffle", seed=1)
    faults = _replay(corrupted, "SYN-TS-SHUFFLE")
    kinds = {f["kind"] for f in faults}
    assert "OUT_OF_ORDER_TIMESTAMP" in kinds


def test_garbage_timestamps_trigger_unparseable_fault():
    """SYNTHETIC: a firmware bug emitting a non-ISO timestamp string."""
    clean = rows_to_clean_messages(_SOURCE_ROWS, cell_id="SYN-TS-GARBAGE")
    corrupted = corrupt_timestamps(clean, mode="garbage")
    faults = _replay(corrupted, "SYN-TS-GARBAGE")
    kinds = {f["kind"] for f in faults}
    assert "UNPARSEABLE_TIMESTAMP" in kinds
    # Every single message should trip it -- all of them are garbage.
    assert sum(1 for f in faults if f["kind"] == "UNPARSEABLE_TIMESTAMP") == len(clean)


def test_duplicate_timestamps_trigger_duplicate_fault():
    """SYNTHETIC: a stuck clock / duplicate publish."""
    clean = rows_to_clean_messages(_SOURCE_ROWS, cell_id="SYN-TS-DUP")
    corrupted = corrupt_timestamps(clean, mode="duplicate")
    faults = _replay(corrupted, "SYN-TS-DUP")
    kinds = {f["kind"] for f in faults}
    assert "DUPLICATE_TIMESTAMP" in kinds


def test_corrupted_units_trigger_implausible_capacity_fault():
    """SYNTHETIC: a real ~1.8-2.0 Ah NASA reading scaled x1000 (an Ah/mAh
    unit mixup) -- a case the pre-existing AnomalyDetector could never
    catch at all, since it never looks at capacity_ah."""
    clean = rows_to_clean_messages(_SOURCE_ROWS, cell_id="SYN-UNITS")
    corrupted = corrupt_units(clean, factor=1000.0)
    faults = _replay(corrupted, "SYN-UNITS")
    kinds = {f["kind"] for f in faults}
    assert "IMPLAUSIBLE_CAPACITY" in kinds
    assert sum(1 for f in faults if f["kind"] == "IMPLAUSIBLE_CAPACITY") == len(clean)


def test_dropped_packets_trigger_gap_fault():
    """SYNTHETIC: simulates real packet loss in transit -- removes ~20% of
    messages at random while preserving the remaining ones' original seq
    numbers, leaving gaps for validate_telemetry() to catch."""
    clean = rows_to_clean_messages(_SOURCE_ROWS, cell_id="SYN-DROP")
    corrupted = drop_packets(clean, fraction=0.3, seed=2)
    assert len(corrupted) < len(clean), "the corruption itself must actually drop something"

    faults = _replay(corrupted, "SYN-DROP")
    kinds = {f["kind"] for f in faults}
    assert "DROPPED_PACKET_GAP" in kinds


def test_combined_corruptions_do_not_crash_the_ingestion_pipeline():
    """Robustness proof: stacking every corruption at once (dropped packets
    + corrupted units + duplicated timestamps) must not raise, and must
    still surface multiple distinct fault kinds -- the ingestion layer
    degrades to a flagged, inspectable state rather than crashing or
    silently accepting garbage."""
    clean = rows_to_clean_messages(_SOURCE_ROWS, cell_id="SYN-COMBINED")
    corrupted = drop_packets(clean, fraction=0.2, seed=3)
    corrupted = corrupt_units(corrupted, factor=1000.0)
    corrupted = corrupt_timestamps(corrupted, mode="duplicate")

    faults = _replay(corrupted, "SYN-COMBINED")  # must not raise
    kinds = {f["kind"] for f in faults}
    assert "IMPLAUSIBLE_CAPACITY" in kinds
    assert "DUPLICATE_TIMESTAMP" in kinds
    assert "DROPPED_PACKET_GAP" in kinds


def test_non_dropped_messages_still_reach_the_telemetry_queue():
    """Faults are flagged, not used as an excuse to discard the message --
    validate_telemetry() runs alongside (not instead of) the existing
    telemetry/anomaly pipeline (see mqtt_stream._on_message())."""
    clean = rows_to_clean_messages(_SOURCE_ROWS, cell_id="SYN-STILL-FLOWS")
    corrupted = corrupt_units(clean, factor=1000.0)

    mqtt_stream._detectors.pop("SYN-STILL-FLOWS", None)
    mqtt_stream.drain_faults(10_000)
    mqtt_stream.drain_telemetry(10_000)
    for m in corrupted:
        mqtt_stream._on_message(None, {"chemistry_map": {}}, _fake_msg(m))

    telemetry = mqtt_stream.drain_telemetry(10_000)
    assert len(telemetry) == len(corrupted)
