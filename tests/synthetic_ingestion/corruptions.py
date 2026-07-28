"""
Pure functions: real cached NASA cycling data -> MQTT telemetry messages
-> deliberately corrupted messages. No network call, no live broker --
these build the same message dicts src/mqtt_stream.py's real publisher/
subscriber exchange, so they can be replayed straight into
mqtt_stream._on_message() (see test_fault_injection.py).

SYNTHETIC: every corrupt_*() function below injects a fault that does not
exist in the source data -- clearly labelled in each docstring. The
underlying cycling values (capacity/resistance/temperature/SOH) are real,
already-committed NASA PCoE measurements (batlab/datasets/nasa.py), not
fabricated; only the corruption itself is synthetic.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone


def load_source_cell_rows(cell_id: str = "B0005", n: int = 30) -> "list[dict]":
    """Real, already-cached NASA cycling data (no download -- reads the
    pre-committed data/raw/{cell_id}_summary.csv) for the first n cycles,
    as plain row dicts. Kept small (n=30 by default) since this harness
    only needs enough rows to exercise the ingestion fault checks, not a
    full cell history."""
    from batlab.datasets.nasa import load_nasa_cells

    df = load_nasa_cells(cell_ids=[cell_id])[cell_id]
    return df.head(n).to_dict("records")


def rows_to_clean_messages(rows: "list[dict]", cell_id: str) -> "list[dict]":
    """Shape real cycling rows into the app's real MQTT telemetry wire
    schema (see mqtt_stream.py's module docstring) -- monotonic ISO-8601
    timestamps one second apart, consecutive seq numbers, voltage
    synthesized from SOH the same way src/mqtt_stream.py's own replay
    publisher does. This is the CLEAN baseline every corrupt_*() function
    below is applied on top of."""
    base_ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    messages = []
    for i, row in enumerate(rows):
        soh = float(row.get("soh_pct", 100.0))
        messages.append({
            "cell_id":       cell_id,
            "cycle":         int(row.get("cycle_number", i + 1)),
            "seq":           i,
            "ts":            (base_ts + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
            "voltage_v":     round(3.0 + (soh / 100.0) * 1.2, 4),
            "current_a":     -2.0,
            "temperature_c": float(row.get("temperature_c", 24.0)),
            "capacity_ah":   float(row.get("capacity_ah", 2.0)),
            "soh_pct":       soh,
        })
    return messages


def corrupt_timestamps(messages: "list[dict]", mode: str = "shuffle", seed: int = 0) -> "list[dict]":
    """
    SYNTHETIC corruption: deliberately breaks the ts field.

    mode="shuffle":   reorders messages so ts arrives out of sequence
                       (simulates network reordering / clock skew).
    mode="garbage":   replaces ts with an unparseable string (simulates a
                       firmware bug emitting a non-ISO timestamp).
    mode="duplicate": repeats the previous message's exact ts (simulates a
                       stuck clock / duplicate publish).
    """
    messages = [dict(m) for m in messages]  # don't mutate the caller's list
    if mode == "shuffle":
        rng = random.Random(seed)
        rng.shuffle(messages)
    elif mode == "garbage":
        for m in messages:
            m["ts"] = "not-a-real-timestamp"
    elif mode == "duplicate":
        for i in range(1, len(messages)):
            messages[i]["ts"] = messages[i - 1]["ts"]
    else:
        raise ValueError(f"Unknown mode {mode!r} -- must be 'shuffle', 'garbage', or 'duplicate'.")
    return messages


def corrupt_units(messages: "list[dict]", factor: float = 1000.0) -> "list[dict]":
    """SYNTHETIC corruption: multiplies capacity_ah by `factor` (default
    1000x), simulating a real-world unit mixup -- a device or gateway
    reporting milliamp-hours into a field the schema expects in
    amp-hours (e.g. a real ~1.8 Ah reading becomes 1800)."""
    messages = [dict(m) for m in messages]
    for m in messages:
        if m.get("capacity_ah") is not None:
            m["capacity_ah"] = m["capacity_ah"] * factor
    return messages


def drop_packets(messages: "list[dict]", fraction: float = 0.2, seed: int = 0) -> "list[dict]":
    """SYNTHETIC corruption: removes a random fraction of messages
    (default 20%) while preserving every remaining message's original seq
    number -- simulates real packet loss in transit, leaving gaps in the
    seq sequence for validate_telemetry()'s DROPPED_PACKET_GAP check to
    catch. Always keeps the first message so seq gap-tracking has a
    baseline to compare against."""
    rng = random.Random(seed)
    kept = [messages[0]] + [
        m for m in messages[1:] if rng.random() >= fraction
    ]
    return kept
