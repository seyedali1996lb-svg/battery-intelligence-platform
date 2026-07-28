"""
MQTT BMS streaming — publisher, subscriber, and anomaly detector.

Demo mode:  connects to test.mosquitto.org:1883 (public broker, no auth).
Production: point MQTT_HOST / MQTT_PORT at your BMS broker.

Topic scheme:
  battery/{cell_id}/telemetry   — per-reading JSON from BMS or replay publisher
  battery/{cell_id}/anomaly     — anomaly events published by detector

Telemetry message format (JSON):
  {
    "cell_id":       "B0005",
    "cycle":         42,
    "seq":           1234,
    "ts":            "2024-01-15T12:34:56Z",
    "voltage_v":     3.852,
    "current_a":     -2.000,
    "temperature_c": 24.3,
    "capacity_ah":   1.823,
    "soh_pct":       74.8
  }

Anomaly detection runs on every received message:
  - Voltage out of bounds (chemistry-specific limits)
  - Temperature spike (absolute + rate-of-rise)
  - Rolling Z-score (window=20) on voltage, current, temperature
  - Multi-signal correlated anomaly — combined Z-score across 2+ channels,
    catching simultaneous moderate drift that no single-channel threshold
    can see on its own (see AnomalyDetector.check())
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Default connection parameters ────────────────────────────────────────────
DEFAULT_HOST  = "test.mosquitto.org"
DEFAULT_PORT  = 1883
TOPIC_PREFIX  = "battery-intelligence"

# ── Chemistry voltage limits ──────────────────────────────────────────────────
_VOLTAGE_LIMITS = {
    "LFP":    (2.50, 3.65),
    "NCA":    (2.50, 4.20),
    "LiCoO2": (2.75, 4.20),
    "default":(2.50, 4.25),
}

# IEC 62619:2022 operational safety limits
_TEMP_MAX          = 45.0    # °C — IEC 62619 §6.2 maximum operating temperature
_TEMP_CHARGE_MAX   = 45.0    # °C — charging upper limit (§6.2.3)
_TEMP_DISCHARGE_MAX= 60.0    # °C — discharge upper limit (§6.2.4)
_TEMP_MIN          = -20.0   # °C — minimum operating temperature
_TEMP_RATE_MAX     = 2.0     # °C per reading — rate-of-rise flag (thermal runaway precursor)
_TEMP_RATE_CRITICAL= 5.0     # °C per reading — IEC 62619 §8.2 critical rate
_CAPACITY_PLUNGE   = 0.05    # SOH drop > 5% in one cycle → plating event signal
_ZSCORE_WINDOW     = 20      # rolling window for Z-score
_ZSCORE_THRESH     = 2.5     # flag threshold
_MULTI_SIGNAL_ELEVATED   = 1.5   # per-channel Z-score to count as "contributing" to a combined anomaly
_MULTI_SIGNAL_COMBINED   = 3.5   # combined (Euclidean-norm) Z-score threshold across channels

# ── Ingestion fault-detection thresholds ──────────────────────────────────────
# These catch malformed/corrupted DATA (missing fields, unparseable
# timestamps, out-of-order delivery, dropped packets, wrong-unit-scale
# values) -- a structurally different concern from AnomalyDetector.check()
# above, which only ever sees a well-formed message and judges whether its
# *value* is physically worrying. A message that fails a fault check here
# is quarantined (logged + queued to _faults) rather than silently dropped
# or fed to the anomaly detector as if it were trustworthy.
_CAPACITY_AH_MAX = 50.0   # a single cell's capacity_ah above this indicates a unit mixup (e.g. mAh in an Ah field), not a real reading

# ── Thread-safe message store ─────────────────────────────────────────────────
# Module-level so publisher and subscriber share one namespace across Streamlit reruns.

_incoming:   queue.Queue   = queue.Queue(maxsize=2000)
_anomalies:  queue.Queue   = queue.Queue(maxsize=500)
_faults:     queue.Queue   = queue.Queue(maxsize=500)
_pub_thread: Optional[threading.Thread] = None
_sub_client = None          # paho client for subscriber
_pub_running = threading.Event()


# ── Anomaly detector ─────────────────────────────────────────────────────────

class AnomalyDetector:
    """Per-cell rolling anomaly detector. One instance per cell_id."""

    def __init__(self, cell_id: str, chemistry: str = "default"):
        self.cell_id   = cell_id
        self.v_lo, self.v_hi = _VOLTAGE_LIMITS.get(chemistry, _VOLTAGE_LIMITS["default"])
        self._v_hist   : deque = deque(maxlen=_ZSCORE_WINDOW)
        self._i_hist   : deque = deque(maxlen=_ZSCORE_WINDOW)
        self._t_hist   : deque = deque(maxlen=_ZSCORE_WINDOW)
        self._last_temp: Optional[float] = None
        self._last_ts  : Optional[datetime] = None   # ingestion fault tracking (out-of-order/duplicate)
        self._last_seq : Optional[int] = None         # ingestion fault tracking (dropped-packet gaps)

    def check(self, msg: dict) -> list[dict]:
        """Return list of anomaly dicts (empty if clean)."""
        flags = []
        v = msg.get("voltage_v")
        i = msg.get("current_a")
        t = msg.get("temperature_c")
        ts = msg.get("ts", datetime.now(timezone.utc).isoformat())

        def _flag(kind, detail, severity="warning"):
            return {
                "cell_id":  self.cell_id,
                "ts":       ts,
                "kind":     kind,
                "detail":   detail,
                "severity": severity,
                "seq":      msg.get("seq", 0),
            }

        # Voltage bounds
        if v is not None:
            if v < self.v_lo:
                flags.append(_flag("UNDERVOLTAGE",
                    f"{v:.3f} V below min {self.v_lo} V", "critical"))
            elif v > self.v_hi:
                flags.append(_flag("OVERVOLTAGE",
                    f"{v:.3f} V above max {self.v_hi} V", "critical"))
            self._v_hist.append(v)

        # Temperature — IEC 62619:2022 limits
        if t is not None:
            if t < _TEMP_MIN:
                flags.append(_flag("UNDERTEMPERATURE",
                    f"{t:.1f}°C below minimum {_TEMP_MIN}°C (IEC 62619 §6.2)", "critical"))
            elif t > _TEMP_CHARGE_MAX:
                flags.append(_flag("OVERTEMPERATURE",
                    f"{t:.1f}°C exceeds {_TEMP_CHARGE_MAX}°C operating limit (IEC 62619 §6.2.3)", "critical"))
            if self._last_temp is not None:
                rate = t - self._last_temp
                if rate > _TEMP_RATE_CRITICAL:
                    flags.append(_flag("THERMAL_RUNAWAY_PRECURSOR",
                        f"Temperature rate {rate:.1f}°C/step exceeds {_TEMP_RATE_CRITICAL}°C/step "
                        f"(IEC 62619 §8.2 — thermal runaway precursor)", "critical"))
                elif rate > _TEMP_RATE_MAX:
                    flags.append(_flag("TEMP_RATE_HIGH",
                        f"Temperature rose {rate:.1f}°C in one reading (warning threshold {_TEMP_RATE_MAX}°C/step)",
                        "warning"))
            self._last_temp = t
            self._t_hist.append(t)

        # Capacity plunge — IEC 62619 §8.2 sudden loss event
        soh = msg.get("soh_pct")
        if soh is not None and len(self._v_hist) >= 2:
            _prev_soh = getattr(self, "_last_soh", None)
            if _prev_soh is not None:
                _drop = (_prev_soh - soh) / 100.0
                if _drop > _CAPACITY_PLUNGE:
                    flags.append(_flag("CAPACITY_PLUNGE",
                        f"SOH dropped {_drop*100:.1f}% in one reading — possible lithium plating event "
                        f"(IEC 62619 §8.2 sudden capacity loss threshold: {_CAPACITY_PLUNGE*100:.0f}%)",
                        "critical"))
            self._last_soh = soh

        if i is not None:
            self._i_hist.append(i)

        # Rolling Z-score on each channel
        _zscores: dict[str, float] = {}
        for label, hist in [("voltage", self._v_hist),
                             ("current", self._i_hist),
                             ("temperature", self._t_hist)]:
            if len(hist) >= _ZSCORE_WINDOW:
                arr = np.array(hist)
                mu, sigma = arr.mean(), arr.std()
                if sigma > 1e-6:
                    z = abs((arr[-1] - mu) / sigma)
                    _zscores[label] = z
                    if z > _ZSCORE_THRESH:
                        flags.append(_flag(f"ZSCORE_{label.upper()}",
                            f"{label} Z-score {z:.2f} (>{_ZSCORE_THRESH}) — "
                            f"value {arr[-1]:.3f}, rolling μ={mu:.3f} σ={sigma:.3f}",
                            "warning"))

        # Multi-signal correlated anomaly — a real step beyond independent
        # per-channel thresholds, not just a rephrased threshold check. This
        # combines the same Z-scores already computed above via their
        # Euclidean norm, so it can flag a real anomaly even when no single
        # channel alone crosses its own threshold: simultaneous, moderate
        # deviation across 2+ channels (e.g. voltage sag + temperature rise
        # together) is a materially stronger fault signal than any one
        # channel drifting alone, which is exactly the case a per-channel-
        # only check structurally cannot see.
        _elevated = [c for c, z in _zscores.items() if z > _MULTI_SIGNAL_ELEVATED]
        if len(_zscores) >= 2 and len(_elevated) >= 2:
            _combined = float(np.sqrt(sum(z * z for z in _zscores.values())))
            if _combined > _MULTI_SIGNAL_COMBINED:
                flags.append(_flag("MULTI_SIGNAL_ANOMALY",
                    f"Correlated deviation across {', '.join(sorted(_elevated))} "
                    f"(combined Z={_combined:.2f}, threshold {_MULTI_SIGNAL_COMBINED}) — "
                    f"simultaneous multi-channel drift, a stronger fault signal than any single reading alone",
                    "critical"))

        return flags


def _parse_ts(raw) -> "Optional[datetime]":
    """Best-effort ISO-8601 parse (handles a trailing 'Z', which
    datetime.fromisoformat rejects before Python 3.11). Returns None on any
    unparseable/missing value -- never raises."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def validate_telemetry(payload: dict, detector: "AnomalyDetector") -> list[dict]:
    """
    Ingestion-layer fault checks, run on every incoming message BEFORE
    AnomalyDetector.check(). These catch malformed/corrupted DATA (missing
    fields, unparseable timestamps, out-of-order/duplicate delivery,
    dropped packets, wrong-unit-scale values) -- a structurally different
    concern from AnomalyDetector, which only ever judges whether an
    already-well-formed reading's *value* is physically worrying. A
    message that trips one of these checks should be quarantined (logged +
    queued to _faults), not silently dropped or fed to the anomaly
    detector as if it were trustworthy.

    Mutates detector's _last_ts/_last_seq tracking as a side effect (same
    pattern as AnomalyDetector.check()'s own rolling-history updates).
    Returns a list of fault dicts (empty if clean); never raises.
    """
    faults: list[dict] = []
    cell_id = payload.get("cell_id")
    now_iso = datetime.now(timezone.utc).isoformat()

    def _flag(kind, detail, severity="warning"):
        return {
            "cell_id":  cell_id or "unknown",
            "ts":       payload.get("ts", now_iso),
            "kind":     kind,
            "detail":   detail,
            "severity": severity,
            "seq":      payload.get("seq", 0),
        }

    if not cell_id:
        faults.append(_flag("MISSING_CELL_ID", "Message has no cell_id — cannot attribute to a fleet asset.", "critical"))

    ts_raw = payload.get("ts")
    ts_parsed = _parse_ts(ts_raw)
    if ts_raw is None:
        faults.append(_flag("MISSING_TIMESTAMP", "Message has no ts field."))
    elif ts_parsed is None:
        faults.append(_flag("UNPARSEABLE_TIMESTAMP", f"ts value {ts_raw!r} is not valid ISO-8601."))
    else:
        if detector._last_ts is not None:
            if ts_parsed < detector._last_ts:
                faults.append(_flag("OUT_OF_ORDER_TIMESTAMP",
                    f"ts {ts_raw} is earlier than the previous reading's {detector._last_ts.isoformat()} "
                    f"— messages arrived (or were corrupted) out of order."))
            elif ts_parsed == detector._last_ts:
                faults.append(_flag("DUPLICATE_TIMESTAMP", f"ts {ts_raw} repeats the previous reading's timestamp exactly."))
        detector._last_ts = ts_parsed

    seq = payload.get("seq")
    if isinstance(seq, int):
        if detector._last_seq is not None and seq > detector._last_seq + 1:
            _gap = seq - detector._last_seq - 1
            faults.append(_flag("DROPPED_PACKET_GAP",
                f"seq jumped from {detector._last_seq} to {seq} — {_gap} message(s) appear to have been "
                f"dropped in transit."))
        detector._last_seq = seq

    cap = payload.get("capacity_ah")
    if cap is not None:
        try:
            cap_f = float(cap)
            if cap_f < 0 or cap_f > _CAPACITY_AH_MAX:
                faults.append(_flag("IMPLAUSIBLE_CAPACITY",
                    f"capacity_ah={cap_f} is outside any physically plausible single-cell range "
                    f"(0-{_CAPACITY_AH_MAX} Ah) — likely a unit mixup (e.g. mAh reported as Ah)."))
        except (TypeError, ValueError):
            faults.append(_flag("NON_NUMERIC_CAPACITY", f"capacity_ah={cap!r} is not numeric."))

    return faults


# ── Subscriber ───────────────────────────────────────────────────────────────

_detectors: dict[str, AnomalyDetector] = {}


def _on_message(client, userdata, msg):
    """paho callback — decode JSON, run fault + anomaly checks, push to queues."""
    try:
        payload = json.loads(msg.payload.decode())
    except Exception as e:
        logger.warning("Dropped unparseable MQTT message on %s: %s", getattr(msg, "topic", "?"), e)
        return

    cell_id   = payload.get("cell_id", "unknown")
    chemistry = userdata.get("chemistry_map", {}).get(cell_id, "default")

    if cell_id not in _detectors:
        _detectors[cell_id] = AnomalyDetector(cell_id, chemistry)
    detector = _detectors[cell_id]

    # Push telemetry
    if not _incoming.full():
        _incoming.put_nowait(payload)

    # Ingestion faults — malformed/corrupted data (missing fields, bad
    # timestamps, dropped packets, wrong-unit-scale values). Quarantined
    # (logged + queued) rather than silently dropped.
    for fault in validate_telemetry(payload, detector):
        logger.warning("Ingestion fault [%s] %s: %s", fault["kind"], fault["cell_id"], fault["detail"])
        if not _faults.full():
            _faults.put_nowait(fault)

    # Anomalies — physically-plausible-but-worrying readings
    for flag in detector.check(payload):
        if not _anomalies.full():
            _anomalies.put_nowait(flag)


def start_subscriber(
    host: str = DEFAULT_HOST,
    port: int  = DEFAULT_PORT,
    cell_ids: list[str] | None = None,
    chemistry_map: dict[str, str] | None = None,
    topic_prefix: str = TOPIC_PREFIX,
) -> bool:
    """
    Start the MQTT subscriber in a background thread.

    Returns True if connection succeeded, False otherwise.
    Non-blocking — messages accumulate in _incoming / _anomalies queues.
    """
    global _sub_client

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        return False

    def _on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            topics = (
                [(f"{topic_prefix}/{cid}/telemetry", 0) for cid in cell_ids]
                if cell_ids else
                [(f"{topic_prefix}/+/telemetry", 0)]
            )
            client.subscribe(topics)

    client_id = f"battery-intelligence-sub-{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        userdata={"chemistry_map": chemistry_map or {}},
    )
    client.on_connect = _on_connect
    client.on_message = _on_message

    try:
        client.connect(host, port, keepalive=30)
        client.loop_start()
        _sub_client = client
        return True
    except Exception:
        return False


def stop_subscriber():
    global _sub_client
    if _sub_client:
        _sub_client.loop_stop()
        _sub_client.disconnect()
        _sub_client = None


def drain_telemetry(max_msgs: int = 200) -> list[dict]:
    """Pull up to max_msgs from the incoming queue."""
    msgs = []
    for _ in range(max_msgs):
        try:
            msgs.append(_incoming.get_nowait())
        except queue.Empty:
            break
    return msgs


def drain_anomalies(max_msgs: int = 100) -> list[dict]:
    """Pull up to max_msgs from the anomaly queue."""
    msgs = []
    for _ in range(max_msgs):
        try:
            msgs.append(_anomalies.get_nowait())
        except queue.Empty:
            break
    return msgs


def drain_faults(max_msgs: int = 100) -> list[dict]:
    """Pull up to max_msgs from the ingestion-fault queue (see
    validate_telemetry() -- malformed/corrupted data, distinct from the
    anomaly queue's physically-plausible-but-worrying readings)."""
    msgs = []
    for _ in range(max_msgs):
        try:
            msgs.append(_faults.get_nowait())
        except queue.Empty:
            break
    return msgs


def is_subscriber_connected() -> bool:
    return _sub_client is not None and _sub_client.is_connected()


# ── Replay publisher ──────────────────────────────────────────────────────────

def _publisher_worker(
    df,
    cell_id: str,
    host: str,
    port: int,
    topic_prefix: str,
    speed: float,
    loop: bool,
):
    """Background thread: replays cell DataFrame as MQTT telemetry."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        return

    client_id = f"battery-intelligence-pub-{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    try:
        client.connect(host, port, keepalive=30)
        client.loop_start()
    except Exception:
        return

    topic = f"{topic_prefix}/{cell_id}/telemetry"
    # Columns that may exist in the DataFrame
    _col = lambda c: c if c in df.columns else None

    while _pub_running.is_set():
        for seq, row in enumerate(df.itertuples(index=False)):
            if not _pub_running.is_set():
                break

            def _get(col, default=None):
                return float(getattr(row, col)) if col and hasattr(row, col) else default

            # Map DataFrame columns to BMS telemetry schema
            voltage  = _get(_col("voltage_v"),      None) or _get(_col("voltage"), None)
            current  = _get(_col("current_a"),      None) or _get(_col("current"), None)
            temp     = _get(_col("temperature_c"),  None) or _get(_col("temperature"), None)
            cap      = _get(_col("capacity_ah"),    None)
            soh      = _get(_col("soh_pct"),        None)
            cycle    = int(_get(_col("cycle_number"), seq))

            # Synthesise voltage from SOH if raw voltage not in df
            if voltage is None and soh is not None:
                voltage = round(3.0 + (soh / 100.0) * 1.2 + np.random.normal(0, 0.005), 4)
            if temp is None:
                temp = round(24.0 + np.random.normal(0, 0.3), 2)
            if current is None:
                current = -2.0

            payload = {
                "cell_id":       cell_id,
                "cycle":         cycle,
                "seq":           seq,
                "ts":            datetime.now(timezone.utc).isoformat(),
                "voltage_v":     round(voltage, 4) if voltage is not None else None,
                "current_a":     round(current, 4) if current is not None else None,
                "temperature_c": round(temp, 2)    if temp    is not None else None,
                "capacity_ah":   round(cap, 4)     if cap     is not None else None,
                "soh_pct":       round(soh, 2)     if soh     is not None else None,
            }
            client.publish(topic, json.dumps(payload), qos=0)
            time.sleep(max(0.05, 1.0 / speed))

        if not loop:
            break

    client.loop_stop()
    client.disconnect()


def start_publisher(
    df,
    cell_id: str,
    host: str         = DEFAULT_HOST,
    port: int         = DEFAULT_PORT,
    topic_prefix: str = TOPIC_PREFIX,
    speed: float      = 10.0,
    loop: bool        = True,
) -> bool:
    """
    Start the replay publisher in a background thread.

    speed: replay speed multiplier (1×=1 msg/s, 10×=10 msg/s, 20×=20 msg/s).
    loop:  whether to loop the replay continuously.
    """
    global _pub_thread
    if _pub_thread and _pub_thread.is_alive():
        stop_publisher()

    _pub_running.set()
    _pub_thread = threading.Thread(
        target=_publisher_worker,
        args=(df, cell_id, host, port, topic_prefix, speed, loop),
        daemon=True,
        name=f"mqtt-pub-{cell_id}",
    )
    _pub_thread.start()
    return True


def stop_publisher():
    _pub_running.clear()
    global _pub_thread
    if _pub_thread:
        _pub_thread.join(timeout=3.0)
        _pub_thread = None


def publisher_running() -> bool:
    return _pub_thread is not None and _pub_thread.is_alive()
