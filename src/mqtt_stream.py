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
    "soc_pct":       74.8
  }

Anomaly detection runs on every received message:
  - Voltage out of bounds (chemistry-specific limits)
  - Temperature spike (absolute + rate-of-rise)
  - Rolling Z-score (window=20) on voltage, current, temperature
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

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

_TEMP_MAX        = 45.0    # °C — absolute limit
_TEMP_RATE_MAX   = 2.0     # °C per reading — rate-of-rise flag
_ZSCORE_WINDOW   = 20      # rolling window for Z-score
_ZSCORE_THRESH   = 2.5     # flag threshold

# ── Thread-safe message store ─────────────────────────────────────────────────
# Module-level so publisher and subscriber share one namespace across Streamlit reruns.

_incoming:   queue.Queue   = queue.Queue(maxsize=2000)
_anomalies:  queue.Queue   = queue.Queue(maxsize=500)
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

        # Temperature absolute + rate-of-rise
        if t is not None:
            if t > _TEMP_MAX:
                flags.append(_flag("OVERTEMPERATURE",
                    f"{t:.1f}°C exceeds {_TEMP_MAX}°C limit", "critical"))
            if self._last_temp is not None:
                rate = t - self._last_temp
                if rate > _TEMP_RATE_MAX:
                    flags.append(_flag("TEMP_SPIKE",
                        f"Temperature rose {rate:.1f}°C in one reading (limit {_TEMP_RATE_MAX}°C/step)",
                        "warning"))
            self._last_temp = t
            self._t_hist.append(t)

        if i is not None:
            self._i_hist.append(i)

        # Rolling Z-score on each channel
        for label, hist in [("voltage", self._v_hist),
                             ("current", self._i_hist),
                             ("temperature", self._t_hist)]:
            if len(hist) >= _ZSCORE_WINDOW:
                arr = np.array(hist)
                mu, sigma = arr.mean(), arr.std()
                if sigma > 1e-6:
                    z = abs((arr[-1] - mu) / sigma)
                    if z > _ZSCORE_THRESH:
                        flags.append(_flag(f"ZSCORE_{label.upper()}",
                            f"{label} Z-score {z:.2f} (>{_ZSCORE_THRESH}) — "
                            f"value {arr[-1]:.3f}, rolling μ={mu:.3f} σ={sigma:.3f}",
                            "warning"))

        return flags


# ── Subscriber ───────────────────────────────────────────────────────────────

_detectors: dict[str, AnomalyDetector] = {}


def _on_message(client, userdata, msg):
    """paho callback — decode JSON, run anomaly check, push to queues."""
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        return

    cell_id   = payload.get("cell_id", "unknown")
    chemistry = userdata.get("chemistry_map", {}).get(cell_id, "default")

    if cell_id not in _detectors:
        _detectors[cell_id] = AnomalyDetector(cell_id, chemistry)

    # Push telemetry
    if not _incoming.full():
        _incoming.put_nowait(payload)

    # Push anomalies
    for flag in _detectors[cell_id].check(payload):
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
                "soc_pct":       round(soh, 2)     if soh     is not None else None,
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
