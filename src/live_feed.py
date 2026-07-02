"""
Simulated BMS live feed.

In a production deployment this would subscribe to an MQTT broker
(topic: battery/telemetry/{cell_id}) via paho-mqtt. For the demo,
a background thread generates synthetic telemetry at configurable intervals.

Architecture:
  LiveFeedThread → _FEED_QUEUE → get_latest_readings()
  Streamlit page calls get_latest_readings() on each rerun to drain the queue.
"""

from __future__ import annotations
import threading
import queue
import time
import random
import datetime
from dataclasses import dataclass, field


_FEED_QUEUE: queue.Queue = queue.Queue(maxsize=500)
_FEED_THREAD: threading.Thread | None = None
_FEED_RUNNING = threading.Event()


@dataclass
class TelemetryReading:
    cell_id: str
    timestamp: str
    voltage_v: float
    current_a: float
    temp_c: float
    soh_estimate: float   # from onboard BMS algorithm (simplified)
    cycle_count: int
    source: str = "SIMULATED · demo feed"


def _generate_tick(cell_id: str, base_soh: float, cycle: int) -> TelemetryReading:
    """Generate one synthetic telemetry reading."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    # Add realistic noise
    voltage  = 3.6 + random.gauss(0, 0.05)    # V
    current  = -(2.0 + random.gauss(0, 0.1))  # A (discharge)
    temp     = 24.0 + random.gauss(0, 1.5)    # °C
    soh_est  = base_soh + random.gauss(0, 0.3)

    return TelemetryReading(
        cell_id=cell_id,
        timestamp=now,
        voltage_v=round(voltage, 3),
        current_a=round(current, 3),
        temp_c=round(temp, 2),
        soh_estimate=round(soh_est, 2),
        cycle_count=cycle,
    )


def start_feed(cell_ids: list[str], base_sohs: dict[str, float],
               base_cycles: dict[str, int], interval_s: float = 2.0) -> None:
    """Start the background feed thread (idempotent — safe to call multiple times)."""
    global _FEED_THREAD

    if _FEED_THREAD is not None and _FEED_THREAD.is_alive():
        return  # already running

    _FEED_RUNNING.set()

    def _run():
        tick = 0
        while _FEED_RUNNING.is_set():
            for cid in cell_ids:
                reading = _generate_tick(
                    cell_id=cid,
                    base_soh=base_sohs.get(cid, 85.0),
                    cycle=base_cycles.get(cid, 100) + tick // len(cell_ids),
                )
                try:
                    _FEED_QUEUE.put_nowait(reading)
                except queue.Full:
                    try:
                        _FEED_QUEUE.get_nowait()   # drop oldest
                        _FEED_QUEUE.put_nowait(reading)
                    except queue.Empty:
                        pass
            tick += 1
            time.sleep(interval_s)

    _FEED_THREAD = threading.Thread(target=_run, daemon=True, name="bms-live-feed")
    _FEED_THREAD.start()


def stop_feed() -> None:
    _FEED_RUNNING.clear()


def get_latest_readings(max_items: int = 100) -> list[TelemetryReading]:
    """Drain up to max_items readings from the queue (call on each Streamlit rerun)."""
    readings = []
    while len(readings) < max_items:
        try:
            readings.append(_FEED_QUEUE.get_nowait())
        except queue.Empty:
            break
    return readings


def is_running() -> bool:
    return _FEED_THREAD is not None and _FEED_THREAD.is_alive()
