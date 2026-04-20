"""OnlineRunner — pure Python continuous monitoring loop.

NO LLM calls anywhere in this file.  This is the hot path — kept cheap
and predictable.  Iterates all monitors whose applicable_modes includes
"online" and calls evaluate(tick) on each, then publishes events to the bus.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from typing import Any, Callable, Dict, List, Optional

from energyx.data.events import BaseEvent
from energyx.monitoring.bus import EventBus, get_event_bus
from energyx.monitoring.monitors.base import BaseMonitor
from energyx.monitoring.monitors import (
    AnomalyDetectors,
    ApplianceBaselineWatcher,
    HabitDriftTracker,
    CarbonTracker,
    RealtimeCostMeter,
    BudgetTrajectoryTracker,
    HistoricStoreWriter,
    ForgotToTurnOffDetector,
    DemandResponseListener,
)

logger = logging.getLogger(__name__)


def _build_online_monitors(config: Dict[str, Any]) -> List[BaseMonitor]:
    """Instantiate all monitors that support online mode."""
    home_id = config.get("home_id", "unknown")
    base_cfg = {"home_id": home_id}

    monitors: List[BaseMonitor] = [
        AnomalyDetectors({**base_cfg, **config.get("anomaly", {})}),
        ApplianceBaselineWatcher({**base_cfg, **config.get("baseline", {})}),
        HabitDriftTracker({**base_cfg, **config.get("habit_drift", {})}),
        CarbonTracker({**base_cfg, **config.get("carbon", {})}),
        RealtimeCostMeter({**base_cfg, **config.get("cost", {})}),
        BudgetTrajectoryTracker({**base_cfg, **config.get("budget", {})}),
        HistoricStoreWriter({**base_cfg, **config.get("store", {})}),
        ForgotToTurnOffDetector({**base_cfg, **config.get("forgot", {})}),
        DemandResponseListener({**base_cfg, **config.get("dr", {})}),
    ]
    # Only return monitors that support online mode
    return [m for m in monitors if "online" in m.applicable_modes]


class OnlineRunner:
    """Consumes a live tick stream; evaluates all online monitors per tick.

    Usage (pull mode)::
        runner = OnlineRunner(config={"home_id": "home001"})
        runner.start()
        # Push individual ticks:
        runner.push_tick({"ts": "...", "value": 350.0, "sensor_type": "electricity_apparent"})
        runner.stop()

    Usage (iterator mode)::
        runner = OnlineRunner(config={"home_id": "home001"})
        for tick in my_tick_source:
            events = runner.process_tick(tick)   # synchronous, no threads

    Events are published to the shared EventBus AND returned from process_tick.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        bus: Optional[EventBus] = None,
        monitors: Optional[List[BaseMonitor]] = None,
    ):
        self._config = config or {}
        self._bus = bus or get_event_bus()
        self._monitors = monitors if monitors is not None else _build_online_monitors(self._config)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tick_queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total_ticks = 0
        self._total_events = 0

    # ------------------------------------------------------------------
    # Synchronous API (single-threaded / test-friendly)
    # ------------------------------------------------------------------

    def process_tick(self, tick: Dict[str, Any]) -> List[BaseEvent]:
        """Process a single tick synchronously. Returns all emitted events."""
        events: List[BaseEvent] = []
        for monitor in self._monitors:
            try:
                result = monitor.evaluate(tick)
                if result:
                    events.extend(result)
            except Exception as e:
                logger.error(f"Monitor {type(monitor).__name__} failed on tick: {e}")
        for event in events:
            self._bus.publish(event)
        self._total_ticks += 1
        self._total_events += len(events)
        return events

    def process_ticks(self, ticks: List[Dict[str, Any]]) -> List[BaseEvent]:
        """Process a list of ticks. Returns all emitted events."""
        all_events: List[BaseEvent] = []
        for tick in ticks:
            all_events.extend(self.process_tick(tick))
        return all_events

    # ------------------------------------------------------------------
    # Threaded API (background loop consuming pushed ticks)
    # ------------------------------------------------------------------

    def start(self):
        """Start the background processing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="OnlineRunner")
        self._thread.start()
        logger.info("OnlineRunner started")

    def stop(self, flush_timeout: float = 5.0):
        """Signal the runner to stop; flush remaining ticks."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=flush_timeout)
        # Flush any remaining
        with self._lock:
            remaining = list(self._tick_queue)
            self._tick_queue.clear()
        for tick in remaining:
            self.process_tick(tick)
        # Flush HistoricStoreWriter buffer
        for m in self._monitors:
            if isinstance(m, HistoricStoreWriter):
                m.flush()
        logger.info(f"OnlineRunner stopped. Processed {self._total_ticks} ticks, {self._total_events} events.")

    def push_tick(self, tick: Dict[str, Any]):
        """Thread-safe: enqueue a tick for the background loop."""
        with self._lock:
            self._tick_queue.append(tick)

    def inject_dr_signal(self, signal: Dict[str, Any]) -> List[BaseEvent]:
        """Inject a DR signal into the DemandResponseListener."""
        events: List[BaseEvent] = []
        for m in self._monitors:
            if isinstance(m, DemandResponseListener):
                result = m.inject_signal(signal)
                events.extend(result)
                for e in result:
                    self._bus.publish(e)
        return events

    def stats(self) -> Dict[str, Any]:
        return {
            "total_ticks": self._total_ticks,
            "total_events": self._total_events,
            "queue_size": len(self._tick_queue),
            "running": self._running,
            "monitor_count": len(self._monitors),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self):
        while self._running:
            with self._lock:
                batch = list(self._tick_queue)
                self._tick_queue.clear()
            for tick in batch:
                self.process_tick(tick)
            if not batch:
                _time.sleep(0.01)  # 10ms idle sleep — keeps CPU free
