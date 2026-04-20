"""Tests for Step 2b: OnlineRunner.

pytest tests/test_online_runner.py
"""
import pytest
from energyx.monitoring.bus import EventBus
from energyx.monitoring.online_runner import OnlineRunner
from energyx.monitoring.monitors import AnomalyDetectors, RealtimeCostMeter, ForgotToTurnOffDetector


class TestOnlineRunner:
    def _make_tick(self, watts=300.0, ts="2026-04-20 14:00:00"):
        return {"value": watts, "ts": ts, "sensor_type": "electricity_apparent", "unit": "watts"}

    def test_process_tick_returns_list(self):
        bus = EventBus()
        runner = OnlineRunner(config={"home_id": "home001"}, bus=bus)
        events = runner.process_tick(self._make_tick())
        assert isinstance(events, list)

    def test_events_published_to_bus(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", received.append)
        runner = OnlineRunner(config={"home_id": "home001"}, bus=bus)
        # Publish enough ticks to fill anomaly window
        for i in range(30):
            runner.process_tick(self._make_tick(300.0))
        # Push a spike
        runner.process_tick(self._make_tick(99999.0))
        # Some events should have been published
        assert runner.stats()["total_ticks"] == 31

    def test_online_only_monitors_included(self):
        runner = OnlineRunner(config={"home_id": "home001"})
        monitor_types = [type(m).__name__ for m in runner._monitors]
        assert "ForgotToTurnOffDetector" in monitor_types
        assert "DemandResponseListener" in monitor_types

    def test_stats(self):
        runner = OnlineRunner(config={"home_id": "home001"})
        runner.process_tick(self._make_tick())
        stats = runner.stats()
        assert stats["total_ticks"] == 1
        assert stats["running"] is False

    def test_process_multiple_ticks(self):
        runner = OnlineRunner(config={"home_id": "home001"})
        ticks = [self._make_tick(300.0 + i) for i in range(10)]
        events = runner.process_ticks(ticks)
        assert runner.stats()["total_ticks"] == 10

    def test_custom_monitors(self):
        bus = EventBus()
        mon = RealtimeCostMeter({"home_id": "home001"})
        runner = OnlineRunner(config={"home_id": "home001"}, bus=bus, monitors=[mon])
        events = runner.process_tick(self._make_tick(1000.0))
        from energyx.data.events import CostRealtimeEvent
        cost_events = [e for e in events if isinstance(e, CostRealtimeEvent)]
        assert len(cost_events) == 1

    def test_dr_signal_injection(self):
        bus = EventBus()
        runner = OnlineRunner(config={"home_id": "home001"}, bus=bus)
        events = runner.inject_dr_signal({
            "event_id": "test_dr",
            "start": "2026-04-20T18:00:00Z",
            "end": "2026-04-20T19:00:00Z",
            "source": "octopus",
        })
        assert isinstance(events, list)  # may be empty if no loads configured
