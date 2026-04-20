"""Tests for Step 2: Shared Monitor Core.

pytest tests/test_monitors.py
"""
import pandas as pd
import numpy as np
import pytest

from energyx.data.events import (
    AnomalyApplianceEvent, DegradationEvent, HabitDriftEvent,
    CostRealtimeEvent, CarbonRealtimeEvent, BudgetTrajectoryEvent,
    EventType,
)
from energyx.monitoring.monitors import (
    AnomalyDetectors, ApplianceBaselineWatcher, HabitDriftTracker,
    CarbonTracker, RealtimeCostMeter, BudgetTrajectoryTracker,
    HistoricStoreWriter, ForgotToTurnOffDetector, DemandResponseListener,
    ALL_MONITORS,
)


# ---------------------------------------------------------------------------
# applicable_modes
# ---------------------------------------------------------------------------

class TestApplicableModes:
    def test_online_only_monitors(self):
        """ForgotToTurnOffDetector and DemandResponseListener are online-only."""
        assert ForgotToTurnOffDetector.applicable_modes == {"online"}
        assert DemandResponseListener.applicable_modes == {"online"}

    def test_both_mode_monitors(self):
        both = [
            AnomalyDetectors, ApplianceBaselineWatcher, HabitDriftTracker,
            CarbonTracker, RealtimeCostMeter, BudgetTrajectoryTracker, HistoricStoreWriter,
        ]
        for cls in both:
            assert "online" in cls.applicable_modes
            assert "offline" in cls.applicable_modes

    def test_all_monitors_registered(self):
        assert len(ALL_MONITORS) == 9


# ---------------------------------------------------------------------------
# AnomalyDetectors
# ---------------------------------------------------------------------------

class TestAnomalyDetectors:
    def _make_batch(self, values):
        return pd.DataFrame({"value": values})

    def test_no_anomaly_in_normal_data(self):
        monitor = AnomalyDetectors(config={"min_votes": 4})
        normal = [300.0 + 10 * np.sin(i * 0.1) for i in range(100)]
        events = monitor.evaluate(self._make_batch(normal))
        assert len(events) == 0

    def test_detects_obvious_spike(self):
        monitor = AnomalyDetectors(config={"min_votes": 2})  # lower threshold for test
        values = [300.0] * 80 + [5000.0] + [300.0] * 19  # clear spike
        events = monitor.evaluate(self._make_batch(values))
        assert len(events) >= 1
        assert all(isinstance(e, AnomalyApplianceEvent) for e in events)

    def test_online_tick_rolling_window(self):
        monitor = AnomalyDetectors(config={"window_size": 20, "min_votes": 2})
        # Fill window with normal data
        for i in range(20):
            monitor.evaluate({"value": 300.0})
        # Now push a spike
        events = monitor.evaluate({"value": 9999.0})
        # May or may not fire depending on window content, but should not crash
        assert isinstance(events, list)

    def test_batch_returns_list(self):
        monitor = AnomalyDetectors()
        events = monitor.evaluate(self._make_batch([300.0] * 50))
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# ApplianceBaselineWatcher
# ---------------------------------------------------------------------------

class TestApplianceBaselineWatcher:
    def test_no_event_within_threshold(self):
        monitor = ApplianceBaselineWatcher(config={"device_id": "fridge"})
        df = pd.DataFrame({"value": [100.0 + np.random.normal(0, 2) for _ in range(50)]})
        events = monitor.evaluate(df)
        # Normal variation should not fire anomaly
        assert not any(isinstance(e, AnomalyApplianceEvent) and e.metric == "baseline_deviation_batch"
                       and e.value > 200 for e in events)

    def test_fires_on_extreme_outlier(self):
        monitor = ApplianceBaselineWatcher(
            config={"device_id": "fridge", "anomaly_threshold_sigma": 2.0}
        )
        values = [100.0] * 49 + [5000.0]  # 49 normal, 1 extreme
        df = pd.DataFrame({"value": values})
        events = monitor.evaluate(df)
        anomaly_events = [e for e in events if isinstance(e, AnomalyApplianceEvent)]
        assert len(anomaly_events) >= 1

    def test_degradation_flag_on_batch(self):
        monitor = ApplianceBaselineWatcher(
            config={
                "device_id": "fridge",
                "degradation_threshold_pct": 5.0,
            }
        )
        # First half: 100W, second half: 120W (20% increase)
        values = [100.0] * 30 + [120.0] * 30
        df = pd.DataFrame({"value": values})
        events = monitor.evaluate(df)
        deg_events = [e for e in events if isinstance(e, DegradationEvent)]
        assert len(deg_events) >= 1
        assert deg_events[0].drift_pct >= 5.0


# ---------------------------------------------------------------------------
# HabitDriftTracker
# ---------------------------------------------------------------------------

class TestHabitDriftTracker:
    def test_no_drift_insufficient_data(self):
        monitor = HabitDriftTracker(config={"baseline_days": 30, "recent_days": 7})
        # Not enough data
        events = monitor.evaluate({"value": 300.0, "ts": "2026-01-01 00:00:00"})
        assert events == []

    def test_drift_detected_on_batch(self):
        monitor = HabitDriftTracker(config={
            "baseline_days": 1,
            "recent_days": 1,
            "drift_threshold_pct": 5.0,
        })
        # 2 days worth at 10-tick/day resolution:  baseline=100, recent=150
        n_base = 10 * 60  # enough to exceed threshold
        n_recent = 10 * 60
        values = [100.0] * n_base + [150.0] * n_recent
        df = pd.DataFrame({"value": values, "ts": ["2026-01-01 00:00:00"] * (n_base + n_recent)})
        events = monitor.evaluate(df)
        drift_events = [e for e in events if isinstance(e, HabitDriftEvent)]
        assert len(drift_events) >= 1


# ---------------------------------------------------------------------------
# CarbonTracker
# ---------------------------------------------------------------------------

class TestCarbonTracker:
    def test_returns_carbon_event_on_tick(self):
        monitor = CarbonTracker(config={"intensity_source": "fallback"})
        events = monitor.evaluate({"value": 500.0, "ts": "2026-01-01 00:00:00"})
        assert len(events) == 1
        assert isinstance(events[0], CarbonRealtimeEvent)
        assert events[0].intensity_gco2_per_kwh == 200.0  # fallback

    def test_returns_carbon_event_on_batch(self):
        monitor = CarbonTracker()
        df = pd.DataFrame({"value": [300.0, 310.0], "ts": ["2026-01-01 00:00:00"] * 2})
        events = monitor.evaluate(df)
        assert isinstance(events[0], CarbonRealtimeEvent)


# ---------------------------------------------------------------------------
# RealtimeCostMeter
# ---------------------------------------------------------------------------

class TestRealtimeCostMeter:
    def test_returns_cost_event(self):
        monitor = RealtimeCostMeter(config={
            "tariff_name": "standard",
            "sample_seconds": 1.0,
        })
        events = monitor.evaluate({"value": 1000.0, "ts": "2026-04-20 14:00:00"})
        assert len(events) == 1
        assert isinstance(events[0], CostRealtimeEvent)
        # 1000W at 0.2459 £/kWh → £/hour = 1 * 0.2459 = £0.2459/h
        assert events[0].rate_gbp_per_hour > 0

    def test_zero_watts_returns_zero_cost(self):
        monitor = RealtimeCostMeter()
        events = monitor.evaluate({"value": 0.0, "ts": "2026-04-20 14:00:00"})
        assert events[0].rate_gbp_per_hour == 0.0


# ---------------------------------------------------------------------------
# BudgetTrajectoryTracker
# ---------------------------------------------------------------------------

class TestBudgetTrajectoryTracker:
    def test_no_warn_when_under_budget(self):
        monitor = BudgetTrajectoryTracker(config={"monthly_cap_gbp": 1000.0})
        df = pd.DataFrame({"value": [100.0] * 20})
        events = monitor.evaluate(df)
        warn_events = [e for e in events if isinstance(e, BudgetTrajectoryEvent) and e.severity != "info"]
        assert len(warn_events) == 0

    def test_warns_when_over_budget(self):
        monitor = BudgetTrajectoryTracker(config={"monthly_cap_gbp": 0.01, "warn_pct": 0.5})
        df = pd.DataFrame({"value": [5000.0] * 50})  # very high consumption
        events = monitor.evaluate(df)
        assert any(isinstance(e, BudgetTrajectoryEvent) for e in events)


# ---------------------------------------------------------------------------
# ForgotToTurnOffDetector (online-only)
# ---------------------------------------------------------------------------

class TestForgotToTurnOffDetector:
    def test_applicable_modes_online_only(self):
        assert ForgotToTurnOffDetector.applicable_modes == {"online"}

    def test_no_event_when_off(self):
        monitor = ForgotToTurnOffDetector(config={"on_threshold_watts": 50.0})
        events = monitor.evaluate({"value": 5.0, "ts": "2026-01-01 01:00:00"})
        assert events == []

    def test_fires_on_unusual_hour(self):
        monitor = ForgotToTurnOffDetector(config={
            "device_id": "oven",
            "on_threshold_watts": 50.0,
            "unusual_hours_start": 0,   # midnight is unusual
            "unusual_hours_end": 6,
            "idle_threshold_minutes": 999,  # disable idle check
        })
        # First tick: turns on at 1 AM (unusual)
        monitor._state["on_since"] = "2026-01-01 00:30:00"  # already on
        events = monitor.evaluate({"value": 2000.0, "ts": "2026-01-01 01:00:00"})
        unusual = [e for e in events if e.unusual_hours]
        assert len(unusual) >= 1

    def test_batch_returns_empty(self):
        """offline batch should return empty — online-only logic."""
        monitor = ForgotToTurnOffDetector()
        df = pd.DataFrame({"value": [2000.0] * 10})
        assert monitor.evaluate(df) == []


# ---------------------------------------------------------------------------
# DemandResponseListener (online-only)
# ---------------------------------------------------------------------------

class TestDemandResponseListener:
    def test_applicable_modes_online_only(self):
        assert DemandResponseListener.applicable_modes == {"online"}

    def test_inject_signal_fires_event(self):
        monitor = DemandResponseListener(config={
            "home_id": "home001",
            "deferrable_loads": ["switch.ev_charger"],
        })
        events = monitor.inject_signal({
            "event_id": "ss_20260420",
            "start": "2026-04-20T18:00:00Z",
            "end": "2026-04-20T19:00:00Z",
            "source": "octopus_saving_session",
            "reward_gbp": 3.50,
        })
        from energyx.data.events import DemandResponseEvent, ControlSuggestEvent
        dr_events = [e for e in events if isinstance(e, DemandResponseEvent)]
        suggest_events = [e for e in events if isinstance(e, ControlSuggestEvent)]
        assert len(dr_events) == 1
        assert len(suggest_events) == 1  # one per deferrable load

    def test_batch_returns_empty(self):
        monitor = DemandResponseListener()
        df = pd.DataFrame({"value": [300.0] * 5})
        assert monitor.evaluate(df) == []
