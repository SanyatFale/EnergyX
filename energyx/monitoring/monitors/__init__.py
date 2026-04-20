"""Shared monitor core — 9 pure-Python monitor classes.

No LLM inside any monitor class.  Each exposes:
    evaluate(tick_or_batch) -> list[Event]

applicable_modes: set[str] declares which orchestration context can use it.
Online-only monitors: ForgotToTurnOffDetector, DemandResponseListener.
"""

from energyx.monitoring.monitors.base import BaseMonitor
from energyx.monitoring.monitors.anomaly_detectors import AnomalyDetectors
from energyx.monitoring.monitors.appliance_baseline_watcher import ApplianceBaselineWatcher
from energyx.monitoring.monitors.habit_drift_tracker import HabitDriftTracker
from energyx.monitoring.monitors.carbon_tracker import CarbonTracker
from energyx.monitoring.monitors.realtime_cost_meter import RealtimeCostMeter
from energyx.monitoring.monitors.budget_trajectory_tracker import BudgetTrajectoryTracker
from energyx.monitoring.monitors.historic_store_writer import HistoricStoreWriter
from energyx.monitoring.monitors.forgot_to_turn_off_detector import ForgotToTurnOffDetector
from energyx.monitoring.monitors.demand_response_listener import DemandResponseListener

ALL_MONITORS = [
    AnomalyDetectors,
    ApplianceBaselineWatcher,
    HabitDriftTracker,
    CarbonTracker,
    RealtimeCostMeter,
    BudgetTrajectoryTracker,
    HistoricStoreWriter,
    ForgotToTurnOffDetector,
    DemandResponseListener,
]

MONITOR_REGISTRY = {cls.__name__: cls for cls in ALL_MONITORS}

__all__ = [
    "BaseMonitor",
    "AnomalyDetectors",
    "ApplianceBaselineWatcher",
    "HabitDriftTracker",
    "CarbonTracker",
    "RealtimeCostMeter",
    "BudgetTrajectoryTracker",
    "HistoricStoreWriter",
    "ForgotToTurnOffDetector",
    "DemandResponseListener",
    "ALL_MONITORS",
    "MONITOR_REGISTRY",
]
