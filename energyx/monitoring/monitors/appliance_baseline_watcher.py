"""ApplianceBaselineWatcher — per-appliance rolling baseline with deviation flags.

Fires AnomalyApplianceEvent when consumption deviates beyond threshold.
Also fires DegradationEvent when sustained upward drift is detected.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from energyx.data.events import AnomalyApplianceEvent, DegradationEvent, Severity
from energyx.monitoring.monitors.base import BaseMonitor

logger = logging.getLogger(__name__)


class ApplianceBaselineWatcher(BaseMonitor):
    """Track per-appliance rolling baseline; flag anomalies and degradation.

    Config keys:
        device_id (str)
        baseline_window (int): number of readings for rolling baseline (default 1440 = 24h at 1/min)
        anomaly_threshold_sigma (float): sigma multiplier for anomaly flag (default 2.5)
        degradation_window (int): readings for degradation comparison (default 10080 = 1 week at 1/min)
        degradation_threshold_pct (float): % sustained increase to fire degradation (default 10.0)
        value_column (str): default "value"
    """

    applicable_modes: Set[str] = {"online", "offline"}

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._device_id = self._cfg("device_id", "unknown")
        self._baseline_win = self._cfg("baseline_window", 1440)
        self._anom_thr = self._cfg("anomaly_threshold_sigma", 2.5)
        self._deg_win = self._cfg("degradation_window", 10080)
        self._deg_thr_pct = self._cfg("degradation_threshold_pct", 10.0)
        self._value_col = self._cfg("value_column", "value")

        if "recent_values" not in self._state:
            self._state["recent_values"] = []
        if "long_values" not in self._state:
            self._state["long_values"] = []

    def evaluate(self, tick_or_batch: Any) -> List:
        if isinstance(tick_or_batch, pd.DataFrame):
            return self._eval_batch(tick_or_batch)
        return self._eval_tick(tick_or_batch)

    def _eval_tick(self, tick: Dict[str, Any]) -> List:
        val = self._extract_value(tick)
        if val is None:
            return []
        events = []
        recent: list = self._state["recent_values"]
        recent.append(val)
        if len(recent) > self._baseline_win:
            recent.pop(0)
        self._state["recent_values"] = recent

        long: list = self._state["long_values"]
        long.append(val)
        if len(long) > self._deg_win:
            long.pop(0)
        self._state["long_values"] = long

        if len(recent) >= max(10, self._baseline_win // 10):
            arr = np.array(recent[:-1])  # exclude current
            mu, sd = np.mean(arr), np.std(arr) + 1e-9
            z = abs(val - mu) / sd
            if z > self._anom_thr:
                sev = Severity.CRITICAL if z > 4.0 else Severity.WARN
                events.append(AnomalyApplianceEvent(
                    device_id=self._device_id,
                    metric="baseline_deviation",
                    value=val,
                    threshold=float(mu + self._anom_thr * sd),
                    context_window=recent[-5:],
                    methods_agreed=1,
                    severity=sev,
                    **self._make_event_kwargs(),
                ))

        if len(long) >= self._deg_win:
            mid = len(long) // 2
            old_mean = np.mean(long[:mid])
            new_mean = np.mean(long[mid:])
            if old_mean > 0:
                drift_pct = (new_mean - old_mean) / old_mean * 100
                if drift_pct >= self._deg_thr_pct:
                    events.append(DegradationEvent(
                        device_id=self._device_id,
                        baseline_watts=float(old_mean),
                        current_watts=float(new_mean),
                        drift_pct=float(drift_pct),
                        observation_days=self._deg_win // 1440,
                        severity=Severity.WARN,
                        **self._make_event_kwargs(),
                    ))
        return events

    def _eval_batch(self, df: pd.DataFrame) -> List:
        if df.empty or self._value_col not in df.columns:
            return []
        events = []
        arr = df[self._value_col].values.astype(float)
        mu, sd = np.mean(arr), np.std(arr) + 1e-9
        for i, val in enumerate(arr):
            if abs(val - mu) / sd > self._anom_thr:
                ctx = arr[max(0, i-2):i+3].tolist()
                events.append(AnomalyApplianceEvent(
                    device_id=self._device_id,
                    metric="baseline_deviation_batch",
                    value=float(val),
                    threshold=float(mu + self._anom_thr * sd),
                    context_window=ctx,
                    methods_agreed=1,
                    severity=Severity.WARN,
                    **self._make_event_kwargs(),
                ))
        # Degradation check on batch
        if len(arr) >= 20:
            mid = len(arr) // 2
            old_mean, new_mean = np.mean(arr[:mid]), np.mean(arr[mid:])
            if old_mean > 0:
                drift_pct = (new_mean - old_mean) / old_mean * 100
                if drift_pct >= self._deg_thr_pct:
                    events.append(DegradationEvent(
                        device_id=self._device_id,
                        baseline_watts=float(old_mean),
                        current_watts=float(new_mean),
                        drift_pct=float(drift_pct),
                        observation_days=len(arr) // 1440,
                        severity=Severity.WARN,
                        **self._make_event_kwargs(),
                    ))
        return events

    def _extract_value(self, tick: Dict[str, Any]) -> Optional[float]:
        val = tick.get(self._value_col, tick.get("value"))
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
