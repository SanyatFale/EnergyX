"""HabitDriftTracker — detects slow behavioural shifts over weeks/months."""

from __future__ import annotations

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from energyx.data.events import HabitDriftEvent, Severity
from energyx.monitoring.monitors.base import BaseMonitor


class HabitDriftTracker(BaseMonitor):
    """Track long-term behavioural patterns and fire on drift.

    Compares a recent window against a historical baseline.
    Designed for weekly-resolution; meaningful results require 2+ weeks of data.

    Config keys:
        pattern (str): pattern name (e.g. "hvac_runtime")
        baseline_days (int): how many days form the historical baseline (default 30)
        recent_days (int): how many recent days to compare (default 7)
        drift_threshold_pct (float): % change to fire event (default 10.0)
        value_column (str): default "value"
    """

    applicable_modes: Set[str] = {"online", "offline"}

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._pattern = self._cfg("pattern", "consumption")
        self._baseline_days = self._cfg("baseline_days", 30)
        self._recent_days = self._cfg("recent_days", 7)
        self._drift_thr = self._cfg("drift_threshold_pct", 10.0)
        self._value_col = self._cfg("value_column", "value")

        if "history" not in self._state:
            self._state["history"] = []  # list of (ts_str, value) tuples

    def evaluate(self, tick_or_batch: Any) -> List[HabitDriftEvent]:
        if isinstance(tick_or_batch, pd.DataFrame):
            return self._eval_batch(tick_or_batch)
        return self._eval_tick(tick_or_batch)

    def _eval_tick(self, tick: Dict[str, Any]) -> List[HabitDriftEvent]:
        val = tick.get(self._value_col, tick.get("value"))
        ts = tick.get("ts", "")
        if val is None:
            return []
        try:
            val = float(val)
        except (TypeError, ValueError):
            return []
        history: list = self._state["history"]
        history.append((str(ts), val))
        # Keep only baseline_days + recent_days worth (approx by count)
        max_len = (self._baseline_days + self._recent_days) * 1440
        if len(history) > max_len:
            history = history[-max_len:]
            self._state["history"] = history
        return self._check_drift(history)

    def _eval_batch(self, df: pd.DataFrame) -> List[HabitDriftEvent]:
        if df.empty or self._value_col not in df.columns:
            return []
        history = [(str(r.get("ts", "")), float(r[self._value_col]))
                   for _, r in df.iterrows() if pd.notna(r[self._value_col])]
        return self._check_drift(history)

    def _check_drift(self, history: list) -> List[HabitDriftEvent]:
        if len(history) < (self._baseline_days + self._recent_days) * 10:
            return []  # Not enough data
        vals = np.array([v for _, v in history])
        split = max(1, len(vals) - self._recent_days * 1440)
        baseline_mean = np.mean(vals[:split])
        recent_mean = np.mean(vals[split:])
        if baseline_mean <= 0:
            return []
        drift_pct = (recent_mean - baseline_mean) / baseline_mean * 100
        if abs(drift_pct) >= self._drift_thr:
            return [HabitDriftEvent(
                pattern=self._pattern,
                drift_magnitude=float(drift_pct),
                baseline_period=f"last {self._baseline_days} days (excluding recent)",
                current_period=f"last {self._recent_days} days",
                severity=Severity.WARN if abs(drift_pct) < 25 else Severity.CRITICAL,
                **self._make_event_kwargs(),
            )]
        return []
