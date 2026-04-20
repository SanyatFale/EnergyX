"""ForgotToTurnOffDetector — online-only; flags appliances running in unusual windows."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from energyx.data.events import ForgotTurnOffEvent, Severity
from energyx.monitoring.monitors.base import BaseMonitor


class ForgotToTurnOffDetector(BaseMonitor):
    """Detect appliances running in unusual time windows.

    Examples: oven at 1 AM, iron running for 30+ minutes idle.
    Online-only — retrospective flagging of past events is not actionable.

    Config keys:
        device_id (str)
        unusual_hours_start (int): hour (24h) when unusual period starts (default 23)
        unusual_hours_end (int): hour when unusual period ends (default 7)
        idle_threshold_minutes (float): fire if appliance on for this long (default 30)
        on_threshold_watts (float): watts above which appliance is "on" (default 50)
        value_column (str): default "value"
    """

    applicable_modes: Set[str] = {"online"}  # Online-only

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._device_id = self._cfg("device_id", "unknown")
        self._unusual_start = self._cfg("unusual_hours_start", 23)
        self._unusual_end = self._cfg("unusual_hours_end", 7)
        self._idle_thr_min = self._cfg("idle_threshold_minutes", 30.0)
        self._on_thr_watts = self._cfg("on_threshold_watts", 50.0)
        self._value_col = self._cfg("value_column", "value")

        if "on_since" not in self._state:
            self._state["on_since"] = None  # ISO str or None

    def evaluate(self, tick_or_batch: Any) -> List[ForgotTurnOffEvent]:
        if isinstance(tick_or_batch, pd.DataFrame):
            # Not applicable in offline batch context; return empty
            return []
        return self._eval_tick(tick_or_batch)

    def _eval_tick(self, tick: Dict[str, Any]) -> List[ForgotTurnOffEvent]:
        watts = tick.get(self._value_col, tick.get("value"))
        ts_val = tick.get("ts")
        if watts is None:
            return []
        try:
            watts = float(watts)
        except (TypeError, ValueError):
            return []

        ts = self._parse_ts(ts_val)
        is_on = watts > self._on_thr_watts

        if is_on and self._state["on_since"] is None:
            self._state["on_since"] = ts_val

        if not is_on:
            self._state["on_since"] = None
            return []

        on_since = self._parse_ts(self._state["on_since"])
        if on_since is None:
            return []

        events = []
        # Check unusual hours
        if ts and self._is_unusual_hour(ts):
            events.append(ForgotTurnOffEvent(
                device_id=self._device_id,
                running_since=str(self._state["on_since"]),
                unusual_hours=True,
                idle_minutes=None,
                severity=Severity.WARN,
                **self._make_event_kwargs(),
            ))

        # Check idle timeout
        if on_since and ts:
            elapsed_min = (ts.timestamp() - on_since.timestamp()) / 60
            if elapsed_min >= self._idle_thr_min:
                events.append(ForgotTurnOffEvent(
                    device_id=self._device_id,
                    running_since=str(self._state["on_since"]),
                    unusual_hours=False,
                    idle_minutes=round(elapsed_min, 1),
                    severity=Severity.WARN,
                    **self._make_event_kwargs(),
                ))
                # Reset so we don't fire every tick after threshold
                self._state["on_since"] = None

        return events

    def _is_unusual_hour(self, ts: datetime) -> bool:
        h = ts.hour
        if self._unusual_start <= self._unusual_end:
            return self._unusual_start <= h < self._unusual_end
        return h >= self._unusual_start or h < self._unusual_end

    def _parse_ts(self, val: Any) -> Optional[datetime]:
        if val is None:
            return None
        try:
            return pd.to_datetime(val).to_pydatetime().replace(tzinfo=None)
        except Exception:
            return None
