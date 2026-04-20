"""BudgetTrajectoryTracker — project end-of-month spend; alert when cap threatened."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from energyx.data.events import BudgetTrajectoryEvent, Severity
from energyx.monitoring.monitors.base import BaseMonitor


class BudgetTrajectoryTracker(BaseMonitor):
    """Project spend trajectory against a user-set monthly cap.

    Online: fires on every tick; proactively warns when cap is at risk.
    Offline (reframed): computes how the batch tracked against the cap.

    Config keys:
        monthly_cap_gbp (float): user cap (default 150.0)
        household_id (str)
        value_column (str): Watts (default "value")
        sample_seconds (float): default 1.0
        warn_pct (float): fire WARN when projected >= warn_pct * cap (default 0.85)
        critical_pct (float): CRITICAL threshold (default 1.0)
    """

    applicable_modes: Set[str] = {"online", "offline"}

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._cap = self._cfg("monthly_cap_gbp", 150.0)
        self._household_id = self._cfg("household_id", self.home_id)
        self._value_col = self._cfg("value_column", "value")
        self._sample_sec = self._cfg("sample_seconds", 1.0)
        self._warn_pct = self._cfg("warn_pct", 0.85)
        self._crit_pct = self._cfg("critical_pct", 1.0)

        if "month_spend" not in self._state:
            self._state["month_spend"] = 0.0
        if "month_start_ts" not in self._state:
            self._state["month_start_ts"] = None
        if "tick_count" not in self._state:
            self._state["tick_count"] = 0

    def evaluate(self, tick_or_batch: Any) -> List[BudgetTrajectoryEvent]:
        if isinstance(tick_or_batch, pd.DataFrame):
            return self._eval_batch(tick_or_batch)
        return self._eval_tick(tick_or_batch)

    def _eval_tick(self, tick: Dict[str, Any]) -> List[BudgetTrajectoryEvent]:
        watts = tick.get(self._value_col, tick.get("value"))
        if watts is None:
            return []
        try:
            watts = float(watts)
        except (TypeError, ValueError):
            return []
        ts = self._parse_ts(tick.get("ts"))
        rate = self._get_rate(ts)
        kwh = watts / 1000.0 * self._sample_sec / 3600
        cost = kwh * rate
        self._state["month_spend"] = self._state["month_spend"] + cost
        self._state["tick_count"] += 1
        spend = self._state["month_spend"]

        # Project to end of month
        ticks_per_day = 86400 / self._sample_sec
        days_elapsed = self._state["tick_count"] / ticks_per_day
        if days_elapsed <= 0:
            return []
        days_in_month = 30.44
        projected = spend / days_elapsed * days_in_month
        pct = projected / self._cap if self._cap > 0 else 0

        if pct >= self._warn_pct:
            days_remaining = days_in_month - days_elapsed
            rate_per_day = spend / days_elapsed if days_elapsed > 0 else 0
            budget_remaining = self._cap - spend
            days_to_breach = (budget_remaining / rate_per_day) if rate_per_day > 0 else None
            sev = Severity.CRITICAL if pct >= self._crit_pct else Severity.WARN
            return [BudgetTrajectoryEvent(
                projected_spend_gbp=round(projected, 2),
                cap_gbp=self._cap,
                days_to_breach=round(days_to_breach, 1) if days_to_breach is not None else None,
                pct_of_cap=round(pct, 3),
                severity=sev,
                **self._make_event_kwargs(),
            )]
        return []

    def _eval_batch(self, df: pd.DataFrame) -> List[BudgetTrajectoryEvent]:
        if df.empty or self._value_col not in df.columns:
            return []
        ts0 = self._parse_ts(df["ts"].iloc[0] if "ts" in df.columns else None)
        rate = self._get_rate(ts0)
        total_kwh = df[self._value_col].mean() / 1000.0 * len(df) * self._sample_sec / 3600
        total_cost = total_kwh * rate
        pct = total_cost / self._cap if self._cap > 0 else 0
        sev = Severity.WARN if pct >= self._warn_pct else Severity.INFO
        return [BudgetTrajectoryEvent(
            projected_spend_gbp=round(total_cost, 2),
            cap_gbp=self._cap,
            days_to_breach=None,
            pct_of_cap=round(pct, 3),
            severity=sev,
            **self._make_event_kwargs(),
        )]

    def _get_rate(self, ts: Optional[datetime]) -> float:
        try:
            from energyx.data.tariffs import get_tariff_cache
            t = get_tariff_cache().get_active_tariff(self._household_id)
            if t and ts:
                return t.unit_rate_at(ts)
        except Exception:
            pass
        return 0.2459

    def _parse_ts(self, val: Any) -> Optional[datetime]:
        if val is None:
            return None
        try:
            return pd.to_datetime(val).to_pydatetime()
        except Exception:
            return None
