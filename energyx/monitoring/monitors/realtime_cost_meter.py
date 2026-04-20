"""RealtimeCostMeter — live £/hour burn rate (online) or historical cost (offline)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from energyx.data.events import CostRealtimeEvent, Severity
from energyx.monitoring.monitors.base import BaseMonitor


class RealtimeCostMeter(BaseMonitor):
    """Compute current cost rate using the active tariff.

    Online: fires on every tick with instantaneous £/hour.
    Offline: computes £ for each row in the batch; fires summary event.

    Config keys:
        tariff_id (str): tariff to use (None → use household default)
        household_id (str): for tariff lookup
        value_column (str): Watts column (default "value")
        sample_seconds (float): seconds per tick (default 1.0)
        top_appliances (dict): {device_id: watts} for top-contributor breakdown
    """

    applicable_modes: Set[str] = {"online", "offline"}

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._household_id = self._cfg("household_id", self.home_id)
        self._value_col = self._cfg("value_column", "value")
        self._sample_sec = self._cfg("sample_seconds", 1.0)
        self._top_appliances: Dict[str, float] = self._cfg("top_appliances", {})

    def evaluate(self, tick_or_batch: Any) -> List[CostRealtimeEvent]:
        if isinstance(tick_or_batch, pd.DataFrame):
            return self._eval_batch(tick_or_batch)
        return self._eval_tick(tick_or_batch)

    def _eval_tick(self, tick: Dict[str, Any]) -> List[CostRealtimeEvent]:
        watts = tick.get(self._value_col, tick.get("value"))
        if watts is None:
            return []
        try:
            watts = float(watts)
        except (TypeError, ValueError):
            return []
        ts = self._parse_ts(tick.get("ts"))
        rate = self._get_rate(ts)
        gbp_per_hour = watts / 1000.0 * rate  # kW * £/kWh = £/h
        contributors = self._compute_contributors(rate)
        return [CostRealtimeEvent(
            rate_gbp_per_hour=round(gbp_per_hour, 4),
            top_contributors=contributors,
            tariff_name=self._cfg("tariff_name", "standard"),
            severity=Severity.INFO,
            **self._make_event_kwargs(),
        )]

    def _eval_batch(self, df: pd.DataFrame) -> List[CostRealtimeEvent]:
        if df.empty or self._value_col not in df.columns:
            return []
        total_kwh = df[self._value_col].mean() / 1000.0 * len(df) * self._sample_sec / 3600
        ts = self._parse_ts(df["ts"].iloc[0] if "ts" in df.columns else None)
        rate = self._get_rate(ts)
        total_cost = total_kwh * rate
        rate_per_h = df[self._value_col].mean() / 1000.0 * rate
        return [CostRealtimeEvent(
            rate_gbp_per_hour=round(rate_per_h, 4),
            top_contributors=self._compute_contributors(rate),
            tariff_name=self._cfg("tariff_name", "standard"),
            severity=Severity.INFO,
            **self._make_event_kwargs(),
        )]

    def _get_rate(self, ts: Optional[datetime]) -> float:
        try:
            from energyx.data.tariffs import get_tariff_cache
            cache = get_tariff_cache()
            tariff = cache.get_active_tariff(self._household_id)
            if tariff and ts:
                return tariff.unit_rate_at(ts)
            if tariff:
                return tariff.rates[0].unit_rate_gbp_per_kwh if tariff.rates else 0.2459
        except Exception:
            pass
        return 0.2459  # Ofgem Q1 2026 cap fallback

    def _parse_ts(self, ts_val: Any) -> Optional[datetime]:
        if ts_val is None:
            return None
        try:
            return pd.to_datetime(ts_val).to_pydatetime()
        except Exception:
            return None

    def _compute_contributors(self, rate: float) -> List:
        return [
            (dev, round(watts / 1000.0 * rate, 4))
            for dev, watts in sorted(
                self._top_appliances.items(), key=lambda x: -x[1]
            )[:3]
        ]
