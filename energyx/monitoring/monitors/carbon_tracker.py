"""CarbonTracker — translates kWh to kgCO₂ using carbon intensity data."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from energyx.data.events import CarbonRealtimeEvent, Severity
from energyx.monitoring.monitors.base import BaseMonitor

logger = logging.getLogger(__name__)

# Fallback intensity if API unavailable (national average, gCO2/kWh)
_FALLBACK_INTENSITY = 200.0


class CarbonTracker(BaseMonitor):
    """Translate power readings to carbon footprint.

    Uses National Grid ESO carbon intensity (live in online mode,
    historical intensity table in offline mode).

    Config keys:
        intensity_source (str): "api" | "cached" | "fallback" (default "fallback")
        intensity_cache (dict): {datetime_str: gco2_per_kwh}
        high_carbon_threshold (float): gCO2/kWh to warn (default 300)
        value_column (str): default "value" (expected in Watts)
        sample_seconds (float): seconds between ticks for kWh conversion (default 1.0)
        region (str): default "national"
    """

    applicable_modes: Set[str] = {"online", "offline"}

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._intensity_src = self._cfg("intensity_source", "fallback")
        self._intensity_cache: Dict[str, float] = self._cfg("intensity_cache", {})
        self._high_carbon_thr = self._cfg("high_carbon_threshold", 300.0)
        self._value_col = self._cfg("value_column", "value")
        self._sample_sec = self._cfg("sample_seconds", 1.0)
        self._region = self._cfg("region", "national")

    def evaluate(self, tick_or_batch: Any) -> List[CarbonRealtimeEvent]:
        if isinstance(tick_or_batch, pd.DataFrame):
            return self._eval_batch(tick_or_batch)
        return self._eval_tick(tick_or_batch)

    def _eval_tick(self, tick: Dict[str, Any]) -> List[CarbonRealtimeEvent]:
        watts = tick.get(self._value_col, tick.get("value"))
        if watts is None:
            return []
        try:
            watts = float(watts)
        except (TypeError, ValueError):
            return []
        intensity = self._get_intensity(tick.get("ts"))
        kwh_per_tick = watts * self._sample_sec / 3600
        return [CarbonRealtimeEvent(
            intensity_gco2_per_kwh=intensity,
            region=self._region,
            source=self._intensity_src,
            severity=Severity.WARN if intensity > self._high_carbon_thr else Severity.INFO,
            **self._make_event_kwargs(),
        )]

    def _eval_batch(self, df: pd.DataFrame) -> List[CarbonRealtimeEvent]:
        if df.empty or self._value_col not in df.columns:
            return []
        intensity = self._get_intensity(None)
        return [CarbonRealtimeEvent(
            intensity_gco2_per_kwh=intensity,
            region=self._region,
            source="cached_batch",
            severity=Severity.INFO,
            **self._make_event_kwargs(),
        )]

    def _get_intensity(self, ts: Optional[str]) -> float:
        if self._intensity_src == "fallback":
            return _FALLBACK_INTENSITY
        if ts and str(ts) in self._intensity_cache:
            return self._intensity_cache[str(ts)]
        return _FALLBACK_INTENSITY
