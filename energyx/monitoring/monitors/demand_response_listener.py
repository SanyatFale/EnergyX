"""DemandResponseListener — online-only; listens for grid DR signals."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from energyx.data.events import DemandResponseEvent, ControlSuggestEvent, Severity
from energyx.monitoring.monitors.base import BaseMonitor

logger = logging.getLogger(__name__)


class DemandResponseListener(BaseMonitor):
    """Listen for Demand Response signals and fire events.

    Online-only — DR signals are live grid events; not applicable in offline mode.

    In online mode, the Orchestrator injects DR signals via inject_signal().
    The tick stream is used only to check whether DR is currently active.

    Config keys:
        enrollment_required (bool): if True and dr_enrollment permission missing, skip (default True)
        deferrable_loads (list): entity_ids of loads that can be deferred
    """

    applicable_modes: Set[str] = {"online"}  # Online-only

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._enrollment_required = self._cfg("enrollment_required", True)
        self._deferrable = self._cfg("deferrable_loads", [])
        if "active_event" not in self._state:
            self._state["active_event"] = None

    def inject_signal(self, signal: Dict[str, Any]) -> List:
        """Called by the Orchestrator when a DR signal arrives from an API."""
        if not signal:
            return []
        event_id = signal.get("event_id", "dr_unknown")
        events = [DemandResponseEvent(
            event_id=event_id,
            start_ts=str(signal.get("start", "")),
            end_ts=str(signal.get("end", "")),
            signal_source=str(signal.get("source", "unknown")),
            reward_gbp=signal.get("reward_gbp"),
            severity=Severity.WARN,
            **self._make_event_kwargs(),
        )]
        # Suggest deferring each registered load
        for entity_id in self._deferrable:
            events.append(ControlSuggestEvent(
                action="defer",
                entity_id=entity_id,
                reason=f"demand_response:{event_id}",
                ha_payload={
                    "domain": "switch",
                    "service": "turn_off",
                    "target": {"entity_id": entity_id},
                },
                severity=Severity.WARN,
                **self._make_event_kwargs(),
            ))
        self._state["active_event"] = event_id
        return events

    def evaluate(self, tick_or_batch: Any) -> List:
        # Tick stream evaluation: just check if an active DR event has ended.
        # Signal injection happens via inject_signal(), not via ticks.
        if isinstance(tick_or_batch, pd.DataFrame):
            return []
        # If there is an active DR event, we could check if it has expired.
        # For now, return empty — expiry logic handled by Orchestrator.
        return []
