"""Event escalation — routes bus events to the correct agent or UI.

Escalation rules (applied identically whether events come from the online
Monitoring Module or the offline Monitor Agent):

  Informational event  → forward to UI
  Diagnostic event     → dispatch to Analysis Agent for explanation
  Actionable event     → dispatch to Control Agent (online only) if
                         auto-consent is on, else surface as advisory
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from energyx.data.events import (
    AnomalyApplianceEvent,
    BudgetTrajectoryEvent,
    ControlSuggestEvent,
    DegradationEvent,
    DemandResponseEvent,
    ForgotTurnOffEvent,
    HabitDriftEvent,
    BaseEvent,
    EventType,
    Severity,
)
from energyx.orchestrator.permission_manager import PermissionManager

logger = logging.getLogger(__name__)

# Map event type → auto_control permission required
_EVENT_PERMISSION_MAP: Dict[str, str] = {
    EventType.CONTROL_SUGGEST: "auto_control:laundry_ev",
    EventType.DR_EVENT: "dr_enrollment",
    EventType.FORGOT_TURN_OFF: "auto_control:emergency_off",
}

# Events that warrant analysis escalation
_DIAGNOSTIC_TYPES = {
    EventType.ANOMALY_APPLIANCE,
    EventType.DEGRADATION,
    EventType.HABIT_DRIFT,
    EventType.BUDGET_TRAJECTORY,
}


class EventEscalator:
    """Subscribe to the EventBus and dispatch events to appropriate handlers.

    Handlers are callables registered by the Orchestrator:
      - ui_handler(event)          → always called (UI display)
      - analysis_handler(event)    → called for diagnostic events
      - control_handler(event)     → called for actionable events (online, perm-gated)
    """

    def __init__(
        self,
        permission_manager: PermissionManager,
        is_online_fn: Callable[[], bool],
        ui_handler: Optional[Callable[[BaseEvent], None]] = None,
        analysis_handler: Optional[Callable[[BaseEvent], None]] = None,
        control_handler: Optional[Callable[[BaseEvent], None]] = None,
    ):
        self._pm = permission_manager
        self._is_online = is_online_fn
        self._ui = ui_handler or (lambda e: None)
        self._analysis = analysis_handler or (lambda e: None)
        self._control = control_handler or (lambda e: None)
        self._ui_queue: List[BaseEvent] = []
        self._advisory_queue: List[BaseEvent] = []

    def handle(self, event: BaseEvent) -> None:
        """Main entry point — called by EventBus subscriber or Orchestrator directly."""
        # Always forward to UI
        self._ui(event)
        self._ui_queue.append(event)

        etype = event.type

        # Diagnostic → Analysis Agent
        if etype in _DIAGNOSTIC_TYPES:
            logger.debug(f"Escalating diagnostic event {etype} to Analysis Agent")
            self._analysis(event)
            return

        # Actionable → Control Agent (online, permission-gated)
        if etype in {EventType.CONTROL_SUGGEST, EventType.DR_EVENT, EventType.FORGOT_TURN_OFF}:
            required_perm = _EVENT_PERMISSION_MAP.get(etype)
            online = self._is_online()
            if online and required_perm and self._pm.is_granted(required_perm):
                logger.debug(f"Auto-dispatching {etype} to Control Agent")
                self._control(event)
            else:
                reason = "offline mode" if not online else f"permission '{required_perm}' not granted"
                logger.debug(f"Control action for {etype} surfaced as advisory ({reason})")
                event_copy = event  # already in advisory queue via UI handler
                self._advisory_queue.append(event_copy)
            return

    def register_with_bus(self, bus) -> None:
        """Subscribe to all event types on an EventBus."""
        bus.subscribe("*", self.handle)

    def pop_advisory(self) -> List[BaseEvent]:
        """Return and clear advisory events (surfaced for user action)."""
        items = list(self._advisory_queue)
        self._advisory_queue.clear()
        return items

    def pop_ui_events(self) -> List[BaseEvent]:
        items = list(self._ui_queue)
        self._ui_queue.clear()
        return items
