"""Control Agent — emits Home Assistant service-call JSON.

ALL actions produce HA service-call JSON (schema per AGENT_MAPPING.md §3.4).
ALL actions are permission-gated via the Orchestrator's PermissionManager.
ALL actions support dry-run mode.
ONLINE ONLY — entirely disabled in offline mode.

Tools:
  - set_appliance_schedule(schedule_json)
  - set_setpoint(entity_id, temperature)
  - defer_load(entity_id, defer_until)
  - power_off(entity_id)
  - read_device_state(entity_id)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from energyx.agents.control.dispatcher import BaseDispatcher, StubDispatcher
from energyx.orchestrator.permission_manager import PermissionManager, PermissionDenied

logger = logging.getLogger(__name__)

# Map tool → required permission
_TOOL_PERMISSIONS = {
    "set_appliance_schedule": "auto_control:laundry_ev",
    "set_setpoint": "auto_control:hvac",
    "defer_load": "auto_control:laundry_ev",
    "power_off": "auto_control:emergency_off",
    "read_device_state": None,  # No permission required for reads
}


def _ha_call(domain: str, service: str, entity_id: str, **service_data) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "domain": domain,
        "service": service,
        "target": {"entity_id": entity_id},
    }
    if service_data:
        payload["service_data"] = service_data
    return payload


class ControlAgent:
    """Lightweight agent that translates requests to HA service-call JSON.

    Not an LLM agentic loop — just a thin permission-gated dispatcher.
    """

    def __init__(
        self,
        permission_manager: Optional[PermissionManager] = None,
        dispatcher: Optional[BaseDispatcher] = None,
        dry_run: bool = False,
    ):
        self._pm = permission_manager or PermissionManager()
        self._dispatcher = dispatcher or StubDispatcher(dry_run=dry_run)
        self._dry_run = dry_run

    def handle(self, query: str) -> Dict[str, Any]:
        """Parse a natural-language control command and dispatch it."""
        q = query.lower()
        if "turn off" in q or "power off" in q:
            entity = self._extract_entity(query)
            return self.power_off(entity)
        if "set temperature" in q or "setpoint" in q:
            entity = self._extract_entity(query)
            temp = self._extract_number(query, default=19.0)
            return self.set_setpoint(entity, temp)
        if "defer" in q or "delay" in q:
            entity = self._extract_entity(query)
            return self.defer_load(entity)
        return {
            "response": "I can turn off appliances, set temperatures, or defer loads. "
                        "Please specify the device and action.",
            "status": "clarification_needed",
        }

    # ------------------------------------------------------------------
    # Tools (all permission-gated, all return HA JSON)
    # ------------------------------------------------------------------

    def set_appliance_schedule(self, schedule_json: str) -> Dict[str, Any]:
        """Dispatch a pre-built HA schedule JSON (from Analysis Agent)."""
        self._require("set_appliance_schedule")
        try:
            schedule = json.loads(schedule_json)
        except Exception as e:
            return {"error": f"Invalid schedule JSON: {e}"}
        result = self._dispatcher.dispatch(schedule)
        return {"tool": "set_appliance_schedule", "schedule_id": schedule.get("schedule_id"), **result}

    def set_setpoint(self, entity_id: str, temperature: float) -> Dict[str, Any]:
        """Set a thermostat setpoint."""
        self._require("set_setpoint")
        payload = _ha_call("climate", "set_temperature", entity_id, temperature=temperature)
        result = self._dispatcher.dispatch(payload)
        return {"tool": "set_setpoint", "entity_id": entity_id, "temperature": temperature, **result}

    def defer_load(self, entity_id: str, defer_until: Optional[str] = None) -> Dict[str, Any]:
        """Defer a flexible load (turn off until defer_until timestamp)."""
        self._require("defer_load")
        payload = _ha_call("switch", "turn_off", entity_id)
        if defer_until:
            payload["service_data"] = {"defer_until": defer_until}
        result = self._dispatcher.dispatch(payload)
        return {"tool": "defer_load", "entity_id": entity_id, "defer_until": defer_until, **result}

    def power_off(self, entity_id: str) -> Dict[str, Any]:
        """Immediately power off a device."""
        self._require("power_off")
        payload = _ha_call("switch", "turn_off", entity_id)
        result = self._dispatcher.dispatch(payload)
        return {"tool": "power_off", "entity_id": entity_id, **result}

    def read_device_state(self, entity_id: str) -> Dict[str, Any]:
        """Read the current state of a device (no permission required)."""
        # In stub mode, return a placeholder state
        logger.info(f"ControlAgent.read_device_state: {entity_id} (stub)")
        return {
            "tool": "read_device_state",
            "entity_id": entity_id,
            "state": "unknown",
            "note": "Real state requires HA REST API connection.",
        }

    # ------------------------------------------------------------------
    def _require(self, tool_name: str) -> None:
        perm = _TOOL_PERMISSIONS.get(tool_name)
        if perm:
            self._pm.require(perm)

    @staticmethod
    def _extract_entity(query: str) -> str:
        import re
        m = re.search(r"(switch\.\w+|climate\.\w+|light\.\w+|sensor\.\w+)", query)
        if m:
            return m.group(1)
        # Guess from keywords
        q = query.lower()
        if "ev" in q or "charger" in q:
            return "switch.ev_charger"
        if "heating" in q or "thermostat" in q or "hvac" in q:
            return "climate.living_room_thermostat"
        if "iron" in q:
            return "switch.iron"
        if "dishwasher" in q:
            return "switch.dishwasher"
        if "washing" in q:
            return "switch.washing_machine"
        return "switch.unknown_device"

    @staticmethod
    def _extract_number(query: str, default: float = 19.0) -> float:
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:°?C|degrees?)?", query)
        if m:
            return float(m.group(1))
        return default
