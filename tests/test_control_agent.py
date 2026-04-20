"""Tests for Step 6: Control Agent.

pytest tests/test_control_agent.py
"""
import json
import pytest
from pathlib import Path

from energyx.agents.control.agent import ControlAgent
from energyx.agents.control.dispatcher import StubDispatcher
from energyx.orchestrator.permission_manager import PermissionManager, PermissionDenied


class TestControlAgent:
    def _make_agent(self, tmp_path, granted_perms=None):
        pm = PermissionManager(path=tmp_path / "perms.json")
        for perm in (granted_perms or []):
            pm.grant(perm)
        dispatcher = StubDispatcher(log_path=tmp_path / "dispatch.log", dry_run=True)
        return ControlAgent(permission_manager=pm, dispatcher=dispatcher, dry_run=True)

    def test_power_off_requires_permission(self, tmp_path):
        agent = self._make_agent(tmp_path, granted_perms=[])
        with pytest.raises(PermissionDenied):
            agent.power_off("switch.iron")

    def test_power_off_with_permission(self, tmp_path):
        agent = self._make_agent(tmp_path, granted_perms=["auto_control:emergency_off"])
        result = agent.power_off("switch.iron")
        assert result["tool"] == "power_off"
        assert result["entity_id"] == "switch.iron"
        assert result["status"] == "dry_run"

    def test_set_setpoint_requires_hvac_permission(self, tmp_path):
        agent = self._make_agent(tmp_path, granted_perms=[])
        with pytest.raises(PermissionDenied):
            agent.set_setpoint("climate.living_room", 19.0)

    def test_set_setpoint_with_permission(self, tmp_path):
        agent = self._make_agent(tmp_path, granted_perms=["auto_control:hvac"])
        result = agent.set_setpoint("climate.living_room", 19.5)
        assert result["temperature"] == 19.5

    def test_read_device_state_no_permission_needed(self, tmp_path):
        agent = self._make_agent(tmp_path, granted_perms=[])
        result = agent.read_device_state("switch.ev_charger")
        assert "entity_id" in result
        assert result["entity_id"] == "switch.ev_charger"

    def test_defer_load_requires_permission(self, tmp_path):
        agent = self._make_agent(tmp_path, granted_perms=[])
        with pytest.raises(PermissionDenied):
            agent.defer_load("switch.dishwasher")

    def test_dispatch_logged_to_file(self, tmp_path):
        agent = self._make_agent(tmp_path, granted_perms=["auto_control:emergency_off"])
        agent.power_off("switch.iron")
        log_path = tmp_path / "dispatch.log"
        assert log_path.exists()
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["dry_run"] is True
        assert entry["payload"]["domain"] == "switch"

    def test_ha_json_schema_compliance(self, tmp_path):
        """HA service-call JSON must have domain, service, target.entity_id."""
        agent = self._make_agent(tmp_path, granted_perms=["auto_control:hvac"])
        result = agent.set_setpoint("climate.bedroom", 18.0)
        log_path = tmp_path / "dispatch.log"
        with open(log_path) as f:
            entry = json.loads(f.readlines()[0])
        payload = entry["payload"]
        assert "domain" in payload
        assert "service" in payload
        assert "target" in payload
        assert "entity_id" in payload["target"]

    def test_schedule_json_validates(self, tmp_path):
        """set_appliance_schedule validates the schedule JSON schema."""
        agent = self._make_agent(tmp_path, granted_perms=["auto_control:laundry_ev"])
        schedule = {
            "schedule_id": "sched_test_001",
            "actions": [
                {
                    "at": "2026-04-20T02:30:00Z",
                    "call": {
                        "domain": "switch",
                        "service": "turn_on",
                        "target": {"entity_id": "switch.ev_charger"},
                    },
                }
            ],
        }
        result = agent.set_appliance_schedule(json.dumps(schedule))
        assert result["schedule_id"] == "sched_test_001"

    def test_dry_run_does_not_dispatch_real(self, tmp_path):
        """With dry_run=True, no actual HA calls should be made."""
        agent = self._make_agent(tmp_path, granted_perms=["auto_control:emergency_off"])
        result = agent.power_off("switch.iron")
        # Status must be dry_run, not "dispatched"
        assert result["status"] == "dry_run"
