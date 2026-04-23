"""Tests for Step 3: Orchestrator.

pytest tests/test_orchestrator.py
"""
import pandas as pd
import pytest

from energyx.monitoring.bus import EventBus
from energyx.orchestrator.mode_manager import Mode, ModeManager
from energyx.orchestrator.permission_manager import PermissionManager, PermissionDenied
from energyx.orchestrator.router import classify_query
from energyx.orchestrator.orchestrator import Orchestrator


class TestModeManager:
    def test_default_offline(self):
        mm = ModeManager(Mode.OFFLINE)
        assert mm.is_offline()
        assert not mm.is_online()

    def test_switch_modes(self):
        mm = ModeManager(Mode.OFFLINE)
        mm.go_online()
        assert mm.is_online()
        mm.go_offline()
        assert mm.is_offline()

    def test_callback_on_change(self):
        history = []
        mm = ModeManager(Mode.OFFLINE)
        mm.on_mode_change(history.append)
        mm.go_online()
        assert history[-1] == Mode.ONLINE
        mm.go_offline()
        assert history[-1] == Mode.OFFLINE


class TestPermissionManager:
    def test_grant_and_check(self, tmp_path):
        pm = PermissionManager(path=tmp_path / "perms.json")
        assert not pm.is_granted("auto_control:hvac")
        pm.grant("auto_control:hvac")
        assert pm.is_granted("auto_control:hvac")

    def test_require_raises_when_not_granted(self, tmp_path):
        pm = PermissionManager(path=tmp_path / "perms.json")
        with pytest.raises(PermissionDenied):
            pm.require("auto_control:hvac")

    def test_require_passes_when_granted(self, tmp_path):
        pm = PermissionManager(path=tmp_path / "perms.json")
        pm.grant("auto_control:hvac")
        pm.require("auto_control:hvac")  # should not raise

    def test_revoke(self, tmp_path):
        pm = PermissionManager(path=tmp_path / "perms.json")
        pm.grant("dr_enrollment")
        pm.revoke("dr_enrollment")
        assert not pm.is_granted("dr_enrollment")

    def test_unknown_permission_raises(self, tmp_path):
        pm = PermissionManager(path=tmp_path / "perms.json")
        with pytest.raises(ValueError):
            pm.grant("made_up_permission")

    def test_all_permissions_dict(self, tmp_path):
        pm = PermissionManager(path=tmp_path / "perms.json")
        pm.grant("continuous_training")
        all_p = pm.all_permissions()
        assert all_p["continuous_training"] is True
        assert all_p["auto_control:hvac"] is False


class TestQueryRouter:
    def test_knowledge_queries(self):
        assert classify_query("Am I eligible for ECO4?", use_llm=False) == "knowledge"
        assert classify_query("How does Economy 7 work?", use_llm=False) == "knowledge"

    def test_control_queries(self):
        assert classify_query("Turn off the iron", use_llm=False) == "control"
        assert classify_query("Set temperature to 19°C", use_llm=False) == "control"

    def test_status_queries(self):
        assert classify_query("What is my current spend?", use_llm=False) == "status"
        assert classify_query("Show me live monitoring status", use_llm=False) == "status"

    def test_analysis_queries(self):
        assert classify_query("Forecast next 7 days", use_llm=False) == "analysis"
        assert classify_query("Explain the anomaly last Tuesday", use_llm=False) == "analysis"

    def test_budget_queries_route_to_analysis(self):
        assert classify_query("Am I on track with my £120/month budget?", use_llm=False) == "analysis"
        assert classify_query("Check my monthly budget", use_llm=False) == "analysis"

    def test_bill_queries_route_to_analysis(self):
        assert classify_query("What changes reduce my bill by £40/month?", use_llm=False) == "analysis"
        assert classify_query("If heating +20%, how does my bill change?", use_llm=False) == "analysis"
        assert classify_query("How will my bill change if consumption increases?", use_llm=False) == "analysis"

    def test_tariff_queries_route_to_analysis(self):
        assert classify_query("Replay consumption against alternative tariffs", use_llm=False) == "analysis"
        assert classify_query("Evaluate tariff switch to Economy 7", use_llm=False) == "analysis"
        assert classify_query("Compare my tariff vs Economy 7", use_llm=False) == "analysis"
        assert classify_query("Would I save on Agile Octopus?", use_llm=False) == "analysis"

    def test_economy7_question_routes_to_knowledge_not_analysis(self):
        # "How does Economy 7 work?" is a knowledge question, not a tariff switch
        assert classify_query("How does Economy 7 work?", use_llm=False) == "knowledge"


class TestOrchestrator:
    def test_offline_mode_by_default(self, tmp_path):
        orch = Orchestrator(
            config={"home_id": "home001"},
            permissions_path=tmp_path / "perms.json",
        )
        assert orch.mode_manager.is_offline()

    def test_status_query_returns_mode(self, tmp_path):
        orch = Orchestrator(
            config={"home_id": "home001"},
            permissions_path=tmp_path / "perms.json",
        )
        result = orch.handle_query("show monitoring status")
        assert result["intent"] == "status"
        assert "mode" in result

    def test_control_disabled_in_offline(self, tmp_path):
        orch = Orchestrator(
            config={"home_id": "home001"},
            permissions_path=tmp_path / "perms.json",
            initial_mode=Mode.OFFLINE,
        )
        result = orch.handle_query("turn off the iron")
        assert result["intent"] == "control"
        assert result.get("advisory") is True

    def test_append_batch_triggers_monitor_agent(self, tmp_path):
        """Batch append returns a result dict with expected keys."""
        orch = Orchestrator(
            config={"home_id": "home001"},
            permissions_path=tmp_path / "perms.json",
        )
        # Don't actually call Monitor Agent (needs LLM); just verify the interface
        batch = pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=5, freq="min"),
            "value": [300.0] * 5,
        })
        # Patch the monitor agent to avoid LLM call
        class StubMonitorAgent:
            def run(self, *a, **kw):
                return {"events": [], "narrative": "stub", "retraining_triggered": False, "batch_id": "b1"}
        orch._monitor_agent = StubMonitorAgent()
        result = orch.append_batch(batch, batch_id="test_batch")
        assert "events" in result
        assert "narrative" in result
        assert "batch_id" in result

    def test_permission_gating(self, tmp_path):
        orch = Orchestrator(
            config={"home_id": "home001"},
            permissions_path=tmp_path / "perms.json",
        )
        # Without auto_control:hvac, control should be advisory or refused
        assert not orch.permissions.is_granted("auto_control:hvac")
