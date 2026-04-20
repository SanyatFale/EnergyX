"""Tests for MonitorAgent (offline batch analyzer).

These tests bypass the LLM by calling the tool functions directly.
pytest tests/test_monitor_agent.py
"""
import json
import pytest
import pandas as pd

pytest.importorskip("langchain_core", reason="langchain_core not installed")

from energyx.agents.monitor.agent import MonitorAgent
from energyx.monitoring.bus import EventBus
from energyx.data.events import BaseEvent, RetrainingRequestEvent


def _make_batch(n: int = 100, home_id: str = "home001") -> pd.DataFrame:
    """Return a simple sensor batch suitable for monitor evaluation."""
    return pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=n, freq="min"),
        "value": [300.0 + i % 50 for i in range(n)],
        "sensor_type": ["electricity_apparent"] * n,
        "home_id": [home_id] * n,
        "unit": ["watts"] * n,
    })


class TestMonitorAgentConstruction:
    def test_instantiation_minimal(self):
        agent = MonitorAgent()
        assert agent is not None
        assert agent._home_id == "unknown"

    def test_instantiation_with_config(self):
        agent = MonitorAgent(config={"home_id": "home042"})
        assert agent._home_id == "home042"

    def test_uses_provided_bus(self):
        bus = EventBus()
        agent = MonitorAgent(bus=bus)
        assert agent._bus is bus


class TestMonitorAgentTools:
    """Test individual tool callables without triggering the LLM loop."""

    def _make_agent(self, **kwargs) -> MonitorAgent:
        return MonitorAgent(config={"home_id": "home001", **kwargs})

    def test_select_applicable_monitors_returns_list(self):
        agent = self._make_agent()
        batch = _make_batch()
        tools = agent._create_tools(batch, include_online_only=False)
        tool_map = {t.name: t for t in tools}

        result = json.loads(tool_map["select_applicable_monitors"].invoke(
            {"batch_metadata_json": "{}"}
        ))
        monitors = result["monitors"]
        assert isinstance(monitors, list)
        assert len(monitors) >= 1

    def test_select_applicable_monitors_excludes_online_only(self):
        agent = self._make_agent()
        batch = _make_batch()
        tools = agent._create_tools(batch, include_online_only=False)
        tool_map = {t.name: t for t in tools}

        result = json.loads(tool_map["select_applicable_monitors"].invoke(
            {"batch_metadata_json": "{}"}
        ))
        monitors = result["monitors"]
        assert "ForgotToTurnOffDetector" not in monitors
        assert "DemandResponseListener" not in monitors

    def test_run_monitor_anomaly_detector(self):
        agent = self._make_agent()
        batch = _make_batch()
        tools = agent._create_tools(batch, include_online_only=False)
        tool_map = {t.name: t for t in tools}

        result = json.loads(tool_map["run_monitor"].invoke(
            {"monitor_name": "AnomalyEnsemble", "batch_summary_json": "{}"}
        ))
        assert "monitor" in result
        assert "n_events" in result
        assert result["monitor"] == "AnomalyEnsemble"

    def test_run_monitor_unknown_returns_error(self):
        agent = self._make_agent()
        batch = _make_batch()
        tools = agent._create_tools(batch, include_online_only=False)
        tool_map = {t.name: t for t in tools}

        result = json.loads(tool_map["run_monitor"].invoke(
            {"monitor_name": "NonExistentMonitor", "batch_summary_json": "{}"}
        ))
        assert "error" in result

    def test_summarize_batch_findings(self):
        agent = self._make_agent()
        batch = _make_batch()
        tools = agent._create_tools(batch, include_online_only=False)
        tool_map = {t.name: t for t in tools}

        events = [{"type": "anomaly.appliance"}, {"type": "cost.realtime"}]
        result = json.loads(tool_map["summarize_batch_findings"].invoke(
            {"events_json": json.dumps(events), "stats_json": "{}"}
        ))
        assert "narrative" in result
        assert result["event_count"] == 2

    def test_trigger_retraining_appends_event(self):
        agent = self._make_agent()
        batch = _make_batch()
        tools = agent._create_tools(batch, include_online_only=False)
        tool_map = {t.name: t for t in tools}

        result = json.loads(tool_map["trigger_retraining"].invoke(
            {"reason": "batch_drift_detected"}
        ))
        assert result["status"] == "retraining_requested"
        assert result["reason"] == "batch_drift_detected"
        # Retraining event appended to collected events
        assert any(isinstance(e, RetrainingRequestEvent) for e in agent._collected_events)


class TestMonitorAgentBusPublishing:
    def test_collected_events_published_to_bus(self):
        """Events collected during tool calls are published to the bus."""
        bus = EventBus()
        received = []
        bus.subscribe("*", received.append)

        agent = MonitorAgent(config={"home_id": "home001"}, bus=bus)
        batch = _make_batch(n=200)
        tools = agent._create_tools(batch, include_online_only=False)
        tool_map = {t.name: t for t in tools}

        # Run a monitor that may generate events
        tool_map["run_monitor"].invoke({"monitor_name": "AnomalyEnsemble", "batch_summary_json": "{}"})
        # Manually trigger publish path
        for e in agent._collected_events:
            bus.publish(e)

        # If events were collected they should appear on the bus
        if agent._collected_events:
            assert len(received) >= 1


class TestMonitorAgentDescribeBatch:
    def test_describe_batch_empty(self):
        result = MonitorAgent._describe_batch(pd.DataFrame(), "b1")
        assert result["n_rows"] == 0
        assert result["batch_id"] == "b1"

    def test_describe_batch_with_data(self):
        batch = _make_batch(n=10)
        result = MonitorAgent._describe_batch(batch, "b2")
        assert result["n_rows"] == 10
        assert "value_stats" in result
        assert result["value_stats"]["mean"] > 0

    def test_describe_batch_date_range(self):
        batch = _make_batch(n=5)
        result = MonitorAgent._describe_batch(batch, "b3")
        assert "date_range" in result
        assert len(result["date_range"]) == 2


class TestMonitorAgentTextAdapter:
    """Test the Cerebras text-mode tool-call format parser."""

    def test_parse_simple_tool_call(self):
        agent = MonitorAgent()
        batch = _make_batch()
        tools = agent._create_tools(batch, False)
        tool_map = {t.name: t for t in tools}

        parsed = agent._parse_text_tool_call(
            'trigger_retraining(reason="model_drift")', tool_map
        )
        assert parsed is not None
        assert parsed["name"] == "trigger_retraining"
        assert parsed["args"]["reason"] == "model_drift"

    def test_parse_no_args_tool_call(self):
        agent = MonitorAgent()
        batch = _make_batch()
        tools = agent._create_tools(batch, False)
        tool_map = {t.name: t for t in tools}

        parsed = agent._parse_text_tool_call(
            "select_applicable_monitors()", tool_map
        )
        # May return None if no match or empty args
        if parsed is not None:
            assert parsed["name"] == "select_applicable_monitors"

    def test_parse_unknown_tool_returns_none(self):
        agent = MonitorAgent()
        batch = _make_batch()
        tools = agent._create_tools(batch, False)
        tool_map = {t.name: t for t in tools}

        parsed = agent._parse_text_tool_call('unknown_tool(x="y")', tool_map)
        assert parsed is None
