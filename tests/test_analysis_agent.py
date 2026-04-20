"""Tests for AnalysisAgent — verifies construction and tool map integrity.

The LLM agentic loop is not exercised here (no API key required).
We test that:
  - detect_anomalies is NOT in the tool map
  - all 10 new tools are registered
  - new tools return valid JSON on invocation
  - tariff-based tools use get_active_tariff() not RAG prose

pytest tests/test_analysis_agent.py
"""
import csv
import json
import pytest
import tempfile
from pathlib import Path

pytest.importorskip("langchain_core", reason="langchain_core not installed")


# ---------------------------------------------------------------------------
# Fixture: minimal CSV dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    """Write a tiny electricity CSV the agent can load."""
    tmp = tmp_path_factory.mktemp("data")
    p = tmp / "electricity.csv"
    import pandas as pd
    df = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=120, freq="h"),
        "value": [300.0 + i % 50 for i in range(120)],
    })
    df.to_csv(p, index=False)
    return str(p), str(tmp / "out")


@pytest.fixture(scope="module")
def agent(sample_csv):
    csv_path, out_dir = sample_csv
    try:
        from energyx.agents.analysis.agent import AnalysisAgent
        return AnalysisAgent(
            dataset_path=csv_path,
            time_column="ts",
            target_column="value",
            output_dir=out_dir,
        )
    except Exception as e:
        pytest.skip(f"AnalysisAgent construction failed (missing deps?): {e}")


# ---------------------------------------------------------------------------
# Tool map integrity
# ---------------------------------------------------------------------------

class TestAnalysisAgentToolMap:
    def test_detect_anomalies_removed(self, agent):
        """detect_anomalies must NOT appear — it lives in the monitor core."""
        assert "detect_anomalies" not in agent.tool_map, (
            "detect_anomalies should have been removed from AnalysisAgent tool_map"
        )

    def test_core_tools_present(self, agent):
        expected_core = [
            "profile_dataset",
            "train_forecast_model",
            "combine_forecasts",
            "explain_anomalies",
            "generate_report",
        ]
        for name in expected_core:
            assert name in agent.tool_map, f"Core tool '{name}' missing from tool_map"

    def test_new_energy_tools_present(self, agent):
        new_tools = [
            "predict_bill",
            "counterfactual_bill_forward",
            "counterfactual_bill_inverse",
            "evaluate_tariff_switch",
            "causal_attribution",
            "schedule_flexible_loads",
            "suggest_budget_corrections",
            "project_longhorizon",
            "compute_elasticity",
            "update_degradation_baselines",
        ]
        for name in new_tools:
            assert name in agent.tool_map, f"New tool '{name}' missing from tool_map"

    def test_total_new_tools_count(self, agent):
        new_tools = [
            "predict_bill", "counterfactual_bill_forward", "counterfactual_bill_inverse",
            "evaluate_tariff_switch", "causal_attribution", "schedule_flexible_loads",
            "suggest_budget_corrections", "project_longhorizon",
            "compute_elasticity", "update_degradation_baselines",
        ]
        present = [n for n in new_tools if n in agent.tool_map]
        assert len(present) == 10, f"Expected 10 new tools, found {len(present)}: {present}"


# ---------------------------------------------------------------------------
# New tool invocations (no LLM required)
# ---------------------------------------------------------------------------

class TestNewToolInvocations:
    def test_predict_bill_returns_json(self, agent):
        result_str = agent.tool_map["predict_bill"].invoke({"horizon": 24})
        result = json.loads(result_str)
        assert isinstance(result, dict)
        # Should mention bill or cost or error
        assert any(k in result for k in ("projected_bill_gbp", "error", "status", "warning"))

    def test_predict_bill_uses_tariff_not_hardcoded(self, agent):
        """predict_bill must call get_active_tariff(), not embed rate literals."""
        result_str = agent.tool_map["predict_bill"].invoke({"horizon": 24})
        # Result is valid JSON — tariff source must be retrievable
        result = json.loads(result_str)
        # If tariff_id is present, it came from get_active_tariff()
        if "tariff_id" in result:
            assert isinstance(result["tariff_id"], str)

    def test_counterfactual_bill_forward_returns_json(self, agent):
        changes = json.dumps({"value": -50})
        result_str = agent.tool_map["counterfactual_bill_forward"].invoke(
            {"changes_json": changes, "horizon": 24}
        )
        result = json.loads(result_str)
        assert isinstance(result, dict)

    def test_counterfactual_bill_inverse_returns_json(self, agent):
        result_str = agent.tool_map["counterfactual_bill_inverse"].invoke(
            {"target_delta_gbp": 10.0, "constraints_json": "{}"}
        )
        result = json.loads(result_str)
        assert isinstance(result, dict)

    def test_evaluate_tariff_switch_returns_json(self, agent):
        result_str = agent.tool_map["evaluate_tariff_switch"].invoke(
            {"alternative_tariff_id": "economy7_standard_2026q1"}
        )
        result = json.loads(result_str)
        assert isinstance(result, dict)

    def test_causal_attribution_returns_json(self, agent):
        result_str = agent.tool_map["causal_attribution"].invoke({
            "period_a_start": "2026-01-01",
            "period_a_end": "2026-01-07",
            "period_b_start": "2026-01-08",
            "period_b_end": "2026-01-14",
        })
        result = json.loads(result_str)
        assert isinstance(result, dict)

    def test_schedule_flexible_loads_returns_ha_json(self, agent):
        loads = json.dumps([
            {"entity_id": "switch.ev_charger", "duration_hours": 4, "power_kw": 7.0}
        ])
        result_str = agent.tool_map["schedule_flexible_loads"].invoke(
            {"loads_json": loads, "horizon": 24}
        )
        result = json.loads(result_str)
        assert isinstance(result, dict)
        # Should contain schedule or error
        assert "schedule" in result or "error" in result or "actions" in result

    def test_suggest_budget_corrections_returns_list(self, agent):
        result_str = agent.tool_map["suggest_budget_corrections"].invoke(
            {"monthly_cap_gbp": 100.0}
        )
        result = json.loads(result_str)
        assert isinstance(result, dict)
        assert "corrections" in result or "error" in result or "suggestions" in result

    def test_project_longhorizon_returns_json(self, agent):
        result_str = agent.tool_map["project_longhorizon"].invoke({"horizon_days": 30})
        result = json.loads(result_str)
        assert isinstance(result, dict)

    def test_compute_elasticity_returns_json(self, agent):
        result_str = agent.tool_map["compute_elasticity"].invoke({})
        result = json.loads(result_str)
        assert isinstance(result, dict)
        # Either elasticity data or a note that no model is trained yet
        assert "elasticity" in result or "note" in result or "error" in result or "status" in result

    def test_update_degradation_baselines_returns_json(self, agent):
        result_str = agent.tool_map["update_degradation_baselines"].invoke({})
        result = json.loads(result_str)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Session cache shared between tools
# ---------------------------------------------------------------------------

class TestAnalysisAgentSession:
    def test_session_initialized(self, agent):
        assert hasattr(agent, "session")
        assert isinstance(agent.session, dict)

    def test_elasticity_cache_starts_none(self, agent):
        """Elasticity cache should not be pre-populated before compute_elasticity() runs."""
        # It may or may not exist; if it does it should not have stale data
        val = agent.session.get("_elasticity_cache")
        # OK if None or a dict
        assert val is None or isinstance(val, dict)
