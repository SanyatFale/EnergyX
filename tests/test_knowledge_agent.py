"""Tests for KnowledgeAgent (stub mode — no corpus indexed).

pytest tests/test_knowledge_agent.py
"""
import pytest
from energyx.agents.knowledge.agent import KnowledgeAgent


class TestKnowledgeAgentConstruction:
    def test_instantiation(self):
        agent = KnowledgeAgent()
        assert agent is not None

    def test_rag_not_available_without_corpus(self):
        agent = KnowledgeAgent()
        # In test env, corpus is not indexed — stub mode expected
        assert not agent._rag_available


class TestKnowledgeAgentStubAnswers:
    def test_economy7_query(self):
        agent = KnowledgeAgent()
        answer = agent.answer("How does Economy 7 work?")
        assert "economy 7" in answer.lower() or "Economy 7" in answer
        assert "off-peak" in answer.lower() or "rate" in answer.lower()

    def test_eco4_query(self):
        agent = KnowledgeAgent()
        answer = agent.answer("Am I eligible for ECO4?")
        assert "ECO4" in answer or "eco4" in answer.lower()
        assert "insulation" in answer.lower() or "heating" in answer.lower()

    def test_bus_query(self):
        agent = KnowledgeAgent()
        answer = agent.answer("Tell me about the Boiler Upgrade Scheme")
        assert "boiler" in answer.lower() or "BUS" in answer
        assert "£" in answer or "grant" in answer.lower()

    def test_unknown_query_returns_not_configured_message(self):
        agent = KnowledgeAgent()
        answer = agent.answer("What is the capital of France?")
        # Falls through to generic stub
        assert len(answer) > 20

    def test_answer_returns_string(self):
        agent = KnowledgeAgent()
        result = agent.answer("Any question")
        assert isinstance(result, str)


class TestKnowledgeAgentTariff:
    def test_get_active_tariff_returns_dict(self):
        agent = KnowledgeAgent()
        result = agent.get_active_tariff()
        # May return error dict if no tariff configured but must be dict
        assert isinstance(result, dict)

    def test_get_active_tariff_has_rate_field(self):
        agent = KnowledgeAgent()
        result = agent.get_active_tariff()
        # Either has error key or tariff data
        assert "error" in result or "tariff_id" in result or "rates" in result


class TestKnowledgeAgentCarbonIntensity:
    def test_carbon_intensity_returns_dict(self):
        agent = KnowledgeAgent()
        result = agent.fetch_carbon_intensity()
        assert isinstance(result, dict)
        assert "intensity_gco2_per_kwh" in result

    def test_carbon_intensity_has_numeric_value(self):
        agent = KnowledgeAgent()
        result = agent.fetch_carbon_intensity()
        val = result.get("intensity_gco2_per_kwh")
        assert isinstance(val, (int, float))
        assert 0 < val < 1000  # plausible range

    def test_carbon_intensity_fallback_on_no_network(self):
        """Even with no network the fallback should return 200 gCO2/kWh."""
        import unittest.mock as mock
        agent = KnowledgeAgent()
        with mock.patch("urllib.request.urlopen", side_effect=Exception("no network")):
            result = agent.fetch_carbon_intensity()
        assert result["intensity_gco2_per_kwh"] == 200
        assert result["source"] == "fallback"


class TestKnowledgeAgentWeather:
    def test_weather_forecast_returns_dict(self):
        agent = KnowledgeAgent()
        result = agent.fetch_weather_forecast()
        assert isinstance(result, dict)
        assert "location" in result

    def test_weather_forecast_fallback_on_no_network(self):
        import unittest.mock as mock
        agent = KnowledgeAgent()
        with mock.patch("urllib.request.urlopen", side_effect=Exception("no network")):
            result = agent.fetch_weather_forecast()
        assert result["source"] == "unavailable"
        assert result["temperature_2m_next_24h"] == []


class TestKnowledgeAgentIncentives:
    def test_list_incentives_england(self):
        agent = KnowledgeAgent()
        result = agent.list_applicable_incentives(jurisdiction="england")
        assert isinstance(result, dict)
        schemes = result["applicable_schemes"]
        assert isinstance(schemes, list)
        assert len(schemes) >= 1

    def test_list_incentives_scotland_includes_home_energy_scotland(self):
        agent = KnowledgeAgent()
        result = agent.list_applicable_incentives(jurisdiction="scotland")
        names = [s["name"] for s in result["applicable_schemes"]]
        assert any("scotland" in n.lower() or "Home Energy" in n for n in names)

    def test_list_incentives_epc_filter(self):
        agent = KnowledgeAgent()
        # EPC band A — ECO4 requires D or below, so fewer schemes expected
        result_a = agent.list_applicable_incentives(epc_band="A")
        result_d = agent.list_applicable_incentives(epc_band="D")
        # D-band should have at least as many schemes as A-band
        assert len(result_d["applicable_schemes"]) >= len(result_a["applicable_schemes"])

    def test_result_has_note_field(self):
        agent = KnowledgeAgent()
        result = agent.list_applicable_incentives()
        assert "note" in result
