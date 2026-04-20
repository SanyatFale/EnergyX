"""LangChain @tool: fetch_weather_forecast — Met Office / Open-Meteo UKMO."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def fetch_weather_forecast(location: str = "london") -> str:
    """Fetch a weather forecast for a UK location (next 24 hours, hourly).

    Primary source: Open-Meteo UKMO (free, no auth required).
    Falls back to a stub if the API is unavailable.

    Args:
        location: City or region name (used for display only; coordinates
                  are currently hardcoded to London — parameterise when
                  geocoding is added).

    Returns:
        JSON string with location, source, temperature_2m_next_24h (list of floats),
        and mean_temp_c.
    """
    try:
        from energyx.agents.knowledge.agent import KnowledgeAgent
        agent = KnowledgeAgent()
        result = agent.fetch_weather_forecast(location=location)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"fetch_weather_forecast failed: {e}")
        return json.dumps({"error": str(e), "location": location,
                           "source": "unavailable", "temperature_2m_next_24h": []})
