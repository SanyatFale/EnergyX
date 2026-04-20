"""LangChain @tool: fetch_carbon_intensity — National Grid ESO Carbon Intensity API."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def fetch_carbon_intensity(region: str = "national") -> str:
    """Fetch current UK grid carbon intensity (gCO2/kWh).

    Source: National Grid ESO Carbon Intensity API (free, no auth required).
    Falls back to 200 gCO2/kWh if the API is unavailable.

    Args:
        region: "national" (default) or a DNO region code. Regional API
                is available at /regional but not yet implemented — uses
                national intensity for all regions currently.

    Returns:
        JSON string with intensity_gco2_per_kwh, index ("very low"|"low"|
        "moderate"|"high"|"very high"), and source.
    """
    try:
        from energyx.agents.knowledge.agent import KnowledgeAgent
        agent = KnowledgeAgent()
        result = agent.fetch_carbon_intensity(region=region)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"fetch_carbon_intensity failed: {e}")
        return json.dumps({
            "intensity_gco2_per_kwh": 200,
            "index": "moderate",
            "source": "fallback",
            "error": str(e),
        })
