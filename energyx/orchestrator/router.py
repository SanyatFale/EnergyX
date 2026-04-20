"""Query classifier — routes user queries to the correct agent.

Returns one of: knowledge | analysis | control | status
Temperature 0.1 (deterministic routing).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

QueryIntent = Literal["knowledge", "analysis", "control", "status"]

_CLASSIFIER_PROMPT = """Classify the following user query into exactly one of these intents:

- knowledge   : Questions about UK energy regulations, tariffs, schemes (ECO4, BUS, SEG),
                 how things work, advice about suppliers, carbon intensity, EPC, weather.
- analysis    : Forecasting, anomaly explanation, what-if scenarios, bill prediction,
                 causal attribution, scheduling recommendations, reports.
- control     : Direct device commands — turn on/off, set temperature, schedule appliance.
- status      : Current readings, live cost, carbon, budget, recent events, monitoring status.

Query: {query}

Respond with ONLY the intent word (knowledge, analysis, control, or status).
"""

# Keyword heuristics for fallback when LLM unavailable
_ANALYSIS_KW = re.compile(
    r"\b(forecast|predict|anomaly|anomalies|explain the|what.?if|counterfactual|"
    r"causal|attribution|schedule.*load|elasticity|degradation|report|projection)\b",
    re.IGNORECASE,
)
_KNOWLEDGE_KW = re.compile(
    r"\b(eco4|bus|seg|grant|scheme|eligible|regulation|tariff|how does|what is|what does|"
    r"economy 7|agile|standing charge|epc|ofgem|supplier|switch|insulation|heat pump"
    r"|rebate|vat|smart meter|sensor|dataset|mean|record)\b",
    re.IGNORECASE,
)
_CONTROL_KW = re.compile(
    r"\b(turn on|turn off|switch on|switch off|set temperature|setpoint|thermostat|"
    r"schedule|defer|charge now|boost|cancel)\b",
    re.IGNORECASE,
)
_STATUS_KW = re.compile(
    r"\b(current|live|now|right now|real.?time|how much.*using|status|latest"
    r"|monitoring|alerts|events|budget|spend so far)\b",
    re.IGNORECASE,
)


def classify_query(query: str, use_llm: bool = True) -> QueryIntent:
    """Classify query intent. Falls back to keyword heuristics on LLM failure."""
    if use_llm:
        try:
            from tinyts.config import get_llm, settings
            llm = get_llm(temperature=0.1)
            prompt = _CLASSIFIER_PROMPT.format(query=query)
            resp = llm.invoke(prompt)
            content = (resp.content if hasattr(resp, "content") else str(resp)).strip().lower()
            # Extract intent word
            for intent in ("knowledge", "analysis", "control", "status"):
                if intent in content:
                    logger.info(f"Router: '{query[:60]}' → {intent} (LLM)")
                    return intent  # type: ignore
        except Exception as e:
            logger.warning(f"Router LLM failed, using heuristics: {e}")

    # Keyword heuristic fallback: control → analysis → status → knowledge
    # Analysis checked before knowledge to prevent "explain the anomaly" → knowledge
    # Status checked before knowledge to prevent "what is my current spend?" → knowledge
    if _CONTROL_KW.search(query):
        intent = "control"
    elif _ANALYSIS_KW.search(query):
        intent = "analysis"
    elif _STATUS_KW.search(query):
        intent = "status"
    elif _KNOWLEDGE_KW.search(query):
        intent = "knowledge"
    else:
        intent = "analysis"

    logger.info(f"Router: '{query[:60]}' → {intent} (heuristic)")
    return intent  # type: ignore
