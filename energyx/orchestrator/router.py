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

# Keyword heuristics — ordered by specificity (checked before LLM for strong matches)
_ANALYSIS_KW = re.compile(
    r"\b(forecast|predict|anomaly|anomalies|explain the|what.?if|counterfactual|"
    r"causal|attribution|schedule.*load|elasticity|degradation|report|projection|"
    r"budget|on track|bill|spend|saving|tariff switch|evaluate tariff|compare tariff|"
    r"compare.*rate|compare.*tariff|vs economy|vs agile|switch tariff|would i save|"
    r"replay|alternative tariff|reduce.*bill|bill.*reduc|changes.*bill|what.*change.*bill|"
    r"save.*£|£.*save|heating.*%|consumption.*change)\b",
    re.IGNORECASE,
)
_KNOWLEDGE_KW = re.compile(
    r"\b(eco4|bus scheme|seg scheme|grant|eligible|regulation|how does|what is a|what does|"
    r"economy 7|agile octopus|standing charge|epc|ofgem|supplier|insulation|heat pump"
    r"|rebate|vat|smart meter)\b",
    re.IGNORECASE,
)
_CONTROL_KW = re.compile(
    r"\b(turn on|turn off|switch on|switch off|set temperature|setpoint|thermostat|"
    r"defer|charge now|boost|cancel schedule)\b",
    re.IGNORECASE,
)
_STATUS_KW = re.compile(
    r"\b(current reading|live reading|right now|real.?time|how much.*using|"
    r"monitoring status|latest reading|recent events|spend so far)\b",
    re.IGNORECASE,
)


def classify_query(query: str, use_llm: bool = True) -> QueryIntent:
    """Classify query intent.

    Heuristics run first for strong keyword matches — llama3.2:3b misclassifies
    clear analysis queries (e.g. 'forecast...') as knowledge too often.
    LLM is used only when no strong heuristic pattern fires.
    """
    # Strong heuristic pre-check: control and analysis patterns are high-confidence
    if _CONTROL_KW.search(query):
        logger.info(f"Router: '{query[:60]}' → control (heuristic-pre)")
        return "control"
    if _ANALYSIS_KW.search(query):
        logger.info(f"Router: '{query[:60]}' → analysis (heuristic-pre)")
        return "analysis"

    # For ambiguous queries, try LLM
    if use_llm:
        try:
            from tinyts.config import get_llm, settings
            llm = get_llm(temperature=0.1)
            prompt = _CLASSIFIER_PROMPT.format(query=query)
            resp = llm.invoke(prompt)
            content = (resp.content if hasattr(resp, "content") else str(resp)).strip().lower()
            for intent in ("knowledge", "analysis", "control", "status"):
                if intent in content:
                    logger.info(f"Router: '{query[:60]}' → {intent} (LLM)")
                    return intent  # type: ignore
        except Exception as e:
            logger.warning(f"Router LLM failed, using heuristics: {e}")

    # Heuristic fallback for remaining cases
    if _STATUS_KW.search(query):
        intent = "status"
    elif _KNOWLEDGE_KW.search(query):
        intent = "knowledge"
    else:
        intent = "analysis"

    logger.info(f"Router: '{query[:60]}' → {intent} (heuristic)")
    return intent  # type: ignore
