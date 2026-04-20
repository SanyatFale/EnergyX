"""LangChain @tool: list_applicable_incentives — UK energy scheme eligibility."""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def list_applicable_incentives(epc_band: Optional[str] = None, jurisdiction: str = "england") -> str:
    """List UK energy incentive schemes applicable to the household.

    Args:
        epc_band: EPC energy rating (A–G). If provided, filters to schemes
                  available at that band. ECO4 requires band D or below.
        jurisdiction: One of "england", "england_wales", "scotland", "ni", "uk".

    Returns:
        JSON string with applicable_schemes list. Each scheme has name, amount, url,
        jurisdiction, and epc_requirement fields.

    Note:
        Amounts and deadlines shown are informational. Always verify at the scheme URL
        before advising a household to apply.
    """
    try:
        from energyx.agents.knowledge.agent import KnowledgeAgent
        agent = KnowledgeAgent()
        result = agent.list_applicable_incentives(epc_band=epc_band, jurisdiction=jurisdiction)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"list_applicable_incentives failed: {e}")
        return json.dumps({"error": str(e), "applicable_schemes": []})
