"""LangChain @tool: get_active_tariff — authoritative source for tariff numbers.

This is the ONLY tool that may return £/kWh rates.
RAG prose chunks must NOT be used for tariff rates.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_active_tariff(household_id: Optional[str] = None) -> str:
    """Return the active electricity tariff for a household as structured JSON.

    This is the ONLY authoritative source for unit rates (£/kWh) and standing charges.
    Never use RAG-retrieved prose for numeric tariff rates.

    Args:
        household_id: Optional household identifier. Uses system default if None.

    Returns:
        JSON string with tariff_id, structure, unit_rate_gbp_per_kwh,
        standing_charge_gbp_per_day, off_peak_rate, off_peak_window.
    """
    try:
        from energyx.data.tariffs import get_tariff_cache
        cache = get_tariff_cache()
        tariff = cache.get_active_tariff(household_id)
        if tariff is None:
            return json.dumps({"error": "No active tariff configured for this household."})
        return json.dumps(tariff.to_dict())
    except Exception as e:
        logger.error(f"get_active_tariff failed: {e}")
        return json.dumps({"error": str(e)})
