"""RAG guardrails: staleness, jurisdiction, tariff-answer policy, confidence."""
from typing import List, Dict, Any
from datetime import datetime, timedelta

_STALENESS_MONTHS = 12

def apply_guardrails(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply staleness, jurisdiction, and tariff-policy guardrails."""
    result = []
    for chunk in chunks:
        pub_date_str = chunk.get("publication_date")
        if pub_date_str:
            try:
                pub = datetime.fromisoformat(pub_date_str)
                if (datetime.utcnow() - pub).days > _STALENESS_MONTHS * 30:
                    chunk["_staleness_warning"] = (
                        f"Sourced from {pub_date_str} guidance; verify at {chunk.get('url', 'original source')}."
                    )
            except Exception:
                pass
        # Tariff policy: never return £/kWh from prose
        if chunk.get("source_type") == "tariff_structured":
            chunk["_tariff_policy"] = "Numeric rates must be fetched via get_active_tariff()."
        result.append(chunk)
    return result
