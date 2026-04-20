"""Phase 1 RAG: classify query → select which collections to search."""
from typing import List

def select_sources(query: str) -> List[str]:
    """Return collection names to search for this query. Stub implementation."""
    q = query.lower()
    collections = []
    if any(w in q for w in ["ideal", "sensor", "dataset", "csv", "electricity_combined"]):
        collections.append("ideal_docs")
    if any(w in q for w in ["eco4", "bus", "seg", "grant", "scheme", "regulation", "ofgem", "eligible"]):
        collections.append("regulatory")
    if any(w in q for w in ["save", "reduce", "insulation", "efficient", "heating"]):
        collections.append("efficiency_guide")
    if any(w in q for w in ["economy 7", "agile", "standing charge", "tariff", "unit rate", "kw"]):
        collections.append("domain_qa")
    return collections or ["domain_qa", "regulatory"]
