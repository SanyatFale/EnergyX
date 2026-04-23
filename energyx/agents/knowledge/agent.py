"""Knowledge Agent — RAG pipeline + live API tools.

RAG corpus is not yet loaded (docs unavailable at build time).
The RAG chain will return an informative stub until the corpus is indexed.

Live API tools (weather, carbon intensity) are stubbed with a clear
"configure API key" message.

Per RAG_PLAN.md §3.3:
  - Tariff numbers NEVER come from RAG prose → always from get_active_tariff()
  - Staleness guard: warn if chunk > 12 months old
  - Jurisdiction guard: flag Scotland/NI vs England/Wales differences
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_NOT_CONFIGURED_MSG = (
    "The Knowledge Agent RAG corpus has not been indexed yet. "
    "Run the ingestion scripts in knowledge/ingestion/ to build the index. "
    "For authoritative answers, consult: https://www.ofgem.gov.uk and https://www.gov.uk/energy."
)


class KnowledgeAgent:
    """Answers UK energy knowledge queries via RAG + live APIs.

    When the corpus is indexed (knowledge/index/chroma_store/ exists),
    this switches from stub mode to full RAG retrieval.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._rag_available = self._check_rag()

    def answer(self, query: str) -> str:
        """Main entry: answer a query using RAG + live APIs."""
        if not self._rag_available:
            return self._stub_answer(query)
        try:
            return self._rag_answer(query)
        except Exception as e:
            logger.error(f"KnowledgeAgent RAG failed: {e}")
            return self._stub_answer(query)

    def get_active_tariff(self, household_id: Optional[str] = None) -> Dict[str, Any]:
        """Return active tariff as structured data (never from RAG prose)."""
        try:
            from energyx.data.tariffs import get_tariff_cache
            tariff = get_tariff_cache().get_active_tariff(household_id)
            if tariff:
                return tariff.to_dict()
            return {"error": "No tariff configured"}
        except Exception as e:
            return {"error": str(e)}

    def fetch_carbon_intensity(self, region: str = "national") -> Dict[str, Any]:
        """Call National Grid ESO Carbon Intensity API (no auth required)."""
        try:
            import urllib.request
            url = "https://api.carbonintensity.org.uk/intensity"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            intensity = data["data"][0]["intensity"]
            return {
                "region": region,
                "intensity_gco2_per_kwh": intensity.get("actual") or intensity.get("forecast", 200),
                "index": intensity.get("index", "moderate"),
                "source": "national_grid_eso",
            }
        except Exception as e:
            logger.warning(f"Carbon intensity API unavailable: {e}")
            return {
                "intensity_gco2_per_kwh": 200,
                "index": "moderate",
                "source": "fallback",
                "note": str(e),
            }

    def list_applicable_incentives(self, epc_band: Optional[str] = None, jurisdiction: str = "england") -> Dict[str, Any]:
        """List UK energy incentive schemes applicable to the household."""
        schemes = [
            {
                "name": "Boiler Upgrade Scheme (BUS)",
                "amount": "Up to £7,500 for ASHP/GSHP",
                "url": "https://www.ofgem.gov.uk/environmental-and-social-schemes/boiler-upgrade-scheme-bus",
                "jurisdiction": "england_wales",
                "epc_requirement": None,
            },
            {
                "name": "ECO4 (Energy Company Obligation)",
                "amount": "Free insulation / heating upgrades",
                "url": "https://www.ofgem.gov.uk/energy-company-obligation-eco",
                "jurisdiction": "uk",
                "epc_requirement": "D or below",
                "end_date": "31 December 2026",
            },
            {
                "name": "Smart Export Guarantee (SEG)",
                "amount": "Varies by supplier",
                "url": "https://www.ofgem.gov.uk/check-if-energy-company-has-to-offer-you-export-tariff",
                "jurisdiction": "uk",
                "epc_requirement": None,
            },
        ]

        if jurisdiction.lower() == "scotland":
            schemes = [s for s in schemes if s["jurisdiction"] in ("uk", "scotland")]
            schemes.append({
                "name": "Home Energy Scotland",
                "amount": "Free advice + grants",
                "url": "https://www.homeenergyscotland.org",
                "jurisdiction": "scotland",
                "epc_requirement": None,
            })

        if epc_band:
            filtered = [s for s in schemes
                        if not s.get("epc_requirement")
                        or epc_band.upper() <= s["epc_requirement"][0].upper()]
        else:
            filtered = schemes

        return {
            "jurisdiction": jurisdiction,
            "epc_band": epc_band,
            "applicable_schemes": filtered,
            "note": "Verify eligibility and current amounts at scheme URLs before applying.",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_rag(self) -> bool:
        """Return True if the vector store index exists."""
        from pathlib import Path
        chroma_dir = Path(__file__).parent.parent.parent.parent / "knowledge" / "index" / "chroma_store"
        return chroma_dir.exists() and any(chroma_dir.iterdir())

    def _stub_answer(self, query: str) -> str:
        # Try to answer common questions with built-in knowledge
        q = query.lower()
        if "economy 7" in q:
            return (
                "Economy 7 is a time-of-use electricity tariff with two rates: "
                "a cheaper off-peak rate (typically 00:30–07:30) and a higher peak rate. "
                "It suits households with storage heaters or EV charging overnight. "
                "For current rates, use get_active_tariff() — rates are not available from RAG prose."
            )
        if "eco4" in q:
            return (
                "ECO4 (Energy Company Obligation 4) provides free insulation and heating upgrades "
                "for eligible low-income households (typically EPC band D or below). "
                "Extended to 31 December 2026 per the January 2026 government response. "
                "See: https://www.ofgem.gov.uk/energy-company-obligation-eco"
            )
        if "bus" in q or "boiler upgrade" in q:
            return (
                "The Boiler Upgrade Scheme (BUS) provides grants for heat pumps: "
                "£7,500 for air-source and ground-source heat pumps; £5,000 for biomass boilers. "
                "Available in England and Wales only. "
                "See: https://www.ofgem.gov.uk/environmental-and-social-schemes/boiler-upgrade-scheme-bus"
            )
        return f"{_NOT_CONFIGURED_MSG}\n\nYour query: '{query}'"

    def _rag_answer(self, query: str) -> str:
        """Full RAG answer — only called when chroma_store exists."""
        try:
            from knowledge.rag.source_selector import select_sources
            from knowledge.rag.hybrid_retriever import retrieve
            from knowledge.rag.reranker import rerank
            from knowledge.rag.synthesizer import synthesize
            from knowledge.rag.guardrails import apply_guardrails

            collections = select_sources(query)
            candidates = retrieve(query, collections, top_k=10)
            top_chunks = rerank(query, candidates, top_n=5)
            top_chunks = apply_guardrails(query, top_chunks)
            return synthesize(query, top_chunks)
        except ImportError:
            return self._stub_answer(query)
