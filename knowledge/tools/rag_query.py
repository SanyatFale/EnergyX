"""LangChain @tool: rag_query — main Knowledge Agent RAG entry point.

Returns grounded text with source citations.
Tariff numbers are NOT returned from this tool — call get_active_tariff() instead.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def rag_query(query: str, jurisdiction: str = "england") -> str:
    """Answer a UK energy knowledge question using the indexed RAG corpus.

    Args:
        query: The user's question (e.g. "Am I eligible for ECO4?")
        jurisdiction: One of "england", "england_wales", "scotland", "ni", "uk"

    Returns:
        JSON string with keys: answer, sources, warnings.
        Tariff numbers (£/kWh) are never returned — use get_active_tariff() for those.
    """
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

        answer = synthesize(query, top_chunks)
        sources = [c.get("url", "") for c in top_chunks if c.get("url")]
        warnings = [c["_staleness_warning"] for c in top_chunks if "_staleness_warning" in c]

        return json.dumps({
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),  # deduplicate, preserve order
            "warnings": warnings,
            "chunks_retrieved": len(top_chunks),
        })

    except Exception as e:
        logger.error(f"rag_query failed: {e}")
        return json.dumps({
            "answer": (
                "The Knowledge Agent corpus is not yet indexed. "
                "Run knowledge/ingestion/ scripts to build the index. "
                "Authoritative source: https://www.ofgem.gov.uk"
            ),
            "sources": [],
            "warnings": [str(e)],
            "chunks_retrieved": 0,
        })
