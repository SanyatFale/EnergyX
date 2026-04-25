"""Grounded answer synthesis from retrieved chunks.

Uses LLM when available (inherits from tinyts config), otherwise returns
a structured excerpt from the top-ranked chunks.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
You are an energy adviser assistant for UK households. Answer the question below using
ONLY the provided source excerpts. If the excerpts do not contain sufficient information,
say so clearly. Do not invent tariff rates or eligibility criteria — refer the user to
the official source URLs listed in the excerpts.

SOURCES:
{context}

QUESTION: {query}

ANSWER (concise, factual, grounded):"""


def synthesize(query: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return (
            "I don't have reliable information on this yet. "
            "Please check: https://www.ofgem.gov.uk or https://www.gov.uk/energy"
        )

    context_parts = []
    sources_seen: set[str] = set()
    for i, chunk in enumerate(chunks[:3], 1):
        meta = chunk.get("meta", {})
        url = meta.get("url", "")
        source_id = chunk.get("source_id", "")
        text = chunk.get("text", "")
        context_parts.append(f"[{i}] {text[:400]}")
        if url and url not in sources_seen:
            sources_seen.add(url)

    context = "\n\n".join(context_parts)
    warnings = [c["_staleness_warning"] for c in chunks if "_staleness_warning" in c]

    # Try LLM synthesis
    try:
        from tinyts.config import get_llm
        llm = get_llm(temperature=0.1)
        prompt = _SYNTHESIS_PROMPT.format(context=context, query=query)
        resp = llm.invoke(prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        logger.debug(f"LLM synthesis unavailable ({e}); using excerpt fallback")
        top = chunks[0]
        answer = top.get("text", "").strip()[:600]
        if len(top.get("text", "")) > 600:
            answer += "…"

    if sources_seen:
        answer += "\n\n**Sources:** " + "  |  ".join(sorted(sources_seen))
    if warnings:
        answer += "\n\n⚠️ " + "  ".join(set(warnings))

    return answer
