"""Minimal LLM-as-judge for synthesis faithfulness evaluation.

Design: single YES/NO question to a judge model, keeping token cost at ~20 tokens
per evaluation call. Based on G-Eval (Liu et al., 2023) but stripped to binary
faithfulness — we don't need gradient scores when running across many model pairs.

Judge model: defaults to cerebras/llama3.3-70b (strongest available locally).
Using a stronger judge than the models under test prevents evaluation bias.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_FAITHFULNESS_PROMPT = """\
Context: {context}

Query: {query}

Response: {response}

Does the response answer the query using ONLY information in the context, without hallucinating facts not present there?
Reply YES or NO, then a reason in ≤15 words."""

_COMPLETENESS_PROMPT = """\
Query: {query}

Response: {response}

Does the response directly and fully address what the query is asking?
Reply YES or NO, then a reason in ≤15 words."""


def _build_judge_llm():
    """Use the strongest available Cerebras model as judge."""
    from benchmark.models import get_model, get_llm_for_model
    judge_cfg = get_model("cerebras/llama3.3-70b")
    if judge_cfg is None:
        # Fallback to 8B if 70B not available
        from benchmark.models import CEREBRAS_MODELS
        judge_cfg = CEREBRAS_MODELS[0]
    return get_llm_for_model(judge_cfg, temperature=0.0, max_tokens=20)


def faithfulness_score(
    response: str,
    context: str,
    query: str,
    llm=None,
    retries: int = 2,
) -> dict:
    """Return {score: 1|0, raw: str, error: str|None}.

    score=1 → judge says YES (faithful)
    score=0 → judge says NO (hallucination detected)
    """
    if llm is None:
        try:
            llm = _build_judge_llm()
        except Exception as e:
            return {"score": -1, "raw": "", "error": f"judge init failed: {e}"}

    prompt = _FAITHFULNESS_PROMPT.format(
        context=context[:800],   # cap context to avoid prompt bloat
        query=query,
        response=response[:400],
    )

    for attempt in range(retries + 1):
        try:
            resp = llm.invoke(prompt)
            raw = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            score = 1 if raw.upper().startswith("YES") else 0
            return {"score": score, "raw": raw, "error": None}
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return {"score": -1, "raw": "", "error": str(e)}

    return {"score": -1, "raw": "", "error": "max retries exceeded"}


def completeness_score(
    response: str,
    query: str,
    llm=None,
    retries: int = 2,
) -> dict:
    """Return {score: 1|0, raw: str, error: str|None}."""
    if llm is None:
        try:
            llm = _build_judge_llm()
        except Exception as e:
            return {"score": -1, "raw": "", "error": f"judge init failed: {e}"}

    prompt = _COMPLETENESS_PROMPT.format(query=query, response=response[:400])

    for attempt in range(retries + 1):
        try:
            resp = llm.invoke(prompt)
            raw = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            score = 1 if raw.upper().startswith("YES") else 0
            return {"score": score, "raw": raw, "error": None}
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return {"score": -1, "raw": "", "error": str(e)}

    return {"score": -1, "raw": "", "error": "max retries exceeded"}
