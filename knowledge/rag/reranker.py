"""Reranker: orders candidates by RRF score from hybrid retrieval."""
from __future__ import annotations

from typing import Any, Dict, List


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top_n candidates sorted by RRF score.

    At corpus size ~50 chunks, hybrid RRF ordering is sufficient without
    a cross-encoder — saves ~300ms per query and one model load.
    """
    if not candidates:
        return []
    return sorted(candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)[:top_n]
