"""Cross-encoder reranker. Falls back to RRF score ordering if model unavailable."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_cross_encoder: Optional[Any] = None
_ce_available: Optional[bool] = None  # None = not yet tried


def _get_cross_encoder():
    global _cross_encoder, _ce_available
    if _ce_available is False:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        from knowledge.config import cfg
        _cross_encoder = CrossEncoder(cfg.reranker_model, max_length=512)
        _ce_available = True
        logger.info(f"Cross-encoder loaded: {cfg.reranker_model}")
    except Exception as e:
        logger.warning(f"Cross-encoder not available ({e}); using RRF scores")
        _ce_available = False
    return _cross_encoder


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Score candidates with cross-encoder and return top_n.

    Falls back to RRF score ordering when the cross-encoder model is unavailable.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    if model is None:
        # Fallback: sort by RRF score already computed by hybrid_retriever
        return sorted(candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)[:top_n]

    try:
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)
        ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return ranked[:top_n]
    except Exception as e:
        logger.warning(f"Cross-encoder scoring failed ({e}); falling back to RRF")
        return sorted(candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)[:top_n]
