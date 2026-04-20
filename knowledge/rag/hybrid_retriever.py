"""Dense + sparse hybrid retrieval over the knowledge corpus."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_KNOWLEDGE_ROOT = Path(__file__).parent.parent

# Module-level lazy singletons — loaded once, reused across calls
_chroma_client: Optional[Any] = None
_embed_model: Optional[Any] = None
_bm25_data: Optional[dict] = None


def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from knowledge.config import cfg
        store_path = str(cfg.chroma_store_dir)
        _chroma_client = chromadb.PersistentClient(path=store_path)
    return _chroma_client


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        from knowledge.config import cfg
        _embed_model = SentenceTransformer(cfg.embedding_model, device=cfg.embedding_device)
    return _embed_model


def _get_bm25_data():
    global _bm25_data
    if _bm25_data is None:
        from knowledge.config import cfg
        pkl_path = cfg.bm25_index_path
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                _bm25_data = pickle.load(f)
        else:
            _bm25_data = {}
    return _bm25_data


def _dense_retrieve(query: str, collections: List[str], top_k: int) -> List[Dict[str, Any]]:
    try:
        client = _get_chroma()
        model = _get_embed_model()
        query_embedding = model.encode([query])[0].tolist()

        results: List[Dict[str, Any]] = []
        for coll_name in collections:
            try:
                coll = client.get_collection(coll_name)
            except Exception:
                continue
            try:
                resp = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, coll.count()),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                logger.warning(f"Chroma query failed for {coll_name}: {e}")
                continue

            docs = resp.get("documents", [[]])[0]
            metas = resp.get("metadatas", [[]])[0]
            dists = resp.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                # Cosine distance → similarity score
                score = float(1.0 - dist)
                results.append({
                    "text": doc,
                    "meta": meta,
                    "source_id": meta.get("source_id", ""),
                    "dense_score": score,
                    "sparse_score": 0.0,
                    "collection": coll_name,
                })
        return results
    except Exception as e:
        logger.warning(f"Dense retrieval failed: {e}")
        return []


def _sparse_retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        data = _get_bm25_data()
        if not data:
            return []
        bm25 = data["bm25"]
        ids = data["ids"]
        corpus = data["corpus"]
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": corpus[idx],
                    "meta": {},
                    "source_id": ids[idx],
                    "dense_score": 0.0,
                    "sparse_score": float(scores[idx]),
                    "collection": "bm25",
                })
        return results
    except Exception as e:
        logger.warning(f"Sparse retrieval failed: {e}")
        return []


def retrieve(query: str, collections: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve candidate chunks using dense + sparse hybrid search.

    Merges results via reciprocal rank fusion (RRF) and returns up to top_k*2
    deduplicated candidates for the reranker to score.
    """
    from knowledge.config import cfg

    dense = _dense_retrieve(query, collections, top_k=cfg.dense_top_k)
    sparse = _sparse_retrieve(query, top_k=cfg.sparse_top_k)

    if not dense and not sparse:
        return []

    # Merge by source_id with RRF scoring
    # RRF score = 1/(rank + k) summed across retrieval methods
    rrf_k = 60
    merged: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(dense):
        sid = item["source_id"]
        if sid not in merged:
            merged[sid] = item.copy()
            merged[sid]["rrf_score"] = 0.0
        merged[sid]["rrf_score"] += 1.0 / (rank + 1 + rrf_k)
        merged[sid]["dense_score"] = item["dense_score"]

    for rank, item in enumerate(sparse):
        sid = item["source_id"]
        if sid not in merged:
            merged[sid] = item.copy()
            merged[sid]["rrf_score"] = 0.0
        merged[sid]["rrf_score"] += 1.0 / (rank + 1 + rrf_k)
        if merged[sid].get("sparse_score", 0) == 0:
            merged[sid]["sparse_score"] = item["sparse_score"]

    candidates = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
    return candidates[: top_k * 2]
