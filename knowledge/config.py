"""Knowledge Agent configuration — API keys, paths, model names.

All secrets should be set via environment variables, not hardcoded here.
Usage::
    from knowledge.config import cfg
    key = cfg.met_office_api_key
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


_KNOWLEDGE_ROOT = Path(__file__).parent


@dataclass
class KnowledgeConfig:
    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------
    corpus_dir: Path = field(default_factory=lambda: _KNOWLEDGE_ROOT / "corpus")
    index_dir: Path = field(default_factory=lambda: _KNOWLEDGE_ROOT / "index")
    chroma_store_dir: Path = field(default_factory=lambda: _KNOWLEDGE_ROOT / "index" / "chroma_store")
    bm25_index_path: Path = field(default_factory=lambda: _KNOWLEDGE_ROOT / "index" / "bm25_index.pkl")
    metadata_db_path: Path = field(default_factory=lambda: _KNOWLEDGE_ROOT / "index" / "metadata.sqlite")
    eval_set_path: Path = field(default_factory=lambda: _KNOWLEDGE_ROOT / "eval" / "eval_set.jsonl")

    # ---------------------------------------------------------------------------
    # Embedding model
    # ---------------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cpu"        # "cuda" if GPU available
    chunk_overlap_sentences: int = 1

    # ---------------------------------------------------------------------------
    # Reranker
    # ---------------------------------------------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 5

    # ---------------------------------------------------------------------------
    # Retrieval settings
    # ---------------------------------------------------------------------------
    dense_top_k: int = 10
    sparse_top_k: int = 10
    staleness_threshold_months: int = 12

    # ---------------------------------------------------------------------------
    # API keys (read from environment — never hardcode)
    # ---------------------------------------------------------------------------
    met_office_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("MET_OFFICE_API_KEY")
    )
    octopus_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OCTOPUS_API_KEY")
    )

    # ---------------------------------------------------------------------------
    # API endpoints
    # ---------------------------------------------------------------------------
    met_office_base_url: str = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point"
    open_meteo_base_url: str = "https://api.open-meteo.com/v1/ukmo"
    carbon_intensity_base_url: str = "https://api.carbonintensity.org.uk"
    octopus_base_url: str = "https://api.octopus.energy/v1"

    # ---------------------------------------------------------------------------
    # Corpus collection names (must match Chroma collection names)
    # ---------------------------------------------------------------------------
    collections: tuple = (
        "ideal_docs",
        "regulatory",
        "efficiency_guide",
        "domain_qa",
    )

    # ---------------------------------------------------------------------------
    # Chunk metadata schema version (bump when metadata fields change)
    # ---------------------------------------------------------------------------
    metadata_schema_version: str = "1.0"


# Singleton — import and use directly
cfg = KnowledgeConfig()
