"""Bootstrap the knowledge corpus and build the vector index.

Runs the seed ingestion for all corpus types (no external files required),
then optionally builds the Chroma + BM25 index if the dependencies are
available.

Usage::
    # Seed-only (no index, no external deps)
    python -m knowledge.ingestion.build_index --seed-only

    # Seed + build index (requires sentence-transformers + chromadb)
    python -m knowledge.ingestion.build_index

    # Seed + index + run eval
    python -m knowledge.ingestion.build_index --eval

After running --seed-only, the corpus/ directories are populated with
Markdown chunks.  Re-run without --seed-only to build the vector index
when dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_KNOWLEDGE_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Seed ingestion (no external files required)
# ---------------------------------------------------------------------------

def run_seed_ingestion(out_root: Path) -> dict[str, int]:
    """Write all built-in seed chunks to the corpus directories."""
    counts: dict[str, int] = {}

    # 1. IDEAL docs seed chunks
    from knowledge.ingestion.ideal_ingest import generate_seed_chunks, write_chunks
    ideal_chunks = list(generate_seed_chunks())
    write_chunks(ideal_chunks, out_root / "ideal_docs")
    counts["ideal_docs"] = len(ideal_chunks)
    logger.info(f"  ideal_docs: {len(ideal_chunks)} seed chunks")

    # 2. Domain Q&A
    from knowledge.ingestion.domain_qa_ingest import load_seeds, entry_to_chunk, write_chunks as write_qa
    seeds_path = _KNOWLEDGE_ROOT / "corpus" / "domain_qa" / "domain_qa_seeds.json"
    if seeds_path.exists():
        entries = load_seeds(seeds_path)
        qa_chunks = [entry_to_chunk(e, i) for i, e in enumerate(entries)]
        write_qa(qa_chunks, out_root / "domain_qa")
        counts["domain_qa"] = len(qa_chunks)
        logger.info(f"  domain_qa: {len(qa_chunks)} seed chunks")
    else:
        logger.warning(f"Domain Q&A seeds not found at {seeds_path}")
        counts["domain_qa"] = 0

    # 3. Regulatory corpus
    from knowledge.ingestion.regulatory_ingest import generate_seed_chunks as gen_reg, write_chunks as write_reg
    reg_chunks = list(gen_reg())
    write_reg(reg_chunks, out_root / "regulatory")
    counts["regulatory"] = len(reg_chunks)
    logger.info(f"  regulatory: {len(reg_chunks)} seed chunks")

    # 4. Efficiency guides
    from knowledge.ingestion.efficiency_ingest import generate_seed_chunks as gen_eff, write_chunks as write_eff
    eff_chunks = list(gen_eff())
    write_eff(eff_chunks, out_root / "efficiency_guide")
    counts["efficiency_guide"] = len(eff_chunks)
    logger.info(f"  efficiency_guide: {len(eff_chunks)} seed chunks")

    return counts


# ---------------------------------------------------------------------------
# Index building (requires sentence-transformers + chromadb)
# ---------------------------------------------------------------------------

def build_chroma_index(corpus_root: Path, index_dir: Path) -> bool:
    """Embed all corpus chunks and write to Chroma vector store."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("chromadb or sentence-transformers not installed. Skipping index build.")
        logger.warning("Install with: pip install chromadb sentence-transformers")
        return False

    from knowledge.config import cfg

    logger.info("Building Chroma index...")
    model = SentenceTransformer(cfg.embedding_model, device=cfg.embedding_device)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma_store"))

    total = 0
    for collection_name in cfg.collections:
        source_dir = corpus_root / collection_name
        if not source_dir.exists():
            logger.warning(f"Corpus dir not found: {source_dir}")
            continue

        md_files = list(source_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"No .md files in {source_dir}")
            continue

        coll = client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids, docs, metadatas = [], [], []
        for md_path in md_files:
            content = md_path.read_text(encoding="utf-8")
            # Parse metadata comment block
            meta = {}
            if content.startswith("<!--"):
                end = content.find("-->")
                if end > 0:
                    try:
                        meta = json.loads(content[4:end].strip())
                        content = content[end + 3:].strip()
                    except json.JSONDecodeError:
                        pass

            ids.append(meta.get("source_id", md_path.stem))
            docs.append(content)
            metadatas.append({k: str(v) for k, v in meta.items() if v is not None})

        if not docs:
            continue

        embeddings = model.encode(docs, show_progress_bar=True).tolist()
        coll.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
        total += len(docs)
        logger.info(f"  {collection_name}: {len(docs)} chunks indexed")

    logger.info(f"Chroma index built. {total} total chunks.")
    return True


def build_bm25_index(corpus_root: Path, index_dir: Path) -> bool:
    """Build BM25 sparse index over all corpus chunks."""
    try:
        from rank_bm25 import BM25Okapi
        import pickle
    except ImportError:
        logger.warning("rank_bm25 not installed. Skipping BM25 index.")
        logger.warning("Install with: pip install rank-bm25")
        return False

    from knowledge.config import cfg

    logger.info("Building BM25 index...")
    corpus: list[str] = []
    ids: list[str] = []

    for collection_name in cfg.collections:
        source_dir = corpus_root / collection_name
        for md_path in source_dir.glob("*.md"):
            content = md_path.read_text(encoding="utf-8")
            if content.startswith("<!--"):
                end = content.find("-->")
                if end > 0:
                    try:
                        meta = json.loads(content[4:end].strip())
                        content = content[end + 3:].strip()
                        ids.append(meta.get("source_id", md_path.stem))
                    except json.JSONDecodeError:
                        ids.append(md_path.stem)
                else:
                    ids.append(md_path.stem)
            else:
                ids.append(md_path.stem)
            corpus.append(content)

    if not corpus:
        logger.warning("No chunks found for BM25 index.")
        return False

    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    import pickle
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = index_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "corpus": corpus}, f)
    logger.info(f"BM25 index written to {bm25_path} ({len(corpus)} docs)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap knowledge corpus and build index")
    parser.add_argument("--seed-only", action="store_true",
                        help="Write seed chunks only; skip vector index building")
    parser.add_argument("--eval", action="store_true",
                        help="Run the evaluation harness after indexing")
    parser.add_argument("--corpus-dir", type=Path,
                        default=_KNOWLEDGE_ROOT / "corpus")
    parser.add_argument("--index-dir", type=Path,
                        default=_KNOWLEDGE_ROOT / "index")
    args = parser.parse_args()

    print("=== EnergyX Knowledge Index Builder ===\n")

    # Step 1: Seed ingestion
    print("Step 1/3: Writing seed corpus chunks...")
    counts = run_seed_ingestion(args.corpus_dir)
    for name, n in counts.items():
        print(f"  {name}: {n} chunks")
    print()

    if args.seed_only:
        print("Seed-only mode — skipping index build.")
        print("Run without --seed-only when sentence-transformers + chromadb are installed.")
        return

    # Step 2: Vector index
    print("Step 2/3: Building vector index (Chroma + BM25)...")
    chroma_ok = build_chroma_index(args.corpus_dir, args.index_dir)
    bm25_ok = build_bm25_index(args.corpus_dir, args.index_dir)
    print(f"  Chroma: {'OK' if chroma_ok else 'SKIPPED'}")
    print(f"  BM25:   {'OK' if bm25_ok else 'SKIPPED'}")
    print()

    # Step 3: Eval
    if args.eval:
        print("Step 3/3: Running evaluation harness...")
        from knowledge.eval.run_eval import run_eval
        from knowledge.config import cfg
        report = run_eval(cfg.eval_set_path)
        agg = report.get("aggregate", {})
        print(f"  n_questions:           {agg.get('n_questions')}")
        print(f"  answer_relevance_mean: {agg.get('answer_relevance_mean')}")
        print(f"  source_cited_rate:     {agg.get('source_cited_rate')}")
        print(f"  pass_rate:             {agg.get('pass_rate')}")
        if agg.get("deploy_block"):
            print("\n  DEPLOY BLOCKED — metric below threshold")
            sys.exit(1)
        else:
            print("\n  Eval passed.")
    else:
        print("Step 3/3: Skipped (pass --eval to run evaluation).")

    print("\nDone.")


if __name__ == "__main__":
    main()
