"""Ingest the built-in domain Q&A seeds into the knowledge corpus.

Source: knowledge/corpus/domain_qa/domain_qa_seeds.json
Each entry becomes one Markdown chunk (question + answer together).
1 chunk per entry, no overlap (Q&A entries are self-contained).

Output: knowledge/corpus/domain_qa/  (Markdown .md files)

Usage::
    python -m knowledge.ingestion.domain_qa_ingest
    python -m knowledge.ingestion.domain_qa_ingest --seeds path/to/custom_qa.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SEEDS_PATH = Path(__file__).parent.parent / "corpus" / "domain_qa" / "domain_qa_seeds.json"
_OUT_DIR = Path(__file__).parent.parent / "corpus" / "domain_qa"


def load_seeds(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def entry_to_chunk(entry: dict, index: int) -> dict:
    """Convert a Q&A entry to a corpus chunk dict."""
    text = f"Q: {entry['question']}\n\nA: {entry['answer']}"
    return {
        "source_id": f"domain_qa.{entry.get('id', f'entry{index:04d}')}",
        "source_type": "domain_qa",
        "scheme": None,
        "jurisdiction": "uk",
        "publication_date": "2026-01-01",
        "version": "v1.0",
        "url": "",
        "heading_path": ["Domain Q&A"] + entry.get("tags", [])[:2],
        "chunk_index": index,
        "text": text,
    }


def write_chunks(chunks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale domain_qa chunk files (not the JSON seeds)
    for f in out_dir.glob("domain_qa.*.md"):
        f.unlink()
    for chunk in chunks:
        meta = {k: v for k, v in chunk.items() if k != "text"}
        content = f"<!--\n{json.dumps(meta, indent=2)}\n-->\n\n{chunk['text']}\n"
        (out_dir / f"{chunk['source_id']}.md").write_text(content, encoding="utf-8")
    logger.info(f"Wrote {len(chunks)} domain Q&A chunks to {out_dir}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest domain Q&A seeds into corpus")
    parser.add_argument("--seeds", type=Path, default=_SEEDS_PATH)
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    args = parser.parse_args()

    entries = load_seeds(args.seeds)
    chunks = [entry_to_chunk(e, i) for i, e in enumerate(entries)]
    write_chunks(chunks, args.out_dir)
    print(f"Done. {len(chunks)} domain Q&A chunks written to {args.out_dir}")


if __name__ == "__main__":
    main()
