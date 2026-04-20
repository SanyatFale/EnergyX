"""Ingest PAS 2035/2030:2023 (and other BSI standard PDFs) into the regulatory corpus.

PAS 2035/2030 is the UK retrofit standard referenced in ECO4 and BUS eligibility.
Updated 30 March 2025 to require Retrofit Coordinator site visits.

The PDF is not freely downloadable — it must be obtained via BSI or the
Publicly Available Specifications portal (sometimes freely available for
PAS documents).

Requires: pdfplumber (pip install pdfplumber)
Output: Markdown chunks in knowledge/corpus/regulatory/

Usage::
    python -m knowledge.ingestion.pas_pdf_ingest --pdf /path/to/pas2035_2023.pdf
    python -m knowledge.ingestion.pas_pdf_ingest --pdf /path/to/smets2.pdf \\
        --source-id govuk.smets2.2024 --jurisdiction uk
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_OUT_DIR = Path(__file__).parent.parent / "corpus" / "regulatory"

# Well-known PDF sources (populate --pdf from these)
_KNOWN_SOURCES = {
    "pas2035": {
        "source_id": "bsi.pas2035.2023",
        "jurisdiction": "uk",
        "scheme": "ECO4 BUS",
        "publication_date": "2023-03-30",
        "url": "https://www.bsigroup.com/en-GB/standards/pas-2035-2030/",
        "version": "2023",
    },
    "smets2": {
        "source_id": "govuk.smets2.2024",
        "jurisdiction": "uk",
        "scheme": None,
        "publication_date": "2024-01-01",
        "url": "https://www.gov.uk/guidance/smart-meters-technical-specifications-2",
        "version": "v2",
    },
}


def _heading_path(line: str, current: list[str]) -> list[str]:
    """Detect section heading and return new heading path, or unchanged."""
    # PAS headings often like "3.2 Definitions" or "Section 4:"
    m = re.match(r"^(\d+\.?\d*\.?\d*)\s+([A-Z][^a-z]{0,3}[^\n]{5,80})", line)
    if m:
        return [m.group(0)[:80]]
    return current


def ingest_pdf(
    pdf_path: Path,
    source_id: str,
    jurisdiction: str,
    scheme: str | None,
    publication_date: str,
    url: str,
    version: str,
    min_tokens: int = 80,
    max_tokens: int = 600,
) -> list[dict]:
    """Parse PDF into section-chunked Markdown dicts."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    chunks = []
    current_heading = ["Overview"]
    current_text: list[str] = []
    chunk_index = 0

    def flush():
        nonlocal chunk_index
        text = " ".join(current_text).strip()
        if len(text.split()) >= min_tokens:
            chunks.append({
                "source_id": f"{source_id}.chunk{chunk_index:04d}",
                "source_type": "regulatory",
                "scheme": scheme,
                "jurisdiction": jurisdiction,
                "publication_date": publication_date,
                "version": version,
                "url": url,
                "heading_path": list(current_heading),
                "chunk_index": chunk_index,
                "text": text,
            })
            chunk_index += 1

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                new_heading = _heading_path(stripped, current_heading)
                if new_heading != current_heading:
                    flush()
                    current_heading = new_heading
                    current_text = []
                else:
                    current_text.append(stripped)
                    if len(" ".join(current_text).split()) >= max_tokens:
                        flush()
                        current_text = []

    flush()
    return chunks


def write_chunks(chunks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        meta = {k: v for k, v in chunk.items() if k != "text"}
        content = f"<!--\n{json.dumps(meta, indent=2)}\n-->\n\n{chunk['text']}\n"
        (out_dir / f"{chunk['source_id']}.md").write_text(content, encoding="utf-8")
    logger.info(f"Wrote {len(chunks)} chunks to {out_dir}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest BSI standard PDFs into regulatory corpus")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to PDF file")
    parser.add_argument("--known", choices=list(_KNOWN_SOURCES.keys()),
                        help="Use metadata from a known source")
    parser.add_argument("--source-id", help="Override source_id")
    parser.add_argument("--jurisdiction", default="uk")
    parser.add_argument("--scheme", default=None)
    parser.add_argument("--publication-date", default=date.today().isoformat())
    parser.add_argument("--url", default="")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    args = parser.parse_args()

    meta = {}
    if args.known:
        meta = _KNOWN_SOURCES[args.known]
    meta.update({k: v for k, v in vars(args).items()
                 if v is not None and k in ("source_id", "jurisdiction", "scheme",
                                             "publication_date", "url", "version")})

    chunks = ingest_pdf(
        args.pdf,
        source_id=meta.get("source_id", f"bsi.custom.{date.today().isoformat()}"),
        jurisdiction=meta.get("jurisdiction", "uk"),
        scheme=meta.get("scheme"),
        publication_date=meta.get("publication_date", date.today().isoformat()),
        url=meta.get("url", ""),
        version=meta.get("version", "v1"),
    )
    write_chunks(chunks, args.out_dir)
    print(f"Done. {len(chunks)} chunks written to {args.out_dir}")


if __name__ == "__main__":
    main()
