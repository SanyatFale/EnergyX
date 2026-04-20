"""Scrape Ofgem guidance pages into the regulatory corpus.

Pages scraped:
  - ECO4 guidance (https://www.ofgem.gov.uk/energy-company-obligation-eco)
  - BUS guidance (https://www.ofgem.gov.uk/environmental-and-social-schemes/boiler-upgrade-scheme-bus)
  - SEG guidance (https://www.ofgem.gov.uk/check-if-energy-company-has-to-offer-you-export-tariff)
  - Price cap methodology pages

Requires: trafilatura (pip install trafilatura)
Output: Markdown chunks in knowledge/corpus/regulatory/

Usage::
    python -m knowledge.ingestion.ofgem_scraper
    python -m knowledge.ingestion.ofgem_scraper --scheme eco4 --scheme bus

Refresh cadence: quarterly (see RAG_PLAN.md §4).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_OUT_DIR = Path(__file__).parent.parent / "corpus" / "regulatory"

# Ofgem scheme pages to scrape (scheme_key → url)
_SCHEME_URLS = {
    "eco4": "https://www.ofgem.gov.uk/energy-company-obligation-eco",
    "bus": "https://www.ofgem.gov.uk/environmental-and-social-schemes/boiler-upgrade-scheme-bus",
    "seg": "https://www.ofgem.gov.uk/check-if-energy-company-has-to-offer-you-export-tariff",
    "price_cap": "https://www.ofgem.gov.uk/check-if-energy-company-has-to-offer-you-export-tariff",
}

# Scheme → jurisdiction (BUS England/Wales only)
_SCHEME_JURISDICTION = {
    "eco4": "uk",
    "bus": "england_wales",
    "seg": "uk",
    "price_cap": "uk",
}


def _clean_markdown(text: str) -> str:
    """Remove excessive blank lines from extracted text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_by_section(text: str, min_tokens: int = 100, max_tokens: int = 600) -> list[str]:
    """Split markdown text into chunks at heading boundaries."""
    sections = re.split(r"\n(?=#+\s)", text)
    chunks = []
    current = ""
    for section in sections:
        words = len(section.split())
        if len(current.split()) + words > max_tokens and current:
            chunks.append(current.strip())
            current = section
        else:
            current = current + "\n\n" + section if current else section
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c.split()) >= min_tokens]


def scrape_url(url: str) -> str | None:
    """Fetch and extract clean Markdown from a URL using trafilatura."""
    try:
        import trafilatura
    except ImportError:
        raise ImportError("Install trafilatura: pip install trafilatura")

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            logger.warning(f"Empty response from {url}")
            return None
        text = trafilatura.extract(
            downloaded,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
        )
        return text
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return None


def ingest_scheme(scheme: str, url: str) -> Iterator[dict]:
    """Scrape one scheme page and yield chunk dicts."""
    logger.info(f"Scraping {scheme} from {url}")
    text = scrape_url(url)
    if not text:
        logger.warning(f"No content extracted for {scheme}")
        return

    text = _clean_markdown(text)
    sections = _chunk_by_section(text)
    jurisdiction = _SCHEME_JURISDICTION.get(scheme, "uk")
    today = date.today().isoformat()
    source_id_base = f"ofgem.{scheme}.{today}"

    for i, section in enumerate(sections):
        heading_match = re.match(r"^#+\s+(.+)", section.split("\n")[0])
        heading = heading_match.group(1) if heading_match else "Content"
        yield {
            "source_id": f"{source_id_base}.chunk{i:04d}",
            "source_type": "regulatory",
            "scheme": scheme.upper().replace("_", " "),
            "jurisdiction": jurisdiction,
            "publication_date": today,
            "version": today,
            "url": url,
            "heading_path": [scheme.upper(), heading],
            "chunk_index": i,
            "text": section,
        }


def write_chunks(chunks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        meta = {k: v for k, v in chunk.items() if k != "text"}
        content = f"<!--\n{json.dumps(meta, indent=2)}\n-->\n\n{chunk['text']}\n"
        (out_dir / f"{chunk['source_id']}.md").write_text(content, encoding="utf-8")
    logger.info(f"Wrote {len(chunks)} chunks to {out_dir}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Scrape Ofgem guidance into regulatory corpus")
    parser.add_argument("--scheme", action="append", choices=list(_SCHEME_URLS.keys()),
                        help="Scheme(s) to scrape (default: all)")
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to wait between requests (be polite)")
    args = parser.parse_args()

    schemes = args.scheme or list(_SCHEME_URLS.keys())
    all_chunks = []
    for scheme in schemes:
        url = _SCHEME_URLS[scheme]
        all_chunks.extend(ingest_scheme(scheme, url))
        time.sleep(args.delay)

    write_chunks(all_chunks, args.out_dir)
    print(f"Done. {len(all_chunks)} chunks written to {args.out_dir}")


if __name__ == "__main__":
    main()
