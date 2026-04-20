"""Scrape GOV.UK energy pages into the regulatory corpus.

Pages scraped:
  - Warm Homes Local Grant
  - Warm Home Discount
  - Smart Export Guarantee (GOV.UK overview)
  - EPC guidance

Requires: trafilatura (pip install trafilatura)
Output: Markdown chunks in knowledge/corpus/regulatory/

Usage::
    python -m knowledge.ingestion.govuk_scraper

Refresh cadence: quarterly.
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

logger = logging.getLogger(__name__)

_OUT_DIR = Path(__file__).parent.parent / "corpus" / "regulatory"

_PAGES = {
    "warm_homes_local_grant": {
        "url": "https://www.gov.uk/guidance/warm-homes-local-grant",
        "scheme": "Warm Homes Local Grant",
        "jurisdiction": "england",
    },
    "warm_home_discount": {
        "url": "https://www.gov.uk/the-warm-home-discount-scheme",
        "scheme": "Warm Home Discount",
        "jurisdiction": "uk",
    },
    "epc_guidance": {
        "url": "https://www.gov.uk/buy-sell-your-home/energy-performance-certificates",
        "scheme": None,
        "jurisdiction": "uk",
    },
    "heat_pump_grant": {
        "url": "https://www.gov.uk/apply-boiler-upgrade-scheme",
        "scheme": "Boiler Upgrade Scheme",
        "jurisdiction": "england_wales",
    },
}


def _extract_text(url: str) -> str | None:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(downloaded, output_format="markdown", include_tables=True)
    except ImportError:
        raise ImportError("Install trafilatura: pip install trafilatura")
    except Exception as e:
        logger.error(f"Failed to extract {url}: {e}")
        return None


def _chunk_by_paragraph(text: str, max_tokens: int = 500) -> list[str]:
    paragraphs = re.split(r"\n\n+", text)
    chunks, current = [], ""
    for para in paragraphs:
        if len((current + " " + para).split()) > max_tokens and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c.split()) >= 30]


def ingest_page(key: str, meta: dict) -> Iterator[dict]:
    url, scheme, jurisdiction = meta["url"], meta["scheme"], meta["jurisdiction"]
    logger.info(f"Scraping {key}: {url}")
    text = _extract_text(url)
    if not text:
        return
    today = date.today().isoformat()
    for i, chunk in enumerate(_chunk_by_paragraph(text)):
        yield {
            "source_id": f"govuk.{key}.{today}.chunk{i:04d}",
            "source_type": "regulatory",
            "scheme": scheme,
            "jurisdiction": jurisdiction,
            "publication_date": today,
            "version": today,
            "url": url,
            "heading_path": [key],
            "chunk_index": i,
            "text": chunk,
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
    parser = argparse.ArgumentParser(description="Scrape GOV.UK energy pages")
    parser.add_argument("--page", action="append", choices=list(_PAGES.keys()))
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    pages = args.page or list(_PAGES.keys())
    all_chunks = []
    for key in pages:
        all_chunks.extend(ingest_page(key, _PAGES[key]))
        time.sleep(args.delay)

    write_chunks(all_chunks, args.out_dir)
    print(f"Done. {len(all_chunks)} chunks written to {args.out_dir}")


if __name__ == "__main__":
    main()
