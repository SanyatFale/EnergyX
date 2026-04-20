"""Scrape Energy Saving Trust guides into the efficiency corpus.

Pages scraped:
  - Home insulation guide
  - Heat pumps guide
  - Solar panels guide
  - EV charging guide
  - Smart meters guide
  - LED lighting guide

Requires: trafilatura (pip install trafilatura)
Output: Markdown chunks in knowledge/corpus/efficiency/

Usage::
    python -m knowledge.ingestion.est_scraper

Refresh cadence: semi-annually.
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

_OUT_DIR = Path(__file__).parent.parent / "corpus" / "efficiency"

_PAGES = {
    "insulation": "https://energysavingtrust.org.uk/home-insulation/",
    "heat_pumps": "https://energysavingtrust.org.uk/advice/heat-pumps/",
    "solar_panels": "https://energysavingtrust.org.uk/advice/solar-panels/",
    "ev_charging": "https://energysavingtrust.org.uk/advice/charging-electric-vehicles/",
    "smart_meters": "https://energysavingtrust.org.uk/advice/smart-meters/",
    "led_lighting": "https://energysavingtrust.org.uk/advice/lighting/",
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


def _chunk_by_paragraph(text: str, max_tokens: int = 450) -> list[str]:
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


def ingest_page(key: str, url: str) -> Iterator[dict]:
    logger.info(f"Scraping EST {key}: {url}")
    text = _extract_text(url)
    if not text:
        return
    today = date.today().isoformat()
    for i, chunk in enumerate(_chunk_by_paragraph(text)):
        yield {
            "source_id": f"est.{key}.{today}.chunk{i:04d}",
            "source_type": "efficiency_guide",
            "scheme": None,
            "jurisdiction": "uk",
            "publication_date": today,
            "version": today,
            "url": url,
            "heading_path": ["Energy Saving Trust", key.replace("_", " ").title()],
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
    parser = argparse.ArgumentParser(description="Scrape Energy Saving Trust guides")
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
