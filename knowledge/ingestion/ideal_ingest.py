"""Ingest IDEAL dataset documentation into the knowledge corpus.

Sources:
  - Pullinger et al. 2021, Scientific Data (PDF)
    DOI: 10.1038/s41597-021-00921-y
  - Edinburgh DataShare landing page
    DOI: 10.7488/ds/2836
  - IDEAL machine-accessible metadata (figshare)
    DOI: 10.6084/m9.figshare.14096993

Output: Markdown chunks written to knowledge/corpus/ideal_docs/
Each chunk has a frontmatter header with required metadata fields.

Usage::
    python -m knowledge.ingestion.ideal_ingest --pdf path/to/pullinger2021.pdf
    python -m knowledge.ingestion.ideal_ingest --metadata path/to/metadata.csv

Action required (from DATA_CARD_COMPLIANCE.md A1–A7):
    After running this script against the real files, verify the sensor slug
    mapping and filename patterns documented in DATA_CARD_COMPLIANCE.md.
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

_OUT_DIR = Path(__file__).parent.parent / "corpus" / "ideal_docs"
_SOURCE_ID_PREFIX = "ideal.pullinger2021"


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str,
    heading_path: list,
    chunk_index: int,
    url: str = "https://doi.org/10.7488/ds/2836",
    publication_date: str = "2021-01-01",
) -> dict:
    return {
        "source_id": f"{_SOURCE_ID_PREFIX}.chunk{chunk_index:04d}",
        "source_type": "ideal_docs",
        "scheme": None,
        "jurisdiction": "uk",
        "publication_date": publication_date,
        "version": "v1.0",
        "url": url,
        "heading_path": heading_path,
        "chunk_index": chunk_index,
        "text": text.strip(),
    }


def _chunk_to_markdown(chunk: dict) -> str:
    meta = {k: v for k, v in chunk.items() if k != "text"}
    return f"<!--\n{json.dumps(meta, indent=2)}\n-->\n\n{chunk['text']}\n"


# ---------------------------------------------------------------------------
# PDF ingestion (requires pdfplumber)
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path) -> Iterator[dict]:
    """Chunk a paper PDF by section heading.  Requires pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    heading_re = re.compile(r"^(\d+\.?\d*)\s+[A-Z]")
    current_heading = ["Introduction"]
    current_text: list[str] = []
    chunk_index = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                stripped = line.strip()
                if heading_re.match(stripped) and len(stripped) < 120:
                    if current_text:
                        yield _make_chunk(
                            " ".join(current_text),
                            heading_path=list(current_heading),
                            chunk_index=chunk_index,
                        )
                        chunk_index += 1
                        current_text = []
                    current_heading = [stripped]
                else:
                    if stripped:
                        current_text.append(stripped)

    if current_text:
        yield _make_chunk(
            " ".join(current_text),
            heading_path=list(current_heading),
            chunk_index=chunk_index,
        )


# ---------------------------------------------------------------------------
# Metadata CSV ingestion (one chunk per sensor type)
# ---------------------------------------------------------------------------

def ingest_metadata_csv(csv_path: Path) -> Iterator[dict]:
    """Convert IDEAL machine-accessible metadata to one chunk per sensor/room type."""
    import csv
    chunk_index = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_parts = [f"Sensor/entity metadata from the IDEAL dataset."]
            for k, v in row.items():
                if v:
                    text_parts.append(f"{k}: {v}")
            yield _make_chunk(
                " ".join(text_parts),
                heading_path=["Metadata", row.get("sensor_type", "unknown")],
                chunk_index=chunk_index,
                url="https://doi.org/10.6084/m9.figshare.14096993",
                publication_date="2021-01-01",
            )
            chunk_index += 1


# ---------------------------------------------------------------------------
# Synthetic seed chunks (embedded, no external file required)
# ---------------------------------------------------------------------------

_SEED_CHUNKS = [
    {
        "heading": ["Dataset Overview"],
        "text": (
            "The IDEAL household energy dataset (Pullinger et al. 2021) covers 255 UK homes "
            "monitored between 2016 and 2018. 39 homes received enhanced monitoring with "
            "appliance-level plug sensors. All timestamps are UTC in the format "
            "YYYY-MM-DD HH:MM:SS. No header row appears in sensor CSV files."
        ),
    },
    {
        "heading": ["Sensor Units", "Temperature"],
        "text": (
            "Temperature readings in the IDEAL dataset are stored as integers in tenths of "
            "a degree Celsius. For example, a raw value of 205 represents 20.5°C. "
            "Room sensors sample at 12-second cadence. Boiler pipe temperatures are also "
            "recorded but their exact cadence should be verified against the readme.txt."
        ),
    },
    {
        "heading": ["Sensor Units", "Electricity"],
        "text": (
            "Whole-home electricity is measured as apparent power in Watts at 1 Hz "
            "(one reading per second). Files are gzip-compressed CSVs with two columns: "
            "timestamp and raw value. The mains electricity sensor records real power for "
            "enhanced homes."
        ),
    },
    {
        "heading": ["Sensor Units", "Gas"],
        "text": (
            "Gas is recorded as pulse counts. One pulse represents a fixed volume of gas "
            "consumed; the conversion coefficient (m³ per pulse) is documented in the "
            "dataset readme.txt. Raw pulse counts are ingested; conversion to kWh requires "
            "the calorific value coefficient."
        ),
    },
    {
        "heading": ["Ethics and Consent"],
        "text": (
            "IDEAL data was collected under informed consent from participating households. "
            "All home IDs are anonymised. Location data is limited to broad UK region. "
            "Occupancy and survey data must be handled according to the dataset's data "
            "sharing agreement (University of Edinburgh DataShare)."
        ),
    },
]


def generate_seed_chunks() -> Iterator[dict]:
    for i, entry in enumerate(_SEED_CHUNKS):
        yield _make_chunk(
            entry["text"],
            heading_path=entry["heading"],
            chunk_index=i,
        )


# ---------------------------------------------------------------------------
# Write chunks to disk
# ---------------------------------------------------------------------------

def write_chunks(chunks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        filename = f"{chunk['source_id']}.md"
        (out_dir / filename).write_text(_chunk_to_markdown(chunk), encoding="utf-8")
    logger.info(f"Wrote {len(chunks)} chunks to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest IDEAL docs into knowledge corpus")
    parser.add_argument("--pdf", type=Path, help="Path to Pullinger 2021 PDF")
    parser.add_argument("--metadata", type=Path, help="Path to IDEAL metadata CSV")
    parser.add_argument("--seed", action="store_true", default=True,
                        help="Write built-in seed chunks (default: True)")
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    args = parser.parse_args()

    chunks = []
    if args.seed:
        chunks.extend(generate_seed_chunks())
        logger.info(f"Generated {len(chunks)} seed chunks")

    if args.pdf and args.pdf.exists():
        pdf_chunks = list(ingest_pdf(args.pdf))
        chunks.extend(pdf_chunks)
        logger.info(f"Ingested {len(pdf_chunks)} chunks from PDF")

    if args.metadata and args.metadata.exists():
        meta_chunks = list(ingest_metadata_csv(args.metadata))
        chunks.extend(meta_chunks)
        logger.info(f"Ingested {len(meta_chunks)} chunks from metadata CSV")

    write_chunks(chunks, args.out_dir)
    print(f"Done. {len(chunks)} chunks written to {args.out_dir}")


if __name__ == "__main__":
    main()
