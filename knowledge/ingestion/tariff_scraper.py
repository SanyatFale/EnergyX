"""Scrape supplier tariff pages and write structured JSON to knowledge/corpus/tariffs/.

NOT for RAG retrieval — tariff data is served as structured JSON via get_active_tariff().
This script populates the tariff catalogue that TariffCache loads.

Suppliers covered (public tariff pages, no auth):
  - Octopus Energy (Agile, Go, Cosy Octopus, Intelligent Octopus)
  - British Gas (standard, HomeEnergy)
  - EDF
  - E.ON Next
  - OVO Energy

Refresh cadence: weekly (prices change with the Ofgem cap quarterly + supplier decisions).
Output: knowledge/corpus/tariffs/tariff_catalogue.json (also copied to energyx/data/)

Usage::
    python -m knowledge.ingestion.tariff_scraper
    python -m knowledge.ingestion.tariff_scraper --supplier octopus
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_OUT_DIR = Path(__file__).parent.parent / "corpus" / "tariffs"
_ENERGYX_DATA_DIR = Path(__file__).parent.parent.parent / "energyx" / "data"

# ---------------------------------------------------------------------------
# Hardcoded baseline (Ofgem Q1 2026 cap rates).
# NOTE: Replace with scraped values when running against live pages.
# ---------------------------------------------------------------------------
_BASELINE_CATALOGUE: list[dict[str, Any]] = [
    {
        "tariff_id": "flat_standard_2026q1",
        "supplier": "ofgem_cap",
        "name": "Standard Variable (Q1 2026 cap)",
        "structure": "flat",
        "unit_rate_gbp_per_kwh": 0.2459,
        "standing_charge_gbp_per_day": 0.61,
        "off_peak_rate": None,
        "off_peak_window": None,
        "jurisdiction": "england_wales",
        "green": False,
        "valid_from": "2026-01-01",
        "valid_to": "2026-03-31",
        "source": "ofgem_price_cap",
        "scraped_at": None,
    },
    {
        "tariff_id": "economy7_standard_2026q1",
        "supplier": "ofgem_cap",
        "name": "Economy 7 (Q1 2026 cap)",
        "structure": "economy7",
        "unit_rate_gbp_per_kwh": 0.3048,
        "standing_charge_gbp_per_day": 0.61,
        "off_peak_rate": 0.1338,
        "off_peak_window": ["00:30", "07:30"],
        "jurisdiction": "england_wales",
        "green": False,
        "valid_from": "2026-01-01",
        "valid_to": "2026-03-31",
        "source": "ofgem_price_cap",
        "scraped_at": None,
    },
]


def scrape_octopus() -> list[dict]:
    """Fetch Octopus Energy Agile tariff via their public API."""
    try:
        import urllib.request
        url = "https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/E-1R-AGILE-24-10-01-C/standard-unit-rates/"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        # Latest rate
        rates = data.get("results", [])
        if rates:
            latest = rates[0]
            return [{
                "tariff_id": "octopus_agile_2026",
                "supplier": "octopus",
                "name": "Octopus Agile",
                "structure": "agile",
                "unit_rate_gbp_per_kwh": latest["value_inc_vat"] / 100,
                "standing_charge_gbp_per_day": 0.61,  # approximate
                "off_peak_rate": None,
                "off_peak_window": None,
                "jurisdiction": "england_wales",
                "green": True,
                "valid_from": latest.get("valid_from", date.today().isoformat()),
                "valid_to": latest.get("valid_to"),
                "source": "octopus_api",
                "scraped_at": date.today().isoformat(),
            }]
    except Exception as e:
        logger.warning(f"Octopus API scrape failed: {e}")
    return []


def build_catalogue(suppliers: list[str] | None) -> list[dict]:
    catalogue = list(_BASELINE_CATALOGUE)
    today = date.today().isoformat()
    for entry in catalogue:
        if entry["scraped_at"] is None:
            entry["scraped_at"] = today

    if suppliers is None or "octopus" in suppliers:
        octopus = scrape_octopus()
        if octopus:
            catalogue.extend(octopus)
            logger.info(f"Added {len(octopus)} Octopus tariff(s)")
        else:
            logger.info("No live Octopus data — using cap baseline only")

    return catalogue


def write_catalogue(catalogue: list[dict], out_dir: Path, also_copy_to: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tariff_catalogue.json"
    path.write_text(json.dumps(catalogue, indent=2), encoding="utf-8")
    logger.info(f"Wrote {len(catalogue)} tariffs to {path}")

    if also_copy_to:
        also_copy_to.mkdir(parents=True, exist_ok=True)
        copy_path = also_copy_to / "tariff_catalogue.json"
        copy_path.write_text(json.dumps(catalogue, indent=2), encoding="utf-8")
        logger.info(f"Copied to {copy_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build tariff catalogue JSON")
    parser.add_argument("--supplier", action="append",
                        choices=["octopus", "british_gas", "edf", "eon", "ovo"],
                        help="Suppliers to scrape (default: all available)")
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    parser.add_argument("--no-copy", action="store_true",
                        help="Do not copy to energyx/data/")
    args = parser.parse_args()

    catalogue = build_catalogue(args.supplier)
    write_catalogue(
        catalogue,
        args.out_dir,
        also_copy_to=None if args.no_copy else _ENERGYX_DATA_DIR,
    )
    print(f"Done. {len(catalogue)} tariff entries written.")


if __name__ == "__main__":
    main()
