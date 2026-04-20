#!/usr/bin/env python
"""Ingest IDEAL dataset sensor files into the EnergyX HistoricStore.

Reads raw .csv.gz sensor files from the IDEAL data directory tree and writes
them into the Parquet-backed HistoricStore in long format:
  (home_id, ts, sensor_type, sensor_id, value, unit)

No pre-stitching or wide-format pivoting — the HistoricStore long schema is
exactly what the AnalysisAgent and MonitorAgent consume.

Default: ingests the 5 selected diverse homes (62, 96, 128, 168, 169).
Override with --home-ids to ingest any subset of the 39 enhanced homes.

Usage::
    # Ingest the 5 default homes
    python scripts/ingest_ideal.py

    # Ingest specific homes
    python scripts/ingest_ideal.py --home-ids 62 96

    # Dry run — discover files and report counts without writing
    python scripts/ingest_ideal.py --dry-run

    # Custom paths
    python scripts/ingest_ideal.py \\
        --data-root data/IDEAL \\
        --store-path data/historic_store

Data roots searched (relative to --data-root):
  household_sensors/sensordata/        mains electricity, gas, tempprobes
  room_and_appliance_sensors/sensordata/  room sensors, appliance power
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from energyx.data.ideal_loader import IDEALLoader, SensorType
from energyx.data.historic_store import HistoricStore, ParquetBackend

logger = logging.getLogger(__name__)

# The 5 selected diverse homes (confirmed 2026-04-20)
_DEFAULT_HOME_IDS = [62, 96, 128, 168, 169]

# Both data sub-trees to scan
_DATA_SUBTREES = [
    "household_sensors/sensordata",
    "room_and_appliance_sensors/sensordata",
    "auxiliarydata/sensordata",
]


def _build_multi_root_loader(data_root: Path) -> IDEALLoader:
    """Create a loader that discovers files across all IDEAL sub-trees."""

    class _MultiRootLoader(IDEALLoader):
        def __init__(self, roots: list[Path]):
            self._roots = roots

        def discover_files(self):
            from energyx.data.ideal_loader import parse_filename
            files = []
            seen: set[Path] = set()
            for root in self._roots:
                if not root.exists():
                    logger.debug(f"Skipping missing subtree: {root}")
                    continue
                # Use *.gz only — it matches both .csv.gz and bare .gz
                # Do NOT use *.csv.gz + *.gz as that double-counts .csv.gz files
                for path in sorted(root.glob("*.gz")):
                    if path in seen:
                        continue
                    seen.add(path)
                    meta = parse_filename(path)
                    if meta:
                        files.append(meta)
            logger.info(f"Discovered {len(files)} IDEAL sensor files across {len(self._roots)} subtrees")
            return files

    roots = [data_root / sub for sub in _DATA_SUBTREES]
    return _MultiRootLoader(roots)


def ingest(
    data_root: Path,
    store_path: Path,
    home_ids: list[int],
    dry_run: bool = False,
    downsample_electricity: bool = True,
) -> dict[str, dict[str, int]]:
    """Stream all sensor files for the given homes into the HistoricStore.

    Returns: {home_id: {sensor_type: row_count}}
    """
    loader = _build_multi_root_loader(data_root)
    all_files = loader.discover_files()

    home_id_strs = {f"home{hid}" for hid in home_ids}
    relevant = [f for f in all_files if f.home_id in home_id_strs]

    if not relevant:
        logger.error(f"No files found for homes {home_ids} under {data_root}")
        return {}

    # Report file discovery
    by_home: dict[str, list] = defaultdict(list)
    for f in relevant:
        by_home[f.home_id].append(f)

    print(f"\nDiscovered {len(relevant)} sensor files for {len(by_home)} homes:\n", flush=True)
    for hid in sorted(by_home):
        type_counts: dict[str, int] = defaultdict(int)
        for f in by_home[hid]:
            type_counts[f.sensor_type.value] += 1
        breakdown = ", ".join(f"{t}×{n}" for t, n in sorted(type_counts.items()))
        print(f"  {hid:>10}: {len(by_home[hid]):3d} files  [{breakdown}]", flush=True)

    if dry_run:
        print("\nDry run — no data written.", flush=True)
        return {}

    import pandas as pd
    from energyx.data.ideal_loader import stream_sensor_file, SensorType as ST

    results: dict[str, dict[str, int]] = {}
    total_rows = 0

    for hid_str in sorted(by_home):
        print(f"\n  Ingesting {hid_str}...", flush=True)
        type_row_counts: dict[str, int] = defaultdict(int)
        t0 = time.time()

        # Write one parquet per sensor per date — no read-modify-write, O(1) RAM per file.
        # Layout: ticks/home_id={home}/date={date}/{sensor_type}_{sensor_id}.parquet
        # read_ticks globs all *.parquet in the date dir.
        ticks_home_dir = store_path / "ticks" / f"home_id={hid_str}"

        for file_meta in by_home[hid_str]:
            # Read entire file into memory (one sensor file at a time)
            chunks = []
            for chunk in stream_sensor_file(file_meta):
                if downsample_electricity and file_meta.sensor_type in {
                    ST.ELECTRICITY_APPARENT, ST.ELECTRICITY_REAL
                }:
                    chunk = IDEALLoader._downsample_electricity(chunk)
                chunks.append(chunk)

            if not chunks:
                continue

            sensor_df = pd.concat(chunks, ignore_index=True)
            sensor_df["_date"] = pd.to_datetime(sensor_df["ts"]).dt.date.astype(str)
            type_row_counts[file_meta.sensor_type.value] += len(sensor_df)

            # Write one parquet per date for this sensor
            safe_id = str(file_meta.sensor_id).replace("/", "_")
            fname = f"{file_meta.sensor_type.value}_{safe_id}.parquet"
            for date_str, grp in sensor_df.groupby("_date"):
                part_dir = ticks_home_dir / f"date={date_str}"
                part_dir.mkdir(parents=True, exist_ok=True)
                grp.drop(columns=["_date"]).to_parquet(part_dir / fname, index=False)

            del sensor_df, chunks  # free RAM immediately

        home_total = sum(type_row_counts.values())
        total_rows += home_total
        elapsed = time.time() - t0
        results[hid_str] = dict(type_row_counts)

        for stype, n in sorted(type_row_counts.items()):
            print(f"    {stype:<30} {n:>10,} rows", flush=True)
        print(f"    {'TOTAL':<30} {home_total:>10,} rows  ({elapsed:.1f}s)", flush=True)

    print(f"\nIngestion complete. {total_rows:,} total rows written to {store_path}", flush=True)
    return results


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Ingest IDEAL sensor data into the EnergyX HistoricStore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--home-ids",
        nargs="+",
        type=int,
        default=_DEFAULT_HOME_IDS,
        metavar="ID",
        help=f"Home IDs to ingest (default: {_DEFAULT_HOME_IDS})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_REPO_ROOT / "data" / "IDEAL",
        help="Root of the IDEAL dataset (default: data/IDEAL/)",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=_REPO_ROOT / "data" / "historic_store",
        help="HistoricStore output path (default: data/historic_store/)",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Write 1 Hz electricity as-is (default: downsample to 1-min averages)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and report files without writing anything",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=== EnergyX IDEAL Ingest ===")
    print(f"  Data root : {args.data_root}")
    print(f"  Store path: {args.store_path}")
    print(f"  Home IDs  : {args.home_ids}")
    print(f"  Downsample: {not args.no_downsample}")
    print(f"  Dry run   : {args.dry_run}")

    if not args.data_root.exists():
        print(f"\nERROR: data root not found: {args.data_root}", file=sys.stderr)
        sys.exit(1)

    ingest(
        data_root=args.data_root,
        store_path=args.store_path,
        home_ids=args.home_ids,
        dry_run=args.dry_run,
        downsample_electricity=not args.no_downsample,
    )


if __name__ == "__main__":
    main()
