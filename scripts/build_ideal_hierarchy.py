#!/usr/bin/env python3
"""Build a human-readable Parquet hierarchy from the IDEAL dataset.

Output structure::

    data/ideal_hierarchy/
      home96/
          electricity_mains.parquet      <- household_sensors (home level)
          gas.parquet
          central_heating_return.parquet
          livingroom/
              temperature.parquet        <- room sensorbox (room level)
              humidity.parquet
              radiator_output.parquet    <- tempprobe in room
              gas_fire/
                  gas_fire.parquet       <- electric-appliance (appliance level)
          kitchen/
              temperature.parquet
              fridgefreezer/
                  fridgefreezer.parquet
          bedroom_1/
              temperature.parquet
          bedroom_2/
              ...
      home128/
          ...

Multiple rooms of the same type are disambiguated by ascending room-id order:
bedroom_1, bedroom_2, etc.  Single rooms keep the plain type name.

Usage::

    python scripts/build_ideal_hierarchy.py --home-ids 96 128
    python scripts/build_ideal_hierarchy.py --home-ids 96 128 --dry-run
    python scripts/build_ideal_hierarchy.py --home-ids 96 128 --no-downsample
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from energyx.data.ideal_loader import (
    FileMetadata,
    IDEALLoader,
    SensorType,
    _FNAME_RE,
    parse_filename,
    stream_sensor_file,
)

logger = logging.getLogger(__name__)

_DEFAULT_HOME_IDS = [96, 128]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "IDEAL"
_DEFAULT_OUT_ROOT = _REPO_ROOT / "data" / "ideal_hierarchy"

_DATA_SUBTREES = [
    ("household_sensors/sensordata", "household"),
    ("room_and_appliance_sensors/sensordata", "room_and_appliance"),
    ("auxiliarydata/sensordata", "auxiliary"),
]

# Files from these sensorboxes become appliance sub-directories under their room
_APPLIANCE_SENSORBOXES = {"electric-appliance"}


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

def _clean(name: str) -> str:
    """Lowercase + replace non-alphanumeric runs with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_room_meta(metadata_dir: Path) -> Dict[str, Dict]:
    """Return {room_id_str: row_dict} from room.csv (BOM-safe)."""
    room_csv = metadata_dir / "room.csv"
    rooms: Dict[str, Dict] = {}
    if not room_csv.exists():
        logger.warning(f"room.csv not found at {room_csv}")
        return rooms
    with open(room_csv, encoding="utf-8-sig") as fh:  # utf-8-sig strips BOM
        reader = csv.DictReader(fh)
        for row in reader:
            rid = row.get("roomid", "").strip()
            if rid:
                rooms[rid] = row
    return rooms


def _build_room_display(
    room_metas: Dict[str, Dict],
    seen_room_ids: List[str],
) -> Dict[str, str]:
    """Map numeric room_id strings → display names.

    When multiple rooms share the same type they are numbered in ascending
    room_id order: bedroom_1, bedroom_2 …  Single rooms keep the plain name.
    """
    by_type: Dict[str, List[str]] = defaultdict(list)
    for rid in seen_room_ids:
        meta = room_metas.get(rid)
        rtype = meta["type"].strip() if meta else f"room{rid}"
        by_type[rtype].append(rid)
    for rtype in by_type:
        by_type[rtype].sort(key=lambda r: int(r))

    display: Dict[str, str] = {}
    for rtype, rids in by_type.items():
        if len(rids) == 1:
            display[rids[0]] = _clean(rtype)
        else:
            for i, rid in enumerate(rids, start=1):
                display[rid] = f"{_clean(rtype)}_{i}"
    return display


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

@dataclass
class _SensorEntry:
    """Everything needed to write one Parquet file."""
    file_meta: FileMetadata
    sensorbox: str    # raw sensorbox from filename
    subtype: str      # raw subtype from filename
    room_type: str    # e.g. "kitchen"
    room_id_num: str  # numeric room id, e.g. "999"
    subtree: str      # "household" | "room_and_appliance" | "auxiliary"


def _discover(data_root: Path, home_ids: List[int]) -> List[_SensorEntry]:
    """Scan all subtrees; return one _SensorEntry per matched sensor file.

    Note: ``_FNAME_RE`` uses a greedy ``[a-z0-9]*`` for ``room_type``, so
    the captured ``room_id`` group may be only the *last* digit of a 4-digit
    room ID (e.g. "bathroom125" + "2" instead of "bathroom" + "1252").
    We always derive the *true* numeric room ID from ``file_meta.room_id``
    (which is the full reconstructed string like "bathroom1252") by stripping
    trailing digits, and similarly extract the clean alpha room type.
    """
    home_strs = {f"home{h}" for h in home_ids}
    entries: List[_SensorEntry] = []

    _trailing_digits = re.compile(r"\d+$")
    _leading_letters = re.compile(r"^([a-z]+)", re.IGNORECASE)

    for rel, label in _DATA_SUBTREES:
        subtree_path = data_root / rel
        if not subtree_path.exists():
            logger.debug(f"Subtree not found, skipping: {subtree_path}")
            continue
        for gz_path in sorted(subtree_path.glob("*.gz")):
            m = _FNAME_RE.match(gz_path.name)
            if not m:
                continue
            if f"home{m.group('home_id')}" not in home_strs:
                continue
            file_meta = parse_filename(gz_path)
            if file_meta is None:
                continue

            # file_meta.room_id is the full compound string e.g. "bathroom1252"
            full_room = file_meta.room_id or ""
            num_m = _trailing_digits.search(full_room)
            room_id_num = num_m.group() if num_m else full_room
            alpha_m = _leading_letters.match(full_room)
            room_type = alpha_m.group(1).lower() if alpha_m else full_room

            entries.append(_SensorEntry(
                file_meta=file_meta,
                sensorbox=m.group("sensorbox").lower(),
                subtype=m.group("sensor_subtype").lower(),
                room_type=room_type,
                room_id_num=room_id_num,
                subtree=label,
            ))
    return entries



# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_paths(
    entries: List[_SensorEntry],
    out_root: Path,
    room_metas: Dict[str, Dict],
) -> Dict[int, Path]:
    """Map entry index → output Parquet path, disambiguating collisions."""

    # Build per-home room display maps from all seen room IDs
    home_room_ids: Dict[str, List[str]] = defaultdict(list)
    for e in entries:
        home_room_ids[e.file_meta.home_id].append(e.room_id_num)

    room_display: Dict[str, Dict[str, str]] = {}
    for home_id_str, rids in home_room_ids.items():
        unique = list(dict.fromkeys(rids))  # deduplicate, preserve first-seen order
        room_display[home_id_str] = _build_room_display(room_metas, unique)

    # First pass — candidate paths
    candidate: Dict[int, Path] = {}
    for i, e in enumerate(entries):
        home_dir = out_root / e.file_meta.home_id
        rd = room_display.get(e.file_meta.home_id, {})
        room_name = rd.get(e.room_id_num, f"{e.room_type}_{e.room_id_num}")
        stem = _clean(e.subtype)

        if e.subtree == "household":
            candidate[i] = home_dir / f"{stem}.parquet"
        elif e.sensorbox in _APPLIANCE_SENSORBOXES:
            candidate[i] = home_dir / room_name / stem / f"{stem}.parquet"
        else:
            candidate[i] = home_dir / room_name / f"{stem}.parquet"

    # Second pass — detect collisions and append sensor_id to disambiguate
    path_count: Dict[Path, int] = defaultdict(int)
    for p in candidate.values():
        path_count[p] += 1

    resolved: Dict[int, Path] = {}
    for i, p in candidate.items():
        if path_count[p] > 1:
            sid = entries[i].file_meta.sensor_id
            stem = _clean(entries[i].subtype)
            new_stem = f"{stem}_{sid}"
            if entries[i].sensorbox in _APPLIANCE_SENSORBOXES:
                p = p.parent.parent / new_stem / f"{new_stem}.parquet"
            else:
                p = p.parent / f"{new_stem}.parquet"
        resolved[i] = p
    return resolved


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_parquet(
    entry: _SensorEntry,
    out_path: Path,
    downsample_electricity: bool,
) -> int:
    """Read one sensor file, optionally downsample, write Parquet. Returns row count."""
    chunks = []
    for chunk in stream_sensor_file(entry.file_meta):
        if downsample_electricity and entry.file_meta.sensor_type in {
            SensorType.ELECTRICITY_APPARENT, SensorType.ELECTRICITY_REAL
        }:
            chunk = IDEALLoader._downsample_electricity(chunk)
        chunks.append(chunk)

    if not chunks:
        return 0

    df = pd.concat(chunks, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_hierarchy(
    data_root: Path,
    out_root: Path,
    home_ids: List[int],
    dry_run: bool = False,
    downsample_electricity: bool = True,
) -> None:
    """Build the full HOME > ROOM > APPLIANCE/PROBE Parquet hierarchy."""
    metadata_dir = data_root / "metadata"
    room_metas = _load_room_meta(metadata_dir)

    print(f"\nScanning IDEAL sensor files for homes {home_ids}...", flush=True)
    entries = _discover(data_root, home_ids)

    if not entries:
        print("No files found — check --data-root path.", file=sys.stderr)
        return

    by_home: Dict[str, List[_SensorEntry]] = defaultdict(list)
    for e in entries:
        by_home[e.file_meta.home_id].append(e)

    print(f"Found {len(entries)} sensor files:\n")
    for hid in sorted(by_home):
        n_h = sum(1 for e in by_home[hid] if e.subtree == "household")
        n_r = sum(1 for e in by_home[hid]
                  if e.subtree != "household" and e.sensorbox not in _APPLIANCE_SENSORBOXES)
        n_a = sum(1 for e in by_home[hid] if e.sensorbox in _APPLIANCE_SENSORBOXES)
        print(f"  {hid:>10}: {len(by_home[hid]):3d} files  "
              f"[home-level={n_h}, room-level={n_r}, appliance-level={n_a}]")

    path_map = _resolve_paths(entries, out_root, room_metas)

    if dry_run:
        print("\nDry run — planned output paths:\n")
        for i, e in enumerate(entries):
            rel = path_map[i].relative_to(out_root)
            print(f"  {rel}")
        print(f"\nTotal: {len(entries)} files — nothing written.")
        return

    print(f"\nWriting Parquet files to {out_root}/\n")
    total_rows = 0
    t0 = time.time()

    for i, e in enumerate(entries):
        out_path = path_map[i]
        rel = out_path.relative_to(out_root)
        try:
            n_rows = _write_parquet(e, out_path, downsample_electricity)
            total_rows += n_rows
            print(f"  [{i + 1:3d}/{len(entries)}] {rel}  ({n_rows:,} rows)", flush=True)
        except Exception as exc:
            print(f"  [{i + 1:3d}/{len(entries)}] ERROR {rel}: {exc}", flush=True)
            logger.exception(f"Error writing {out_path}")

    elapsed = time.time() - t0
    print(f"\nDone. {total_rows:,} rows across {len(entries)} Parquet files "
          f"in {elapsed:.1f}s  →  {out_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Build a human-readable Parquet hierarchy from the IDEAL dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--home-ids", nargs="+", type=int, default=_DEFAULT_HOME_IDS, metavar="ID",
                   help=f"Home IDs to process (default: {_DEFAULT_HOME_IDS})")
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT,
                   help="Root of the IDEAL dataset (default: data/IDEAL/)")
    p.add_argument("--out-root", type=Path, default=_DEFAULT_OUT_ROOT,
                   help="Output hierarchy root (default: data/ideal_hierarchy/)")
    p.add_argument("--no-downsample", action="store_true",
                   help="Write electricity at full 1 Hz resolution (default: 1-min averages)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show planned output paths without writing anything")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=== EnergyX IDEAL Hierarchy Builder ===")
    print(f"  Data root : {args.data_root}")
    print(f"  Output    : {args.out_root}")
    print(f"  Home IDs  : {args.home_ids}")
    print(f"  Downsample: {not args.no_downsample}")
    print(f"  Dry run   : {args.dry_run}")

    if not args.data_root.exists():
        print(f"\nERROR: data root not found: {args.data_root}", file=sys.stderr)
        sys.exit(1)

    build_hierarchy(
        data_root=args.data_root,
        out_root=args.out_root,
        home_ids=args.home_ids,
        dry_run=args.dry_run,
        downsample_electricity=not args.no_downsample,
    )


if __name__ == "__main__":
    main()
