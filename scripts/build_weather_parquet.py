#!/usr/bin/env python3
"""Build per-home weather.parquet from Edinburgh MIDAS station data (03160).

Source files (hourly, UTC, station 03160 – Edinburgh Airport):
  data/IDEAL/03160_2016.csv.gz
  data/IDEAL/03160_2017.csv.gz
  data/IDEAL/03160_2018.csv.gz

Output: data/ideal_hierarchy/home{id}/weather.parquet

Columns in output:
  ts   – UTC timestamp (minute-resolution, timezone-aware)
  temp – air temperature (°C)
  rhum – relative humidity (%)
  wdir – wind direction (°, 0-360; note: linearly interpolated, not circular)
  wspd – wind speed (km/h)
  pres – sea-level pressure (hPa)

Processing steps:
  1. Concatenate the three yearly CSVs and build a UTC DatetimeIndex.
  2. Coerce all values to float; NaNs remain as-is at this stage.
  3. Reindex to 1-minute resolution (upsample from 1 h → 1 min).
  4. Linearly interpolate ALL gaps (original NaNs + newly created minute gaps).
  5. Forward-fill / back-fill any residual edge NaNs.
  6. Clip to each home's [starttime, endtime] window from home.csv.
  7. Write weather.parquet alongside the sensor parquets.

Usage::
    venv/bin/python scripts/build_weather_parquet.py
    venv/bin/python scripts/build_weather_parquet.py --home-ids 96 128
    venv/bin/python scripts/build_weather_parquet.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

logger = logging.getLogger(__name__)

_WEATHER_FILES = [
    _REPO / "data" / "IDEAL" / "03160_2016.csv.gz",
    _REPO / "data" / "IDEAL" / "03160_2017.csv.gz",
    _REPO / "data" / "IDEAL" / "03160_2018.csv.gz",
]
_HOME_CSV     = _REPO / "data" / "IDEAL" / "metadata" / "home.csv"
_HIER_ROOT    = _REPO / "data" / "ideal_hierarchy"
_WEATHER_COLS = ["temp", "rhum", "wdir", "wspd", "pres"]
_DATE_FMT     = "%d/%m/%Y %H:%M"


# ---------------------------------------------------------------------------
# Load & build hourly weather series
# ---------------------------------------------------------------------------

def _load_hourly_weather() -> pd.DataFrame:
    """Read all three yearly CSVs, build a UTC DatetimeIndex, return hourly df."""
    frames = []
    for path in _WEATHER_FILES:
        df = pd.read_csv(
            path,
            usecols=["year", "month", "day", "hour"] + _WEATHER_COLS,
        )
        frames.append(df)

    wx = pd.concat(frames, ignore_index=True)

    # Build UTC timestamps from year/month/day/hour columns
    wx["ts"] = pd.to_datetime(
        wx[["year", "month", "day", "hour"]].rename(columns={"hour": "H"}
        ).assign(minute=0, second=0).rename(columns={"H": "hour"}),
        utc=True,
    )
    wx = wx.set_index("ts").sort_index()
    wx = wx[_WEATHER_COLS].apply(pd.to_numeric, errors="coerce")

    # Drop any exact duplicate timestamps (shouldn't exist but just in case)
    wx = wx[~wx.index.duplicated(keep="first")]
    return wx


def _to_minute_resolution(hourly: pd.DataFrame) -> pd.DataFrame:
    """Upsample from 1-hour → 1-minute and interpolate all gaps."""
    # Build a complete 1-minute index spanning the data
    minute_idx = pd.date_range(
        start=hourly.index[0],
        end=hourly.index[-1],
        freq="1min",
        tz="UTC",
    )
    # Reindex: places NaN at all non-hour positions, keeps hourly values at :00
    minute_df = hourly.reindex(minute_idx)

    # Linear interpolation handles both the upsampling gaps AND original NaNs
    minute_df = minute_df.interpolate(method="time", limit_direction="both")

    # Clip wdir to [0, 360) — linear interp can go slightly outside
    minute_df["wdir"] = minute_df["wdir"].clip(0, 360)

    # Final safety: forward-fill then back-fill any remaining edge NaNs
    minute_df = minute_df.ffill().bfill()
    return minute_df


# ---------------------------------------------------------------------------
# Home metadata
# ---------------------------------------------------------------------------

def _load_home_windows() -> dict[str, tuple[datetime, datetime]]:
    """Return {homeid_str: (start_utc, end_utc)} from home.csv."""
    windows: dict[str, tuple[datetime, datetime]] = {}
    with open(_HOME_CSV, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hid = row["homeid"].strip()
            start_str = row["starttime"].strip()
            end_str   = row["endtime"].strip()
            if not start_str or not end_str:
                continue
            try:
                start = pd.Timestamp(datetime.strptime(start_str, _DATE_FMT), tz="UTC")
                end   = pd.Timestamp(datetime.strptime(end_str,   _DATE_FMT), tz="UTC")
                windows[hid] = (start, end)
            except ValueError as exc:
                logger.warning(f"home {hid}: unparseable dates ({exc}), skipping")
    return windows



# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_weather(
    home_ids: list[int] | None = None,
    dry_run: bool = False,
) -> None:
    print("=== EnergyX Weather Parquet Builder ===")
    print(f"  Source files : {[f.name for f in _WEATHER_FILES]}")
    print(f"  Output root  : {_HIER_ROOT}")
    print(f"  Dry run      : {dry_run}\n")

    # 1. Build full minute-resolution weather series
    print("Loading and upsampling weather data...", flush=True)
    hourly = _load_hourly_weather()
    print(f"  Hourly rows  : {len(hourly):,}  ({hourly.index[0]} → {hourly.index[-1]})")

    for col in _WEATHER_COLS:
        n = int(hourly[col].isna().sum())
        if n:
            print(f"  {col}: {n} missing hourly values → will be interpolated")

    minute_wx = _to_minute_resolution(hourly)
    print(f"  Minute rows  : {len(minute_wx):,}  (after upsample + interpolation)\n")

    # 2. Discover home directories under ideal_hierarchy
    home_windows = _load_home_windows()
    home_dirs = sorted(_HIER_ROOT.glob("home*"))
    if not home_dirs:
        print(f"ERROR: No home directories found under {_HIER_ROOT}", file=sys.stderr)
        print("       Run build_ideal_hierarchy.py first.", file=sys.stderr)
        return

    if home_ids:
        requested = {f"home{h}" for h in home_ids}
        home_dirs = [d for d in home_dirs if d.name in requested]

    if not home_dirs:
        print(f"ERROR: No matching home directories for ids {home_ids}", file=sys.stderr)
        return

    # 3. Clip to monitoring window and write per home
    for home_dir in home_dirs:
        hid_str  = home_dir.name           # "home96"
        hid_num  = hid_str.replace("home", "")
        out_path = home_dir / "weather.parquet"

        window = home_windows.get(hid_num)
        if window is None:
            print(f"  {hid_str}: no home.csv entry — skipping")
            continue

        start, end = window
        clipped = minute_wx.loc[
            (minute_wx.index >= start) & (minute_wx.index <= end)
        ].copy()

        if clipped.empty:
            print(f"  {hid_str}: weather data outside window {start} – {end} — skipping")
            continue

        # Reset index so 'ts' becomes a plain column (matches sensor parquet schema)
        clipped = clipped.reset_index().rename(columns={"index": "ts"})
        clipped.insert(0, "home_id", hid_str)

        n_rows         = len(clipped)
        span_start     = clipped["ts"].iloc[0].strftime("%Y-%m-%d %H:%M")
        span_end       = clipped["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M")
        remaining_nan  = int(clipped[_WEATHER_COLS].isna().sum().sum())

        print(f"  {hid_str}: {n_rows:,} rows  [{span_start} → {span_end}]  "
              f"NaN remaining: {remaining_nan}")

        if dry_run:
            print(f"    → [dry run] would write {out_path.relative_to(_REPO)}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        clipped.to_parquet(out_path, index=False)
        size_kb = out_path.stat().st_size // 1024
        print(f"    → wrote {out_path.relative_to(_REPO)}  ({size_kb:,} KB)", flush=True)

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Build per-home weather.parquet from Edinburgh station data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--home-ids", nargs="+", type=int, metavar="ID",
                   help="Limit to specific home IDs (default: all found in ideal_hierarchy/)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned actions without writing any files")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    build_weather(home_ids=args.home_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
