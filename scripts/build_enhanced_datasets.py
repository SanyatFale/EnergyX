#!/usr/bin/env python3
"""Build enhanced datasets from raw IDEAL hierarchy and ASHRAE CSVs.

IDEAL enhancements (four variants: 5-min, 15-min, 30-min, 60-min):
  - Deduplication of same-timestamp rows
  - Resampling to target frequency averages
  - log1p(value) column added for electricity and appliance power targets
  - lag_24h and lag_168h columns added to all sensor files (step count
    adjusted per variant so they always represent 24 h and 168 h of history)
  - Calendar and weather files resampled to same grid

ASHRAE enhancements:
  - Fix roll_mean_24 and roll_std_24 look-ahead leakage (recompute with shift(1))
  - All other columns preserved as-is

Output:
  data/enhanced/ideal/variant_5/   (same folder structure as ideal_hierarchy)
  data/enhanced/ideal/variant_15/
  data/enhanced/ideal/variant_30/
  data/enhanced/ideal/variant_60/
  data/enhanced/ashrae/

Usage:
    python scripts/build_enhanced_datasets.py
    python scripts/build_enhanced_datasets.py --skip-ideal
    python scripts/build_enhanced_datasets.py --skip-ashrae
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
_IDEAL_SRC = _REPO_ROOT / "data" / "ideal_hierarchy"
_ENHANCED_ROOT = _REPO_ROOT / "data" / "enhanced"
_ASHRAE_SRC = _REPO_ROOT / "data" / "raw"

# sensor_type values whose 'value' column is an energy/power target → add log1p
_ELECTRICITY_TYPES = {
    "electricity_real",
    "electricity_apparent",
    "appliance_power",
    "gas_pulse",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_sensor(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Deduplicate, sort, resample a sensor parquet to freq. Returns new df."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset="ts", keep="first").sort_values("ts")

    # Preserve scalar metadata columns (same value in every row)
    meta_cols = [c for c in df.columns if c not in ("ts", "value")]
    meta = {c: df[c].iloc[0] for c in meta_cols}

    numeric = df.set_index("ts")[["value"]].resample(freq).mean()
    numeric = numeric.reset_index()

    for col, val in meta.items():
        numeric[col] = val

    # Reorder columns to match original
    orig_cols = list(df.columns)
    numeric = numeric[[c for c in orig_cols if c in numeric.columns]]
    return numeric


def _resample_weather(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Deduplicate + resample multi-column weather parquet."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset="ts", keep="first").sort_values("ts")
    home_id = df["home_id"].iloc[0]
    value_cols = [c for c in df.columns if c not in ("ts", "home_id")]
    resampled = df.set_index("ts")[value_cols].resample(freq).mean().reset_index()
    resampled.insert(0, "home_id", home_id)
    return resampled


def _resample_calendar(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample calendar to target freq.

    The source calendar is hourly. For sub-hourly targets (30-min) we upsample
    and forward-fill so every slot in the new grid has a valid calendar label.
    For hourly targets the grid matches exactly and no fill is needed.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset="ts", keep="first").sort_values("ts")
    home_id = df["home_id"].iloc[0]
    cal_cols = [c for c in df.columns if c not in ("ts", "home_id")]

    # Reindex to the target frequency grid and forward-fill gaps
    full_idx = pd.date_range(df["ts"].iloc[0], df["ts"].iloc[-1], freq=freq, tz="UTC")
    resampled = (
        df.set_index("ts")[cal_cols]
        .reindex(full_idx)
        .ffill()
        .reset_index()
        .rename(columns={"index": "ts"})
    )
    resampled.insert(1, "home_id", home_id)
    return resampled


def _add_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """Add value_log1p column if 'value' exists and is non-negative."""
    if "value" in df.columns:
        df = df.copy()
        df["value_log1p"] = np.log1p(df["value"].clip(lower=0))
    return df


def _add_lags(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Add lag_24h and lag_168h columns to a sensor dataframe.

    Steps are derived from the resampling frequency so both lags always
    represent exactly 24 h and 168 h of history regardless of variant.
    Lag values are NaN for rows where history is insufficient (series start).
    """
    if "value" not in df.columns:
        return df

    freq_minutes = pd.tseries.frequencies.to_offset(freq).nanos / 60e9
    steps_24h  = int(round(24  * 60 / freq_minutes))
    steps_168h = int(round(168 * 60 / freq_minutes))

    df = df.copy()
    df["lag_24h"]  = df["value"].shift(steps_24h)
    df["lag_168h"] = df["value"].shift(steps_168h)
    return df


# ---------------------------------------------------------------------------
# IDEAL pipeline
# ---------------------------------------------------------------------------

def process_ideal(freq: str, variant_name: str) -> None:
    out_root = _ENHANCED_ROOT / "ideal" / variant_name
    print(f"\n--- IDEAL {variant_name} ({freq}) → {out_root} ---")

    parquet_files = sorted(_IDEAL_SRC.glob("**/*.parquet"))
    if not parquet_files:
        print("  No parquet files found in ideal_hierarchy — check path.", file=sys.stderr)
        return

    for i, src in enumerate(parquet_files):
        rel = src.relative_to(_IDEAL_SRC)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src)

        if "value" in df.columns and "ts" in df.columns:
            # Standard sensor file (electricity, temperature, appliance, etc.)
            sensor_type = df["sensor_type"].iloc[0] if "sensor_type" in df.columns else ""
            resampled = _resample_sensor(df, freq)
            if sensor_type in _ELECTRICITY_TYPES:
                resampled = _add_log1p(resampled)
            resampled = _add_lags(resampled, freq)
            resampled.to_parquet(dst, index=False)

        elif "temp" in df.columns:
            # Weather parquet (multi-column, no 'value')
            resampled = _resample_weather(df, freq)
            resampled.to_parquet(dst, index=False)

        elif "hour" in df.columns:
            # Calendar parquet
            resampled = _resample_calendar(df, freq)
            resampled.to_parquet(dst, index=False)

        else:
            # Unknown structure — copy as-is
            df.to_parquet(dst, index=False)
            logger.warning(f"Unknown structure, copied as-is: {rel}")

        suffix = " [+ log1p]" if dst.exists() and "value_log1p" in pd.read_parquet(dst).columns else ""
        print(f"  [{i+1:3d}/{len(parquet_files)}] {rel}{suffix}", flush=True)

    print(f"  Done — {len(parquet_files)} files written to {out_root}")


# ---------------------------------------------------------------------------
# ASHRAE pipeline
# ---------------------------------------------------------------------------

def process_ashrae() -> None:
    out_dir = _ENHANCED_ROOT / "ashrae"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- ASHRAE → {out_dir} ---")

    csv_files = [
        _ASHRAE_SRC / "building_energy_data (copy 1)_elec.csv",
        _ASHRAE_SRC / "building_energy_data (copy 1)_chilled.csv",
    ]

    for src in csv_files:
        if not src.exists():
            print(f"  Not found, skipping: {src.name}", file=sys.stderr)
            continue

        df = pd.read_csv(src)

        # Identify timestamp column (first col for chilled has no header)
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values(ts_col).reset_index(drop=True)

        # Fix rolling leakage: recompute roll_mean_24 and roll_std_24
        # Correct definition: trailing 24-hour window ending BEFORE current row
        if "roll_mean_24" in df.columns:
            df["roll_mean_24"] = df["meter_reading"].shift(1).rolling(24).mean()
        if "roll_std_24" in df.columns:
            df["roll_std_24"] = df["meter_reading"].shift(1).rolling(24).std()

        out_name = src.stem.replace("building_energy_data (copy 1)_", "building_energy_data_") + ".csv"
        dst = out_dir / out_name
        df.to_csv(dst, index=False)
        print(f"  Written: {dst.name}  ({len(df)} rows, rolling features recomputed)")

    print(f"  Done — ASHRAE files in {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-ideal", action="store_true", help="Skip IDEAL processing")
    p.add_argument("--skip-ashrae", action="store_true", help="Skip ASHRAE processing")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=== EnergyX Enhanced Dataset Builder ===")
    print(f"  Source IDEAL : {_IDEAL_SRC}")
    print(f"  Source ASHRAE: {_ASHRAE_SRC}")
    print(f"  Output root  : {_ENHANCED_ROOT}")

    if not args.skip_ideal:
        process_ideal("5min",  "variant_5")
        process_ideal("15min", "variant_15")
        process_ideal("30min", "variant_30")
        process_ideal("60min", "variant_60")

    if not args.skip_ashrae:
        process_ashrae()

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
