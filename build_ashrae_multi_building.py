"""Build ASHRAE electricity-meter datasets for 10 diverse buildings.

Outputs one CSV per building to data/enhanced/ashrae/multi_building/,
in the same format as building_energy_data_elec.csv so evaluate_energy_benchmark.py
can load them directly via _load_ashrae().

Selected buildings (diverse use types, full-year coverage, >70% non-zero, mean > 10 kWh):
  106  - Education,                     site 1,  small  (high variance)
  492  - Education,                     site 3,  large  (low variance)
  1195 - Office,                        site 13, small  (high variance)
  1274 - Office,                        site 14, small  (low variance)
  965  - Lodging/residential,           site 9,  medium (high variance)
  614  - Lodging/residential,           site 4,  large  (low variance)
  743  - Entertainment/public assembly, site 5,  small  (high variance)
  451  - Entertainment/public assembly, site 3,  small  (low variance)
  455  - Healthcare,                    site 3,  medium (mid variance)
  1243 - Healthcare,                    site 14, large  (low variance)

Usage:
    python build_ashrae_multi_building.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ASHRAE = ROOT / "data/ashrae-energy-prediction"
OUT_DIR = ROOT / "data/enhanced/ashrae/multi_building"

BUILDING_IDS = [106, 492, 1195, 1274, 965, 614, 743, 451, 455, 1243]

LAG_COLS  = {"lag_1": 1, "lag_24": 24, "lag_168": 168}
ROLL_COLS = {"roll_mean_24": 24, "roll_std_24": 24}


def build_building(
    building_id: int,
    train: pd.DataFrame,
    meta: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    df = train[(train["building_id"] == building_id) & (train["meter"] == 0)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    df = df.resample("h").mean().ffill()
    df["meter_reading"] = df["meter_reading"].clip(lower=0)

    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["day"]        = df.index.day
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    brow    = meta[meta["building_id"] == building_id].iloc[0]
    site_id = int(brow["site_id"])

    site_wx = weather[weather["site_id"] == site_id].copy()
    site_wx["timestamp"] = pd.to_datetime(site_wx["timestamp"])
    site_wx = site_wx.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    site_wx = site_wx.resample("h").mean().ffill()

    for col in ["air_temperature", "cloud_coverage", "dew_temperature",
                "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]:
        if col in site_wx.columns:
            df[col] = site_wx[col].reindex(df.index).ffill().bfill()

    for name, shift in LAG_COLS.items():
        df[name] = df["meter_reading"].shift(shift)
    for name, window in ROLL_COLS.items():
        base = df["meter_reading"].shift(1)
        df[name] = base.rolling(window).mean() if "mean" in name else base.rolling(window).std()

    df = df.dropna(subset=["lag_168"])

    df = df.reset_index()
    df["building_id"]  = building_id
    df["site_id"]      = site_id
    df["primary_use"]  = brow["primary_use"]
    df["square_feet"]  = brow["square_feet"]

    keep = [
        "timestamp", "meter_reading",
        "air_temperature", "cloud_coverage", "dew_temperature",
        "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",
        "hour", "dayofweek", "month", "day", "is_weekend",
        "lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_24",
        "building_id", "site_id", "primary_use", "square_feet",
    ]
    return df[[c for c in keep if c in df.columns]]


def main():
    print("Loading ASHRAE source files...")
    train   = pd.read_csv(ASHRAE / "train.csv")
    meta    = pd.read_csv(ASHRAE / "building_metadata.csv")
    weather = pd.read_csv(ASHRAE / "weather_train.csv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for bid in BUILDING_IDS:
        print(f"  Building {bid}...", end=" ", flush=True)
        df = build_building(bid, train, meta, weather)
        out_path = OUT_DIR / f"building_{bid}_elec.csv"
        df.to_csv(out_path, index=False)
        print(f"{len(df)} rows → {out_path.name}")

    print(f"\nDone. {len(BUILDING_IDS)} datasets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
