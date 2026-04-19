"""
Consolidate ASHRAE energy, weather, metadata, and LEAD anomaly data
for the top 5 buildings by anomaly count from LEAD 1.0.

Top 5 (by LEAD anomaly count):
    Rank 1: building_id=1319  (775 anomalies, 8.82%)
    Rank 2: building_id=1258  (713 anomalies, 8.12%)
    Rank 3: building_id=439   (567 anomalies, 6.47%)
    Rank 4: building_id=247   (479 anomalies, 5.45%)
    Rank 5: building_id=1225  (463 anomalies, 5.27%)

Source data (all 2016):
    - ASHRAE train energy  : building_id, meter, timestamp, meter_reading
    - ASHRAE weather_train : site_id, timestamp, weather features
    - Building metadata    : site_id, building_id, primary_use, sq_ft, ...
    - LEAD 1.0             : building_id, timestamp, anomaly (1=anomalous)

Output: data/consolidated/building_{id}_consolidated.csv  (one per building)

Schema per output file:
    building_id, meter, meter_type, timestamp,
    meter_reading,
    site_id, primary_use, square_feet, year_built, floor_count,
    air_temperature, cloud_coverage, dew_temperature,
    precip_depth_1_hr, sea_level_pressure, wind_direction, wind_speed,
    anomaly    <- 1 if LEAD flags anomaly at that timestamp, empty otherwise
"""

import os
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data/ashrae-energy-prediction"
LEAD_PATH = "lead1.0-small.csv"
OUT_DIR   = "data/consolidated"
TOP5      = [1319, 1258, 439, 247, 1225]

METER_NAMES = {0: "electricity", 1: "chilled_water", 2: "steam", 3: "hot_water"}
WEATHER_COLS = [
    "air_temperature", "cloud_coverage", "dew_temperature",
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",
]

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Metadata ───────────────────────────────────────────────────────────────
print("Loading building metadata...")
meta = pd.read_csv(f"{DATA_DIR}/building_metadata.csv")
meta_top5 = meta[meta["building_id"].isin(TOP5)].copy()
print(f"  {len(meta_top5)} rows for top 5 buildings")

# ── 2. ASHRAE train energy (2016) ─────────────────────────────────────────────
print("Loading ASHRAE train energy (streaming)...")
train_chunks = []
for chunk in pd.read_csv(f"{DATA_DIR}/train.csv", chunksize=500_000,
                         parse_dates=["timestamp"]):
    subset = chunk[chunk["building_id"].isin(TOP5)]
    if len(subset):
        train_chunks.append(subset)
energy = pd.concat(train_chunks, ignore_index=True)
energy["meter_type"] = energy["meter"].map(METER_NAMES)
energy = energy.sort_values(["building_id", "meter", "timestamp"]).reset_index(drop=True)
print(f"  {len(energy):,} energy rows loaded")

# ── 3. Weather train (2016) ───────────────────────────────────────────────────
print("Loading weather_train...")
site_ids = meta_top5["site_id"].unique().tolist()
weather = pd.read_csv(f"{DATA_DIR}/weather_train.csv", parse_dates=["timestamp"])
weather = weather[weather["site_id"].isin(site_ids)].sort_values(
    ["site_id", "timestamp"]
).reset_index(drop=True)
print(f"  {len(weather):,} weather rows for sites {site_ids}")

# ── 4. LEAD anomaly labels (2016) ─────────────────────────────────────────────
print("Loading LEAD anomaly labels...")
lead = pd.read_csv(LEAD_PATH, parse_dates=["timestamp"],
                   usecols=["building_id", "timestamp", "anomaly"])
lead_top5 = lead[lead["building_id"].isin(TOP5)].copy()
# Only keep rows where anomaly == 1; everything else stays empty (NaN) in output
lead_flags = (
    lead_top5[lead_top5["anomaly"] == 1][["building_id", "timestamp"]]
    .copy()
)
lead_flags["anomaly"] = 1
print(f"  {len(lead_flags):,} anomaly-flagged timestamps across top 5 buildings")

# ── 5. Build one CSV per building ─────────────────────────────────────────────
for bid in TOP5:
    print(f"\n{'='*60}")
    print(f"Building {bid}")
    print(f"{'='*60}")

    # --- energy ---
    bld_energy = energy[energy["building_id"] == bid].copy()

    # --- metadata (broadcast) ---
    bld_meta = meta_top5[meta_top5["building_id"] == bid].iloc[0]
    site_id = int(bld_meta["site_id"])
    for col in ["site_id", "primary_use", "square_feet", "year_built", "floor_count"]:
        bld_energy[col] = bld_meta[col]

    # --- weather (merge_asof on timestamp, nearest within 1h) ---
    bld_weather = (
        weather[weather["site_id"] == site_id][["timestamp"] + WEATHER_COLS]
        .sort_values("timestamp")
        .copy()
    )
    bld_energy = bld_energy.sort_values("timestamp")
    bld_energy = pd.merge_asof(
        bld_energy,
        bld_weather,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    # --- LEAD anomaly (building-level: same flag across all meters at that ts) ---
    bld_flags = lead_flags[lead_flags["building_id"] == bid][["timestamp", "anomaly"]].copy()
    bld_energy = bld_energy.merge(bld_flags, on="timestamp", how="left")
    # Left join: rows with no LEAD match stay NaN — correct (anomaly = empty)

    # --- column order ---
    cols = (
        ["building_id", "meter", "meter_type", "timestamp", "meter_reading"]
        + ["site_id", "primary_use", "square_feet", "year_built", "floor_count"]
        + WEATHER_COLS
        + ["anomaly"]
    )
    bld_energy = bld_energy[cols].sort_values(["meter", "timestamp"]).reset_index(drop=True)

    # --- summary ---
    meters = sorted(bld_energy["meter"].unique().tolist())
    meter_labels = [METER_NAMES[m] for m in meters]
    n_anom = int(bld_energy["anomaly"].notna().sum())
    print(f"  Rows      : {len(bld_energy):,}")
    print(f"  Meters    : {meter_labels}")
    print(f"  Date range: {bld_energy['timestamp'].min()} -> {bld_energy['timestamp'].max()}")
    print(f"  Anomalies : {n_anom} flagged timestamps (across all meters)")
    print(f"  Missing meter_reading: {bld_energy['meter_reading'].isna().sum():,}")
    print(f"  Missing weather rows : {bld_energy['air_temperature'].isna().sum():,}")

    # --- write ---
    out_path = f"{OUT_DIR}/building_{bid}_consolidated.csv"
    bld_energy.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  -> {out_path}  ({size_mb:.1f} MB)")

print("\n\nAll 5 CSVs written to data/consolidated/")
print("Summary:")
for bid in TOP5:
    path = f"{OUT_DIR}/building_{bid}_consolidated.csv"
    mb = os.path.getsize(path) / 1e6
    df = pd.read_csv(path, usecols=["building_id", "anomaly"])
    print(f"  building_{bid}_consolidated.csv  |  {mb:.1f} MB  |  {len(df):,} rows  |  {int(df['anomaly'].notna().sum())} anomalies")
