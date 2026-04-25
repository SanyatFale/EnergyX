# Data Findings & Preprocessing Recommendations

---

## 0. Dataset Map

### Raw / Source Data

| Path | Contents |
|---|---|
| `data/IDEAL/` | Raw IDEAL sensor data (`.csv.gz`), 1 Hz, organised by subtree |
| `data/IDEAL/household_sensors/sensordata/` | Home-level electricity (mains, electric-combined, cooker, shower) |
| `data/IDEAL/room_and_appliance_sensors/sensordata/` | Room sensorboxes + appliance clamps (fridge, washer, kettle, etc.) |
| `data/IDEAL/auxiliarydata/` | Battery, propagation, hourly backup readings, anomalous readings |
| `data/IDEAL/metadata/room.csv` | Room metadata: room ID → type (kitchen, bedroom, etc.) |
| `data/raw/building_energy_data (copy 1)_elec.csv` | ASHRAE electricity — hourly, pre-engineered features, 5,238 rows |
| `data/raw/building_energy_data (copy 1)_chilled.csv` | ASHRAE chilled water — hourly, pre-engineered features, 6,955 rows |

### Processed Hierarchy (`data/ideal_hierarchy/`)

1-min parquet files extracted and downsampled from raw gz by `scripts/build_ideal_hierarchy.py`. Folder structure mirrors physical layout: `homeXXX/` → room folder → appliance folder.

| File pattern | Sensor type | Contents |
|---|---|---|
| `homeXXX/mains.parquet` | electricity_real (W) | Whole-home real power from utility-room subcircuit meter |
| `homeXXX/electric_combined.parquet` | electricity_apparent (VA) | Whole-home apparent power from main clamp (longer coverage) |
| `homeXXX/gas.parquet` | gas_pulse (Wh) | Gas consumption |
| `homeXXX/cooker.parquet` | electricity_real (W) | Cooker subcircuit (home96 only) |
| `homeXXX/shower.parquet` | electricity_real (W) | Shower subcircuit (home128 only) |
| `homeXXX/weather.parquet` | — | Outdoor: temp, rhum, wdir, wspd, pres (1-min interpolated) |
| `homeXXX/calendar.parquet` | — | Hourly: hour, dayofweek, month, is_weekend, is_holiday (Scottish) |
| `homeXXX/<room>/temperature.parquet` | temperature_room | Room air temperature |
| `homeXXX/<room>/humidity.parquet` | humidity | Room relative humidity |
| `homeXXX/<room>/radiator_*.parquet` | temperature_probe | Radiator flow/return/output probes |
| `homeXXX/<room>/<appliance>/<appliance>.parquet` | appliance_power (W) | Per-appliance power (fridge, washer, kettle, microwave, etc.) |

### Enhanced Data (`data/enhanced/`)

Produced by `scripts/build_enhanced_datasets.py`. Original data untouched.

| Path | Contents |
|---|---|
| `data/enhanced/ideal/variant_5/` | IDEAL — 5-min, deduped, resampled, log1p + lag_24h + lag_168h added |
| `data/enhanced/ideal/variant_15/` | IDEAL — 15-min, same processing |
| `data/enhanced/ideal/variant_30/` | IDEAL — 30-min, same processing |
| `data/enhanced/ideal/variant_60/` | IDEAL — 60-min, same processing |
| `data/enhanced/ashrae/building_energy_data_elec.csv` | ASHRAE elec — rolling leakage fixed |
| `data/enhanced/ashrae/building_energy_data_chilled.csv` | ASHRAE chilled — rolling leakage fixed |

Each IDEAL enhanced variant preserves the full `ideal_hierarchy/` folder structure. Sensor files gain columns: `value_log1p` (electricity/appliance/gas only), `lag_24h`, `lag_168h`. Lag step counts are scaled per variant so both always represent exactly 24 h and 168 h of wall-clock history.

---

## 1. IDEAL Dataset

### 1.1 Structure

Two homes, stored as a hierarchical Parquet tree: home-level, room-level, and appliance-level. Electricity is the forecasting target; non-electric sensors (temperature, humidity, radiator) serve as covariates.

| File | Sensor type | Raw freq | Date range | Rows |
|---|---|---|---|---|
| home96/electric_combined | electricity_apparent (VA) | 1-min | Feb 2017 – Jun 2018 | 704,979 |
| home96/mains | electricity_real (W) | 1-min | Aug 9–21 2017 only | 17,508 |
| home128/mains | electricity_real (W) | 1-min | Aug 2017 – Jun 2018 | 450,708 |
| home128/electric_combined | electricity_apparent (VA) | 1-min | Jun 2017 – Jun 2018 | 540,738 |

> **home96/mains covers only 12 days** — too short to use as primary target for home96. Use `electric_combined` for home96.
> `electric_combined` = apparent power (VA); `mains` = real power (W). Do not mix them as one target.

---

### 1.2 Data Quality Issues

#### Duplicate timestamps
All files contain duplicate entries at the same timestamp (same-second repeated readings). These must be deduplicated before resampling.

**Fix:** `.drop_duplicates(subset='ts', keep='first')` before any further processing.

#### Gaps (missing intervals)
After resampling to 1-min grid, gap rates are significant:

| Series | Gap % at 1-min | Gap % at 1-hour |
|---|---|---|
| home96/electric_combined | 28.6% | ~7% (estimated) |
| home128/mains | 8.7% | ~2% |
| home96/washingmachine | 48.5% | high |
| home128/fridgefreezer | 16.6% | moderate |
| home96/fridgefreezer | 0% | 0% |

**home96/electric_combined has 28.6% 1-min gaps** — the largest data quality problem in the dataset.

**Fix:** Resample to 1-hour with `.resample('1h').mean()`. Remaining hourly NaNs: linearly interpolate up to 3 consecutive hours; drop windows with gaps > 3 hours from training.

#### Near-zero appliance series
Most appliances are rarely-on devices:

| Appliance | Zeros % | Median (W) | Forecastable? |
|---|---|---|---|
| home96/fridgefreezer | 0% | 159 W | Yes — continuous cycling |
| home96/washingmachine | 27% | 1 W | Yes — event-driven, enough data |
| home128/fridgefreezer | 20% | 41 W | Yes — with gap handling |
| home96/microwave | 55% | 0 W | No — spike-only |
| home96/kettle | 78% | 0 W | No — skip |
| home128/kettle/toaster/vacuumcleaner | 85–100% | 0 W | No — skip |

**Fix for forecastable appliances:** apply `log1p` transform; use rolling binary on/off flag as auxiliary feature.
**Fix for skip appliances:** exclude from regression; optionally model as binary on/off classifier separately.

---

### 1.3 Missing Feature Groups

#### Resampling (required)
All IDEAL data is at 1-min. Models need hourly data to learn meaningful patterns and to align with weather covariates.

**Fix:** `.resample('1h').mean()` as the first step after loading + deduplication.

#### Calendar features (added — `calendar.parquet`)
IDEAL had no time-of-day, day-of-week, or holiday features.

**Created:** `data/ideal_hierarchy/home96/calendar.parquet` and `home128/calendar.parquet`.

Columns: `ts, home_id, hour, dayofweek (0=Mon), month, is_weekend, is_holiday`

Scottish public holidays used (2017–2018): 2 Jan, Good Friday, May BH, Spring BH, Summer BH (Scotland), St Andrew's Day (30 Nov), Christmas, Boxing Day.

Hourly granularity, aligned to UTC, local time (Europe/London) used for all calendar derivations.

---

### 1.4 Correlation Analysis (Pearson, variant_60 / hourly enhanced data)

Computed on `data/enhanced/ideal/variant_60` — deduplicated, hourly-resampled. Lags and rolling features computed with `shift(1)` prior to rolling to avoid leakage. `lag_1h` = 1-step (1 hour); `lag_24h` = 24 steps (24 h); `lag_168h` = 168 steps (1 week).

#### home96 electric_combined
| Feature | Corr |
|---|---|
| lag_1h | 0.563 |
| lag_168h | 0.486 |
| lag_24h | 0.464 |
| hour | 0.423 |
| kitchen_temp | 0.397 |
| roll_mean_24h | 0.243 |
| roll_std_24h | 0.219 |
| kitchen_hum | -0.147 |
| month | -0.113 |
| rhum | -0.077 |
| temp (outdoor) | -0.074 |
| wspd | 0.073 |
| is_weekend | 0.063 |
| dayofweek | 0.061 |
| pres | -0.035 |
| is_holiday | -0.019 |

**Key insights:** Strong lag-1h and lag-168h (weekly pattern). `hour` is the strongest calendar feature. Kitchen temperature outperforms outdoor temperature — indoor environment correlates with occupancy-driven electricity use. Outdoor weather has very weak direct correlation (likely mediated through indoor temp and month).

#### home128 mains
| Feature | Corr |
|---|---|
| lag_1h | 0.402 |
| lag_168h | 0.218 |
| roll_mean_24h | 0.194 |
| hour | 0.181 |
| lag_24h | 0.167 |
| is_weekend | 0.129 |
| dayofweek | 0.117 |
| living_temp | 0.113 |
| pres | -0.026 |
| month | 0.025 |
| wspd | 0.024 |
| temp (outdoor) | -0.021 |
| rhum | -0.011 |
| is_holiday | -0.009 |
| living_hum | 0.004 |
| roll_std_24h | 0.003 |

**Key insights:** Weaker autocorrelation than home96 (lower usage, more variable). Weekend effect is the strongest non-lag signal (0.129). Outdoor weather near-zero throughout.

#### home96 fridgefreezer
| Feature | Corr |
|---|---|
| lag_1h | 0.306 |
| kitchen_temp | 0.231 |
| temp (outdoor) | 0.191 |
| hour | 0.112 |
| lag_168h | 0.112 |
| rhum | -0.108 |
| wspd | 0.103 |
| roll_mean_24h | 0.091 |
| month | -0.075 |
| pres | -0.049 |
| roll_std_24h | 0.043 |
| kitchen_hum | -0.040 |
| dayofweek | 0.034 |
| is_weekend | 0.021 |
| lag_24h | 0.020 |

**Key insights:** Kitchen temperature is second-best after lag_1h — physically sensible (warmer ambient = compressor works harder). Outdoor temperature also relevant. lag_24h near zero confirms no strict daily cycle — fridge cycles are thermostat-driven, not time-of-day-driven.

#### home128 fridgefreezer
| Feature | Corr |
|---|---|
| kitchen_temp | 0.099 |
| roll_mean_24h | 0.066 |
| lag_1h | -0.055 |
| temp (outdoor) | 0.051 |
| hour | -0.051 |
| lag_24h | 0.046 |
| roll_std_24h | 0.038 |
| lag_168h | 0.031 |
| all others | <0.02 |

**Key insights:** All correlations very weak — largely noise. Series has 16.6% hourly gaps and low median consumption (41 W). Exclude from main regression evaluation; flag for two-stage treatment if needed.

#### home96 washingmachine
| Feature | Corr |
|---|---|
| lag_1h | 0.274 |
| lag_168h | 0.100 |
| hour | 0.093 |
| lag_24h | 0.052 |
| rhum | -0.034 |
| pres | 0.031 |
| dayofweek | 0.025 |
| is_weekend | 0.021 |
| all others | <0.015 |

**Key insights:** Very weak overall signal. Discrete wash cycles don't align with hourly bins — per-cycle energy is smeared across hours. Two-stage (on/off + level) model better suited than direct regression.

---

## 2. ASHRAE Dataset

### 2.1 Structure

Pre-processed CSVs with engineered features already included. Hourly, no timestamp gaps, no nulls.

| File | Rows | Target range | Meter type |
|---|---|---|---|
| _elec.csv | 5,238 | 0–459 (kWh) | Electricity (meter=0) |
| _chilled.csv | 6,955 | 0–5,572 (kWh) | Chilled water (meter=1) |

---

### 2.2 Data Quality Issues

#### Zero readings
- Elec: 1 zero (negligible)
- Chilled: 29 zeros (0.4%) — likely sensor dropouts, not real zero consumption

**Fix for chilled zeros:** forward-fill.

#### Outliers (chilled only)
Chilled p99 = 4,390 but max = 5,572 (27% spike above p99). Could be genuine demand or meter error.

**Fix:** winsorize at p99.5 or cap at rolling 3×IQR before training.

#### Roll_mean_24 and roll_std_24 — data leakage
`lag_1`, `lag_24`, `lag_168` are correctly computed (shift only, verified: max diff = 0.0). However, `roll_mean_24` and `roll_std_24` are computed **without a prior shift** — they include the current timestep's value in the rolling window. This constitutes look-ahead leakage for forecasting.

**Fix:** recompute as `meter_reading.shift(1).rolling(24).mean()` and `.std()` before any model training. The `target` column (log of meter_reading) has the same issue if it is used to compute rolling features.

---

### 2.3 Correlation Analysis (Pearson vs meter_reading, enhanced data)

Computed on `data/enhanced/ashrae/` — rolling leakage fixed (`roll_mean_24` and `roll_std_24` recomputed with `shift(1)` prior to rolling window).

#### ASHRAE Electricity
| Feature | Corr |
|---|---|
| target (log) | 0.963 |
| lag_1 | 0.939 |
| lag_168 | 0.893 |
| lag_24 | 0.836 |
| roll_mean_24 | 0.353 |
| roll_std_24 | 0.351 |
| hour | 0.350 |
| is_weekend | -0.292 |
| dayofweek | -0.265 |
| cloud_coverage | -0.102 |
| wind_direction | 0.077 |
| month | 0.046 |
| precip_depth_1_hr | -0.044 |
| wind_speed | -0.024 |
| air_temperature | 0.017 |
| day | -0.016 |
| dew_temperature | -0.006 |
| sea_level_pressure | -0.005 |

**Key insights:** Lags dominate — strong persistence and weekly seasonality. Weather irrelevant for electricity (climate-controlled building). Weekend effect strongly negative (office/commercial building). Rolling features dropped from 0.370/0.368 → 0.353/0.351 after leakage fix — the prior inflation was small but present. Log-transform of target is near-linear with raw (corr=0.963), confirming it is appropriate.

#### ASHRAE Chilled Water
| Feature | Corr |
|---|---|
| lag_1 | 0.875 |
| lag_24 | 0.813 |
| roll_mean_24 | 0.686 |
| lag_168 | 0.660 |
| air_temperature | 0.598 |
| dew_temperature | 0.594 |
| roll_std_24 | 0.400 |
| hour | 0.305 |
| sea_level_pressure | -0.227 |
| month | -0.154 |
| cloud_coverage | 0.114 |
| wind_speed | -0.071 |
| wind_direction | -0.068 |
| is_weekend | -0.036 |
| dayofweek | -0.023 |
| day | -0.022 |
| precip_depth_1_hr | 0.009 |

**Key insights:** Outdoor temperature is a strong driver of chilled water (0.598) — physically expected as cooling load rises with ambient heat. roll_mean_24 dropped from 0.699 → 0.686 after leakage fix, confirming the prior value was inflated. lag_168 weaker than for electricity (0.660 vs 0.893). Weekend effect near-zero — HVAC runs continuously regardless of occupancy.

---

## 3. Enhanced Datasets (Applied)

All preprocessing was applied by `scripts/build_enhanced_datasets.py`. Original data in `data/ideal_hierarchy/` and `data/raw/` is untouched.

### IDEAL — `data/enhanced/ideal/`

Four resampling variants, each containing all 102 parquet files in the same folder hierarchy as `ideal_hierarchy/`:

| Variant | Freq | Folder |
|---|---|---|
| variant_5 | 5-min | `data/enhanced/ideal/variant_5/` |
| variant_15 | 15-min | `data/enhanced/ideal/variant_15/` |
| variant_30 | 30-min | `data/enhanced/ideal/variant_30/` |
| variant_60 | 60-min | `data/enhanced/ideal/variant_60/` |

Per-file processing applied in all variants:

| File type | Dedup | Resample | `value_log1p` added |
|---|---|---|---|
| Electricity (mains, electric_combined, cooker, shower) | Yes | variant freq, mean | Yes |
| Appliance power (fridge, washer, kettle, etc.) | Yes | variant freq, mean | Yes |
| Gas | Yes | variant freq, mean | Yes |
| Temperature / humidity / radiator probes | Yes | variant freq, mean | No |
| Weather | Yes | variant freq, mean | No |
| Calendar | Yes | forward-fill to grid | No |

`value_log1p = log1p(value.clip(lower=0))` added as an extra column alongside the raw `value` — both are available for model selection.

**Calendar parquets** (`home96/calendar.parquet`, `home128/calendar.parquet`) were generated in `ideal_hierarchy/` first (source of truth), then resampled into each variant. Scottish public holidays used (2017–2018). Calendar uses `Europe/London` local time for all derivations; stored in UTC. Columns: `ts, home_id, hour, dayofweek (0=Mon), month, is_weekend, is_holiday`.

### ASHRAE — `data/enhanced/ashrae/`

| File | Change |
|---|---|
| `building_energy_data_elec.csv` | `roll_mean_24` and `roll_std_24` recomputed as `shift(1).rolling(24)` |
| `building_energy_data_chilled.csv` | same rolling fix applied |

No other columns were changed. Zeros, outliers, and other issues are deferred.

---

## 4. Remaining Actions (Deferred)

| Issue | Dataset | Priority | Proposed Fix |
|---|---|---|---|
| Gap handling | IDEAL (esp. home96 at 28.6%) | High | Interpolate ≤3 consecutive NaN hours; drop windows >3h from training |
| Near-zero appliance series | IDEAL appliances | High | Skip (kettle/toaster/vacuum) or two-stage on/off + level model |
| Zero readings | ASHRAE chilled (29 zeros, 0.4%) | Medium | Forward-fill |
| Outlier cap | ASHRAE chilled (max 27% above p99) | Medium | Winsorize at p99.5 |
