# EnergyX — Assumptions and Limitations

## 1. Data Assumptions

### 1.1 Dataset Scope
- The system is developed and evaluated on **two homes only** (homes 96 and 128) from the IDEAL dataset. These were selected as they belong to the enhanced-monitoring cohort with appliance-level electricity sub-metering.
- All sensor data originates from **Edinburgh, Scotland**, during **2017**. Findings may not generalise to homes in other climates, geographies, or time periods.

### 1.2 Data Quality
- **Sensor gaps and dropouts** are assumed to be Missing At Random (MAR). The pipeline does not perform imputation on training data — gaps propagate as `NaN` and are silently dropped during feature creation. This may bias model training if gaps are systematic (e.g., correlated with high-consumption events).
- The IDEAL dataset uses **1-second electricity** and **12-second room/appliance** sampling. All data is downsampled to **1-minute averages** for warm storage and model training. Sub-minute dynamics (e.g., appliance start/stop transients) are lost.
- Weather data is resampled from **hourly** to **1-minute** resolution via linear interpolation. Intra-hour weather variability is therefore smoothed out.

### 1.3 Target Variable
- All forecasting targets are treated as **univariate or lightly multivariate** (lag features + concurrent weather/room sensors). Complex interactions between appliances or occupancy-driven demand patterns are not explicitly modelled.
- Electricity readings are in **Watts (apparent power)** for mains and appliances. The distinction between apparent and real power is preserved in sensor types but not in the unified forecast target.

---

## 2. Model Assumptions

### 2.1 Stationarity
- ARIMA assumes weak stationarity. `pmdarima.auto_arima` performs ADF/KPSS-based differencing selection, but the bounded search (`max_p=2`, `max_q=2`) may not capture long-memory processes present in some homes.
- Tree-based models (RF, LightGBM) make no stationarity assumptions, but rely on the lag matrix accurately representing temporal dependencies. With `n_lags=720` (12 hours of history at 1-minute resolution), the model sees half a daily cycle as direct features; the full 24-hour and 7-day periodicities are not directly captured.

### 2.2 Seasonality
- The system does not apply explicit seasonal decomposition before model fitting. SeasonalNaive uses a period of 60 (1-hour cycle at 1-minute resolution), which is a simplification — energy demand has strong daily (1440-step) and weekly periodicities that are not fully exploited.
- N-BEATS uses `n_lags=1440` (24 hours of history at 1-minute resolution) with a simple MLP architecture. It is not the full N-BEATS paper architecture with trend/seasonality stacks.

### 2.3 Cross-Validation
- Rolling-origin CV uses **3 folds** with a `min_train` of `max(50, horizon * 2)` rows. For short series (e.g., individual appliances with sparse activation), this may yield only 1 fold or degenerate splits.
- Hyperparameter search is intentionally minimal (fixed hyperparameters for live queries) to reduce latency. This means the deployed models are not fully optimised — the `MODEL_TEMPLATES` defaults are reasonable but not tuned per home.

### 2.4 Ensemble
- The inverse-SMAPE weighting ensemble assumes that lower CV error on historical data predicts lower error on the forecast horizon. This is a standard but imperfect assumption, particularly for non-stationary or drifting series.

---

## 3. RAG and Knowledge Assumptions

### 3.1 Corpus
- The RAG corpus (regulatory documents, efficiency guides, tariff data) must be **indexed separately** by running `knowledge/ingestion/build_index.py`. Until indexed, all RAG queries fall back to a stub response directing users to `ofgem.gov.uk`.
- Regulatory content is assumed to be **UK-specific** (England/Wales/Scotland/NI). The jurisdiction filter is keyword-based and may mis-route queries for devolved policy areas (e.g., Scottish government schemes).

### 3.2 Tariff Data
- Tariff rates (£/kWh, standing charge) are read from a **static config file**, not fetched live. They reflect Ofgem price cap values at the time of configuration. The system does not automatically update when the price cap changes quarterly.
- Carbon intensity is fetched live from the **National Grid ESO API** with a **200 gCO₂/kWh fallback** when the API is unavailable. This fallback is a UK grid average and will be inaccurate at times of high renewable generation.

---

## 4. LLM and Inference Assumptions

### 4.1 Model
- The LLM (`llama3.1-8b` via Cerebras) is a relatively small open-weight model. It may produce incorrect tool-call formats, hallucinate parameter values, or misclassify query intent for ambiguous queries.
- A **text-mode adapter** (`_parse_text_tool_call`) handles cases where the model emits tool calls as free text rather than structured JSON. This parser covers two formats (JSON object and `tool_name(arg=val)` syntax) but may fail on novel formats.
- **Cerebras `llama3.1-8b` is scheduled for deprecation on May 27, 2026.** The replacement model (`gpt-oss-120b`) may require prompt or adapter changes.

### 4.2 Query Routing
- `classify_query()` applies heuristic keyword patterns first (high-confidence control and analysis keywords). The LLM is only invoked for ambiguous queries. Heuristics may mis-route novel phrasings not covered by the keyword patterns.

### 4.3 Human-in-the-Loop
- HIL form parameters (horizon, models, level) **override** LLM plan values. If the user submits a HIL form with incorrect values, those values will be used without further validation.

---

## 5. Deployment Limitations

### 5.1 Scale
- The current system is designed for **offline / research use** on a local machine. It is not hardened for multi-user or production deployment.
- The Streamlit app loads an entire calendar month of data per home into memory on startup. For homes with many sensors this can require several hundred MB of RAM.

### 5.2 Real-Time Operation
- The **OnlineRunner** is architecturally designed for real-time tick processing but is integrated in the Streamlit app in a batch-simulation mode — it processes historical data as if it were arriving in real time. True WebSocket or MQTT integration with live smart meters is not implemented.
- Home Assistant integration (ControlAgent) emits correctly formatted service-call JSON but uses a **StubDispatcher** by default. No live HA instance connection is wired up out of the box.

### 5.3 Scheduling
- The `EnergyXScheduler` requires **APScheduler** to be installed. If not present, all scheduled jobs are silently disabled (a warning is logged).

### 5.4 Feature Cache Validity
- The Parquet feature cache is keyed by an **MD5 hash of the raw `y` array bytes**. If the training window shifts (e.g., due to a data re-ingestion) the hash changes and a fresh cache is written. However, stale cache files from old runs are **not automatically deleted** and will accumulate in `data/feature_cache/`.
