# EnergyX — Challenges Encountered and Potential Future Improvements

## 1. Challenges Encountered

### 1.1 Forecasting Latency — The JSON Serialization Bottleneck

**Challenge:** Early end-to-end forecast runs took **2–6 minutes** per query. Profiling revealed the bottleneck was not the models themselves but the feature engineering step: an O(n) Python loop that appended 18,000+ individual NumPy slices to a list before calling `np.array()` to stack them. Running this inside LangChain's `@tool` interface added a further JSON serialization/deserialization round-trip (converting arrays to lists, stringifying to JSON, then parsing back) for every cross-validation fold.

**Resolution:** Two changes in combination reduced end-to-end forecast time from ~6 minutes to under 5 seconds:
1. Replaced the Python loop with `numpy.lib.stride_tricks.sliding_window_view`, a single C-level stride operation that builds the (N × 720) lag matrix in ~0.6 ms (≈ 200,000× speedup).
2. Introduced a "fast path" (`_cv_tree_numpy`) that bypasses the `@tool` interface for tree-based models entirely, operating directly on NumPy array slices.

### 1.2 Persistent Feature Caching

**Challenge:** Even with vectorized feature creation, the lag matrix was being recomputed on every Streamlit rerun (triggered by any UI interaction). For a 14-day window this added ~44 ms per interaction — small individually but noticeable in rapid UI sessions and wasteful for a fixed dataset.

**Resolution:** Implemented a content-keyed Parquet cache (`data/feature_cache/{md5_hash}_lags720.parquet`). The MD5 hash of the raw `y` array bytes serves as the cache key, so the cache auto-invalidates if the underlying data changes. Subsequent loads read the pre-computed matrix in ~36 ms instead of recomputing it.

### 1.3 LLM Tool-Call Format Compatibility

**Challenge:** Cerebras `llama3.1-8b` does not always emit tool calls in the structured JSON format that LangChain's `bind_tools` expects. It sometimes outputs tool invocations as free prose (`tool_name(arg=val)`) or as a JSON object embedded in the response text. This caused the agentic loop to stall or misfire, silently skipping tool calls.

**Resolution:** Implemented a `_parse_text_tool_call` adapter in the AnalysisAgent that handles three formats:
- Native LangChain structured tool calls (pass-through)
- JSON object with `name`/`parameters` keys (parsed from text)
- `tool_name(arg=val, ...)` syntax (parsed with regex)

A `_strip_thinking` method also removes `<think>...</think>` blocks emitted by reasoning-tuned model variants before parsing.

### 1.4 Data Hierarchy Restructuring

**Challenge:** The raw IDEAL dataset uses a flat file naming convention (`home{id}_{roomtype}{roomid}_sensor{id}_{sensorbox}_{subtype}.csv.gz`) that is difficult to navigate programmatically and requires loading all files to understand the sensor graph of a given home.

**Resolution:** Built `scripts/build_ideal_hierarchy.py` to parse the filename convention and write a `HOME > ROOM > APPLIANCE` Parquet hierarchy on disk. This makes sensor discovery O(1) per level (directory listing) and allows the Streamlit app to build room/appliance selectors without reading any data files.

### 1.5 Training Window vs. Data Quality Trade-off

**Challenge:** Using the full calendar month (~44,640 rows at 1-minute resolution) for model training caused prohibitively slow CV on statistical models (ARIMA re-fits on ~44,000 rows per fold). Reducing to a very short window (e.g., 3 days) risked missing weekly seasonality patterns essential for realistic forecasts.

**Resolution:** Set `_FORECAST_WINDOW_DAYS = 14` — two full weeks captures one complete weekly cycle while reducing training rows by ≈ 4× versus the full month. The Overview and Monitor tabs continue to display full-month data from a separate load path.

---

## 2. Potential Future Improvements

### 2.1 Longer Context and Seasonal Features

The current lag matrix uses `n_lags=720` (12 hours of history). Adding **Fourier terms** (sine/cosine pairs for 24-hour and 7-day periods) as exogenous features would allow tree-based models to explicitly encode daily and weekly seasonality signals without requiring 1,440 or 10,080 raw lags. This is a standard technique in energy demand forecasting and would likely reduce MAPE for all lag-based models.

### 2.2 Expanded Home Coverage

The system currently covers two homes. Expanding to the full enhanced cohort (39 homes) would enable **cross-home transfer learning** and more statistically robust benchmark comparisons. The `build_ideal_hierarchy.py` script already accepts arbitrary home IDs via `--home-ids`.

### 2.3 Probabilistic Forecasting

All current models produce point forecasts. Adding **prediction intervals** (e.g., via quantile regression forests or conformal prediction wrappers) would enable uncertainty-aware scheduling and budget projections. The `evaluate_probabilistic.py` script already scaffolds this evaluation.

### 2.4 Live Smart Meter Integration

The OnlineRunner is architecturally ready for real-time operation but currently runs in batch-simulation mode. Integrating a **MQTT or WebSocket consumer** to ingest live smart meter readings (e.g., via the SMETS2 n3rgy API or a Home Assistant Zigbee bridge) would make the system deployable on real hardware.

### 2.5 Automatic RAG Corpus Refresh

The knowledge corpus must currently be re-indexed manually when regulatory documents change. An **automated ingestion pipeline** (e.g., a weekly job that fetches updated Ofgem PDFs and diffs against the existing index) would keep the RAG system current without manual intervention.

### 2.6 LLM Upgrade Path

`llama3.1-8b` is scheduled for deprecation on Cerebras on **May 27, 2026**. Migrating to a larger model (e.g., `llama3.1-70b` or `gpt-oss-120b`) would improve plan accuracy and tool-call format reliability, reducing reliance on the text-mode adapter. The config is fully parameterised via `.env`, so swapping models requires changing only `CEREBRAS_MODEL`.

### 2.7 Hyperparameter Tuning Per Home

Current `MODEL_TEMPLATES` use fixed hyperparameters for live queries to minimise latency. Running a **one-time offline grid search** per home at system setup (storing best-params in a home-specific config) would improve forecast accuracy without increasing per-query latency.

### 2.8 Feature Cache Lifecycle Management

The Parquet feature cache does not prune stale files. Adding a **cache eviction policy** (e.g., delete entries older than 7 days, or cap total cache size to 100 MB) would prevent unbounded growth in `data/feature_cache/`.

### 2.9 Home Assistant Live Dispatch

The ControlAgent currently uses a `StubDispatcher` that logs HA service-call JSON without sending it. Wiring up a real `HomeAssistantDispatcher` (HTTP POST to the HA webhook endpoint with the bearer token from config) would enable genuine appliance control from the chat interface.
