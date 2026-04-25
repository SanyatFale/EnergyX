# EnergyX — System Design Documentation

## 1. Overview

EnergyX is a multi-agent Home Energy Management System (HEMS) built on the IDEAL residential sensor dataset. It combines LLM-driven agentic reasoning with deterministic time-series analysis to deliver energy forecasting, anomaly detection, cost monitoring, regulatory knowledge, and appliance control in a unified Streamlit interface.

---

## 2. Data Pipeline

### 2.1 Dataset

The system uses the **IDEAL Household Energy Dataset** (University of Edinburgh), which provides 1-second electricity and 12-second room/appliance sensor readings from 255 UK homes. EnergyX focuses on **homes 96 and 128** (enhanced monitoring homes with complete appliance-level coverage).

### 2.2 Hierarchy Structure

Raw sensor files are restructured into a human-readable Parquet hierarchy via `scripts/build_ideal_hierarchy.py`:

```
data/ideal_hierarchy/
  home96/
    electricity_mains.parquet       ← household-level mains
    gas.parquet
    weather.parquet                 ← Edinburgh weather (minute-aligned)
    livingroom/
      temperature.parquet           ← room sensorbox
      gas_fire/
        gas_fire.parquet            ← appliance-level
    kitchen/
      temperature.parquet
      fridgefreezer/
        fridgefreezer.parquet
    ...
  home128/
    ...
```

Hierarchy depth: **HOME → ROOM → APPLIANCE / PROBE**

### 2.3 Weather Integration

Edinburgh Belhaven weather data (3 yearly CSVs) is upsampled from 1-hour to 1-minute resolution via linear interpolation and clipped to each home's monitoring window. Written to `data/ideal_hierarchy/{home}/weather.parquet`.

### 2.4 Historic Store

Ingested sensor data is also persisted into a Parquet-on-disk `HistoricStore` (partitioned by `home_id` and `date`), backed by `ParquetBackend`. The façade interface supports swapping to DuckDB without changing agent code.

---

## 3. Agent Architecture

```
User Query (Streamlit / CLI)
        │
        ▼
  Orchestrator  ──  classify_query()  ──►  "knowledge" → KnowledgeAgent
        │                                  "control"   → ControlAgent
        │                                  "analysis"  → AnalysisAgent
        │                                  "status"    → status stub
        │
        ▼ (analysis path)
   AnalysisAgent
     Phase 1: plan()   — LLM → UserTaskPlan JSON
     Phase 2: execute() — LLM tool-calling loop (max 25 steps)
```

### 3.1 AnalysisAgent (TinyTS)

The core reasoning agent. Follows a strict two-phase design:

**Plan phase** — `agent.plan(query)`
- Profiles the dataset deterministically (statistics, stationarity, seasonality detection).
- Sends profile + query to the LLM (Cerebras `llama3.1-8b`) via structured prompt.
- Returns a `UserTaskPlan` specifying: `task_type`, `models_included`, `horizon`, `is_multivariate`, `needs_explanation`.

**Execute phase** — `agent.execute(plan)`
- Builds a task-specific system prompt (forecast / anomaly / counterfactual / budget).
- Runs an LLM tool-calling loop (up to 25 iterations).
- A text-mode adapter (`_parse_text_tool_call`) handles Cerebras's free-text tool-call format when native function-calling is unavailable.
- Human-in-the-Loop (HIL) form values (`hil_params`) can override any plan parameter before execution.

**Analysis tools exposed:**
`forecast_univariate`, `forecast_multivariate`, `explain_forecast`, `detect_anomalies`, `explain_anomalies`, `counterfactual_fwd`, `counterfactual_inv`, `get_cost_analysis`, `get_consumption_analysis`, `make_budget`, `weather_impact_analysis`, `suggest_budget_correction`, `evaluate_tariff_switch`, `generate_report`

### 3.2 KnowledgeAgent

Answers UK energy knowledge queries via a hybrid RAG pipeline:

```
query
  │  source_selector.py  — keyword-based domain classification
  ▼
hybrid_retriever.py
  ├── BM25Index  (knowledge/index/bm25_index.pkl)
  └── ChromaDB   (knowledge/index/chroma_store/)
  │  Reciprocal Rank Fusion (k=60) merges both ranked lists
  ▼
reranker.py  — cross-encoder (falls back to RRF score order)
  ▼
guardrails.py  — staleness check (12-month threshold), tariff policy, citation enforcement
  ▼
synthesizer.py  — LLM generates grounded answer (excerpt fallback if LLM unavailable)
```

Also provides: `get_active_tariff()`, `fetch_carbon_intensity()` (National Grid ESO API), `list_applicable_incentives()`.

### 3.3 ControlAgent

Translates natural-language control commands to **Home Assistant service-call JSON**. Permission-gated via `PermissionManager`. Dry-run mode available. Tools: `set_appliance_schedule`, `set_setpoint`, `defer_load`, `power_off`, `read_device_state`.

### 3.4 MonitorAgent (Offline Batch)

LLM-wrapped batch analyzer. Follows the same plan/execute pattern as the AnalysisAgent. Calls the shared monitor core and synthesizes a narrative summary. Can trigger retraining via `trigger_retraining()`.

### 3.5 OnlineRunner (Hot Path)

Deterministic, no-LLM streaming monitor. Processes each sensor tick through a fixed pipeline of 9 monitors:

| Monitor | Event |
|---|---|
| AnomalyDetectors | `anomaly.appliance` (Z-score + IQR + STL ensemble) |
| ApplianceBaselineWatcher | Rolling deviation flags |
| HabitDriftTracker | `habit.drift` (slow behavioural shift) |
| CarbonTracker | `carbon.realtime` (kWh → gCO₂ via National Grid ESO) |
| RealtimeCostMeter | `cost.realtime` (live £/hour burn rate) |
| BudgetTrajectoryTracker | End-of-month spend projection |
| HistoricStoreWriter | Persists ticks to ParquetBackend |
| ForgotToTurnOffDetector | `forgot.turn_off` (appliance on at unusual hours) |
| DemandResponseListener | Grid stress signal detection |

---

## 4. TinyTS Forecasting Engine

### 4.1 Model Suite

| Type | Models |
|---|---|
| Statistical | Naive, SeasonalNaive, ARIMA (`pmdarima` auto_arima, `max_p=2`, `max_q=2`) |
| Gradient Boosting | LightGBM (univariate lag-based, multivariate) |
| Ensemble Trees | RandomForest (univariate lag-based, multivariate) |
| Neural | N-BEATS (simple MLP variant, 15 epochs) |
| Exponential Smoothing | ETS (`statsmodels`) |

### 4.2 Feature Engineering & Caching

Lag features (n_lags=720, i.e. **12 hours** at 1-minute resolution) are built once per session using `numpy.lib.stride_tricks.sliding_window_view` — a vectorized C-level operation. The resulting (N × 720) matrix is persisted to `data/feature_cache/{md5_hash}_lags720.parquet` and reloaded on subsequent runs (~36 ms read vs. ~44 ms compute for a 14-day window). N-BEATS uses n_lags=1440 (24 hours), which is handled separately by the neural training path.

### 4.3 Cross-Validation Strategy

Expanding-window rolling-origin CV (3 folds). Tree-based models use a "fast path" (`_cv_tree_numpy`) that slices the pre-computed lag matrix directly — no JSON serialization. Statistical models go through `_cv_univariate` via the LangChain `@tool` interface.

### 4.4 Ensemble

Final predictions combine all models via inverse-SMAPE weighting. Models with lower CV error receive proportionally higher weight.

### 4.5 Training Window

Capped at **14 days** (≈ 20,160 rows at 1-minute resolution) for agent queries. Full monthly data is still loaded for Overview and Monitor tabs.

---

## 5. LLM Configuration

- **Provider:** Cerebras (cloud, OpenAI-compatible endpoint)
- **Model:** `llama3.1-8b`
- **Config location:** `.env` (`LLM_PROVIDER`, `CEREBRAS_API_KEY`, `CEREBRAS_MODEL`, `CEREBRAS_BASE_URL`)
- **Routing temperature:** 0.1 (low, for deterministic planning)
- **Fallback:** Text-mode tool-call adapter handles models that emit tool calls as prose rather than structured JSON

---

## 6. Scheduler

`EnergyXScheduler` wraps APScheduler with five recurring jobs:
- **Monday 03:00 UTC** — Elasticity recompute per home
- **Daily 02:00 UTC** — Degradation baseline update
- **Sunday 06:00 UTC** — Weekly report
- **1st of month 06:00 UTC** — Monthly report
- **Monday 04:00 UTC** — Continuous training (permission-gated)

MonitorAgent can also trigger immediate retraining via `on_retraining_requested()`.

---

## 7. UI

**Streamlit** (`app.py`) provides four tabs:
- **Overview** — Monthly consumption charts, top appliances/rooms by variance
- **Monitor** — Live monitor outputs (anomalies, cost, budget, carbon)
- **Forecast** — Per-appliance/room forecast via AnalysisAgent (HIL form for horizon & models)
- **Chat** — Free-text query → Orchestrator → Agent → response
