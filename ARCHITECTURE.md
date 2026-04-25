# EnergyX — Architecture

EnergyX is a multi-agent Home Energy Management System. An LLM-driven Orchestrator classifies incoming queries and routes them to specialised agents. A separate hot-path monitoring layer runs without any LLM calls, processing every live tick through nine streaming monitors. The system operates in two distinct modes with a clean decoupling between the data producer, the data bus, and the consumer UI.

---

## Design Principles

1. **LLMs for reasoning, not computation.** The LLM selects tools, interprets results, and generates explanations. It never sees raw data or performs numerical operations.
2. **All models as tools.** Every forecasting model, anomaly detector, and explainability method is a LangChain `@tool` with JSON in/out.
3. **Hot path has no LLM.** The OnlineRunner is pure Python — deterministic, cheap, predictable. LLM analysis runs only on-demand from the Chat tab.
4. **Producer–bus–consumer decoupling.** The data source (stream_ideal.py), the processing bus (FastAPI + OnlineRunner), and the UI (Streamlit) are independent processes communicating only via HTTP. No shared memory, no imports between them.
5. **Mode-aware behaviour.** Online mode = live streaming dashboard. Offline mode = historical batch analysis. Monitors declare `applicable_modes`; the Orchestrator enforces mode context.
6. **No external weather APIs.** IDEAL dataset has `temperature_room` and `temperature_probe` sensors. All meteo context comes from HistoricStore.

---

## System Overview

```
                    stream_ideal.py  (external producer)
                         │
                         │  POST /ingest/tick  (HTTP)
                         ▼
              ┌──────────────────────────┐
              │  FastAPI data bus         │
              │  energyx/api/main.py      │
              │                          │
              │  owns:                   │
              │  - LiveBuffer (singleton) │
              │  - OnlineRunner          │
              └────────────┬─────────────┘
                           │  GET /live/ticks?since=N
                           │  GET /live/events?since=N
                           │  GET /live/status
                           ▼
              ┌──────────────────────────┐
              │  Streamlit dashboard      │
              │  app.py                  │
              │                          │
              │  Online mode:            │
              │  - st.fragment polling   │
              │  - live KPIs + charts    │
              │  - event feed            │
              │                          │
              │  Offline mode:           │
              │  - HistoricStore reads   │
              │  - 4-tab batch analysis  │
              └────────────┬─────────────┘
                           │  NL query
                           ▼
              ┌──────────────────────────┐
              │  Orchestrator             │
              │  orchestrator.py         │
              │                          │
              │  - classify_query()      │
              │  - mode manager          │
              │  - permission gate       │
              │  - event escalation      │
              └─────┬────────┬───────────┘
                    │        │
         ┌──────────┘        └──────────────┐
         ▼                                  ▼
  ┌─────────────┐                   ┌──────────────┐
  │ Knowledge   │                   │  Analysis    │
  │ Agent       │                   │  Agent       │
  │             │                   │              │
  │ RAG retrieval                   │ plan()       │
  │ Tariff lookup                   │ execute()    │
  │ Carbon API  │                   │ LLM tool loop│
  │ Incentives  │                   │              │
  └─────────────┘                   └──────────────┘
         ▼                                  ▼
  ┌─────────────┐                   ┌──────────────┐
  │  Control    │                   │  Monitor     │
  │  Agent      │                   │  Agent       │
  │             │                   │              │
  │ NL → HA JSON│                   │ Offline batch│
  │ HA dispatch │                   │ summaries    │
  └─────────────┘                   └──────────────┘
```

---

## Mode Architecture

### Online Mode

```
stream_ideal.py
   POST /stream/start  {"home_id": "home96"}
   POST /ingest/tick   (loop, N ticks)
   POST /stream/complete
   POST /stream/persist  ← saves buffer to HistoricStore
   POST /stream/reset
```

The FastAPI process owns one `LiveBuffer` (thread-safe list + Lock) and one `OnlineRunner` (9 monitors). Every `POST /ingest/tick` call:
1. Calls `_buffer.add_tick(tick)`
2. Calls `_runner.process_tick(tick)` → returns `List[BaseEvent]`
3. Calls `_buffer.add_events(events)`

Streamlit polls `GET /live/ticks?since=N` and `GET /live/events?since=N` using integer offset slicing so it never re-receives old data.

`st.fragment(run_every=timedelta(seconds=N))` fragments auto-refresh the live overview and monitor tabs independently without a full page rerun.

When the producer calls `/stream/complete`, `is_complete=True` appears in the next `/live/status` poll. The dashboard renders a session-complete banner offering three options: persist to HistoricStore (via `/stream/persist`), reset and clear, or continue watching.

### Offline Mode

HistoricStore is a Parquet-backed store partitioned as:
```
data/historic_store/
  ticks/home_id=home96/date=2017-09-01/part.parquet
  events/home_id=home96/date=2017-09-01/part.parquet
  forecasts/home_id=home96/run_ts=.../part.parquet
```

The user selects home + date window in the sidebar. Queries route through the Orchestrator to the appropriate agent. The MonitorAgent runs the full 9-monitor pipeline synchronously on each batch append.

---

## Agent Architecture

### Orchestrator

`classify_query(query)` maps natural language to a destination:

| Pattern | Destination |
|---|---|
| `forecast`, `predict`, `projection` | AnalysisAgent |
| `anomaly`, `unusual`, `spike` | AnalysisAgent |
| `bill`, `tariff`, `cost`, `elasticity` | AnalysisAgent |
| `turn on/off`, `schedule`, `set temperature` | ControlAgent |
| `regulation`, `ECO4`, `grant`, `scheme` | KnowledgeAgent |
| `carbon`, `intensity` | KnowledgeAgent |
| `what if`, `counterfactual` | AnalysisAgent |

### AnalysisAgent — Two-Phase Design

**Phase 1: Plan** (`agent.plan(query)`)
- Profiles dataset (statistics, stationarity, seasonality) deterministically.
- Sends profile + query to LLM with structured prompt → `UserTaskPlan` JSON specifying task type, models, horizon, explanation flags.

**Phase 2: Execute** (`agent.execute(plan)`)
- Builds task-specific system prompt (forecast / anomaly / counterfactual-forward / counterfactual-inverse).
- LLM tool-calling loop (max 25 iterations): emit tool call → execute → append result → decide next action.
- Format adapter handles Cerebras/Ollama text-format tool call outputs.

### KnowledgeAgent

Owns:
- `answer(query)` — RAG over hybrid BM25+ChromaDB corpus, synthesizes grounded answer
- `get_active_tariff()` — reads tariff config, returns rates
- `fetch_carbon_intensity()` — National Grid ESO API with 200 gCO₂/kWh fallback
- `list_applicable_incentives(jurisdiction, epc_band)` — ECO4, BUS, GBIS, WHD, MEES

RAG pipeline: `source_selector` → `hybrid_retriever` (BM25 + vector) → `reranker` → `synthesizer` (LLM with hallucination guardrails).

### ControlAgent

`handle(command)` translates natural language control commands to Home Assistant service-call JSON. Uses LLM to extract: domain, service, entity_id, service_data. Returns structured JSON that a HA webhook can consume directly.

---

## OnlineRunner — Hot Path

9 monitors, no LLM, deterministic:

```
process_tick(tick_dict)
   │
   ├── AnomalyDetectors.evaluate(tick)          → anomaly.appliance events
   ├── ApplianceBaselineWatcher.evaluate(tick)  → deviation flags
   ├── HabitDriftTracker.evaluate(tick)         → habit.drift events
   ├── CarbonTracker.evaluate(tick)             → carbon.realtime events
   ├── RealtimeCostMeter.evaluate(tick)         → cost.realtime events
   ├── BudgetTrajectoryTracker.evaluate(tick)   → budget projection
   ├── HistoricStoreWriter.evaluate(tick)       → writes to ParquetBackend
   ├── ForgotToTurnOffDetector.evaluate(tick)   → forgot.turn_off events
   └── DemandResponseListener.evaluate(tick)   → DR signal detection
```

Each monitor implements `BaseMonitor.evaluate(tick) → List[BaseEvent]`. Monitors declare `applicable_modes = ["online"]` or `["online", "offline"]`. The OnlineRunner filters at startup; the MonitorAgent uses the offline-eligible set.

---

## Data Layer

### HistoricStore

```python
store = HistoricStore(backend=ParquetBackend(store_path))
df = store.read_ticks("home96", start=datetime(...), end=datetime(...))
store.write_ticks("home96", df)
```

`read_ticks` returns a DataFrame with columns: `home_id`, `ts`, `sensor_type`, `sensor_id`, `value`, `unit`.

IDEAL sensor types present in home96:
- `electricity_apparent` — apparent power in watts
- `electricity_real` — real power in watts
- `gas_pulse` — cumulative Wh pulses
- `temperature_room` — room temperature (tenths °C in raw; converted to °C)
- `temperature_probe` — probe temperature (tenths °C)
- `humidity` — relative humidity %
- `appliance_power` — per-appliance power (39 enhanced homes only)

### LiveBuffer

Thread-safe (`threading.Lock`) in-memory store for the FastAPI process:

```
start(home_id)  →  add_tick() / add_events()  →  complete()  →  reset()
```

Consumers poll via `get_ticks_since(offset)` and `get_events_since(offset)` — O(1) list slice, no copy overhead until polled.

---

## Tool Architecture

All AnalysisAgent tools are created by `create_agent_tools()` (core TinyTS tools) and `create_analysis_extensions()` (energy-specific extensions). Both are factories returning closures bound to a dataset session dict.

### Core TinyTS Tools

| Tool | Returns |
|---|---|
| `profile_dataset` | Statistics, frequency, seasonality, stationarity |
| `train_forecast_model` | MAPE, std, best params, predictions |
| `train_and_explain_forecast` | Above + FI%, SHAP%, STL, lags, correlations |
| `combine_forecasts` | Ensemble predictions (inverse-MAPE weights) |
| `detect_anomalies` | Count, per-method counts, agreement rate |
| `explain_anomalies` | Z-scores, IQR, context snapshots, method agreement |
| `counterfactual_forward` | Baseline vs modified forecast, impact |
| `counterfactual_inverse` | Feature recommendations via Nelder-Mead |
| `generate_report` | LLM-synthesized markdown report |

### Energy-Specific Extensions (`new_tools.py`)

| Tool | Returns |
|---|---|
| `predict_bill` | Projected bill: consumption × tariff schedule |
| `counterfactual_bill_forward` | Bill impact of consumption change |
| `counterfactual_bill_inverse` | Feature targets to hit a bill reduction |
| `evaluate_tariff_switch` | Savings from switching tariff |
| `causal_attribution` | Consumption Δ decomposed into drivers |
| `schedule_flexible_loads` | HA service-call JSON for shiftable appliances |
| `suggest_budget_corrections` | Corrective actions for budget breach |
| `project_longhorizon` | Annual/multi-year projection |
| `compute_elasticity` | sklearn PDP on RF: ∂consumption/∂feature curves |
| `update_degradation_baselines` | Refresh per-appliance baseline consumption |

### Appliance-Level Forecasting (`utils/appliance_forecast.py`)

`forecast_appliance(appl_df, sensor_id, horizon, cv_folds)`:
- Resample to 1-hour mean
- Train ARIMA(1,1,1) + RandomForest with lag features
- Rolling-origin CV (3 folds) for MAPE estimation
- Inverse-MAPE ensemble weighting
- Returns: `{status, predictions, ensemble_mape, best_model, n_train}`

---

## Knowledge Layer

### Corpus

```
knowledge/corpus/
  regulatory/          # ECO4, BUS, MEES, Ofgem price cap, WHD, VAT, SEG, smart meters
  efficiency_guide/    # Insulation, heat pump, EV charging, solar PV, appliances
  ideal_docs/          # Pullinger 2021 dataset paper (chunked)
  domain_qa/           # 20 expert Q&A pairs on UK energy management
```

### RAG Pipeline

```
query
  │  source_selector.py  — domain classification
  ▼
hybrid_retriever.py
  ├── BM25Index (knowledge/index/bm25_index.pkl)
  └── ChromaDB (knowledge/index/chroma_store/)
  │  k=8 candidates, reciprocal rank fusion
  ▼
reranker.py  — cross-encoder or keyword re-score
  ▼
synthesizer.py  — LLM generates grounded answer
  │  guardrails.py — blocks hallucinations, requires citation
  ▼
answer string
```

---

## Scheduling

`energyx/scheduler/jobs.py` defines:
- **Weekly**: `recompute_elasticity` — re-trains elasticity model per home, caches result
- **Nightly**: `update_degradation_baselines` — refreshes per-appliance consumption baselines

**Status**: Scheduler job definitions exist but APScheduler wiring and startup integration are not yet implemented (see WORK_TODO.md).

---

## Cross-Validation

All models use rolling-origin CV with 3 folds. Minimum training size: `max(20, 2*horizon)`. Ensemble weights are inverse-MAPE computed on validation folds only — no data leakage.

---

## Configuration

| Provider | Model | Use case |
|---|---|---|
| Ollama (default) | llama3.2:3b | Local, privacy-preserving, free |
| OpenRouter | meta-llama/llama-3.3-70b-instruct | Cloud, better reasoning quality |

Two-temperature pattern:
- **Routing (0.1)**: Deterministic query classification and tool selection
- **Synthesis (0.7)**: Report generation and explanation narrative

---

## Extension Points

**Add a monitor:**
1. Subclass `BaseMonitor`, implement `evaluate(tick) → List[BaseEvent]`, set `applicable_modes`.
2. Add to `_build_online_monitors()` in `online_runner.py`.

**Add an analysis tool:**
1. Define `@tool` function inside `create_analysis_extensions()` in `new_tools.py`.
2. Add to returned tools list — no other wiring needed.

**Add a corpus document:**
1. Write markdown to `knowledge/corpus/<category>/`.
2. Re-run `python knowledge/ingestion/build_index.py` to rebuild BM25 + ChromaDB.

**Add a new home from IDEAL:**
1. Run `python scripts/ingest_ideal.py --home homeXXX --start YYYY-MM-DD --end YYYY-MM-DD`.
2. Add default window to `_WINDOWS` dict in `stream_ideal.py`.
