# EnergyX

A **multi-agent Home Energy Management System** built on the IDEAL residential dataset. An LLM-driven Orchestrator routes natural-language queries to specialised agents that forecast consumption, detect anomalies, explain drivers, and dispatch smart-home control commands. The system operates in two distinct modes: **Online** (live streaming dashboard) and **Offline** (historical batch analysis).

## Key Contributions

1. **Multi-Agent Architecture** — Orchestrator routes queries to KnowledgeAgent (RAG + regulatory tools), AnalysisAgent (forecasting/anomaly/counterfactual), ControlAgent (HA dispatch), and MonitorAgent (offline batch). OnlineRunner handles the hot path without LLM calls.
2. **Real-Time Data Bus** — FastAPI process owns a thread-safe LiveBuffer and OnlineRunner. An independent producer script streams IDEAL ticks at configurable speed and density. Streamlit polls via offset-based HTTP endpoints.
3. **Multi-Layered Explainability** — SHAP, feature importance, STL decomposition, lag correlations, causal attribution, and counterfactual billing, all grounded in IDEAL sensor signals without external weather APIs.
4. **9-Monitor Pipeline** — Anomaly detection, appliance baseline tracking, habit drift, carbon intensity, real-time cost, budget trajectory, forgot-to-turn-off, demand response, and historic store writer — all running on every live tick.
5. **RAG Knowledge Layer** — Hybrid BM25 + ChromaDB retrieval over a corpus of UK energy regulations (ECO4, BUS, MEES, Ofgem price cap), efficiency guides, IDEAL dataset docs, and domain Q&A pairs.

## Quick Start

### Prerequisites

- Python 3.10+
- Local Ollama installation (`ollama pull llama3.2:3b`) **or** an OpenRouter API key
- IDEAL dataset ingested into `data/historic_store/` (see [STARTUP.md](STARTUP.md))

### Installation

```bash
git clone <repo-url> && cd EnergyX
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set LLM_PROVIDER and model settings
```

### Run (Offline Mode)

```bash
streamlit run app.py
```

Select a home (home96 / home128 / home62) and date window in the sidebar. Ask questions in the Chat tab: `"forecast next 7 days"`, `"detect anomalies in September"`, `"what drives my consumption?"`.

### Run (Online Mode)

Three processes, three terminals:

```bash
# Terminal 1 — data bus
uvicorn energyx.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — dashboard (switch to Online in sidebar)
streamlit run app.py

# Terminal 3 — data producer
python scripts/stream_ideal.py --home home96 --every 5 --speed 0.05
```

See [STARTUP.md](STARTUP.md) for full mode-by-mode instructions.

## Architecture

```
                    stream_ideal.py
                         │  POST /ingest/tick
                         ▼
                  ┌─────────────┐        ┌──────────────────┐
                  │  FastAPI    │──────► │  OnlineRunner    │
                  │  data bus   │        │  9 monitors      │
                  └─────────────┘        └──────────────────┘
                         │  GET /live/*
                         ▼
                  ┌─────────────┐
                  │  Streamlit  │
                  │  dashboard  │
                  └──────┬──────┘
                         │ NL query
                         ▼
                  ┌─────────────┐
                  │ Orchestrator│
                  └──┬──────────┘
          ┌──────────┼──────────┬────────────┐
          ▼          ▼          ▼            ▼
    Knowledge    Analysis    Control      Monitor
     Agent        Agent       Agent        Agent
   (RAG/tools) (forecast/  (HA JSON)   (offline
               anomaly/cf)              batch)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Agent Inventory

| Agent | Trigger | Capabilities |
|---|---|---|
| KnowledgeAgent | R | RAG over corpus, tariff lookup, carbon intensity, incentives |
| AnalysisAgent | R | Forecast, anomaly, counterfactual, bill prediction, elasticity, causal attribution |
| ControlAgent | R | NL → HA service-call JSON, schedule flexible loads |
| MonitorAgent | C/S | Offline batch monitoring, periodic summaries |
| OnlineRunner | C (hot path) | 9 streaming monitors, no LLM, direct tick processing |

## Monitor Inventory (Online Hot Path)

| Monitor | Event Type | Description |
|---|---|---|
| AnomalyDetectors | `anomaly.appliance` | Z-score + IQR + STL ensemble |
| ApplianceBaselineWatcher | — | Rolling per-appliance deviation flags |
| HabitDriftTracker | `habit.drift` | Slow behavioural shift detection |
| CarbonTracker | `carbon.realtime` | kWh → gCO₂ via National Grid ESO |
| RealtimeCostMeter | `cost.realtime` | Live £/hour burn rate |
| BudgetTrajectoryTracker | — | End-of-month spend projection |
| HistoricStoreWriter | — | Persists every tick to ParquetBackend |
| ForgotToTurnOffDetector | `forgot.turn_off` | Appliance on at unusual hours |
| DemandResponseListener | — | Grid stress signal detection |

## Analysis Tools

| Tool | What it does |
|---|---|
| `predict_bill` | Project upcoming bill: consumption forecast × tariff schedule |
| `counterfactual_bill_forward` | "If heating +20%, how does my bill change?" |
| `counterfactual_bill_inverse` | "What changes reduce my bill by £40/month?" |
| `evaluate_tariff_switch` | Replay consumption against alternative tariffs |
| `causal_attribution` | Decompose consumption change into weather/occupancy/behaviour Δ |
| `schedule_flexible_loads` | Optimise run times for shiftable appliances, output HA JSON |
| `suggest_budget_corrections` | Corrective actions when budget trajectory breaches cap |
| `project_longhorizon` | Annual/multi-year projection with weather normalisation |
| `compute_elasticity` | Partial dependence (sklearn PDP on RF) per driver feature |
| `update_degradation_baselines` | Refresh per-appliance consumption baselines |

## IDEAL Dataset

| Property | Value |
|---|---|
| Homes | 255 UK households |
| Enhanced homes (appliance-level) | 39/255 |
| Sensor types | `electricity_apparent`, `electricity_real`, `gas_pulse`, `temperature_room`, `temperature_probe`, `humidity`, `light`, `appliance_power` |
| Temperature encoding | Tenths of °C |
| Gas encoding | Cumulative Wh pulses |
| Pre-ingested windows | home96: 2017-09-01→14, home128: 2017-10-01→14, home62: 2017-03-01→14 |

## Project Structure

```
app.py                              # Streamlit UI (Online + Offline modes)
energyx/
  agents/
    analysis/agent.py               # AnalysisAgent — plan/execute via LLM tools
    analysis/new_tools.py           # Energy-specific tool extensions
    control/agent.py                # ControlAgent — NL → HA JSON
    knowledge/agent.py              # KnowledgeAgent — RAG + external tools
    monitor/agent.py                # MonitorAgent — offline batch
  api/
    main.py                         # FastAPI data bus (stream lifecycle + polling)
    live_buffer.py                  # Thread-safe tick + event buffer
  data/
    historic_store.py               # ParquetBackend partitioned by home/date
    events.py                       # BaseEvent dataclass
  monitoring/
    online_runner.py                # Hot-path tick processor (no LLM)
    monitors/                       # 9 streaming monitors
  orchestrator/
    orchestrator.py                 # Query router + mode manager
    router.py                       # classify_query() — keyword/intent routing
  utils/
    appliance_forecast.py           # Per-appliance ARIMA+RF ensemble forecast
    replayer.py                     # Synchronous HistoricStore → OnlineRunner replayer
knowledge/
  corpus/                           # Markdown docs: regulatory, efficiency, IDEAL, Q&A
  index/                            # BM25 pkl + ChromaDB store (pre-built)
  rag/                              # HybridRetriever, reranker, synthesizer
  tools/                            # LangChain @tools: carbon, tariff, weather, incentives
scripts/
  ingest_ideal.py                   # One-time IDEAL CSV → HistoricStore ingestion
  stream_ideal.py                   # Live data producer (posts to FastAPI bus)
```

## Configuration

```bash
# .env
LLM_PROVIDER=ollama                 # or "openrouter"
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434/v1

# OpenRouter alternative (comment out ollama lines above)
# LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=sk-or-...
# OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

ROUTING_TEMPERATURE=0.1
SYNTHESIS_TEMPERATURE=0.7
RANDOM_SEED=42
```

## License

MIT
