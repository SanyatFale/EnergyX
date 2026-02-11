# TinyTS-Scientist

A **Local, Human-in-the-Loop Agentic Time-Series System** with Streamlit chat UI, LLM-powered query understanding, 7-method anomaly ensemble, and explainability.

## Architecture

```
Streamlit Chat UI (app.py)  /  CLI (cli.py)
  |
  DataProfiler  ->  QueryUnderstanding  ->  TaskPlanner  ->
  Training  ->  StrategySelector  ->  Execution  ->
  Explainability  ->  ReportGenerator  ->  END
```

Each node has a distinct responsibility with explicit, typed state (Pydantic + TypedDict).

### Key Principles

1. **LLMs for reasoning, NOT computation** -- LLMs parse queries, suggest strategies, and explain results. They never see raw data.
2. **All models as LangChain tools** -- Forecasting and anomaly detection models are tool-wrapped and invoked by nodes.
3. **Two-LLM temperature pattern** -- Low temp (0.1) for deterministic routing/parsing, high temp (0.7) for synthesis/explanation.
4. **No data leakage** -- Validation metrics computed only on held-out folds.
5. **Dual LLM provider** -- Cerebras (cloud, default) or Ollama (local) via `get_llm()`.

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) Ollama for local LLM: `ollama pull llama3.2:3b`
- (Optional) Cerebras API key for cloud LLM

### Installation

```bash
cd tiny-agent
poetry install

cp .env.example .env
# Edit .env -- set LLM_PROVIDER, CEREBRAS_API_KEY, etc.
```

### Streamlit UI

```bash
streamlit run app.py
```

Upload a CSV, select time/target columns, type a query like "forecast next 7 days", and the pipeline runs stage-by-stage with interactive results.

### CLI

```bash
poetry run python main.py run data/example.csv \
  --time date --target sales \
  --query "forecast next 7 days and explain"
```

### Python API

```python
from tinyts.graph import create_graph

graph = create_graph()
state = graph.run(
    dataset_path="data/example.csv",
    time_column="date",
    target_column="sales",
    user_query="forecast next 14 days",
)
print(state["predictions"])
print(state["explainability_result"].llm_explanation)
```

## Pipeline Nodes

| Node | Type | Purpose |
|------|------|---------|
| **DataProfiler** | Deterministic | Frequency, stationarity (ADF/KPSS), seasonality, outliers, Plotly plots |
| **QueryUnderstanding** | LLM (low temp) | Parse user query into structured `UserTaskPlan` |
| **TaskPlanner** | Deterministic | Convert plan to model configs, CV settings, handle multivariate |
| **Training** | Deterministic | Rolling-origin CV, hyperparameter search, per-model metrics |
| **StrategySelector** | LLM-assisted | Ensemble weights + LLM explanation of strategy choice |
| **Execution** | Deterministic | Final predictions, Plotly visualizations |
| **Explainability** | LLM + compute | STL decomposition, SHAP, feature importance, lag correlations, LLM synthesis |
| **ReportGenerator** | LLM | Concise scientific report with all-stage outputs |

## Models

### Forecasting
- **Statistical:** Naive, SeasonalNaive, ARIMA, ETS
- **Tree-based:** RandomForest, LightGBM (univariate + multivariate)
- **Neural:** N-BEATS, TinyTimeMixer

### Anomaly Detection (7-method ensemble)
- Z-score, Modified Z-score (MAD), Rolling statistics, IQR, STL residuals, Isolation Forest, DBSCAN
- Majority voting via `run_anomaly_ensemble`

### Explainability
- Statistical summary (trend, recent averages)
- STL decomposition (trend/seasonal strength)
- Lag contributions (autocorrelation at lags 1, 6, 12, 24)
- Feature importance (tree models)
- SHAP values (optional, requires `shap` package)
- Feature correlations
- LLM-synthesized explanation grounded in metrics

## Project Structure

```
app.py                        # Streamlit chat interface
tinyts/
  config.py                   # Settings + get_llm() helper
  state.py                    # Pydantic models + AgentState
  graph.py                    # LangGraph workflow
  cli.py                      # CLI with --query flag
  nodes/
    data_profiler.py          # Enhanced profiling + Plotly
    query_understanding.py    # LLM query parsing
    task_planner.py           # Plan -> model configs
    training.py               # CV + multivariate support
    strategy_selector.py      # LLM-assisted ensemble
    execution.py              # Predictions + Plotly plots
    explainability.py         # SHAP, STL, correlations, LLM
    report_generator.py       # LLM report with explainability
    _reasoning_agent_old.py   # Deprecated
    _human_approval_old.py    # Deprecated
    _model_planner_old.py     # Deprecated
  tools/
    statistical.py
    tree_based.py             # + multivariate variants
    neural.py
    anomaly.py                # 7-method ensemble
    ensemble.py
    explainability.py         # Computation utilities
```

## Configuration

```bash
# .env
LLM_PROVIDER=cerebras          # or "ollama"
CEREBRAS_API_KEY=your_key
CEREBRAS_MODEL=llama-3.3-70b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
ROUTING_TEMPERATURE=0.1
SYNTHESIS_TEMPERATURE=0.7
MAX_WORKERS=4
DEVICE=cuda
```

## License

MIT
