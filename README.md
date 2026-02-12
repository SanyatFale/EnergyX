# TinyTS-Scientist

An **agentic time-series forecasting and anomaly detection system** with multi-layered explainability. The system uses LLM-driven tool calling to autonomously profile data, train models, detect anomalies, and generate grounded explanations combining statistical metrics, model attributions, feature analysis, counterfactual reasoning, and contextual snapshots.

Submitted to the **ICLR 2026 Workshop on Time Series in the Age of Large Models (TSALM)**.

## Key Contributions

1. **Agentic Architecture**: LLM autonomously orchestrates a pipeline of forecasting/anomaly tools via native tool calling, with human-in-the-loop plan approval.
2. **Multi-Layered Explainability**: Combines STL decomposition, lag correlations, SHAP values, feature importance, feature correlations, and contextual anomaly snapshots into grounded, interpretable output.
3. **Counterfactual Analysis**: Forward (what-if scenarios under modified features) and inverse (Nelder-Mead optimization to find feature changes needed to reach a target).
4. **7-Method Anomaly Ensemble**: Z-score, MAD, Rolling Statistics, IQR, STL Residuals, Isolation Forest, and DBSCAN with majority voting.

## Quick Start

### Prerequisites

- Python 3.10+
- A Cerebras API key (free tier) **or** a local Ollama installation

### Installation

```bash
# Clone and enter the repository
git clone <repo-url> && cd tiny-agent

# Create virtual environment and install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set LLM_PROVIDER and CEREBRAS_API_KEY (or OLLAMA settings)
```

### Run the Streamlit UI

```bash
streamlit run app.py
```

1. The default dataset (building energy) loads automatically.
2. Select time/target/feature columns in the sidebar.
3. Type a query: `"forecast next 3 days"`, `"detect anomalies and explain"`, or `"what if temperature drops by 5 degrees?"`.
4. Review and approve the plan, then the agent executes autonomously.

### Run via CLI

```bash
python -m tinyts.cli run "data/raw/building_energy_data (copy 1)_elec.csv" \
  --time timestamp --target meter_reading \
  --query "forecast next 3 days and explain"
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

```
User Query
    |
    v
[Plan Phase]  DataProfiler -> QueryUnderstanding (LLM) -> UserTaskPlan
    |
    v
[Human Approval]  Edit models, horizon, explanation toggles
    |
    v
[Execute Phase]  LLM tool-calling loop:
    |   train_forecast_model() / train_and_explain_forecast()
    |   combine_forecasts() / detect_anomalies() / explain_anomalies()
    |   counterfactual_forward() / counterfactual_inverse()
    |   generate_report()
    v
[Output]  Predictions + Plots + Grounded Explanation
```

## Models

| Family | Models | Multivariate |
|--------|--------|:------------:|
| Statistical | Naive, SeasonalNaive, ARIMA (auto), ETS | No |
| Tree-based | RandomForest, LightGBM | Yes |
| Neural | N-BEATS | No |

## Explainability Layers

| Layer | Source | Scope |
|-------|--------|-------|
| Statistical summary | Train/test split stats | Dataset |
| STL decomposition | Trend, seasonal, residual | Dataset |
| Lag correlations | ACF at lags 1, 6, 12, 24 | Dataset |
| Feature importance | Tree model `.feature_importances_` | Per model |
| SHAP values | TreeExplainer | Per model |
| Feature correlations | Pearson with target | Dataset |
| Anomaly context | 5-value window of target + features | Per anomaly |
| Counterfactual | Forward (what-if) and inverse (target-seeking) | Scenario |

## Project Structure

```
app.py                          # Streamlit chat UI
tinyts/
  agent.py                      # Central LLM agent (plan + execute)
  agent_tools.py                # Tool factory with closures (~1200 lines)
  config.py                     # Settings + get_llm() (Cerebras / Ollama)
  state.py                      # Pydantic models (UserTaskPlan, etc.)
  cli.py                        # CLI interface
  nodes/
    data_profiler.py            # Frequency, stationarity, seasonality
    query_understanding.py      # LLM query -> structured plan
  tools/
    statistical.py              # Naive, SeasonalNaive, ARIMA, ETS
    tree_based.py               # RandomForest, LightGBM (uni + multivariate)
    neural.py                   # N-BEATS
    anomaly.py                  # 7-method ensemble
    ensemble.py                 # Weighted combination
    explainability.py           # SHAP, FI, STL, lags, correlations
data/
  raw/                          # Building energy datasets, UCI household power
  processed/                    # Preprocessed datasets
```

## Configuration

```bash
# .env
LLM_PROVIDER=cerebras              # or "ollama"
CEREBRAS_API_KEY=your_key_here
CEREBRAS_MODEL=llama-3.3-70b
CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
ROUTING_TEMPERATURE=0.1
SYNTHESIS_TEMPERATURE=0.7
DEVICE=cpu                         # or "cuda"
```

## Reproducibility

```bash
# Exact environment
pip install -r requirements.txt    # Pinned versions

# Deterministic seeds
RANDOM_SEED=42                     # In .env, used by all stochastic models
```

## License

MIT
