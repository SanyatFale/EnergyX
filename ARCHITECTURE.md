# Architecture

TinyTS-Scientist is an agentic system where an LLM autonomously orchestrates time-series forecasting, anomaly detection, and explainability through native tool calling. This document details the system design with emphasis on the agentic loop and explainability pipeline.

## Design Principles

1. **LLMs for reasoning, not computation.** The LLM selects tools, interprets results, and generates explanations. It never sees raw data or performs numerical operations.
2. **All models as tools.** Every forecasting model, anomaly detector, and explainability method is a LangChain `@tool` function with JSON in/out.
3. **Explainability is first-class.** Every analysis mode (forecast, anomaly, counterfactual) has a corresponding explanation layer that provides grounded, metric-backed interpretation.
4. **Human-in-the-loop.** The LLM proposes a plan; the user edits and approves before execution begins.
5. **No data leakage.** Cross-validation uses rolling-origin splits. Ensemble weights come from validation folds only.

## System Overview

```
                        ┌──────────────────────────┐
                        │      User Interface       │
                        │   Streamlit UI / CLI      │
                        └────────────┬─────────────┘
                                     │ query
                                     v
                        ┌──────────────────────────┐
                        │      Plan Phase           │
                        │  DataProfiler             │
                        │  QueryUnderstanding (LLM) │
                        │  -> UserTaskPlan          │
                        └────────────┬─────────────┘
                                     │ plan
                                     v
                        ┌──────────────────────────┐
                        │   Human Approval Gate     │
                        │  Edit models, horizon,    │
                        │  explanation, task type    │
                        └────────────┬─────────────┘
                                     │ approved plan
                                     v
                   ┌─────────────────────────────────────┐
                   │         Agentic Execute Loop         │
                   │                                      │
                   │  LLM receives: system prompt +       │
                   │  workflow description + plan params   │
                   │                                      │
                   │  Loop:                                │
                   │    1. LLM emits tool_call             │
                   │    2. Tool executes, returns JSON     │
                   │    3. Result appended to messages     │
                   │    4. LLM decides next tool or stops  │
                   │                                      │
                   │  Format adapter: if Cerebras outputs  │
                   │  tool call as text, parse and execute │
                   └────────────┬────────────────────────┘
                                │
                                v
                   ┌──────────────────────────┐
                   │   Output & Visualization  │
                   │  Predictions, plots,      │
                   │  grounded explanation,     │
                   │  optional report           │
                   └──────────────────────────┘
```

## Agentic Architecture

### Two-Phase Design

**Phase 1: Plan** (`TinyTSAgent.plan()`)
- Runs `profile_dataset` deterministically (statistics, stationarity tests, seasonality detection, Plotly plots).
- Sends profile summary to the LLM with a structured prompt. The LLM outputs a `UserTaskPlan` (JSON) specifying task type, models, horizon, explanation needs, and counterfactual parameters.
- Fallback heuristics handle LLM parse failures.

**Phase 2: Execute** (`TinyTSAgent.execute()`)
- Builds a task-specific system prompt (forecast, anomaly, counterfactual forward/inverse) with no explicit step enumeration. The prompt describes the task generically: *"Train each model in the list one at a time, then combine."*
- The LLM decides each tool call dynamically in a loop (max 25 iterations).
- Each tool call returns structured JSON. The LLM uses prior results to decide the next action.
- When all tools are done, the LLM provides a final analysis grounded in the returned metrics.

### Format Adapter

Cerebras LLama-3.3-70b occasionally outputs tool calls as plain text instead of structured `tool_calls` format. The execute loop includes a regex-based parser that detects tool call patterns in text output (e.g., `train_forecast_model(model_name='ARIMA', horizon=72)`), extracts the name and arguments, and wraps them as a proper `AIMessage` with `tool_calls`. This is format translation, not error recovery: the LLM's decision is preserved.

### Prompt Engineering

System prompts are intentionally generic to prevent the LLM from "reciting a script" as text:
- `FORECAST_WORKFLOW`: Lists models and tool signature, says "train each one at a time, then combine."
- `FORECAST_EXPLAIN_WORKFLOW`: Same but instructs grounded analysis of all returned metrics.
- `ANOMALY_EXPLAIN_WORKFLOW`: Instructs causal inference from context snapshots.
- `COUNTERFACTUAL_*_WORKFLOW`: Instructs practical interpretation of intervention impacts.

## Tool Architecture

All tools are created by `create_agent_tools()` — a factory that returns closures bound to a dataset session. The session dict accumulates results across tool calls (model results, explanations, predictions).

### Tool Inventory

| Tool | Arguments | Returns |
|------|-----------|---------|
| `profile_dataset` | none | Dataset statistics, frequency, seasonality |
| `train_forecast_model` | model_name, horizon | MAPE, std, best params, predictions |
| `train_and_explain_forecast` | model_name, horizon | Above + FI%, SHAP%, decomposition, lags, correlations |
| `combine_forecasts` | none | Ensemble predictions, weights, strategy |
| `detect_anomalies` | none | Anomaly count, per-method counts, agreement |
| `explain_anomalies` | none | Z-scores, IQR, STL, method agreement, percentiles, context snapshots |
| `generate_report` | none | LLM-synthesized markdown report |
| `counterfactual_forward` | changes_json, horizon | Baseline vs modified forecast, impact |
| `counterfactual_inverse` | target_value, constraints_json | Feature recommendations, optimization result |

### Cross-Validation

All models use rolling-origin cross-validation with 3 folds. Minimum training size: `max(50, 2*horizon)`. Hyperparameter search explores up to 10 random combinations from predefined search spaces.

### Ensemble Strategy

`combine_forecasts` computes inverse-MAPE weights automatically: `weight_i = (1/MAPE_i) / sum(1/MAPE_j)`. No separate ensemble selection step.

## Explainability Pipeline

### Shared Metrics (computed once, cached in session)

| Metric | Computation | Interpretation |
|--------|-------------|----------------|
| Statistical summary | Train/test means, std, trend direction | Dataset-level behavior |
| STL decomposition | Trend, seasonal, residual strength | Decomposed signal structure |
| Lag correlations | ACF at lags 1, 6, 12, 24 | Temporal dependence patterns |
| Feature correlations | Pearson r between features and target | Linear feature-target relationships |

### Per-Model Metrics

| Metric | Computation | Interpretation |
|--------|-------------|----------------|
| Feature importance (%) | Tree `.feature_importances_`, normalized to sum=100% | Which features the model relies on |
| SHAP values (%) | TreeExplainer, normalized to sum=100% | Directional feature contributions |

### Reconciliation

The system produces reconciliation hints combining FI%, SHAP%, and correlations:
- **SHAP > 10% and correlation > 0.2**: Direct driver
- **Correlation < 0.1 but FI > 5%**: Non-linear effect (feature matters through interactions, not simple linear relationship)

### Anomaly Context Snapshots

For each detected anomaly, `explain_anomalies` provides a 5-value window of the target variable and all feature columns centered on the anomaly. This enables the LLM to infer probable causes:
- Target dropped but features stable → measurement error or external event
- Target and feature co-moved → causal relationship

## Counterfactual Analysis

### Forward (What-If)

1. Forecast each feature univariately via ARIMA.
2. Apply user-specified deltas to selected features.
3. Run the best multivariate model (RandomForest/LightGBM) on both baseline and modified feature matrices.
4. Compare baseline vs counterfactual predictions.

### Inverse (Target-Seeking)

1. Select top 3 features by FI/SHAP importance.
2. Bound each feature to its historical min/max.
3. Nelder-Mead optimization minimizes `(prediction - target)^2`.
4. Return recommended feature changes with feasibility assessment.

## Data Flow

```
Dataset CSV
    |
    v
create_agent_tools() -> [tools], session{}
    |
    v
profile_dataset() -> session["profile"]
    |
    v
train_and_explain_forecast("LightGBM", 72)
    |-> session["model_results"]["LightGBM"]
    |-> session["model_explanations"]["LightGBM"]
    |-> session["_shared_explain"] (cached: stat, decomp, lags, corrs)
    |
    v
combine_forecasts()
    |-> session["final_predictions"]
    |-> final_forecast.html, model_comparison.html
    |
    v
LLM final analysis (grounded in all returned JSON metrics)
```

## Configuration

The system supports two LLM providers via `config.py`:

| Provider | Use Case | Model |
|----------|----------|-------|
| Cerebras | Cloud, fast inference | LLama-3.3-70b |
| Ollama | Local, privacy-preserving | LLama 3.2:3b |

Two-temperature pattern:
- **Routing (0.1)**: Deterministic query parsing and tool selection.
- **Synthesis (0.7)**: Creative report generation.

## Extension Points

**Adding a new model:**
1. Add tool function to `tinyts/tools/`.
2. Add entry to `MODEL_TEMPLATES` in `agent_tools.py`.
3. Add to `_UNI_TOOLS` or `_MV_TOOLS` in `_get_model_tools()`.

**Adding a new explainability layer:**
1. Add computation to `tinyts/tools/explainability.py`.
2. Include in `train_and_explain_forecast` shared or per-model metrics.
3. Update workflow prompt to instruct LLM to interpret the new metric.

**Adding a new tool:**
1. Define `@tool` function inside `create_agent_tools()`.
2. Add to returned tools list.
3. Add to `_TOOL_RE` regex in `agent.py` for format adapter.
4. Add to `BASE_PROMPT` tool listing.
