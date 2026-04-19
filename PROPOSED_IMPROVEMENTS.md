# EnergyX Proposed Improvements

**Prepared:** April 2026  
**Scope:** Production pipeline enhancements, additional studies, and architectural changes.  
Format per entry: **Old Structure → Reason for Change → Proposed Change**

---

## 1. Counterfactual Analysis Methods

### Old Structure
Single inverse optimizer in `tinyts/agent_tools.py` using Nelder-Mead simplex (`scipy.optimize.minimize(method='Nelder-Mead')`). Forward tool forecasts feature deltas per-feature via ARIMA, then applies a fixed delta. No uncertainty estimate on recommended feature values.

### Reason for Change
Nelder-Mead is a local, gradient-free optimizer. On non-convex energy response landscapes it frequently converges to local minima and is sensitive to starting conditions. A single recommended point gives no operational flexibility — in practice an energy manager wants several feasible options. The forward tool's per-feature ARIMA forecasts treat features as independent, which ignores correlated changes (e.g., temperature and humidity move together). There is no causal model behind the "what-if."

### Proposed Change
- **Inverse tool:** Replace Nelder-Mead with Bayesian Optimization (Optuna `TPESampler`). Sample-efficient, handles non-convex objectives, provides uncertainty estimates over the objective surface.
- **DiCE integration:** Generate a diverse set of counterfactuals (3–5) rather than one point. Uses the `dice-ml` library with the existing tree-based model as the ML backend. Operationally, this gives "here are 3 different ways to reach your target."
- **Forward tool — causal model:** Replace independent per-feature ARIMA with a DoWhy/CausalImpact structural model that respects inter-feature correlations (e.g., weather covariates co-move). This makes the forward counterfactual a genuine causal claim, not a correlation extrapolation.
- **SHAP-guided counterfactuals (free):** Use the existing SHAP attribution signs to produce a first-pass counterfactual direction before any optimization. Already computed per session — zero extra inference cost.

---

## 2. Anomaly Detection Ground-Truth Evaluation (LEAD Dataset)

### Old Structure
`evaluate_anomaly.py` uses synthetic injection only: point spikes (±5σ), level shifts (+3σ windows), and contextual anomalies injected into the test set. Precision/Recall/F1 are computed against the synthetic labels.

### Reason for Change
Synthetic injection evaluates what the methods were designed to catch. Reviewers (ICLR W3) correctly noted this is a circular evaluation — real-world anomalies are messier, often unlabeled co-occurring, and don't follow the clean injection model. The LEAD 1.0 dataset (`lead1.0-small.csv`, 200 buildings, 1.75M rows, 37,296 labeled anomalies, full year 2016) provides real human-verified ground truth. Using it makes the evaluation externally credible.

### Proposed Change
- Add `evaluate_lead.py` that loads LEAD, runs the same 7-method ensemble pipeline (`tinyts/tools/anomaly.py` unchanged), and computes P/R/F1/AUROC against the LEAD anomaly labels.
- Evaluate per-building and aggregate across the 200-building corpus.
- Cross-reference synthetic results: report where real-world performance diverges from synthetic (expected: level shift and contextual detection will be harder on real data).
- Extend paper Table 3 (anomaly results) with a LEAD column alongside the synthetic ETT/ASHRAE columns.
- Report anomaly rate distribution across buildings (see Section 9 of this document for the full LEAD ranking analysis).

---

## 3. Data-Agnostic Cleaning and Semantic Column Understanding

### Old Structure
Data loading in `tinyts/agent_tools.py` trims only leading NaN/zero values from the target column. Categorical filtering supports only `==`/`!=` operators. No imputation for mid-series missing values, no unit detection, no duplicate timestamp handling, no schema inference beyond column name matching.

### Reason for Change
Real building datasets have mid-series gaps (meter outages, communication failures), duplicate timestamps (overlapping sensor exports), categorical columns with ambiguous roles (is `meter_type` a filter or a feature to encode?), and inconsistent units (kW vs kWh vs W mixed across files). The current pipeline silently drops or propagates NaN through to model training — behavior that is invisible to the user and produces degraded results without any warning.

### Proposed Change
Extend `tinyts/nodes/data_profiler.py` to emit a `data_quality_flags` list included in `DataProfile`. Surface these flags to the user during the plan approval gate. No silent fixes mid-execution.

| Problem | Detection | Action |
|---|---|---|
| Mid-series missing values | Count NaN per column, gap length distribution | Short gaps (≤2 periods): forward fill. Medium gaps (3–10): linear interpolation. Long gaps (>10% of series): flag to user, ask whether to impute or exclude. |
| Duplicate timestamps | `df.index.duplicated()` check | Aggregate duplicates by mean, warn user of count. |
| Constant / all-zero columns | std = 0 | Drop with warning; never pass to models. |
| Categorical columns | `object` or `bool` dtype detection | Enumerate unique values in `DataProfile`; during plan approval, agent asks: "Column `meter_type` has values [electricity, chilled_water, steam] — should I filter to one, or encode it as a feature?" |
| Unit ambiguity | Pattern-match column names (kw, kwh, w, mw) | Flag scale mismatches; suggest normalisation if target mean > 10× feature mean. |
| Timezone inconsistency | Parse timezone from timestamp strings | Warn and coerce to UTC if mixed. |

The LLM plan approval phase is the right hook — corrections happen before training, not during.

---

## 4. Real-Time Continuous Data Injection and Periodic Forecasting

### Old Structure
Fully batch pipeline. Data is loaded once at agent initialization, a single plan is executed, and results are returned. No mechanism for new data ingestion, periodic retraining, or model updating.

### Reason for Change
Building energy systems generate data continuously. A batch pipeline must be manually re-run to incorporate new readings; models trained weeks ago degrade under concept drift (seasonal regime changes, occupancy pattern shifts, equipment changes). Periodic automated retraining also unlocks advanced online-capable models that learn incrementally.

### Proposed Change
Phased implementation to manage architectural scope:

**Phase A — File-watch triggered retraining (low effort):**  
A scheduler watches a configured data directory. When ≥N new rows are detected (configurable, default 24 for hourly data = 1 day), it triggers the EnergyX pipeline with the updated dataset and stores the new forecast. Implemented via the harness `CronCreate` or a lightweight `watchdog` file monitor.

**Phase B — Concept drift detection (medium effort):**  
After each retraining, compute rolling MAPE on recent windows. If MAPE exceeds a threshold (e.g., 2× baseline CV MAPE), trigger an immediate retraining regardless of data volume. Use CUSUM or ADWIN on the residual stream.

**Phase C — Incremental model updating (larger effort):**  
LightGBM supports `model.fit(init_model=prev_model)` — warm-start retraining on new data only. Add an `incremental=True` flag to the LGBM tool. Combined with drift detection, this enables efficient continuous learning without full retrains.

**Phase D — Streaming ingestion:**  
HTTP push endpoint or MQTT subscriber for direct sensor feeds. Requires a persistent process model (FastAPI or similar), out of scope for the current Streamlit/CLI architecture but compatible as a future service layer.

---

## 5. Advanced Forecasting and Anomaly Detection Models

### Old Structure
**Forecasting:** Naive, SeasonalNaive, ARIMA (auto), ETS, N-BEATS (simplified PyTorch, 50 epochs), RandomForest, LightGBM.  
**Anomaly detection:** 7-method ensemble (Z-Score, MAD, Rolling Stats, IQR, STL, Isolation Forest, DBSCAN).

DBSCAN is included in the univariate ensemble despite achieving F1=0.020 across all benchmark datasets. N-BEATS is a single-stacked implementation without the original paper's double-stack (trend + seasonality) architecture. No foundation model is available.

### Reason for Change
**Forecasting gap:** ETT benchmark shows EnergyX ensemble at MAE=2.59 vs TimeSeriesScientist at 1.81 on ETTh1-96. The gap is architectural — classical/ML ensembles vs. pre-trained foundation model weights. TinyTimeMixer (TTM, IBM Research, 1–3M params) is a lightweight zero-shot/fine-tunable foundation model that runs on CPU and has shown competitive results on ETT without the compute requirements of Moirai or Lag-Llama.

**Anomaly detection gaps from benchmark:**  
- Level shifts: best F1=0.329 (IQR) — no dedicated changepoint detector exists.  
- Contextual anomalies: only STL detects them (F1=0.218) — forecast-residual methods would add a second coverage path.  
- DBSCAN univariate: F1=0.020 — contributes near-zero signal, wastes one vote slot in the majority scheme.

### Proposed Change
**Forecasting — add TinyTimeMixer:**  
Add `tinyts/tools/neural.py:forecast_tinytimemixer()`. Load IBM TTM checkpoint via `huggingface_hub` with a graceful fallback to N-BEATS if offline. Expose as a tool option in the agent. Expected to close 30–40% of the ETT accuracy gap vs. TSS.

**Anomaly detection — targeted additions:**
| Gap | Addition | Library |
|---|---|---|
| Level shifts / changepoints | PELT (Pruned Exact Linear Time) | `ruptures` |
| Contextual anomalies (second path) | Forecast-residual method: train ARIMA, flag \|residual\| > k×MAD | existing `pmdarima` |
| DBSCAN in univariate mode | Gate DBSCAN to multivariate-only mode | `anomaly.py` routing logic |

Bump ensemble to 8-method (or 7 with DBSCAN gated): ZScore, MAD, Rolling, IQR, STL, IsolationForest, PELT, ForecastResidual. Update majority threshold accordingly.

---

## 6. Cross-Validation and Hyperparameter Tuning

### Old Structure
3-fold rolling-origin CV with up to 10 random hyperparameter combinations. Folds evaluated sequentially. Primary metric: MAPE (undefined when actuals=0, returns `inf`, then filtered). Fixed search space sizes regardless of dataset size. No early stopping of bad trials. RF univariate CV alone takes 42–163s (benchmark timing data).

### Reason for Change
Random search with 10 trials is a brute-force approach with no learning between trials. Sequential fold evaluation wastes time — folds are independent and can trivially be parallelised. MAPE produces `inf` on zero-valued series (common in building energy during unoccupied periods) requiring defensive filtering throughout the code. The CV fold count is fixed regardless of dataset size, causing underfitting on small datasets and over-computation on large ones.

### Proposed Change

**Replace random search with Optuna TPE:**  
`optuna.create_study(sampler=TPESampler())` with a `TimeoutCallback(seconds=60)` replaces the fixed 10-trial random loop. TPE uses previous trial results to propose better candidates — typically achieves equivalent quality in 5–6 trials vs. 10 random. The time budget approach (60s cap) is more principled than a fixed trial count.

**Early fold stopping:**  
After fold 1, if trial MAPE > 2× current best mean MAPE, skip folds 2 and 3. Implemented as a simple check in `_cv_univariate()` / `_cv_multivariate()`. Expected to eliminate 30–50% of wasted fold evaluations on bad hyperparameter regions.

**Parallel fold evaluation:**  
Wrap the fold loop in `joblib.Parallel(n_jobs=-1)`. Folds are stateless (each creates its own model instance). On a 4-core machine, 3-fold CV runs in ~⅓ the wall-clock time.

**Switch primary metric from MAPE to MASE:**  
MASE (Mean Absolute Scaled Error) = MAE / (MAE of naïve in-sample forecast). Zero-safe, scale-invariant, interpretable (MASE < 1 means better than naïve). Replace `_compute_metric('mape')` default with `mase` throughout `agent_tools.py`. Remove `inf`-filtering defensive code — no longer needed.

**Adaptive fold count:**  
`n_folds = 2 if n < 500 else (3 if n < 5000 else 5)`. Prevents CV underfitting on short ASHRAE_chilled segments and avoids over-computation on long ETT series.

**Reduce RF search space:**  
`n_estimators: [100, 200]`, `max_depth: [10, 15]`, `n_lags: [12, 24]` (from 3×3×3=27 to 2×2×2=8 combos). RF is the bottleneck at 42–163s; reducing its space while adding early stopping cuts expected CV time to ~25s.

---

## 7. Multi-Sensor Data Stitching and UI

### Old Structure
Single CSV/Parquet/XLSX file input only. Streamlit sidebar in `app.py` has upload + column pickers but no multi-file support. When data spans multiple meters or sensors it must be pre-joined externally. No time-alignment logic, no sensor metadata layer, no hierarchical grouping.

### Reason for Change
Real building deployments produce separate data exports per meter type (electricity, chilled water, steam, gas), per floor, or per sensor. The current workaround — manually joining externally or using the categorical filter — is a friction point that limits the system to pre-processed datasets. Multi-sensor data also enables hierarchical forecasting (building total = sum of floor-level forecasts) and richer multivariate anomaly detection across meter types.

### Proposed Change
Extend the Streamlit UI with a **Dataset Studio** tab (pre-chat, before plan approval):

- **Multi-file upload panel:** Accept N CSV/Parquet files. Display schema preview (columns, dtypes, row count, date range) per file.
- **Role assignment:** User assigns each file a role: Primary (the series to forecast/detect), Weather, Occupancy, Auxiliary. The agent uses role metadata during plan generation.
- **Column namespace deduplication:** Prefix file-of-origin to column names on merge (`file1_temperature`, `file2_temperature`).
- **Time alignment:** Auto-detect sampling frequency per file. Resample all to the lowest common frequency. Flag and visualise gaps before merging.
- **Sensor grouping:** If a file has a `building_id`/`floor`/`meter_type` grouping column, offer a "split by" selector — generates one sub-dataset per group that can be processed in sequence or aggregate.
- **Hierarchical aggregation:** When multiple sensors of the same type exist (e.g., floor-level electricity meters), offer a "create aggregate" option that sums them into a building-level series and links them for reconciliation.

CLI equivalent: `tinyts run file1.csv file2.csv --join-on timestamp --roles primary,weather`.

---

## 8. Additional Study: Conformal Prediction Intervals

### Old Structure
Probabilistic evaluation (`evaluate_probabilistic.py`) covers 7 PI methods. Only ARIMA analytic achieves calibrated coverage (mean PICP=0.933). LGBM quantile fails catastrophically (PICP=0.301). All empirical residual methods systematically undercover (PICP=0.34–0.58).

### Reason for Change
The PI failure is structural: empirical residual methods underestimate out-of-sample variance; quantile LGBM compounds recursive forecast error. These are not fixable by tuning — they require a different coverage guarantee mechanism. Conformal prediction (specifically split conformal / MAPIE) provides distribution-free marginal coverage for any base forecaster. It requires only a held-out calibration set, which is already available in the rolling-origin CV setup.

### Proposed Change
- Implement split conformal intervals in `tinyts/tools/` using `MAPIE` library (`MapieRegressor` or `MapieTimeSeriesRegressor`).
- Wrap LGBM, RF, and N-BEATS forecasters to produce conformal PIs alongside point forecasts.
- Add `evaluate_conformal.py` comparing conformal-corrected LGBM to ARIMA analytic on PICP + PINAW.
- Expected result: LGBM conformal achieves PICP ≥ 0.88 by construction, with sharper intervals than ARIMA analytic at short horizons.
- Add a `return_intervals=True` flag to `forecast_lgbm` and `forecast_rf` tools.

---

## 9. Additional Study: Anomaly Root-Cause Attribution

### Old Structure
Anomaly explanation in `tinyts/tools/anomaly.py` / `explainability.py` provides context snapshots (5-value windows around flagged points) and per-feature z-scores. The counterfactual inverse tool and the anomaly detector operate as independent tools — there is no automated path from "anomaly detected" to "what caused it."

### Reason for Change
Detection without attribution is operationally incomplete. A building operator receiving an anomaly alert needs to know which subsystem to inspect. The counterfactual inverse tool already finds which feature combination produces a given output — applying it to the timestamp of each detected anomaly would produce a root-cause ranking. No new model is needed; it reuses existing infrastructure.

### Proposed Change
Add an `explain_anomaly(timestamp)` tool that:
1. Retrieves the feature values at the anomaly timestamp from the session data.
2. Runs SHAP on the best multivariate model at that point.
3. Calls `counterfactual_inverse` to find the minimal feature change that would have moved the prediction into the normal range.
4. Returns a ranked list: `{feature: "air_temperature", actual: 34.2, normal_range: [18–26], contribution_pct: 42%}`.

This closes the loop between anomaly detection, SHAP attribution, and counterfactual reasoning — a unique capability no current open-source energy analytics tool offers.

---

## 10. Additional Study: Energy-Domain Feature Engineering

### Old Structure
Features are whatever the user provides in the input CSV. The profiler identifies numeric columns and passes them through. No domain-specific features are auto-generated. Calendar features (hour, day, dayofweek) are the top LightGBM predictors in ASHRAE benchmarks but must be present in the raw data — they are not auto-derived from the timestamp.

### Reason for Change
Energy consumption is structurally driven by time-of-use patterns, weather, and occupancy. These signals are always derivable from a timestamp + temperature column — they do not require additional data collection. The ASHRAE benchmark shows calendar features rank 1–5 in LightGBM importance, yet the pipeline currently only uses them if they happen to be pre-engineered in the uploaded file.

### Proposed Change
Add `tinyts/tools/feature_engineering.py` with auto-generated features from the timestamp index and available weather columns:

| Feature | Source | Notes |
|---|---|---|
| hour, dayofweek, month, day | Timestamp | Always generated |
| is_weekend | Timestamp | Binary |
| Heating Degree Days (HDD) | Temperature column (if present) | `max(0, 18 - T)` per timestep |
| Cooling Degree Days (CDD) | Temperature column (if present) | `max(0, T - 18)` per timestep |
| Holiday flag | `holidays` library (country-configurable) | Binary; requires country setting in config |
| Occupancy proxy | Timestamp | Weekday 08:00–18:00 = 1, else 0 |
| Tariff period | Timestamp + tariff config (optional) | On-peak/off-peak binary |

The profiler auto-generates the always-available features and flags which optional ones can be derived if the user confirms the building location/country. No user action required for the timestamp-only features.

---

## Summary Roadmap

| # | Change | Effort | Impact | Priority |
|---|---|---|---|---|
| 2 | LEAD ground-truth anomaly evaluation | Low | High (paper W3) | 1 |
| 6 | CV/tuning: Optuna + parallel folds + MASE | Medium | High (3–5× speedup) | 2 |
| 8 | Conformal prediction intervals | Low–Medium | High (fixes PI gap) | 3 |
| 3 | Data cleaning + semantic column understanding | Medium | High (production) | 4 |
| 9 | Anomaly root-cause attribution | Low | High (novel capability) | 5 |
| 10 | Energy-domain feature engineering | Low | Medium–High | 6 |
| 7 | Multi-sensor UI + dataset stitching | Medium–High | High (deployability) | 7 |
| 5 | TinyTimeMixer + changepoint detection | Medium | Medium–High | 8 |
| 1 | Counterfactual methods (Bayesian opt + DiCE) | Medium | Medium | 9 |
| 4 | Real-time streaming (phased) | Very High | Long-term | 10 |
