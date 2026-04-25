# EnergyX — Work To Be Done

Remaining implementation work from the update plan. Entries are grouped by subsystem and ordered roughly by dependency. Each entry notes where the gap is and what specifically needs building.

---

## 1. Scheduler Wiring — APScheduler Integration

**Status:** Job definitions exist in `energyx/scheduler/jobs.py` but are never started.

**What's missing:**
- Install APScheduler: `pip install apscheduler` (add to `requirements.txt`)
- Wire `EnergyXScheduler.start()` into FastAPI lifespan (`@app.on_event("startup")` or `asynccontextmanager` lifespan in `energyx/api/main.py`)
- Two jobs to register:
  - **Weekly** (`interval`, weeks=1): `recompute_elasticity(home_id)` — re-trains elasticity RF from HistoricStore, caches PDP result
  - **Nightly** (`cron`, hour=2): `update_degradation_baselines(home_id)` — refreshes per-appliance baseline consumption

**Why it matters:** Without the scheduler, elasticity and degradation data never update. `compute_elasticity` and `update_degradation_baselines` tools work on-demand but won't auto-refresh.

---

## 2. Occupancy Modelling — `infer_occupancy` (Feature 18)

**Status:** Not implemented. Zero code.

**What's needed:**
- New file: `energyx/utils/occupancy.py`
- Function: `infer_occupancy(df, home_id) → pd.Series` with binary occupancy signal
- Method: change-point detection on `temperature_room` + `humidity` + `light` using ruptures or PELT
- Output: binary series at 30-minute resolution, keyed by timestamp
- Expose as a feature-engineering primitive used by AnalysisAgent tools (forecasting lag features, anomaly context)
- Add `@tool infer_occupancy` to `create_analysis_extensions()` so the LLM can call it when asked about occupancy patterns

**IDEAL signals to use:** `temperature_room` (IDEAL has consistent room temp data), `humidity`, and where available `light` — all in HistoricStore.

---

## 3. Comfort/Cost Trade-Off Dial (Feature 21)

**Status:** Not implemented.

**What's needed:**
- New `@tool comfort_cost_tradeoff(preference: float) -> str` in `new_tools.py`
  - `preference` in [0.0, 1.0]: 0 = savings-first, 1 = comfort-first
  - Re-weights the objective function in `schedule_flexible_loads` (currently minimises cost only)
  - At `preference=0.5`: balance cost reduction against temperature setpoint deviation
  - At `preference=1.0`: no deferral of heating/cooling regardless of tariff
- Integrate with `schedule_flexible_loads` tool so the scheduler respects the dial
- Expose slider in Streamlit Control tab

---

## 4. Multi-Site Aggregation (Feature 38)

**Status:** Not implemented.

**What's needed:**
- New `@tool aggregate_multi_site(home_ids: list, metric: str) -> str` in `new_tools.py`
- Reads HistoricStore for multiple homes over a shared date window
- Computes: total consumption, per-home consumption, benchmark ranking, cross-site anomaly rate
- Returns portfolio summary JSON + league table
- Dashboard: add a "Portfolio" section in Offline mode sidebar when multiple homes have data

---

## 5. Real Causal Attribution (Feature 35)

**Status:** `causal_attribution` tool exists in `new_tools.py` (~line 267–314) but uses keyword bucketing to assign blame (if cold day → weather, if weekend → behaviour). This is not causal attribution — it is correlation labelling.

**What's needed:**
- Replace with partial-effect decomposition:
  1. Train RF on full feature set
  2. For each attribution class (weather features, time-of-day features, occupancy features, appliance baseline features), compute PD contribution using `sklearn.inspection.partial_dependence`
  3. Δattribution = PD(this_period) − PD(last_period) per class
  4. Report each class's share of the total Δ
- Alternative (higher quality): DoWhy structural causal model with weather as exogenous, occupancy as latent, consumption as outcome — if reviewer-quality causal inference is needed

---

## 6. New IDEAL-Native Analysis Functions

These were identified as viable given IDEAL sensor types but have zero implementation:

### 6a. `detect_phantom_loads`
- Identify devices drawing power continuously below a threshold (≤5W) that never drop to zero
- Method: cluster electricity_apparent into always-on vs. duty-cycle devices using GMM or simple threshold scan
- Output: list of sensors with estimated standby wattage and annual cost

### 6b. `subcircuit_breakdown`
- Decompose whole-home electricity_apparent into per-sensor contributions
- Method: NILM-lite — for homes with both circuit-level and appliance-level data, regress appliance signals against whole-home signal
- Only applicable to the 39 enhanced IDEAL homes with `appliance_power` data

### 6c. `analyse_heating_efficiency`
- Ratio of gas consumption to indoor temperature gain per degree-day
- Requires `gas_pulse` (cumulative Wh) + `temperature_room` + `temperature_probe`
- Output: seasonal COP proxy, trend over months, flag if efficiency is degrading

### 6d. `normalise_consumption_by_temperature`
- Weather-normalise electricity/gas consumption to a standard 15°C baseline
- Separates structural trend from weather-driven variation
- Required by `project_longhorizon` for accurate multi-year projections

### 6e. `room_comfort_monitor`
- Sustained deviation from comfort band (18–22°C) by room
- Uses `temperature_room` per sensor_id (each room has its own sensor in IDEAL)
- Output: per-room comfort score, hours outside band, worst room

### 6f. `peak_demand_profiler`
- Identify daily/weekly peak demand windows
- Compute coincidence factor: how often does this home's peak coincide with grid peak (5–7 PM)?
- Output: peak time distribution, coincidence score, recommended shift windows

---

## 7. Test Coverage Gaps

**`tests/test_knowledge_agent.py`** — Fixed in this session (replaced removed `fetch_weather_forecast` method calls with standalone tool tests).

**Missing test files that should exist:**

| File | What to test |
|---|---|
| `tests/test_live_buffer.py` | Thread safety (concurrent add_tick + get_ticks_since), offset slicing, session lifecycle state transitions |
| `tests/test_api_endpoints.py` | `/stream/start` → `/ingest/tick` → `/stream/complete` → `/stream/persist` end-to-end with `httpx.AsyncClient` |
| `tests/test_appliance_forecast.py` | `forecast_appliance()` with synthetic appliance data, MAPE computation, ensemble weighting |
| `tests/test_new_tools.py` | `predict_bill`, `compute_elasticity` (mock RF), `causal_attribution` on fixture data |

---

## 8. `energyx/api/__init__.py` — Expose New App

**Status:** `energyx/api/__init__.py` likely exports the old `ingest.py` app.

**Fix:** Update to expose `from energyx.api.main import app`. Check what is currently exported and update accordingly.

---

## 9. Counterfactual Quality Improvements (from PROPOSED_IMPROVEMENTS.md §1)

**Status:** `counterfactual_inverse` uses Nelder-Mead (local, gradient-free optimizer).

**Proposed upgrades (in priority order):**
1. **Bayesian Optimization** — replace Nelder-Mead with Optuna `TPESampler`; handles non-convex objectives, sample-efficient
2. **DiCE integration** — generate 3–5 diverse counterfactuals instead of one point (`pip install dice-ml`)
3. **SHAP-guided first pass** — use existing SHAP signs to produce a cheap initial direction before optimization (zero extra cost)

---

## 10. Online Mode — Incremental Polling Throughput

**Observed:** The HTTP round-trip bottleneck limits `stream_ideal.py` to ~4 ticks/sec even at `--speed 0`. This is fine for demo purposes but too slow for streaming a full 14-day window at full density.

**Options:**
- Use `POST /ingest/batch` (already implemented, accepts up to 500 ticks per call) in `stream_ideal.py` instead of per-tick POSTs — would yield ~200× throughput improvement
- Current single-tick streaming is intentional for the "realistic data arrival" demo case; batch mode would be a separate `--batch` flag

---

## Priority Order (suggested)

1. **Scheduler wiring** (§1) — makes elasticity and degradation actually work
2. **Live buffer + API tests** (§7) — test coverage for the new architecture  
3. **Occupancy modelling** (§2) — needed as a feature-engineering primitive for other tools
4. **Phantom loads + heating efficiency** (§6a, §6c) — high impact, straightforward from IDEAL signals
5. **Causal attribution fix** (§5) — currently misleading output
6. **Multi-site aggregation** (§4) — useful for portfolio use case
7. **Comfort/cost dial** (§3) — UI polish
8. **Counterfactual upgrades** (§9) — research quality improvement
9. **Remaining IDEAL tools** (§6b, §6d, §6e, §6f) — complete the function inventory
10. **Batch streaming mode** (§10) — performance optimisation
