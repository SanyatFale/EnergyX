#!/usr/bin/env python
"""End-to-end smoke test of EnergyX functionality on real IDEAL data.

Tests:
  1. HistoricStore  — windowed reads, all 3 homes
  2. Monitors       — AnomalyDetectors, RealtimeCostMeter, BudgetTrajectoryTracker,
                      CarbonTracker, ApplianceBaselineWatcher
  3. Knowledge Agent — Q&A, tariff API, incentives, carbon intensity
  4. Analysis tools  — profile, anomaly explanation, bill prediction
  5. Orchestrator   — query routing classification
  6. Control Agent  — NL command parsing (no HA connection needed)

Usage:
    venv/bin/python scripts/smoke_test_real_data.py
"""

from __future__ import annotations
import json, os, sys, tempfile
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from energyx.data.historic_store import HistoricStore, ParquetBackend

STORE = HistoricStore(backend=ParquetBackend(_ROOT / "data" / "historic_store"))
SEP = "─" * 64

def section(title): print(f"\n{SEP}\n  {title}\n{SEP}")
def ok(msg):   print(f"  ✓  {msg}")
def warn(msg): print(f"  ~  {msg}")
def err(msg):  print(f"  ✗  {msg}")


# ══════════════════════════════════════════════
# 1. HistoricStore — windowed reads
# ══════════════════════════════════════════════
section("1. HistoricStore — windowed reads (2-week windows)")

WINDOWS = {
    "home128": (datetime(2017, 10, 1), datetime(2017, 10, 14)),
    "home62":  (datetime(2017, 3,  1), datetime(2017, 3,  14)),
    "home96":  (datetime(2017, 9,  1), datetime(2017, 9,  14)),
}
HOME_DFS: dict[str, pd.DataFrame] = {}

for hid, (s, e) in WINDOWS.items():
    df = STORE.read_ticks(hid, start=s, end=e)
    HOME_DFS[hid] = df
    types = sorted(df["sensor_type"].unique())
    ok(f"{hid}  {len(df):>9,} rows  sensors={types}")

assert all(len(d) > 0 for d in HOME_DFS.values())


# ══════════════════════════════════════════════
# 2. Monitors
# ══════════════════════════════════════════════
section("2. Monitors — home96 two-week batch")

from energyx.monitoring.monitors.anomaly_detectors import AnomalyDetectors
from energyx.monitoring.monitors.realtime_cost_meter import RealtimeCostMeter
from energyx.monitoring.monitors.budget_trajectory_tracker import BudgetTrajectoryTracker
from energyx.monitoring.monitors.carbon_tracker import CarbonTracker
from energyx.monitoring.monitors.appliance_baseline_watcher import ApplianceBaselineWatcher

df96 = HOME_DFS["home96"]

# ── electricity series for whole-home monitors
elec96 = (df96[df96["sensor_type"] == "electricity_apparent"]
          [["ts", "value"]].copy()
          .rename(columns={"value": "electricity_apparent"})
          .set_index("ts").sort_index())

# AnomalyDetectors
ad = AnomalyDetectors(config={"home_id": "home96", "value_column": "electricity_apparent"})
anom_events = ad.evaluate(elec96)
ok(f"AnomalyDetectors:          {len(anom_events):>4} events  "
   f"(severity breakdown: "
   f"{dict(pd.Series([str(getattr(e,'severity','?')) for e in anom_events]).value_counts().to_dict())})")
if anom_events:
    e0 = anom_events[0]
    print(f"       sample → val={e0.value:.1f}W  methods_agreed={e0.methods_agreed}  ts={e0.ts}")

# RealtimeCostMeter
rcm = RealtimeCostMeter(config={"home_id": "home96", "value_column": "electricity_apparent"})
cost_events = rcm.evaluate(elec96)
ok(f"RealtimeCostMeter:         {len(cost_events):>4} events")
if cost_events:
    e0 = cost_events[0]
    rate = getattr(e0, "burn_rate_gbp_per_hour", getattr(e0, "value", "?"))
    print(f"       sample → burn_rate≈{rate}")

# BudgetTrajectoryTracker
btt = BudgetTrajectoryTracker(config={"home_id": "home96", "monthly_cap_gbp": 120.0,
                                       "value_column": "electricity_apparent"})
budget_events = btt.evaluate(elec96)
ok(f"BudgetTrajectoryTracker:   {len(budget_events):>4} events  (cap=£120)")
if budget_events:
    e0 = budget_events[0]
    proj = getattr(e0, "projected_spend_gbp", getattr(e0, "value", "?"))
    print(f"       sample → projected_spend=£{proj}")

# CarbonTracker
ct = CarbonTracker(config={"home_id": "home96", "value_column": "electricity_apparent"})
carbon_events = ct.evaluate(elec96)
ok(f"CarbonTracker:             {len(carbon_events):>4} events  (uses stub 200 gCO₂/kWh)")

# ApplianceBaselineWatcher — per-appliance sensors
appl96 = df96[df96["sensor_type"] == "appliance_power"].copy()
if not appl96.empty:
    appl_wide = (appl96.pivot_table(
        index="ts", columns="sensor_id", values="value", aggfunc="mean"
    ).sort_index())
    abw = ApplianceBaselineWatcher(config={"home_id": "home96"})
    appl_events = abw.evaluate(appl_wide)
    ok(f"ApplianceBaselineWatcher:  {len(appl_events):>4} events  "
       f"({appl96['sensor_id'].nunique()} appliance sensors)")
    if appl_events:
        e0 = appl_events[0]
        print(f"       sample → device={getattr(e0,'device_id','?')}  "
              f"val={getattr(e0,'value','?'):.1f}W")
else:
    warn("ApplianceBaselineWatcher:  no appliance data in this window")


# ══════════════════════════════════════════════
# 3. Knowledge Agent
# ══════════════════════════════════════════════
section("3. Knowledge Agent — Q&A, tariff, incentives, carbon")

from energyx.agents.knowledge.agent import KnowledgeAgent
ka = KnowledgeAgent()

qa_pairs = [
    "What does the electricity_combined sensor mean in the IDEAL dataset?",
    "Am I eligible for ECO4 if I have EPC band D and I'm on Universal Credit?",
    "How does Economy 7 work and is it suitable for homes with storage heaters?",
]
for q in qa_pairs:
    ans = ka.answer(q)
    truncated = ans[:110] + "…" if len(ans) > 110 else ans
    ok(f"Q: {q[:60]}")
    print(f"     A: {truncated}")

# Structured tariff (must NOT come from RAG prose)
tariff = ka.get_active_tariff("home96")
unit_rate = (tariff.get("rates") or [{}])[0].get("unit_rate_gbp_per_kwh", "?")
ok(f"get_active_tariff: unit_rate={unit_rate}  "
   f"standing={tariff.get('standing_charge_gbp_per_day','?')}")
assert tariff.get("tariff_id"), "tariff missing tariff_id key"

# Incentives
inc = ka.list_applicable_incentives(epc_band="D", jurisdiction="england_wales")
schemes = [s["name"] for s in inc.get("applicable_schemes", [])]
ok(f"Incentives (EPC-D, England/Wales): {schemes}")
assert any("eco" in s.lower() or "bus" in s.lower() for s in schemes), "ECO4/BUS missing"

# Carbon intensity (no auth, fallback stub)
ci = ka.fetch_carbon_intensity("South West England")
ok(f"Carbon intensity: {ci.get('intensity_gco2_per_kwh','?')} gCO₂/kWh  "
   f"source={ci.get('source','?')}")


# ══════════════════════════════════════════════
# 4. Analysis Agent tools on real data
# ══════════════════════════════════════════════
section("4. Analysis Agent tools — home128 two-week electricity")

import json as _json

df128 = HOME_DFS["home128"]

# Build electricity+temp CSV for the AnalysisAgent
elec128 = (df128[df128["sensor_type"] == "electricity_apparent"]
           [["ts", "value"]].copy()
           .rename(columns={"value": "electricity_apparent"})
           .sort_values("ts"))

ok(f"Electricity series: {len(elec128):,} rows  "
   f"({elec128['ts'].min().date()} – {elec128['ts'].max().date()})")

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tf:
    csv_path = tf.name
    elec128.to_csv(tf, index=False)

try:
    from energyx.agents.analysis.agent import AnalysisAgent
    agent = AnalysisAgent(
        dataset_path=csv_path,
        time_column="ts",
        target_column="electricity_apparent",
    )
    ok(f"AnalysisAgent initialised  tools={sorted(agent.tool_map.keys())[:5]}…")

    # profile_dataset
    profile_raw = agent.tool_map["profile_dataset"].invoke({})
    prof = _json.loads(profile_raw) if isinstance(profile_raw, str) else profile_raw
    col = prof.get("electricity_apparent", prof.get("columns", {}).get("electricity_apparent", {}))
    mean_w = col.get("mean", prof.get("mean", "?"))
    ok(f"profile_dataset:    mean≈{float(mean_w):.1f}W  rows={prof.get('n_rows', len(elec128)):,}")

    # evaluate_tariff_switch (uses profile mean — no forecast required)
    switch_raw = agent.tool_map["evaluate_tariff_switch"].invoke({
        "alternative_tariff_id": "economy7_2026q1"
    })
    switch = _json.loads(switch_raw) if isinstance(switch_raw, str) else switch_raw
    assert switch.get("status") == "ok", f"evaluate_tariff_switch error: {switch.get('error')}"
    flat_annual = switch.get("current_annual_cost_gbp", "?")
    e7_annual   = switch.get("alternative_annual_cost_gbp", "?")
    saving      = switch.get("projected_annual_saving_gbp", "?")
    ok(f"evaluate_tariff_switch: flat=£{flat_annual}/yr  E7=£{e7_annual}/yr  "
       f"saving=£{saving}  ({switch.get('recommendation','?')})")

    # suggest_budget_corrections (uses profile — no forecast required)
    budget_raw = agent.tool_map["suggest_budget_corrections"].invoke({
        "monthly_cap_gbp": 120.0
    })
    budget = _json.loads(budget_raw) if isinstance(budget_raw, str) else budget_raw
    ok(f"suggest_budget_corrections:  status={budget.get('status', budget.get('error','?'))}")

finally:
    os.unlink(csv_path)


# ══════════════════════════════════════════════
# 5. Orchestrator query routing
# ══════════════════════════════════════════════
section("5. Orchestrator — classify_query routing")

from energyx.orchestrator.router import classify_query

cases = [
    ("What does the IDEAL electricity sensor record?",         "knowledge"),
    ("Show me anomalies in home96 last week",                  "analysis"),
    ("What is my current energy spend this month?",            "status"),
    ("Forecast my heating usage for next 7 days",              "analysis"),
    ("Am I eligible for ECO4?",                                "knowledge"),
    ("Turn off the washing machine now",                       "control"),
    ("Set the thermostat to 19 degrees",                       "control"),
    ("Defer the dishwasher until after 11pm",                  "control"),
]

passed = 0
for query, expected in cases:
    got = classify_query(query, use_llm=False)
    if got == expected:
        passed += 1
        print(f"  ✓  [{got:<12}]  {query}")
    else:
        print(f"  ✗  [{got:<12}] (want {expected})  {query}")

ok(f"Routing: {passed}/{len(cases)} correct")


# ══════════════════════════════════════════════
# 6. Control Agent — NL command parsing
# ══════════════════════════════════════════════
section("6. Control Agent — NL command parsing (no HA)")

from energyx.agents.control.agent import ControlAgent
from energyx.orchestrator.permission_manager import PermissionManager

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
    perm_path = tf.name
try:
    pm = PermissionManager(path=perm_path)
    for perm in ["auto_control:hvac", "auto_control:laundry_ev", "auto_control:emergency_off"]:
        pm.grant(perm)
    ca = ControlAgent(permission_manager=pm)

    commands = [
        "Turn off the washing machine",
        "Set the living room thermostat to 19 degrees",
        "Defer the dishwasher until 11pm tonight",
        "Power off the electric heater immediately",
    ]
    for cmd in commands:
        result = ca.handle(cmd)
        action  = result.get("action",  result.get("intent",  "?"))
        status  = result.get("status",  result.get("result",  "?"))
        entity  = result.get("entity_id", result.get("device_id", "—"))
        ok(f"{cmd[:52]:<54}  action={action}  entity={entity}")
finally:
    os.unlink(perm_path)


# ══════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════
section("SUMMARY")
total_rows = sum(len(d) for d in HOME_DFS.values())
print(f"  Homes loaded:  {list(WINDOWS.keys())}")
print(f"  Total rows:    {total_rows:,} (2-week windows each)")
print(f"  All sections completed.\n")
