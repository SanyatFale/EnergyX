"""EnergyX — Multi-agent energy management dashboard.

Streamlit UI backed by the full EnergyX agent stack:
  • Knowledge Agent  — UK energy Q&A, tariffs, grants
  • Analysis Agent   — Forecasting, bill prediction, tariff comparison
  • Control Agent    — NL device commands
  • Monitors         — Anomaly, cost, budget, carbon, appliance

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="EnergyX",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* typography */
    h1 { color: #0f2942; font-weight: 700; }
    h2, h3 { color: #1a3a5c; }

    /* card-style metric boxes */
    div[data-testid="metric-container"] {
        background: #f0f6ff;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        border-left: 4px solid #2563eb;
    }

    /* chat bubbles */
    .stChatMessage { border-radius: 12px; margin-bottom: 6px; }

    /* tighter sidebar */
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* event badges */
    .ev-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .ev-anomaly  { background:#fee2e2; color:#991b1b; }
    .ev-cost     { background:#fef9c3; color:#854d0e; }
    .ev-carbon   { background:#dcfce7; color:#166534; }
    .ev-budget   { background:#ede9fe; color:#5b21b6; }
    .ev-appliance{ background:#dbeafe; color:#1e40af; }
</style>
""",
    unsafe_allow_html=True,
)

# ── constants ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_STORE_PATH = _ROOT / "data" / "historic_store"

_HOMES = {
    "home96":  {"label": "Home 96 — 5-person house  (Sep 2017)",
                "start": datetime(2017, 9,  1), "end": datetime(2017, 9,  14)},
    "home128": {"label": "Home 128 — 1-person flat  (Oct 2017)",
                "start": datetime(2017, 10, 1), "end": datetime(2017, 10, 14)},
    "home62":  {"label": "Home 62 — 2-person flat   (Mar 2017)",
                "start": datetime(2017, 3,  1), "end": datetime(2017, 3,  14)},
}

# ── session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "home_id": "home96",
    "messages": [],          # chat history
    "monitor_events": None,  # cached monitor results
    "home_df": None,         # cached tick DataFrame
    "home_df_key": "",       # cache invalidation key
    "perm_path": None,       # temp file for PermissionManager
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_store():
    from energyx.data.historic_store import HistoricStore, ParquetBackend
    return HistoricStore(backend=ParquetBackend(_STORE_PATH))


def _load_home(home_id: str) -> pd.DataFrame:
    """Load 2-week tick window for the selected home (cached by key)."""
    key = home_id
    if st.session_state.home_df_key == key and st.session_state.home_df is not None:
        return st.session_state.home_df
    meta = _HOMES[home_id]
    store = _get_store()
    df = store.read_ticks(home_id, start=meta["start"], end=meta["end"])
    st.session_state.home_df = df
    st.session_state.home_df_key = key
    return df


def _elec_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return electricity_apparent as a time-indexed series."""
    sub = df[df["sensor_type"] == "electricity_apparent"][["ts", "value"]].copy()
    sub = sub.rename(columns={"value": "watts"}).sort_values("ts")
    return sub


def _export_elec_csv(df: pd.DataFrame) -> str:
    """Write electricity series to a temp CSV and return its path."""
    elec = _elec_series(df).rename(columns={"watts": "electricity_apparent"})
    tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    elec.to_csv(tf, index=False)
    tf.close()
    return tf.name


@st.cache_resource
def _get_knowledge_agent():
    from energyx.agents.knowledge.agent import KnowledgeAgent
    return KnowledgeAgent()


@st.cache_resource
def _get_permission_manager():
    import tempfile, os
    from energyx.orchestrator.permission_manager import PermissionManager
    tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tf.close()
    pm = PermissionManager(path=tf.name)
    for perm in ["auto_control:hvac", "auto_control:laundry_ev", "auto_control:emergency_off"]:
        pm.grant(perm)
    return pm


def _run_monitors(df: pd.DataFrame, home_id: str) -> dict:
    """Run all 5 monitors on the electricity series. Returns {monitor: [events]}."""
    from energyx.monitoring.monitors.anomaly_detectors import AnomalyDetectors
    from energyx.monitoring.monitors.realtime_cost_meter import RealtimeCostMeter
    from energyx.monitoring.monitors.budget_trajectory_tracker import BudgetTrajectoryTracker
    from energyx.monitoring.monitors.carbon_tracker import CarbonTracker
    from energyx.monitoring.monitors.appliance_baseline_watcher import ApplianceBaselineWatcher

    elec = (df[df["sensor_type"] == "electricity_apparent"][["ts", "value"]]
            .rename(columns={"value": "electricity_apparent"})
            .set_index("ts").sort_index())

    cfg_base = {"home_id": home_id, "value_column": "electricity_apparent"}
    results = {}

    results["anomaly"] = AnomalyDetectors(config=cfg_base).evaluate(elec)
    results["cost"]    = RealtimeCostMeter(config=cfg_base).evaluate(elec)
    results["budget"]  = BudgetTrajectoryTracker(
        config={**cfg_base, "monthly_cap_gbp": 120.0}).evaluate(elec)
    results["carbon"]  = CarbonTracker(config=cfg_base).evaluate(elec)

    appl = df[df["sensor_type"] == "appliance_power"].copy()
    if not appl.empty:
        appl_wide = (appl.pivot_table(index="ts", columns="sensor_id",
                                      values="value", aggfunc="mean").sort_index())
        results["appliance"] = ApplianceBaselineWatcher(
            config={"home_id": home_id}).evaluate(appl_wide)
    else:
        results["appliance"] = []

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Chat routing
# ═══════════════════════════════════════════════════════════════════════════════

def _chat_respond(query: str, home_id: str, df: pd.DataFrame) -> str:
    """Route a chat query through the appropriate EnergyX agent."""
    from energyx.orchestrator.router import classify_query

    intent = classify_query(query, use_llm=True)

    if intent == "knowledge":
        ka = _get_knowledge_agent()
        return ka.answer(query)

    if intent == "control":
        from energyx.agents.control.agent import ControlAgent
        pm = _get_permission_manager()
        ca = ControlAgent(permission_manager=pm)
        result = ca.handle(query)
        action  = result.get("action",    result.get("intent",    "parsed"))
        entity  = result.get("entity_id", result.get("device_id", "—"))
        status  = result.get("status",    result.get("result",    "ok"))
        return (
            f"**Control command parsed**\n\n"
            f"- Action: `{action}`\n"
            f"- Entity: `{entity}`\n"
            f"- Status: `{status}`\n\n"
            f"*Note: No Home Assistant connection — command captured for dispatch.*"
        )

    if intent == "status":
        events = st.session_state.get("monitor_events") or _run_monitors(df, home_id)
        lines = [f"**Live monitor status for {home_id}:**\n"]
        for name, evts in events.items():
            lines.append(f"- **{name.title()}**: {len(evts)} events")
            if evts:
                e = evts[0]
                val = getattr(e, "value", "?")
                try:
                    lines.append(f"  → latest value: {float(val):.1f}")
                except Exception:
                    pass
        return "\n".join(lines)

    # analysis (default)
    csv_path = _export_elec_csv(df)
    try:
        from energyx.agents.analysis.agent import AnalysisAgent
        agent = AnalysisAgent(
            dataset_path=csv_path,
            time_column="ts",
            target_column="electricity_apparent",
        )
        # Profile first (required for tariff tools)
        agent.tool_map["profile_dataset"].invoke({})

        # Try to match specific analysis intents
        q_lower = query.lower()

        if any(w in q_lower for w in ("bill", "cost", "spend", "tariff", "economy 7", "e7")):
            raw = agent.tool_map["evaluate_tariff_switch"].invoke(
                {"alternative_tariff_id": "economy7_2026q1"})
            r = json.loads(raw) if isinstance(raw, str) else raw
            if r.get("status") == "ok":
                return (
                    f"**Tariff comparison for {home_id}:**\n\n"
                    f"| Tariff | Annual cost |\n|---|---|\n"
                    f"| {r.get('current_tariff','Flat')} | £{r.get('current_annual_cost_gbp')} |\n"
                    f"| {r.get('alternative_tariff','E7')} | £{r.get('alternative_annual_cost_gbp')} |\n\n"
                    f"**Recommendation:** {r.get('recommendation','—')} "
                    f"(annual saving: £{r.get('projected_annual_saving_gbp','?')})\n\n"
                    f"*Based on mean consumption of {r.get('annual_kwh_estimate','?')} kWh/year.*"
                )

        if any(w in q_lower for w in ("budget", "cap", "over budget", "projection")):
            raw = agent.tool_map["suggest_budget_corrections"].invoke({"monthly_cap_gbp": 120.0})
            r = json.loads(raw) if isinstance(raw, str) else raw
            if r.get("status") == "ok":
                lines = [f"**Budget analysis for {home_id} (cap £120/mo):**\n"]
                for rec in r.get("recommendations", [])[:5]:
                    lines.append(f"- {rec}")
                return "\n".join(lines) if len(lines) > 1 else json.dumps(r, indent=2)

        # Generic: profile summary
        raw = agent.tool_map["profile_dataset"].invoke({})
        r = json.loads(raw) if isinstance(raw, str) else raw
        mean_w = r.get("mean", "?")
        try:
            mean_w = f"{float(mean_w):.1f}"
        except Exception:
            pass
        return (
            f"**Analysis of {home_id} electricity data:**\n\n"
            f"- Mean power: **{mean_w} W**\n"
            f"- Rows: {r.get('n_rows', '?'):,}\n"
            f"- Frequency: {r.get('frequency', '?')}\n"
            f"- Missing: {r.get('missing_pct', '?')}%\n"
            f"- Outlier rate: {r.get('outlier_pct', '?')}%\n\n"
            f"*Try asking: 'compare tariffs', 'what's my budget status', or 'am I eligible for ECO4?'*"
        )
    finally:
        try:
            import os; os.unlink(csv_path)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ EnergyX")
    st.caption("Multi-agent energy management")
    st.divider()

    st.subheader("Home")
    home_id = st.selectbox(
        "Select home",
        options=list(_HOMES.keys()),
        format_func=lambda h: _HOMES[h]["label"],
        key="home_id",
    )

    meta = _HOMES[home_id]
    st.caption(
        f"Data window: {meta['start'].strftime('%d %b')} – "
        f"{meta['end'].strftime('%d %b %Y')}"
    )

    st.divider()

    # Quick KPIs from tariff
    try:
        from energyx.agents.knowledge.agent import KnowledgeAgent
        _ka = KnowledgeAgent()
        _t = _ka.get_active_tariff(home_id)
        _rates = (_t.get("rates") or [{}])[0]
        st.metric("Unit rate", f"{float(_rates.get('unit_rate_gbp_per_kwh', 0.2459))*100:.1f} p/kWh")
        st.metric("Standing charge", f"{float(_t.get('standing_charge_gbp_per_day', 0.61))*100:.0f} p/day")
    except Exception:
        pass

    st.divider()

    # Carbon intensity
    try:
        _ci = _ka.fetch_carbon_intensity("South West England")
        _ci_val = _ci.get("intensity_gco2_per_kwh", "—")
        st.metric("Grid carbon", f"{_ci_val} gCO₂/kWh")
    except Exception:
        pass

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.monitor_events = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════════
st.title("⚡ EnergyX")
st.caption(f"Real IDEAL household data · {_HOMES[home_id]['label']}")

tab_overview, tab_chat, tab_monitor, tab_control = st.tabs(
    ["📊 Overview", "💬 Chat", "🔔 Monitor", "🎛️ Control"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    with st.spinner("Loading home data…"):
        df = _load_home(home_id)

    if df.empty:
        st.warning("No data found for this home in the store. Run `scripts/ingest_ideal.py` first.")
        st.stop()

    elec = _elec_series(df)

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    mean_w   = elec["watts"].mean()
    peak_w   = elec["watts"].max()
    n_days   = (elec["ts"].max() - elec["ts"].min()).total_seconds() / 86400
    kwh_tot  = elec["watts"].mean() / 1000 * n_days * 24
    cost_est = kwh_tot * 0.2459 + 0.61 * n_days

    c1.metric("Mean power",   f"{mean_w:.0f} W")
    c2.metric("Peak power",   f"{peak_w:.0f} W")
    c3.metric("Est. energy",  f"{kwh_tot:.1f} kWh",  f"{n_days:.0f}-day window")
    c4.metric("Est. cost",    f"£{cost_est:.2f}",    "at cap rate")

    st.divider()

    # ── Electricity trend ────────────────────────────────────────────────────
    st.subheader("Electricity — apparent power")

    # Resample to hourly for a readable chart
    elec_hr = (elec.set_index("ts")["watts"]
               .resample("1h").mean()
               .reset_index()
               .rename(columns={"ts": "time", "watts": "W"}))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=elec_hr["time"], y=elec_hr["W"],
        mode="lines", name="Electricity (W)",
        line=dict(color="#2563eb", width=1.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
    ))
    fig.add_hline(y=mean_w, line_dash="dot", line_color="gray",
                  annotation_text=f"mean {mean_w:.0f} W")
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=None, yaxis_title="Watts",
        showlegend=False, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Sensor breakdown ─────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Sensor breakdown")
        breakdown = (df.groupby("sensor_type")["value"]
                     .agg(["count", "mean", "max"])
                     .rename(columns={"count": "rows", "mean": "mean_val", "max": "max_val"})
                     .reset_index())
        breakdown["mean_val"] = breakdown["mean_val"].round(2)
        breakdown["max_val"]  = breakdown["max_val"].round(2)
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    with col_r:
        st.subheader("Temperature probes (daily avg)")
        temp_probes = df[df["sensor_type"] == "temperature_probe"][["ts", "sensor_id", "value"]].copy()
        if not temp_probes.empty:
            # Show top 3 probes by row count
            top_sensors = (temp_probes.groupby("sensor_id")["value"]
                           .count().nlargest(3).index.tolist())
            tp_hr = (temp_probes[temp_probes["sensor_id"].isin(top_sensors)]
                     .pivot_table(index="ts", columns="sensor_id", values="value", aggfunc="mean")
                     .resample("6h").mean()
                     .reset_index())
            fig2 = go.Figure()
            for col in tp_hr.columns[1:]:
                fig2.add_trace(go.Scatter(
                    x=tp_hr["ts"], y=tp_hr[col],
                    mode="lines", name=str(col)[:20],
                    line=dict(width=1.2),
                ))
            fig2.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="°C", template="plotly_white",
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No temperature probe data in this window.")

    # ── Gas usage (daily totals) ──────────────────────────────────────────────
    gas = df[df["sensor_type"] == "gas_pulse"][["ts", "value"]].copy()
    if not gas.empty:
        st.subheader("Gas — cumulative readings (Wh)")
        gas_daily = (gas.set_index("ts")["value"]
                     .resample("1D").last()
                     .diff().fillna(0)
                     .clip(lower=0)
                     .reset_index()
                     .rename(columns={"value": "Wh_consumed"}))
        fig3 = go.Figure(go.Bar(
            x=gas_daily["ts"], y=gas_daily["Wh_consumed"],
            marker_color="#f97316", name="Gas Wh/day",
        ))
        fig3.update_layout(
            height=220, margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white", showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask EnergyX anything")
    st.caption(
        "Routed automatically to **Knowledge** · **Analysis** · **Control** · **Status** agents."
    )

    # Welcome message on first load
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"Hello! I'm your EnergyX assistant for **{_HOMES[home_id]['label']}**.\n\n"
                "Try asking me:\n"
                "- *Am I eligible for ECO4?*\n"
                "- *Compare flat vs Economy 7 tariff for this home*\n"
                "- *What's the Boiler Upgrade Scheme grant for a heat pump?*\n"
                "- *What is my budget status?*\n"
                "- *Turn off the washing machine*\n"
            ),
        })

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about energy, tariffs, grants, or your home data…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        df = _load_home(home_id)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = _chat_respond(user_input, home_id, df)
                except Exception as e:
                    reply = f"⚠️ Error: {e}"
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Monitor
# ─────────────────────────────────────────────────────────────────────────────
with tab_monitor:
    st.subheader("Monitor events")
    st.caption("Runs all 5 monitors over the 2-week data window on demand.")

    run_col, _ = st.columns([1, 3])
    if run_col.button("▶  Run monitors", type="primary", use_container_width=True):
        df = _load_home(home_id)
        with st.spinner("Running monitors…"):
            st.session_state.monitor_events = _run_monitors(df, home_id)

    events = st.session_state.monitor_events
    if events is None:
        st.info("Click **Run monitors** to analyse the data window.")
    else:
        total = sum(len(v) for v in events.values())
        st.success(f"{total} total events across 5 monitors")

        _BADGE = {
            "anomaly":   ("ev-anomaly",   "Anomaly"),
            "cost":      ("ev-cost",       "Cost"),
            "budget":    ("ev-budget",     "Budget"),
            "carbon":    ("ev-carbon",     "Carbon"),
            "appliance": ("ev-appliance",  "Appliance"),
        }

        for monitor_name, evts in events.items():
            badge_cls, badge_label = _BADGE.get(monitor_name, ("ev-badge", monitor_name))
            header = (
                f'<span class="ev-badge {badge_cls}">{badge_label}</span>'
                f" **{len(evts)} events**"
            )
            with st.expander(f"{badge_label} monitor — {len(evts)} events", expanded=(len(evts) > 0)):
                if not evts:
                    st.write("No events in this window.")
                else:
                    rows = []
                    for e in evts:
                        row = {
                            "ts":       str(getattr(e, "ts", "?")),
                            "severity": str(getattr(e, "severity", "—")),
                            "value":    round(float(getattr(e, "value", 0) or 0), 2),
                        }
                        # Extra fields
                        for attr in ("burn_rate_gbp_per_hour", "projected_spend_gbp",
                                     "intensity_gco2_per_kwh", "methods_agreed", "device_id"):
                            v = getattr(e, attr, None)
                            if v is not None:
                                row[attr] = v
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Control
# ─────────────────────────────────────────────────────────────────────────────
with tab_control:
    st.subheader("Device control")
    st.caption(
        "NL command parser — no live Home Assistant connection. "
        "Commands are parsed and would be dispatched in online mode."
    )

    st.info(
        "Permissions granted: `auto_control:hvac` · `auto_control:laundry_ev` · "
        "`auto_control:emergency_off`",
        icon="🔑",
    )

    # Quick-action buttons
    st.markdown("**Quick commands**")
    qcols = st.columns(3)
    _QUICK = [
        ("Turn off washing machine", "🧺"),
        ("Set thermostat to 19°C",   "🌡️"),
        ("Defer dishwasher to 11pm", "🍽️"),
        ("Boost hot water",          "🔥"),
        ("Power off electric heater","❄️"),
        ("Cancel scheduled charge",  "🔌"),
    ]
    _pending_cmd: Optional[str] = None
    for i, (cmd, icon) in enumerate(_QUICK):
        col = qcols[i % 3]
        if col.button(f"{icon} {cmd}", key=f"qcmd_{i}", use_container_width=True):
            _pending_cmd = cmd

    st.divider()

    # Free-text command
    st.markdown("**Or type a command**")
    ctrl_input = st.text_input(
        "NL command",
        placeholder="e.g. 'Set the living room to 20 degrees' or 'Turn off the tumble dryer'",
        label_visibility="collapsed",
    )
    if st.button("Send command", type="primary"):
        _pending_cmd = ctrl_input.strip() or None

    # Execute pending command
    if _pending_cmd:
        from energyx.agents.control.agent import ControlAgent
        pm = _get_permission_manager()
        ca = ControlAgent(permission_manager=pm)
        with st.spinner(f"Parsing: *{_pending_cmd}*"):
            try:
                result = ca.handle(_pending_cmd)
            except Exception as e:
                result = {"error": str(e)}

        st.markdown("**Result**")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Action",  str(result.get("action",    result.get("intent",    "—"))))
        rc2.metric("Entity",  str(result.get("entity_id", result.get("device_id", "—"))))
        rc3.metric("Status",  str(result.get("status",    result.get("result",    "—"))))

        with st.expander("Full response"):
            st.json(result)


# ─── footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("EnergyX · IDEAL dataset · Sanyat · 2026")
