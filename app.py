"""EnergyX — Multi-agent home energy management dashboard.

Two operating modes
-------------------
Offline Analysis
    Pick a home from the HistoricStore, explore 14-day windows, chat with agents.

Online (Live)
    Connects to the EnergyX FastAPI data bus (energyx/api/main.py).
    A separate producer script (scripts/stream_ideal.py) posts IDEAL ticks to
    the bus; this dashboard polls the bus every few seconds and renders a
    growing, real-time view of consumption, monitoring events, and alerts.
    The app and the producer are fully decoupled — start them independently.

How to run
----------
1. Start the data bus:
       uvicorn energyx.api.main:app --host 0.0.0.0 --port 8000

2. Start the dashboard:
       streamlit run app.py

3. (when ready) Start the producer:
       python scripts/stream_ideal.py --home home96 --every 5 --speed 0.05

Agents routed via Orchestrator (plan → execute for Analysis):
  Knowledge · Analysis · Control · Monitor
"""

from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EnergyX",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    h1 { color: #0f2942; font-weight: 700; }
    h2, h3 { color: #1a3a5c; }
    div[data-testid="metric-container"] {
        background: #f0f6ff; border-radius: 10px;
        padding: 0.6rem 1rem; border-left: 4px solid #2563eb;
    }
    .stChatMessage { border-radius: 12px; margin-bottom: 6px; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    .ev-badge { display:inline-block; padding:2px 10px; border-radius:20px;
                font-size:0.78rem; font-weight:600; margin-right:4px; }
    .ev-anomaly  { background:#fee2e2; color:#991b1b; }
    .ev-cost     { background:#fef9c3; color:#854d0e; }
    .ev-carbon   { background:#dcfce7; color:#166534; }
    .ev-budget   { background:#ede9fe; color:#5b21b6; }
    .ev-appliance{ background:#dbeafe; color:#1e40af; }
    .ev-critical { background:#fca5a5; color:#7f1d1d; }
    .ev-warn     { background:#fde68a; color:#92400e; }
    .ev-info     { background:#bfdbfe; color:#1e3a8a; }
    .live-dot    { color:#dc2626; animation:blink 1s step-end infinite; }
    @keyframes blink { 50%{opacity:0} }
</style>
""", unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────────────────────
_ROOT           = Path(__file__).parent
_STORE_PATH     = _ROOT / "data" / "historic_store"
_HIERARCHY_ROOT = _ROOT / "data" / "ideal_hierarchy"

_HOMES: Dict[str, Dict[str, Any]] = {
    "home96":  {"label": "Home 96 — 5-person house",
                "start": datetime(2017, 9, 1), "end": datetime(2017, 9, 14)},
    "home128": {"label": "Home 128 — 1-person flat",
                "start": datetime(2017, 10, 1), "end": datetime(2017, 10, 14)},
}

# Human-readable names for appliance types and sensor types
_APPLIANCE_LABELS: Dict[str, str] = {
    "washingmachine": "Washing Machine", "dishwasher": "Dishwasher",
    "fridgefreezer": "Fridge/Freezer",   "kettle": "Kettle",
    "microwave": "Microwave",            "toaster": "Toaster",
    "vacuumcleaner": "Vacuum Cleaner",   "tumbledryer": "Tumble Dryer",
    "cooker": "Cooker",                  "other": "Other Appliance",
    "bath": "Bath",                      "shower": "Shower",
    "sink": "Sink",
}
_SENSOR_LABELS: Dict[str, str] = {
    "temperature": "Temperature (°C×10)", "humidity": "Humidity (%)",
    "radiator_input": "Radiator Inlet (°C×10)", "radiator_output": "Radiator Outlet (°C×10)",
    "central_heating_flow": "CH Flow Temp", "central_heating_return": "CH Return Temp",
    "hot_water_hot_pipe": "Hot Water Pipe", "hot_water_cold_pipe": "Cold Water Pipe",
    "mains": "Mains Electricity (W)", "gas": "Gas (W)", "electric_combined": "Electricity (W)",
}
_ROOM_ICONS: Dict[str, str] = {
    "kitchen": "🍳", "livingroom": "🛋️", "bedroom": "🛏️",
    "bathroom": "🚿", "hall": "🚪", "bedroom_1": "🛏️", "bedroom_2": "🛏️",
    "bedroom_3": "🛏️", "bedroom_4": "🛏️", "bedroom_5": "🛏️",
    "bathroom_1": "🚿", "bathroom_2": "🚿", "bathroom_3": "🚿",
    "hall_1": "🚪", "hall_2": "🚪",
}

_SEVERITY_CSS = {"CRITICAL": "ev-critical", "WARNING": "ev-warn", "INFO": "ev-info"}

# ── session state defaults ─────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    # shared
    "app_mode":    "offline",
    "home_id":     "home96",
    "messages":    [],
    # offline
    "monitor_events": None,
    "home_df":        None,
    "home_df_key":    "",
    # human-in-the-loop: plan approval state
    # {query, agent, plan, csv_path, feature_columns} or None
    "pending_analysis": None,
    "pending_budget":   None,
    # online — live polling state
    "api_url":          "http://localhost:8000",
    "live_ticks":       [],    # accumulated tick dicts from bus
    "live_events":      [],    # accumulated event dicts from bus
    "last_tick_offset": 0,
    "last_evt_offset":  0,
    "live_status":      {},    # last /live/status response
    "live_home_id":     None,  # home reported by active stream
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# Cached resources
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_store():
    from energyx.data.historic_store import HistoricStore, ParquetBackend
    return HistoricStore(backend=ParquetBackend(_STORE_PATH))


@st.cache_data(show_spinner=False)
def _parse_home_structure(home_id: str) -> Dict[str, Any]:
    """Walk ideal_hierarchy dir and return room/sensor/appliance map."""
    base = _HIERARCHY_ROOT / home_id
    if not base.exists():
        return {"rooms": {}, "home_level": [], "has_weather": False}
    rooms: Dict[str, Any] = {}
    home_level: List[str] = []
    has_weather = False
    for path in sorted(base.iterdir()):
        if path.name == "weather.parquet":
            has_weather = True
        elif path.is_file() and path.suffix == ".parquet":
            home_level.append(path.stem)
        elif path.is_dir():
            sensors, appliances = [], []
            for item in sorted(path.iterdir()):
                if item.is_file() and item.suffix == ".parquet":
                    sensors.append(item.stem)
                elif item.is_dir():
                    # appliance subdirectory: name = appliance type
                    appliances.append(item.name)
            rooms[path.name] = {"sensors": sensors, "appliances": appliances}
    return {"rooms": rooms, "home_level": home_level, "has_weather": has_weather}


@st.cache_data(show_spinner=False, ttl=600)
def _load_hierarchy_home(home_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load ticks from ideal_hierarchy parquets in the standard tick format."""
    base = _HIERARCHY_ROOT / home_id
    if not base.exists():
        return pd.DataFrame()
    struct = _parse_home_structure(home_id)
    frames: List[pd.DataFrame] = []

    def _read(path: Path, sensor_type_override: Optional[str] = None,
              sensor_id_override: Optional[str] = None) -> None:
        if not path.exists():
            return
        df = pd.read_parquet(path)
        if "ts" not in df.columns or "value" not in df.columns:
            return
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        if sensor_type_override:
            df["sensor_type"] = sensor_type_override
        if sensor_id_override:
            df["sensor_id"] = sensor_id_override
        if "sensor_id" not in df.columns:
            df["sensor_id"] = path.stem
        if "unit" not in df.columns:
            df["unit"] = ""
        if "home_id" not in df.columns:
            df["home_id"] = home_id
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        frames.append(df[["home_id", "ts", "sensor_type", "sensor_id", "value", "unit"]])

    # Home-level files (mains, gas, electric_combined, heating pipes…)
    for name in struct["home_level"]:
        sensor_type = "electricity_real" if name in ("mains", "electric_combined") else name
        _read(base / f"{name}.parquet", sensor_type_override=sensor_type)

    # Room sensors and appliances
    for room, info in struct["rooms"].items():
        room_path = base / room
        for sensor in info["sensors"]:
            _read(room_path / f"{sensor}.parquet",
                  sensor_id_override=f"{room}_{sensor}")
        for appl in info["appliances"]:
            _read(room_path / appl / f"{appl}.parquet",
                  sensor_type_override="appliance_power",
                  sensor_id_override=f"{room}_{appl}")

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    mask = (combined["ts"] >= pd.Timestamp(start)) & (combined["ts"] < pd.Timestamp(end))
    return combined[mask].sort_values("ts").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_appliance_series(home_id: str, room: str, appl: str,
                           start: datetime, end: datetime) -> pd.DataFrame:
    """Load a single appliance's power series, resampled to 1-minute."""
    path = _HIERARCHY_ROOT / home_id / room / appl / f"{appl}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    mask = (df["ts"] >= pd.Timestamp(start)) & (df["ts"] < pd.Timestamp(end))
    df = df[mask][["ts", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    if df.empty:
        return df
    return df.set_index("ts")["value"].resample("1min").mean().dropna().reset_index()


@st.cache_data(show_spinner=False)
def _load_room_sensors(home_id: str, room: str,
                       start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Load all sensor parquets for a room, resampled to 5-minute."""
    room_path = _HIERARCHY_ROOT / home_id / room
    if not room_path.exists():
        return {}
    result = {}
    struct = _parse_home_structure(home_id)
    sensors = struct["rooms"].get(room, {}).get("sensors", [])
    for sensor in sensors:
        path = room_path / f"{sensor}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "ts" not in df.columns or "value" not in df.columns:
            continue
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        mask = (df["ts"] >= pd.Timestamp(start)) & (df["ts"] < pd.Timestamp(end))
        s = df[mask].set_index("ts")["value"].resample("5min").mean().dropna()
        if not s.empty:
            result[sensor] = s.reset_index()
    return result


@st.cache_data(show_spinner=False)
def _load_weather(home_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load weather data for a home."""
    path = _HIERARCHY_ROOT / home_id / "weather.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    mask = (df["ts"] >= pd.Timestamp(start)) & (df["ts"] < pd.Timestamp(end))
    num_cols = [c for c in df.columns if c not in ("home_id", "ts")]
    return df[mask].set_index("ts")[num_cols].apply(pd.to_numeric, errors="coerce").resample("1h").mean().reset_index()


@st.cache_resource
def _get_knowledge_agent():
    from energyx.agents.knowledge.agent import KnowledgeAgent
    return KnowledgeAgent()


@st.cache_resource
def _get_orchestrator(home_id: str):
    from energyx.orchestrator.orchestrator import Orchestrator
    from energyx.orchestrator.mode_manager import Mode
    tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tf.close()
    orch = Orchestrator(
        config={"home_id": home_id},
        permissions_path=tf.name,
        initial_mode=Mode.OFFLINE,
    )
    for perm in ["auto_control:hvac", "auto_control:laundry_ev", "auto_control:emergency_off"]:
        orch.permissions.grant(perm)
    return orch


def _llm_label() -> str:
    try:
        from tinyts.config import settings
        p = settings.llm_provider
        if p == "openrouter": return f"OpenRouter · {settings.openrouter_model.split('/')[-1]}"
        if p == "cerebras":   return f"Cerebras · {settings.cerebras_model}"
        return f"Ollama · {settings.ollama_model}"
    except Exception:
        return "LLM: unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Offline data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_home(home_id: str) -> pd.DataFrame:
    key = home_id
    if st.session_state.home_df_key == key and st.session_state.home_df is not None:
        return st.session_state.home_df
    meta = _HOMES[home_id]
    # Prefer hierarchy parquets (richer, already structured)
    if (_HIERARCHY_ROOT / home_id).exists():
        df = _load_hierarchy_home(home_id, meta["start"], meta["end"])
    else:
        store = _get_store()
        df = store.read_ticks(home_id, start=meta["start"], end=meta["end"])
    st.session_state.home_df     = df
    st.session_state.home_df_key = key
    return df


def _elec_series(df: pd.DataFrame) -> pd.DataFrame:
    elec_types = ("electricity_apparent", "electricity_real", "mains")
    sub = df[df["sensor_type"].isin(elec_types)][["ts", "value"]].copy()
    # Deduplicate timestamps that arise from multi-sensor homes
    sub = sub.drop_duplicates("ts")
    return sub.rename(columns={"value": "watts"}).sort_values("ts")


def _export_elec_csv(df: pd.DataFrame) -> str:
    """Legacy single-column export (kept for monitors)."""
    elec = _elec_series(df).rename(columns={"watts": "electricity_apparent"})
    tf   = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    elec.to_csv(tf, index=False)
    tf.close()
    return tf.name


def _export_analysis_csv(df: pd.DataFrame, query: str = "") -> tuple:
    """Build a wide-format CSV with electricity + ALL available IDEAL features.

    Feature extraction strategy:
    - Temperature: mean indoor + per-room columns (up to 6 rooms)
    - Humidity: mean indoor
    - Light: mean indoor (occupancy / daytime proxy)
    - Gas: total gas (heating proxy)
    - Appliance subcircuits: total + top 5 by variance (direct load components)
    - Electricity subcircuits: if present (electricity_real by circuit)

    The query is scanned for explicit feature mentions. If none are found,
    all available features are returned so the planner can decide.
    Returns (csv_path, feature_columns, available_feature_map).
    """
    elec = _elec_series(df).rename(columns={"watts": "electricity_apparent"}).set_index("ts")
    # Deduplicate electricity index (IDEAL data can have duplicate 1-Hz timestamps)
    elec = elec[~elec.index.duplicated(keep="first")]
    idx = elec.index

    def _align(series: pd.Series, col_name: str, fill: str = "interpolate") -> Optional[pd.Series]:
        """Resample/interpolate a sensor series onto the electricity time grid."""
        if series.empty:
            return None
        s = series.sort_index()
        # Deduplicate before any reindex — IDEAL sensors can share the same timestamp
        s = s[~s.index.duplicated(keep="first")]
        # Expand to the union of sensor timestamps + electricity grid, then interpolate
        combined_idx = s.index.union(idx)
        s = s.reindex(combined_idx).sort_index()
        if fill == "interpolate":
            s = s.interpolate("time", limit=10).ffill().bfill()
        else:
            s = s.ffill().fillna(0.0)
        s = s.reindex(idx)
        if s.notna().sum() < 5:
            return None
        return s.rename(col_name)

    feature_cols: List[str] = []

    # ── Temperature ──────────────────────────────────────────────────────────
    for stype in ("temperature_probe", "temperature_room", "temperature"):
        temp_raw = df[df["sensor_type"] == stype][["ts", "sensor_id", "value"]].copy()
        if temp_raw.empty:
            continue
        temp_raw = temp_raw.set_index("ts")
        # Mean across all sensors → temperature_mean
        s = _align(temp_raw.groupby(level=0)["value"].mean(), "temperature_mean")
        if s is not None and "temperature_mean" not in feature_cols:
            elec["temperature_mean"] = s
            feature_cols.append("temperature_mean")
        # Per-room columns: top 3 by variance, labeled temp_room_1/2/3
        rooms = temp_raw["sensor_id"].unique()
        if len(rooms) > 1:
            room_vars = {
                rid: temp_raw[temp_raw["sensor_id"] == rid]["value"].var()
                for rid in rooms
            }
            top_rooms = sorted(room_vars, key=room_vars.get, reverse=True)[:3]
            for i, rid in enumerate(top_rooms, 1):
                col = f"temp_room_{i}"
                s2 = _align(temp_raw[temp_raw["sensor_id"] == rid]["value"], col)
                if s2 is not None:
                    elec[col] = s2
                    feature_cols.append(col)

    # ── Humidity ─────────────────────────────────────────────────────────────
    hum_raw = df[df["sensor_type"] == "humidity"][["ts", "value"]].copy()
    if not hum_raw.empty:
        s = _align(hum_raw.set_index("ts")["value"].groupby(level=0).mean(), "humidity_mean")
        if s is not None:
            elec["humidity_mean"] = s
            feature_cols.append("humidity_mean")

    # ── Light (occupancy / daytime proxy) ────────────────────────────────────
    light_raw = df[df["sensor_type"] == "light"][["ts", "value"]].copy()
    if not light_raw.empty:
        s = _align(light_raw.set_index("ts")["value"].groupby(level=0).mean(), "light_mean")
        if s is not None:
            elec["light_mean"] = s
            feature_cols.append("light_mean")

    # ── Gas (heating proxy) ──────────────────────────────────────────────────
    gas_raw = df[df["sensor_type"].isin(("gas_pulse", "gas"))][["ts", "value"]].copy()
    if not gas_raw.empty:
        s = _align(gas_raw.set_index("ts")["value"].groupby(level=0).sum(), "gas_watts", fill="fill")
        if s is not None:
            elec["gas_watts"] = s
            feature_cols.append("gas_watts")

    # ── Appliances ───────────────────────────────────────────────────────────
    appl_raw = df[df["sensor_type"] == "appliance_power"][["ts", "sensor_id", "value"]].copy()
    if not appl_raw.empty:
        appl_raw = appl_raw.set_index("ts")
        # Total appliance load
        s = _align(appl_raw.groupby(level=0)["value"].sum(), "appliance_total_watts")
        if s is not None:
            elec["appliance_total_watts"] = s
            feature_cols.append("appliance_total_watts")
        # Top 5 individual appliances by variance (most informative for causal analysis)
        appl_var = (appl_raw.groupby("sensor_id")["value"].var()
                    .dropna().sort_values(ascending=False))
        for aid in appl_var.head(3).index:
            col = f"appl_{aid}"
            s2 = _align(appl_raw[appl_raw["sensor_id"] == aid]["value"], col)
            if s2 is not None:
                elec[col] = s2
                feature_cols.append(col)

    # ── Electricity subcircuits (only when multiple circuits exist, not mains) ──
    real_raw = df[df["sensor_type"] == "electricity_real"][["ts", "sensor_id", "value"]].copy()
    if not real_raw.empty:
        real_raw = real_raw.set_index("ts")
        circuits = [c for c in real_raw["sensor_id"].unique()
                    if str(c) not in ("mains", "electric_combined", "None", "nan")]
        for cid in circuits[:4]:
            col = f"circuit_{cid}"
            s = _align(real_raw[real_raw["sensor_id"] == cid]["value"], col)
            if s is not None:
                elec[col] = s
                feature_cols.append(col)

    # ── Weather (from weather.parquet via session_state, if loaded) ─────────────
    weather_cache = st.session_state.get("_weather_cache", {})
    home_id = st.session_state.get("home_id")
    if home_id and home_id not in weather_cache:
        try:
            meta = _HOMES.get(home_id, {})
            w_df = _load_weather(home_id, meta.get("start"), meta.get("end"))
            weather_cache[home_id] = w_df
            st.session_state["_weather_cache"] = weather_cache
        except Exception:
            pass
    w_df = weather_cache.get(home_id) if home_id else None
    if w_df is not None and not w_df.empty:
        w_df = w_df.copy()
        if "ts" in w_df.columns:
            w_df = w_df.set_index("ts")
        # temp column: IDEAL weather stores actual °C (not ×10)
        if "temp" in w_df.columns:
            s = _align(pd.to_numeric(w_df["temp"], errors="coerce"), "weather_temp_c")
            if s is not None:
                elec["weather_temp_c"] = s
                feature_cols.append("weather_temp_c")
        if "rhum" in w_df.columns:
            s = _align(pd.to_numeric(w_df["rhum"], errors="coerce"), "weather_humidity_pct")
            if s is not None:
                elec["weather_humidity_pct"] = s
                feature_cols.append("weather_humidity_pct")

    # ── Query-guided filtering ────────────────────────────────────────────────
    # If the user explicitly named features, keep only those + always-on columns.
    # Otherwise return all available features (planner decides).
    active_cols = _filter_features_by_query(query, feature_cols) if query.strip() else feature_cols

    tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    # Only write columns that are actually needed
    out_cols = ["ts", "electricity_apparent"] + active_cols
    elec.reset_index()[out_cols].to_csv(tf, index=False)
    tf.close()
    return tf.name, active_cols


# Maps query keywords → feature column name patterns
_FEATURE_KEYWORD_MAP: Dict[str, List[str]] = {
    "temperature": ["temperature_mean", "temp_"],
    "thermal":     ["temperature_mean", "temp_"],
    "warm":        ["temperature_mean", "gas_watts"],
    "cold":        ["temperature_mean", "gas_watts"],
    "heating":     ["gas_watts", "temperature_mean"],
    "gas":         ["gas_watts"],
    "boiler":      ["gas_watts"],
    "humidity":    ["humidity_mean"],
    "light":       ["light_mean"],
    "lighting":    ["light_mean", "appl_"],
    "occupancy":   ["light_mean", "humidity_mean"],
    "appliance":   ["appliance_total_watts", "appl_"],
    "appliances":  ["appliance_total_watts", "appl_"],
    "circuit":     ["circuit_"],
    "subcircuit":  ["circuit_"],
    "room":        ["temp_"],
    "weather":     ["weather_temp_c", "weather_humidity_pct"],
    "outdoor":     ["weather_temp_c", "weather_humidity_pct"],
    "outside":     ["weather_temp_c", "weather_humidity_pct"],
}


def _filter_features_by_query(query: str, available: List[str]) -> List[str]:
    """Return the subset of available features relevant to the query keywords.

    If no explicit feature words are detected, returns all available features.
    """
    q = query.lower()
    matched_patterns: List[str] = []
    for keyword, patterns in _FEATURE_KEYWORD_MAP.items():
        if keyword in q:
            matched_patterns.extend(patterns)

    if not matched_patterns:
        return available  # no explicit mention → all features

    result: List[str] = []
    for col in available:
        if any(col == p or col.startswith(p) for p in matched_patterns):
            result.append(col)
    # Always include temperature_mean if anything matched (it's always a good causal driver)
    if result and "temperature_mean" in available and "temperature_mean" not in result:
        result.insert(0, "temperature_mean")
    return result or available  # fallback to all if filter was too aggressive


def _run_monitors(df: pd.DataFrame, home_id: str) -> dict:
    from energyx.monitoring.monitors.anomaly_detectors import AnomalyDetectors
    from energyx.monitoring.monitors.realtime_cost_meter import RealtimeCostMeter
    from energyx.monitoring.monitors.budget_trajectory_tracker import BudgetTrajectoryTracker
    from energyx.monitoring.monitors.carbon_tracker import CarbonTracker
    from energyx.monitoring.monitors.appliance_baseline_watcher import ApplianceBaselineWatcher

    # Accept both legacy (electricity_apparent) and hierarchy (electricity_real / mains) types
    elec_types = ("electricity_apparent", "electricity_real", "mains")
    elec_raw = df[df["sensor_type"].isin(elec_types)][["ts", "value"]]
    if elec_raw.empty:
        elec_raw = df[df["sensor_type"] == "electricity_real"][["ts", "value"]]
    elec = (elec_raw.rename(columns={"value": "electricity_apparent"})
            .set_index("ts").sort_index())
    elec = elec[~elec.index.duplicated(keep="first")]
    cfg = {"home_id": home_id, "value_column": "electricity_apparent"}
    results = {
        "anomaly": AnomalyDetectors(config=cfg).evaluate(elec),
        "cost":    RealtimeCostMeter(config=cfg).evaluate(elec),
        "budget":  BudgetTrajectoryTracker(config={**cfg, "monthly_cap_gbp": 120.0}).evaluate(elec),
        "carbon":  CarbonTracker(config=cfg).evaluate(elec),
    }
    appl = df[df["sensor_type"] == "appliance_power"].copy()
    if not appl.empty:
        appl_wide = appl.pivot_table(
            index="ts", columns="sensor_id", values="value", aggfunc="mean"
        ).sort_index()
        results["appliance"] = ApplianceBaselineWatcher(
            config={"home_id": home_id}).evaluate(appl_wide)
    else:
        results["appliance"] = []
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Chat — via Orchestrator (plan → execute for Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def _chat_plan(query: str, home_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Phase 1: classify query. If analysis, run plan phase and return pending state.

    Returns {is_analysis, pending} for analysis queries (plan awaiting approval),
    or {is_analysis: False, result} for non-analysis queries handled immediately.
    """
    from energyx.orchestrator.router import classify_query
    intent = classify_query(query)

    if intent != "analysis":
        # Non-analysis queries go straight through
        orch = _get_orchestrator(home_id)
        csv_path, feat_cols = _export_analysis_csv(df, query)
        try:
            result = orch.handle_query(
                query=query,
                dataset_path=csv_path,
                time_column="ts",
                target_column="electricity_apparent",
                feature_columns=feat_cols or None,
            )
        except Exception as e:
            result = {"intent": intent, "response": f"⚠️ {e}"}
        finally:
            try:
                import os; os.unlink(csv_path)
            except Exception:
                pass
        return {"is_analysis": False, "result": result}

    # Analysis — run plan phase only, return for human approval
    from energyx.agents.analysis.agent import AnalysisAgent
    from tinyts.config import settings
    csv_path, feat_cols = _export_analysis_csv(df, query)
    output_dir = str(settings.output_dir / f"run_{home_id}_{int(pd.Timestamp.now().timestamp())}")
    try:
        agent = AnalysisAgent(
            dataset_path=csv_path,
            time_column="ts",
            target_column="electricity_apparent",
            output_dir=output_dir,
            feature_columns=feat_cols or None,
        )
        plan = agent.plan(query)
        # Strip filters on numeric and time columns — only categorical filters are valid.
        if plan.data_filters:
            try:
                import pandas as _pd_fp
                df_peek = _pd_fp.read_csv(csv_path, nrows=2)
                numeric_cols = set(df_peek.select_dtypes(include="number").columns)
            except Exception:
                numeric_cols = set()
            plan.data_filters = [
                f for f in plan.data_filters
                if f.get("column") not in numeric_cols
                and f.get("column") not in ("ts", "timestamp", "time", "date")
            ]
        # Reinitialize tools with LLM-resolved columns/filters
        agent.reinitialize_with_plan(plan)
    except Exception as e:
        try:
            import os; os.unlink(csv_path)
        except Exception:
            pass
        return {"is_analysis": False, "result": {"intent": "analysis", "response": f"⚠️ Planning failed: {e}"}}

    return {
        "is_analysis": True,
        "pending": {
            "query":           query,
            "agent":           agent,
            "plan":            plan,
            "csv_path":        csv_path,
            "feature_columns": feat_cols,
        },
    }


def _chat_execute(
    pending: Dict[str, Any],
    mode: str,
    hil_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 2: execute an approved (possibly user-modified) plan."""
    agent    = pending["agent"]
    plan     = pending["plan"]
    csv_path = pending["csv_path"]
    steps: List[str] = []

    plan.is_multivariate = bool(plan.feature_columns)

    # Strip filters on numeric columns — only categorical column filters are valid.
    # The prompt now instructs the LLM not to create these, but guard defensively.
    if plan.data_filters:
        try:
            import pandas as _pd_f
            df_peek = _pd_f.read_csv(csv_path, nrows=2)
            numeric_cols = set(df_peek.select_dtypes(include="number").columns)
        except Exception:
            numeric_cols = set()
        plan.data_filters = [
            f for f in plan.data_filters
            if f.get("column") not in numeric_cols
            and f.get("column") not in ("ts", "timestamp", "time", "date")
        ]

    try:
        agent.reinitialize_with_plan(plan)
    except Exception:
        pass

    # Fresh model cache per query — prevents stale predictions across queries
    agent.session["_model_cache"] = {}

    def _on_tool(name: str, args: Any, result_json: Any) -> None:
        steps.append(name)

    try:
        result = agent.execute(plan, on_tool_call=_on_tool, hil_params=hil_params)
        result["intent"] = "analysis"
    except Exception as e:
        result = {"intent": "analysis", "response": f"⚠️ Execution failed: {e}"}
    finally:
        try:
            import os; os.unlink(csv_path)
        except Exception:
            pass

    # Check if make_budget phase 1 left an awaiting_budget_input flag
    awaiting_budget = agent.session.get("awaiting_budget_input", False)

    response = result.get("response", "")
    if mode == "offline" and result.get("intent") == "control":
        response = "**[Advisory — offline mode]** " + response
    if steps:
        response += f"\n\n*Tools: {' → '.join(steps)}*"
    return {
        "response":        response or "Done.",
        "output_dir":      result.get("output_dir"),
        "plan":            plan,
        "steps":           steps,
        "intent":          "analysis",
        "awaiting_budget": awaiting_budget,
    }


def _render_chat_plan(plan: Any) -> None:
    """Show the executed plan in a collapsible expander (post-execution summary)."""
    if plan is None:
        return
    with st.expander("📋 Analysis plan used", expanded=False):
        cols = st.columns(2)
        cols[0].markdown(f"**Task:** {getattr(plan, 'task_type', '—')}")
        cols[0].markdown(f"**Horizon:** {getattr(plan, 'horizon', '—')} steps")
        cols[0].markdown(f"**Multivariate:** {getattr(plan, 'is_multivariate', False)}")
        cols[1].markdown(f"**Target:** `{getattr(plan, 'target_column', '—')}`")
        models = getattr(plan, 'models_included', [])
        if models:
            cols[1].markdown(f"**Models:** {', '.join(models)}")
        feats = getattr(plan, 'feature_columns', [])
        if feats:
            st.markdown(f"**Features:** {', '.join(feats)}")


_VALID_MODELS_LIST = ["Naive", "SeasonalNaive", "ARIMA", "ETS", "N-BEATS", "RandomForest", "LightGBM"]


_UV_MODELS = ["Naive", "SeasonalNaive", "ARIMA", "ETS", "N-BEATS"]
_MV_MODELS = ["RandomForest", "LightGBM"]


def _render_plan_approval(pending: Dict[str, Any], mode: str, home_id: str) -> None:
    """Human-in-the-loop: show the plan, allow edits, approve or cancel."""
    plan  = pending["plan"]
    query = pending["query"]

    # Non-forecast tasks (budget, cost, weather, consumption, anomaly) don't
    # need the forecast form — auto-execute them immediately.
    _NON_FORECAST_TASKS = {"budget", "cost", "weather", "consumption",
                           "tariff", "counterfactual"}
    task_type_str = getattr(plan, "task_type", "forecast") or "forecast"
    is_non_forecast = any(t in task_type_str.lower() for t in _NON_FORECAST_TASKS)
    # Also detect by query keywords for cases where LLM labels them "forecast"
    _q = query.lower()
    if not is_non_forecast:
        if any(w in _q for w in ("budget", "cost", "spend", "bill", "weather",
                                  "consumption", "usage pattern", "tariff")):
            is_non_forecast = True

    if is_non_forecast:
        st.info(f"**Query:** {query}", icon="🔍")
        with st.spinner("Running analysis…"):
            chat_result = _chat_execute(pending, mode)
        st.session_state.pending_analysis = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": chat_result["response"],
            "output_dir": chat_result.get("output_dir"),
            "plan": chat_result.get("plan"),
        })
        if chat_result.get("awaiting_budget"):
            st.session_state.pending_budget = pending
        st.rerun()
        return

    st.info(f"**Query:** {query}", icon="🔍")
    st.markdown("##### Review the analysis plan before execution")
    st.caption("The agent has profiled the dataset and proposed this plan. Edit if needed, then approve.")

    all_available = pending.get("feature_columns", [])
    _plan_task = getattr(plan, "task_type", "forecast") or "forecast"
    _is_anomaly_task = _plan_task == "anomaly" or any(
        w in query.lower() for w in ("anomal", "unusual", "spike", "outlier", "detect")
    )

    # ── Forecast controls — hidden for anomaly queries ──
    is_mv = False
    horizon = max(1, int(getattr(plan, "horizon", None) or 24))
    models: List[str] = getattr(plan, "models_included", []) or ["ARIMA", "ETS", "Naive"]
    selected_feats: List[str] = []
    needs_explain = bool(getattr(plan, "needs_explanation", False))
    target_col = getattr(plan, "target_column", "electricity_apparent")

    if not _is_anomaly_task:
        default_task = "Multivariate" if (getattr(plan, "is_multivariate", False) or bool(all_available)) else "Univariate"
        task_type = st.radio(
            "**Forecast type**",
            ["Univariate (electricity only)", "Multivariate (with external features)"],
            index=0 if default_task == "Univariate" else 1,
            key="plan_task_type",
            help="Univariate: Naive/ARIMA/ETS/N-BEATS on electricity alone.  "
                 "Multivariate: RandomForest/LightGBM using temperature, appliances, etc. as causal features.",
            horizontal=True,
        )
        is_mv = task_type.startswith("Multivariate")

    with st.form("plan_approval_form"):
        if not _is_anomaly_task:
            cols = st.columns(2)

            # Target column dropdown
            agent_profile = pending.get("agent")
            if agent_profile and hasattr(agent_profile, "session"):
                p = agent_profile.session.get("profile")
                numeric_opts = getattr(p, "numeric_columns", ["electricity_apparent"])[:20] if p else ["electricity_apparent"]
            else:
                numeric_opts = ["electricity_apparent"]
            plan_target = getattr(plan, "target_column", "electricity_apparent")
            if plan_target not in numeric_opts:
                numeric_opts = [plan_target] + numeric_opts
            target_col = cols[0].selectbox(
                "Target column", options=numeric_opts,
                index=numeric_opts.index(plan_target) if plan_target in numeric_opts else 0,
            )

            horizon = cols[1].number_input(
                "Horizon (steps)", min_value=1, max_value=8760,
                value=max(1, int(getattr(plan, "horizon", None) or 24)),
            )

            # Models — filtered by task type
            if is_mv:
                model_options = _MV_MODELS
                plan_models   = [m for m in getattr(plan, "models_included", _MV_MODELS) if m in _MV_MODELS]
                default_models = plan_models or _MV_MODELS
                st.caption("🔀 Multivariate models use the feature columns below as causal inputs.")
            else:
                model_options = _UV_MODELS
                plan_models   = [m for m in getattr(plan, "models_included", ["ARIMA", "ETS", "Naive"]) if m in _UV_MODELS]
                default_models = plan_models or ["ARIMA", "ETS", "Naive"]
                st.caption("📈 Univariate models forecast from electricity history only.")

            models = st.multiselect("Models to train", options=model_options, default=default_models)

            # Feature columns — only shown for multivariate
            if is_mv:
                plan_feats    = getattr(plan, "feature_columns", []) or []
                default_feats = [f for f in (plan_feats if plan_feats else all_available) if f in all_available]
                selected_feats = st.multiselect(
                    "Causal feature columns",
                    options=all_available,
                    default=default_feats,
                    help="Signals from the dataset used as causal drivers in the forecast.",
                )
                if all_available:
                    st.caption(
                        "**Available:** " +
                        " · ".join(f"`{c}`" for c in all_available)
                    )

            needs_explain = st.checkbox(
                "Include explainability (SHAP, feature importance, STL, lag analysis)",
                value=bool(getattr(plan, "needs_explanation", False)),
            )

        # ── Anomaly HIL: method selection (shown only for anomaly queries) ──
        _ALL_ANOMALY_METHODS = ["zscore", "mad", "rolling", "iqr", "stl",
                                "isolationforest", "dbscan"]
        anomaly_methods: List[str] = _ALL_ANOMALY_METHODS
        anomaly_level = "home"
        if _is_anomaly_task:
            st.markdown("**Anomaly detection parameters**")
            anomaly_methods = st.multiselect(
                "Detection methods",
                options=_ALL_ANOMALY_METHODS,
                default=_ALL_ANOMALY_METHODS,
                help="Select which anomaly detection methods to run. All 7 is recommended.",
            )
            anomaly_level = st.radio(
                "Detection level",
                ["home", "appliance"],
                index=0,
                horizontal=True,
                help="Home: whole-home electricity. Appliance: also checks appl_* columns.",
            )

        col_approve, col_cancel = st.columns(2)
        approved  = col_approve.form_submit_button("▶  Run analysis", type="primary", use_container_width=True)
        cancelled = col_cancel.form_submit_button("✕  Cancel", use_container_width=True)

    if approved:
        plan.horizon           = horizon
        plan.target_column     = target_col
        plan.models_included   = models or (["RandomForest"] if is_mv else ["ARIMA", "ETS", "Naive"])
        plan.is_multivariate   = is_mv
        plan.feature_columns   = selected_feats if is_mv else []
        plan.needs_explanation = needs_explain
        pending["plan"] = plan

        # Build HIL params dict to pass into agent.execute()
        hil_params: Dict[str, Any] = {
            "forecast_univariate":   {"horizon": horizon, "models": ", ".join(plan.models_included)},
            "forecast_multivariate": {"horizon": horizon, "models": ", ".join(plan.models_included)},
            "detect_anomalies":      {"methods": ", ".join(anomaly_methods) if anomaly_methods else "all",
                                      "level": anomaly_level},
        }

        with st.spinner("Executing analysis…"):
            chat_result = _chat_execute(pending, mode, hil_params=hil_params)

        st.session_state.pending_analysis = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": chat_result["response"],
            "output_dir": chat_result.get("output_dir"),
            "plan": chat_result.get("plan"),
        })
        # If make_budget phase 1 is waiting for budget input, stash the pending
        # agent so we can call make_budget(budget_gbp=X) in phase 2.
        if chat_result.get("awaiting_budget"):
            st.session_state.pending_budget = pending
        st.rerun()

    if cancelled:
        st.session_state.pending_analysis = None
        st.session_state.messages.append({
            "role": "assistant", "content": "Analysis cancelled.",
            "output_dir": None, "plan": None,
        })
        st.rerun()


def _render_budget_input(pending_budget: Dict[str, Any], mode: str) -> None:
    """Phase 2 of make_budget: ask user for their monthly budget target."""
    st.markdown("---")
    st.markdown("#### 💰 Set your monthly energy budget")
    st.caption(
        "The pie chart above shows your current appliance breakdown. "
        "Enter your monthly target below to get personalised saving suggestions."
    )
    with st.form("budget_input_form"):
        budget_val = st.number_input(
            "Monthly budget (£)",
            min_value=10.0, max_value=1000.0, value=100.0, step=5.0,
            help="How much would you like to spend on electricity per month?",
        )
        col_set, col_skip = st.columns(2)
        set_budget  = col_set.form_submit_button("Set budget & get suggestions",
                                                  type="primary", use_container_width=True)
        skip_budget = col_skip.form_submit_button("Skip", use_container_width=True)

    if set_budget:
        agent = pending_budget["agent"]
        fn = agent.tool_map.get("make_budget")
        sug_fn = agent.tool_map.get("suggest_budget_correction")
        response_parts = []
        if fn:
            try:
                r = json.loads(fn.invoke({"budget_gbp": float(budget_val)}))
                proj  = r.get("projected_monthly_gbp", 0)
                over  = r.get("overshoot_gbp", 0)
                on_tr = r.get("on_track", False)
                if on_tr:
                    response_parts.append(
                        f"✅ **On track.** Projected: £{proj:.2f}/month vs budget £{budget_val:.2f}/month."
                    )
                else:
                    response_parts.append(
                        f"⚠️ **Over budget** by £{over:.2f}. Projected: £{proj:.2f}, budget: £{budget_val:.2f}/month."
                    )
                for s in r.get("suggestions", [])[:5]:
                    response_parts.append(
                        f"- **{s.get('action','')}** — save ~£{s.get('potential_saving_gbp',0):.2f}/month"
                    )
            except Exception as e:
                response_parts.append(f"Budget error: {e}")
        if sug_fn:
            try:
                sr = json.loads(sug_fn.invoke({}))
                if sr.get("appliance_actions"):
                    response_parts.append("\n**Appliance-level actions:**")
                    for a in sr["appliance_actions"][:5]:
                        response_parts.append(
                            f"- {a['appliance']}: {a['action']} "
                            f"(save ~£{a['potential_saving_gbp']:.2f}/month, {a['effort']} effort)"
                        )
            except Exception:
                pass

        st.session_state.messages.append({
            "role": "assistant",
            "content": "\n".join(response_parts) or "Budget set.",
            "output_dir": pending_budget.get("agent", {}).output_dir
                          if hasattr(pending_budget.get("agent", {}), "output_dir") else None,
            "plan": None,
        })
        st.session_state.pending_budget = None
        st.rerun()

    if skip_budget:
        st.session_state.pending_budget = None
        st.rerun()


_CHAT_PLOT_WHITELIST = {"final_forecast", "model_comparison", "anomaly_detection",
                        "counterfactual", "budget_pie"}

def _render_chat_plots(output_dir: Optional[str]) -> None:
    """Render whitelisted Plotly HTML files saved by the analysis agent."""
    if not output_dir:
        return
    import glob as _glob
    for fpath in sorted(_glob.glob(str(Path(output_dir) / "*.html"))):
        if Path(fpath).stem not in _CHAT_PLOT_WHITELIST:
            continue
        try:
            html = Path(fpath).read_text(encoding="utf-8")
            label = Path(fpath).stem.replace("_", " ").title()
            with st.expander(f"📈 {label}", expanded=True):
                st.components.v1.html(html, height=520, scrolling=False)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Online mode — live polling helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _api_get(path: str, base: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """HTTP GET to the live bus.  Returns None on any error."""
    url = (base or st.session_state.api_url).rstrip("/") + path
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _api_post(path: str, body: Dict[str, Any], base: Optional[str] = None) -> Optional[Dict[str, Any]]:
    url  = (base or st.session_state.api_url).rstrip("/") + path
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data,
                                   headers={"Content-Type": "application/json"},
                                   method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _poll_live_data() -> bool:
    """Fetch any new ticks and events since last poll.  Returns True if new data."""
    got_data = False

    tick_resp = _api_get(f"/live/ticks?since={st.session_state.last_tick_offset}")
    if tick_resp:
        new_ticks = tick_resp.get("ticks", [])
        if new_ticks:
            st.session_state.live_ticks.extend(new_ticks)
            st.session_state.last_tick_offset = tick_resp.get("total", st.session_state.last_tick_offset)
            got_data = True

    evt_resp = _api_get(f"/live/events?since={st.session_state.last_evt_offset}")
    if evt_resp:
        new_evts = evt_resp.get("events", [])
        if new_evts:
            st.session_state.live_events.extend(new_evts)
            st.session_state.last_evt_offset = evt_resp.get("total", st.session_state.last_evt_offset)
            got_data = True

    status = _api_get("/live/status")
    if status:
        st.session_state.live_status  = status
        st.session_state.live_home_id = status.get("home_id")

    return got_data


def _live_ticks_df() -> pd.DataFrame:
    """Convert accumulated live ticks to a DataFrame."""
    ticks = st.session_state.live_ticks
    if not ticks:
        return pd.DataFrame(columns=["ts", "sensor_id", "sensor_type", "value", "unit"])
    df = pd.DataFrame(ticks)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def _live_events_df() -> pd.DataFrame:
    evts = st.session_state.live_events
    if not evts:
        return pd.DataFrame()
    return pd.DataFrame(evts)


def _export_live_elec_csv() -> Optional[str]:
    """Export accumulated electricity_apparent ticks to a temp CSV for Analysis Agent."""
    df = _live_ticks_df()
    elec = df[df["sensor_type"] == "electricity_apparent"][["ts", "value"]].copy()
    if elec.empty:
        return None
    elec = elec.rename(columns={"value": "electricity_apparent"})
    tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    elec.to_csv(tf, index=False)
    tf.close()
    return tf.name


# ═══════════════════════════════════════════════════════════════════════════════
# Online mode — tab renderers (wrapped in st.fragment for auto-refresh)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_live_overview(refresh_s: int) -> None:

    @st.fragment(run_every=timedelta(seconds=refresh_s))
    def _fragment():
        _poll_live_data()
        df    = _live_ticks_df()
        evts  = _live_events_df()
        status = st.session_state.live_status

        tick_count  = status.get("tick_count",  len(st.session_state.live_ticks))
        event_count = status.get("event_count", len(st.session_state.live_events))
        is_active   = status.get("is_active",   False)
        is_complete = status.get("is_complete", False)

        # ── Status bar ────────────────────────────────────────────────────────
        if is_active:
            st.markdown('<span class="live-dot">●</span> **LIVE** — receiving data',
                        unsafe_allow_html=True)
        elif is_complete:
            st.success("✓ Stream complete — all ticks received.")
        else:
            st.info("Waiting for data producer to start…  "
                    "Run: `python scripts/stream_ideal.py --home home96`")

        # ── KPIs ──────────────────────────────────────────────────────────────
        elec = df[df["sensor_type"] == "electricity_apparent"] if not df.empty else pd.DataFrame()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Ticks received",  f"{tick_count:,}")
        c2.metric("Events fired",    str(event_count))
        if not elec.empty:
            mean_w = elec["value"].mean()
            peak_w = elec["value"].max()
            n_h    = (elec["ts"].max() - elec["ts"].min()).total_seconds() / 3600
            kwh    = mean_w / 1000 * n_h
            c3.metric("Mean power",  f"{mean_w:.0f} W")
            c4.metric("Peak power",  f"{peak_w:.0f} W")
            c5.metric("Energy so far", f"{kwh:.2f} kWh")

        # ── Live electricity chart ─────────────────────────────────────────────
        if not elec.empty:
            st.subheader("Live electricity — apparent power")
            elec_hr = (elec.set_index("ts")["value"]
                       .resample("5min").mean()
                       .reset_index()
                       .rename(columns={"ts": "time", "value": "W"}))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=elec_hr["time"], y=elec_hr["W"],
                mode="lines", name="Electricity",
                line=dict(color="#2563eb", width=1.5),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
            ))
            # Overlay critical/warning event markers
            if not evts.empty and "ts" in evts.columns:
                crit = evts[evts.get("severity", pd.Series()).str.upper().isin(
                    ["CRITICAL", "WARNING"])] if "severity" in evts.columns else pd.DataFrame()
                for _, row in crit.head(30).iterrows():
                    try:
                        ts = pd.to_datetime(row["ts"])
                        col = "#dc2626" if str(row.get("severity","")).upper() == "CRITICAL" else "#f59e0b"
                        fig.add_vline(x=ts, line_dash="dot", line_color=col,
                                      line_width=1.2,
                                      annotation_text=str(row.get("type",""))[:10],
                                      annotation_font_size=8)
                    except Exception:
                        pass
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                               yaxis_title="Watts", showlegend=False,
                               template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No electricity data yet.  Start the producer to begin streaming.")

        # ── Sensor summary ────────────────────────────────────────────────────
        if not df.empty:
            st.subheader("Sensor summary")
            summary = (df.groupby("sensor_type")["value"]
                       .agg(["count", "mean", "max"])
                       .rename(columns={"count": "ticks", "mean": "mean_val", "max": "max_val"})
                       .reset_index())
            summary["mean_val"] = summary["mean_val"].round(2)
            summary["max_val"]  = summary["max_val"].round(2)
            st.dataframe(summary, use_container_width=True, hide_index=True)

    _fragment()


def _render_live_monitor(refresh_s: int) -> None:

    @st.fragment(run_every=timedelta(seconds=refresh_s))
    def _fragment():
        _poll_live_data()
        evts_df = _live_events_df()
        status  = st.session_state.live_status

        st.subheader("Live event feed")
        st.caption(f"{status.get('event_count', 0)} total events · "
                   f"refreshes every {refresh_s}s")

        if evts_df.empty:
            st.info("No monitoring events yet.")
            return

        # Recent events first
        disp = evts_df.copy()
        if "ts" in disp.columns:
            try:
                disp["ts"] = pd.to_datetime(disp["ts"], errors="coerce") \
                               .dt.strftime("%d %b %H:%M:%S")
            except Exception:
                pass

        # Event-type counts as badges
        if "type" in disp.columns:
            type_counts = disp["type"].value_counts().to_dict()
            cols = st.columns(min(len(type_counts), 6))
            for i, (t, c) in enumerate(type_counts.items()):
                cols[i % len(cols)].metric(t.replace("Event", ""), str(c))

        # Severity counts
        if "severity" in disp.columns:
            sv = disp["severity"].str.upper().value_counts().to_dict()
            st.write("Severity: " + "  ·  ".join(f"**{k}**: {v}" for k, v in sv.items()))

        # Build a clean display: ts, type, severity, detail (type-specific value)
        def _event_detail(row) -> str:
            t = str(row.get("type", ""))
            if t == "carbon.realtime":
                v = row.get("intensity_gco2_per_kwh")
                return f"{v:.0f} gCO₂/kWh" if v is not None else ""
            if t == "cost.realtime":
                v = row.get("rate_gbp_per_hour")
                return f"£{v:.4f}/hr" if v is not None else ""
            if t == "habit.drift":
                return str(row.get("description", row.get("message", "")))[:60]
            if t == "forgot.turn_off":
                dev = row.get("device_id", "")
                mins = row.get("idle_minutes")
                return f"{dev} idle {mins:.0f} min" if mins else str(dev)
            if t == "anomaly.appliance":
                sid = row.get("sensor_id", "")
                sc = row.get("zscore", row.get("score", ""))
                return f"{sid} score={sc}" if sc else str(sid)
            if t == "budget.trajectory":
                proj = row.get("projected_gbp")
                cap = row.get("monthly_cap_gbp")
                return f"projected £{proj:.2f} / cap £{cap:.2f}" if proj else ""
            # Generic fallback: first non-standard key
            skip = {"type", "ts", "severity", "home_id", "source", "region"}
            for k, v in row.items():
                if k not in skip and v is not None and str(v).strip():
                    return f"{k}={v}"
            return ""

        clean = pd.DataFrame({
            "Time":     disp.get("ts", pd.Series(dtype=str)),
            "Event":    disp.get("type", pd.Series(dtype=str)),
            "Severity": disp.get("severity", pd.Series(dtype=str)),
            "Detail":   disp.apply(_event_detail, axis=1),
        })
        st.dataframe(
            clean.tail(200)[::-1],
            use_container_width=True,
            hide_index=True,
        )

    _fragment()


def _render_live_chat(home_id: str, mode: str) -> None:
    st.subheader("Ask EnergyX anything")
    st.caption("Analysis queries show a plan for review before execution.")

    live_home = st.session_state.live_home_id or home_id

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"I'm monitoring **{live_home}** in real time. "
                "Ask me to forecast, compare tariffs, or check your budget — "
                "I'll work with the data received so far.\n\n"
                "- *Forecast electricity for the next 24 hours and explain*\n"
                "- *Am I on track with my £120/month budget?*\n"
                "- *Compare flat rate vs Economy 7*\n"
                "- *What is the BUS grant for a heat pump?*\n"
            ),
            "output_dir": None,
            "plan": None,
        })

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("output_dir"):
                _render_chat_plots(msg["output_dir"])
            if msg.get("plan") and msg["role"] == "assistant":
                _render_chat_plan(msg["plan"])

    # Human-in-the-loop plan approval
    if st.session_state.pending_analysis:
        _render_plan_approval(st.session_state.pending_analysis, mode, live_home)
        return

    # make_budget phase 2: budget input form
    if st.session_state.pending_budget:
        _render_budget_input(st.session_state.pending_budget, mode)
        return

    user_input = st.chat_input("Ask about your live energy data…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "output_dir": None, "plan": None})
        with st.chat_message("user"):
            st.markdown(user_input)

        csv_path = _export_live_elec_csv()
        if not csv_path:
            msg = "No electricity data received yet. Start the producer and wait for some ticks."
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "output_dir": None, "plan": None})
            st.rerun()
            return

        try:
            df_live = pd.read_csv(csv_path, parse_dates=["ts"])
        finally:
            try:
                import os; os.unlink(csv_path)
            except Exception:
                pass

        with st.status("Planning…", expanded=False) as s:
            try:
                plan_result = _chat_plan(user_input, live_home, df_live)
                s.update(label="Done", state="complete")
            except Exception as e:
                plan_result = {"is_analysis": False, "result": {"intent": "error", "response": f"⚠️ {e}"}}
                s.update(label="Error", state="error")

        if plan_result["is_analysis"]:
            st.session_state.pending_analysis = plan_result["pending"]
            st.rerun()
        else:
            result   = plan_result["result"]
            response = result.get("response", "Done.")
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response, "output_dir": None, "plan": None})
            st.rerun()


def _render_session_complete_banner() -> None:
    """Show when the producer has finished and prompt the user for next steps."""
    status = st.session_state.live_status
    if not status.get("is_complete"):
        return

    st.warning(
        f"✅  **Data stream complete** — {status.get('tick_count', 0):,} ticks received "
        f"from **{status.get('home_id', '?')}**.  "
        "What would you like to do next?",
        icon="🏁",
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("📊 Switch to Offline Analysis", use_container_width=True,
                     key="banner_offline"):
            # Persist data to HistoricStore so offline mode can read it
            with st.spinner("Persisting data to HistoricStore…"):
                resp = _api_post("/stream/persist", {})
            if resp and resp.get("status") in ("persisted", "nothing_to_persist"):
                n = resp.get("tick_count", 0)
                st.success(f"✓ {n:,} ticks saved to HistoricStore. Switch the mode to Offline to analyse.")
            else:
                st.warning("Persist call failed — data remains in the live buffer until reset.")
            st.session_state.app_mode = "offline"
            st.rerun()

    with col_b:
        if st.button("🗑️ Reset & Clear Data", use_container_width=True,
                     key="banner_reset", type="secondary"):
            resp = _api_post("/stream/reset", {})
            if resp:
                st.session_state.live_ticks        = []
                st.session_state.live_events       = []
                st.session_state.last_tick_offset  = 0
                st.session_state.last_evt_offset   = 0
                st.session_state.live_status       = {}
                st.session_state.live_home_id      = None
                st.session_state.messages          = []
                st.rerun()

    with col_c:
        st.write("")  # keep listening label
        st.caption("Or start the producer again to stream another window.")


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ EnergyX")
    st.caption("Multi-agent home energy management")
    st.divider()

    # ── Mode toggle ───────────────────────────────────────────────────────────
    st.subheader("Mode")
    mode = st.radio(
        "Operating mode",
        options=["offline", "online"],
        format_func=lambda m: "📊 Offline Analysis" if m == "offline" else "🔴 Online (Live)",
        key="app_mode",
        label_visibility="collapsed",
    )

    st.divider()

    if mode == "online":
        # ── API connection ────────────────────────────────────────────────────
        st.subheader("Live Bus")
        api_url = st.text_input(
            "API URL", value=st.session_state.api_url,
            key="api_url_input",
            label_visibility="visible",
        )
        if api_url != st.session_state.api_url:
            st.session_state.api_url = api_url

        # Connection status
        health = _api_get("/health", base=api_url)
        if health:
            st.success("Connected", icon="🟢")
            srv_status = health
        else:
            st.error("Not reachable", icon="🔴")
            st.caption(f"Start:  `uvicorn energyx.api.main:app --port 8000`")
            srv_status = {}

        # Stream info
        live_s = st.session_state.live_status
        if live_s.get("is_active"):
            st.markdown(f'<span class="live-dot">●</span> **LIVE** — `{live_s.get("home_id","?")}`',
                        unsafe_allow_html=True)
            st.caption(f'Ticks: {live_s.get("tick_count",0):,}  ·  '
                       f'Events: {live_s.get("event_count",0):,}')
        elif live_s.get("is_complete"):
            st.markdown("✅ Stream complete")

        # Refresh rate
        st.divider()
        refresh_s = st.slider(
            "Refresh interval (s)", min_value=1, max_value=15, value=3, step=1,
            key="live_refresh",
        )

        st.divider()
        if st.button("🗑️ Clear session", use_container_width=True, key="clear_live"):
            _api_post("/stream/reset", {}, base=api_url)
            st.session_state.live_ticks       = []
            st.session_state.live_events      = []
            st.session_state.last_tick_offset = 0
            st.session_state.last_evt_offset  = 0
            st.session_state.live_status      = {}
            st.session_state.messages         = []
            st.rerun()

    else:
        # ── Offline home selector ─────────────────────────────────────────────
        st.subheader("Home")
        home_id = st.selectbox(
            "Select home",
            options=list(_HOMES.keys()),
            format_func=lambda h: _HOMES[h]["label"],
            key="home_id",
        )
        meta = _HOMES[home_id]
        struct_sb = _parse_home_structure(home_id)
        n_appl_sb = sum(len(v["appliances"]) for v in struct_sb["rooms"].values())
        st.caption(f"Data: {meta['start'].strftime('%d %b')} – "
                   f"{meta['end'].strftime('%d %b %Y')}"
                   f"\n{len(struct_sb['rooms'])} rooms · {n_appl_sb} appliances")

        st.divider()
        try:
            _ka    = _get_knowledge_agent()
            _t     = _ka.get_active_tariff(home_id)
            _rates = (_t.get("rates") or [{}])[0]
            st.metric("Unit rate",
                      f"{float(_rates.get('unit_rate_gbp_per_kwh', 0.2459))*100:.1f} p/kWh")
            st.metric("Standing charge",
                      f"{float(_t.get('standing_charge_gbp_per_day', 0.61))*100:.0f} p/day")
        except Exception:
            pass
        try:
            _ci  = _ka.fetch_carbon_intensity("national")
            _civ = _ci.get("intensity_gco2_per_kwh", "—")
            st.metric("Grid carbon", f"{_civ} gCO₂/kWh")
        except Exception:
            pass

        st.divider()
        if st.button("Clear chat", use_container_width=True, key="clear_chat_offline"):
            st.session_state.messages       = []
            st.session_state.monitor_events = None
            st.rerun()

    st.divider()
    st.caption(f"🤖 {_llm_label()}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════════

# Read mode from session (may have been changed by banner reset above)
mode = st.session_state.app_mode

if mode == "online":
    home_id = st.session_state.live_home_id or st.session_state.home_id
    refresh_s = st.session_state.get("live_refresh", 3)

    st.title("⚡ EnergyX — Live")
    st.caption(
        f'🔴 Online monitoring · home: **{home_id or "waiting…"}** · '
        f"bus: `{st.session_state.api_url}`"
    )

    # Session-complete banner (above tabs)
    _render_session_complete_banner()

    # ── How-to banner if nothing connected yet ────────────────────────────────
    if not st.session_state.live_ticks and not st.session_state.live_status.get("is_active"):
        with st.expander("▶  How to start a live session", expanded=True):
            st.markdown(f"""
**Step 1 — Start the data bus** (separate terminal):
```bash
uvicorn energyx.api.main:app --host 0.0.0.0 --port 8000
```

**Step 2 — Keep this dashboard open** (it will auto-refresh every {refresh_s}s)

**Step 3 — Start the producer** (another terminal):
```bash
python scripts/stream_ideal.py --home home96 --every 5 --speed 0.05
```

Options for the producer:
| Flag | Default | Meaning |
|------|---------|---------|
| `--home` | home96 | Which IDEAL home to stream |
| `--every N` | 5 | Keep every Nth tick (density) |
| `--speed S` | 0.05 | Seconds between posts (0.05 = 20/s) |
| `--start` | home default | Start date YYYY-MM-DD |
| `--end` | home default | End date YYYY-MM-DD |
""")

    # ── Four tabs (same structure as offline but live content) ────────────────
    tab_ov, tab_chat, tab_mon, tab_ctrl = st.tabs(
        ["📊 Overview", "💬 Chat", "🔔 Monitor", "🎛️ Control"]
    )

    with tab_ov:
        _render_live_overview(refresh_s)

    with tab_chat:
        _render_live_chat(home_id, mode)

    with tab_mon:
        _render_live_monitor(refresh_s)

    with tab_ctrl:
        st.subheader("Device control")
        st.caption("Advisory in live mode — commands are parsed but not dispatched "
                   "until a Home Assistant connection is configured.")
        st.info("Permissions: `auto_control:hvac` · `auto_control:laundry_ev` · "
                "`auto_control:emergency_off`", icon="🔑")

        qcols = st.columns(3)
        _QUICK = [
            ("Turn off washing machine", "🧺"),
            ("Set thermostat to 19°C",   "🌡️"),
            ("Defer dishwasher to 11pm", "🍽️"),
            ("Boost hot water",          "🔥"),
            ("Power off electric heater","❄️"),
            ("Cancel scheduled charge",  "🔌"),
        ]
        _pending: Optional[str] = None
        for i, (cmd, icon) in enumerate(_QUICK):
            if qcols[i % 3].button(f"{icon} {cmd}", key=f"lqcmd_{i}",
                                    use_container_width=True):
                _pending = cmd

        ctrl_txt = st.text_input("Or type a command", key="live_ctrl_txt",
                                  label_visibility="collapsed",
                                  placeholder="e.g. 'Set the living room to 20°C'")
        if st.button("Send", type="primary", key="live_ctrl_send"):
            _pending = ctrl_txt.strip() or None

        if _pending:
            live_home = st.session_state.live_home_id or st.session_state.home_id
            from energyx.agents.control.agent import ControlAgent
            orch = _get_orchestrator(live_home)
            ca   = ControlAgent(permission_manager=orch.permissions)
            with st.spinner(f"Parsing: *{_pending}*"):
                try:
                    res = ca.handle(_pending)
                except Exception as e:
                    res = {"error": str(e)}
            r1, r2, r3 = st.columns(3)
            r1.metric("Action", str(res.get("action",    res.get("intent",    "—"))))
            r2.metric("Entity", str(res.get("entity_id", res.get("device_id", "—"))))
            r3.metric("Status", str(res.get("status",    res.get("result",    "—"))))
            with st.expander("Full response"):
                st.json(res)

    st.stop()  # don't render offline content


# ═══════════════════════════════════════════════════════════════════════════════
# Offline mode — four tabs
# ═══════════════════════════════════════════════════════════════════════════════

home_id = st.session_state.home_id

st.title("⚡ EnergyX")
st.caption(f"📊 Offline analysis · {_HOMES[home_id]['label']}")

tab_overview, tab_chat, tab_monitor, tab_control = st.tabs(
    ["📊 Overview", "💬 Chat", "🔔 Monitor", "🎛️ Control"]
)


# ── Tab 1 — Overview ──────────────────────────────────────────────────────────
with tab_overview:
    with st.spinner("Loading home data…"):
        df = _load_home(home_id)

    if df.empty:
        st.warning("No data found. Run `scripts/ingest_ideal.py` first.")
        st.stop()

    elec = _elec_series(df)
    c1, c2, c3, c4 = st.columns(4)
    mean_w  = elec["watts"].mean()
    peak_w  = elec["watts"].max()
    n_days  = (elec["ts"].max() - elec["ts"].min()).total_seconds() / 86400
    kwh_tot = mean_w / 1000 * n_days * 24
    cost_e  = kwh_tot * 0.2459 + 0.61 * n_days
    c1.metric("Mean power",  f"{mean_w:.0f} W")
    c2.metric("Peak power",  f"{peak_w:.0f} W")
    c3.metric("Est. energy", f"{kwh_tot:.1f} kWh", f"{n_days:.0f}-day window")
    c4.metric("Est. cost",   f"£{cost_e:.2f}", "at cap rate")

    st.divider()
    st.subheader("Electricity — apparent power (hourly)")
    elec_hr = (elec.set_index("ts")["watts"].resample("1h").mean()
               .reset_index().rename(columns={"ts": "time", "watts": "W"}))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=elec_hr["time"], y=elec_hr["W"], mode="lines",
                             line=dict(color="#2563eb", width=1.5),
                             fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
    fig.add_hline(y=mean_w, line_dash="dot", line_color="gray",
                  annotation_text=f"mean {mean_w:.0f} W")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title="Watts", showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Sensor breakdown")
        df_num = df.copy()
        df_num["value"] = pd.to_numeric(df_num["value"], errors="coerce")
        bd = (df_num.groupby("sensor_type")["value"].agg(["count", "mean", "max"])
              .rename(columns={"count": "rows", "mean": "mean_val", "max": "max_val"})
              .reset_index())
        bd["mean_val"] = bd["mean_val"].round(2)
        bd["max_val"]  = bd["max_val"].round(2)
        st.dataframe(bd, use_container_width=True, hide_index=True)

    with col_r:
        st.subheader("Room temperatures (6-hour avg)")
        # Use hierarchy room sensor data when available
        tp = df[df["sensor_type"].isin(
            ("temperature", "temperature_probe", "temperature_room")
        )][["ts", "sensor_id", "value"]].copy()
        if not tp.empty:
            tp["ts"] = pd.to_datetime(tp["ts"])
            top3 = tp.groupby("sensor_id")["value"].count().nlargest(3).index.tolist()
            tp_hr = (tp[tp["sensor_id"].isin(top3)]
                     .pivot_table(index="ts", columns="sensor_id", values="value", aggfunc="mean")
                     .resample("6h").mean().reset_index())
            fig2 = go.Figure()
            for col in tp_hr.columns[1:]:
                room_label = str(col).replace("_temperature", "").replace("_", " ").title()
                fig2.add_trace(go.Scatter(x=tp_hr["ts"], y=tp_hr[col] / 10,
                                          mode="lines", name=room_label[:20],
                                          line=dict(width=1.2)))
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                                yaxis_title="°C", template="plotly_white",
                                legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No temperature data.")

    # ── Bill accumulation ─────────────────────────────────────────────────────
    st.subheader("Bill accumulation")
    UNIT_RATE  = 0.2459   # £/kWh
    STANDING_C = 0.61     # £/day
    _bill_elec = _elec_series(df).copy()
    if not _bill_elec.empty:
        _bill_elec["ts"] = pd.to_datetime(_bill_elec["ts"])
        _bill_elec = _bill_elec.set_index("ts").sort_index()

        # Hourly energy cost: mean_W × 1h / 1000 × unit_rate
        _hourly = _bill_elec["watts"].resample("1h").mean().dropna()
        _hourly_cost = _hourly / 1000 * UNIT_RATE
        # Add standing charge pro-rated per hour
        _hourly_cost += STANDING_C / 24
        _cum_cost = _hourly_cost.cumsum().reset_index()
        _cum_cost.columns = ["time", "£"]

        # Daily cost bars (for context)
        _daily_cost = _hourly_cost.resample("1D").sum().reset_index()
        _daily_cost.columns = ["date", "£/day"]

        bc1, bc2 = st.columns([2, 1])
        with bc1:
            fig_bill = go.Figure()
            fig_bill.add_trace(go.Scatter(
                x=_cum_cost["time"], y=_cum_cost["£"],
                mode="lines", name="Cumulative bill",
                fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
                line=dict(color="#2563eb", width=2),
            ))
            fig_bill.update_layout(
                height=260, template="plotly_white",
                yaxis_title="£ (inc. standing charge)",
                margin=dict(l=10, r=10, t=20, b=10),
                title="Cumulative electricity + standing charge",
            )
            st.plotly_chart(fig_bill, use_container_width=True)
        with bc2:
            fig_daily = go.Figure(go.Bar(
                x=_daily_cost["date"], y=_daily_cost["£/day"],
                marker_color="#f59e0b",
            ))
            fig_daily.update_layout(
                height=260, template="plotly_white",
                yaxis_title="£/day",
                margin=dict(l=10, r=10, t=20, b=10),
                title="Daily cost",
            )
            st.plotly_chart(fig_daily, use_container_width=True)


# ── Tab 2 — Chat ──────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask EnergyX anything")
    st.caption("Analysis queries show a plan for you to review and edit before execution.")

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"Hello! EnergyX assistant for **{_HOMES[home_id]['label']}**.\n\n"
                "- *Am I eligible for ECO4?*\n"
                "- *Forecast electricity for the next 24 hours and explain*\n"
                "- *Compare flat rate vs Economy 7*\n"
                "- *What's my budget status?*\n"
                "- *Turn off the washing machine*\n"
            ),
            "output_dir": None,
            "plan": None,
        })

    # Replay message history (including plots for past analysis results)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("output_dir"):
                _render_chat_plots(msg["output_dir"])
            if msg.get("plan") and msg["role"] == "assistant":
                _render_chat_plan(msg["plan"])

    # Human-in-the-loop: plan approval or budget phase 2
    if st.session_state.pending_analysis:
        _render_plan_approval(st.session_state.pending_analysis, mode, home_id)
    elif st.session_state.pending_budget:
        _render_budget_input(st.session_state.pending_budget, mode)
    else:
        user_input = st.chat_input("Ask about energy, tariffs, grants, forecasts…")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input, "output_dir": None, "plan": None})
            with st.chat_message("user"):
                st.markdown(user_input)

            df_chat = _load_home(home_id)
            with st.status("Planning…", expanded=False) as s:
                try:
                    plan_result = _chat_plan(user_input, home_id, df_chat)
                    s.update(label="Done", state="complete")
                except Exception as e:
                    plan_result = {"is_analysis": False, "result": {"intent": "error", "response": f"⚠️ {e}"}}
                    s.update(label="Error", state="error")

            if plan_result["is_analysis"]:
                # Store pending plan and rerun to show approval UI
                st.session_state.pending_analysis = plan_result["pending"]
                st.rerun()
            else:
                # Non-analysis result (knowledge, control, status) — show immediately
                result  = plan_result["result"]
                intent  = result.get("intent", "")
                response = result.get("response", "Done.")
                if intent == "control" and mode == "offline":
                    response = "**[Advisory — offline mode]** " + response
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "output_dir": None, "plan": None})
                st.rerun()


# ── Tab 3 — Monitor ───────────────────────────────────────────────────────────
with tab_monitor:
    meta_mon = _HOMES[home_id]
    struct   = _parse_home_structure(home_id)

    # ── Section A: Home Structure Diagram ─────────────────────────────────────
    st.subheader("🏠 Home structure")
    if struct["rooms"]:
        # Build a Plotly treemap: Home → Room → Sensor/Appliance
        labels, parents, values, colors = ["Home"], [""], [1], ["#1e40af"]
        for room, info in struct["rooms"].items():
            room_label = f"{_ROOM_ICONS.get(room, '🏠')} {room.replace('_', ' ').title()}"
            labels.append(room_label); parents.append("Home")
            values.append(max(1, len(info["sensors"]) + len(info["appliances"])))
            colors.append("#3b82f6")
            for s in info["sensors"]:
                labels.append(_SENSOR_LABELS.get(s, s.replace("_", " ").title()))
                parents.append(room_label); values.append(1); colors.append("#93c5fd")
            for a in info["appliances"]:
                labels.append(f"⚡ {_APPLIANCE_LABELS.get(a, a.title())}")
                parents.append(room_label); values.append(1); colors.append("#fbbf24")
        # Add home-level sensors
        for hl in struct["home_level"]:
            labels.append(_SENSOR_LABELS.get(hl, hl.replace("_", " ").title()))
            parents.append("Home"); values.append(1); colors.append("#6ee7b7")
        if struct["has_weather"]:
            labels.append("🌤️ Weather"); parents.append("Home")
            values.append(1); colors.append("#a78bfa")

        fig_tree = go.Figure(go.Treemap(
            labels=labels, parents=parents, values=values,
            marker=dict(colors=colors, line=dict(width=1.5, color="white")),
            textinfo="label",
            hovertemplate="<b>%{label}</b><extra></extra>",
        ))
        fig_tree.update_layout(height=360, margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(fig_tree, use_container_width=True)

        n_rooms = len(struct["rooms"])
        n_appl  = sum(len(v["appliances"]) for v in struct["rooms"].values())
        n_sens  = sum(len(v["sensors"]) for v in struct["rooms"].values()) + len(struct["home_level"])
        st.caption(f"{n_rooms} rooms · {n_appl} appliances · {n_sens} sensor types"
                   + (" · 🌤️ weather" if struct["has_weather"] else ""))
    else:
        st.info("No hierarchy data found. Build it with `scripts/build_ideal_hierarchy.py`.")

    st.divider()

    # ── Section B: Appliance Monitor ─────────────────────────────────────────
    st.subheader("⚡ Appliance monitor")

    # Build appliance options: "Room — Natural Name" → (room, appl_key)
    appl_options: Dict[str, tuple] = {}
    for room, info in struct["rooms"].items():
        for appl in info["appliances"]:
            room_nice = room.replace("_", " ").title()
            appl_nice = _APPLIANCE_LABELS.get(appl, appl.title())
            key       = f"{room_nice} — {appl_nice}"
            appl_options[key] = (room, appl)

    if not appl_options:
        st.info("No appliances found in hierarchy data.")
    else:
        sel_key = st.selectbox(
            "Select appliance",
            options=list(appl_options.keys()),
            key="mon_appl_sel",
        )
        sel_room, sel_appl = appl_options[sel_key]

        with st.spinner(f"Loading {sel_key}…"):
            appl_df = _load_appliance_series(
                home_id, sel_room, sel_appl, meta_mon["start"], meta_mon["end"]
            )

        if appl_df.empty:
            st.warning("No data for this appliance in the selected window.")
        else:
            # Also load home mains to compute % contribution
            df_for_pct = _load_home(home_id)
            mains_df   = df_for_pct[df_for_pct["sensor_type"].isin(
                ("electricity_apparent", "electricity_real", "mains")
            )][["ts", "value"]].drop_duplicates("ts").set_index("ts").sort_index()

            appl_series = appl_df.set_index("ts")["value"]
            mean_w      = float(appl_series.mean())
            total_kwh   = float(appl_series.mean() / 1000 * len(appl_series) / 60)  # 1-min samples
            n_days_appl = (appl_df["ts"].max() - appl_df["ts"].min()).total_seconds() / 86400

            # Predicted consumption = mean_W × remaining hours in data period
            hours_in_period = (meta_mon["end"] - meta_mon["start"]).total_seconds() / 3600
            pred_kwh = mean_w / 1000 * hours_in_period

            # % of home total electricity
            if not mains_df.empty:
                mains_kwh = float(mains_df["value"].mean() / 1000
                                  * len(mains_df) / (mains_df.index.to_series().diff().dt.total_seconds().median() / 60 or 1))
                pct_bill  = (total_kwh / mains_kwh * 100) if mains_kwh > 0 else 0.0
            else:
                mains_kwh = pct_bill = 0.0

            cost_appl = total_kwh * 0.2459

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean power",        f"{mean_w:.0f} W")
            m2.metric("Energy consumed",   f"{total_kwh:.2f} kWh",
                      f"est. £{cost_appl:.2f}")
            m3.metric("Predicted (period)", f"{pred_kwh:.1f} kWh")
            m4.metric("Share of home bill", f"{pct_bill:.1f}%",
                      help="Fraction of mains electricity attributed to this appliance")

            # Consumption chart (hourly)
            appl_hr = appl_series.resample("1h").mean().dropna()
            fig_appl = go.Figure()
            fig_appl.add_trace(go.Scatter(
                x=appl_hr.index, y=appl_hr.values, mode="lines",
                fill="tozeroy", fillcolor="rgba(251,191,36,0.15)",
                line=dict(color="#f59e0b", width=1.5),
                name="Power (W)",
            ))
            fig_appl.add_hline(y=mean_w, line_dash="dot", line_color="#94a3b8",
                               annotation_text=f"mean {mean_w:.0f} W")
            fig_appl.update_layout(
                height=260, template="plotly_white",
                yaxis_title="Watts", margin=dict(l=10, r=10, t=20, b=10),
                title=f"{sel_key} — hourly power",
            )
            st.plotly_chart(fig_appl, use_container_width=True)

            # Probe data (sink/bath/shower in same room)
            room_sensors = _load_room_sensors(
                home_id, sel_room, meta_mon["start"], meta_mon["end"]
            )
            probe_types = [s for s in room_sensors if s in ("sink", "bath", "shower",
                                                              "radiator_input", "radiator_output")]
            if probe_types:
                with st.expander("🔍 Room probes for this appliance", expanded=False):
                    for probe in probe_types:
                        s = room_sensors[probe].set_index("ts")["value"]
                        probe_label = _SENSOR_LABELS.get(probe, probe.title())
                        fig_probe = go.Figure(go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            line=dict(width=1.2, color="#06b6d4"), name=probe_label,
                        ))
                        fig_probe.update_layout(
                            height=180, template="plotly_white",
                            yaxis_title=probe_label,
                            margin=dict(l=10, r=10, t=20, b=5),
                            title=probe_label,
                        )
                        st.plotly_chart(fig_probe, use_container_width=True)

    st.divider()

    # ── Section C: Room Analysis ───────────────────────────────────────────────
    st.subheader("🏠 Room analysis")
    if struct["rooms"]:
        sel_room_r = st.selectbox(
            "Select room",
            options=list(struct["rooms"].keys()),
            format_func=lambda r: f"{_ROOM_ICONS.get(r, '🏠')} {r.replace('_', ' ').title()}",
            key="mon_room_sel",
        )
        with st.spinner(f"Loading room sensors…"):
            room_data = _load_room_sensors(
                home_id, sel_room_r, meta_mon["start"], meta_mon["end"]
            )

        if not room_data:
            st.info("No sensor data for this room in the selected window.")
        else:
            # Temperature & humidity side-by-side
            temp_s = room_data.get("temperature")
            hum_s  = room_data.get("humidity")
            rc1, rc2 = st.columns(2)
            if temp_s is not None:
                with rc1:
                    ts = temp_s.set_index("ts")["value"]
                    fig_t = go.Figure(go.Scatter(
                        x=ts.index, y=ts.values / 10, mode="lines",
                        line=dict(color="#ef4444", width=1.2),
                    ))
                    fig_t.update_layout(height=200, template="plotly_white",
                                        yaxis_title="°C", title="Temperature",
                                        margin=dict(l=5, r=5, t=30, b=5))
                    st.plotly_chart(fig_t, use_container_width=True)
            if hum_s is not None:
                with rc2:
                    hs = hum_s.set_index("ts")["value"]
                    fig_h = go.Figure(go.Scatter(
                        x=hs.index, y=hs.values, mode="lines",
                        line=dict(color="#3b82f6", width=1.2),
                    ))
                    fig_h.update_layout(height=200, template="plotly_white",
                                        yaxis_title="%", title="Humidity",
                                        margin=dict(l=5, r=5, t=30, b=5))
                    st.plotly_chart(fig_h, use_container_width=True)

            # Radiator delta (inlet − outlet = heat delivered)
            rad_in  = room_data.get("radiator_input")
            rad_out = room_data.get("radiator_output")
            if rad_in is not None and rad_out is not None:
                ri = rad_in.set_index("ts")["value"]
                ro = rad_out.set_index("ts")["value"]
                delta = (ri - ro).dropna() / 10  # convert ×10 encoding to °C
                fig_rad = go.Figure(go.Scatter(
                    x=delta.index, y=delta.values, mode="lines",
                    fill="tozeroy", fillcolor="rgba(249,115,22,0.1)",
                    line=dict(color="#f97316", width=1.2),
                ))
                fig_rad.update_layout(height=180, template="plotly_white",
                                      yaxis_title="ΔT (°C)",
                                      title="Radiator heat delivery (inlet − outlet)",
                                      margin=dict(l=5, r=5, t=30, b=5))
                st.plotly_chart(fig_rad, use_container_width=True)

    st.divider()

    # ── Section D: Weather ─────────────────────────────────────────────────────
    if struct["has_weather"]:
        with st.expander("🌤️ Weather conditions", expanded=False):
            wx = _load_weather(home_id, meta_mon["start"], meta_mon["end"])
            if not wx.empty:
                wc1, wc2 = st.columns(2)
                with wc1:
                    fig_wt = go.Figure(go.Scatter(
                        x=wx["ts"], y=wx["temp"], mode="lines",
                        line=dict(color="#f59e0b"), name="Temp (°C)",
                    ))
                    fig_wt.update_layout(height=200, template="plotly_white",
                                         yaxis_title="°C", title="Outdoor temperature",
                                         margin=dict(l=5, r=5, t=30, b=5))
                    st.plotly_chart(fig_wt, use_container_width=True)
                with wc2:
                    fig_wh = go.Figure()
                    fig_wh.add_trace(go.Scatter(x=wx["ts"], y=wx["rhum"],
                                                mode="lines", name="Humidity (%)",
                                                line=dict(color="#3b82f6")))
                    fig_wh.update_layout(height=200, template="plotly_white",
                                         yaxis_title="%", title="Outdoor humidity",
                                         margin=dict(l=5, r=5, t=30, b=5))
                    st.plotly_chart(fig_wh, use_container_width=True)

    st.divider()

    # ── Section E: Monitor Events (anomaly / cost / carbon) ───────────────────
    with st.expander("🔔 Monitor events (anomaly / cost / carbon)", expanded=False):
        if st.button("▶  Run monitors", type="primary", key="run_mon"):
            df_mon = _load_home(home_id)
            with st.spinner("Running…"):
                st.session_state.monitor_events = _run_monitors(df_mon, home_id)

        evs = st.session_state.monitor_events
        if evs is None:
            st.info("Click **Run monitors** to analyse the window.")
        else:
            total = sum(len(v) for v in evs.values())
            st.success(f"{total} total events across 5 monitors")
            _BADGE = {"anomaly": ("ev-anomaly","Anomaly"), "cost": ("ev-cost","Cost"),
                      "budget": ("ev-budget","Budget"),    "carbon": ("ev-carbon","Carbon"),
                      "appliance": ("ev-appliance","Appliance")}
            for mn, elist in evs.items():
                _, label = _BADGE.get(mn, ("ev-badge", mn))
                with st.expander(f"{label} — {len(elist)} events",
                                 expanded=(len(elist) > 0)):
                    if not elist:
                        st.write("No events.")
                    else:
                        rows = []
                        for e in elist:
                            row: Dict[str, Any] = {
                                "ts":       str(getattr(e, "ts", "?")),
                                "severity": str(getattr(e, "severity", "—")),
                                "value":    round(float(getattr(e, "value", 0) or 0), 2),
                            }
                            for attr in ("burn_rate_gbp_per_hour", "projected_spend_gbp",
                                         "intensity_gco2_per_kwh", "methods_agreed", "device_id"):
                                v = getattr(e, attr, None)
                                if v is not None:
                                    row[attr] = v
                            rows.append(row)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                     hide_index=True)


# ── Tab 4 — Control ───────────────────────────────────────────────────────────
with tab_control:
    st.subheader("Device control")
    st.caption("Advisory in offline mode. Switch to Online for live dispatch.")
    st.info("Permissions: `auto_control:hvac` · `auto_control:laundry_ev` · "
            "`auto_control:emergency_off`", icon="🔑")

    qcols = st.columns(3)
    _QUICK2 = [
        ("Turn off washing machine", "🧺"), ("Set thermostat to 19°C", "🌡️"),
        ("Defer dishwasher to 11pm", "🍽️"), ("Boost hot water",         "🔥"),
        ("Power off electric heater","❄️"), ("Cancel scheduled charge", "🔌"),
    ]
    _pending2: Optional[str] = None
    for i, (cmd, icon) in enumerate(_QUICK2):
        if qcols[i % 3].button(f"{icon} {cmd}", key=f"qcmd2_{i}", use_container_width=True):
            _pending2 = cmd

    st.divider()
    ctrl_in = st.text_input("Or type a command", label_visibility="collapsed",
                             placeholder="e.g. 'Set the living room to 20°C'",
                             key="ctrl_txt2")
    if st.button("Send command", type="primary", key="ctrl_send2"):
        _pending2 = ctrl_in.strip() or None

    if _pending2:
        from energyx.agents.control.agent import ControlAgent
        orch = _get_orchestrator(home_id)
        ca   = ControlAgent(permission_manager=orch.permissions)
        with st.spinner(f"Parsing: *{_pending2}*"):
            try:
                res = ca.handle(_pending2)
            except Exception as e:
                res = {"error": str(e)}
        r1, r2, r3 = st.columns(3)
        r1.metric("Action", str(res.get("action",    res.get("intent",    "—"))))
        r2.metric("Entity", str(res.get("entity_id", res.get("device_id", "—"))))
        r3.metric("Status", str(res.get("status",    res.get("result",    "—"))))
        with st.expander("Full response"):
            st.json(res)


# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"EnergyX · IDEAL dataset · {_llm_label()} · 2026")
