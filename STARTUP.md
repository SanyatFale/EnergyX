# EnergyX — Startup & Operations Guide

Practical reference for booting, refreshing, and using the system in every mode. Read this when you come back after a break and can't remember what to start first.

---

## One-Time Setup

```bash
cd ~/Documents/GitHub/EnergyX

# Create venv and install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Copy env template and fill in LLM settings
cp .env.example .env
# Edit .env — default is ollama/llama3.2:3b (local, free)
```

### IDEAL Data Ingestion (run once per home)

The raw IDEAL dataset must be converted to Parquet before the system can use it.

```bash
# Ingest the three pre-configured homes
python scripts/ingest_ideal.py --home home96  --start 2017-09-01 --end 2017-09-14
python scripts/ingest_ideal.py --home home128 --start 2017-10-01 --end 2017-10-14
python scripts/ingest_ideal.py --home home62  --start 2017-03-01 --end 2017-03-14

# Verify
python -c "
from energyx.data.historic_store import HistoricStore, ParquetBackend
from pathlib import Path
s = HistoricStore(backend=ParquetBackend(Path('data/historic_store')))
df = s.read_ticks('home96')
print(len(df), 'ticks loaded for home96')
"
```

### RAG Index Build (run once, or after adding corpus docs)

```bash
python knowledge/ingestion/build_index.py
```

This builds `knowledge/index/bm25_index.pkl` and `knowledge/index/chroma_store/`. The index ships pre-built in the repo — only rebuild if you add new corpus documents.

---

## Mode 1 — Offline (Historical Batch Analysis)

**What it is:** Load stored IDEAL data for a home + date window and ask questions. No external processes needed.

```bash
# Terminal 1 (only terminal needed)
source venv/bin/activate
streamlit run app.py
```

In the sidebar:
1. Mode: **Offline**
2. Home: `home96` / `home128` / `home62`
3. Date range: pick from the home's available window
4. Click **Load Data**

Chat tab examples:
```
forecast next 7 days
detect anomalies and explain the top 5
what is driving my consumption this week?
what if temperature drops 5 degrees — how does my bill change?
am I eligible for ECO4?
predict my bill for this period
```

Monitor tab shows the offline monitoring run — anomaly counts, event timeline, budget trajectory.

**To refresh data** (load a different home or date window): change sidebar selections and click Load Data again.

---

## Mode 2 — Online (Live Streaming Dashboard)

**What it is:** A live dashboard that updates in real time as a data producer streams IDEAL ticks. Mimics a real smart meter or IoT feed. Requires three processes.

### Step 1 — Start the data bus

```bash
# Terminal 1
source venv/bin/activate
uvicorn energyx.api.main:app --host 0.0.0.0 --port 8000

# Verify
curl http://localhost:8000/health
# Expected: {"status": "ok", "tick_count": 0, "is_active": false, ...}
```

### Step 2 — Start the dashboard

```bash
# Terminal 2
source venv/bin/activate
streamlit run app.py
```

In the sidebar:
1. Mode: **Online**
2. API URL: `http://localhost:8000` (default — change if bus is on another host/port)
3. Click **Check Connection** — should show green "Connected"

The dashboard is now listening. It shows a how-to expander and waits for data.

### Step 3 — Start the data producer

```bash
# Terminal 3
source venv/bin/activate

# Basic: stream home96, full 14-day window, 20% density, 20 ticks/sec
python scripts/stream_ideal.py --home home96

# Custom: 1-day window, every tick, as fast as possible
python scripts/stream_ideal.py \
  --home home96 \
  --start 2017-09-01 \
  --end 2017-09-02 \
  --every 1 \
  --speed 0

# Slower, more realistic: 5 ticks/sec, 10% density
python scripts/stream_ideal.py \
  --home home96 \
  --start 2017-09-01 \
  --end 2017-09-07 \
  --every 10 \
  --speed 0.2
```

**Producer arguments:**

| Arg | Default | Meaning |
|---|---|---|
| `--home` | `home96` | IDEAL home ID (`home96`, `home128`, `home62`) |
| `--start` | home's default | Start date `YYYY-MM-DD` |
| `--end` | home's default | End date `YYYY-MM-DD` |
| `--every N` | `5` | Keep every Nth tick (1=all, 10=10% density) |
| `--speed S` | `0.05` | Seconds between POSTs (0 = unlimited) |
| `--host` | `http://localhost:8000` | FastAPI base URL |
| `--dry-run` | off | Print first 3 payloads, don't post |

### What happens while streaming

The dashboard overview tab auto-refreshes (configurable 2–10 s) showing:
- Live electricity chart with event markers
- KPI row: tick count, event count, total kWh, current burn rate
- Sensor type summary table

The monitor tab shows a live event feed: anomalies, carbon alerts, cost updates, habit drift, forgot-to-turn-off.

The chat tab works on accumulated data — ask questions at any point during streaming.

### When the producer finishes

A banner appears with three options:

| Option | What it does |
|---|---|
| **Switch to Offline Analysis** | Calls `POST /stream/persist`, saves buffer to HistoricStore, reloads in Offline mode |
| **Reset & Clear** | Calls `POST /stream/reset`, clears buffer, ready for a new session |
| **Keep Watching** | Dismisses banner, dashboard continues showing static data |

### Stopping and restarting

```bash
# Stop the producer: Ctrl-C in Terminal 3

# Reset the bus for a new session (also available via dashboard banner)
curl -X POST http://localhost:8000/stream/reset

# Check bus state at any time
curl http://localhost:8000/live/status
```

---

## Useful Commands

### Check bus health and state

```bash
curl http://localhost:8000/health | python3 -m json.tool
curl http://localhost:8000/live/status | python3 -m json.tool
```

### Inspect buffered ticks / events

```bash
# Last 10 ticks
curl "http://localhost:8000/live/ticks?since=0" | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('total:', d['total'])
for t in d['ticks'][-3:]: print(t['ts'], t['sensor_type'], t['value'])
"

# Event type breakdown
curl "http://localhost:8000/live/events?since=0" | python3 -c "
from collections import Counter; import json,sys
d=json.load(sys.stdin)
print(Counter(e['type'] for e in d['events']).most_common())
"
```

### Dry-run a stream (no server needed)

```bash
python scripts/stream_ideal.py --home home96 --start 2017-09-01 --end 2017-09-02 --dry-run
```

### Persist current buffer to HistoricStore manually

```bash
curl -X POST http://localhost:8000/stream/persist | python3 -m json.tool
```

### Verify persisted data

```bash
venv/bin/python -c "
from energyx.data.historic_store import HistoricStore, ParquetBackend
from pathlib import Path; from datetime import datetime
s = HistoricStore(backend=ParquetBackend(Path('data/historic_store')))
df = s.read_ticks('home96', start=datetime(2017,9,1), end=datetime(2017,9,2))
print(len(df), 'rows,', df['sensor_type'].nunique(), 'sensor types')
"
```

### Run tests

```bash
source venv/bin/activate
pytest tests/ -v                              # all tests
pytest tests/test_knowledge_agent.py -v      # knowledge layer only
pytest tests/test_monitors.py -v             # monitoring monitors
pytest tests/test_online_runner.py -v        # online runner
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: fastapi` | `pip install fastapi "uvicorn[standard]"` |
| `Cannot reach http://localhost:8000/health` | Start the bus first: `uvicorn energyx.api.main:app --port 8000` |
| `No active stream. Call POST /stream/start first.` | Producer calls `/stream/start` automatically — check producer is running |
| Dashboard shows "no data yet" | Producer hasn't started yet — check Terminal 3 |
| Streamlit fragment auto-refresh not working | Requires Streamlit ≥ 1.55 — run `pip install --upgrade streamlit` |
| Ollama model not found | `ollama pull llama3.2:3b` then restart Ollama |
| `store.read_ticks` returns empty DataFrame | Run `ingest_ideal.py` for that home first |
| BM25 index missing | Run `python knowledge/ingestion/build_index.py` |

---

## Process Map Summary

| Process | Command | Required for |
|---|---|---|
| Ollama | `ollama serve` | Chat/analysis in both modes |
| FastAPI bus | `uvicorn energyx.api.main:app --port 8000` | Online mode only |
| Streamlit app | `streamlit run app.py` | Both modes |
| Data producer | `python scripts/stream_ideal.py --home home96` | Online mode only |
