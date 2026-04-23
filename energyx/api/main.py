"""EnergyX Live Data Bus — FastAPI application.

Architecture
------------
                    stream_ideal.py
                         │  POST /ingest/tick
                         ▼
                  ┌─────────────┐
                  │  FastAPI    │  owns LiveBuffer + OnlineRunner
                  │  (this app) │──── writes events ──► LiveBuffer
                  └─────────────┘
                         │  GET /live/*  (polling)
                         ▼
                  Streamlit dashboard

Lifecycle
---------
1. Producer calls  POST /stream/start   {"home_id": "home96"}
2. Producer sends  POST /ingest/tick    (many times)
3. Producer calls  POST /stream/complete
4. Dashboard polls GET  /live/status    until is_complete=True
5. Dashboard calls POST /stream/reset   to clear for next session

Start:
    uvicorn energyx.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    logger.warning("FastAPI not installed — install with: pip install fastapi uvicorn")

if _FASTAPI_OK:
    from energyx.api.live_buffer import LiveBuffer

    # ── Singleton services (owned by this process) ────────────────────────────
    _buffer: LiveBuffer = LiveBuffer()
    _runner = None   # OnlineRunner, created lazily when stream starts

    # ── App ───────────────────────────────────────────────────────────────────
    app = FastAPI(
        title="EnergyX Live Data Bus",
        version="1.0.0",
        description="Real-time tick ingestion and live event polling for EnergyX.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request / response models ─────────────────────────────────────────────

    class SensorReading(BaseModel):
        sensor_id:   str
        sensor_type: str
        value:       float
        unit:        str = "watts"

    class TickPayload(BaseModel):
        home_id:  str = Field(..., description="IDEAL home id, e.g. home96")
        ts:       str = Field(..., description="UTC timestamp YYYY-MM-DD HH:MM:SS")
        readings: List[SensorReading] = Field(..., min_items=1)
        metadata: Optional[Dict[str, Any]] = None

        @validator("ts")
        def _check_ts(cls, v: str) -> str:
            import re
            if not re.match(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", v):
                raise ValueError("ts must be YYYY-MM-DD HH:MM:SS")
            return v

    class BatchPayload(BaseModel):
        ticks: List[TickPayload] = Field(..., max_items=500)

    class StartPayload(BaseModel):
        home_id: str

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_runner(home_id: str):
        global _runner
        if _runner is None:
            from energyx.monitoring.online_runner import OnlineRunner
            _runner = OnlineRunner(config={"home_id": home_id})
            logger.info(f"OnlineRunner started for {home_id}")

    def _process_tick(tick: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run tick through OnlineRunner, add to buffer, return events."""
        _buffer.add_tick(tick)
        events: list = []
        if _runner is not None:
            try:
                events = _runner.process_tick(tick)
            except Exception as e:
                logger.debug(f"Monitor error on tick: {e}")
        _buffer.add_events(events)
        return events

    # ── Endpoints ─────────────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": "1.0.0", **_buffer.status()}

    # ── Stream lifecycle ──────────────────────────────────────────────────────

    @app.post("/stream/start", status_code=200)
    def stream_start(body: StartPayload) -> Dict[str, Any]:
        """Initialise a new live session.  Clears any existing buffer."""
        global _runner
        _runner = None  # force fresh OnlineRunner for this home
        _buffer.start(body.home_id)
        _ensure_runner(body.home_id)
        logger.info(f"Stream started: {body.home_id}")
        return {"status": "started", "home_id": body.home_id}

    @app.post("/stream/complete", status_code=200)
    def stream_complete() -> Dict[str, Any]:
        """Producer signals that all ticks have been sent."""
        _buffer.complete()
        logger.info("Stream marked complete")
        return {"status": "complete", **_buffer.status()}

    @app.post("/stream/reset", status_code=200)
    def stream_reset() -> Dict[str, Any]:
        """Clear the buffer and stop the current session."""
        global _runner
        _runner = None
        _buffer.reset()
        logger.info("Stream reset")
        return {"status": "reset"}

    @app.post("/stream/persist", status_code=200)
    def stream_persist() -> Dict[str, Any]:
        """Persist all buffered ticks to the HistoricStore for offline analysis.

        Called by the dashboard when the user clicks 'Switch to Offline Analysis'
        after the producer finishes.  Safe to call multiple times — ParquetBackend
        merges into existing partitions.
        """
        status = _buffer.status()
        home_id = status.get("home_id")
        if not home_id:
            raise HTTPException(400, detail="No home_id in buffer — was a session started?")

        ticks = _buffer.get_ticks_since(0)
        if not ticks:
            return {"status": "nothing_to_persist", "tick_count": 0}

        try:
            import pandas as pd
            from energyx.data.historic_store import HistoricStore, ParquetBackend
            from pathlib import Path

            store_path = Path(__file__).parent.parent.parent / "data" / "historic_store"
            store = HistoricStore(backend=ParquetBackend(store_path))

            df = pd.DataFrame(ticks)
            df["ts"] = pd.to_datetime(df["ts"])
            store.write_ticks(home_id, df)

            logger.info(f"Persisted {len(df)} ticks for {home_id} to HistoricStore")
            return {
                "status": "persisted",
                "home_id": home_id,
                "tick_count": len(df),
            }
        except Exception as e:
            logger.error(f"stream_persist failed: {e}")
            raise HTTPException(500, detail=f"Persist failed: {e}")

    # ── Tick ingestion ────────────────────────────────────────────────────────

    @app.post("/ingest/tick", status_code=202)
    def ingest_tick(payload: TickPayload) -> Dict[str, Any]:
        """Receive a single aggregated tick (all sensors at one timestamp)."""
        if not _buffer.status()["is_active"]:
            raise HTTPException(
                409,
                detail="No active stream. Call POST /stream/start first.",
            )
        n_events = 0
        for reading in payload.readings:
            tick = {
                "home_id":    payload.home_id,
                "ts":         payload.ts,
                "sensor_id":  reading.sensor_id,
                "sensor_type": reading.sensor_type,
                "value":      reading.value,
                "unit":       reading.unit,
            }
            events = _process_tick(tick)
            n_events += len(events)
        return {
            "status":    "accepted",
            "n_readings": len(payload.readings),
            "n_events":  n_events,
            "tick_total": _buffer.tick_count(),
        }

    @app.post("/ingest/batch", status_code=202)
    def ingest_batch(payload: BatchPayload) -> Dict[str, Any]:
        """Receive up to 500 ticks in a single request."""
        if not _buffer.status()["is_active"]:
            raise HTTPException(409, detail="No active stream.")
        accepted = 0
        errors   = 0
        for tp in payload.ticks:
            for reading in tp.readings:
                tick = {
                    "home_id":    tp.home_id,
                    "ts":         tp.ts,
                    "sensor_id":  reading.sensor_id,
                    "sensor_type": reading.sensor_type,
                    "value":      reading.value,
                    "unit":       reading.unit,
                }
                try:
                    _process_tick(tick)
                    accepted += 1
                except Exception as e:
                    logger.debug(f"Batch tick error: {e}")
                    errors += 1
        return {"status": "accepted", "accepted": accepted, "errors": errors,
                "tick_total": _buffer.tick_count()}

    # ── Live polling endpoints (Streamlit polls these) ────────────────────────

    @app.get("/live/status")
    def live_status() -> Dict[str, Any]:
        return _buffer.status()

    @app.get("/live/ticks")
    def live_ticks(since: int = Query(0, ge=0)) -> Dict[str, Any]:
        """Return ticks from index `since` onwards."""
        ticks = _buffer.get_ticks_since(since)
        return {
            "total": _buffer.tick_count(),
            "since": since,
            "count": len(ticks),
            "ticks": ticks,
        }

    @app.get("/live/events")
    def live_events(since: int = Query(0, ge=0)) -> Dict[str, Any]:
        """Return monitoring events from index `since` onwards."""
        events = _buffer.get_events_since(since)
        return {
            "total":  _buffer.event_count(),
            "since":  since,
            "count":  len(events),
            "events": events,
        }

else:
    # Stub for environments without FastAPI installed
    class _StubApp:
        def get(self, *a, **kw):    return lambda f: f
        def post(self, *a, **kw):   return lambda f: f
        def add_middleware(self, *a, **kw): pass

    app = _StubApp()
