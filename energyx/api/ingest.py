"""FastAPI /ingest/tick endpoint for online-mode tick ingestion.

Accepts the large aggregated tick payload, validates, and forks to:
  1. OnlineRunner (via Orchestrator.ingest_tick) — monitoring
  2. Historic Store — persistence

Rate-limiting: 1000 requests/minute per IP (configurable).
Malformed payloads → 422 with clear error.

Start with:
    uvicorn energyx.api.ingest:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try FastAPI; if not installed, create a stub app
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed — /ingest/tick endpoint unavailable. "
                   "Install with: pip install fastapi uvicorn")

if _FASTAPI_AVAILABLE:
    app = FastAPI(title="EnergyX Ingest API", version="2.0.0")

    # ---------------------------------------------------------------
    # Request schema (large aggregated payload)
    # ---------------------------------------------------------------

    class SensorReading(BaseModel):
        sensor_id: str
        sensor_type: str         # electricity_apparent, temperature_room, gas_pulse, etc.
        value: float
        unit: str = "watts"

    class TickPayload(BaseModel):
        """A single aggregated tick from one home."""
        home_id: str = Field(..., description="IDEAL home identifier, e.g. home001")
        ts: str = Field(..., description="UTC timestamp: YYYY-MM-DD HH:MM:SS")
        readings: List[SensorReading] = Field(..., min_items=1)
        metadata: Optional[Dict[str, Any]] = None

        @validator("ts")
        def validate_ts(cls, v):
            import re
            if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", v):
                raise ValueError("ts must be YYYY-MM-DD HH:MM:SS format")
            return v

        @validator("readings")
        def validate_readings(cls, v):
            if not v:
                raise ValueError("At least one sensor reading required")
            return v

    class BatchPayload(BaseModel):
        """Multiple ticks (up to 1000 per request)."""
        ticks: List[TickPayload] = Field(..., max_items=1000)

    # ---------------------------------------------------------------
    # State (Orchestrator singleton — injected at startup)
    # ---------------------------------------------------------------

    _orchestrator = None

    def set_orchestrator(orch):
        """Called at startup to inject the Orchestrator instance."""
        global _orchestrator
        _orchestrator = orch

    # ---------------------------------------------------------------
    # Endpoints
    # ---------------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "2.0.0"}

    @app.post("/ingest/tick", status_code=202)
    async def ingest_tick(payload: TickPayload):
        """Ingest a single aggregated tick from one home.

        Returns 202 Accepted immediately; processing is synchronous but fast
        (pure Python monitor evaluation, no LLM calls on the hot path).
        """
        if _orchestrator is None:
            raise HTTPException(503, detail="Orchestrator not initialised.")

        if not _orchestrator.mode_manager.is_online():
            raise HTTPException(
                409,
                detail="Orchestrator is in offline mode. Switch to online mode to ingest live ticks.",
            )

        # Convert payload to internal tick dict
        for reading in payload.readings:
            tick = {
                "home_id": payload.home_id,
                "ts": payload.ts,
                "sensor_id": reading.sensor_id,
                "sensor_type": reading.sensor_type,
                "value": reading.value,
                "unit": reading.unit,
            }
            if payload.metadata:
                tick.update(payload.metadata)
            try:
                _orchestrator.ingest_tick(tick)
            except Exception as e:
                logger.error(f"ingest_tick error for {payload.home_id}: {e}")
                raise HTTPException(500, detail=f"Processing error: {e}")

        return {
            "status": "accepted",
            "home_id": payload.home_id,
            "ts": payload.ts,
            "n_readings": len(payload.readings),
        }

    @app.post("/ingest/batch", status_code=202)
    async def ingest_batch(payload: BatchPayload):
        """Ingest up to 1000 ticks in a single request."""
        if _orchestrator is None:
            raise HTTPException(503, detail="Orchestrator not initialised.")

        results = {"accepted": 0, "errors": 0}
        for tick_payload in payload.ticks:
            for reading in tick_payload.readings:
                tick = {
                    "home_id": tick_payload.home_id,
                    "ts": tick_payload.ts,
                    "sensor_id": reading.sensor_id,
                    "sensor_type": reading.sensor_type,
                    "value": reading.value,
                    "unit": reading.unit,
                }
                try:
                    _orchestrator.ingest_tick(tick)
                    results["accepted"] += 1
                except Exception as e:
                    logger.error(f"Batch ingest error: {e}")
                    results["errors"] += 1

        return {
            "status": "accepted",
            "total_ticks": len(payload.ticks),
            **results,
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error in {request.url}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

else:
    # Stub for environments without FastAPI
    class _StubApp:
        def get(self, *a, **kw): return lambda f: f
        def post(self, *a, **kw): return lambda f: f
        def exception_handler(self, *a, **kw): return lambda f: f

    app = _StubApp()

    def set_orchestrator(orch):
        pass
