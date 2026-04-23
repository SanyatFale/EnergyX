"""LiveBuffer — thread-safe in-memory store for the real-time data bus.

The FastAPI process owns one singleton LiveBuffer.  Every incoming tick is
appended here AND forwarded to the OnlineRunner.  The Streamlit process polls
the REST endpoints that read from this buffer.

Design:
  - Pure Python list + threading.Lock (no external dependencies).
  - Ticks and events are stored as plain dicts (JSON-serialisable).
  - Callers request slices by integer offset so they can resume polling
    without re-receiving old data.
  - Session lifecycle: start → [ticks...] → complete → (reset by user)
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class LiveBuffer:
    """Thread-safe accumulator for live ticks and monitoring events."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._ticks:  List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []

        self._home_id:    Optional[str] = None
        self._is_active:  bool = False
        self._is_complete: bool = False
        self._started_at: Optional[str] = None
        self._completed_at: Optional[str] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start(self, home_id: str) -> None:
        with self._lock:
            self._ticks.clear()
            self._events.clear()
            self._home_id     = home_id
            self._is_active   = True
            self._is_complete = False
            self._started_at  = datetime.now(timezone.utc).isoformat()
            self._completed_at = None

    def complete(self) -> None:
        with self._lock:
            self._is_active   = False
            self._is_complete = True
            self._completed_at = datetime.now(timezone.utc).isoformat()

    def reset(self) -> None:
        with self._lock:
            self._ticks.clear()
            self._events.clear()
            self._home_id     = None
            self._is_active   = False
            self._is_complete = False
            self._started_at  = None
            self._completed_at = None

    # ------------------------------------------------------------------
    # Writes (called from the ingest endpoint, hot path)
    # ------------------------------------------------------------------

    def add_tick(self, tick: Dict[str, Any]) -> None:
        with self._lock:
            self._ticks.append(tick)

    def add_events(self, events: List[Any]) -> None:
        """Accept BaseEvent objects or plain dicts."""
        if not events:
            return
        serialised = []
        for e in events:
            if isinstance(e, dict):
                serialised.append(e)
            else:
                # BaseEvent dataclass → dict
                try:
                    serialised.append(e.to_dict() if hasattr(e, "to_dict") else vars(e))
                except Exception:
                    serialised.append({"type": type(e).__name__, "raw": str(e)})
        with self._lock:
            self._events.extend(serialised)

    # ------------------------------------------------------------------
    # Reads (called from polling endpoints, not hot path)
    # ------------------------------------------------------------------

    def get_ticks_since(self, offset: int) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._ticks[offset:])

    def get_events_since(self, offset: int) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events[offset:])

    def tick_count(self) -> int:
        with self._lock:
            return len(self._ticks)

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    # ------------------------------------------------------------------
    # Status snapshot (JSON-safe)
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "home_id":       self._home_id,
                "is_active":     self._is_active,
                "is_complete":   self._is_complete,
                "tick_count":    len(self._ticks),
                "event_count":   len(self._events),
                "started_at":    self._started_at,
                "completed_at":  self._completed_at,
            }
