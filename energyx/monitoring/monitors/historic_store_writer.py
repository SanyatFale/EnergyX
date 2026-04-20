"""HistoricStoreWriter — persists every tick and event to the Historic Store."""

from __future__ import annotations

from typing import Any, Dict, List, Set

import pandas as pd

from energyx.data.events import BaseEvent
from energyx.monitoring.monitors.base import BaseMonitor


class HistoricStoreWriter(BaseMonitor):
    """Write ticks and events to the persistent Historic Store.

    This monitor has no detection logic — it's a persistence side-effect
    that runs on every tick.  Returns an empty event list always.

    Config keys:
        home_id (str): required
        store_path (str): override default store path
        batch_size (int): buffer ticks before flushing (default 100; 1 = flush every tick)
    """

    applicable_modes: Set[str] = {"online", "offline"}

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._batch_size = self._cfg("batch_size", 100)
        self._store_path = self._cfg("store_path", None)
        if "buffer" not in self._state:
            self._state["buffer"] = []

    def evaluate(self, tick_or_batch: Any) -> List[BaseEvent]:
        if isinstance(tick_or_batch, pd.DataFrame):
            self._flush_df(tick_or_batch)
        else:
            self._state["buffer"].append(tick_or_batch)
            if len(self._state["buffer"]) >= self._batch_size:
                self._flush_buffer()
        return []

    def flush(self):
        """Force-flush any buffered ticks (call on shutdown)."""
        self._flush_buffer()

    def _flush_buffer(self):
        buf: list = self._state.get("buffer", [])
        if not buf:
            return
        df = pd.DataFrame(buf)
        self._flush_df(df)
        self._state["buffer"] = []

    def _flush_df(self, df: pd.DataFrame):
        if df.empty or not self.home_id:
            return
        try:
            from energyx.data.historic_store import get_historic_store
            from pathlib import Path
            path = Path(self._store_path) if self._store_path else None
            store = get_historic_store(store_path=path)
            store.write_ticks(self.home_id, df)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"HistoricStoreWriter flush failed: {e}")
