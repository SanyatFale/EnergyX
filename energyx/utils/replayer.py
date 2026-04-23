"""DataReplayer — replays historical IDEAL ticks through the OnlineRunner.

Used by the Streamlit UI to simulate what the live monitoring system
would have caught over a past time window.  Runs synchronously so
Streamlit can track progress with a normal progress bar.

Example::

    replayer = DataReplayer(home_id="home96", config={"home_id": "home96"})
    result = replayer.replay(
        store=store,
        start=datetime(2017, 9, 1),
        end=datetime(2017, 9, 2),
        sensor_filter={"electricity_apparent", "appliance_power"},
        progress_callback=lambda p: progress_bar.progress(p),
    )
    events = result["events"]
    tick_df = result["tick_df"]
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd

from energyx.data.events import BaseEvent

logger = logging.getLogger(__name__)

# Sensor types the monitors actually care about — skip others to stay fast
_DEFAULT_SENSOR_FILTER: Set[str] = {"electricity_apparent", "appliance_power"}


class DataReplayer:
    """Replays a historical tick window through OnlineRunner monitors.

    Reads from HistoricStore, feeds ticks one-by-one to OnlineRunner in
    timestamp order, and collects all fired events.  This gives an exact
    reconstruction of what live monitoring would have produced.

    Args:
        home_id: Home identifier (must exist in the store).
        config:  Config dict forwarded to OnlineRunner (and its monitors).
    """

    def __init__(self, home_id: str, config: Optional[Dict[str, Any]] = None):
        self.home_id = home_id
        self.config = config or {"home_id": home_id}
        if "home_id" not in self.config:
            self.config["home_id"] = home_id

    # ------------------------------------------------------------------

    def replay(
        self,
        store: Any,
        start: datetime,
        end: datetime,
        sensor_filter: Optional[Set[str]] = None,
        sample_every: int = 1,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """Replay ticks in [start, end] through all online monitors.

        Args:
            store:            HistoricStore instance.
            start / end:      Inclusive time window.
            sensor_filter:    Only feed these sensor_type values.  Pass None
                              to replay every sensor type (slower).
            sample_every:     Feed every Nth tick (1 = all, 5 = 20 % etc.).
                              Monitors maintain rolling windows so mild
                              sampling doesn't break anomaly detection.
            progress_callback: Called with float 0–1 as replay progresses.

        Returns:
            {
                "ticks_processed": int,
                "ticks_total":     int,     # before sampling
                "events":          List[BaseEvent],
                "elapsed_seconds": float,
                "tick_df":         pd.DataFrame,  # all ticks (for charting)
            }
        """
        from energyx.monitoring.online_runner import OnlineRunner

        runner = OnlineRunner(config=self.config)

        df = store.read_ticks(self.home_id, start=start, end=end)
        if df.empty:
            logger.warning(f"DataReplayer: no ticks for {self.home_id} in window")
            return {
                "ticks_processed": 0,
                "ticks_total": 0,
                "events": [],
                "elapsed_seconds": 0.0,
                "tick_df": df,
            }

        df = df.sort_values("ts").reset_index(drop=True)

        # Filter to sensors the monitors care about
        if sensor_filter is None:
            sensor_filter = _DEFAULT_SENSOR_FILTER
        mask = df["sensor_type"].isin(sensor_filter)
        filtered = df[mask].reset_index(drop=True)

        # Sample
        if sample_every > 1:
            filtered = filtered.iloc[::sample_every].reset_index(drop=True)

        n = len(filtered)
        all_events: List[BaseEvent] = []
        t0 = time.perf_counter()

        for i, (_, row) in enumerate(filtered.iterrows()):
            tick = row.to_dict()
            # Ensure ts is a plain string (some monitors stringify it)
            ts = tick.get("ts")
            if hasattr(ts, "isoformat"):
                tick["ts"] = ts.isoformat()

            try:
                events = runner.process_tick(tick)
                all_events.extend(events)
            except Exception as e:
                logger.debug(f"DataReplayer tick {i} failed: {e}")

            if progress_callback and (i % max(1, n // 200) == 0):
                progress_callback(i / n)

        if progress_callback:
            progress_callback(1.0)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"DataReplayer: {n} ticks → {len(all_events)} events "
            f"in {elapsed:.1f}s ({self.home_id})"
        )

        return {
            "ticks_processed": n,
            "ticks_total": len(df),
            "events": all_events,
            "elapsed_seconds": elapsed,
            "tick_df": df,  # full (unsampled) for charting
        }
