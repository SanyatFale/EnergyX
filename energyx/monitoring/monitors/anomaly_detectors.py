"""AnomalyDetectors monitor.

Wraps the 7-method ensemble from tinyts/tools/anomaly.py as a monitor class.
Works on both single ticks (online, rolling window) and batch DataFrames (offline).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from energyx.data.events import AnomalyApplianceEvent, Severity
from energyx.monitoring.monitors.base import BaseMonitor

logger = logging.getLogger(__name__)


class AnomalyDetectors(BaseMonitor):
    """7-method anomaly detection ensemble.

    Online:  maintains a rolling window of recent values; fires when majority
             of methods agree.
    Offline: runs over a full batch DataFrame.

    Config keys:
        window_size (int): rolling window length (default 120)
        min_votes (int): minimum method agreement to flag (default 4)
        z_threshold (float): default 3.0
        rolling_z_threshold (float): default 2.5
        device_id (str): sensor / appliance identifier
        value_column (str): column name in batch DataFrames (default "value")
    """

    applicable_modes: Set[str] = {"online", "offline"}

    def __init__(self, config=None, state_store=None):
        super().__init__(config, state_store)
        self._window_size = self._cfg("window_size", 120)
        self._min_votes = self._cfg("min_votes", 4)
        self._z_thr = self._cfg("z_threshold", 3.0)
        self._roll_z_thr = self._cfg("rolling_z_threshold", 2.5)
        self._device_id = self._cfg("device_id", "unknown")
        self._value_col = self._cfg("value_column", "value")

        # Rolling window for online mode
        if "window" not in self._state:
            self._state["window"] = deque(maxlen=self._window_size)

    # ------------------------------------------------------------------
    def evaluate(self, tick_or_batch: Any) -> List[AnomalyApplianceEvent]:
        if isinstance(tick_or_batch, pd.DataFrame):
            return self._eval_batch(tick_or_batch)
        return self._eval_tick(tick_or_batch)

    # ------------------------------------------------------------------
    def _eval_tick(self, tick: Dict[str, Any]) -> List[AnomalyApplianceEvent]:
        val = tick.get(self._value_col, tick.get("value"))
        if val is None:
            return []
        try:
            val = float(val)
        except (TypeError, ValueError):
            return []

        window: deque = self._state["window"]
        window.append(val)
        if len(window) < max(10, self._window_size // 4):
            return []

        arr = np.array(window, dtype=float)
        votes = self._run_methods(arr)
        last_idx = len(arr) - 1
        if votes[last_idx] >= self._min_votes:
            return [AnomalyApplianceEvent(
                device_id=self._device_id,
                metric="ensemble",
                value=val,
                threshold=float(self._min_votes),
                context_window=list(arr[-5:]),
                methods_agreed=int(votes[last_idx]),
                severity=Severity.WARN if votes[last_idx] < 6 else Severity.CRITICAL,
                **self._make_event_kwargs(),
            )]
        return []

    def _eval_batch(self, df: pd.DataFrame) -> List[AnomalyApplianceEvent]:
        if df.empty or self._value_col not in df.columns:
            return []
        arr = df[self._value_col].values.astype(float)
        votes = self._run_methods(arr)
        events = []
        for idx in np.where(votes >= self._min_votes)[0]:
            window = arr[max(0, idx-2):idx+3].tolist()
            events.append(AnomalyApplianceEvent(
                device_id=self._device_id,
                metric="ensemble",
                value=float(arr[idx]),
                threshold=float(self._min_votes),
                context_window=window,
                methods_agreed=int(votes[idx]),
                severity=Severity.WARN,
                **self._make_event_kwargs(),
            ))
        return events

    # ------------------------------------------------------------------
    def _run_methods(self, arr: np.ndarray) -> np.ndarray:
        """Run all 7 methods; return per-index vote counts."""
        n = len(arr)
        votes = np.zeros(n, dtype=int)

        # 1. Z-score
        mu, sd = np.mean(arr), np.std(arr) + 1e-9
        votes += (np.abs(arr - mu) / sd > self._z_thr).astype(int)

        # 2. Modified Z-score (MAD)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 1e-9
        votes += (0.6745 * np.abs(arr - med) / mad > 3.5).astype(int)

        # 3. Rolling statistics (window=min(30, n//4))
        w = max(5, min(30, n // 4))
        for i in range(n):
            seg = arr[max(0, i-w):i+1]
            if len(seg) < 3:
                continue
            m_, s_ = np.mean(seg[:-1]), np.std(seg[:-1]) + 1e-9
            if abs(arr[i] - m_) / s_ > self._roll_z_thr:
                votes[i] += 1

        # 4. IQR
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1 + 1e-9
        votes += ((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)).astype(int)

        # 5. STL residuals (simplified: detrend with rolling mean)
        roll_mean = np.convolve(arr, np.ones(w) / w, mode="same")
        residuals = arr - roll_mean
        res_sd = np.std(residuals) + 1e-9
        votes += (np.abs(residuals) / res_sd > 2.5).astype(int)

        # 6. Isolation Forest (requires sklearn)
        try:
            from sklearn.ensemble import IsolationForest
            if n >= 10:
                clf = IsolationForest(contamination=0.02, random_state=42)
                labels = clf.fit_predict(arr.reshape(-1, 1))
                votes += (labels == -1).astype(int)
        except Exception:
            pass

        # 7. DBSCAN
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            if n >= 10:
                scaled = StandardScaler().fit_transform(arr.reshape(-1, 1))
                labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(scaled)
                votes += (labels == -1).astype(int)
        except Exception:
            pass

        return votes
