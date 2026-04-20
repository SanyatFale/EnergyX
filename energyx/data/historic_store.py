"""Historic Store — persistent, queryable storage for EnergyX.

Default backend: Parquet on disk, partitioned by home_id and date.
Designed to be swappable to DuckDB — implement StorageBackend interface.

Stores:
  - Energy ticks (electricity, gas, temperature)
  - Events (anomaly, cost, budget, etc.)
  - Forecasts
  - Batch summary records (offline mode)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from energyx.data.events import BaseEvent

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = Path(__file__).parent.parent.parent / "data" / "historic_store"


class StorageBackend(ABC):
    """Interface that concrete backends must implement."""

    @abstractmethod
    def write_ticks(self, home_id: str, df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def read_ticks(
        self,
        home_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def write_events(self, events: List[Dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def read_events(
        self,
        home_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def write_forecast(self, record: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def write_batch_summary(self, record: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def read_batch_summaries(self, home_id: Optional[str] = None) -> List[Dict[str, Any]]:
        ...


class ParquetBackend(StorageBackend):
    """Parquet-on-disk backend, partitioned by home_id/date.

    Layout:
      store_path/
        ticks/home_id=XXX/date=YYYY-MM-DD/part.parquet
        events/home_id=XXX/date=YYYY-MM-DD/part.parquet
        forecasts/home_id=XXX/run_ts=YYYY-MM-DDTHH:MM:SS/part.parquet
        batch_summaries/home_id=XXX/batch_id=XXX/summary.json
    """

    def __init__(self, store_path: Optional[Path] = None):
        self._root = Path(store_path or _DEFAULT_STORE_PATH)
        self._root.mkdir(parents=True, exist_ok=True)

    # --- Ticks -----------------------------------------------------------

    def write_ticks(self, home_id: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
            df["_date"] = df["ts"].dt.date.astype(str)
        else:
            df["_date"] = datetime.utcnow().date().isoformat()

        for date_str, group in df.groupby("_date"):
            part_dir = self._root / "ticks" / f"home_id={home_id}" / f"date={date_str}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out = part_dir / "part.parquet"
            if out.exists():
                existing = pd.read_parquet(out)
                group = pd.concat([existing, group.drop(columns=["_date"])], ignore_index=True)
            else:
                group = group.drop(columns=["_date"])
            group.to_parquet(out, index=False)

    def read_ticks(
        self,
        home_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        base = self._root / "ticks" / f"home_id={home_id}"
        if not base.exists():
            return pd.DataFrame()
        parts = []
        for date_dir in sorted(base.iterdir()):
            if not date_dir.is_dir():
                continue
            date_str = date_dir.name.replace("date=", "")
            try:
                d = date.fromisoformat(date_str)
            except ValueError:
                continue
            if start and d < start.date():
                continue
            if end and d > end.date():
                continue
            # Support both legacy single-file layout (part.parquet) and
            # per-sensor layout ({sensor_type}_{sensor_id}.parquet)
            parquet_files = sorted(date_dir.glob("*.parquet"))
            for pq in parquet_files:
                try:
                    parts.append(pd.read_parquet(pq))
                except Exception as e:
                    logger.warning(f"Skipping corrupt parquet {pq}: {e}")
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            if start:
                df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
            if end:
                df = df[df["ts"] <= pd.Timestamp(end, tz="UTC")]
        return df.sort_values("ts") if "ts" in df.columns else df

    # --- Events ----------------------------------------------------------

    def write_events(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        df = pd.DataFrame(events)
        if "ts" not in df.columns:
            df["ts"] = datetime.utcnow().isoformat() + "Z"
        df["ts"] = pd.to_datetime(df["ts"])
        df["_date"] = df["ts"].dt.date.astype(str)
        home_id = str(events[0].get("home_id", "unknown"))
        for date_str, group in df.groupby("_date"):
            part_dir = self._root / "events" / f"home_id={home_id}" / f"date={date_str}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out = part_dir / "part.parquet"
            group = group.drop(columns=["_date"])
            # Convert non-primitive columns to JSON strings for Parquet compatibility
            for col in group.columns:
                if group[col].dtype == object:
                    group[col] = group[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x) if x is not None else None
                    )
            if out.exists():
                existing = pd.read_parquet(out)
                group = pd.concat([existing, group], ignore_index=True)
            group.to_parquet(out, index=False)

    def read_events(
        self,
        home_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        base = self._root / "events"
        if not base.exists():
            return []
        dirs = []
        if home_id:
            d = base / f"home_id={home_id}"
            if d.exists():
                dirs = [d]
        else:
            dirs = [d for d in base.iterdir() if d.is_dir()]
        records = []
        for home_dir in dirs:
            for date_dir in sorted(home_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                date_str = date_dir.name.replace("date=", "")
                try:
                    d = date.fromisoformat(date_str)
                except ValueError:
                    continue
                if start and d < start.date():
                    continue
                if end and d > end.date():
                    continue
                part = date_dir / "part.parquet"
                if part.exists():
                    df = pd.read_parquet(part)
                    records.extend(df.to_dict(orient="records"))
        if event_type:
            records = [r for r in records if r.get("type") == event_type]
        return records

    # --- Forecasts -------------------------------------------------------

    def write_forecast(self, record: Dict[str, Any]) -> None:
        home_id = record.get("household_id", "unknown")
        run_ts = record.get("run_ts", datetime.utcnow().isoformat())
        safe_ts = run_ts.replace(":", "-").replace(".", "-")[:19]
        part_dir = self._root / "forecasts" / f"home_id={home_id}" / f"run_ts={safe_ts}"
        part_dir.mkdir(parents=True, exist_ok=True)
        with open(part_dir / "forecast.json", "w") as f:
            json.dump(record, f, indent=2)

    # --- Batch summaries -------------------------------------------------

    def write_batch_summary(self, record: Dict[str, Any]) -> None:
        batch_id = record.get("batch_id", datetime.utcnow().isoformat())
        home_id = record.get("home_id", "unknown")
        safe_id = str(batch_id).replace(":", "-").replace(".", "-")[:40]
        out_dir = self._root / "batch_summaries" / f"home_id={home_id}" / f"batch_id={safe_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(record, f, indent=2)

    def read_batch_summaries(self, home_id: Optional[str] = None) -> List[Dict[str, Any]]:
        base = self._root / "batch_summaries"
        if not base.exists():
            return []
        dirs = []
        if home_id:
            d = base / f"home_id={home_id}"
            if d.exists():
                dirs = [d]
        else:
            dirs = [d for d in base.iterdir() if d.is_dir()]
        summaries = []
        for home_dir in dirs:
            for batch_dir in sorted(home_dir.iterdir()):
                f = batch_dir / "summary.json"
                if f.exists():
                    with open(f) as fp:
                        summaries.append(json.load(fp))
        return summaries


class HistoricStore:
    """Façade over a StorageBackend.  Swap the backend to change persistence."""

    def __init__(self, backend: Optional[StorageBackend] = None, store_path: Optional[Path] = None):
        self._backend = backend or ParquetBackend(store_path)

    # --- Public API ------------------------------------------------------

    def write_ticks(self, home_id: str, df: pd.DataFrame) -> None:
        try:
            self._backend.write_ticks(home_id, df)
        except Exception as e:
            logger.error(f"HistoricStore.write_ticks failed: {e}")

    def read_ticks(
        self,
        home_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        return self._backend.read_ticks(home_id, start, end)

    def write_event(self, event: BaseEvent) -> None:
        self.write_events([event])

    def write_events(self, events: Union[List[BaseEvent], List[Dict[str, Any]]]) -> None:
        dicts = []
        for e in events:
            dicts.append(e.to_dict() if isinstance(e, BaseEvent) else e)
        try:
            self._backend.write_events(dicts)
        except Exception as e:
            logger.error(f"HistoricStore.write_events failed: {e}")

    def read_events(
        self,
        home_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        return self._backend.read_events(home_id, event_type, start, end)

    def write_forecast(self, record: Dict[str, Any]) -> None:
        try:
            self._backend.write_forecast(record)
        except Exception as e:
            logger.error(f"HistoricStore.write_forecast failed: {e}")

    def write_batch_summary(self, record: Dict[str, Any]) -> None:
        try:
            self._backend.write_batch_summary(record)
        except Exception as e:
            logger.error(f"HistoricStore.write_batch_summary failed: {e}")

    def read_batch_summaries(self, home_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._backend.read_batch_summaries(home_id)


_store: Optional[HistoricStore] = None


def get_historic_store(store_path: Optional[Path] = None) -> HistoricStore:
    global _store
    if _store is None:
        _store = HistoricStore(store_path=store_path)
    return _store
