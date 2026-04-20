"""IDEAL dataset loader for EnergyX.

Authoritative spec: Pullinger et al. 2021, Scientific Data
DOI: 10.7488/ds/2836

Key invariants (from the paper):
  - Timestamps are UTC in "YYYY-MM-DD hh:mm:ss" format, NO header line in sensor files.
  - Electricity (electric-mains) is 1 Hz apparent power (Watts).
  - Electricity (electric-subcircuit, electric-appliance) is Watts at variable/1 Hz.
  - Temperatures stored in TENTHS of degrees Celsius — loader divides by 10.
  - Humidity stored in TENTHS of percent — loader divides by 10.
  - Gas (gas-pulse) stored in Watt-hours (NOT raw pulse counts).
  - Room sensors are 12-second cadence (temp, humidity, light).
  - Boiler-pipe / radiator tempprobes are 12-second cadence.
  - 255 total homes; 39 have enhanced monitoring (appliance-level data).
  - Files are gzip-compressed CSVs (extension .csv.gz or bare .gz).
  - Known issue: data 08:50–09:50 on 17 April 2018 is unreliable across all homes.
  - Home 223 has no mains electric data.

File naming convention (real files):
  home{homeid}_{roomtype}{roomid}_sensor{sensorid}_{sensorbox}_{sensorsubtype}.csv.gz
  e.g. home62_kitchen710_sensor1779_electric-appliance_fridgefreezer.csv.gz
       home62_hall705_sensor1662c1666_electric-mains_electric-combined.csv.gz
       home96_livingroom997_sensor4112_gas-pulse_gas.csv.gz
"""

from __future__ import annotations

import gzip
import io
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class SensorType(str, Enum):
    ELECTRICITY_APPARENT = "electricity_apparent"   # electric-mains, 1 Hz, Watts (apparent)
    ELECTRICITY_REAL = "electricity_real"           # electric-subcircuit, Watts
    GAS_PULSE = "gas_pulse"                         # gas-pulse sensorbox, Watt-hours
    TEMPERATURE_ROOM = "temperature_room"           # room sensorbox/temperature, 12s, tenths °C → ÷10
    TEMPERATURE_PROBE = "temperature_probe"         # tempprobe sensorbox, 12s, tenths °C → ÷10
    HUMIDITY = "humidity"                           # room sensorbox/humidity, 12s, tenths % → ÷10
    LIGHT = "light"                                 # room sensorbox/light, 12s, uncalibrated
    APPLIANCE_POWER = "appliance_power"             # electric-appliance, enhanced only, Watts


# Columns produced by the loader
TICK_COLUMNS = ["home_id", "ts", "sensor_type", "sensor_id", "value", "unit"]

# Temperature sensors: value must be divided by 10
_TEMPERATURE_TYPES = {SensorType.TEMPERATURE_ROOM, SensorType.TEMPERATURE_PROBE}

# Standard cadence expectations (seconds)
SENSOR_CADENCE = {
    SensorType.ELECTRICITY_APPARENT: 1,
    SensorType.ELECTRICITY_REAL: 1,
    SensorType.GAS_PULSE: None,           # event-driven
    SensorType.TEMPERATURE_ROOM: 12,
    SensorType.TEMPERATURE_PROBE: 12,
    SensorType.HUMIDITY: 12,
    SensorType.LIGHT: 12,
    SensorType.APPLIANCE_POWER: 1,
}

# For warm storage, electricity is down-sampled to 1-minute averages
ELECTRICITY_DOWNSAMPLE = "1min"


# ---------------------------------------------------------------------------
# File-name parser
# ---------------------------------------------------------------------------

# Real IDEAL naming convention:
#   home{homeid}_{roomtype}{roomid}_sensor{sensorid}_{sensorbox}_{sensorsubtype}.csv.gz
#
# sensorid may be composite: e.g. 1662c1666 (two merged sensor IDs for mains)
# sensorbox values: room, electric-appliance, electric-mains, electric-subcircuit,
#                   gas-pulse, tempprobe, heatcook, heater
# Extension: .csv.gz  (most files) or bare .gz (rare)
_FNAME_RE = re.compile(
    r"home(?P<home_id>\d+)"
    r"_(?P<room_type>[a-z][a-z0-9]*)(?P<room_id>\d+)"
    r"_sensor(?P<sensor_id>[0-9]+(?:c[0-9]+)?)"
    r"_(?P<sensorbox>[a-z][a-z0-9-]*)"
    r"_(?P<sensor_subtype>[a-z][a-z0-9-]*)"
    r"(?:\.csv)?\.gz$",
    re.IGNORECASE,
)

# sensorbox → SensorType (for non-room sensorboxes)
_SENSORBOX_TO_TYPE: Dict[str, SensorType] = {
    "electric-appliance": SensorType.APPLIANCE_POWER,
    "electric-mains": SensorType.ELECTRICITY_APPARENT,
    "electric-subcircuit": SensorType.ELECTRICITY_REAL,
    "gas-pulse": SensorType.GAS_PULSE,
    "tempprobe": SensorType.TEMPERATURE_PROBE,
    "heatcook": SensorType.TEMPERATURE_PROBE,
    "heater": SensorType.TEMPERATURE_PROBE,
}

# room sensorbox subtype → SensorType
_ROOM_SUBTYPE_TO_TYPE: Dict[str, SensorType] = {
    "temperature": SensorType.TEMPERATURE_ROOM,
    "humidity": SensorType.HUMIDITY,
    "light": SensorType.LIGHT,
}


@dataclass
class FileMetadata:
    path: Path
    home_id: str
    sensor_type: SensorType
    room_id: Optional[str]
    sensor_id: str
    is_enhanced: bool = False

    @property
    def is_temperature(self) -> bool:
        return self.sensor_type in _TEMPERATURE_TYPES


def parse_filename(path: Path) -> Optional[FileMetadata]:
    """Parse a real IDEAL sensor filename into FileMetadata. Returns None if unrecognised."""
    m = _FNAME_RE.match(path.name)
    if not m:
        logger.debug(f"Unrecognised IDEAL filename: {path.name}")
        return None

    sensorbox = m.group("sensorbox").lower()
    subtype = m.group("sensor_subtype").lower()

    if sensorbox == "room":
        sensor_type = _ROOM_SUBTYPE_TO_TYPE.get(subtype, SensorType.TEMPERATURE_ROOM)
    else:
        sensor_type = _SENSORBOX_TO_TYPE.get(sensorbox, SensorType.ELECTRICITY_APPARENT)

    room_id = f"{m.group('room_type')}{m.group('room_id')}"
    return FileMetadata(
        path=path,
        home_id=f"home{m.group('home_id')}",
        sensor_type=sensor_type,
        room_id=room_id,
        sensor_id=m.group("sensor_id"),
    )


# ---------------------------------------------------------------------------
# Low-level row reader (streaming)
# ---------------------------------------------------------------------------

def _open_file(path: Path):
    """Open a plain or gzipped CSV as a text stream."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_sensor_file(
    meta: FileMetadata,
    chunk_size: int = 10_000,
) -> Generator[pd.DataFrame, None, None]:
    """Yield chunks of rows from an IDEAL sensor file.

    Applies:
      - UTC timestamp parsing (YYYY-MM-DD HH:MM:SS, NO header row)
      - Temperature ÷ 10 conversion
      - Column labelling
    """
    with _open_file(meta.path) as fh:
        while True:
            lines = []
            try:
                for _ in range(chunk_size):
                    line = next(fh)
                    lines.append(line)
            except StopIteration:
                pass
            if not lines:
                break
            buf = io.StringIO("".join(lines))
            # IDEAL sensor files have NO header line: columns are [timestamp, value]
            chunk = pd.read_csv(buf, header=None, names=["ts", "raw_value"])
            chunk["ts"] = pd.to_datetime(chunk["ts"], format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["ts"])
            chunk["raw_value"] = pd.to_numeric(chunk["raw_value"], errors="coerce")

            # Apply scaling per PDF spec
            if meta.is_temperature:
                # All temperatures (room + probe) stored as tenths of °C
                chunk["value"] = chunk["raw_value"] / 10.0
                chunk["unit"] = "celsius"
            elif meta.sensor_type == SensorType.HUMIDITY:
                # Humidity stored as tenths of percent
                chunk["value"] = chunk["raw_value"] / 10.0
                chunk["unit"] = "pct"
            elif meta.sensor_type == SensorType.GAS_PULSE:
                # Gas stored as Watt-hours (not raw pulse counts) per Pullinger et al.
                chunk["value"] = chunk["raw_value"]
                chunk["unit"] = "wh"
            elif meta.sensor_type == SensorType.LIGHT:
                # Light is uncalibrated auxiliary — pass through as-is
                chunk["value"] = chunk["raw_value"]
                chunk["unit"] = "uncalibrated"
            else:
                # All electricity types (apparent, real, appliance) in Watts
                chunk["value"] = chunk["raw_value"]
                chunk["unit"] = "watts"

            chunk["home_id"] = meta.home_id
            chunk["sensor_type"] = meta.sensor_type.value
            chunk["sensor_id"] = meta.sensor_id
            yield chunk[TICK_COLUMNS]

            if len(lines) < chunk_size:
                break


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

class IDEALLoader:
    """Load IDEAL dataset files into DataFrames ready for the HistoricStore.

    Usage::
        loader = IDEALLoader("/path/to/ideal/data")
        for df in loader.stream_home("home001"):
            historic_store.write_ticks("home001", df)

    Electricity is down-sampled to 1-min averages for warm storage.
    Raw 1s data available via stream_home(downsample_electricity=False).
    """

    ENHANCED_HOME_COUNT = 39
    TOTAL_HOME_COUNT = 255

    def __init__(self, data_root: str | Path):
        self._root = Path(data_root)
        if not self._root.exists():
            raise FileNotFoundError(f"IDEAL data root not found: {self._root}")

    def discover_files(self) -> List[FileMetadata]:
        """Recursively discover all IDEAL sensor files."""
        files = []
        for path in self._root.rglob("*.csv*"):
            meta = parse_filename(path)
            if meta:
                files.append(meta)
        logger.info(f"Discovered {len(files)} IDEAL sensor files under {self._root}")
        return files

    def discover_homes(self) -> List[str]:
        """Return sorted list of unique home IDs found in the data root."""
        return sorted({m.home_id for m in self.discover_files()})

    def stream_home(
        self,
        home_id: str,
        sensor_types: Optional[List[SensorType]] = None,
        downsample_electricity: bool = True,
        chunk_size: int = 10_000,
    ) -> Generator[pd.DataFrame, None, None]:
        """Yield DataFrame chunks for all sensors of a single home.

        Args:
            home_id: e.g. "home001"
            sensor_types: Filter by sensor type (None = all)
            downsample_electricity: Resample 1 Hz to 1-min averages
            chunk_size: Rows per chunk
        """
        for meta in self.discover_files():
            if meta.home_id != home_id:
                continue
            if sensor_types and meta.sensor_type not in sensor_types:
                continue
            for chunk in stream_sensor_file(meta, chunk_size=chunk_size):
                if downsample_electricity and meta.sensor_type in {
                    SensorType.ELECTRICITY_APPARENT, SensorType.ELECTRICITY_REAL
                }:
                    chunk = self._downsample_electricity(chunk)
                yield chunk

    def load_home(
        self,
        home_id: str,
        sensor_types: Optional[List[SensorType]] = None,
        downsample_electricity: bool = True,
    ) -> pd.DataFrame:
        """Load all data for a home into a single DataFrame (use with small homes)."""
        parts = list(self.stream_home(home_id, sensor_types, downsample_electricity))
        if not parts:
            return pd.DataFrame(columns=TICK_COLUMNS)
        return pd.concat(parts, ignore_index=True).sort_values("ts")

    def load_all_to_store(
        self,
        store,
        downsample_electricity: bool = True,
        home_ids: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Load all discovered data into a HistoricStore. Returns {home_id: row_count}."""
        from energyx.data.historic_store import HistoricStore
        counts: Dict[str, int] = {}
        all_homes = home_ids or self.discover_homes()
        for home_id in all_homes:
            total = 0
            for chunk in self.stream_home(home_id, downsample_electricity=downsample_electricity):
                store.write_ticks(home_id, chunk)
                total += len(chunk)
            counts[home_id] = total
            logger.info(f"Loaded {total:,} rows for {home_id}")
        return counts

    # --- Helpers ----------------------------------------------------------

    @staticmethod
    def _downsample_electricity(chunk: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-Hz electricity to 1-minute means."""
        if chunk.empty:
            return chunk
        chunk = chunk.copy()
        chunk = chunk.set_index("ts")
        meta_cols = ["home_id", "sensor_type", "sensor_id", "unit"]
        first_meta = {c: chunk[c].iloc[0] for c in meta_cols if c in chunk.columns}
        numeric = chunk[["value"]].resample(ELECTRICITY_DOWNSAMPLE).mean()
        for c, v in first_meta.items():
            numeric[c] = v
        numeric = numeric.reset_index().rename(columns={"ts": "ts"})
        return numeric[TICK_COLUMNS]


# ---------------------------------------------------------------------------
# Synthetic test data generator  (no real IDEAL files needed for tests)
# ---------------------------------------------------------------------------

def create_synthetic_ideal_files(output_dir: str | Path, n_homes: int = 2) -> List[Path]:
    """Create minimal synthetic IDEAL-format files for testing.

    Uses the real IDEAL naming convention:
      home{id}_{roomtype}{roomid}_sensor{sensorid}_{sensorbox}_{subtype}.csv.gz

    Produces gzipped CSVs with no header row, UTC timestamps, and:
      - Mains electricity (electric-mains, 1 Hz apparent power, Watts)
      - Room temperature (room/temperature, 12s, tenths of °C — 200 = 20.0°C)
      - Room humidity (room/humidity, 12s, tenths of % — 500 = 50.0%)
      - Gas (gas-pulse, sparse Watt-hours)
      - Appliance (electric-appliance/fridgefreezer, sparse Watts)

    Returns list of created file paths.
    """
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created = []
    rng = np.random.default_rng(42)

    for i in range(1, n_homes + 1):
        home = f"home{i:03d}"
        # Use incrementing sensor IDs to avoid collisions across homes
        sid_base = (i - 1) * 10 + 1
        base_ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

        # --- Mains electricity (1 Hz, 1 hour = 3600 rows) ---
        n_elec = 3600
        ts_elec = pd.date_range(base_ts, periods=n_elec, freq="1s").strftime("%Y-%m-%d %H:%M:%S")
        watts = (300 + 50 * rng.standard_normal(n_elec)).clip(0).round(1)
        # Real naming: home{id}_hall{roomid}_sensor{id}c{id}_electric-mains_electric-combined.csv.gz
        elec_path = out / f"{home}_hall1_sensor{sid_base}c{sid_base+1}_electric-mains_electric-combined.csv.gz"
        with gzip.open(elec_path, "wt") as f:
            for t, w in zip(ts_elec, watts):
                f.write(f"{t},{w}\n")
        created.append(elec_path)

        # --- Room temperature (12s, tenths of °C: 200 = 20.0°C) ---
        n_temp = n_elec // 12
        ts_temp = pd.date_range(base_ts, periods=n_temp, freq="12s").strftime("%Y-%m-%d %H:%M:%S")
        temp_tenths = (200 + 10 * rng.standard_normal(n_temp)).round().astype(int).clip(100, 350)
        temp_path = out / f"{home}_livingroom2_sensor{sid_base+2}_room_temperature.csv.gz"
        with gzip.open(temp_path, "wt") as f:
            for t, v in zip(ts_temp, temp_tenths):
                f.write(f"{t},{v}\n")
        created.append(temp_path)

        # --- Room humidity (12s, tenths of %: 500 = 50.0%) ---
        humid_tenths = (500 + 20 * rng.standard_normal(n_temp)).round().astype(int).clip(200, 900)
        humid_path = out / f"{home}_livingroom2_sensor{sid_base+3}_room_humidity.csv.gz"
        with gzip.open(humid_path, "wt") as f:
            for t, v in zip(ts_temp, humid_tenths):
                f.write(f"{t},{v}\n")
        created.append(humid_path)

        # --- Gas (sparse Watt-hours) ---
        gas_times = pd.date_range(base_ts, periods=10, freq="6min").strftime("%Y-%m-%d %H:%M:%S")
        gas_path = out / f"{home}_kitchen3_sensor{sid_base+4}_gas-pulse_gas.csv.gz"
        with gzip.open(gas_path, "wt") as f:
            for t in gas_times:
                f.write(f"{t},112\n")
        created.append(gas_path)

        # --- Appliance (fridgefreezer, sparse Watts) ---
        appl_times = pd.date_range(base_ts, periods=20, freq="3min").strftime("%Y-%m-%d %H:%M:%S")
        appl_watts = (80 + 10 * rng.standard_normal(20)).clip(0).round(1)
        appl_path = out / f"{home}_kitchen3_sensor{sid_base+5}_electric-appliance_fridgefreezer.csv.gz"
        with gzip.open(appl_path, "wt") as f:
            for t, w in zip(appl_times, appl_watts):
                f.write(f"{t},{w}\n")
        created.append(appl_path)

    logger.info(f"Created {len(created)} synthetic IDEAL files in {out}")
    return created
