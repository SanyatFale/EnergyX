"""Tests for Step 1: Data Layer.

pytest tests/test_data_layer.py
"""
import gzip
import json
import tempfile
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import pytest

from energyx.data.events import (
    AnomalyApplianceEvent, BudgetTrajectoryEvent, EventType, Severity, event_from_dict,
)
from energyx.data.ideal_loader import (
    IDEALLoader, create_synthetic_ideal_files, parse_filename, stream_sensor_file,
    SensorType, FileMetadata,
)
from energyx.data.tariffs import (
    TariffCache, TariffRate, TariffSchedule, TariffStructure, make_default_tariffs,
)
from energyx.data.appliances import Appliance, ApplianceClass, ApplianceRegistry
from energyx.data.historic_store import HistoricStore, ParquetBackend


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_anomaly_event_serialization(self):
        e = AnomalyApplianceEvent(
            device_id="fridge_01",
            metric="z_score",
            value=4.2,
            threshold=3.0,
            context_window=[100.0, 105.0, 420.0, 110.0],
            methods_agreed=5,
            severity=Severity.WARN,
            home_id="home001",
        )
        d = e.to_dict()
        assert d["type"] == EventType.ANOMALY_APPLIANCE
        assert d["device_id"] == "fridge_01"
        assert d["value"] == 4.2

    def test_event_round_trip_json(self):
        e = BudgetTrajectoryEvent(
            projected_spend_gbp=165.0,
            cap_gbp=150.0,
            pct_of_cap=1.1,
            severity=Severity.CRITICAL,
        )
        restored = event_from_dict(json.loads(e.to_json()))
        assert restored.type == EventType.BUDGET_TRAJECTORY
        assert restored.projected_spend_gbp == 165.0

    def test_event_from_dict_unknown_type(self):
        """Unknown event types fall back to BaseEvent."""
        from energyx.data.events import BaseEvent
        e = event_from_dict({"type": "custom.unknown", "ts": "2026-01-01T00:00:00Z"})
        assert isinstance(e, BaseEvent)


# ---------------------------------------------------------------------------
# IDEAL Loader — synthetic files
# ---------------------------------------------------------------------------

class TestIDEALLoader:
    @pytest.fixture
    def synthetic_dir(self, tmp_path):
        paths = create_synthetic_ideal_files(tmp_path, n_homes=2)
        return tmp_path, paths

    def test_synthetic_file_creation(self, synthetic_dir):
        out_dir, paths = synthetic_dir
        assert len(paths) == 10  # 5 files per home × 2 homes
        for p in paths:
            assert p.exists()

    def test_temperature_divided_by_10(self, synthetic_dir):
        """CRITICAL: temperature values must be ÷10 on ingest."""
        out_dir, _ = synthetic_dir
        loader = IDEALLoader(out_dir)
        metas = [m for m in loader.discover_files()
                 if m.sensor_type == SensorType.TEMPERATURE_ROOM and m.home_id == "home001"]
        assert metas, "No temperature files found"
        for chunk in stream_sensor_file(metas[0]):
            # Raw values are ~200 (tenths), loaded values should be ~20.0 °C
            assert chunk["value"].max() < 50.0, \
                "Temperature was not divided by 10 — values should be in °C (≈20), not tenths (≈200)"
            assert chunk["value"].min() > -20.0
            assert chunk["unit"].iloc[0] == "celsius"
            break

    def test_electricity_is_watts(self, synthetic_dir):
        out_dir, _ = synthetic_dir
        loader = IDEALLoader(out_dir)
        metas = [m for m in loader.discover_files()
                 if m.sensor_type == SensorType.ELECTRICITY_APPARENT and m.home_id == "home001"]
        assert metas
        for chunk in stream_sensor_file(metas[0]):
            assert chunk["unit"].iloc[0] == "watts"
            # Synthetic electricity is ~300W
            assert 0 < chunk["value"].mean() < 10_000
            break

    def test_no_header_row(self, synthetic_dir):
        """Sensor files must not have a header row — first row should be data."""
        out_dir, paths = synthetic_dir
        # electric-mains files match the real naming convention
        elec_files = [p for p in paths if "electric-mains" in p.name]
        assert elec_files
        with gzip.open(elec_files[0], "rt") as f:
            first_line = f.readline().strip()
        # First line should be "YYYY-MM-DD HH:MM:SS,<value>" not a header
        assert first_line[0].isdigit() or first_line[0] == "2", \
            f"First line looks like a header: {first_line!r}"

    def test_discover_homes(self, synthetic_dir):
        out_dir, _ = synthetic_dir
        loader = IDEALLoader(out_dir)
        homes = loader.discover_homes()
        assert "home001" in homes
        assert "home002" in homes

    def test_load_home_dataframe(self, synthetic_dir):
        out_dir, _ = synthetic_dir
        loader = IDEALLoader(out_dir)
        df = loader.load_home("home001")
        assert not df.empty
        assert "ts" in df.columns
        assert "value" in df.columns
        assert "home_id" in df.columns

    def test_electricity_downsampled(self, synthetic_dir):
        """Downsampled electricity should have ~60 rows per hour (1/min from 1Hz)."""
        out_dir, _ = synthetic_dir
        loader = IDEALLoader(out_dir)
        df = loader.load_home("home001", sensor_types=[SensorType.ELECTRICITY_APPARENT])
        # 1 hour of 1Hz data → ~60 rows at 1min resolution
        assert 50 < len(df) <= 70, f"Expected ~60 rows after downsampling, got {len(df)}"

    def test_humidity_divided_by_10(self, synthetic_dir):
        """CRITICAL: humidity values must be ÷10 on ingest (stored as tenths of %)."""
        out_dir, _ = synthetic_dir
        loader = IDEALLoader(out_dir)
        metas = [m for m in loader.discover_files()
                 if m.sensor_type == SensorType.HUMIDITY and m.home_id == "home001"]
        assert metas, "No humidity files found"
        for chunk in stream_sensor_file(metas[0]):
            # Raw values ~500 (tenths), loaded values should be ~50.0%
            assert chunk["value"].max() <= 100.0, \
                "Humidity was not divided by 10 — values should be ≤100%, not tenths (≈500)"
            assert chunk["unit"].iloc[0] == "pct"
            break

    def test_gas_unit_is_wh(self, synthetic_dir):
        """Gas values must carry unit='wh' (Watt-hours), not 'pulse'."""
        out_dir, _ = synthetic_dir
        loader = IDEALLoader(out_dir)
        metas = [m for m in loader.discover_files()
                 if m.sensor_type == SensorType.GAS_PULSE and m.home_id == "home001"]
        assert metas, "No gas files found"
        for chunk in stream_sensor_file(metas[0]):
            assert chunk["unit"].iloc[0] == "wh", \
                f"Gas unit should be 'wh' (Watt-hours), got {chunk['unit'].iloc[0]!r}"
            break

    def test_appliance_power_sensor_discovered(self, synthetic_dir):
        """Appliance (electric-appliance) files must be discovered and typed correctly."""
        out_dir, _ = synthetic_dir
        loader = IDEALLoader(out_dir)
        metas = [m for m in loader.discover_files()
                 if m.sensor_type == SensorType.APPLIANCE_POWER and m.home_id == "home001"]
        assert metas, "No appliance_power files found — check filename pattern"
        for chunk in stream_sensor_file(metas[0]):
            assert chunk["unit"].iloc[0] == "watts"
            break

    def test_real_filename_pattern_parses(self):
        """Real IDEAL filenames must parse to correct sensor types."""
        from energyx.data.ideal_loader import parse_filename
        cases = [
            ("home62_kitchen710_sensor1779_electric-appliance_fridgefreezer.csv.gz",
             SensorType.APPLIANCE_POWER),
            ("home96_livingroom997_sensor4112_gas-pulse_gas.csv.gz",
             SensorType.GAS_PULSE),
            ("home62_hall705_sensor1662c1666_electric-mains_electric-combined.csv.gz",
             SensorType.ELECTRICITY_APPARENT),
            ("home96_utility1608_sensor9067_electric-subcircuit_cooker.csv.gz",
             SensorType.ELECTRICITY_REAL),
            ("home62_hall705_sensor1656_room_humidity.csv.gz",
             SensorType.HUMIDITY),
            ("home62_bathroom709_sensor1690_room_temperature.csv.gz",
             SensorType.TEMPERATURE_ROOM),
            ("home62_kitchen710_sensor1701_tempprobe_hot-water-cold-pipe.csv.gz",
             SensorType.TEMPERATURE_PROBE),
        ]
        for fname, expected_type in cases:
            meta = parse_filename(Path(fname))
            assert meta is not None, f"Failed to parse: {fname}"
            assert meta.sensor_type == expected_type, \
                f"{fname}: expected {expected_type}, got {meta.sensor_type}"


# ---------------------------------------------------------------------------
# Tariffs
# ---------------------------------------------------------------------------

class TestTariffs:
    def test_flat_rate(self):
        cache = make_default_tariffs()
        tariff = cache.get_active_tariff()
        assert tariff is not None
        rate = tariff.unit_rate_at(datetime(2026, 4, 20, 14, 0, 0))
        assert rate == pytest.approx(0.2459, rel=1e-3)

    def test_economy7_off_peak(self):
        cache = make_default_tariffs()
        e7 = next(t for t in cache.list_tariffs() if t.structure == TariffStructure.ECONOMY_7)
        # Off-peak: 00:30–07:30
        ts_offpeak = datetime(2026, 4, 20, 2, 0, 0)
        ts_peak = datetime(2026, 4, 20, 10, 0, 0)
        assert e7.unit_rate_at(ts_offpeak) < e7.unit_rate_at(ts_peak)

    def test_tariff_lookup_by_household(self):
        cache = make_default_tariffs()
        cache.assign_household("home001", "flat_standard_2026q1")
        t = cache.get_active_tariff("home001")
        assert t is not None
        assert t.tariff_id == "flat_standard_2026q1"

    def test_cost_calculation(self):
        cache = make_default_tariffs()
        tariff = cache.get_active_tariff()
        ts = datetime(2026, 4, 20, 14, 0, 0)
        cost = tariff.cost_for_kwh(1.0, ts)
        assert cost == pytest.approx(0.2459, rel=1e-3)


# ---------------------------------------------------------------------------
# Appliances
# ---------------------------------------------------------------------------

class TestAppliances:
    def test_from_ideal(self):
        a = Appliance.from_ideal("home001", "fridge_freezer", room="kitchen", index=0)
        assert a.appliance_id == "home001_fridge_freezer_00"
        assert a.appliance_class == ApplianceClass.REFRIGERATION
        assert a.home_id == "home001"

    def test_registry_round_trip(self, tmp_path):
        reg = ApplianceRegistry(path=tmp_path / "appliances.json")
        a = Appliance.from_ideal("home001", "boiler", room="utility")
        reg.register(a)
        reg.save()
        reg2 = ApplianceRegistry(path=tmp_path / "appliances.json")
        loaded = reg2.get(a.appliance_id)
        assert loaded is not None
        assert loaded.appliance_class == ApplianceClass.HVAC

    def test_by_class(self):
        reg = ApplianceRegistry.__new__(ApplianceRegistry)
        reg._path = Path("/tmp/test_appliances.json")
        reg._appliances = {}
        reg.register(Appliance.from_ideal("home001", "fridge", room="kitchen"))
        reg.register(Appliance.from_ideal("home001", "washing_machine", room="utility"))
        fridges = reg.by_class("home001", ApplianceClass.REFRIGERATION)
        assert len(fridges) == 1


# ---------------------------------------------------------------------------
# Historic Store
# ---------------------------------------------------------------------------

class TestHistoricStore:
    pyarrow = pytest.importorskip("pyarrow", reason="pyarrow not installed")

    def test_write_read_ticks(self, tmp_path):
        store = HistoricStore(backend=ParquetBackend(tmp_path))
        df = pd.DataFrame({
            "home_id": ["home001"] * 5,
            "ts": pd.date_range("2026-01-01", periods=5, freq="min"),
            "sensor_type": ["electricity_apparent"] * 5,
            "sensor_id": ["0"] * 5,
            "value": [300.0, 310.0, 290.0, 305.0, 315.0],
            "unit": ["watts"] * 5,
        })
        store.write_ticks("home001", df)
        result = store.read_ticks("home001")
        assert len(result) == 5
        assert set(result["home_id"]) == {"home001"}

    def test_write_read_events(self, tmp_path):
        store = HistoricStore(backend=ParquetBackend(tmp_path))
        e = AnomalyApplianceEvent(
            device_id="fridge", metric="z_score", value=4.5, threshold=3.0,
            context_window=[], methods_agreed=4, home_id="home001",
        )
        store.write_event(e)
        events = store.read_events(home_id="home001")
        assert len(events) >= 1

    def test_write_read_batch_summary(self, tmp_path):
        store = HistoricStore(backend=ParquetBackend(tmp_path))
        record = {
            "batch_id": "batch_001",
            "home_id": "home001",
            "narrative": "Test narrative.",
            "n_events": 5,
        }
        store.write_batch_summary(record)
        summaries = store.read_batch_summaries("home001")
        assert len(summaries) == 1
        assert summaries[0]["narrative"] == "Test narrative."
