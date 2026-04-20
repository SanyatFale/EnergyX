"""Tariff schedule model for EnergyX.

Supports: flat, Economy 7, Economy 10, half-hourly Agile, standing charge, VAT,
regional variation.

Tariff *numbers* (£/kWh) MUST come from get_active_tariff() — never from RAG prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TariffStructure(str, Enum):
    FLAT = "flat"
    ECONOMY_7 = "economy_7"
    ECONOMY_10 = "economy_10"
    AGILE_HALFHOURLY = "agile_halfhourly"
    TIME_OF_USE = "time_of_use"


@dataclass
class TariffRate:
    """A single unit rate that applies during a time window."""
    name: str                     # e.g. "peak", "off_peak", "standard"
    unit_rate_gbp_per_kwh: float  # £/kWh, inclusive of VAT
    start_time: Optional[time] = None   # None = always active
    end_time: Optional[time] = None

    def applies_at(self, ts: datetime) -> bool:
        if self.start_time is None:
            return True
        t = ts.time()
        if self.start_time <= self.end_time:
            return self.start_time <= t < self.end_time
        # Overnight span (e.g. 23:00 – 07:00)
        return t >= self.start_time or t < self.end_time


@dataclass
class TariffSchedule:
    """Complete tariff definition for a household."""
    tariff_id: str
    supplier: str
    plan_name: str
    structure: TariffStructure
    region: str                           # e.g. "london", "scotland", "national"
    standing_charge_gbp_per_day: float    # £/day inclusive of VAT
    rates: List[TariffRate] = field(default_factory=list)
    vat_rate: float = 0.05                # 5% reduced rate on domestic energy
    exit_fee_gbp: float = 0.0
    green_credentials: bool = False
    valid_from: Optional[str] = None      # ISO date string
    valid_to: Optional[str] = None
    # For Agile: half-hourly prices dict keyed by UTC ISO datetime string
    halfhourly_prices: Dict[str, float] = field(default_factory=dict)

    def unit_rate_at(self, ts: datetime) -> float:
        """Return the £/kWh rate applicable at the given timestamp."""
        if self.structure == TariffStructure.AGILE_HALFHOURLY:
            # Find the matching half-hour slot
            slot = ts.replace(minute=0 if ts.minute < 30 else 30, second=0, microsecond=0)
            key = slot.isoformat()
            if key in self.halfhourly_prices:
                return self.halfhourly_prices[key]
            # Fallback: use flat rate if Agile prices not loaded
            return self.rates[0].unit_rate_gbp_per_kwh if self.rates else 0.34
        for rate in self.rates:
            if rate.applies_at(ts):
                return rate.unit_rate_gbp_per_kwh
        # Fallback to first rate
        return self.rates[0].unit_rate_gbp_per_kwh if self.rates else 0.0

    def cost_for_kwh(self, kwh: float, ts: datetime) -> float:
        """£ cost for a given kWh consumption at a timestamp."""
        return kwh * self.unit_rate_at(ts)

    def daily_standing_charge(self) -> float:
        return self.standing_charge_gbp_per_day

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tariff_id": self.tariff_id,
            "supplier": self.supplier,
            "plan_name": self.plan_name,
            "structure": self.structure,
            "region": self.region,
            "standing_charge_gbp_per_day": self.standing_charge_gbp_per_day,
            "vat_rate": self.vat_rate,
            "rates": [
                {
                    "name": r.name,
                    "unit_rate_gbp_per_kwh": r.unit_rate_gbp_per_kwh,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "end_time": r.end_time.isoformat() if r.end_time else None,
                }
                for r in self.rates
            ],
        }


class TariffCache:
    """In-memory + JSON-file tariff cache.

    get_active_tariff() is the ONLY authoritative source for numeric rates.
    RAG can explain tariff structure but not return numbers.
    """

    _DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "tariffs.json"

    def __init__(self, cache_path: Optional[Path] = None):
        self._path = cache_path or self._DEFAULT_PATH
        self._tariffs: Dict[str, TariffSchedule] = {}
        self._household_map: Dict[str, str] = {}  # household_id -> tariff_id
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        """Load tariffs from JSON file if it exists."""
        if self._path.exists():
            with open(self._path) as f:
                raw = json.load(f)
            for item in raw.get("tariffs", []):
                t = self._from_dict(item)
                self._tariffs[t.tariff_id] = t
            self._household_map = raw.get("household_map", {})

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(
                {
                    "tariffs": [t.to_dict() for t in self._tariffs.values()],
                    "household_map": self._household_map,
                },
                f, indent=2,
            )

    # ------------------------------------------------------------------
    def register_tariff(self, tariff: TariffSchedule):
        self._tariffs[tariff.tariff_id] = tariff

    def assign_household(self, household_id: str, tariff_id: str):
        self._household_map[household_id] = tariff_id

    def get_active_tariff(self, household_id: Optional[str] = None) -> Optional[TariffSchedule]:
        """Return the active TariffSchedule for a household, or the default."""
        if household_id and household_id in self._household_map:
            tid = self._household_map[household_id]
            return self._tariffs.get(tid)
        # Return first tariff as default
        return next(iter(self._tariffs.values()), None)

    def list_tariffs(self) -> List[TariffSchedule]:
        return list(self._tariffs.values())

    # ------------------------------------------------------------------
    @staticmethod
    def _from_dict(d: Dict[str, Any]) -> TariffSchedule:
        rates = []
        for r in d.get("rates", []):
            st = time.fromisoformat(r["start_time"]) if r.get("start_time") else None
            et = time.fromisoformat(r["end_time"]) if r.get("end_time") else None
            rates.append(TariffRate(
                name=r["name"],
                unit_rate_gbp_per_kwh=r["unit_rate_gbp_per_kwh"],
                start_time=st,
                end_time=et,
            ))
        return TariffSchedule(
            tariff_id=d["tariff_id"],
            supplier=d["supplier"],
            plan_name=d["plan_name"],
            structure=TariffStructure(d["structure"]),
            region=d["region"],
            standing_charge_gbp_per_day=d["standing_charge_gbp_per_day"],
            rates=rates,
            vat_rate=d.get("vat_rate", 0.05),
        )


def make_default_tariffs() -> TariffCache:
    """Seed a TariffCache with representative UK tariff examples (2026 Q1)."""
    cache = TariffCache.__new__(TariffCache)
    cache._path = TariffCache._DEFAULT_PATH
    cache._tariffs = {}
    cache._household_map = {}

    # Standard flat tariff (~Ofgem Q1 2026 cap)
    flat = TariffSchedule(
        tariff_id="flat_standard_2026q1",
        supplier="generic",
        plan_name="Standard Variable",
        structure=TariffStructure.FLAT,
        region="national",
        standing_charge_gbp_per_day=0.61,
        rates=[TariffRate(name="standard", unit_rate_gbp_per_kwh=0.2459)],
    )

    # Economy 7 (off-peak 00:30–07:30 UTC)
    e7 = TariffSchedule(
        tariff_id="economy7_2026q1",
        supplier="generic",
        plan_name="Economy 7",
        structure=TariffStructure.ECONOMY_7,
        region="national",
        standing_charge_gbp_per_day=0.61,
        rates=[
            TariffRate("peak", 0.2859, time(7, 30), time(0, 30)),
            TariffRate("off_peak", 0.1189, time(0, 30), time(7, 30)),
        ],
    )

    cache.register_tariff(flat)
    cache.register_tariff(e7)
    return cache


# Module-level singleton
_tariff_cache: Optional[TariffCache] = None


def get_tariff_cache() -> TariffCache:
    global _tariff_cache
    if _tariff_cache is None:
        _tariff_cache = TariffCache()
        if not _tariff_cache._tariffs:
            _tariff_cache = make_default_tariffs()
    return _tariff_cache
