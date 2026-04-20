"""Appliance registry for EnergyX.

Stable IDs, install date, nameplate rating, class enum, IDEAL appliance_type mapping.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ApplianceClass(str, Enum):
    """Broad class used for scheduling and permission gating."""
    HVAC = "hvac"              # Heating, ventilation, AC
    LAUNDRY = "laundry"        # Washing machine, tumble dryer, dishwasher
    EV = "ev"                  # EV charger
    WATER_HEATING = "water_heating"
    COOKING = "cooking"
    REFRIGERATION = "refrigeration"
    ENTERTAINMENT = "entertainment"
    LIGHTING = "lighting"
    SMALL_APPLIANCE = "small_appliance"
    OTHER = "other"


# Mapping from IDEAL dataset appliance_type strings to ApplianceClass
IDEAL_TYPE_TO_CLASS: Dict[str, ApplianceClass] = {
    "boiler": ApplianceClass.HVAC,
    "heat_pump": ApplianceClass.HVAC,
    "electric_heater": ApplianceClass.HVAC,
    "air_conditioner": ApplianceClass.HVAC,
    "washing_machine": ApplianceClass.LAUNDRY,
    "tumble_dryer": ApplianceClass.LAUNDRY,
    "dishwasher": ApplianceClass.LAUNDRY,
    "ev_charger": ApplianceClass.EV,
    "immersion_heater": ApplianceClass.WATER_HEATING,
    "hot_water_cylinder": ApplianceClass.WATER_HEATING,
    "electric_oven": ApplianceClass.COOKING,
    "microwave": ApplianceClass.COOKING,
    "kettle": ApplianceClass.COOKING,
    "fridge": ApplianceClass.REFRIGERATION,
    "fridge_freezer": ApplianceClass.REFRIGERATION,
    "freezer": ApplianceClass.REFRIGERATION,
    "television": ApplianceClass.ENTERTAINMENT,
    "computer": ApplianceClass.ENTERTAINMENT,
    "games_console": ApplianceClass.ENTERTAINMENT,
    "lighting": ApplianceClass.LIGHTING,
    "iron": ApplianceClass.SMALL_APPLIANCE,
    "vacuum": ApplianceClass.SMALL_APPLIANCE,
    "other": ApplianceClass.OTHER,
}


@dataclass
class Appliance:
    """One registered appliance in a household."""
    appliance_id: str          # Stable EnergyX ID (e.g. "home001_fridge_01")
    home_id: str
    ideal_appliance_type: str  # Original IDEAL appliance_type string
    appliance_class: ApplianceClass
    room: Optional[str] = None
    nameplate_watts: Optional[float] = None   # Rated power from nameplate
    install_date: Optional[str] = None        # ISO date
    entity_id: Optional[str] = None          # Home Assistant entity_id if known
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Appliance":
        d = dict(d)
        d["appliance_class"] = ApplianceClass(d.get("appliance_class", "other"))
        return cls(**d)

    @classmethod
    def from_ideal(
        cls,
        home_id: str,
        ideal_type: str,
        room: Optional[str] = None,
        index: int = 0,
        nameplate_watts: Optional[float] = None,
        entity_id: Optional[str] = None,
    ) -> "Appliance":
        """Construct an Appliance from IDEAL dataset metadata."""
        cls_enum = IDEAL_TYPE_TO_CLASS.get(ideal_type, ApplianceClass.OTHER)
        safe_type = ideal_type.replace(" ", "_").lower()
        appliance_id = f"{home_id}_{safe_type}_{index:02d}"
        return cls(
            appliance_id=appliance_id,
            home_id=home_id,
            ideal_appliance_type=ideal_type,
            appliance_class=cls_enum,
            room=room,
            nameplate_watts=nameplate_watts,
            entity_id=entity_id,
        )


class ApplianceRegistry:
    """In-memory + JSON-persisted registry of all appliances."""

    _DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "appliances.json"

    def __init__(self, path: Optional[Path] = None):
        self._path = path or self._DEFAULT_PATH
        self._appliances: Dict[str, Appliance] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path) as f:
                for d in json.load(f):
                    a = Appliance.from_dict(d)
                    self._appliances[a.appliance_id] = a

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump([a.to_dict() for a in self._appliances.values()], f, indent=2)

    def register(self, appliance: Appliance):
        self._appliances[appliance.appliance_id] = appliance

    def get(self, appliance_id: str) -> Optional[Appliance]:
        return self._appliances.get(appliance_id)

    def for_home(self, home_id: str) -> List[Appliance]:
        return [a for a in self._appliances.values() if a.home_id == home_id]

    def by_class(self, home_id: str, cls: ApplianceClass) -> List[Appliance]:
        return [a for a in self.for_home(home_id) if a.appliance_class == cls]

    def all(self) -> List[Appliance]:
        return list(self._appliances.values())


_registry: Optional[ApplianceRegistry] = None


def get_appliance_registry() -> ApplianceRegistry:
    global _registry
    if _registry is None:
        _registry = ApplianceRegistry()
    return _registry
