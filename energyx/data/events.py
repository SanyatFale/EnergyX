"""Structured event schema for EnergyX event bus.

Every event emitted by the Monitoring Module (online) or Monitor Agent (offline)
uses one of these dataclasses. The Orchestrator routes events based on their type.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EventType(str, Enum):
    ANOMALY_APPLIANCE = "anomaly.appliance"
    COST_REALTIME = "cost.realtime"
    CARBON_REALTIME = "carbon.realtime"
    BUDGET_TRAJECTORY = "budget.trajectory"
    CONTROL_SUGGEST = "control.suggest"
    DR_EVENT = "dr.event"
    DEGRADATION = "degradation.appliance"
    HABIT_DRIFT = "habit.drift"
    FORGOT_TURN_OFF = "forgot.turn_off"
    STORE_TICK = "store.tick"
    RETRAINING_REQUEST = "retraining.request"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class BaseEvent:
    type: str
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    severity: str = Severity.INFO
    home_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class AnomalyApplianceEvent(BaseEvent):
    """Fired when an appliance deviates from its baseline."""
    type: str = EventType.ANOMALY_APPLIANCE
    device_id: str = ""
    metric: str = ""           # e.g. "z_score", "iqr", "isolation_forest"
    value: float = 0.0
    threshold: float = 0.0
    context_window: List[float] = field(default_factory=list)
    methods_agreed: int = 0


@dataclass
class CostRealtimeEvent(BaseEvent):
    """Fired on each monitoring tick with current burn rate."""
    type: str = EventType.COST_REALTIME
    rate_gbp_per_hour: float = 0.0
    top_contributors: List[Tuple[str, float]] = field(default_factory=list)  # [(name, gbp/h)]
    tariff_name: str = ""


@dataclass
class CarbonRealtimeEvent(BaseEvent):
    """Fired on each tick with current carbon intensity."""
    type: str = EventType.CARBON_REALTIME
    intensity_gco2_per_kwh: float = 0.0
    region: str = "national"
    source: str = "national_grid_eso"


@dataclass
class BudgetTrajectoryEvent(BaseEvent):
    """Fired when projected end-of-month spend threatens the cap."""
    type: str = EventType.BUDGET_TRAJECTORY
    projected_spend_gbp: float = 0.0
    cap_gbp: float = 0.0
    days_to_breach: Optional[float] = None
    pct_of_cap: float = 0.0


@dataclass
class ControlSuggestEvent(BaseEvent):
    """Advisory action emitted by monitors (e.g. turn off iron)."""
    type: str = EventType.CONTROL_SUGGEST
    action: str = ""           # "turn_off", "defer", "set_temperature"
    entity_id: str = ""
    reason: str = ""
    ha_payload: Optional[Dict[str, Any]] = None


@dataclass
class DemandResponseEvent(BaseEvent):
    """Fired when a DR signal is received from the grid."""
    type: str = EventType.DR_EVENT
    event_id: str = ""
    start_ts: str = ""
    end_ts: str = ""
    signal_source: str = ""   # "octopus_saving_session", "national_grid"
    reward_gbp: Optional[float] = None


@dataclass
class DegradationEvent(BaseEvent):
    """Fired when sustained upward drift detected on an appliance."""
    type: str = EventType.DEGRADATION
    device_id: str = ""
    baseline_watts: float = 0.0
    current_watts: float = 0.0
    drift_pct: float = 0.0
    observation_days: int = 0


@dataclass
class HabitDriftEvent(BaseEvent):
    """Fired when slow behavioural shift is detected."""
    type: str = EventType.HABIT_DRIFT
    pattern: str = ""          # e.g. "hvac_runtime_increasing"
    drift_magnitude: float = 0.0
    baseline_period: str = ""
    current_period: str = ""


@dataclass
class ForgotTurnOffEvent(BaseEvent):
    """Fired when appliance runs in unusual time window."""
    type: str = EventType.FORGOT_TURN_OFF
    device_id: str = ""
    running_since: str = ""
    unusual_hours: bool = True
    idle_minutes: Optional[float] = None


@dataclass
class RetrainingRequestEvent(BaseEvent):
    """Fired by Monitor Agent when batch reveals model drift."""
    type: str = EventType.RETRAINING_REQUEST
    reason: str = ""
    affected_models: List[str] = field(default_factory=list)
    batch_id: str = ""


def event_from_dict(d: Dict[str, Any]) -> BaseEvent:
    """Deserialize an event dict to the correct dataclass."""
    _MAP = {
        EventType.ANOMALY_APPLIANCE: AnomalyApplianceEvent,
        EventType.COST_REALTIME: CostRealtimeEvent,
        EventType.CARBON_REALTIME: CarbonRealtimeEvent,
        EventType.BUDGET_TRAJECTORY: BudgetTrajectoryEvent,
        EventType.CONTROL_SUGGEST: ControlSuggestEvent,
        EventType.DR_EVENT: DemandResponseEvent,
        EventType.DEGRADATION: DegradationEvent,
        EventType.HABIT_DRIFT: HabitDriftEvent,
        EventType.FORGOT_TURN_OFF: ForgotTurnOffEvent,
        EventType.RETRAINING_REQUEST: RetrainingRequestEvent,
    }
    cls = _MAP.get(d.get("type"), BaseEvent)
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
