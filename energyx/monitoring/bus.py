"""In-process event bus for EnergyX.

Simple synchronous publish/subscribe.  Upgrade to async or a message broker
(Redis Streams, etc.) for production without changing the subscribe/publish API.

Both the online Monitoring Module and the offline Monitor Agent publish to this
bus using the same schema (energyx.data.events).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from energyx.data.events import BaseEvent

logger = logging.getLogger(__name__)

# Subscriber callback type: (event) -> None
Subscriber = Callable[[BaseEvent], None]


class EventBus:
    """Synchronous publish/subscribe event bus.

    Usage::
        bus = EventBus()

        def on_anomaly(event):
            print(event.type, event.severity)

        bus.subscribe("anomaly.appliance", on_anomaly)
        bus.subscribe("*", on_anomaly)          # wildcard — all events
        bus.publish(AnomalyApplianceEvent(...))
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Subscriber]] = defaultdict(list)
        self._history: List[BaseEvent] = []
        self._max_history: int = 1000

    def subscribe(self, event_type: str, callback: Subscriber) -> None:
        """Subscribe a callback to an event type or "*" for all events."""
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Subscriber) -> None:
        try:
            self._subscribers[event_type].remove(callback)
        except ValueError:
            pass

    def publish(self, event: BaseEvent) -> None:
        """Publish an event to all matching subscribers."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Call type-specific subscribers
        for cb in list(self._subscribers.get(event.type, [])):
            try:
                cb(event)
            except Exception as e:
                logger.error(f"EventBus subscriber error ({event.type}): {e}")

        # Call wildcard subscribers
        for cb in list(self._subscribers.get("*", [])):
            try:
                cb(event)
            except Exception as e:
                logger.error(f"EventBus wildcard subscriber error: {e}")

    def publish_many(self, events: List[BaseEvent]) -> None:
        for event in events:
            self.publish(event)

    def recent(
        self,
        event_type: Optional[str] = None,
        n: int = 100,
    ) -> List[BaseEvent]:
        """Return recent events, optionally filtered by type."""
        if event_type:
            return [e for e in self._history if e.type == event_type][-n:]
        return self._history[-n:]

    def clear_history(self):
        self._history.clear()


# Module-level singleton
_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
