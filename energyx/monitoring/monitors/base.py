"""Base class for all EnergyX monitors.

Each monitor:
  - declares applicable_modes ({"online"}, {"offline"}, or {"online","offline"})
  - exposes evaluate(tick_or_batch) -> list[Event]
  - contains NO LLM calls
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from energyx.data.events import BaseEvent


class BaseMonitor(ABC):
    """Abstract base for all monitors.

    Args:
        config: Dict of monitor-specific config (thresholds, etc.)
        state_store: Optional dict for persisting rolling state across calls.
                     If None, state lives only in memory for this instance.
    """

    # Subclasses MUST declare this
    applicable_modes: Set[str] = {"online", "offline"}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        state_store: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self._state: Dict[str, Any] = state_store if state_store is not None else {}
        self.home_id: Optional[str] = self.config.get("home_id")

    @abstractmethod
    def evaluate(self, tick_or_batch: Any) -> List[BaseEvent]:
        """Process a single tick (dict) or batch (DataFrame) and return events.

        Must NOT make any LLM calls.  Must be idempotent on the same input.
        Updates self._state for rolling state.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def _set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def _cfg(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def _make_event_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.home_id:
            kwargs["home_id"] = self.home_id
        return kwargs
