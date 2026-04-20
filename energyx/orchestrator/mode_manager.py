"""Mode manager — tracks online vs offline state."""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional


class Mode(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class ModeManager:
    """Holds and broadcasts current mode (online / offline).

    Online:  live API stream active; Monitoring Module running;
             Control Agent enabled; scheduled jobs running.
    Offline: batch append active; Monitor Agent runs per-batch;
             Control Agent disabled.
    """

    def __init__(self, initial_mode: Mode = Mode.OFFLINE):
        self._mode = initial_mode
        self._listeners: List[Callable[[Mode], None]] = []

    @property
    def mode(self) -> Mode:
        return self._mode

    def is_online(self) -> bool:
        return self._mode == Mode.ONLINE

    def is_offline(self) -> bool:
        return self._mode == Mode.OFFLINE

    def switch_to(self, mode: Mode) -> None:
        if mode != self._mode:
            self._mode = mode
            for cb in self._listeners:
                try:
                    cb(mode)
                except Exception:
                    pass

    def go_online(self) -> None:
        self.switch_to(Mode.ONLINE)

    def go_offline(self) -> None:
        self.switch_to(Mode.OFFLINE)

    def on_mode_change(self, callback: Callable[[Mode], None]) -> None:
        """Register a callback invoked whenever mode changes."""
        self._listeners.append(callback)
