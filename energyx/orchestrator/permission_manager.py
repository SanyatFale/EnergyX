"""Permission manager — token-based consent, persisted to local JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "permissions.json"

# All known permission tokens
ALL_PERMISSIONS = {
    "continuous_training",       # opt-in weekly refit (online only)
    "auto_control:hvac",         # auto-dispatch HVAC commands
    "auto_control:laundry_ev",   # auto-dispatch laundry/EV commands
    "auto_control:emergency_off",# auto-dispatch forgot-to-turn-off
    "dr_enrollment",             # receive and act on DR signals
    "external_data_share",       # outbound telemetry allowed
}


class PermissionDenied(Exception):
    """Raised when a required permission is not granted."""


class PermissionManager:
    """Token-based permission store, persisted to a local JSON file.

    Usage::
        pm = PermissionManager()
        pm.grant("auto_control:hvac")
        pm.require("auto_control:hvac")  # raises PermissionDenied if not granted
        pm.is_granted("continuous_training")  # → bool
    """

    def __init__(self, path=None):
        self._path = Path(path) if path is not None else _DEFAULT_PATH
        self._granted: Set[str] = set()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._granted = set(data.get("granted", []))
            except Exception as e:
                logger.warning(f"PermissionManager: could not load {self._path}: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump({"granted": sorted(self._granted)}, f, indent=2)
        except Exception as e:
            logger.error(f"PermissionManager: save failed: {e}")

    def grant(self, permission: str) -> None:
        if permission not in ALL_PERMISSIONS:
            raise ValueError(f"Unknown permission: {permission}. Valid: {ALL_PERMISSIONS}")
        self._granted.add(permission)
        self._save()
        logger.info(f"Permission granted: {permission}")

    def revoke(self, permission: str) -> None:
        self._granted.discard(permission)
        self._save()
        logger.info(f"Permission revoked: {permission}")

    def is_granted(self, permission: str) -> bool:
        return permission in self._granted

    def require(self, permission: str) -> None:
        """Raise PermissionDenied if the permission is not granted."""
        if not self.is_granted(permission):
            raise PermissionDenied(
                f"Permission '{permission}' is required but not granted. "
                f"Grant it via the permission panel or PermissionManager.grant()."
            )

    def granted_set(self) -> Set[str]:
        return set(self._granted)

    def all_permissions(self) -> Dict[str, bool]:
        return {p: p in self._granted for p in sorted(ALL_PERMISSIONS)}
