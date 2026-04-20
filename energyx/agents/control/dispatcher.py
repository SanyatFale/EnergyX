"""Control action dispatcher.

StubDispatcher: logs HA JSON to control_dispatch.log (no real HA connection).
HARestDispatcher: real Home Assistant REST API (out of scope for this build — stub).

Swap by passing dispatcher=HARestDispatcher() to ControlAgent.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).parent.parent.parent.parent / "control_dispatch.log"


class BaseDispatcher(ABC):
    @abstractmethod
    def dispatch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send the HA service-call JSON to the target system."""
        ...


class StubDispatcher(BaseDispatcher):
    """Logs every action to control_dispatch.log — no real dispatch."""

    def __init__(self, log_path: Path = _LOG_PATH, dry_run: bool = False):
        self._log_path = log_path
        self._dry_run = dry_run

    def dispatch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "dispatched_at": datetime.utcnow().isoformat() + "Z",
            "dry_run": self._dry_run,
            "payload": payload,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"StubDispatcher: failed to write log: {e}")
        if self._dry_run:
            logger.info(f"DRY-RUN dispatch: {json.dumps(payload)}")
        else:
            logger.info(f"STUB dispatch (logged): {payload.get('service')} → {payload.get('target')}")
        return {"status": "dispatched" if not self._dry_run else "dry_run", "payload": payload}


class HARestDispatcher(BaseDispatcher):
    """Real Home Assistant REST API dispatcher (stub — implement for production)."""

    def __init__(self, ha_url: str, ha_token: str, dry_run: bool = False):
        self._ha_url = ha_url
        self._ha_token = ha_token
        self._dry_run = dry_run

    def dispatch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._dry_run:
            logger.info(f"HA dry-run: {payload}")
            return {"status": "dry_run", "payload": payload}
        # Production implementation:
        # import requests
        # domain = payload["domain"]
        # service = payload["service"]
        # url = f"{self._ha_url}/api/services/{domain}/{service}"
        # headers = {"Authorization": f"Bearer {self._ha_token}"}
        # resp = requests.post(url, json=payload.get("service_data", {}), headers=headers, timeout=10)
        # return {"status": "ok" if resp.ok else "error", "http_status": resp.status_code}
        raise NotImplementedError(
            "HARestDispatcher is not yet connected to a real HA instance. "
            "Use StubDispatcher for development."
        )
