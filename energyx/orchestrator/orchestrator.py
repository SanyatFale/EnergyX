"""Central Orchestrator for EnergyX.

Responsibilities:
  - Classify incoming queries → route to correct agent
  - Own mode state (online / offline)
  - Permission manager gate
  - Event escalation (online: subscribe to bus; offline: receive from Monitor Agent)
  - Plan approval gate delegation to Analysis Agent
  - Invoke Monitor Agent on each offline batch append
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from energyx.data.events import BaseEvent
from energyx.monitoring.bus import EventBus, get_event_bus
from energyx.orchestrator.event_escalation import EventEscalator
from energyx.orchestrator.mode_manager import Mode, ModeManager
from energyx.orchestrator.permission_manager import PermissionManager
from energyx.orchestrator.router import classify_query

logger = logging.getLogger(__name__)


class Orchestrator:
    """Top-level coordinator for the EnergyX multi-agent system.

    Usage (offline mode)::

        orch = Orchestrator(config={"home_id": "home001"})
        # Batch append
        result = orch.append_batch(df)
        # Query routing
        response = orch.handle_query("forecast next 7 days", dataset_path="...", ...)

    Usage (online mode)::

        orch = Orchestrator(config={"home_id": "home001"})
        orch.go_online()
        # Tick ingestion (called by /ingest/tick endpoint)
        orch.ingest_tick(tick_dict)
        # Query routing works identically
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        bus: Optional[EventBus] = None,
        permissions_path=None,
        initial_mode: Mode = Mode.OFFLINE,
    ):
        self._config = config or {}
        self._home_id = self._config.get("home_id", "unknown")

        # Core services
        self.permissions = PermissionManager(path=permissions_path)
        self.mode_manager = ModeManager(initial_mode=initial_mode)
        self.bus = bus or get_event_bus()

        # Online runner (lazy-initialized)
        self._online_runner = None

        # Monitor Agent (lazy-initialized)
        self._monitor_agent = None

        # Historic store
        self._historic_store = None

        # Escalator
        self.escalator = EventEscalator(
            permission_manager=self.permissions,
            is_online_fn=self.mode_manager.is_online,
            ui_handler=self._ui_handler,
            analysis_handler=self._analysis_handler,
            control_handler=self._control_handler,
        )
        self.escalator.register_with_bus(self.bus)

        # Pending analysis requests (from monitoring escalations)
        self._pending_analysis: List[BaseEvent] = []
        # UI event inbox
        self._ui_events: List[BaseEvent] = []

        # Mode change hook
        self.mode_manager.on_mode_change(self._on_mode_change)

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def go_online(self, runner_config: Optional[Dict[str, Any]] = None) -> None:
        """Switch to online mode and start the OnlineRunner."""
        from energyx.monitoring.online_runner import OnlineRunner
        cfg = {**self._config, **(runner_config or {})}
        self._online_runner = OnlineRunner(config=cfg, bus=self.bus)
        self._online_runner.start()
        self.mode_manager.go_online()
        logger.info(f"Orchestrator: switched to ONLINE mode (home={self._home_id})")

    def go_offline(self) -> None:
        """Switch to offline mode and stop the OnlineRunner if active."""
        if self._online_runner:
            self._online_runner.stop()
            self._online_runner = None
        self.mode_manager.go_offline()
        logger.info(f"Orchestrator: switched to OFFLINE mode (home={self._home_id})")

    # ------------------------------------------------------------------
    # Online tick ingestion
    # ------------------------------------------------------------------

    def ingest_tick(self, tick: Dict[str, Any]) -> List[BaseEvent]:
        """Online mode: process a live tick through the OnlineRunner.

        Also writes to Historic Store.
        """
        events: List[BaseEvent] = []
        if self._online_runner:
            events = self._online_runner.process_tick(tick)
        # Persist to Historic Store
        try:
            store = self._get_store()
            df = pd.DataFrame([tick])
            store.write_ticks(self._home_id, df)
        except Exception as e:
            logger.error(f"Orchestrator.ingest_tick: store write failed: {e}")
        return events

    def inject_dr_signal(self, signal: Dict[str, Any]) -> List[BaseEvent]:
        """Inject a DR signal into the online runner."""
        if self._online_runner:
            return self._online_runner.inject_dr_signal(signal)
        return []

    # ------------------------------------------------------------------
    # Offline batch append
    # ------------------------------------------------------------------

    def append_batch(
        self,
        batch: pd.DataFrame,
        batch_id: Optional[str] = None,
        on_tool_call: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Offline mode: append a batch → invoke Monitor Agent → return results.

        Writes batch to Historic Store; Monitor Agent events go to bus + store.
        """
        # Persist raw batch
        try:
            store = self._get_store()
            store.write_ticks(self._home_id, batch)
        except Exception as e:
            logger.error(f"Orchestrator.append_batch: store write failed: {e}")

        agent = self._get_monitor_agent()
        result = agent.run(batch, batch_id=batch_id, on_tool_call=on_tool_call)
        return result

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    def handle_query(
        self,
        query: str,
        dataset_path: Optional[str] = None,
        time_column: Optional[str] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        plan_callback: Optional[Callable] = None,
        tool_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Classify and route a user query.

        Returns a dict with at least: {intent, response, ...agent-specific fields}
        """
        intent = classify_query(query)

        if intent == "knowledge":
            return self._route_knowledge(query)

        if intent == "control":
            return self._route_control(query)

        if intent == "status":
            return self._route_status(query)

        # intent == "analysis" (default)
        return self._route_analysis(
            query, dataset_path, time_column, target_column,
            feature_columns, output_dir, plan_callback, tool_callback,
        )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _route_analysis(
        self,
        query: str,
        dataset_path: Optional[str],
        time_column: Optional[str],
        target_column: Optional[str],
        feature_columns: Optional[List[str]],
        output_dir: Optional[str],
        plan_callback: Optional[Callable],
        tool_callback: Optional[Callable],
    ) -> Dict[str, Any]:
        if not dataset_path or not time_column or not target_column:
            return {
                "intent": "analysis",
                "response": "Analysis requires dataset_path, time_column, and target_column.",
                "error": True,
            }
        try:
            from energyx.agents.analysis.agent import AnalysisAgent
            agent = AnalysisAgent(
                dataset_path=dataset_path,
                time_column=time_column,
                target_column=target_column,
                output_dir=output_dir,
                feature_columns=feature_columns,
            )
            plan = agent.plan(query)
            if plan_callback:
                plan_callback(plan)
            result = agent.execute(plan, on_tool_call=tool_callback)
            result["intent"] = "analysis"
            return result
        except Exception as e:
            logger.error(f"Analysis routing failed: {e}")
            return {"intent": "analysis", "response": f"Analysis failed: {e}", "error": True}

    def _route_knowledge(self, query: str) -> Dict[str, Any]:
        try:
            from energyx.agents.knowledge.agent import KnowledgeAgent
            agent = KnowledgeAgent()
            return {"intent": "knowledge", "response": agent.answer(query)}
        except Exception as e:
            return {"intent": "knowledge", "response": f"Knowledge agent unavailable: {e}"}

    def _route_control(self, query: str) -> Dict[str, Any]:
        if not self.mode_manager.is_online():
            return {
                "intent": "control",
                "response": "Control Agent is disabled in offline mode. Actions are advisory only.",
                "advisory": True,
            }
        try:
            from energyx.agents.control.agent import ControlAgent
            agent = ControlAgent(permission_manager=self.permissions)
            return {"intent": "control", **agent.handle(query)}
        except Exception as e:
            return {"intent": "control", "response": f"Control agent error: {e}"}

    def _route_status(self, query: str) -> Dict[str, Any]:
        ui_events = self.escalator.pop_ui_events()
        advisory = self.escalator.pop_advisory()
        return {
            "intent": "status",
            "mode": self.mode_manager.mode.value,
            "recent_events": [e.to_dict() for e in ui_events[-20:]],
            "advisory_actions": [e.to_dict() for e in advisory],
            "permissions": self.permissions.all_permissions(),
            "response": f"Mode: {self.mode_manager.mode.value}. {len(ui_events)} recent events.",
        }

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    def _ui_handler(self, event: BaseEvent) -> None:
        self._ui_events.append(event)

    def _analysis_handler(self, event: BaseEvent) -> None:
        self._pending_analysis.append(event)
        logger.info(f"Orchestrator: queued {event.type} for Analysis Agent")

    def _control_handler(self, event: BaseEvent) -> None:
        if not self.mode_manager.is_online():
            return
        logger.info(f"Orchestrator: forwarding {event.type} to Control Agent")

    def _on_mode_change(self, mode: Mode) -> None:
        logger.info(f"Orchestrator: mode changed to {mode.value}")

    # ------------------------------------------------------------------
    # Lazy init helpers
    # ------------------------------------------------------------------

    def _get_store(self):
        if self._historic_store is None:
            from energyx.data.historic_store import get_historic_store
            self._historic_store = get_historic_store()
        return self._historic_store

    def _get_monitor_agent(self):
        if self._monitor_agent is None:
            from energyx.agents.monitor.agent import MonitorAgent
            self._monitor_agent = MonitorAgent(
                config=self._config,
                historic_store=self._get_store(),
                bus=self.bus,
            )
        return self._monitor_agent
