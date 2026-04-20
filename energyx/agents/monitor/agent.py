"""Monitor Agent — offline LLM-wrapped batch analyzer.

Follows the same plan/execute pattern as the Analysis Agent (TinyTSAgent).
Calls the shared monitor core functions over a batch; the LLM:
  1. Selects which monitors to run (plan phase)
  2. Calls run_monitor() for each (execute phase — tool-calling loop)
  3. Synthesizes a narrative summary from collected events

Does NOT dispatch to Control Agent.  Events are written to the Historic Store
and published on the shared Event Bus for Orchestrator-side escalation.

Reuses the Cerebras text-mode tool-call format adapter from TinyTSAgent.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from energyx.data.events import BaseEvent, RetrainingRequestEvent
from energyx.monitoring.bus import get_event_bus
from energyx.monitoring.monitors import MONITOR_REGISTRY

logger = logging.getLogger(__name__)

# Monitors that require liveness — skip by default in offline mode
_ONLINE_ONLY = {"ForgotToTurnOffDetector", "DemandResponseListener"}

# ----- System prompts -------------------------------------------------

_MONITOR_AGENT_SYSTEM = """You are EnergyX Monitor Agent, analyzing a historical batch of energy data.

PROTOCOL:
- Call ONE tool per response. No text commentary, just the tool call.
- After all monitors are run, call summarize_batch_findings() with a comprehensive summary.
- If you detect model drift or significant anomalies, call trigger_retraining().

TOOLS:
- select_applicable_monitors(batch_metadata_json) — select monitors for this batch
- run_monitor(monitor_name, batch_summary_json) — run one monitor over the batch
- summarize_batch_findings(events_json, stats_json) — produce narrative summary
- trigger_retraining(reason) — request Analysis Agent to refresh models

RULES:
- Skip ForgotToTurnOffDetector and DemandResponseListener unless user opts in.
- Run monitors in this order: AnomalyDetectors, ApplianceBaselineWatcher,
  HabitDriftTracker, CarbonTracker, RealtimeCostMeter, BudgetTrajectoryTracker,
  HistoricStoreWriter.
- Do NOT make up data — base all claims on tool outputs only.
"""


class MonitorAgent:
    """Offline batch analyzer.  Runs shared monitor core + LLM narrative.

    Args:
        config: dict with home_id, household_id, budget_cap_gbp, etc.
        historic_store: HistoricStore instance for persisting events.
        bus: EventBus to publish collected events.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        historic_store=None,
        bus=None,
    ):
        self._config = config or {}
        self._home_id = self._config.get("home_id", "unknown")
        self._historic_store = historic_store
        self._bus = bus or get_event_bus()
        self._collected_events: List[BaseEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        batch: pd.DataFrame,
        batch_id: Optional[str] = None,
        include_online_only: bool = False,
        on_tool_call: Optional[Callable] = None,
        max_steps: int = 20,
    ) -> Dict[str, Any]:
        """Analyze a batch and return events + narrative.

        Returns::
            {
                "events": [...],       # list of event dicts
                "narrative": "...",    # LLM narrative summary
                "retraining_triggered": bool,
                "batch_id": batch_id,
            }
        """
        import uuid
        batch_id = batch_id or str(uuid.uuid4())[:8]
        self._collected_events = []

        tools = self._create_tools(batch, include_online_only)
        tool_map = {t.name: t for t in tools}

        from tinyts.config import get_llm, settings
        llm = get_llm(temperature=settings.routing_temperature)
        llm_with_tools = llm.bind_tools(tools)

        batch_meta = self._describe_batch(batch, batch_id)
        messages = [
            SystemMessage(content=_MONITOR_AGENT_SYSTEM),
            HumanMessage(content=f"Analyze this batch:\n{json.dumps(batch_meta, indent=2)}"),
        ]

        narrative = ""
        retraining_triggered = False

        for step in range(max_steps):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error(f"MonitorAgent LLM call failed step {step}: {e}")
                time.sleep(2)
                try:
                    response = llm_with_tools.invoke(messages)
                except Exception as e2:
                    narrative = f"Agent failed: {e2}"
                    break

            tool_calls = response.tool_calls
            if not tool_calls and response.content:
                parsed = self._parse_text_tool_call(response.content, tool_map)
                if parsed:
                    fid = f"text_{step}"
                    tool_calls = [{"name": parsed["name"], "args": parsed["args"], "id": fid}]
                    response = AIMessage(
                        content="",
                        tool_calls=[{"name": parsed["name"], "args": parsed["args"], "id": fid}],
                    )

            messages.append(response)

            if not tool_calls:
                narrative = response.content or ""
                break

            for tc in tool_calls:
                name, args, tid = tc["name"], tc["args"], tc["id"]
                fn = tool_map.get(name)
                if fn is None:
                    res = json.dumps({"error": f"Unknown tool: {name}"})
                else:
                    try:
                        res = fn.invoke(args)
                    except Exception as e:
                        res = json.dumps({"error": f"{name} failed: {e}"})

                if name == "trigger_retraining":
                    retraining_triggered = True

                messages.append(ToolMessage(content=res, tool_call_id=tid))
                if on_tool_call:
                    on_tool_call(name, args, res)

        # Persist events
        event_dicts = [e.to_dict() if isinstance(e, BaseEvent) else e
                       for e in self._collected_events]
        if self._historic_store and event_dicts:
            try:
                self._historic_store.write_events(event_dicts)
            except Exception as e:
                logger.error(f"MonitorAgent: failed to write events: {e}")

        # Publish to bus
        for e in self._collected_events:
            try:
                self._bus.publish(e)
            except Exception as ex:
                logger.error(f"MonitorAgent: bus publish failed: {ex}")

        # Persist batch summary
        summary_record = {
            "batch_id": batch_id,
            "home_id": self._home_id,
            "narrative": narrative,
            "n_events": len(event_dicts),
            "retraining_triggered": retraining_triggered,
            "event_types": list({e.get("type") for e in event_dicts}),
        }
        if self._historic_store:
            try:
                self._historic_store.write_batch_summary(summary_record)
            except Exception as e:
                logger.error(f"MonitorAgent: failed to write batch summary: {e}")

        return {
            "events": event_dicts,
            "narrative": narrative,
            "retraining_triggered": retraining_triggered,
            "batch_id": batch_id,
        }

    # ------------------------------------------------------------------
    # Tool factory
    # ------------------------------------------------------------------

    def _create_tools(self, batch: pd.DataFrame, include_online_only: bool):
        agent = self

        @tool
        def select_applicable_monitors(batch_metadata_json: str) -> str:
            """Select which monitors to run on this batch.
            Returns JSON list of monitor names."""
            available = [
                name for name in MONITOR_REGISTRY
                if "offline" in MONITOR_REGISTRY[name].applicable_modes
                or (include_online_only and name not in _ONLINE_ONLY)
            ]
            if not include_online_only:
                available = [n for n in available if n not in _ONLINE_ONLY]
            return json.dumps({"monitors": available, "skipped": list(_ONLINE_ONLY)})

        @tool
        def run_monitor(monitor_name: str, batch_summary_json: str = "{}") -> str:
            """Run a named monitor over the batch. Returns collected events as JSON."""
            cls = MONITOR_REGISTRY.get(monitor_name)
            if cls is None:
                return json.dumps({"error": f"Unknown monitor: {monitor_name}"})
            cfg = {**agent._config}
            try:
                monitor = cls(config=cfg)
                events = monitor.evaluate(batch)
                agent._collected_events.extend(events)
                event_dicts = [e.to_dict() if isinstance(e, BaseEvent) else e for e in events]
                return json.dumps({
                    "monitor": monitor_name,
                    "n_events": len(events),
                    "events": event_dicts[:20],  # cap to avoid huge messages
                })
            except Exception as e:
                return json.dumps({"error": f"{monitor_name} failed: {e}"})

        @tool
        def summarize_batch_findings(events_json: str, stats_json: str = "{}") -> str:
            """Produce a narrative summary of all batch findings.
            Returns the narrative text."""
            try:
                events = json.loads(events_json) if events_json else []
            except Exception:
                events = []
            n = len(events)
            types = {}
            for e in events:
                t = e.get("type", "unknown")
                types[t] = types.get(t, 0) + 1
            summary = f"Batch analysis complete. {n} events detected: {json.dumps(types)}."
            return json.dumps({"narrative": summary, "event_count": n})

        @tool
        def trigger_retraining(reason: str) -> str:
            """Request the Analysis Agent to refresh models or elasticity baselines.
            Call when batch reveals model drift or systematic degradation."""
            event = RetrainingRequestEvent(
                reason=reason,
                home_id=agent._home_id,
            )
            agent._collected_events.append(event)
            logger.info(f"MonitorAgent: retraining requested — {reason}")
            return json.dumps({"status": "retraining_requested", "reason": reason})

        return [select_applicable_monitors, run_monitor, summarize_batch_findings, trigger_retraining]

    # ------------------------------------------------------------------
    # Helpers (format adapter preserved from TinyTSAgent)
    # ------------------------------------------------------------------

    _TOOL_RE = re.compile(
        r'\b(select_applicable_monitors|run_monitor|summarize_batch_findings|trigger_retraining'
        r')\s*\(([^)]*)\)'
    )
    _ARG_RE = re.compile(
        r"(\w+)\s*=\s*('(?:[^']*)'|\"(?:[^\"]*)\"|[^,)]+?)\s*(?:,|$)"
    )

    def _parse_text_tool_call(self, text: str, tool_map: dict) -> Optional[dict]:
        m = self._TOOL_RE.search(text)
        if not m or m.group(1) not in tool_map:
            return None
        name = m.group(1)
        raw = m.group(2).strip()
        if not raw:
            return {"name": name, "args": {}}
        args = {}
        for am in self._ARG_RE.finditer(raw):
            k, v = am.group(1), am.group(2).strip()
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1]
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            args[k] = v
        return {"name": name, "args": args}

    @staticmethod
    def _describe_batch(batch: pd.DataFrame, batch_id: str) -> Dict[str, Any]:
        if batch.empty:
            return {"batch_id": batch_id, "n_rows": 0}
        meta: Dict[str, Any] = {
            "batch_id": batch_id,
            "n_rows": len(batch),
            "columns": list(batch.columns),
        }
        if "ts" in batch.columns:
            meta["date_range"] = [str(batch["ts"].min()), str(batch["ts"].max())]
        if "value" in batch.columns:
            meta["value_stats"] = {
                "mean": round(float(batch["value"].mean()), 2),
                "std": round(float(batch["value"].std()), 2),
                "min": round(float(batch["value"].min()), 2),
                "max": round(float(batch["value"].max()), 2),
            }
        return meta
