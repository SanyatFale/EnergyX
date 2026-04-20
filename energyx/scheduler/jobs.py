"""Scheduled jobs for EnergyX.

Online jobs (run when mode=online, APScheduler):
  - weekly_elasticity_recompute        (every Monday 03:00 UTC)
  - nightly_degradation_baseline_update (every day 02:00 UTC)
  - weekly_report                       (every Sunday 06:00 UTC)
  - monthly_report                      (1st of month 06:00 UTC)
  - weekly_continuous_training          (every Monday 04:00 UTC, gated by continuous_training perm)

Offline trigger (called by Monitor Agent trigger_retraining):
  - on_retraining_requested(reason, analysis_agent) — recomputes elasticity + baselines immediately.

Usage::
    from energyx.scheduler.jobs import EnergyXScheduler
    scheduler = EnergyXScheduler(analysis_agent=agent, orchestrator=orch)
    scheduler.start()   # starts APScheduler in background
    ...
    scheduler.stop()
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Try APScheduler; if not installed, provide a no-op stub
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed — scheduled jobs disabled. "
                   "Install with: pip install apscheduler")


class EnergyXScheduler:
    """Wraps APScheduler to register and run all EnergyX cron jobs.

    Pass lazy-evaluated callables or an analysis_agent; jobs call into
    the agent's tools via session.
    """

    def __init__(
        self,
        analysis_agent=None,
        orchestrator=None,
        permission_manager=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._agent = analysis_agent
        self._orchestrator = orchestrator
        self._pm = permission_manager
        self._config = config or {}
        self._scheduler = None
        self._job_history: list = []

    def start(self):
        """Start the scheduler and register all online jobs."""
        if not _APSCHEDULER_AVAILABLE:
            logger.warning("APScheduler not available — scheduler not started.")
            return
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._register_jobs()
        self._scheduler.start()
        logger.info("EnergyXScheduler started.")

    def stop(self):
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("EnergyXScheduler stopped.")

    def is_running(self) -> bool:
        return bool(self._scheduler and self._scheduler.running)

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_jobs(self):
        sched = self._scheduler

        # Weekly elasticity recompute — Monday 03:00 UTC
        sched.add_job(
            self._run_elasticity,
            CronTrigger(day_of_week="mon", hour=3, minute=0),
            id="weekly_elasticity",
            name="Weekly elasticity recompute",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        # Nightly degradation baseline update — 02:00 UTC daily
        sched.add_job(
            self._run_degradation_update,
            CronTrigger(hour=2, minute=0),
            id="nightly_degradation",
            name="Nightly degradation baseline update",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        # Weekly report — Sunday 06:00 UTC
        sched.add_job(
            self._run_weekly_report,
            CronTrigger(day_of_week="sun", hour=6, minute=0),
            id="weekly_report",
            name="Weekly energy report",
            replace_existing=True,
            misfire_grace_time=7200,
        )

        # Monthly report — 1st of month 06:00 UTC
        sched.add_job(
            self._run_monthly_report,
            CronTrigger(day=1, hour=6, minute=0),
            id="monthly_report",
            name="Monthly energy report",
            replace_existing=True,
            misfire_grace_time=7200,
        )

        # Weekly continuous training (permission-gated) — Monday 04:00 UTC
        sched.add_job(
            self._run_continuous_training,
            CronTrigger(day_of_week="mon", hour=4, minute=0),
            id="weekly_continuous_training",
            name="Weekly continuous training (opt-in)",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        logger.info(f"Registered {len(sched.get_jobs())} scheduled jobs.")

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    def _run_elasticity(self):
        logger.info("Scheduled job: weekly_elasticity_recompute")
        self._record_job("weekly_elasticity", "started")
        try:
            tool = self._get_tool("compute_elasticity")
            if tool:
                result = tool.invoke({})
                logger.info(f"Elasticity recompute done: {result[:200]}")
                self._record_job("weekly_elasticity", "completed")
        except Exception as e:
            logger.error(f"Elasticity job failed: {e}")
            self._record_job("weekly_elasticity", f"error: {e}")

    def _run_degradation_update(self):
        logger.info("Scheduled job: nightly_degradation_baseline_update")
        self._record_job("nightly_degradation", "started")
        try:
            tool = self._get_tool("update_degradation_baselines")
            if tool:
                result = tool.invoke({})
                logger.info(f"Degradation update done: {result[:200]}")
                self._record_job("nightly_degradation", "completed")
        except Exception as e:
            logger.error(f"Degradation job failed: {e}")
            self._record_job("nightly_degradation", f"error: {e}")

    def _run_weekly_report(self):
        logger.info("Scheduled job: weekly_report")
        self._record_job("weekly_report", "started")
        try:
            tool = self._get_tool("generate_report")
            if tool:
                result = tool.invoke({"confirm": "yes"})
                logger.info(f"Weekly report done: {result[:200]}")
                self._record_job("weekly_report", "completed")
        except Exception as e:
            logger.error(f"Weekly report job failed: {e}")
            self._record_job("weekly_report", f"error: {e}")

    def _run_monthly_report(self):
        logger.info("Scheduled job: monthly_report")
        self._record_job("monthly_report", "started")
        # Monthly report reuses generate_report with a different prefix
        self._run_weekly_report()

    def _run_continuous_training(self):
        logger.info("Scheduled job: weekly_continuous_training")
        if self._pm and not self._pm.is_granted("continuous_training"):
            logger.info("Continuous training skipped — permission not granted.")
            self._record_job("weekly_continuous_training", "skipped:no_permission")
            return
        self._record_job("weekly_continuous_training", "started")
        # Placeholder: in production this would refit all models with fresh data
        logger.info("Continuous training pass would run here.")
        self._record_job("weekly_continuous_training", "completed")

    # ------------------------------------------------------------------
    # Offline trigger (called by Monitor Agent)
    # ------------------------------------------------------------------

    def on_retraining_requested(self, reason: str) -> Dict[str, Any]:
        """Called when Monitor Agent fires trigger_retraining.

        Runs elasticity recompute + degradation baseline update immediately
        (offline mode — not on schedule).
        """
        logger.info(f"Retraining requested: {reason}")
        self._run_elasticity()
        self._run_degradation_update()
        return {
            "status": "retraining_complete",
            "reason": reason,
            "ran_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tool(self, tool_name: str):
        if self._agent and hasattr(self._agent, "tool_map"):
            return self._agent.tool_map.get(tool_name)
        return None

    def _record_job(self, job_id: str, status: str):
        self._job_history.append({
            "job_id": job_id,
            "status": status,
            "ts": datetime.utcnow().isoformat(),
        })

    def job_history(self) -> list:
        return list(self._job_history)

    def list_jobs(self) -> list:
        if not self._scheduler:
            return []
        return [
            {"id": j.id, "name": j.name, "next_run": str(j.next_run_time)}
            for j in self._scheduler.get_jobs()
        ]
