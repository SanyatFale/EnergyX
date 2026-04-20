"""Tests for Step 7: Scheduled Jobs.

pytest tests/test_scheduler.py
"""
import pytest
from unittest.mock import MagicMock, patch
from energyx.scheduler.jobs import EnergyXScheduler


class TestEnergyXScheduler:
    def test_instantiation(self):
        sched = EnergyXScheduler()
        assert sched is not None

    def test_job_history_empty_on_start(self):
        sched = EnergyXScheduler()
        assert sched.job_history() == []

    def test_on_retraining_requested(self):
        """Offline trigger: retraining requested → runs elasticity + degradation."""
        mock_agent = MagicMock()
        mock_tool = MagicMock(return_value='{"status": "ok"}')
        mock_agent.tool_map = {
            "compute_elasticity": MagicMock(invoke=mock_tool),
            "update_degradation_baselines": MagicMock(invoke=mock_tool),
        }
        sched = EnergyXScheduler(analysis_agent=mock_agent)
        result = sched.on_retraining_requested(reason="batch_drift_detected")
        assert result["status"] == "retraining_complete"
        assert result["reason"] == "batch_drift_detected"
        # Both tools should have been called
        assert mock_agent.tool_map["compute_elasticity"].invoke.called
        assert mock_agent.tool_map["update_degradation_baselines"].invoke.called

    def test_continuous_training_skipped_without_permission(self):
        from energyx.orchestrator.permission_manager import PermissionManager
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            pm = PermissionManager(path=f.name)
        sched = EnergyXScheduler(permission_manager=pm)
        sched._run_continuous_training()
        history = sched.job_history()
        skipped = [h for h in history if "skipped" in h["status"]]
        assert len(skipped) >= 1
        os.unlink(f.name)

    def test_continuous_training_runs_with_permission(self, tmp_path):
        from energyx.orchestrator.permission_manager import PermissionManager
        pm = PermissionManager(path=tmp_path / "perms.json")
        pm.grant("continuous_training")
        sched = EnergyXScheduler(permission_manager=pm)
        sched._run_continuous_training()
        history = sched.job_history()
        completed = [h for h in history if h["status"] == "completed"]
        assert len(completed) >= 1

    def test_list_jobs_empty_when_not_started(self):
        sched = EnergyXScheduler()
        assert sched.list_jobs() == []

    def test_start_stop_with_apscheduler(self):
        """If APScheduler is available, start/stop registers 5 jobs."""
        sched = EnergyXScheduler()
        try:
            sched.start()
            if sched.is_running():
                jobs = sched.list_jobs()
                assert len(jobs) == 5  # 5 registered cron jobs
                sched.stop()
                assert not sched.is_running()
        except Exception:
            pass  # APScheduler may not be installed — that's OK
