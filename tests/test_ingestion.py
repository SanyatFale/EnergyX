"""Tests for Step 9: FastAPI /ingest/tick endpoint.

pytest tests/test_ingestion.py
"""
import pytest

try:
    from fastapi.testclient import TestClient
    from energyx.api.ingest import app, set_orchestrator
    _FASTAPI = True
except ImportError:
    _FASTAPI = False


@pytest.mark.skipif(not _FASTAPI, reason="FastAPI not installed")
class TestIngestEndpoint:
    @pytest.fixture
    def client(self, tmp_path):
        from energyx.orchestrator.orchestrator import Orchestrator
        from energyx.orchestrator.mode_manager import Mode

        orch = Orchestrator(
            config={"home_id": "home001"},
            permissions_path=tmp_path / "perms.json",
            initial_mode=Mode.ONLINE,
        )
        # Prevent OnlineRunner from actually starting
        class FakeRunner:
            def process_tick(self, tick): return []
            def stop(self): pass

        orch._online_runner = FakeRunner()
        set_orchestrator(orch)
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ingest_tick_valid_payload(self, client):
        payload = {
            "home_id": "home001",
            "ts": "2026-04-20 14:00:00",
            "readings": [
                {"sensor_id": "elec_0", "sensor_type": "electricity_apparent", "value": 350.0, "unit": "watts"}
            ],
        }
        resp = client.post("/ingest/tick", json=payload)
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["n_readings"] == 1

    def test_ingest_tick_invalid_timestamp(self, client):
        payload = {
            "home_id": "home001",
            "ts": "not-a-timestamp",
            "readings": [
                {"sensor_id": "e0", "sensor_type": "electricity_apparent", "value": 300.0, "unit": "watts"}
            ],
        }
        resp = client.post("/ingest/tick", json=payload)
        assert resp.status_code == 422

    def test_ingest_tick_missing_readings(self, client):
        payload = {
            "home_id": "home001",
            "ts": "2026-04-20 14:00:00",
            "readings": [],
        }
        resp = client.post("/ingest/tick", json=payload)
        assert resp.status_code == 422

    def test_ingest_batch(self, client):
        payload = {
            "ticks": [
                {
                    "home_id": "home001",
                    "ts": "2026-04-20 14:00:00",
                    "readings": [
                        {"sensor_id": "e0", "sensor_type": "electricity_apparent", "value": 300.0, "unit": "watts"}
                    ],
                },
                {
                    "home_id": "home001",
                    "ts": "2026-04-20 14:00:01",
                    "readings": [
                        {"sensor_id": "e0", "sensor_type": "electricity_apparent", "value": 305.0, "unit": "watts"}
                    ],
                },
            ]
        }
        resp = client.post("/ingest/batch", json=payload)
        assert resp.status_code == 202
        assert resp.json()["total_ticks"] == 2

    def test_ingest_offline_mode_rejected(self, tmp_path):
        """Ticks to an offline-mode orchestrator should return 409."""
        from energyx.orchestrator.orchestrator import Orchestrator
        from energyx.orchestrator.mode_manager import Mode

        orch = Orchestrator(
            config={"home_id": "home002"},
            permissions_path=tmp_path / "perms2.json",
            initial_mode=Mode.OFFLINE,
        )
        set_orchestrator(orch)
        client = TestClient(app)
        payload = {
            "home_id": "home002",
            "ts": "2026-04-20 14:00:00",
            "readings": [
                {"sensor_id": "e0", "sensor_type": "electricity_apparent", "value": 300.0, "unit": "watts"}
            ],
        }
        resp = client.post("/ingest/tick", json=payload)
        assert resp.status_code == 409
