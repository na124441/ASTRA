"""Integration tests for FastAPI and WebSocket edge endpoints."""

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app, set_orchestrator
from astra.runtime.orchestrator import EdgeRuntimeOrchestrator
from astra.video.camera import MockCamera


@pytest.fixture
def test_client(tmp_path):
    """Fixture providing isolated EdgeRuntimeOrchestrator and TestClient."""
    db_file = tmp_path / "test_api.db"
    camera = MockCamera()
    orch = EdgeRuntimeOrchestrator(
        camera=camera,
        db_path=db_file,
        tts_enabled=True,
        mock_tts=True,
    )
    set_orchestrator(orch)

    # Use TestClient without background lifespan to avoid background camera thread contention in tests
    with TestClient(app) as client:
        yield client, orch

    orch.shutdown()


def test_health_and_dashboard(test_client):
    client, _ = test_client

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert data["system"] == "ASTRA-E"

    # 2. Dashboard UI
    res_ui = client.get("/dashboard")
    assert res_ui.status_code == 200
    assert "ASTRA-E" in res_ui.text


def test_experiment_lifecycle_and_status(test_client):
    client, orch = test_client

    # 1. Start Experiment
    start_res = client.post("/api/v1/experiment/start", json={"experiment_id": "EXP001"})
    assert start_res.status_code == 200
    data = start_res.json()
    assert data["status"] == "STARTED"
    run_id = data["run_id"]
    assert run_id.startswith("RUN-")

    # 2. Query Status
    status_res = client.get("/api/v1/experiment/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["run_id"] == run_id
    assert status_data["status"] == "RUNNING"
    assert status_data["procedure"]["id"] == "PROC-EXP-001"

    # 3. Pause
    pause_res = client.post("/api/v1/experiment/pause")
    assert pause_res.status_code == 200
    assert pause_res.json()["status"] == "PAUSED"

    # 4. Resume
    resume_res = client.post("/api/v1/experiment/resume")
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "RUNNING"

    # 5. Reset
    reset_res = client.post("/api/v1/experiment/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "IDLE"


def test_runs_and_audit_report(test_client):
    client, orch = test_client

    # Start run and step a frame to generate telemetry
    start_res = client.post("/api/v1/experiment/start", json={"experiment_id": "EXP001"})
    run_id = start_res.json()["run_id"]

    orch.step_frame()

    # List runs
    runs_res = client.get("/api/v1/experiment/runs")
    assert runs_res.status_code == 200
    runs = runs_res.json()
    assert len(runs) >= 1
    assert any(r["run_id"] == run_id for r in runs)

    # Export audit report
    report_res = client.get(f"/api/v1/experiment/runs/{run_id}/report")
    assert report_res.status_code == 200
    report = report_res.json()
    assert report["run_id"] == run_id
    assert "total_events" in report


def test_audio_assistance_endpoint(test_client):
    client, orch = test_client

    res = client.post("/api/v1/assistance/speak", json={"text": "Step complete", "priority": "HIGH"})
    assert res.status_code == 200
    data = res.json()
    assert data["spoken"] is True
    assert data["text"] == "Step complete"


def test_websocket_telemetry_streaming(test_client):
    client, orch = test_client

    # Connect to WebSocket endpoint
    with client.websocket_connect("/ws/telemetry") as ws:
        # Should receive initial telemetry packet
        msg = ws.receive_json()
        assert "system" in msg
        assert msg["system"] == "ASTRA-E"
        assert "procedure" in msg
