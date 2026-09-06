"""
============================================================================
OWNER: Backend Developer 2 (Kratika)
PURPOSE: Comprehensive unit tests for Work Package Tasks 1, 2 & 3.
         Tests demo inference (JSON & multipart), health monitoring endpoint,
         bug report submission endpoint, and router integration in FastAPI.
============================================================================
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app

client = TestClient(app)


def test_demo_inference_json_nominal():
    """Task 1: Verify POST /api/v1/demo/inference with JSON payload for sample-01 (nominal)."""
    response = client.post(
        "/api/v1/demo/inference",
        json={"clip_id": "sample-01"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["step_id"] == 3
    assert data["action_name"] == "Place Red Vial in Slot A"
    assert data["status"] == "NOMINAL"
    assert data["confidence"] == 0.942
    assert 110 <= data["inference_ms"] <= 140
    assert data["anomaly_detected"] is False
    assert "timestamp" in data


def test_demo_inference_json_fault():
    """Task 1: Verify POST /api/v1/demo/inference with JSON payload for sample-02 (fault)."""
    response = client.post(
        "/api/v1/demo/inference",
        json={"clip_id": "sample-02"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["step_id"] == 4
    assert data["status"] == "FAULT"
    assert data["anomaly_detected"] is True


def test_demo_inference_multipart_file():
    """Task 1: Verify POST /api/v1/demo/inference with multipart/form-data video upload."""
    files = {"file": ("test_clip.mp4", b"dummy video bytes", "video/mp4")}
    response = client.post("/api/v1/demo/inference", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["step_id"] == 3
    assert data["status"] == "NOMINAL"
    assert 110 <= data["inference_ms"] <= 140


def test_system_health():
    """Task 3: Verify GET /api/v1/health status endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gpu_status" in data
    assert "cpu_status" in data
    assert data["fps"] == 30.0
    assert data["model_readiness"]["action_classifier"] is True
    assert data["model_readiness"]["hoi_detector"] is True
    assert data["model_readiness"]["procedure_engine"] is True


def test_bug_report_submission():
    """Task 2: Verify POST /api/v1/telemetry/bug-report endpoint."""
    payload = {
        "title": "Camera latency spike",
        "description": "Observed frame drop during experiment EXP001",
        "severity": "high",
    }
    response = client.post("/api/v1/telemetry/bug-report", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["ticket_id"].startswith("ASTRA-BUG-")
    assert "timestamp" in data
