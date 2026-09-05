"""Unit tests for task dispenser, seeding, and inspection."""

import pytest
from fastapi.testclient import TestClient

from apps.upload_api.main import app, set_collector_service
from apps.upload_api.schemas import CollectionTask, TaskStatus
from apps.upload_api.service import CollectorService
from apps.upload_api.storage import ChunkStorageManager


@pytest.fixture
def clean_service(tmp_path):
    storage = ChunkStorageManager(base_staging_dir=tmp_path / "staging")
    svc = CollectorService(storage_manager=storage)
    set_collector_service(svc)
    return svc


def test_get_next_task_assignment(clean_service):
    client = TestClient(app)

    # Fetch next task
    resp = client.get("/api/v1/collector/tasks/next?collector_id=COL-001")
    assert resp.status_code == 200
    task = resp.json()
    assert task is not None
    assert task["task_id"] == "TASK-0001"
    assert task["status"] == "assigned"
    assert task["assigned_to"] == "COL-001"
    assert task["scenario_type"] == "nominal"
    assert len(task["procedure_steps"]) > 0

    # Fetching again for same collector returns the same in-progress task
    resp2 = client.get("/api/v1/collector/tasks/next?collector_id=COL-001")
    assert resp2.status_code == 200
    assert resp2.json()["task_id"] == "TASK-0001"

    # Fetching for different collector returns next task
    resp3 = client.get("/api/v1/collector/tasks/next?collector_id=COL-002")
    assert resp3.status_code == 200
    assert resp3.json()["task_id"] == "TASK-0002"
    assert resp3.json()["assigned_to"] == "COL-002"


def test_get_task_details_and_seed(clean_service):
    client = TestClient(app)

    # Detail lookup
    resp = client.get("/api/v1/collector/tasks/TASK-0001")
    assert resp.status_code == 200
    assert resp.json()["experiment_id"] == "EXP001"

    # Seed custom task
    seed_payload = {
        "tasks": [
            {
                "task_id": "TASK-9999",
                "experiment_id": "EXP001",
                "run_id": "RUN-0999",
                "camera_id": "CAM-01",
                "scenario_type": "fault_injection",
                "required_object": "BLUE_COMPONENT",
                "target": "TARGET_C",
                "duration_min": 20,
                "duration_max": 40,
                "orientation": "landscape",
                "instruction_version": "EXP001-v1.0",
                "procedure_steps": ["Step 1", "Step 2"],
                "status": "available",
            }
        ]
    }
    seed_resp = client.post("/api/v1/collector/tasks/seed", json=seed_payload)
    assert seed_resp.status_code == 200
    assert seed_resp.json()["seeded"] == 1

    lookup_resp = client.get("/api/v1/collector/tasks/TASK-9999")
    assert lookup_resp.status_code == 200
    assert lookup_resp.json()["required_object"] == "BLUE_COMPONENT"
