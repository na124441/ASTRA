"""Integration test covering complete end-to-end ASTRA Collector upload pipeline."""

import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from apps.upload_api.audit_logger import CollectorAuditLogger
from apps.upload_api.hf_uploader import HuggingFaceDatasetUploader
from apps.upload_api.main import app, set_collector_service
from apps.upload_api.service import CollectorService
from apps.upload_api.storage import ChunkStorageManager


@pytest.fixture
def clean_environment(tmp_path):
    staging = tmp_path / "staging"
    audit_dir = tmp_path / "audit"
    storage = ChunkStorageManager(base_staging_dir=staging)
    audit = CollectorAuditLogger(log_dir=audit_dir)
    uploader = HuggingFaceDatasetUploader(mock_mode=True)
    svc = CollectorService(storage_manager=storage, hf_uploader=uploader, audit_logger=audit)
    set_collector_service(svc)
    return svc, tmp_path


def test_complete_e2e_collection_workflow(clean_environment):
    svc, tmp_path = clean_environment
    client = TestClient(app)

    # 1. Device Registration
    reg_resp = client.post(
        "/api/v1/collector/auth/register",
        json={
            "collector_id": "COL-042",
            "device_id": "device-uuid-phone-01",
            "app_version": "1.0.0",
            "device_model": "Google Pixel 8 Pro",
        },
    )
    assert reg_resp.status_code == 200
    token = reg_resp.json()["auth_token"]

    # 2. Get Next Task
    task_resp = client.get("/api/v1/collector/tasks/next?collector_id=COL-042")
    assert task_resp.status_code == 200
    task = task_resp.json()
    assert task["task_id"] == "TASK-0001"
    assert task["status"] == "assigned"
    assert task["run_id"] == "RUN-0041"
    assert task["camera_id"] == "CAM-01"

    # 3. Simulate Recording Video on Phone
    simulated_video_data = b"RIFF_HEADER_TEST_VIDEO_DATA_STREAM_" * 1000
    file_size = len(simulated_video_data)
    video_sha256 = hashlib.sha256(simulated_video_data).hexdigest()

    local_phone_video = tmp_path / "DCIM_Camera_EXP001_temp.mp4"
    local_phone_video.write_bytes(simulated_video_data)

    metadata = {
        "schema_version": "1.0",
        "experiment_id": task["experiment_id"],
        "run_id": task["run_id"],
        "recording_id": f"{task['experiment_id']}_{task['run_id']}_{task['camera_id']}",
        "collector_id": "COL-042",
        "camera_id": task["camera_id"],
        "scenario_type": task["scenario_type"],
        "object": task["required_object"],
        "target": task["target"],
        "duration_seconds": 42.5,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "orientation": "landscape",
        "file_size_bytes": file_size,
        "sha256": video_sha256,
        "app_version": "1.0.0",
        "protocol_version": "EXP001-v1.0",
        "created_at": "2026-09-05T21:00:00Z",
    }

    # 4. Initiate Upload (divide into 2 chunks)
    chunk_size = file_size // 2
    chunk0 = simulated_video_data[:chunk_size]
    chunk1 = simulated_video_data[chunk_size:]
    total_chunks = 2

    init_resp = client.post(
        "/api/v1/collector/uploads/initiate",
        json={
            "task_id": task["task_id"],
            "collector_id": "COL-042",
            "file_size_bytes": file_size,
            "total_chunks": total_chunks,
            "sha256": video_sha256,
            "metadata": metadata,
        },
    )
    assert init_resp.status_code == 200
    upload_id = init_resp.json()["upload_id"]

    # 5. Stream Chunks
    resp_c0 = client.put(
        f"/api/v1/collector/uploads/{upload_id}/chunks/0",
        content=chunk0,
        headers={"X-Chunk-SHA256": hashlib.sha256(chunk0).hexdigest()},
    )
    assert resp_c0.status_code == 200
    assert resp_c0.json()["chunks_completed"] == 1

    resp_c1 = client.put(
        f"/api/v1/collector/uploads/{upload_id}/chunks/1",
        content=chunk1,
        headers={"X-Chunk-SHA256": hashlib.sha256(chunk1).hexdigest()},
    )
    assert resp_c1.status_code == 200
    assert resp_c1.json()["chunks_completed"] == 2

    # 6. Complete and Finalize
    comp_resp = client.post(
        f"/api/v1/collector/uploads/{upload_id}/complete",
        json={
            "upload_id": upload_id,
            "expected_sha256": video_sha256,
            "metadata": metadata,
        },
    )
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert comp_data["verified"] is True
    assert comp_data["status"] == "verified"
    assert comp_data["remote_path"] == f"videos/exp001/{task['run_id']}/{task['camera_id']}.mp4"

    # 7. Query Audit Logs
    audit_resp = client.get(f"/api/v1/collector/audit/logs?upload_id={upload_id}")
    assert audit_resp.status_code == 200
    records = audit_resp.json()
    assert len(records) == 1
    assert records[0]["sha256"] == video_sha256
    assert records[0]["collector_id"] == "COL-042"

    # 8. Check Task Completion
    task_after = client.get(f"/api/v1/collector/tasks/{task['task_id']}").json()
    assert task_after["status"] == "completed"

    # 9. Verify Local Phone Deletion Handshake
    if comp_data["verified"] is True and comp_data["status"] == "verified":
        local_phone_video.unlink()

    assert not local_phone_video.exists(), "Local file should be deleted upon verified remote persistence"
