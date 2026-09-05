"""Unit and contract tests explicitly enforcing the critical invariant:
NO VERIFIED REMOTE UPLOAD -> NO LOCAL DELETE.
"""

import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from apps.upload_api.main import app, set_collector_service
from apps.upload_api.service import CollectorService
from apps.upload_api.storage import ChunkStorageManager


@pytest.fixture
def clean_service(tmp_path):
    storage = ChunkStorageManager(base_staging_dir=tmp_path / "staging")
    svc = CollectorService(storage_manager=storage)
    set_collector_service(svc)
    return svc, tmp_path


def test_invariant_checksum_mismatch_fails_closed(clean_service):
    """
    Simulates a client with a corrupted upload or wrong advertised checksum.
    The server MUST reject verification and mark status FAILED.
    Client code MUST check verified == True before deleting local video.
    """
    svc, tmp_path = clean_service
    client = TestClient(app)

    # 1. Initiate upload
    init_payload = {
        "task_id": "TASK-0001",
        "collector_id": "COL-007",
        "file_size_bytes": 100,
        "total_chunks": 1,
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",  # False hash
        "metadata": {
            "schema_version": "1.0",
            "experiment_id": "EXP001",
            "run_id": "RUN-0041",
            "recording_id": "EXP001_RUN-0041_CAM-01",
            "collector_id": "COL-007",
            "camera_id": "CAM-01",
            "scenario_type": "nominal",
            "object": "RED_COMPONENT",
            "target": "TARGET_A",
            "duration_seconds": 35.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "orientation": "landscape",
            "file_size_bytes": 100,
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "app_version": "0.1.0",
            "protocol_version": "EXP001-v1.0",
            "created_at": "2026-09-05T20:00:00Z",
        },
    }
    init_resp = client.post("/api/v1/collector/uploads/initiate", json=init_payload)
    assert init_resp.status_code == 200
    upload_id = init_resp.json()["upload_id"]

    # 2. Upload actual chunk (100 bytes of 'X')
    chunk_data = b"X" * 100
    chunk_resp = client.put(f"/api/v1/collector/uploads/{upload_id}/chunks/0", content=chunk_data)
    assert chunk_resp.status_code == 200

    # 3. Client calls complete with false expected SHA-256
    complete_payload = {
        "upload_id": upload_id,
        "expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "metadata": init_payload["metadata"],
    }
    comp_resp = client.post(f"/api/v1/collector/uploads/{upload_id}/complete", json=complete_payload)
    assert comp_resp.status_code == 200
    res = comp_resp.json()

    # Crucial assertions:
    assert res["verified"] is False
    assert res["status"] == "failed"
    assert res["sha256_match"] is False
    assert "Checksum mismatch" in res["error_message"]

    # 4. Invariant assertion: Local deletion client simulation
    # The client simulates checking 'verified':
    local_video_file = tmp_path / "local_phone_video.mp4"
    local_video_file.write_bytes(chunk_data)

    def simulate_client_cleanup(server_response: dict, local_path: Path) -> bool:
        if server_response.get("verified") is True and server_response.get("status") == "verified":
            local_path.unlink()
            return True
        return False

    deleted = simulate_client_cleanup(res, local_video_file)
    assert deleted is False
    assert local_video_file.exists(), "INVARIANT VIOLATION: Local file was deleted without verified remote upload!"
