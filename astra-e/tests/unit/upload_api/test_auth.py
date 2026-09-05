"""Unit tests for device registration, authentication, and revocation."""

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
    return svc


def test_device_registration_and_login(clean_service):
    client = TestClient(app)

    # 1. Register device
    reg_payload = {
        "collector_id": "COL-007",
        "device_id": "dev-uuid-12345",
        "app_version": "0.1.0",
        "device_model": "Pixel 8 Pro",
    }
    resp = client.post("/api/v1/collector/auth/register", json=reg_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["collector_id"] == "COL-007"
    assert data["status"] == "active"
    token = data["auth_token"]
    assert token.startswith("astra_tok_")

    # 2. Re-register (check-in) returns same token
    resp2 = client.post("/api/v1/collector/auth/register", json=reg_payload)
    assert resp2.status_code == 200
    assert resp2.json()["auth_token"] == token

    # 3. Login verification
    login_payload = {
        "collector_id": "COL-007",
        "device_id": "dev-uuid-12345",
        "auth_token": token,
    }
    login_resp = client.post("/api/v1/collector/auth/login", json=login_payload)
    assert login_resp.status_code == 200
    assert login_resp.json()["status"] == "active"


def test_device_revocation_blocks_access(clean_service):
    client = TestClient(app)

    # 1. Register
    reg_payload = {
        "collector_id": "COL-008",
        "device_id": "dev-uuid-99999",
        "app_version": "0.1.0",
    }
    resp = client.post("/api/v1/collector/auth/register", json=reg_payload)
    token = resp.json()["auth_token"]

    # 2. Revoke device
    revoke_resp = client.post("/api/v1/collector/admin/revoke/COL-008")
    assert revoke_resp.status_code == 200
    assert revoke_resp.json()["status"] == "revoked"

    # 3. Login must fail with 403 Forbidden
    login_resp = client.post(
        "/api/v1/collector/auth/login",
        json={"collector_id": "COL-008", "device_id": "dev-uuid-99999", "auth_token": token},
    )
    assert login_resp.status_code == 403

    # 4. Attempt to re-register must fail with 403 Forbidden
    rereg_resp = client.post("/api/v1/collector/auth/register", json=reg_payload)
    assert rereg_resp.status_code == 403
