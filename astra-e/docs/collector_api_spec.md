# ASTRA Collector Upload API Specification

## 1. Architectural Overview
The ASTRA Collector Upload API is an authenticated gateway for receiving experimental procedure videos (`EXP001`) from distributed mobile collection terminals and persisting them directly into a private Hugging Face Dataset repository (`na124441/astra-e-raw`).

### Security Posture
- **Zero Third-Party Tokens on Device**: Mobile clients hold **no** Hugging Face credentials. Hugging Face fine-grained write tokens (`HF_TOKEN`) exist **exclusively on the backend server**.
- **Device Authentication & Revocation**: Each phone registers with a `collector_id` and receives a unique device bearer token. Administrators can instantaneously revoke compromised or decommissioned devices.
- **Fail-Closed Storage Invariant**: `NO VERIFIED REMOTE UPLOAD -> NO LOCAL DELETE`.

---

## 2. API Endpoints

### 2.1 Device Authentication

#### `POST /api/v1/collector/auth/register`
Registers a synthesizer device or reconnects an existing client.

**Request Body**:
```json
{
  "collector_id": "COL-007",
  "device_id": "8f3b2d1c-9a4e-4b8f-bc52-123456789abc",
  "app_version": "0.1.0",
  "device_model": "Google Pixel 8 Pro"
}
```

**Response (200 OK)**:
```json
{
  "collector_id": "COL-007",
  "auth_token": "astra_tok_xyz123...",
  "status": "active",
  "message": "Device authenticated successfully."
}
```

---

### 2.2 Task Dispenser

#### `GET /api/v1/collector/tasks/next?collector_id=COL-007`
Dispenses the next available collection task for the given collector.

**Response (200 OK)**:
```json
{
  "task_id": "TASK-0042",
  "experiment_id": "EXP001",
  "run_id": "RUN-0042",
  "camera_id": "CAM-01",
  "scenario_type": "nominal",
  "required_object": "RED_COMPONENT",
  "target": "TARGET_A",
  "duration_min": 30,
  "duration_max": 60,
  "orientation": "landscape",
  "instruction_version": "EXP001-v1.0",
  "procedure_steps": [
    "1. Start with both objects stationary on workstation",
    "2. Approach RED component with right hand",
    "3. Grasp RED component firmly",
    "4. Pick up RED component from table surface",
    "5. Move RED component smoothly toward Target A",
    "6. Place RED component squarely on Target A",
    "7. Release RED component",
    "8. Return hand to neutral resting position"
  ],
  "status": "assigned",
  "assigned_to": "COL-007",
  "assigned_at": "2026-09-05T21:40:00Z"
}
```

---

### 2.3 Resumable Chunked Upload Protocol

#### `POST /api/v1/collector/uploads/initiate`
Initializes a multi-chunk upload session.

**Request Body**:
```json
{
  "task_id": "TASK-0042",
  "collector_id": "COL-007",
  "file_size_bytes": 193482931,
  "total_chunks": 24,
  "sha256": "3a7b9c1d...",
  "metadata": {
    "schema_version": "1.0",
    "experiment_id": "EXP001",
    "run_id": "RUN-0042",
    "recording_id": "EXP001_RUN-0042_CAM-01",
    "collector_id": "COL-007",
    "camera_id": "CAM-01",
    "scenario_type": "nominal",
    "object": "RED_COMPONENT",
    "target": "TARGET_A",
    "duration_seconds": 43.2,
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "orientation": "landscape",
    "file_size_bytes": 193482931,
    "sha256": "3a7b9c1d...",
    "app_version": "0.1.0",
    "protocol_version": "EXP001-v1.0",
    "created_at": "2026-09-05T21:42:00Z"
  }
}
```

**Response (200 OK)**:
```json
{
  "upload_id": "upl_abc987...",
  "chunk_size_bytes": 8388608,
  "total_chunks": 24,
  "status": "initiated"
}
```

#### `PUT /api/v1/collector/uploads/{upload_id}/chunks/{chunk_index}`
Receives a raw binary chunk.
- **Header**: `X-Chunk-SHA256: <hex-digest>` (Optional, validated if provided)
- **Body**: Raw binary octet stream (up to 8 MB)

**Response (200 OK)**:
```json
{
  "upload_id": "upl_abc987...",
  "chunk_index": 0,
  "bytes_received": 8388608,
  "chunk_sha256": "4e5f6a7b...",
  "chunks_completed": 1,
  "total_chunks": 24
}
```

#### `POST /api/v1/collector/uploads/{upload_id}/complete`
Finalizes upload, reassembles file, validates full-file SHA-256 against advertised hash, uploads to Hugging Face Hub, and logs immutable audit record.

**Request Body**:
```json
{
  "upload_id": "upl_abc987...",
  "expected_sha256": "3a7b9c1d...",
  "metadata": { ... }
}
```

**Response (200 OK - Verified)**:
```json
{
  "upload_id": "upl_abc987...",
  "task_id": "TASK-0042",
  "status": "verified",
  "verified": true,
  "remote_path": "videos/exp001/RUN-0042/CAM-01.mp4",
  "sha256_match": true,
  "error_message": null
}
```

**Response (200 OK - Failed / Mismatch)**:
```json
{
  "upload_id": "upl_abc987...",
  "task_id": "TASK-0042",
  "status": "failed",
  "verified": false,
  "remote_path": null,
  "sha256_match": false,
  "error_message": "Checksum mismatch: computed xyz, expected abc"
}
```

---

### 2.4 Audit & Administration

#### `GET /api/v1/collector/audit/logs`
Returns verified collection audit records.

#### `POST /api/v1/collector/admin/revoke/{collector_id}`
Revokes access for a specified collector device.
