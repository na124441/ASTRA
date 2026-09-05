"""Pydantic data schemas and contracts for ASTRA Collector API."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class DeviceStatus(str, Enum):
    """Device registration and authorization status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class DeviceRegisterRequest(BaseModel):
    """Payload for registering or checking in an ASTRA Collector client device."""
    collector_id: str = Field(..., description="Unique human-assigned collector identifier, e.g. COL-007")
    device_id: str = Field(..., description="Hardware or Android installation UUID")
    app_version: str = Field(..., description="ASTRA Collector app semantic version")
    device_model: str | None = Field(default=None, description="Device brand / model")


class DeviceAuthResponse(BaseModel):
    """Authentication response returning device session credentials."""
    collector_id: str
    auth_token: str
    status: DeviceStatus
    message: str


class DeviceLoginRequest(BaseModel):
    """Device login payload."""
    collector_id: str
    device_id: str
    auth_token: str


class TaskStatus(str, Enum):
    """Lifecycle state of an experiment collection task."""
    AVAILABLE = "available"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CollectionTask(BaseModel):
    """Structured collection task dispensed to an ASTRA Collector device."""
    task_id: str = Field(..., description="Unique task identifier, e.g. TASK-00421")
    experiment_id: str = Field(default="EXP001", description="Experiment identifier")
    run_id: str = Field(..., description="Target execution run identifier, e.g. RUN-0042")
    camera_id: str = Field(..., description="Assigned camera perspective, e.g. CAM-01")
    scenario_type: str = Field(default="nominal", description="nominal, wrong_object, wrong_target, order_violation")
    required_object: str = Field(default="RED_COMPONENT", description="Object required for procedure")
    target: str = Field(default="TARGET_A", description="Target receptacle or location")
    duration_min: int = Field(default=30, description="Minimum duration in seconds")
    duration_max: int = Field(default=60, description="Maximum duration in seconds")
    orientation: str = Field(default="landscape", description="Required camera orientation")
    instruction_version: str = Field(default="EXP001-v1.0", description="Version of experimental protocol")
    procedure_steps: list[str] = Field(
        default_factory=lambda: [
            "1. Start with both objects stationary on workstation",
            "2. Approach RED component with right hand",
            "3. Grasp RED component firmly",
            "4. Pick up RED component from table surface",
            "5. Move RED component smoothly toward Target A",
            "6. Place RED component squarely on Target A",
            "7. Release RED component",
            "8. Return hand to neutral resting position",
        ],
        description="Step-by-step instructions for synthesizer",
    )
    status: TaskStatus = Field(default=TaskStatus.AVAILABLE)
    assigned_to: str | None = Field(default=None, description="Collector ID assigned to this task")
    assigned_at: str | None = Field(default=None, description="ISO timestamp of task assignment")


class TaskSeedRequest(BaseModel):
    """Batch seed request for populating collection tasks."""
    tasks: list[CollectionTask]


class RecordingMetadata(BaseModel):
    """Cryptographic and experimental metadata packaged alongside every video."""
    schema_version: str = Field(default="1.0", description="Contract schema version")
    experiment_id: str = Field(default="EXP001")
    run_id: str
    recording_id: str
    collector_id: str
    camera_id: str
    scenario_type: str = Field(default="nominal")
    object: str
    target: str
    duration_seconds: float
    width: int = Field(default=1920)
    height: int = Field(default=1080)
    fps: float = Field(default=30.0)
    orientation: str = Field(default="landscape")
    file_size_bytes: int
    sha256: str
    app_version: str = Field(default="0.1.0")
    protocol_version: str = Field(default="EXP001-v1.0")
    created_at: str


class UploadState(str, Enum):
    """Resumable upload session state."""
    INITIATED = "initiated"
    RECEIVING_CHUNKS = "receiving_chunks"
    ASSEMBLING = "assembling"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"


class UploadInitiateRequest(BaseModel):
    """Payload to request an upload session."""
    task_id: str
    collector_id: str
    file_size_bytes: int
    total_chunks: int
    sha256: str
    metadata: RecordingMetadata


class UploadInitiateResponse(BaseModel):
    """Session initialization response returning upload parameters."""
    upload_id: str
    chunk_size_bytes: int = Field(default=8 * 1024 * 1024, description="Standard 8 MB chunk size")
    total_chunks: int
    status: UploadState


class ChunkUploadResult(BaseModel):
    """Acknowledgement of individual chunk upload."""
    upload_id: str
    chunk_index: int
    bytes_received: int
    chunk_sha256: str
    chunks_completed: int
    total_chunks: int


class UploadCompleteRequest(BaseModel):
    """Payload to finalize chunk assembly and trigger remote verification."""
    upload_id: str
    expected_sha256: str
    metadata: RecordingMetadata


class UploadStatusResponse(BaseModel):
    """Current verification and persistence status of an upload."""
    upload_id: str
    task_id: str
    status: UploadState
    verified: bool
    remote_path: str | None = None
    sha256_match: bool = False
    error_message: str | None = None


class AuditRecord(BaseModel):
    """Immutable audit entry for dataset collection."""
    upload_id: str
    task_id: str
    collector_id: str
    run_id: str
    recording_id: str
    sha256: str
    file_size: int
    remote_path: str
    uploaded_at: str
    status: str = "verified"
