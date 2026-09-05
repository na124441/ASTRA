"""Business logic service coordinator for ASTRA Collector API."""

from __future__ import annotations

import datetime
import logging
import secrets
import tempfile
from pathlib import Path
from typing import Any

from .audit_logger import CollectorAuditLogger
from .hf_uploader import HuggingFaceDatasetUploader
from .schemas import (
    AuditRecord,
    CollectionTask,
    DeviceAuthResponse,
    DeviceRegisterRequest,
    DeviceStatus,
    RecordingMetadata,
    TaskStatus,
    UploadCompleteRequest,
    UploadInitiateRequest,
    UploadInitiateResponse,
    UploadState,
    UploadStatusResponse,
)
from .storage import ChunkStorageManager

logger = logging.getLogger("astra.collector.service")


class CollectorService:
    """Core service managing device registration, tasks, and chunked uploads."""

    def __init__(
        self,
        storage_manager: ChunkStorageManager | None = None,
        hf_uploader: HuggingFaceDatasetUploader | None = None,
        audit_logger: CollectorAuditLogger | None = None,
    ) -> None:
        self.storage = storage_manager or ChunkStorageManager()
        self.hf_uploader = hf_uploader or HuggingFaceDatasetUploader()
        self.audit = audit_logger or CollectorAuditLogger()

        # In-memory storage (can be backed by SQLite/PostgreSQL)
        self.devices: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, CollectionTask] = {}
        self.upload_sessions: dict[str, dict[str, Any]] = {}

        # Pre-seed initial tasks for EXP001 if none exist
        self._seed_default_exp001_tasks()

    def _seed_default_exp001_tasks(self) -> None:
        """Populate initial set of EXP001 collection tasks."""
        scenarios = [
            ("RUN-0041", "CAM-01", "nominal", "RED_COMPONENT", "TARGET_A"),
            ("RUN-0041", "CAM-02", "nominal", "RED_COMPONENT", "TARGET_A"),
            ("RUN-0042", "CAM-01", "wrong_object", "YELLOW_COMPONENT", "TARGET_A"),
            ("RUN-0042", "CAM-02", "wrong_object", "YELLOW_COMPONENT", "TARGET_A"),
            ("RUN-0043", "CAM-01", "wrong_target", "RED_COMPONENT", "TARGET_B"),
            ("RUN-0043", "CAM-02", "wrong_target", "RED_COMPONENT", "TARGET_B"),
            ("RUN-0044", "CAM-01", "order_violation", "RED_COMPONENT", "TARGET_A"),
            ("RUN-0045", "CAM-01", "nominal", "RED_COMPONENT", "TARGET_A"),
        ]
        for idx, (run, cam, scenario, obj, tgt) in enumerate(scenarios, start=1):
            tid = f"TASK-{idx:04d}"
            self.tasks[tid] = CollectionTask(
                task_id=tid,
                experiment_id="EXP001",
                run_id=run,
                camera_id=cam,
                scenario_type=scenario,
                required_object=obj,
                target=tgt,
                duration_min=30,
                duration_max=60,
                orientation="landscape",
                instruction_version="EXP001-v1.0",
                status=TaskStatus.AVAILABLE,
            )

    # -------------------------------------------------------------------------
    # Authentication & Device Management
    # -------------------------------------------------------------------------

    def register_device(self, req: DeviceRegisterRequest) -> DeviceAuthResponse:
        """Register a new device or return credentials for existing device."""
        existing = self.devices.get(req.collector_id)
        if existing:
            if existing["status"] == DeviceStatus.REVOKED:
                raise PermissionError(f"Collector ID {req.collector_id} has been revoked by system administrator.")
            # Refresh last seen
            existing["last_seen"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            existing["app_version"] = req.app_version
            return DeviceAuthResponse(
                collector_id=req.collector_id,
                auth_token=existing["auth_token"],
                status=existing["status"],
                message="Device authenticated successfully.",
            )

        token = f"astra_tok_{secrets.token_urlsafe(24)}"
        self.devices[req.collector_id] = {
            "collector_id": req.collector_id,
            "device_id": req.device_id,
            "auth_token": token,
            "status": DeviceStatus.ACTIVE,
            "app_version": req.app_version,
            "device_model": req.device_model,
            "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        return DeviceAuthResponse(
            collector_id=req.collector_id,
            auth_token=token,
            status=DeviceStatus.ACTIVE,
            message="Device registered successfully.",
        )

    def verify_auth_token(self, collector_id: str, auth_token: str) -> None:
        """Verify that collector exists, token matches, and device is active."""
        device = self.devices.get(collector_id)
        if not device:
            raise KeyError(f"Collector ID {collector_id} not registered.")
        if device["status"] == DeviceStatus.REVOKED:
            raise PermissionError(f"Collector ID {collector_id} is REVOKED.")
        if device["auth_token"] != auth_token:
            raise PermissionError("Invalid authentication token.")

    def revoke_device(self, collector_id: str) -> None:
        """Administratively revoke access for a collector device."""
        if collector_id in self.devices:
            self.devices[collector_id]["status"] = DeviceStatus.REVOKED
            logger.warning("Device %s has been revoked.", collector_id)

    # -------------------------------------------------------------------------
    # Task Dispenser
    # -------------------------------------------------------------------------

    def get_next_task(self, collector_id: str) -> CollectionTask | None:
        """Fetch next available task and assign it to collector."""
        # 1. Check if this collector already has an active incomplete task
        for task in self.tasks.values():
            if task.assigned_to == collector_id and task.status in (
                TaskStatus.ASSIGNED,
                TaskStatus.IN_PROGRESS,
            ):
                return task

        # 2. Find first available unassigned task
        for task in self.tasks.values():
            if task.status == TaskStatus.AVAILABLE:
                task.status = TaskStatus.ASSIGNED
                task.assigned_to = collector_id
                task.assigned_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                return task

        return None

    def get_task(self, task_id: str) -> CollectionTask | None:
        """Lookup task by ID."""
        return self.tasks.get(task_id)

    def seed_tasks(self, new_tasks: list[CollectionTask]) -> int:
        """Seed a batch of tasks into the registry."""
        count = 0
        for t in new_tasks:
            self.tasks[t.task_id] = t
            count += 1
        return count

    # -------------------------------------------------------------------------
    # Resumable Chunked Uploads
    # -------------------------------------------------------------------------

    def initiate_upload(self, req: UploadInitiateRequest) -> UploadInitiateResponse:
        """Initiate an upload session for a task."""
        task = self.tasks.get(req.task_id)
        if not task:
            raise KeyError(f"Task {req.task_id} not found.")

        upload_id = f"upl_{secrets.token_urlsafe(16)}"
        chunk_size = 8 * 1024 * 1024  # 8 MB standard chunk

        self.upload_sessions[upload_id] = {
            "upload_id": upload_id,
            "task_id": req.task_id,
            "collector_id": req.collector_id,
            "file_size_bytes": req.file_size_bytes,
            "total_chunks": req.total_chunks,
            "expected_sha256": req.sha256,
            "metadata": req.metadata,
            "status": UploadState.INITIATED,
            "remote_path": None,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        task.status = TaskStatus.IN_PROGRESS
        logger.info("Initiated upload session %s for task %s", upload_id, req.task_id)

        return UploadInitiateResponse(
            upload_id=upload_id,
            chunk_size_bytes=chunk_size,
            total_chunks=req.total_chunks,
            status=UploadState.INITIATED,
        )

    def handle_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        chunk_data: bytes,
        expected_chunk_sha256: str | None = None,
    ) -> tuple[int, str, int, int]:
        """Save chunk and return (bytes_written, chunk_sha256, completed_chunks, total_chunks)."""
        session = self.upload_sessions.get(upload_id)
        if not session:
            raise KeyError(f"Upload session {upload_id} not found.")

        bytes_written, chunk_sha256 = self.storage.write_chunk(upload_id, chunk_index, chunk_data)

        if expected_chunk_sha256 and expected_chunk_sha256.lower() != chunk_sha256.lower():
            raise ValueError(
                f"Chunk {chunk_index} checksum mismatch: computed {chunk_sha256}, expected {expected_chunk_sha256}"
            )

        session["status"] = UploadState.RECEIVING_CHUNKS
        uploaded_chunks = self.storage.list_uploaded_chunks(upload_id)

        return bytes_written, chunk_sha256, len(uploaded_chunks), session["total_chunks"]

    def complete_upload(self, req: UploadCompleteRequest) -> UploadStatusResponse:
        """
        Assemble chunks, verify full-file SHA-256 against expected hash,
        upload to Hugging Face Hub, write audit log, and return verified status.
        """
        session = self.upload_sessions.get(req.upload_id)
        if not session:
            raise KeyError(f"Upload session {req.upload_id} not found.")

        total_chunks = session["total_chunks"]
        uploaded = self.storage.list_uploaded_chunks(req.upload_id)
        if len(uploaded) < total_chunks:
            missing = set(range(total_chunks)) - uploaded
            session["status"] = UploadState.FAILED
            raise ValueError(f"Cannot complete upload: missing chunks {sorted(list(missing))}")

        # Assemble file
        session["status"] = UploadState.ASSEMBLING
        assembled_dir = self.storage.base_staging_dir / "assembled"
        assembled_dir.mkdir(parents=True, exist_ok=True)
        assembled_file = assembled_dir / f"{req.upload_id}.mp4"

        try:
            computed_sha256 = self.storage.assemble_file(req.upload_id, total_chunks, assembled_file)
            
            # Cryptographic checksum verification
            if computed_sha256.lower() != req.expected_sha256.lower():
                session["status"] = UploadState.FAILED
                session["error_message"] = (
                    f"Checksum mismatch: computed {computed_sha256}, expected {req.expected_sha256}"
                )
                logger.error("Upload %s failed checksum verification: %s", req.upload_id, session["error_message"])
                return UploadStatusResponse(
                    upload_id=req.upload_id,
                    task_id=session["task_id"],
                    status=UploadState.FAILED,
                    verified=False,
                    sha256_match=False,
                    error_message=session["error_message"],
                )

            # Upload to Hugging Face Hub
            session["status"] = UploadState.VERIFYING
            remote_video_path, _ = self.hf_uploader.upload_recording(assembled_file, req.metadata)

            # Success transition
            session["status"] = UploadState.VERIFIED
            session["remote_path"] = remote_video_path

            # Update task
            task = self.tasks.get(session["task_id"])
            if task:
                task.status = TaskStatus.COMPLETED

            # Record audit log
            audit_entry = AuditRecord(
                upload_id=req.upload_id,
                task_id=session["task_id"],
                collector_id=session["collector_id"],
                run_id=req.metadata.run_id,
                recording_id=req.metadata.recording_id,
                sha256=computed_sha256,
                file_size=session["file_size_bytes"],
                remote_path=remote_video_path,
                uploaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status="verified",
            )
            self.audit.record_verified_upload(audit_entry)

            # Cleanup staging chunks and assembled temp file
            self.storage.cleanup_staging(req.upload_id)
            if assembled_file.exists():
                assembled_file.unlink(missing_ok=True)

            logger.info("Upload %s fully VERIFIED. Remote path: %s", req.upload_id, remote_video_path)

            return UploadStatusResponse(
                upload_id=req.upload_id,
                task_id=session["task_id"],
                status=UploadState.VERIFIED,
                verified=True,
                remote_path=remote_video_path,
                sha256_match=True,
            )

        except Exception as e:
            session["status"] = UploadState.FAILED
            session["error_message"] = str(e)
            logger.exception("Error during completion of upload %s: %s", req.upload_id, e)
            return UploadStatusResponse(
                upload_id=req.upload_id,
                task_id=session["task_id"],
                status=UploadState.FAILED,
                verified=False,
                sha256_match=False,
                error_message=str(e),
            )

    def get_upload_status(self, upload_id: str) -> UploadStatusResponse:
        """Inspect status of an upload session."""
        session = self.upload_sessions.get(upload_id)
        if not session:
            raise KeyError(f"Upload session {upload_id} not found.")

        return UploadStatusResponse(
            upload_id=upload_id,
            task_id=session["task_id"],
            status=session["status"],
            verified=(session["status"] == UploadState.VERIFIED),
            remote_path=session.get("remote_path"),
            sha256_match=(session["status"] == UploadState.VERIFIED),
            error_message=session.get("error_message"),
        )
