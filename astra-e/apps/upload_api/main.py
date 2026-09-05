"""FastAPI REST Application for ASTRA Collector Upload Service."""

from __future__ import annotations

import logging
from typing import Any

from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audit_logger import CollectorAuditLogger
from .hf_uploader import HuggingFaceDatasetUploader
from .schemas import (
    AuditRecord,
    ChunkUploadResult,
    CollectionTask,
    DeviceAuthResponse,
    DeviceLoginRequest,
    DeviceRegisterRequest,
    TaskSeedRequest,
    UploadCompleteRequest,
    UploadInitiateRequest,
    UploadInitiateResponse,
    UploadStatusResponse,
)
from .service import CollectorService
from .storage import ChunkStorageManager

logger = logging.getLogger("astra.collector.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Singleton service instance
_service: CollectorService | None = None


def get_collector_service() -> CollectorService:
    """Dependency accessor for CollectorService."""
    global _service
    if _service is None:
        _service = CollectorService()
    return _service


def set_collector_service(service: CollectorService) -> None:
    """Override service for testing."""
    global _service
    _service = service


app = FastAPI(
    title="ASTRA Collector Upload API",
    description="Secure distributed data collection and upload service for ASTRA-E EXP001",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
@app.get("/collector")
def get_collector_webapp() -> FileResponse:
    """Serve the ASTRA Collector mobile web application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web application static files not found.")
    return FileResponse(str(index_file))



# -----------------------------------------------------------------------------
# Authentication Endpoints
# -----------------------------------------------------------------------------


@app.post("/api/v1/collector/auth/register", response_model=DeviceAuthResponse)
def register_device(
    req: DeviceRegisterRequest,
    service: CollectorService = Depends(get_collector_service),
) -> DeviceAuthResponse:
    """Register a new synthesizer device or reconnect existing one."""
    try:
        return service.register_device(req)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@app.post("/api/v1/collector/auth/login", response_model=DeviceAuthResponse)
def login_device(
    req: DeviceLoginRequest,
    service: CollectorService = Depends(get_collector_service),
) -> DeviceAuthResponse:
    """Verify device credentials."""
    try:
        service.verify_auth_token(req.collector_id, req.auth_token)
        return DeviceAuthResponse(
            collector_id=req.collector_id,
            auth_token=req.auth_token,
            status=service.devices[req.collector_id]["status"],
            message="Authenticated successfully.",
        )
    except (KeyError, PermissionError) as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


# -----------------------------------------------------------------------------
# Task Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/v1/collector/tasks/next", response_model=CollectionTask | None)
def get_next_task(
    collector_id: str,
    service: CollectorService = Depends(get_collector_service),
) -> CollectionTask | None:
    """Dispense the next available experimental recording task for a collector."""
    task = service.get_next_task(collector_id)
    if task is None:
        return None
    return task


@app.get("/api/v1/collector/tasks/{task_id}", response_model=CollectionTask)
def get_task_details(
    task_id: str,
    service: CollectorService = Depends(get_collector_service),
) -> CollectionTask:
    """Inspect collection task details."""
    task = service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found.")
    return task


@app.post("/api/v1/collector/tasks/seed")
def seed_tasks(
    req: TaskSeedRequest,
    service: CollectorService = Depends(get_collector_service),
) -> dict[str, int]:
    """Seed tasks into the registry."""
    count = service.seed_tasks(req.tasks)
    return {"seeded": count}


# -----------------------------------------------------------------------------
# Resumable Chunked Upload Endpoints
# -----------------------------------------------------------------------------


@app.post("/api/v1/collector/uploads/initiate", response_model=UploadInitiateResponse)
def initiate_upload(
    req: UploadInitiateRequest,
    service: CollectorService = Depends(get_collector_service),
) -> UploadInitiateResponse:
    """Initialize a chunked upload session for a recorded video."""
    try:
        return service.initiate_upload(req)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.put(
    "/api/v1/collector/uploads/{upload_id}/chunks/{chunk_index}",
    response_model=ChunkUploadResult,
)
async def upload_chunk(
    upload_id: str,
    chunk_index: int,
    request: Request,
    x_chunk_sha256: str | None = Header(default=None),
    service: CollectorService = Depends(get_collector_service),
) -> ChunkUploadResult:
    """Receive a raw binary chunk for an upload session."""
    chunk_data = await request.body()
    if not chunk_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty chunk payload received.")

    try:
        bytes_written, chunk_sha256, completed, total = service.handle_chunk(
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_data=chunk_data,
            expected_chunk_sha256=x_chunk_sha256,
        )
        return ChunkUploadResult(
            upload_id=upload_id,
            chunk_index=chunk_index,
            bytes_received=bytes_written,
            chunk_sha256=chunk_sha256,
            chunks_completed=completed,
            total_chunks=total,
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@app.post("/api/v1/collector/uploads/{upload_id}/complete", response_model=UploadStatusResponse)
def complete_upload(
    upload_id: str,
    req: UploadCompleteRequest,
    service: CollectorService = Depends(get_collector_service),
) -> UploadStatusResponse:
    """
    Finalize chunk upload session. Assembles the file, verifies full SHA-256,
    streams to Hugging Face Hub, and records immutable audit log.
    """
    if req.upload_id != upload_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Path upload_id does not match body upload_id.")

    try:
        status_resp = service.complete_upload(req)
        return status_resp
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@app.get("/api/v1/collector/uploads/{upload_id}/status", response_model=UploadStatusResponse)
def get_upload_status(
    upload_id: str,
    service: CollectorService = Depends(get_collector_service),
) -> UploadStatusResponse:
    """Check the verification status of an upload session."""
    try:
        return service.get_upload_status(upload_id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# -----------------------------------------------------------------------------
# Admin & Audit Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/v1/collector/audit/logs", response_model=list[AuditRecord])
def get_audit_logs(
    collector_id: str | None = None,
    run_id: str | None = None,
    upload_id: str | None = None,
    limit: int = 100,
    service: CollectorService = Depends(get_collector_service),
) -> list[AuditRecord]:
    """Retrieve verified collection audit records."""
    return service.audit.query_records(
        collector_id=collector_id,
        run_id=run_id,
        upload_id=upload_id,
        limit=limit,
    )


@app.post("/api/v1/collector/admin/revoke/{collector_id}")
def revoke_collector_device(
    collector_id: str,
    service: CollectorService = Depends(get_collector_service),
) -> dict[str, str]:
    """Revoke access for a collector device."""
    service.revoke_device(collector_id)
    return {"collector_id": collector_id, "status": "revoked"}


@app.get("/health")
@app.get("/api/v1/collector/health")
def health_check() -> dict[str, str]:
    """Service health check."""
    return {"status": "healthy", "service": "ASTRA Collector Upload Service"}
