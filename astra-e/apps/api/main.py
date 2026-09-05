"""FastAPI & WebSocket Server for ASTRA-E Edge Platform (BAS)."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from astra.contracts.assistance import AssistancePriority
from astra.runtime.orchestrator import EdgeRuntimeOrchestrator

logger = logging.getLogger("astra.api.main")

# Global orchestrator reference
_orchestrator: EdgeRuntimeOrchestrator | None = None


def get_orchestrator() -> EdgeRuntimeOrchestrator:
    """Dependency accessor for EdgeRuntimeOrchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = EdgeRuntimeOrchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: EdgeRuntimeOrchestrator) -> None:
    """Override orchestrator instance (e.g. for testing or custom camera feeds)."""
    global _orchestrator
    _orchestrator = orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and graceful shutdown lifecycle for edge daemon."""
    orch = get_orchestrator()
    orch.start_loop(target_fps=30.0)
    logger.info("ASTRA-E FastAPI Edge Service initialized and loop started.")
    try:
        yield
    finally:
        orch.shutdown()
        logger.info("ASTRA-E FastAPI Edge Service terminated.")


app = FastAPI(
    title="ASTRA-E Edge API",
    description="Autonomous Space Task Recognition & Assistance for Experiments (BAS)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


# Request & Response Models
class StartExperimentRequest(BaseModel):
    experiment_id: str = Field(default="EXP001", description="Experiment procedure directory name")
    run_id: str | None = Field(default=None, description="Optional unique execution run ID")


class SpeakRequest(BaseModel):
    text: str = Field(description="Utterance text to synthesize")
    priority: str = Field(default="LOW", description="Priority level: LOW, MEDIUM, HIGH, CRITICAL")


# Endpoints
@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health status and telemetry snapshot."""
    orch = get_orchestrator()
    return {
        "status": "OK",
        "system": "ASTRA-E",
        "mode": "EDGE_OFFLINE",
        "orchestrator_status": orch.status,
        "fps": round(orch._last_fps, 1),
    }


@app.get("/")
@app.get("/dashboard")
def get_dashboard() -> FileResponse:
    """Serve embedded astronaut HUD dashboard."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found.")
    return FileResponse(html_path)


@app.post("/api/v1/experiment/start")
def start_experiment(req: StartExperimentRequest) -> dict[str, Any]:
    """Initialize and start an experiment procedure execution."""
    orch = get_orchestrator()
    run_id = orch.start_experiment(experiment_id=req.experiment_id, run_id=req.run_id)
    return {
        "status": "STARTED",
        "run_id": run_id,
        "experiment_id": req.experiment_id,
    }


@app.post("/api/v1/experiment/pause")
def pause_experiment() -> dict[str, str]:
    """Pause experiment processing."""
    orch = get_orchestrator()
    orch.pause_experiment()
    return {"status": "PAUSED"}


@app.post("/api/v1/experiment/resume")
def resume_experiment() -> dict[str, str]:
    """Resume experiment processing."""
    orch = get_orchestrator()
    orch.resume_experiment()
    return {"status": "RUNNING"}


@app.post("/api/v1/experiment/reset")
def reset_experiment() -> dict[str, str]:
    """Reset experiment state to IDLE."""
    orch = get_orchestrator()
    orch.reset_experiment()
    return {"status": "IDLE"}


@app.get("/api/v1/experiment/status")
def get_experiment_status() -> dict[str, Any]:
    """Current procedure state, next steps, confidence, and telemetry."""
    orch = get_orchestrator()
    return orch.get_telemetry()


@app.get("/api/v1/experiment/runs")
def list_experiment_runs() -> list[dict[str, Any]]:
    """List historical runs from local SQLite ledger."""
    orch = get_orchestrator()
    runs = orch.ledger.list_runs()
    return [
        {
            "run_id": r.run_id,
            "experiment_id": r.experiment_id,
            "procedure_id": r.procedure_id,
            "status": r.status,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "metadata": r.metadata,
        }
        for r in runs
    ]


@app.get("/api/v1/experiment/runs/{run_id}/report")
def get_run_report(run_id: str) -> dict[str, Any]:
    """Export complete forensic audit report for ground downlink."""
    orch = get_orchestrator()
    try:
        report = orch.ledger.export_audit_report(run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "run_id": report.run_id,
        "experiment_id": report.experiment_id,
        "procedure_id": report.procedure_id,
        "status": report.status,
        "start_time": report.start_time,
        "end_time": report.end_time,
        "duration_seconds": report.duration_seconds,
        "total_events": report.total_events,
        "total_confirmed_actions": report.total_confirmed_actions,
        "total_violations": report.total_violations,
        "total_assistance_alerts": report.total_assistance_alerts,
        "violations": report.violations,
        "assistance": report.assistance,
    }


@app.post("/api/v1/assistance/speak")
def speak_utterance(req: SpeakRequest) -> dict[str, Any]:
    """Manually synthesize auditory voice prompt via offline TTS engine."""
    orch = get_orchestrator()
    try:
        p_enum = AssistancePriority(req.priority.upper())
    except ValueError:
        p_enum = AssistancePriority.LOW

    queued = orch.tts.speak(req.text, priority=p_enum)
    return {"spoken": queued, "text": req.text, "priority": p_enum.value}


@app.get("/api/v1/video/stream")
def video_stream() -> StreamingResponse:
    """MJPEG live camera video stream with HUD telemetry overlays."""
    orch = get_orchestrator()

    def generate_frames():
        while True:
            jpeg_bytes = orch.get_annotated_jpeg()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
            time.sleep(0.033)  # ~30 FPS

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    """High-throughput real-time telemetry streaming to connected dashboards."""
    await websocket.accept()
    orch = get_orchestrator()
    try:
        while True:
            telemetry = orch.get_telemetry()
            await websocket.send_json(telemetry)
            await asyncio.sleep(0.05)  # 20 Hz push rate
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket telemetry error: {e}")
