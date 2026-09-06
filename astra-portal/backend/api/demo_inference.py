"""
============================================================================
OWNER: Backend Developer 2 (Kratika)
PURPOSE: Live Action Classification Sandbox Endpoint (POST /api/v1/demo/inference).
         Handles multipart video/image uploads or JSON clip IDs.
         Simulates lightweight action classifier with realistic 110-140ms latency.
============================================================================
"""

import time
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, File, Form, UploadFile, Request

router = APIRouter(prefix="/api/v1/demo/inference", tags=["Inference Sandbox"])


class DemoInferenceResponse(BaseModel):
    step_id: int
    action_name: str
    status: str
    confidence: float
    inference_ms: int
    anomaly_detected: bool
    timestamp: str


@router.post("", response_model=DemoInferenceResponse)
async def run_demo_inference(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    clip_id: Optional[str] = Form(default=None),
):
    """
    Run action classification on a demo clip or uploaded video/image payload.
    Supports both application/json payload and multipart/form-data video upload.
    """
    start_time = time.time()
    target_clip_id = clip_id

    # Check if request is application/json
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                target_clip_id = body.get("clip_id") or target_clip_id
        except Exception:
            pass

    target_id = target_clip_id or (file.filename if file else "sample-01")
    is_fault = "fault" in target_id.lower() or target_id == "sample-02"

    # Simulate realistic microgravity inference compute delay (110-140ms)
    time.sleep(0.118)

    elapsed_ms = int((time.time() - start_time) * 1000)
    # Ensure reported latency stays cleanly within required 110-140ms bracket
    inference_ms = max(110, min(140, elapsed_ms))

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if is_fault:
        return DemoInferenceResponse(
            step_id=4,
            action_name="FAULT DETECTED: Gasket Seal Misaligned on Chamber B",
            status="FAULT",
            confidence=0.942,
            inference_ms=inference_ms,
            anomaly_detected=True,
            timestamp=now_iso,
        )

    return DemoInferenceResponse(
        step_id=3,
        action_name="Place Red Vial in Slot A",
        status="NOMINAL",
        confidence=0.942,
        inference_ms=inference_ms,
        anomaly_detected=False,
        timestamp=now_iso,
    )
