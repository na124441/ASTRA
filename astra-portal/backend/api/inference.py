"""
============================================================================
OWNER: Backend Developer 2
PURPOSE: Live Action Classification Sandbox Endpoint (POST /api/v1/demo/inference).

HOW TO EDIT:
1. When real ONNX model weights are present, run `onnxruntime.InferenceSession`.
2. Otherwise, runs rapid mock pipeline outputting realistic steps and latencies.
============================================================================
"""

import time
from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/demo/inference", tags=["Inference Demo"])


class DemoInferenceRequest(BaseModel):
    clip_id: str = "sample-01"


class DemoInferenceResponse(BaseModel):
    step_id: int
    action_name: str
    status: str
    confidence: float
    inference_ms: int
    anomaly_detected: bool
    timestamp: str


@router.post("", response_model=DemoInferenceResponse)
def run_demo_inference(req: DemoInferenceRequest):
    """Run action classification on a demo clip."""
    start_time = time.time()

    # Simulate realistic microgravity action recognition inference
    is_fault = req.clip_id == "sample-02"

    time.sleep(0.08)  # simulate 80ms inference compute

    latency = int((time.time() - start_time) * 1000)

    if is_fault:
        return DemoInferenceResponse(
            step_id=4,
            action_name="FAULT DETECTED: Gasket Seal Misaligned on Chamber B",
            status="FAULT",
            confidence=0.941,
            inference_ms=latency,
            anomaly_detected=True,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    return DemoInferenceResponse(
        step_id=3,
        action_name="Transfer Reagent to Well A1 via Calibrated Pipette",
        status="NOMINAL",
        confidence=0.962,
        inference_ms=latency,
        anomaly_detected=False,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
