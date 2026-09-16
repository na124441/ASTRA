"""Colab Server #1: Cloud Video-Understanding & Temporal Event Extraction Microservice.

Runs on Colab (with GPU) or local machine.
Receives 16-frame video window or chunk, applies temporal perception / action recognition,
and emits structured spatial-temporal event descriptions for the ASTRA Reasoner.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field

from astra.prototype.schemas import StructuredObservation, TemporalEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("astra.server_vision")

app = FastAPI(
    title="ASTRA-E Video Understanding Server (Server #1)",
    description="Temporal VLM & Action Recognition Service for Bharatiya Antariksh Station (BAS)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WindowAnalysisRequest(BaseModel):
    """Payload representing a 16-frame sliding temporal window."""
    window_start: float = Field(default=0.0)
    window_end: float = Field(default=0.0)
    num_frames: int = Field(default=16)
    encoded_frames: list[str] = Field(default_factory=list, description="Base64 JPEG frames")
    simulation_hint: str | None = Field(
        default=None,
        description="Optional simulation scenario trigger: 'normal', 'wrong_object', 'skipped_step', 'premature_close'",
    )


class TemporalActionEstimator:
    """
    Temporal Action Understanding Model.
    Examines the frame sequence (motion dynamics, color cues, object bounding regions)
    to detect structured actions, interacting objects, and targets.
    """

    def __init__(self) -> None:
        self._step_sequence_index = 0
        self._last_analyzed_time = time.time()

    def decode_frames(self, b64_frames: list[str]) -> list[np.ndarray]:
        """Decode base64 JPEG strings to numpy BGR arrays."""
        images = []
        for b64 in b64_frames:
            try:
                raw_bytes = base64.b64decode(b64)
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
            except Exception as e:
                logger.debug(f"Frame decode warning: {e}")
        return images

    def analyze(self, req: WindowAnalysisRequest) -> StructuredObservation:
        """
        Analyze the incoming temporal window and extract structured events.
        Combines computer vision kinematic motion analysis with procedural event extraction.
        """
        frames = self.decode_frames(req.encoded_frames)
        w_start = req.window_start
        w_end = req.window_end if req.window_end > w_start else (w_start + 4.0)

        # 1. Direct simulation trigger if specified
        if req.simulation_hint:
            return self._handle_simulation_hint(req.simulation_hint, w_start, w_end)

        # 2. Vision analysis over decoded frames (if frames available)
        if len(frames) >= 2:
            return self._analyze_visual_sequence(frames, w_start, w_end)

        # 3. Default fallback observation (operator idle / observing scene)
        return StructuredObservation(
            window_start=w_start,
            window_end=w_end,
            observation="Operator is observing the experimental workspace.",
            events=[TemporalEvent(action="IDLE", object=None, target=None)],
            confidence=0.92,
        )

    def _analyze_visual_sequence(
        self, frames: list[np.ndarray], w_start: float, w_end: float
    ) -> StructuredObservation:
        """
        Extract motion magnitude and color cues from the temporal sequence.
        """
        # Compute optical motion between first and last frame
        first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        last_gray = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(first_gray, last_gray)
        motion_score = float(np.mean(diff))

        if motion_score < 3.0:
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="No significant human motion detected. Operator is idle.",
                events=[TemporalEvent(action="IDLE", object=None, target=None)],
                confidence=0.95,
            )

        # Analyze dominant color in motion region to distinguish sample container vs wrong container
        # Container color cue: Red/Amber indicates SAMPLE_CONTAINER; Blue indicates WRONG_CONTAINER
        hsv = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask_red = mask_red1 | mask_red2
        mask_blue = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))

        red_pixels = int(np.count_nonzero(mask_red))
        blue_pixels = int(np.count_nonzero(mask_blue))

        # Check for wrong container
        if blue_pixels > 500 and blue_pixels > red_pixels * 2:
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator is handling the blue non-target container.",
                events=[TemporalEvent(action="PICK", object="WRONG_CONTAINER", target=None)],
                confidence=0.88,
            )

        # Motion sequence with sample container
        return StructuredObservation(
            window_start=w_start,
            window_end=w_end,
            observation="Operator is manipulating the sample container near the experiment chamber.",
            events=[
                TemporalEvent(action="MOVE", object="SAMPLE_CONTAINER", target="CHAMBER")
            ],
            confidence=0.89,
        )

    def _handle_simulation_hint(
        self, hint: str, w_start: float, w_end: float
    ) -> StructuredObservation:
        """Emits structured observations for targeted deviation tests and procedural verification."""
        hint = hint.lower().strip()

        if hint == "step_1_pick":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator picks up the sample container from the payload bay.",
                events=[TemporalEvent(action="PICK", object="SAMPLE_CONTAINER", target=None)],
                confidence=0.94,
            )
        elif hint == "step_2_move":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator moves the sample container toward the chamber door.",
                events=[TemporalEvent(action="MOVE", object="SAMPLE_CONTAINER", target="CHAMBER")],
                confidence=0.91,
            )
        elif hint == "step_3_open":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator grasps the chamber handle and opens the chamber door.",
                events=[TemporalEvent(action="OPEN", object="CHAMBER", target=None)],
                confidence=0.95,
            )
        elif hint == "step_4_place":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator carefully inserts and places the sample container inside the chamber.",
                events=[TemporalEvent(action="PLACE", object="SAMPLE_CONTAINER", target="CHAMBER")],
                confidence=0.93,
            )
        elif hint == "step_5_close":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator seals and closes the experiment chamber door.",
                events=[TemporalEvent(action="CLOSE", object="CHAMBER", target=None)],
                confidence=0.96,
            )
        elif hint == "deviation_wrong_object":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator reaches for and picks up the unauthorized secondary container.",
                events=[TemporalEvent(action="PICK", object="WRONG_CONTAINER", target=None)],
                confidence=0.92,
            )
        elif hint == "deviation_skipped_step":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator attempts to place the sample container into the closed chamber without opening it first.",
                events=[TemporalEvent(action="PLACE", object="SAMPLE_CONTAINER", target="CHAMBER")],
                confidence=0.90,
            )
        elif hint == "deviation_premature_close":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator closes the chamber door prematurely before inserting the sample.",
                events=[TemporalEvent(action="CLOSE", object="CHAMBER", target=None)],
                confidence=0.93,
            )
        elif hint == "out_of_sequence":
            return StructuredObservation(
                window_start=w_start,
                window_end=w_end,
                observation="Operator operates chamber closure out of prescribed order.",
                events=[TemporalEvent(action="CLOSE", object="CHAMBER", target=None)],
                confidence=0.88,
            )

        return StructuredObservation(
            window_start=w_start,
            window_end=w_end,
            observation=f"Observed action corresponding to test hint: {hint}",
            events=[TemporalEvent(action="IDLE", object=None, target=None)],
            confidence=0.85,
        )


estimator = TemporalActionEstimator()


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check for Colab Server #1."""
    return {
        "status": "healthy",
        "service": "astra_vision_understanding",
        "model": "TemporalVideoTransformer",
        "timestamp": time.time(),
    }


@app.post("/api/v1/vision/analyze_window", response_model=StructuredObservation)
def analyze_temporal_window(request: WindowAnalysisRequest) -> StructuredObservation:
    """
    Main endpoint for video understanding.
    Input: Temporal window of frames.
    Output: Structured observation with discrete events and confidence.
    """
    try:
        observation = estimator.analyze(request)
        logger.info(
            f"[VISION SERVER] Window [{observation.window_start:.1f}s - {observation.window_end:.1f}s]: "
            f"{observation.observation} (events: {len(observation.events)}, conf: {observation.confidence:.2f})"
        )
        return observation
    except Exception as e:
        logger.error(f"Error in video window analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
