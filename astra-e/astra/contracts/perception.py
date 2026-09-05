"""Perception layer contracts and structured observation schemas."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from astra.contracts.base import BaseMessage, current_timestamp


class DetectedHuman(BaseModel):
    """Detected person in the payload workspace."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique tracking identifier, e.g. human-01")
    bbox: list[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")


class DetectedObject(BaseModel):
    """Detected experimental object or container."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable object tracking ID, e.g. obj-red-01")
    type: str = Field(description="Semantic class, e.g. RED_COMPONENT, CONTAINER")
    bbox: list[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    tracking_state: str | None = Field(default=None, description="Tracking status: TRACKED, LOST, NEW")

    @property
    def label(self) -> str:
        """Alias for type semantic label."""
        return self.type


class HandLandmark(BaseModel):
    """Hand position estimation."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Hand identifier, e.g. hand-right, hand-left")
    owner_id: str = Field(description="Associated human ID")
    position: list[float] = Field(description="Coordinates [x, y] or [x, y, z]")
    confidence: float = Field(ge=0.0, le=1.0, description="Landmark confidence")


class SceneObservation(BaseMessage):
    """
    Contract 02: SceneObservation.
    High-level structured state of entities in the frame.
    """
    camera_id: str = Field(description="Camera that generated the observation")
    event_time: float = Field(default_factory=current_timestamp, description="Timestamp of frame capture")
    humans: list[DetectedHuman] = Field(default_factory=list)
    objects: list[DetectedObject] = Field(default_factory=list)
    hands: list[HandLandmark] = Field(default_factory=list)
    poses: list[dict[str, Any]] = Field(default_factory=list)
    scene_metadata: dict[str, Any] = Field(default_factory=dict)


class EntityDetection(BaseModel):
    """
    Detector-agnostic entity localization and tracking confidence.
    Standardized payload format consumed by KinematicFeatureExtractor regardless
    of whether detections originate from YOLO, MediaPipe, synthetic simulation, or ArUco.
    """
    model_config = ConfigDict(frozen=True)

    pos: list[float] = Field(description="[x, y] coordinates in pixel space")
    conf: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")


def scene_observation_to_detections(obs: SceneObservation) -> dict[str, Any]:
    """
    Converts a rich edge SceneObservation into the agnostic detector contract dictionary.
    Guarantees that the production feature extractor only ever consumes the canonical detector dict.
    """
    detections: dict[str, Any] = {
        "event_time": obs.event_time,
    }
    if obs.hands:
        h = obs.hands[0]
        detections["hand"] = {
            "pos": [float(h.position[0]), float(h.position[1])],
            "conf": float(h.confidence),
        }
    for obj in obs.objects:
        c = [(obj.bbox[0] + obj.bbox[2]) / 2.0, (obj.bbox[1] + obj.bbox[3]) / 2.0]
        t = obj.type.upper()
        if "RED" in t:
            detections["red"] = {"pos": [float(c[0]), float(c[1])], "conf": float(obj.confidence)}
        elif "YELLOW" in t:
            detections["yellow"] = {"pos": [float(c[0]), float(c[1])], "conf": float(obj.confidence)}
        elif "TARGET_A" in t:
            detections["target_a"] = {"pos": [float(c[0]), float(c[1])], "conf": float(obj.confidence)}
        elif "TARGET_B" in t:
            detections["target_b"] = {"pos": [float(c[0]), float(c[1])], "conf": float(obj.confidence)}
        elif "CONTAINER" in t:
            detections["container"] = {"pos": [float(c[0]), float(c[1])], "conf": float(obj.confidence)}
    return detections

