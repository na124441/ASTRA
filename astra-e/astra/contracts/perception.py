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
