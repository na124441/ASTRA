"""Video ingestion and frame reference contracts."""

from __future__ import annotations

from pydantic import Field
from astra.contracts.base import BaseMessage, current_timestamp


class VideoFrame(BaseMessage):
    """
    Contract 01: VideoFrame reference contract.
    Carries frame metadata and buffer reference without raw pixel serialization.
    """
    frame_id: int = Field(description="Monotonically increasing sequence frame index")
    camera_id: str = Field(description="Unique camera identifier, e.g. CAM-01")
    width: int = Field(description="Frame width in pixels")
    height: int = Field(description="Frame height in pixels")
    format: str = Field(default="BGR", description="Color format: BGR, RGB, GRAY")
    frame_reference: str = Field(description="Shared memory URI or disk pointer to buffer")
    event_time: float = Field(default_factory=current_timestamp, description="Timestamp of physical frame capture")
