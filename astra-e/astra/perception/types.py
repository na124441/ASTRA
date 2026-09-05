"""Data types and internal data structures for the perception engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDetection:
    """Unassociated single-frame bounding box detection."""
    class_name: str
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    centroid: list[float] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.centroid and len(self.bbox) == 4:
            x1, y1, x2, y2 = self.bbox
            self.centroid = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


@dataclass
class TrackedEntity:
    """Entity with persistent cross-frame tracking identity."""
    track_id: str
    class_name: str
    bbox: list[float]
    centroid: list[float]
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])  # [vx, vy] in px/sec
    confidence: float = 1.0
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
