"""Spatial relationship utilities and geometric heuristics for Human-Object Interaction (HOI)."""

from __future__ import annotations

import math
from typing import Sequence


def bbox_centroid(bbox: Sequence[float]) -> list[float]:
    """Compute (cx, cy) centroid from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox[:4]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Calculate 2D or 3D Euclidean distance."""
    return math.dist(p1[:2], p2[:2])


def point_in_bbox(point: Sequence[float], bbox: Sequence[float], margin: float = 0.0) -> bool:
    """Check if point (x, y) is inside bounding box [x1, y1, x2, y2] with optional margin."""
    x, y = point[:2]
    x1, y1, x2, y2 = bbox[:4]
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def compute_iou(bbox1: Sequence[float], bbox2: Sequence[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1_a, y1_a, x2_a, y2_a = bbox1[:4]
    x1_b, y1_b, x2_b, y2_b = bbox2[:4]

    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compute_co_movement_score(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Calculate directional and speed similarity between two velocity vectors [vx, vy].
    Returns score in [0.0, 1.0].
    """
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 < 2.0 or mag2 < 2.0:
        # Stationary or minimal movement
        return 0.5 if (mag1 < 2.0 and mag2 < 2.0) else 0.1

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_sim = dot / (mag1 * mag2)
    # Clamp to [0, 1]
    normalized_cos = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))
    speed_ratio = min(mag1, mag2) / max(mag1, mag2)

    return float(normalized_cos * 0.7 + speed_ratio * 0.3)
