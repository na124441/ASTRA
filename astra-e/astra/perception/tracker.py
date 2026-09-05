"""MultiObjectTracker providing stable identities across time for temporal reasoning."""

from __future__ import annotations

import math
from typing import Any
from astra.perception.types import RawDetection, TrackedEntity


class MultiObjectTracker:
    """
    Centroid and proximity-based Multi-Object Tracker (MOT).
    Ensures stable entity tracking IDs (e.g. 'obj-red-1') across video frames,
    satisfying page 179 of the ASTRA-E specification.
    Computes velocity vectors to detect hand-object co-movement.
    """

    def __init__(
        self,
        max_distance: float = 80.0,
        max_missing_frames: int = 15,
    ) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.tracks: dict[str, TrackedEntity] = {}
        self._class_counters: dict[str, int] = {}
        self._last_timestamp: float | None = None

    def update(
        self,
        detections: list[RawDetection],
        timestamp: float | None = None,
    ) -> list[TrackedEntity]:
        """
        Associate new detections with existing tracks and update kinematics.
        """
        dt = 1.0 / 30.0  # default 30 fps
        if timestamp is not None and self._last_timestamp is not None:
            delta = timestamp - self._last_timestamp
            if delta > 0:
                dt = delta
        if timestamp is not None:
            self._last_timestamp = timestamp

        # Group detections by class
        detections_by_class: dict[str, list[RawDetection]] = {}
        for det in detections:
            detections_by_class.setdefault(det.class_name, []).append(det)

        updated_track_ids: set[str] = set()

        # Update per class to avoid cross-class ID swapping
        all_classes = set(list(self.tracks.keys()) + list(detections_by_class.keys()))
        for cls_name in set(t.class_name for t in self.tracks.values()) | set(detections_by_class.keys()):
            class_tracks = [t for t in self.tracks.values() if t.class_name == cls_name]
            class_dets = detections_by_class.get(cls_name, [])

            # Compute pairwise distance matrix
            unmatched_dets = list(range(len(class_dets)))
            unmatched_tracks = list(range(len(class_tracks)))

            if class_tracks and class_dets:
                # Greedy nearest neighbor matching
                pairs: list[tuple[float, int, int]] = []
                for t_idx, track in enumerate(class_tracks):
                    for d_idx, det in enumerate(class_dets):
                        dist = math.dist(track.centroid, det.centroid)
                        if dist <= self.max_distance:
                            pairs.append((dist, t_idx, d_idx))

                pairs.sort(key=lambda x: x[0])
                matched_t = set()
                matched_d = set()

                for dist, t_idx, d_idx in pairs:
                    if t_idx in matched_t or d_idx in matched_d:
                        continue
                    matched_t.add(t_idx)
                    matched_d.add(d_idx)

                    track = class_tracks[t_idx]
                    det = class_dets[d_idx]

                    # Update velocity
                    vx = (det.centroid[0] - track.centroid[0]) / dt
                    vy = (det.centroid[1] - track.centroid[1]) / dt

                    track.bbox = det.bbox
                    track.centroid = det.centroid
                    track.velocity = [vx, vy]
                    track.confidence = det.confidence
                    track.hits += 1
                    track.age += 1
                    track.time_since_update = 0
                    updated_track_ids.add(track.track_id)

                unmatched_dets = [i for i in range(len(class_dets)) if i not in matched_d]
                unmatched_tracks = [i for i in range(len(class_tracks)) if i not in matched_t]

            # Mark unmatched tracks as aged
            for t_idx in unmatched_tracks:
                track = class_tracks[t_idx]
                track.time_since_update += 1
                track.age += 1

            # Create new tracks for unmatched detections
            for d_idx in unmatched_dets:
                det = class_dets[d_idx]
                prefix = self._get_prefix(det.class_name)
                idx = self._class_counters.get(det.class_name, 1)
                self._class_counters[det.class_name] = idx + 1
                new_id = f"{prefix}-{idx:02d}"

                new_track = TrackedEntity(
                    track_id=new_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    centroid=det.centroid,
                    velocity=[0.0, 0.0],
                    confidence=det.confidence,
                    age=1,
                    hits=1,
                    time_since_update=0,
                )
                self.tracks[new_id] = new_track
                updated_track_ids.add(new_id)

        # Remove dead tracks
        dead_ids = [
            tid for tid, track in self.tracks.items()
            if track.time_since_update > self.max_missing_frames
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        return [t for t in self.tracks.values() if t.time_since_update == 0]

    def _get_prefix(self, class_name: str) -> str:
        mapping = {
            "RED_COMPONENT": "obj-red",
            "YELLOW_COMPONENT": "obj-yellow",
            "CONTAINER": "container",
            "TARGET_A": "target-a",
            "TARGET_B": "target-b",
            "HAND": "hand",
            "HUMAN": "human",
        }
        return mapping.get(class_name, "entity")

    def reset(self) -> None:
        """Clear all active tracks."""
        self.tracks.clear()
        self._class_counters.clear()
        self._last_timestamp = None
