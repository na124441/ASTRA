"""PerceptionPipeline generating standardized SceneObservation contracts from video frames."""

from __future__ import annotations

import time
from typing import Any
import numpy as np
from astra.contracts.base import default_uuid
from astra.contracts.perception import (
    DetectedHuman,
    DetectedObject,
    HandLandmark,
    SceneObservation,
)
from astra.contracts.video import VideoFrame
from astra.events.bus import EventBus
from astra.perception.detector import BaseDetector, ColorExperimentDetector
from astra.perception.tracker import MultiObjectTracker


class PerceptionPipeline:
    """
    Perception Subsystem Pipeline.
    Converts raw image frames into structured, contract-compliant SceneObservation objects.
    Decouples raw computer vision models from downstream temporal and procedure engines.
    """

    def __init__(
        self,
        detector: BaseDetector | None = None,
        tracker: MultiObjectTracker | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.detector = detector or ColorExperimentDetector()
        self.tracker = tracker or MultiObjectTracker()
        self.event_bus = event_bus

    def process_frame(
        self,
        video_frame: VideoFrame,
        frame_data: np.ndarray,
    ) -> SceneObservation:
        """
        Execute detection and tracking on the input frame.
        Produces an immutable SceneObservation contract.
        """
        # 1. Run detection
        raw_detections = self.detector.detect(frame_data)

        # 2. Update multi-object tracker
        active_tracks = self.tracker.update(
            raw_detections,
            timestamp=video_frame.event_time,
        )

        humans: list[DetectedHuman] = []
        objects: list[DetectedObject] = []
        hands: list[HandLandmark] = []

        for track in active_tracks:
            if track.class_name == "HUMAN":
                humans.append(
                    DetectedHuman(
                        id=track.track_id,
                        bbox=track.bbox,
                        confidence=track.confidence,
                    )
                )
            elif track.class_name == "HAND":
                hands.append(
                    HandLandmark(
                        id=track.track_id,
                        owner_id="human-01",
                        position=track.centroid,
                        confidence=track.confidence,
                    )
                )
            else:
                # Experiment object (RED_COMPONENT, CONTAINER, TARGET_A, etc.)
                objects.append(
                    DetectedObject(
                        id=track.track_id,
                        type=track.class_name,
                        bbox=track.bbox,
                        confidence=track.confidence,
                        tracking_state="TRACKED",
                    )
                )

        scene_obs = SceneObservation(
            message_id=f"obs-{default_uuid()[:8]}",
            schema_version="1.0",
            timestamp=time.time(),
            source="perception-engine",
            correlation_id=video_frame.correlation_id,
            camera_id=video_frame.camera_id,
            event_time=video_frame.event_time,
            humans=humans,
            objects=objects,
            hands=hands,
            poses=[],
            scene_metadata={
                "frame_id": video_frame.frame_id,
                "entity_count": len(active_tracks),
                "frame_reference": video_frame.frame_reference,
            },
        )

        if self.event_bus is not None:
            self.event_bus.publish("scene.observation", scene_obs)

        return scene_obs
