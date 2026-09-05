"""InteractionAnalyzer implementing spatial relationship and HOI state tracking."""

from __future__ import annotations

import time
from typing import Any
from astra.contracts.base import default_uuid
from astra.contracts.interaction import InteractionEvent, InteractionEvidence
from astra.contracts.perception import DetectedObject, HandLandmark, SceneObservation
from astra.interaction.spatial import (
    bbox_centroid,
    compute_co_movement_score,
    euclidean_distance,
    point_in_bbox,
)


class InteractionAnalyzer:
    """
    Analyzes spatial-temporal hand-object relationships.
    Detects primitive interaction transitions:
    - APPROACH: Hand moves close to experiment object
    - TOUCH: Hand reaches object perimeter
    - GRASP / PICK: Hand seizes object and object begins following hand
    - MOVE: Hand transports object across the rack
    - PLACE: Object brought inside target boundary (TARGET_A / TARGET_B)
    - RELEASE: Hand detaches from object inside target zone
    """

    def __init__(
        self,
        approach_threshold: float = 90.0,
        touch_threshold: float = 40.0,
    ) -> None:
        self.approach_threshold = approach_threshold
        self.touch_threshold = touch_threshold

        # Active grasp state: {hand_id: object_id}
        self._grasped_objects: dict[str, str] = {}
        # Object previous centroids: {obj_id: [cx, cy]}
        self._prev_centroids: dict[str, list[float]] = {}
        # Object placed status: {obj_id: target_id}
        self._placed_objects: dict[str, str] = {}

    def reset(self) -> None:
        """Reset internal interaction tracking state."""
        self._grasped_objects.clear()
        self._prev_centroids.clear()
        self._placed_objects.clear()

    def analyze(self, observation: SceneObservation) -> list[InteractionEvent]:
        """
        Analyze scene entities and return list of active interaction events.
        """
        events: list[InteractionEvent] = []
        hands = observation.hands
        objects = observation.objects

        # Separate targets from manipulable objects
        target_objects = [o for o in objects if "TARGET" in o.type.upper()]
        manipulable_objects = [
            o for o in objects
            if "TARGET" not in o.type.upper() and "CONTAINER" not in o.type.upper()
        ]
        container_objects = [o for o in objects if "CONTAINER" in o.type.upper()]

        for hand in hands:
            hand_pos = hand.position[:2]
            actor_id = hand.owner_id or "human-01"

            # 1. Evaluate interactions with manipulable components (RED_COMPONENT, YELLOW_COMPONENT)
            for obj in manipulable_objects:
                obj_centroid = bbox_centroid(obj.bbox)
                dist = euclidean_distance(hand_pos, obj_centroid)
                prev_pos = self._prev_centroids.get(obj.id, obj_centroid)
                obj_displaced = euclidean_distance(obj_centroid, prev_pos) > 2.0

                is_grasped = self._grasped_objects.get(hand.id) == obj.id

                # Check if inside any target zone
                current_target_id: str | None = None
                for tgt in target_objects:
                    if point_in_bbox(obj_centroid, tgt.bbox, margin=10.0):
                        current_target_id = tgt.id
                        break

                evidence = InteractionEvidence(
                    hand_distance=round(dist, 2),
                    relative_motion=1.0 if (is_grasped and obj_displaced) else 0.0,
                    details={
                        "object_type": obj.type,
                        "target_zone": current_target_id,
                        "hand_id": hand.id,
                    },
                )

                # State Transition Logic:
                if dist <= self.touch_threshold:
                    if not is_grasped:
                        # Transition from proximity to grasp
                        self._grasped_objects[hand.id] = obj.id
                        events.append(
                            self._create_event(
                                observation=observation,
                                interaction_type="GRASP",
                                actor_id=actor_id,
                                hand_id=hand.id,
                                object_id=obj.id,
                                target_id=None,
                                confidence=0.95,
                                evidence=evidence,
                            )
                        )
                        events.append(
                            self._create_event(
                                observation=observation,
                                interaction_type="PICK",
                                actor_id=actor_id,
                                hand_id=hand.id,
                                object_id=obj.id,
                                target_id=None,
                                confidence=0.94,
                                evidence=evidence,
                            )
                        )
                    else:
                        # Already grasped and moving
                        if current_target_id:
                            # Object is currently inside a target zone
                            events.append(
                                self._create_event(
                                    observation=observation,
                                    interaction_type="PLACE",
                                    actor_id=actor_id,
                                    hand_id=hand.id,
                                    object_id=obj.id,
                                    target_id=current_target_id,
                                    confidence=0.96,
                                    evidence=evidence,
                                )
                            )
                        elif obj_displaced:
                            events.append(
                                self._create_event(
                                    observation=observation,
                                    interaction_type="MOVE",
                                    actor_id=actor_id,
                                    hand_id=hand.id,
                                    object_id=obj.id,
                                    target_id=None,
                                    confidence=0.92,
                                    evidence=evidence,
                                )
                            )

                elif is_grasped and dist > self.touch_threshold:
                    # Hand released the object
                    del self._grasped_objects[hand.id]
                    events.append(
                        self._create_event(
                            observation=observation,
                            interaction_type="RELEASE",
                            actor_id=actor_id,
                            hand_id=hand.id,
                            object_id=obj.id,
                            target_id=current_target_id,
                            confidence=0.95,
                            evidence=evidence,
                        )
                    )
                    if current_target_id:
                        self._placed_objects[obj.id] = current_target_id

                elif dist <= self.approach_threshold:
                    # Hand is approaching
                    events.append(
                        self._create_event(
                            observation=observation,
                            interaction_type="APPROACH",
                            actor_id=actor_id,
                            hand_id=hand.id,
                            object_id=obj.id,
                            target_id=None,
                            confidence=0.90,
                            evidence=evidence,
                        )
                    )

                self._prev_centroids[obj.id] = obj_centroid

            # 2. Evaluate interactions with CONTAINER (Open/Close touch)
            for cnt in container_objects:
                cnt_center = bbox_centroid(cnt.bbox)
                dist = euclidean_distance(hand_pos, cnt_center)
                if dist <= (self.touch_threshold + 40.0):
                    ev = InteractionEvidence(
                        hand_distance=round(dist, 2),
                        details={"container_id": cnt.id},
                    )
                    events.append(
                        self._create_event(
                            observation=observation,
                            interaction_type="TOUCH",
                            actor_id=actor_id,
                            hand_id=hand.id,
                            object_id=cnt.id,
                            target_id=None,
                            confidence=0.91,
                            evidence=ev,
                        )
                    )

        return events

    def _create_event(
        self,
        observation: SceneObservation,
        interaction_type: str,
        actor_id: str,
        hand_id: str,
        object_id: str | None,
        target_id: str | None,
        confidence: float,
        evidence: InteractionEvidence,
    ) -> InteractionEvent:
        return InteractionEvent(
            message_id=f"int-{default_uuid()[:8]}",
            schema_version="1.0",
            timestamp=time.time(),
            source="interaction-engine",
            correlation_id=observation.correlation_id,
            interaction_type=interaction_type,
            actor_id=actor_id,
            hand_id=hand_id,
            object_id=object_id,
            target_id=target_id,
            confidence=confidence,
            event_time=observation.event_time,
            evidence=evidence,
        )
