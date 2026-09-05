"""ActivityPipeline linking SceneObservations to ConfirmedActions via causal ML and confirmation."""

from __future__ import annotations

import logging
from astra.activity.confirmation import ActionConfirmationEngine
from astra.activity.features import KinematicFeatureExtractor
from astra.activity.recognizer import ActivityRecognizer
from astra.contracts.activity import ActionObservation, ConfirmedAction
from astra.contracts.perception import SceneObservation
from astra.contracts.system import EventTopic
from astra.events.bus import EventBus

logger = logging.getLogger("astra.activity.pipeline")


class ActivityPipeline:
    """
    Complete Temporal Activity Recognition and Confirmation Pipeline.
    SceneObservation -> Kinematic Features -> Causal LSTM -> Calibrated ActionObservation
    -> Temporal Confirmation (EMA & Hysteresis) -> ConfirmedAction.
    """

    def __init__(
        self,
        feature_extractor: KinematicFeatureExtractor | None = None,
        recognizer: ActivityRecognizer | None = None,
        confirmation_engine: ActionConfirmationEngine | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.extractor = feature_extractor or KinematicFeatureExtractor()
        self.recognizer = recognizer or ActivityRecognizer()
        self.confirmation = confirmation_engine or ActionConfirmationEngine()
        self.event_bus = event_bus

    def reset(self) -> None:
        """Reset internal pipeline buffers."""
        self.extractor.reset()
        self.recognizer.reset()
        self.confirmation.reset()

    def process_observation(self, observation: SceneObservation) -> ConfirmedAction | None:
        """
        Process single SceneObservation.
        Returns ConfirmedAction if a stable action was confirmed.
        """
        # 1. Feature extraction
        features = self.extractor.extract_frame_features(observation)

        # 2. Causal temporal ML inference
        action_obs = self.recognizer.push_frame_features(
            features=features,
            timestamp=observation.event_time,
            correlation_id=observation.correlation_id,
        )

        if action_obs is None:
            return None

        # Publish raw action observation
        if self.event_bus is not None:
            self.event_bus.publish(EventTopic.ACTION_OBSERVED, action_obs)

        # 3. Confirmation layer evaluation
        confirmed_action = self.confirmation.process_observation(action_obs)

        if confirmed_action is not None and self.event_bus is not None:
            self.event_bus.publish(EventTopic.ACTION_CONFIRMED, confirmed_action)

        return confirmed_action
