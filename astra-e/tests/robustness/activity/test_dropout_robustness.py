"""Robustness tests for perception tracking dropout and sensor noise."""

import random
import numpy as np
from astra.activity.features import KinematicFeatureExtractor
from astra.contracts.perception import DetectedObject, HandLandmark, SceneObservation


def test_feature_extractor_robust_to_dropped_hand():
    """Verify feature extractor handles missing hand detections without crashing or returning NaNs."""
    extractor = KinematicFeatureExtractor()

    # Frame 1: Normal with hand
    obs1 = SceneObservation(
        source="test", correlation_id="RUN-ROBUST", camera_id="CAM-1",
        hands=[HandLandmark(id="hand-right", owner_id="human-1", position=[300.0, 200.0], confidence=0.9)],
        objects=[DetectedObject(id="obj-red", type="RED_COMPONENT", bbox=[100.0, 100.0, 140.0, 140.0], confidence=0.9)],
    )
    f1 = extractor.extract_frame_features(obs1)
    assert not np.isnan(f1).any()
    assert f1[23] == 0.9  # hand conf

    # Frame 2: Hand dropped (occluded) -> hands list empty
    obs2 = SceneObservation(
        source="test", correlation_id="RUN-ROBUST", camera_id="CAM-1",
        hands=[],
        objects=[DetectedObject(id="obj-red", type="RED_COMPONENT", bbox=[100.0, 100.0, 140.0, 140.0], confidence=0.9)],
    )
    f2 = extractor.extract_frame_features(obs2)
    assert not np.isnan(f2).any()
    assert f2[23] == 0.0  # hand conf dropped to 0
    # Velocity should be zeroed rather than producing extreme values
    assert f2[2] == 0.0
    assert f2[3] == 0.0
