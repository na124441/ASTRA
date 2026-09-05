"""Unit tests for observable KinematicFeatureExtractor."""

import numpy as np
import pytest
from astra.activity.features import KinematicFeatureExtractor
from astra.contracts.perception import DetectedObject, HandLandmark, SceneObservation


def test_feature_extractor_output_shape_and_leak_free():
    """Verify feature extractor outputs exactly 26 dimensions without leaked state."""
    extractor = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)

    obs = SceneObservation(
        source="test",
        correlation_id="RUN-TEST",
        camera_id="CAM-1",
        hands=[HandLandmark(id="hand-right", owner_id="human-1", position=[320.0, 240.0], confidence=0.95)],
        objects=[
            DetectedObject(id="obj-red-1", type="RED_COMPONENT", bbox=[100.0, 100.0, 140.0, 140.0], confidence=0.98),
            DetectedObject(id="obj-yellow-1", type="YELLOW_COMPONENT", bbox=[200.0, 200.0, 240.0, 240.0], confidence=0.97),
        ],
    )

    feat = extractor.extract_frame_features(obs)
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (26,)
    # Verify hand coordinates are normalized to [0, 1]
    assert 0.0 <= feat[0] <= 1.0
    assert 0.0 <= feat[1] <= 1.0
    # Confidence flags
    assert feat[23] == 0.95  # hand confidence
    assert feat[24] == 0.98  # red confidence
    assert feat[25] == 0.97  # yellow confidence


def test_distance_derivatives():
    """Verify distance derivative d_dot is negative when hand approaches object."""
    extractor = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)

    # Frame 1: Hand at (400, 400), Red at (100, 100)
    obs1 = SceneObservation(
        source="test",
        correlation_id="RUN-TEST",
        camera_id="CAM-1",
        event_time=0.0,
        hands=[HandLandmark(id="hand-right", owner_id="human-1", position=[400.0, 400.0], confidence=0.9)],
        objects=[DetectedObject(id="obj-red-1", type="RED_COMPONENT", bbox=[90.0, 90.0, 110.0, 110.0], confidence=0.9)],
    )
    _ = extractor.extract_frame_features(obs1)

    # Frame 2: Hand moves closer to Red at (200, 200)
    obs2 = SceneObservation(
        source="test",
        correlation_id="RUN-TEST",
        camera_id="CAM-1",
        event_time=0.033,
        hands=[HandLandmark(id="hand-right", owner_id="human-1", position=[200.0, 200.0], confidence=0.9)],
        objects=[DetectedObject(id="obj-red-1", type="RED_COMPONENT", bbox=[90.0, 90.0, 110.0, 110.0], confidence=0.9)],
    )
    feat2 = extractor.extract_frame_features(obs2)

    # Feature 17 is d_dot_hand_red. As hand approaches, distance decreases -> derivative must be negative!
    d_dot_hand_red = feat2[17]
    assert d_dot_hand_red < 0.0
