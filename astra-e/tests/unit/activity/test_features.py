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


def test_reset_clears_all_state_and_cached_positions():
    """Verify reset() restores both temporal derivatives and cached occlusion positions."""
    extractor = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)

    # Mutate state with an observation
    obs = SceneObservation(
        source="test",
        correlation_id="RUN-TEST",
        camera_id="CAM-1",
        hands=[HandLandmark(id="h1", owner_id="u1", position=[10.0, 20.0], confidence=0.8)],
        objects=[
            DetectedObject(id="o1", type="RED_COMPONENT", bbox=[30.0, 30.0, 50.0, 50.0], confidence=0.9),
            DetectedObject(id="o2", type="YELLOW_COMPONENT", bbox=[60.0, 60.0, 80.0, 80.0], confidence=0.9),
        ],
    )
    _ = extractor.extract(obs)
    assert extractor._last_hand_pos == [10.0, 20.0]

    # Reset
    extractor.reset()
    assert extractor._prev_time is None
    assert extractor._prev_dist_hand_red is None
    assert extractor._last_hand_pos == [640.0 * 0.8, 480.0 * 0.8]
    assert extractor._last_red_pos == [640.0 * 0.25, 480.0 * 0.5]
    assert extractor._last_yellow_pos == [640.0 * 0.35, 480.0 * 0.6]


def test_extract_from_detector_dict_parity():
    """Verify single production KinematicFeatureExtractor handles frozen detector dicts."""
    extractor = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)

    detections = {
        "hand": {"pos": [320.0, 240.0], "conf": 0.95},
        "red": {"pos": [120.0, 120.0], "conf": 0.98},
        "yellow": {"pos": [220.0, 220.0], "conf": 0.97},
        "container": {"pos": [180.0, 280.0]},
        "target_a": {"pos": [460.0, 150.0]},
        "target_b": {"pos": [460.0, 310.0]},
    }

    feat = extractor.extract(detections, event_time=0.0)
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (26,)
    assert feat[0] == pytest.approx(320.0 / 640.0)
    assert feat[1] == pytest.approx(240.0 / 480.0)
    assert feat[23] == 0.95  # conf_hand
    assert feat[24] == 0.98  # conf_red
    assert feat[25] == 0.97  # conf_yellow


def test_frozen_detector_contract_exact_spec():
    """Verify the exact user-specified detector dictionary contract."""
    extractor = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)

    # Frame 1
    det_f1 = {
        "hand": {"pos": [300.0, 200.0], "conf": 0.95},
        "red": {"pos": [150.0, 150.0], "conf": 0.92},
        "yellow": {"pos": [250.0, 250.0], "conf": 0.89},
        "container": {"pos": [180.0, 280.0]},
        "target_a": {"pos": [460.0, 150.0]},
        "target_b": {"pos": [460.0, 310.0]},
    }
    f1 = extractor.extract(det_f1, event_time=0.0)
    assert f1.shape == (26,)
    assert f1[23] == 0.95
    assert f1[24] == 0.92
    assert f1[25] == 0.89

    # Frame 2 (dt = 0.1s): Hand moves by dx = +32px, dy = +24px
    det_f2 = {
        "hand": {"pos": [332.0, 224.0], "conf": 0.96},
        "red": {"pos": [150.0, 150.0], "conf": 0.93},
        "yellow": {"pos": [250.0, 250.0], "conf": 0.90},
        "container": {"pos": [180.0, 280.0]},
        "target_a": {"pos": [460.0, 150.0]},
        "target_b": {"pos": [460.0, 310.0]},
    }
    f2 = extractor.extract(det_f2, event_time=0.1)
    # Velocity vx = (332 - 300) / 0.1 = 320 px/sec -> normalized vx = 320 / 640 = 0.5 1/sec
    assert f2[2] == pytest.approx(0.5, abs=1e-4)
    # Velocity vy = (224 - 200) / 0.1 = 240 px/sec -> normalized vy = 240 / 480 = 0.5 1/sec
    assert f2[3] == pytest.approx(0.5, abs=1e-4)


def test_frozen_detector_contract_missing_entities_zero_velocity_and_confidence():
    """Verify omitted entities in detector dict zero out velocity and confidence safely."""
    extractor = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)

    # Frame 1: Hand and Red detected
    det1 = {
        "hand": {"pos": [300.0, 200.0], "conf": 0.95},
        "red": {"pos": [150.0, 150.0], "conf": 0.92},
    }
    _ = extractor.extract(det1, event_time=0.0)

    # Frame 2: Hand occluded / lost (omitted from detections)
    det2 = {
        "red": {"pos": [150.0, 150.0], "conf": 0.92},
    }
    f2 = extractor.extract(det2, event_time=0.033)
    assert not np.isnan(f2).any()
    assert f2[23] == 0.0  # hand confidence zeroed
    assert f2[2] == 0.0   # hand vx zeroed
    assert f2[3] == 0.0   # hand vy zeroed
    # Hand position retained from frame 1
    assert f2[0] == pytest.approx(300.0 / 640.0)
    assert f2[1] == pytest.approx(200.0 / 480.0)


def test_scene_observation_parity_with_detector_dict():
    """Verify SceneObservation produces identical features as direct detector dict."""
    extractor1 = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)
    extractor2 = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)

    obs = SceneObservation(
        source="test",
        correlation_id="RUN-PARITY",
        camera_id="CAM-1",
        event_time=1.0,
        hands=[HandLandmark(id="h1", owner_id="p1", position=[320.0, 240.0], confidence=0.95)],
        objects=[
            DetectedObject(id="o1", type="RED_COMPONENT", bbox=[100.0, 100.0, 140.0, 140.0], confidence=0.92),
            DetectedObject(id="o2", type="YELLOW_COMPONENT", bbox=[200.0, 200.0, 240.0, 240.0], confidence=0.89),
            DetectedObject(id="o3", type="CONTAINER", bbox=[160.0, 260.0, 200.0, 300.0], confidence=0.99),
            DetectedObject(id="o4", type="TARGET_A", bbox=[440.0, 130.0, 480.0, 170.0], confidence=0.99),
            DetectedObject(id="o5", type="TARGET_B", bbox=[440.0, 290.0, 480.0, 330.0], confidence=0.99),
        ],
    )

    detections = {
        "event_time": 1.0,
        "hand": {"pos": [320.0, 240.0], "conf": 0.95},
        "red": {"pos": [120.0, 120.0], "conf": 0.92},
        "yellow": {"pos": [220.0, 220.0], "conf": 0.89},
        "container": {"pos": [180.0, 280.0], "conf": 0.99},
        "target_a": {"pos": [460.0, 150.0], "conf": 0.99},
        "target_b": {"pos": [460.0, 310.0], "conf": 0.99},
    }

    f_obs = extractor1.extract(obs)
    f_dict = extractor2.extract(detections)

    assert np.allclose(f_obs, f_dict, atol=1e-6)

