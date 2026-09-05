"""Unit tests for Perception detectors, tracking, and pipeline."""

from astra.contracts.video import VideoFrame
from astra.perception.detector import ColorExperimentDetector
from astra.perception.pipeline import PerceptionPipeline
from astra.perception.tracker import MultiObjectTracker
from astra.perception.types import RawDetection
from astra.video.camera import MockCamera


def test_color_detector_on_mock_frame():
    """Verify ColorExperimentDetector identifies payload objects in synthetic scene."""
    cam = MockCamera(width=640, height=480)
    cam.start()
    ok, frame, ts = cam.read()
    assert ok and frame is not None

    detector = ColorExperimentDetector()
    detections = detector.detect(frame)

    classes_found = {d.class_name for d in detections}
    assert "RED_COMPONENT" in classes_found
    assert "YELLOW_COMPONENT" in classes_found
    assert "CONTAINER" in classes_found
    assert "TARGET_A" in classes_found
    assert "TARGET_B" in classes_found
    assert "HAND" in classes_found

    cam.stop()


def test_multi_object_tracker_id_stability():
    """Verify tracker preserves stable identity across consecutive frames."""
    tracker = MultiObjectTracker(max_distance=50.0)

    # Frame 1: Red component at (100, 100)
    det1 = [RawDetection(class_name="RED_COMPONENT", bbox=[90.0, 90.0, 110.0, 110.0], confidence=0.95)]
    tracks1 = tracker.update(det1, timestamp=0.0)
    assert len(tracks1) == 1
    assigned_id = tracks1[0].track_id
    assert assigned_id.startswith("obj-red")

    # Frame 2: Red component moves slightly to (105, 103)
    det2 = [RawDetection(class_name="RED_COMPONENT", bbox=[95.0, 93.0, 115.0, 113.0], confidence=0.96)]
    tracks2 = tracker.update(det2, timestamp=0.033)
    assert len(tracks2) == 1
    # Must retain the EXACT SAME track ID
    assert tracks2[0].track_id == assigned_id
    # Must have computed velocity
    assert tracks2[0].velocity != [0.0, 0.0]


def test_perception_pipeline_scene_observation():
    """Verify PerceptionPipeline produces valid SceneObservation contracts."""
    cam = MockCamera(width=640, height=480)
    cam.start()
    ok, frame, ts = cam.read()
    assert ok and frame is not None

    vf = VideoFrame(
        source="cam",
        correlation_id="RUN-TEST",
        frame_id=1,
        camera_id="CAM-01",
        width=640,
        height=480,
        frame_reference="memory://frame/1",
    )

    pipeline = PerceptionPipeline()
    obs = pipeline.process_frame(vf, frame)

    assert obs.camera_id == "CAM-01"
    assert obs.correlation_id == "RUN-TEST"
    assert len(obs.objects) >= 5
    assert len(obs.hands) >= 1

    cam.stop()
