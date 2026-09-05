"""Unit tests for Spatial heuristics and InteractionAnalyzer."""

from astra.contracts.perception import DetectedObject, HandLandmark, SceneObservation
from astra.interaction.analyzer import InteractionAnalyzer
from astra.interaction.spatial import (
    bbox_centroid,
    compute_co_movement_score,
    compute_iou,
    euclidean_distance,
    point_in_bbox,
)


def test_spatial_primitives():
    """Verify centroid, euclidean distance, point containment, and IoU calculations."""
    bbox = [10.0, 20.0, 30.0, 40.0]
    assert bbox_centroid(bbox) == [20.0, 30.0]
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == 5.0
    assert point_in_bbox([20.0, 30.0], bbox) is True
    assert point_in_bbox([50.0, 50.0], bbox) is False

    # Exact overlap IoU
    assert compute_iou(bbox, bbox) == 1.0
    # Disjoint boxes
    assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_co_movement_scoring():
    """Verify velocity co-movement calculation."""
    # Both moving in same direction with same velocity
    score_same = compute_co_movement_score([10.0, 0.0], [10.0, 0.0])
    assert score_same > 0.8

    # Moving in opposite directions
    score_opp = compute_co_movement_score([10.0, 0.0], [-10.0, 0.0])
    assert score_opp <= 0.3


def test_interaction_analyzer_grasp_and_release():
    """Verify grasp and release state transitions."""
    analyzer = InteractionAnalyzer(approach_threshold=80.0, touch_threshold=30.0)

    # Frame 1: Hand approaches Red Component
    obs1 = SceneObservation(
        source="test",
        correlation_id="RUN-1",
        camera_id="CAM-1",
        hands=[HandLandmark(id="hand-right", owner_id="human-1", position=[100.0, 100.0], confidence=0.9)],
        objects=[
            DetectedObject(id="obj-red-1", type="RED_COMPONENT", bbox=[150.0, 100.0, 170.0, 120.0], confidence=0.9),
            DetectedObject(id="target-a-1", type="TARGET_A", bbox=[300.0, 100.0, 400.0, 200.0], confidence=0.9),
        ],
    )
    events1 = analyzer.analyze(obs1)
    types1 = [e.interaction_type for e in events1]
    assert "APPROACH" in types1

    # Frame 2: Hand reaches object (distance <= 30px) -> GRASP and PICK
    obs2 = SceneObservation(
        source="test",
        correlation_id="RUN-1",
        camera_id="CAM-1",
        hands=[HandLandmark(id="hand-right", owner_id="human-1", position=[155.0, 105.0], confidence=0.9)],
        objects=[
            DetectedObject(id="obj-red-1", type="RED_COMPONENT", bbox=[150.0, 100.0, 170.0, 120.0], confidence=0.9),
            DetectedObject(id="target-a-1", type="TARGET_A", bbox=[300.0, 100.0, 400.0, 200.0], confidence=0.9),
        ],
    )
    events2 = analyzer.analyze(obs2)
    types2 = [e.interaction_type for e in events2]
    assert "GRASP" in types2
    assert "PICK" in types2

    # Frame 3: Hand moves object into TARGET_A (position: 350, 150) -> PLACE
    obs3 = SceneObservation(
        source="test",
        correlation_id="RUN-1",
        camera_id="CAM-1",
        hands=[HandLandmark(id="hand-right", owner_id="human-1", position=[350.0, 150.0], confidence=0.9)],
        objects=[
            DetectedObject(id="obj-red-1", type="RED_COMPONENT", bbox=[340.0, 140.0, 360.0, 160.0], confidence=0.9),
            DetectedObject(id="target-a-1", type="TARGET_A", bbox=[300.0, 100.0, 400.0, 200.0], confidence=0.9),
        ],
    )
    events3 = analyzer.analyze(obs3)
    types3 = [e.interaction_type for e in events3]
    assert "PLACE" in types3

    # Frame 4: Hand moves away while object stays in TARGET_A -> RELEASE
    obs4 = SceneObservation(
        source="test",
        correlation_id="RUN-1",
        camera_id="CAM-1",
        hands=[HandLandmark(id="hand-right", owner_id="human-1", position=[450.0, 250.0], confidence=0.9)],
        objects=[
            DetectedObject(id="obj-red-1", type="RED_COMPONENT", bbox=[340.0, 140.0, 360.0, 160.0], confidence=0.9),
            DetectedObject(id="target-a-1", type="TARGET_A", bbox=[300.0, 100.0, 400.0, 200.0], confidence=0.9),
        ],
    )
    events4 = analyzer.analyze(obs4)
    types4 = [e.interaction_type for e in events4]
    assert "RELEASE" in types4
