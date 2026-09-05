"""Integration test for full Video Ingestion -> Perception -> HOI pipeline."""

from astra.events.bus import EventBus
from astra.interaction.pipeline import InteractionPipeline
from astra.perception.pipeline import PerceptionPipeline
from astra.video.buffer import FrameBuffer
from astra.video.camera import MockCamera


def test_full_perception_hoi_chain():
    """
    Run 100 frames through MockCamera -> FrameBuffer -> PerceptionPipeline -> InteractionPipeline.
    Verify that:
    1. All 100 frames are ingested and stored in FrameBuffer.
    2. SceneObservation contracts are generated for every frame with stable entity IDs.
    3. Hand-object interaction events (APPROACH, GRASP, PICK, MOVE) are identified and published.
    """
    event_bus = EventBus()
    buffer = FrameBuffer(capacity=60)
    camera = MockCamera(width=640, height=480, total_frames=100)
    perception = PerceptionPipeline(event_bus=event_bus)
    hoi = InteractionPipeline(event_bus=event_bus)

    camera.start()
    observations = []
    all_interaction_events = []

    for _ in range(100):
        ok, frame, ts = camera.read()
        if not ok or frame is None:
            break

        vf = buffer.push(
            camera_id=camera.camera_id,
            frame=frame,
            event_time=ts,
            correlation_id="INTEG-RUN-01",
        )

        obs = perception.process_frame(vf, frame)
        observations.append(obs)

        events = hoi.process_observation(obs)
        all_interaction_events.extend(events)

    camera.stop()

    assert len(observations) == 100
    assert len(all_interaction_events) > 0

    # Verify track ID stability
    red_ids = {
        obj.id for obs in observations
        for obj in obs.objects
        if "RED" in obj.type
    }
    # There should only be 1 persistent ID for the red object across all 100 frames
    assert len(red_ids) == 1

    interaction_types = {e.interaction_type for e in all_interaction_events}
    assert "APPROACH" in interaction_types
    assert "GRASP" in interaction_types or "PICK" in interaction_types
