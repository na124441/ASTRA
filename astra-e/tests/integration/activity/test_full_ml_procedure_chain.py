"""End-to-End Integration test: Camera -> Perception -> Causal ML -> Confirmation -> Procedure Engine."""

from astra.activity.pipeline import ActivityPipeline
from astra.activity.recognizer import ActivityRecognizer
from astra.contracts.activity import ConfirmedAction
from astra.contracts.procedure import ProcedureDefinition, ProcedureStep
from astra.events.bus import EventBus
from astra.perception.pipeline import PerceptionPipeline
from astra.procedure.engine import ProcedureEngine
from astra.video.buffer import FrameBuffer
from astra.video.camera import MockCamera


def test_full_ml_to_procedure_integration():
    """
    Verify complete integrated pipeline executes without failure:
    MockCamera -> FrameBuffer -> PerceptionPipeline -> ActivityPipeline -> ProcedureEngine.
    """
    event_bus = EventBus()
    buffer = FrameBuffer(capacity=60)
    camera = MockCamera(width=640, height=480, total_frames=120)
    perception = PerceptionPipeline(event_bus=event_bus)

    # Load trained model weights
    recognizer = ActivityRecognizer(model_path="models/activity/temporal_model.pt")
    activity_pipe = ActivityPipeline(recognizer=recognizer, event_bus=event_bus)

    proc_def = ProcedureDefinition(
        id="PROC-INTEG",
        experiment_id="EXP-001",
        steps=[
            ProcedureStep(id="S01", action="APPROACH", object="RED_COMPONENT", allowed_next=["S02"]),
            ProcedureStep(id="S02", action="PICK", object="RED_COMPONENT", allowed_next=["S03"]),
            ProcedureStep(id="S03", action="MOVE", object="RED_COMPONENT", allowed_next=[]),
        ],
        initial_step_id="S01",
    )
    engine = ProcedureEngine(event_bus=event_bus)
    engine.start(run_id="INTEG-RUN-01", procedure=proc_def)

    camera.start()
    confirmed_actions: list[ConfirmedAction] = []

    for _ in range(120):
        ok, frame, ts = camera.read()
        if not ok or frame is None:
            break

        vf = buffer.push(camera.camera_id, frame, ts, correlation_id="INTEG-RUN-01")
        obs = perception.process_frame(vf, frame)
        confirmed = activity_pipe.process_observation(obs)

        if confirmed is not None:
            confirmed_actions.append(confirmed)
            engine.process(confirmed)

    camera.stop()

    # The pipeline should have processed 120 frames and produced confirmed actions
    assert len(buffer) > 0
    assert len(confirmed_actions) >= 1
    # Check that confirmed action carries confirmation evidence
    assert confirmed_actions[0].confirmation is not None
    assert confirmed_actions[0].confirmation.stable_frames >= 4
