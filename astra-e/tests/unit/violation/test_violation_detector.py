"""Unit tests for deterministic ViolationDetector."""

import time
import pytest
from astra.contracts.activity import ConfirmedAction
from astra.contracts.base import ViolationType
from astra.contracts.procedure import ProcedureDefinition, ProcedureStep
from astra.events.bus import EventBus
from astra.procedure.engine import ProcedureEngine
from astra.violation.detector import ViolationDetector


@pytest.fixture
def procedure():
    return ProcedureDefinition(
        id="PROC-VIOL-TEST",
        experiment_id="EXP-V",
        name="Violation Test Procedure",
        objects=["RED", "YELLOW"],
        targets=["TARGET_A", "TARGET_B"],
        steps=[
            ProcedureStep(id="S01", action="OPEN_CONTAINER", allowed_next=["S02"]),
            ProcedureStep(id="S02", action="PICK", object="RED", allowed_next=["S03"]),
            ProcedureStep(id="S03", action="PLACE", object="RED", target="TARGET_A", allowed_next=["S04"]),
            ProcedureStep(id="S04", action="PICK", object="YELLOW", allowed_next=["S05"]),
            ProcedureStep(id="S05", action="PLACE", object="YELLOW", target="TARGET_B", allowed_next=["S06"]),
            ProcedureStep(id="S06", action="CLOSE_CONTAINER", allowed_next=[]),
        ],
        initial_step_id="S01",
        terminal_step_ids=["S06"],
    )


def test_violation_wrong_object(procedure):
    """Verify WRONG_OBJECT classification when incorrect object is interacted with."""
    engine = ProcedureEngine()
    engine.start("RUN-V1", procedure)

    # Valid S01
    engine.process(ConfirmedAction(
        message_id="m1", source="conf", correlation_id="RUN-V1",
        action="OPEN_CONTAINER", confidence=0.95,
    ))

    # Next expected is S02: PICK RED. Astronaut attempts PICK YELLOW!
    bad_action = ConfirmedAction(
        message_id="m2", source="conf", correlation_id="RUN-V1",
        action="PICK", object_id="YELLOW", confidence=0.93,
    )
    decision = engine.process(bad_action)
    assert decision.accepted is False
    assert decision.violation is not None
    assert decision.violation.violation_type == ViolationType.WRONG_OBJECT


def test_violation_wrong_target(procedure):
    """Verify WRONG_TARGET classification when object is placed into incorrect zone."""
    engine = ProcedureEngine()
    engine.start("RUN-V2", procedure)

    # S01: OPEN
    engine.process(ConfirmedAction(message_id="m1", source="conf", correlation_id="RUN-V2", action="OPEN_CONTAINER", confidence=0.95))
    # S02: PICK RED
    engine.process(ConfirmedAction(message_id="m2", source="conf", correlation_id="RUN-V2", action="PICK", object_id="RED", confidence=0.95))

    # Expected S03: PLACE RED -> TARGET_A. Astronaut places in TARGET_B!
    bad_place = ConfirmedAction(
        message_id="m3", source="conf", correlation_id="RUN-V2",
        action="PLACE", object_id="RED", target_id="TARGET_B", confidence=0.94,
    )
    decision = engine.process(bad_place)
    assert decision.accepted is False
    assert decision.violation is not None
    assert decision.violation.violation_type == ViolationType.WRONG_TARGET


def test_violation_skipped_step(procedure):
    """Verify SKIPPED_STEP classification when intermediate step is skipped."""
    engine = ProcedureEngine()
    engine.start("RUN-V3", procedure)

    # S01: OPEN
    engine.process(ConfirmedAction(message_id="m1", source="conf", correlation_id="RUN-V3", action="OPEN_CONTAINER", confidence=0.95))
    # S02: PICK RED
    engine.process(ConfirmedAction(message_id="m2", source="conf", correlation_id="RUN-V3", action="PICK", object_id="RED", confidence=0.95))
    # S03: PLACE RED -> TARGET_A
    engine.process(ConfirmedAction(message_id="m3", source="conf", correlation_id="RUN-V3", action="PLACE", object_id="RED", target_id="TARGET_A", confidence=0.95))

    # S04 is PICK YELLOW. But astronaut directly tries to CLOSE_CONTAINER (S06)!
    skip_action = ConfirmedAction(
        message_id="m4", source="conf", correlation_id="RUN-V3",
        action="CLOSE_CONTAINER", confidence=0.91,
    )
    decision = engine.process(skip_action)
    assert decision.accepted is False
    assert decision.violation is not None
    assert decision.violation.violation_type == ViolationType.SKIPPED_STEP
    assert "S04" in decision.violation.expected.get("skipped_steps", [])


def test_violation_repeated_action(procedure):
    """Verify REPEATED_ACTION detection on re-executing an already completed step."""
    engine = ProcedureEngine()
    engine.start("RUN-V4", procedure)

    # S01: OPEN
    engine.process(ConfirmedAction(message_id="m1", source="conf", correlation_id="RUN-V4", action="OPEN_CONTAINER", confidence=0.95))
    # S02: PICK RED
    engine.process(ConfirmedAction(message_id="m2", source="conf", correlation_id="RUN-V4", action="PICK", object_id="RED", confidence=0.95))

    # Astronaut attempts S01 again: OPEN_CONTAINER
    repeat_action = ConfirmedAction(
        message_id="m3", source="conf", correlation_id="RUN-V4",
        action="OPEN_CONTAINER", confidence=0.90,
    )
    decision = engine.process(repeat_action)
    assert decision.accepted is False
    assert decision.violation is not None
    assert decision.violation.violation_type == ViolationType.REPEATED_ACTION


def test_alert_suppression_cooldown(procedure):
    """Verify FR-022 alert suppression prevents continuous duplicate event spam."""
    detector = ViolationDetector(suppression_cooldown_seconds=1.0)
    engine = ProcedureEngine(violation_detector=detector)
    engine.start("RUN-V5", procedure)

    bad_action = ConfirmedAction(
        message_id="m1", source="conf", correlation_id="RUN-V5",
        action="PICK", object_id="YELLOW", confidence=0.93,
    )

    # First attempt: violation returned, suppression check is False (alert allowed)
    dec1 = engine.process(bad_action)
    assert dec1.violation is not None
    # Now check should_suppress_alert directly on next immediate arrival
    assert detector.should_suppress_alert(dec1.violation) is True

    # After cooldown expires, alert should be permitted again
    time.sleep(1.05)
    assert detector.should_suppress_alert(dec1.violation) is False
