"""Unit tests for deterministic ProcedureEngine."""

import pytest
from astra.contracts.activity import ConfirmedAction
from astra.contracts.base import DecisionType, RunStatus
from astra.contracts.procedure import ProcedureDefinition, ProcedureStep
from astra.contracts.system import EventTopic
from astra.events.bus import EventBus
from astra.procedure.engine import ProcedureEngine


@pytest.fixture
def sample_procedure():
    return ProcedureDefinition(
        id="PROC-TEST",
        experiment_id="EXP-TEST",
        name="Test Experiment",
        steps=[
            ProcedureStep(id="S1", action="OPEN_CONTAINER", allowed_next=["S2"]),
            ProcedureStep(id="S2", action="PICK", object="RED", allowed_next=["S3"]),
            ProcedureStep(id="S3", action="PLACE", object="RED", target="TARGET_A", allowed_next=[]),
        ],
        terminal_step_ids=["S3"],
    )


def test_engine_initialization(sample_procedure):
    """Verify clean start of procedure engine."""
    bus = EventBus()
    started_events = []
    bus.subscribe(EventTopic.PROCEDURE_STARTED, lambda e: started_events.append(e))

    engine = ProcedureEngine(event_bus=bus)
    engine.start("RUN-01", sample_procedure)

    assert len(started_events) == 1
    assert engine.state.status == RunStatus.RUNNING
    assert engine.state.current_step_id is None
    assert engine.state.next_expected == ["S1"]


def test_engine_valid_progression(sample_procedure):
    """Verify sequential transitions."""
    engine = ProcedureEngine()
    engine.start("RUN-02", sample_procedure)

    # Step 1: OPEN_CONTAINER
    act1 = ConfirmedAction(
        message_id="a1",
        source="conf",
        correlation_id="RUN-02",
        action="OPEN_CONTAINER",
        confidence=0.95,
    )
    dec1 = engine.process(act1)
    assert dec1.accepted is True
    assert dec1.decision == DecisionType.VALID
    assert dec1.next_state == "S1"
    assert engine.state.current_step_id == "S1"

    # Step 2: PICK RED
    act2 = ConfirmedAction(
        message_id="a2",
        source="conf",
        correlation_id="RUN-02",
        action="PICK",
        object_id="RED",
        confidence=0.92,
    )
    dec2 = engine.process(act2)
    assert dec2.accepted is True
    assert dec2.next_state == "S2"

    # Step 3: PLACE RED -> TARGET_A
    act3 = ConfirmedAction(
        message_id="a3",
        source="conf",
        correlation_id="RUN-02",
        action="PLACE",
        object_id="RED",
        target_id="TARGET_A",
        confidence=0.97,
    )
    dec3 = engine.process(act3)
    assert dec3.accepted is True
    assert dec3.next_state == "S3"
    assert engine.is_completed is True
    assert engine.state.status == RunStatus.COMPLETED


def test_engine_idle_action_ignored(sample_procedure):
    """Verify that IDLE actions do not advance state or trigger violations."""
    engine = ProcedureEngine()
    engine.start("RUN-03", sample_procedure)

    act_idle = ConfirmedAction(
        message_id="a-idle",
        source="conf",
        correlation_id="RUN-03",
        action="IDLE",
        confidence=0.99,
    )
    dec = engine.process(act_idle)
    assert dec.accepted is False
    assert dec.decision == DecisionType.IGNORED
    assert dec.violation is None
    assert engine.state.current_step_id is None
