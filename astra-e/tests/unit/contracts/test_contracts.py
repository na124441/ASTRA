"""Unit tests for immutable Pydantic contracts."""

import pytest
from pydantic import ValidationError
from astra.contracts.activity import (
    ActionObservation,
    ConfirmationMetadata,
    ConfirmedAction,
)
from astra.contracts.base import (
    ActionType,
    BaseMessage,
    DecisionType,
    RunStatus,
    Severity,
    ViolationType,
)
from astra.contracts.procedure import (
    ProcedureDecision,
    ProcedureDefinition,
    ProcedureRuntimeState,
    ProcedureStep,
)
from astra.contracts.violation import ViolationEvent
from astra.contracts.assistance import AssistanceEvent, AssistancePriority


def test_base_message_immutability():
    """Verify BaseMessage contracts are frozen and immutable."""
    msg = BaseMessage(
        message_id="msg-1",
        schema_version="1.0",
        source="test-source",
        correlation_id="corr-1",
    )
    with pytest.raises(ValidationError):
        # Trying to mutate a frozen model field must raise ValidationError
        msg.source = "new-source"  # type: ignore


def test_confirmed_action_serialization():
    """Verify ConfirmedAction contract serializes and deserializes accurately."""
    action = ConfirmedAction(
        message_id="act-101",
        source="confidence-mgr",
        correlation_id="run-1",
        action=ActionType.PICK.value,
        object_id="RED_COMPONENT",
        target_id=None,
        confidence=0.98,
        confirmation=ConfirmationMetadata(
            stable_frames=10,
            minimum_confidence=0.90,
            temporal_consistency=0.95,
        ),
    )
    json_data = action.model_dump_json()
    restored = ConfirmedAction.model_validate_json(json_data)

    assert restored.message_id == "act-101"
    assert restored.action == "PICK"
    assert restored.object_id == "RED_COMPONENT"
    assert restored.confidence == 0.98
    assert restored.confirmation is not None
    assert restored.confirmation.stable_frames == 10


def test_procedure_definition_contract():
    """Verify ProcedureDefinition structure."""
    proc = ProcedureDefinition(
        id="PROC-01",
        experiment_id="EXP-01",
        name="Test",
        objects=["RED_OBJ"],
        steps=[
            ProcedureStep(
                id="S01",
                action="PICK",
                object="RED_OBJ",
                allowed_next=["S02"],
            ),
            ProcedureStep(
                id="S02",
                action="PLACE",
                object="RED_OBJ",
                target="TARGET_A",
                allowed_next=[],
            ),
        ],
        terminal_step_ids=["S02"],
    )
    assert len(proc.steps) == 2
    assert proc.steps[0].id == "S01"
    assert proc.steps[1].id == "S02"


def test_violation_event_contract():
    """Verify ViolationEvent model fields."""
    viol = ViolationEvent(
        message_id="viol-1",
        source="violation-engine",
        correlation_id="run-1",
        violation_type=ViolationType.WRONG_OBJECT,
        expected={"object": "RED"},
        observed={"object": "YELLOW"},
        severity=Severity.WARNING,
        message="Wrong object picked",
    )
    assert viol.violation_type == ViolationType.WRONG_OBJECT
    assert viol.severity == Severity.WARNING
    assert viol.expected["object"] == "RED"


def test_assistance_event_contract():
    """Verify AssistanceEvent delivery channels and priority."""
    assist = AssistanceEvent(
        message_id="assist-1",
        source="assistance-engine",
        correlation_id="run-1",
        type="PROCEDURE_WARNING",
        priority=AssistancePriority.HIGH,
        message="Please use Target B",
    )
    assert assist.priority == AssistancePriority.HIGH
    assert "TTS" in [ch.value for ch in assist.channels]
