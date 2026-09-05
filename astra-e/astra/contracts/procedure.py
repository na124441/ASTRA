"""Procedure definition, state machine tracking, and transition decision contracts."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from astra.contracts.base import BaseMessage, DecisionType, RunStatus
from astra.contracts.violation import ViolationEvent


class ProcedureStep(BaseModel):
    """Definition of an individual step in the procedure graph."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Step ID, e.g. S01, S02")
    action: str = Field(description="Expected action verb, e.g. PICK, PLACE, OPEN_CONTAINER")
    object: str | None = Field(default=None, description="Expected target object, e.g. RED_COMPONENT")
    target: str | None = Field(default=None, description="Expected target container/zone, e.g. TARGET_A")
    description: str = Field(default="", description="Human-readable description of this step")
    allowed_next: list[str] = Field(default_factory=list, description="Explicit outgoing transitions to next step IDs")
    optional: bool = Field(default=False, description="Whether this step can be bypassed legitimately")
    repeatable: bool = Field(default=False, description="Whether this step can be re-executed multiple times")


class ProcedureDefinition(BaseModel):
    """
    Contract 06: ProcedureDefinition.
    Immutable declarative schema of the experiment procedure graph.
    """
    model_config = ConfigDict(frozen=True)

    id: str = Field(default="PROC-001", description="Unique procedure identifier")
    experiment_id: str = Field(description="Associated experiment ID, e.g. EXP-001")
    name: str = Field(default="", description="Descriptive name of the procedure")
    version: str = Field(default="1.0", description="Procedure version string")
    objects: list[str] = Field(default_factory=list, description="Registered objects in experiment workspace")
    targets: list[str] = Field(default_factory=list, description="Registered target zones in experiment workspace")
    steps: list[ProcedureStep] = Field(default_factory=list, description="All states/steps forming the procedure graph")
    initial_step_id: str | None = Field(default=None, description="Starting step ID (defaults to first step if None)")
    terminal_step_ids: list[str] = Field(default_factory=list, description="Valid terminal step IDs")


class ProcedureRuntimeState(BaseModel):
    """
    Procedure state snapshot maintained during experiment execution.
    """
    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(description="Experiment identifier")
    procedure_version: str = Field(description="Active procedure version")
    run_id: str = Field(description="Current execution run ID")
    current_step_id: str | None = Field(default=None, description="Current step ID in graph, None if START")
    current_action: str | None = Field(default=None, description="Last validated action name")
    next_expected: list[str] = Field(default_factory=list, description="Next valid step IDs")
    completed_steps: list[str] = Field(default_factory=list, description="List of completed step IDs")
    pending_steps: list[str] = Field(default_factory=list, description="List of steps yet to be executed")
    status: RunStatus = Field(default=RunStatus.RUNNING, description="Current state machine status")
    last_action: str | None = Field(default=None, description="Summary of last processed action")
    violations_count: int = Field(default=0, description="Total violations encountered so far")


class ProcedureDecision(BaseMessage):
    """
    Contract 07: ProcedureDecision.
    Result of evaluating an action against the deterministic procedure state machine.
    """
    accepted: bool = Field(description="True if transition was accepted, False if invalid or rejected")
    current_state: str | None = Field(default=None, description="Step ID prior to transition")
    next_state: str | None = Field(default=None, description="Step ID after transition")
    observed_action: dict[str, Any] = Field(default_factory=dict, description="Action details evaluated")
    decision: DecisionType = Field(description="VALID, INVALID, UNCERTAIN, IGNORED")
    violation: ViolationEvent | None = Field(default=None, description="Attached violation event if INVALID")
