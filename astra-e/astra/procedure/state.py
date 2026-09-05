"""Runtime procedure state tracking and snapshot management."""

from __future__ import annotations

from astra.contracts.base import RunStatus
from astra.contracts.procedure import ProcedureDefinition, ProcedureRuntimeState, ProcedureStep
from astra.procedure.graph import ProcedureGraph


class ProcedureStateManager:
    """
    Maintains mutable runtime execution state for an active experiment run.
    Produces immutable ProcedureRuntimeState snapshots.
    """

    def __init__(self, run_id: str, graph: ProcedureGraph) -> None:
        self.run_id = run_id
        self.graph = graph
        self.definition = graph.definition
        self.current_step: ProcedureStep | None = None
        self.completed_steps: list[str] = []
        self.violations_count: int = 0
        self.status: RunStatus = RunStatus.INITIALIZING
        self.last_action_desc: str | None = None

    def initialize(self) -> None:
        """Mark procedure ready to receive the first step."""
        self.status = RunStatus.RUNNING
        self.current_step = None
        self.completed_steps = []
        self.violations_count = 0
        self.last_action_desc = "INITIALIZED"

    def advance(self, next_step: ProcedureStep, action_desc: str) -> None:
        """Advance the state machine to next_step."""
        if self.current_step:
            self.completed_steps.append(self.current_step.id)
        self.current_step = next_step
        self.last_action_desc = action_desc

        if self.graph.is_terminal_step(next_step.id):
            self.status = RunStatus.COMPLETED

    def record_violation(self) -> None:
        """Increment count of observed procedural violations."""
        self.violations_count += 1

    def pause(self) -> None:
        """Pause procedure tracking."""
        self.status = RunStatus.PAUSED

    def resume(self) -> None:
        """Resume procedure tracking."""
        if self.status == RunStatus.PAUSED:
            self.status = RunStatus.RUNNING

    def abort(self) -> None:
        """Abort procedure execution."""
        self.status = RunStatus.ABORTED

    def get_snapshot(self) -> ProcedureRuntimeState:
        """Generate an immutable ProcedureRuntimeState contract."""
        current_id = self.current_step.id if self.current_step else None
        next_allowed = [s.id for s in self.graph.get_allowed_next_steps(current_id)]

        all_step_ids = [s.id for s in self.definition.steps]
        pending = [
            sid for sid in all_step_ids
            if sid not in self.completed_steps and sid != current_id
        ]

        return ProcedureRuntimeState(
            experiment_id=self.definition.experiment_id,
            procedure_version=self.definition.version,
            run_id=self.run_id,
            current_step_id=current_id,
            current_action=self.current_step.action if self.current_step else None,
            next_expected=next_allowed,
            completed_steps=list(self.completed_steps),
            pending_steps=pending,
            status=self.status,
            last_action=self.last_action_desc,
            violations_count=self.violations_count,
        )
