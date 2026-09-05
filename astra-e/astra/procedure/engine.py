"""Core deterministic ProcedureEngine for ASTRA-E."""

from __future__ import annotations

import logging
import time
from typing import Any
from astra.contracts.activity import ConfirmedAction
from astra.contracts.base import DecisionType, RunStatus, default_uuid
from astra.contracts.procedure import (
    ProcedureDecision,
    ProcedureDefinition,
    ProcedureRuntimeState,
)
from astra.contracts.system import EventTopic
from astra.events.bus import EventBus
from astra.procedure.graph import ProcedureGraph
from astra.procedure.state import ProcedureStateManager
from astra.procedure.transition import TransitionEvaluator
from astra.procedure.validator import ProcedureValidator
from astra.violation.detector import ViolationDetector

logger = logging.getLogger("astra.procedure.engine")


class ProcedureEngine:
    """
    Deterministic Procedure Engine.
    Evaluates ConfirmedAction observations against formal procedure graphs.
    Zero neural networks or LLMs are used to validate transitions or safety.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        violation_detector: ViolationDetector | None = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.violation_detector = violation_detector or ViolationDetector()
        self.graph: ProcedureGraph | None = None
        self.state_manager: ProcedureStateManager | None = None
        self._current_run_id: str | None = None

    def load_procedure(self, definition: ProcedureDefinition) -> None:
        """Validate and compile procedure definition into an executable graph."""
        ProcedureValidator.validate(definition)
        self.graph = ProcedureGraph(definition)

    def start(self, run_id: str, procedure: ProcedureDefinition | None = None) -> None:
        """
        Start tracking a new experiment run.
        """
        if procedure is not None:
            self.load_procedure(procedure)

        if self.graph is None:
            raise RuntimeError("Cannot start ProcedureEngine without a loaded ProcedureDefinition.")

        self._current_run_id = run_id
        self.state_manager = ProcedureStateManager(run_id=run_id, graph=self.graph)
        self.state_manager.initialize()
        self.violation_detector.reset()

        logger.info(f"ProcedureEngine started for run '{run_id}', experiment '{self.graph.definition.experiment_id}'.")

        self.event_bus.publish(
            EventTopic.PROCEDURE_STARTED,
            {
                "run_id": run_id,
                "experiment_id": self.graph.definition.experiment_id,
                "procedure_id": self.graph.definition.id,
                "timestamp": time.time(),
            },
        )

    def process(self, action: ConfirmedAction) -> ProcedureDecision:
        """
        Process a confirmed action against current state machine.
        Deterministic transition evaluation: T(S_t, A_t) -> S_{t+1}.
        """
        if self.graph is None or self.state_manager is None:
            raise RuntimeError("ProcedureEngine is not started. Call start() before processing actions.")

        curr_state = self.state_manager.get_snapshot()
        curr_step_id = curr_state.current_step_id

        # Evaluate transition
        evaluation = TransitionEvaluator.evaluate(
            current_step_id=curr_step_id,
            action=action,
            graph=self.graph,
        )

        observed_dict = {
            "action": action.action,
            "object_id": action.object_id,
            "target_id": action.target_id,
            "confidence": action.confidence,
            "actor_id": action.actor_id,
        }

        # 1. Handled as valid transition
        if evaluation.decision_type == DecisionType.VALID and evaluation.matched_step:
            next_step = evaluation.matched_step
            prev_id = curr_step_id
            self.state_manager.advance(
                next_step=next_step,
                action_desc=f"{action.action}:{action.object_id or ''}->{action.target_id or ''}",
            )
            self.violation_detector.reset()

            decision = ProcedureDecision(
                message_id=f"dec-{default_uuid()[:8]}",
                timestamp=time.time(),
                source="procedure-engine",
                correlation_id=action.correlation_id,
                accepted=True,
                current_state=prev_id,
                next_state=next_step.id,
                observed_action=observed_dict,
                decision=DecisionType.VALID,
                violation=None,
            )

            # Publish transition event
            self.event_bus.publish(
                EventTopic.PROCEDURE_TRANSITIONED,
                {
                    "run_id": self._current_run_id,
                    "previous_step": prev_id,
                    "new_step": next_step.id,
                    "decision": decision.model_dump(),
                },
            )

            # If reached terminal step
            if self.graph.is_terminal_step(next_step.id):
                logger.info(f"Procedure reached terminal step '{next_step.id}'. Run complete.")
                self.event_bus.publish(
                    EventTopic.PROCEDURE_COMPLETED,
                    {
                        "run_id": self._current_run_id,
                        "terminal_step": next_step.id,
                        "timestamp": time.time(),
                    },
                )

            return decision

        # 2. Ignored action (e.g. IDLE)
        if evaluation.decision_type == DecisionType.IGNORED:
            return ProcedureDecision(
                message_id=f"dec-{default_uuid()[:8]}",
                timestamp=time.time(),
                source="procedure-engine",
                correlation_id=action.correlation_id,
                accepted=False,
                current_state=curr_step_id,
                next_state=curr_step_id,
                observed_action=observed_dict,
                decision=DecisionType.IGNORED,
                violation=None,
            )

        # 3. Invalid transition -> invoke ViolationDetector
        self.state_manager.record_violation()
        violation = self.violation_detector.evaluate(
            current_step_id=curr_step_id,
            observed_action=action,
            graph=self.graph,
            completed_step_ids=curr_state.completed_steps,
        )

        decision = ProcedureDecision(
            message_id=f"dec-{default_uuid()[:8]}",
            timestamp=time.time(),
            source="procedure-engine",
            correlation_id=action.correlation_id,
            accepted=False,
            current_state=curr_step_id,
            next_state=curr_step_id,
            observed_action=observed_dict,
            decision=DecisionType.INVALID,
            violation=violation,
        )

        # Enforce alert suppression before publishing violation
        if violation is not None:
            if not self.violation_detector.should_suppress_alert(violation):
                self.event_bus.publish(
                    EventTopic.VIOLATION_DETECTED,
                    violation,
                )

        return decision

    @property
    def state(self) -> ProcedureRuntimeState:
        """Get current procedure state snapshot."""
        if self.state_manager is None:
            raise RuntimeError("ProcedureEngine not initialized.")
        return self.state_manager.get_snapshot()

    @property
    def is_completed(self) -> bool:
        """Check if active run has reached completion."""
        if self.state_manager is None:
            return False
        return self.state_manager.status == RunStatus.COMPLETED
