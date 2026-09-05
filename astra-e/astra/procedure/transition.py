"""Transition evaluation logic."""

from __future__ import annotations

from dataclasses import dataclass
from astra.contracts.activity import ConfirmedAction
from astra.contracts.base import DecisionType
from astra.contracts.procedure import ProcedureStep
from astra.procedure.graph import ProcedureGraph


@dataclass(frozen=True)
class TransitionEvaluation:
    """Internal result of evaluating an action against the procedure graph."""
    decision_type: DecisionType
    matched_step: ProcedureStep | None
    expected_steps: list[ProcedureStep]
    reason: str


class TransitionEvaluator:
    """Evaluates whether a ConfirmedAction represents a valid graph transition."""

    @staticmethod
    def evaluate(
        current_step_id: str | None,
        action: ConfirmedAction,
        graph: ProcedureGraph,
    ) -> TransitionEvaluation:
        """
        Evaluate ConfirmedAction against current procedure position.
        """
        allowed_next = graph.get_allowed_next_steps(current_step_id)

        # Ignore pure IDLE or no-op actions
        if action.action.upper() == "IDLE":
            return TransitionEvaluation(
                decision_type=DecisionType.IGNORED,
                matched_step=None,
                expected_steps=allowed_next,
                reason="IDLE action ignored during procedure evaluation.",
            )

        # Check for immediate valid match
        matched = graph.find_matching_next_step(
            current_step_id=current_step_id,
            action=action.action,
            object_id=action.object_id,
            target_id=action.target_id,
        )

        if matched:
            return TransitionEvaluation(
                decision_type=DecisionType.VALID,
                matched_step=matched,
                expected_steps=allowed_next,
                reason=f"Action '{action.action}' matches valid transition to step '{matched.id}'.",
            )

        # Transition is invalid
        return TransitionEvaluation(
            decision_type=DecisionType.INVALID,
            matched_step=None,
            expected_steps=allowed_next,
            reason=f"Action '{action.action}' is not a valid transition from current step '{current_step_id}'.",
        )
