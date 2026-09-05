"""Deterministic ViolationDetector for procedural deviations and alert suppression."""

from __future__ import annotations

import time
from typing import Any
from astra.contracts.activity import ConfirmedAction
from astra.contracts.base import ViolationType, default_uuid
from astra.contracts.procedure import ProcedureStep
from astra.contracts.violation import ViolationEvent
from astra.procedure.graph import ProcedureGraph
from astra.violation.classifier import ViolationClassifier


class ViolationDetector:
    """
    Deterministic evaluation engine for classifying procedural deviations.
    Enforces FR-013, FR-014, FR-015, FR-016, and FR-022 (Alert Suppression).
    Does NOT use AI/LLMs to judge correctness.
    """

    def __init__(self, suppression_cooldown_seconds: float = 3.0) -> None:
        self.cooldown = suppression_cooldown_seconds
        # Maps violation_signature -> last_alert_time
        self._suppressed_alerts: dict[str, float] = {}
        self.history: list[ViolationEvent] = []

    def reset(self) -> None:
        """Clear alert suppression states when a procedure transitions legitimately."""
        self._suppressed_alerts.clear()
        self.history.clear()

    def evaluate(
        self,
        current_step_id: str | None,
        observed_action: ConfirmedAction,
        graph: ProcedureGraph,
        completed_step_ids: list[str] | None = None,
    ) -> ViolationEvent | None:
        """
        Evaluate an invalid action and return a structured ViolationEvent.
        """
        completed = set(completed_step_ids or [])
        allowed_next = graph.get_allowed_next_steps(current_step_id)

        # Expected step details for contract
        expected_dict: dict[str, Any] = {}
        if allowed_next:
            primary_expected = allowed_next[0]
            expected_dict = {
                "step_id": primary_expected.id,
                "action": primary_expected.action,
                "object": primary_expected.object,
                "target": primary_expected.target,
                "description": primary_expected.description,
            }
        else:
            expected_dict = {"status": "EXPERIMENT_COMPLETE", "step_id": None}

        observed_dict: dict[str, Any] = {
            "action": observed_action.action,
            "object": observed_action.object_id,
            "target": observed_action.target_id,
            "confidence": observed_action.confidence,
        }

        # 1. Check for WRONG_OBJECT
        for exp_step in allowed_next:
            if exp_step.action.upper() == observed_action.action.upper():
                if exp_step.object and observed_action.object_id:
                    if exp_step.object.upper() != observed_action.object_id.upper():
                        msg = (
                            f"Wrong object for step '{exp_step.id}': expected '{exp_step.object}', "
                            f"observed '{observed_action.object_id}'."
                        )
                        return self._create_violation(
                            violation_type=ViolationType.WRONG_OBJECT,
                            expected=expected_dict,
                            observed=observed_dict,
                            message=msg,
                            action=observed_action,
                        )

        # 2. Check for WRONG_TARGET
        for exp_step in allowed_next:
            if (
                exp_step.action.upper() == observed_action.action.upper()
                and (not exp_step.object or (observed_action.object_id and exp_step.object.upper() == observed_action.object_id.upper()))
            ):
                if exp_step.target and observed_action.target_id:
                    if exp_step.target.upper() != observed_action.target_id.upper():
                        msg = (
                            f"Wrong target for step '{exp_step.id}': expected '{exp_step.target}', "
                            f"observed '{observed_action.target_id}'."
                        )
                        return self._create_violation(
                            violation_type=ViolationType.WRONG_TARGET,
                            expected=expected_dict,
                            observed=observed_dict,
                            message=msg,
                            action=observed_action,
                        )

        # 3. Check for REPEATED_ACTION
        for comp_id in completed:
            step = graph.get_step(comp_id)
            if step and graph.match_step(
                step,
                observed_action.action,
                observed_action.object_id,
                observed_action.target_id,
            ):
                if not step.repeatable:
                    msg = f"Step '{step.id}' ({step.action}) has already been completed and cannot be repeated."
                    return self._create_violation(
                        violation_type=ViolationType.REPEATED_ACTION,
                        expected=expected_dict,
                        observed=observed_dict,
                        message=msg,
                        action=observed_action,
                    )

        # 4. Check for SKIPPED_STEP
        # Check if the observed action matches a future reachable step in the graph
        matching_future_steps = graph.find_any_matching_steps(
            observed_action.action,
            observed_action.object_id,
            observed_action.target_id,
        )

        for future_step in matching_future_steps:
            if future_step.id not in completed:
                skipped = graph.get_skipped_steps(current_step_id, future_step.id)
                if skipped:
                    skipped_names = [f"{s.id} ({s.action})" for s in skipped]
                    msg = (
                        f"Skipped required step(s) [{', '.join(skipped_names)}] "
                        f"before executing '{future_step.id}'."
                    )
                    expected_dict["skipped_steps"] = [s.id for s in skipped]
                    return self._create_violation(
                        violation_type=ViolationType.SKIPPED_STEP,
                        expected=expected_dict,
                        observed=observed_dict,
                        message=msg,
                        action=observed_action,
                    )

        # 5. Check for OUT_OF_ORDER vs UNAUTHORIZED_ACTION
        all_matches = graph.find_any_matching_steps(
            observed_action.action,
            observed_action.object_id,
            observed_action.target_id,
        )
        if all_matches:
            target_ids = [s.id for s in all_matches]
            msg = (
                f"Action '{observed_action.action}' is a valid experiment action (steps: {target_ids}), "
                f"but is out-of-order at state '{current_step_id}'."
            )
            return self._create_violation(
                violation_type=ViolationType.OUT_OF_ORDER,
                expected=expected_dict,
                observed=observed_dict,
                message=msg,
                action=observed_action,
            )

        # 6. UNAUTHORIZED_ACTION
        msg = f"Action '{observed_action.action}' is not part of this experiment procedure."
        return self._create_violation(
            violation_type=ViolationType.UNAUTHORIZED_ACTION,
            expected=expected_dict,
            observed=observed_dict,
            message=msg,
            action=observed_action,
        )

    def _create_violation(
        self,
        violation_type: ViolationType,
        expected: dict[str, Any],
        observed: dict[str, Any],
        message: str,
        action: ConfirmedAction,
    ) -> ViolationEvent:
        """Helper to build ViolationEvent with classified severity."""
        severity = ViolationClassifier.classify_severity(violation_type)
        event = ViolationEvent(
            message_id=f"viol-{default_uuid()[:8]}",
            timestamp=time.time(),
            source="violation-engine",
            correlation_id=action.correlation_id,
            violation_type=violation_type,
            expected=expected,
            observed=observed,
            severity=severity,
            message=message,
            event_time=action.event_time,
        )
        self.history.append(event)
        return event

    def should_suppress_alert(self, violation: ViolationEvent) -> bool:
        """
        Enforces FR-022 alert suppression.
        Returns True if an identical violation was alerted recently within cooldown window.
        """
        sig = f"{violation.violation_type}:{violation.expected.get('step_id')}:{violation.observed.get('action')}:{violation.observed.get('object')}:{violation.observed.get('target')}"
        now = time.time()
        last_time = self._suppressed_alerts.get(sig)

        if last_time is not None and (now - last_time) < self.cooldown:
            return True

        self._suppressed_alerts[sig] = now
        return False
