"""AssistanceManager for transforming system state and deviations into astronaut guidance."""

from __future__ import annotations

import logging
import time
from typing import Any
from astra.contracts.assistance import AssistanceChannel, AssistanceEvent, AssistancePriority
from astra.contracts.base import default_uuid
from astra.contracts.procedure import ProcedureDefinition, ProcedureStep
from astra.contracts.system import EventTopic
from astra.contracts.violation import ViolationEvent
from astra.events.bus import EventBus

logger = logging.getLogger("astra.assistance.manager")


class AssistanceManager:
    """
    Assistance Subsystem Manager (FR-018, FR-020, FR-021).
    Translates machine events into deterministic, clear astronaut guidance and alerts.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.history: list[AssistanceEvent] = []
        self._procedure_def: ProcedureDefinition | None = None
        self._subscribe()

    def _subscribe(self) -> None:
        """Register listeners on event bus."""
        self.event_bus.subscribe(EventTopic.VIOLATION_DETECTED, self._on_violation)
        self.event_bus.subscribe(EventTopic.PROCEDURE_TRANSITIONED, self._on_transition)
        self.event_bus.subscribe(EventTopic.PROCEDURE_COMPLETED, self._on_complete)

    def set_procedure(self, procedure_def: ProcedureDefinition) -> None:
        """Cache current procedure definition for context resolution."""
        self._procedure_def = procedure_def

    def _on_violation(self, violation: ViolationEvent) -> None:
        """Generate high-priority corrective warning when a violation is reported."""
        expected = violation.expected
        expected_desc = expected.get("description") or f"{expected.get('action')} {expected.get('object') or ''}".strip()
        expected_target = expected.get("target")

        # Formulate concise astronaut alert
        msg = f"Warning: {violation.message}"
        if expected_target:
            msg += f" Please use target {expected_target}."
        elif expected_desc:
            msg += f" Please perform: {expected_desc}."

        assist_event = AssistanceEvent(
            message_id=f"assist-{default_uuid()[:8]}",
            timestamp=time.time(),
            source="assistance-manager",
            correlation_id=violation.correlation_id,
            type="PROCEDURE_WARNING",
            priority=AssistancePriority.HIGH,
            message=msg,
            channels=[AssistanceChannel.GUI, AssistanceChannel.TTS],
            target_step_id=expected.get("step_id"),
            event_time=violation.event_time,
        )

        self.history.append(assist_event)
        self.event_bus.publish(EventTopic.ASSISTANCE_ISSUED, assist_event)
        logger.warning(f"[ASSISTANT ALERT] {assist_event.message}")

    def _on_transition(self, data: dict[str, Any]) -> None:
        """Generate guidance for the upcoming step following a successful transition."""
        pass  # Guidance can be requested or emitted when next step is known

    def _on_complete(self, data: dict[str, Any]) -> None:
        """Notify astronaut of successful procedure completion."""
        run_id = data.get("run_id", "RUN")
        assist_event = AssistanceEvent(
            message_id=f"assist-{default_uuid()[:8]}",
            timestamp=time.time(),
            source="assistance-manager",
            correlation_id=run_id,
            type="PROCEDURE_COMPLETE",
            priority=AssistancePriority.MEDIUM,
            message="Experiment procedure completed successfully.",
            channels=[AssistanceChannel.GUI, AssistanceChannel.TTS],
            target_step_id=None,
            event_time=time.time(),
        )
        self.history.append(assist_event)
        self.event_bus.publish(EventTopic.ASSISTANCE_ISSUED, assist_event)
        logger.info(f"[ASSISTANT COMPLETE] {assist_event.message}")

    def provide_guidance_for_step(self, step: ProcedureStep, correlation_id: str) -> AssistanceEvent:
        """Produce proactive next-step instruction."""
        desc = step.description or f"{step.action} {step.object or ''} {step.target or ''}".strip()
        msg = f"Next step: {desc}"

        event = AssistanceEvent(
            message_id=f"assist-{default_uuid()[:8]}",
            timestamp=time.time(),
            source="assistance-manager",
            correlation_id=correlation_id,
            type="PROCEDURE_GUIDANCE",
            priority=AssistancePriority.LOW,
            message=msg,
            channels=[AssistanceChannel.GUI, AssistanceChannel.TTS],
            target_step_id=step.id,
            event_time=time.time(),
        )
        self.history.append(event)
        self.event_bus.publish(EventTopic.ASSISTANCE_ISSUED, event)
        return event
