"""ASTRA Reasoner: Procedural State Machine, Temporal Alignment & Deviation Engine.

Key Architectural Principle:
  "The video model observes; the experiment engine reasons."
Does NOT allow the video transformer to decide procedural correctness.
Performs deterministic state transitions, tracks procedure alignment,
evaluates procedural violations against VIOLATION_VOCAB, and executes
the Alert Policy Engine (cooldown, thresholding, guidance mode).
"""

from __future__ import annotations

import logging
import time
from typing import Any
import yaml

from astra.prototype.schemas import (
    AssistanceMode,
    ExperimentRuntimeState,
    ProceduralAlignmentStatus,
    SeverityLevel,
    StepProgress,
    StructuredObservation,
    TemporalEvent,
    VoiceInstruction,
)

logger = logging.getLogger("astra.prototype.alignment")

# Procedural violations vocabulary
VIOLATION_VOCAB = [
    "NONE",
    "WRONG_OBJECT",
    "WRONG_TARGET",
    "SKIPPED_STEP",
    "REPEATED_STEP",
    "PREMATURE_CLOSE",
    "OUT_OF_SEQUENCE",
    "AMBIGUOUS",
]


class ExperimentStep:
    """Internal representation of a formal experiment step."""

    def __init__(
        self,
        id: str,
        step_number: int,
        action: str,
        object: str | None = None,
        target: str | None = None,
        description: str = "",
        allowed_next: list[str] | None = None,
        optional: bool = False,
    ) -> None:
        self.id = id
        self.step_number = step_number
        self.action = action.upper()
        self.object = object.upper() if object else None
        self.target = target.upper() if target else None
        self.description = description
        self.allowed_next = allowed_next or []
        self.optional = optional


class AlertPolicyEngine:
    """
    Alert Policy Engine (Section 10).
    Acoustic fatigue mitigation:
      - Minimum confidence gate
      - 5.0 second cooldown per distinct alert message
      - Suppresses repetitive speech utterances
    """

    def __init__(self, cooldown_seconds: float = 5.0, min_confidence: float = 0.75) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self._last_alert_time: dict[str, float] = {}

    def should_emit_alert(self, instruction: VoiceInstruction, confidence: float) -> bool:
        """Determines whether voice instruction should be forwarded to TTS server."""
        if confidence < self.min_confidence:
            logger.info(f"[ALERT POLICY] Suppressed alert due to low confidence ({confidence:.2f} < {self.min_confidence})")
            return False

        # Routine guidance in Guidance mode has normal low cooldown
        norm_key = instruction.message.strip().lower()
        now = time.time()
        last_time = self._last_alert_time.get(norm_key, 0.0)

        if (now - last_time) < self.cooldown_seconds:
            logger.info(f"[ALERT POLICY COOLDOWN] Suppressed duplicate alert: '{instruction.message}' ({now - last_time:.1f}s ago)")
            return False

        self._last_alert_time[norm_key] = now
        return True

    def reset(self) -> None:
        """Clear alert history."""
        self._last_alert_time.clear()


class TemporalAlignmentEngine:
    """
    Temporal Alignment and Procedural State Machine (Sections 6, 7, 8, 10, 11).
    Maintains:
      EXPECTED STATE + OBSERVED EVENT STREAM -> TEMPORAL ALIGNMENT -> STATE TRANSITION
    """

    def __init__(
        self,
        procedure_yaml_path: str | None = None,
        mode: AssistanceMode = AssistanceMode.GUIDANCE,
        cooldown_seconds: float = 5.0,
    ) -> None:
        self.mode = mode
        self.alert_policy = AlertPolicyEngine(cooldown_seconds=cooldown_seconds)
        self.steps: list[ExperimentStep] = []
        self.step_map: dict[str, ExperimentStep] = {}
        self.experiment_id: str = "EXP001"
        self.experiment_name: str = "Sample Transfer"

        # Runtime State
        self.current_step_index: int = 0
        self.completed_step_ids: list[str] = []
        self.alignment_status: ProceduralAlignmentStatus = ProceduralAlignmentStatus.ALIGNED
        self.belief_confidence: float = 1.0
        self.latest_observation_text: str = "Experiment initialized. Ready for step 1."
        self.last_deviation_message: str | None = None
        self.active_guidance_text: str | None = None
        self.event_history: list[tuple[float, StructuredObservation, bool]] = []

        if procedure_yaml_path:
            self.load_procedure_yaml(procedure_yaml_path)
        else:
            self._load_default_procedure()

    def _load_default_procedure(self) -> None:
        """Loads default 5-step Sample Transfer experiment."""
        default_steps = [
            ExperimentStep(
                id="S01",
                step_number=1,
                action="PICK",
                object="SAMPLE_CONTAINER",
                target=None,
                description="Pick up the sample container",
                allowed_next=["S02"],
            ),
            ExperimentStep(
                id="S02",
                step_number=2,
                action="MOVE",
                object="SAMPLE_CONTAINER",
                target="CHAMBER",
                description="Move the sample container toward the chamber",
                allowed_next=["S03"],
            ),
            ExperimentStep(
                id="S03",
                step_number=3,
                action="OPEN",
                object="CHAMBER",
                target=None,
                description="Open the experiment chamber door",
                allowed_next=["S04"],
            ),
            ExperimentStep(
                id="S04",
                step_number=4,
                action="PLACE",
                object="SAMPLE_CONTAINER",
                target="CHAMBER",
                description="Place the sample container inside the chamber",
                allowed_next=["S05"],
            ),
            ExperimentStep(
                id="S05",
                step_number=5,
                action="CLOSE",
                object="CHAMBER",
                target=None,
                description="Close the chamber door",
                allowed_next=[],
            ),
        ]
        self._set_steps("EXP001", "Sample Transfer & Chamber Enclosure", default_steps)

    def load_procedure_yaml(self, yaml_path: str) -> None:
        """Load and parse procedure from a YAML specification file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        proc = data.get("procedure", data)
        exp_id = proc.get("experiment_id", "EXP001")
        exp_name = proc.get("name", "Sample Transfer")
        steps_raw = proc.get("steps", [])

        compiled_steps: list[ExperimentStep] = []
        for idx, s in enumerate(steps_raw, start=1):
            step = ExperimentStep(
                id=str(s.get("id", f"S{idx:02d}")),
                step_number=idx,
                action=s.get("action", ""),
                object=s.get("object"),
                target=s.get("target"),
                description=s.get("description", ""),
                allowed_next=s.get("allowed_next", []),
                optional=s.get("optional", False),
            )
            compiled_steps.append(step)

        self._set_steps(exp_id, exp_name, compiled_steps)
        logger.info(f"Loaded procedure '{exp_name}' with {len(compiled_steps)} steps.")

    def _set_steps(self, exp_id: str, exp_name: str, steps: list[ExperimentStep]) -> None:
        self.experiment_id = exp_id
        self.experiment_name = exp_name
        self.steps = steps
        self.step_map = {s.id: s for s in steps}
        self.reset()

    def reset(self) -> None:
        """Reset state machine to initial starting state."""
        self.current_step_index = 0
        self.completed_step_ids = []
        self.alignment_status = ProceduralAlignmentStatus.ALIGNED
        self.belief_confidence = 1.0
        self.latest_observation_text = "Experiment initialized. Ready for step 1."
        self.last_deviation_message = None
        self.active_guidance_text = self._get_initial_guidance()
        self.alert_policy.reset()
        self.event_history.clear()

    def _get_initial_guidance(self) -> str:
        if self.steps:
            return f"Please proceed with step 1: {self.steps[0].description}."
        return "Ready to begin experiment."

    @property
    def current_step(self) -> ExperimentStep | None:
        """Returns the currently active expected step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def is_completed(self) -> bool:
        """Check if all procedure steps are completed."""
        return self.current_step_index >= len(self.steps)

    def process_observation(
        self, observation: StructuredObservation
    ) -> tuple[ProceduralAlignmentStatus, VoiceInstruction | None]:
        """
        Main reasoning cycle:
        Given an incoming structured observation from Colab Server #1:
        1. Evaluate events against current expected step and graph
        2. Perform state transition or flag deviation
        3. Consult Alert Policy Engine
        4. Return updated status and optional VoiceInstruction
        """
        self.latest_observation_text = observation.observation
        self.belief_confidence = observation.confidence

        if self.is_completed:
            self.alignment_status = ProceduralAlignmentStatus.COMPLETED
            return self.alignment_status, None

        expected = self.current_step
        if expected is None:
            return self.alignment_status, None

        # Filter meaningful non-idle events
        meaningful_events = [e for e in observation.events if e.action.upper() != "IDLE"]
        if not meaningful_events:
            # Idle observation; remain aligned and quiet
            return self.alignment_status, None

        voice_instruction: VoiceInstruction | None = None
        matched_transition = False

        for event in meaningful_events:
            match_result, violation_type, deviation_info = self._evaluate_event(event, expected)

            if match_result:
                # Aligned step execution!
                matched_transition = True
                completed_step = expected
                self.completed_step_ids.append(completed_step.id)
                self.current_step_index += 1
                self.alignment_status = ProceduralAlignmentStatus.ALIGNED
                self.last_deviation_message = None

                logger.info(
                    f"[STATE ADVANCE] Step {completed_step.step_number}/{len(self.steps)} "
                    f"'{completed_step.id}: {completed_step.action}' COMPLETED."
                )

                # Check if experiment completed
                if self.is_completed:
                    self.alignment_status = ProceduralAlignmentStatus.COMPLETED
                    self.active_guidance_text = "Experiment successfully completed. All steps verified."
                    voice_instruction = VoiceInstruction(
                        severity=SeverityLevel.INFO,
                        message="Experiment complete.",
                        guidance="All protocol steps have been successfully executed and verified.",
                        interrupt=True,
                    )
                elif self.mode == AssistanceMode.GUIDANCE:
                    # Proactive next-step guidance
                    next_step = self.current_step
                    step_announcement = f"Step {completed_step.step_number} complete."
                    next_prompt = f"Next, please {next_step.description.lower()}." if next_step else ""
                    self.active_guidance_text = f"{step_announcement} {next_prompt}"
                    voice_instruction = VoiceInstruction(
                        severity=SeverityLevel.INFO,
                        message=step_announcement,
                        guidance=next_prompt,
                        interrupt=False,
                    )
                break  # Advance step once per window

            else:
                # Procedural mismatch detected!
                severity, message, guidance = deviation_info
                self.alignment_status = (
                    ProceduralAlignmentStatus.CRITICAL
                    if severity == SeverityLevel.CRITICAL
                    else ProceduralAlignmentStatus.DEVIATION
                )
                self.last_deviation_message = message
                self.active_guidance_text = guidance

                logger.warning(f"[DEVIATION DETECTED] {violation_type} ({severity.value}): {message}")

                candidate_instruction = VoiceInstruction(
                    severity=severity,
                    message=message,
                    guidance=guidance,
                    interrupt=(severity == SeverityLevel.CRITICAL),
                )

                # Test alert policy cooldown and threshold
                if self.alert_policy.should_emit_alert(candidate_instruction, observation.confidence):
                    voice_instruction = candidate_instruction
                break

        self.event_history.append((time.time(), observation, matched_transition))
        return self.alignment_status, voice_instruction

    def _evaluate_event(
        self, event: TemporalEvent, expected: ExperimentStep
    ) -> tuple[bool, str, tuple[SeverityLevel, str, str]]:
        """
        Compares an observed event against expected step and classifies deviations.
        Returns:
          (is_match, violation_type, (SeverityLevel, diagnostic_message, corrective_guidance))
        """
        ev_action = event.action.upper()
        ev_obj = event.object.upper() if event.object else None
        ev_tgt = event.target.upper() if event.target else None

        exp_action = expected.action
        exp_obj = expected.object
        exp_tgt = expected.target

        # 1. Exact match check
        action_match = (ev_action == exp_action)
        obj_match = (exp_obj is None or ev_obj == exp_obj or ev_obj is None)
        tgt_match = (exp_tgt is None or ev_tgt == exp_tgt or ev_tgt is None)

        if action_match and obj_match and tgt_match:
            return True, "NONE", (SeverityLevel.INFO, "Step aligned.", "")

        # 2. Critical Check: WRONG_OBJECT (interacted with unauthorized item)
        if ev_obj and ev_obj in ("WRONG_CONTAINER", "SECONDARY_CONTAINER", "UNAUTHORIZED_OBJECT"):
            return (
                False,
                "WRONG_OBJECT",
                (
                    SeverityLevel.CRITICAL,
                    "Warning. The operator appears to be interacting with the wrong component.",
                    f"Please put down the secondary container and interact only with the {exp_obj or 'sample container'}.",
                ),
            )

        if ev_obj and exp_obj and ev_obj != exp_obj and ev_obj != "NONE":
            return (
                False,
                "WRONG_OBJECT",
                (
                    SeverityLevel.CRITICAL,
                    f"Expected interaction with {exp_obj}, but {ev_obj} was detected.",
                    f"Please switch to the correct {exp_obj}.",
                ),
            )

        # 3. Deviation Check: SKIPPED_STEP (e.g. attempting to PLACE before OPENing chamber)
        if exp_action == "OPEN" and ev_action == "PLACE":
            return (
                False,
                "SKIPPED_STEP",
                (
                    SeverityLevel.DEVIATION,
                    "The expected chamber opening step has not been detected.",
                    "Please open the chamber door before attempting to place the sample.",
                ),
            )

        # 4. Deviation Check: PREMATURE_CLOSE (closing before placement)
        if ev_action == "CLOSE" and "S04" not in self.completed_step_ids:
            return (
                False,
                "PREMATURE_CLOSE",
                (
                    SeverityLevel.DEVIATION,
                    "Premature chamber closure detected.",
                    "The sample has not been placed inside yet. Please open the chamber and place the sample.",
                ),
            )

        # 5. Deviation Check: REPEATED_STEP (repeating an already completed step)
        for completed_id in self.completed_step_ids:
            comp_step = self.step_map.get(completed_id)
            if comp_step and comp_step.action == ev_action and comp_step.object == ev_obj:
                return (
                    False,
                    "REPEATED_STEP",
                    (
                        SeverityLevel.WARNING,
                        f"Step {comp_step.step_number} ({comp_step.action}) was already executed.",
                        f"Please proceed forward to step {expected.step_number}: {expected.description}.",
                    ),
                )

        # 6. Deviation Check: OUT_OF_SEQUENCE (executing a future step ahead of time)
        for future_step in self.steps[self.current_step_index + 1 :]:
            if future_step.action == ev_action:
                return (
                    False,
                    "OUT_OF_SEQUENCE",
                    (
                        SeverityLevel.DEVIATION,
                        f"Action {ev_action} was detected out of sequence.",
                        f"Expected step {expected.step_number} is: {expected.description}.",
                    ),
                )

        # 7. Fallback Generic Deviation
        return (
            False,
            "OUT_OF_SEQUENCE",
            (
                SeverityLevel.WARNING,
                f"Observed action '{ev_action}' does not match expected step '{exp_action}'.",
                f"Please verify procedure step {expected.step_number}: {expected.description}.",
            ),
        )

    def get_runtime_state(self) -> ExperimentRuntimeState:
        """Constructs an immutable snapshot of current engine state for UI and dialogue."""
        step_progress_list: list[StepProgress] = []
        for s in self.steps:
            if s.id in self.completed_step_ids:
                status = "completed"
            elif self.current_step and s.id == self.current_step.id:
                status = "deviation" if self.alignment_status in (ProceduralAlignmentStatus.DEVIATION, ProceduralAlignmentStatus.CRITICAL) else "active"
            else:
                status = "pending"

            step_progress_list.append(
                StepProgress(
                    id=s.id,
                    step_number=s.step_number,
                    action=s.action,
                    object=s.object,
                    target=s.target,
                    description=s.description,
                    status=status,
                )
            )

        curr_id = self.current_step.id if self.current_step else None
        curr_num = self.current_step.step_number if self.current_step else len(self.steps)

        return ExperimentRuntimeState(
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            current_step_id=curr_id,
            current_step_number=curr_num,
            total_steps=len(self.steps),
            alignment_status=self.alignment_status,
            confidence=self.belief_confidence,
            latest_observation=self.latest_observation_text,
            completed_steps=list(self.completed_step_ids),
            steps=step_progress_list,
            last_deviation_message=self.last_deviation_message,
            active_guidance_text=self.active_guidance_text,
            mode=self.mode,
            timestamp=time.time(),
        )
