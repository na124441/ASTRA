"""ActionConfirmationEngine: Temporal state machine with EMA smoothing, hysteresis, and abstention."""

from __future__ import annotations

import time
from typing import Any
from astra.contracts.activity import ActionObservation, ConfirmationMetadata, ConfirmedAction
from astra.contracts.base import default_uuid


class ActionConfirmationEngine:
    """
    Temporal Confirmation Layer (FR-009, NFR-003, and pages 255-256 of SIH.pdf).
    Transitions:
      OBSERVED -> CANDIDATE -> CONFIRMED -> REVOKED/TRANSITION
    Applies:
      1. EMA probability smoothing: S_t = alpha * S_{t-1} + (1 - alpha) * P(a_t)
      2. Multi-frame persistence: requires K >= min_support_frames consecutive support
      3. Action schema compatibility validation
      4. Physical plausibility check
      5. Explicit ABSTENTION under uncertainty (confidence < tau_abstain)
    """

    def __init__(
        self,
        ema_alpha: float = 0.65,
        confirmation_threshold: float = 0.70,
        abstain_threshold: float = 0.50,
        min_support_frames: int = 4,
        max_contradictions: int = 2,
    ) -> None:
        self.alpha = ema_alpha
        self.confirmation_thresh = confirmation_threshold
        self.abstain_thresh = abstain_threshold
        self.min_support = min_support_frames
        self.max_contradictions = max_contradictions

        # Internal state
        self._candidate_action: str | None = None
        self._candidate_object: str | None = None
        self._candidate_target: str | None = None
        self._smoothed_score: float = 0.0
        self._support_count: int = 0
        self._contradiction_count: int = 0
        self._last_confirmed_action: str | None = None
        self._min_confidence_in_window: float = 1.0

        # Physical transition plausibility rules
        self._physical_transitions = {
            "IDLE": {"APPROACH", "OPEN_CONTAINER", "IDLE"},
            "OPEN_CONTAINER": {"APPROACH", "PICK", "IDLE"},
            "APPROACH": {"TOUCH", "GRASP", "PICK", "APPROACH", "IDLE"},
            "TOUCH": {"GRASP", "PICK", "APPROACH", "IDLE"},
            "GRASP": {"MOVE", "PICK", "RELEASE", "IDLE"},
            "PICK": {"MOVE", "PLACE", "IDLE"},
            "MOVE": {"PLACE", "RELEASE", "MOVE", "IDLE"},
            "PLACE": {"RELEASE", "CLOSE_CONTAINER", "IDLE"},
            "RELEASE": {"APPROACH", "CLOSE_CONTAINER", "IDLE"},
            "CLOSE_CONTAINER": {"IDLE"},
        }

    def reset(self) -> None:
        """Reset confirmation tracking state."""
        self._candidate_action = None
        self._candidate_object = None
        self._candidate_target = None
        self._smoothed_score = 0.0
        self._support_count = 0
        self._contradiction_count = 0
        self._last_confirmed_action = None
        self._min_confidence_in_window = 1.0

    def process_observation(self, obs: ActionObservation) -> ConfirmedAction | None:
        """
        Process single-frame ActionObservation and return ConfirmedAction when stable.
        Returns None if evidence is accumulating, contradicted, or abstained.
        """
        # 1. Abstention check
        if obs.confidence < self.abstain_thresh or obs.action == "UNKNOWN":
            self._contradiction_count += 1
            if self._contradiction_count > self.max_contradictions:
                self._support_count = 0
            return None

        # 2. Action Schema Compatibility Check
        is_compatible, clean_obj, clean_tgt = self._validate_compatibility(
            obs.action, obs.object_id, obs.target_id
        )
        if not is_compatible:
            return None

        # 3. Check Physical Transition Plausibility
        if not self._is_physically_plausible(obs.action):
            return None

        current_key = f"{obs.action}:{clean_obj}:{clean_tgt}"
        candidate_key = (
            f"{self._candidate_action}:{self._candidate_object}:{self._candidate_target}"
            if self._candidate_action else None
        )

        if self._candidate_action is None:
            # First valid observation initializes candidate
            self._candidate_action = obs.action
            self._candidate_object = clean_obj
            self._candidate_target = clean_tgt
            self._smoothed_score = obs.confidence
            self._support_count = 1
            self._contradiction_count = 0
            self._min_confidence_in_window = obs.confidence
        elif candidate_key == current_key:
            # Accumulate evidence with EMA
            self._smoothed_score = (self.alpha * self._smoothed_score) + ((1.0 - self.alpha) * obs.confidence)
            self._support_count += 1
            self._contradiction_count = 0
            self._min_confidence_in_window = min(self._min_confidence_in_window, obs.confidence)
        else:
            # Different action observed
            self._contradiction_count += 1
            if self._contradiction_count > self.max_contradictions:
                # Switch candidate
                self._candidate_action = obs.action
                self._candidate_object = clean_obj
                self._candidate_target = clean_tgt
                self._smoothed_score = obs.confidence
                self._support_count = 1
                self._contradiction_count = 0
                self._min_confidence_in_window = obs.confidence

        # 4. Confirmation Gate
        if (
            self._support_count >= self.min_support
            and self._smoothed_score >= self.confirmation_thresh
        ):
            action_signature = f"{self._candidate_action}:{self._candidate_object}:{self._candidate_target}"
            if self._last_confirmed_action != action_signature:
                self._last_confirmed_action = action_signature

                confirmed = ConfirmedAction(
                    message_id=f"conf-{default_uuid()[:8]}",
                    schema_version="1.0",
                    timestamp=time.time(),
                    source="confirmation-engine",
                    correlation_id=obs.correlation_id,
                    action=self._candidate_action,
                    actor_id=obs.actor_id,
                    object_id=self._candidate_object,
                    target_id=self._candidate_target,
                    confidence=round(self._smoothed_score, 4),
                    event_time=obs.event_time,
                    confirmation=ConfirmationMetadata(
                        stable_frames=self._support_count,
                        minimum_confidence=round(self._min_confidence_in_window, 4),
                        temporal_consistency=round(self._smoothed_score, 4),
                    ),
                )
                return confirmed

        return None

    def _validate_compatibility(
        self,
        verb: str,
        obj: str | None,
        tgt: str | None,
    ) -> tuple[bool, str | None, str | None]:
        """Verify semantic validity of multi-head verb/object/target combination."""
        clean_obj = obj if obj and obj != "NONE" else None
        clean_tgt = tgt if tgt and tgt != "NONE" else None

        if verb in ("IDLE", "UNKNOWN"):
            return True, None, None

        if verb == "OPEN_CONTAINER" or verb == "CLOSE_CONTAINER":
            return True, "CONTAINER", None

        if verb == "PICK":
            # Pick operates on an object, target should be None
            return True, clean_obj, None

        if verb == "PLACE":
            # Place requires both an object and a target
            return True, clean_obj, clean_tgt

        return True, clean_obj, clean_tgt

    def _is_physically_plausible(self, next_action: str) -> bool:
        """Verify that transition from previous action is physically possible."""
        if self._last_confirmed_action is None:
            return True

        last_verb = self._last_confirmed_action.split(":")[0]
        if next_action == last_verb:
            return True
        if last_verb not in self._physical_transitions:
            return True

        allowed = self._physical_transitions[last_verb]
        return next_action in allowed
