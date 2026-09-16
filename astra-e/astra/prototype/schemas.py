"""Data contracts, schemas, and typed definitions for ASTRA-E Prototype v0.1."""

from __future__ import annotations

from enum import Enum
import time
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ProceduralAlignmentStatus(str, Enum):
    """Overall health / alignment status of the running experiment."""
    ALIGNED = "ALIGNED"
    DEVIATION = "DEVIATION"
    CRITICAL = "CRITICAL"
    COMPLETED = "COMPLETED"


class SeverityLevel(str, Enum):
    """Severity classification for procedural discrepancies."""
    INFO = "INFO"
    WARNING = "WARNING"
    DEVIATION = "DEVIATION"
    CRITICAL = "CRITICAL"


class AssistanceMode(str, Enum):
    """Assistance guidance mode for operator interaction."""
    MONITOR = "MONITOR"      # Stays quiet unless deviation occurs
    GUIDANCE = "GUIDANCE"    # Proactively announces completed steps and next actions


class TemporalEvent(BaseModel):
    """A discrete recognized action event within a temporal window."""
    model_config = ConfigDict(frozen=True)

    action: str = Field(description="Action verb, e.g. PICK, MOVE, OPEN, PLACE, CLOSE, IDLE")
    object: str | None = Field(default=None, description="Interacted object, e.g. SAMPLE_CONTAINER, CHAMBER")
    target: str | None = Field(default=None, description="Target receptacle or destination, e.g. CHAMBER")


class StructuredObservation(BaseModel):
    """
    Structured temporal observation output from Colab Server #1 (Video Understanding).
    Matches the specification:
    {
      "window_start": 12.0,
      "window_end": 16.0,
      "observation": "Operator picks up the sample container and moves it toward the chamber.",
      "events": [{"action": "PICK", "object": "SAMPLE_CONTAINER"}],
      "confidence": 0.87
    }
    """
    model_config = ConfigDict(frozen=True)

    window_start: float = Field(default=0.0, description="Start timestamp of video window in seconds")
    window_end: float = Field(default=0.0, description="End timestamp of video window in seconds")
    observation: str = Field(description="Natural language summary of the observed activity")
    events: list[TemporalEvent] = Field(default_factory=list, description="Extracted structured events")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="Perceptual confidence score")


class VoiceInstruction(BaseModel):
    """
    Structured voice guidance or alert instruction issued by ASTRA Reasoner to Colab Server #2 (TTS).
    Matches the specification:
    {
      "severity": "WARNING",
      "message": "The expected chamber opening step has not been detected.",
      "guidance": "Please open the chamber before placing the sample.",
      "interrupt": true
    }
    """
    model_config = ConfigDict(frozen=True)

    severity: SeverityLevel = Field(default=SeverityLevel.INFO)
    message: str = Field(description="Diagnostic explanation of what happened or what is expected")
    guidance: str = Field(description="Clear, actionable next instruction for the operator")
    interrupt: bool = Field(default=False, description="Whether this alert should preempt ongoing non-critical speech")
    timestamp: float = Field(default_factory=time.time)

    def to_speech_text(self) -> str:
        """Formulates natural conversational spoken text from message and guidance."""
        if self.severity in (SeverityLevel.CRITICAL, SeverityLevel.DEVIATION):
            prefix = "Warning. " if self.severity == SeverityLevel.DEVIATION else "Critical alert. "
            return f"{prefix}{self.message} {self.guidance}"
        if self.severity == SeverityLevel.WARNING:
            return f"Caution. {self.message} {self.guidance}"
        return f"{self.message} {self.guidance}".strip()


class StepProgress(BaseModel):
    """Live state tracking for an individual experiment step in the UI checklist."""
    model_config = ConfigDict(frozen=True)

    id: str
    step_number: int
    action: str
    object: str | None = None
    target: str | None = None
    description: str
    status: str = "pending"  # "completed", "active", "pending", "deviation"


class ExperimentRuntimeState(BaseModel):
    """Complete snapshot of the experiment state for UI and conversational grounding."""
    model_config = ConfigDict(frozen=True)

    experiment_id: str
    experiment_name: str
    current_step_id: str | None
    current_step_number: int
    total_steps: int
    alignment_status: ProceduralAlignmentStatus
    confidence: float
    latest_observation: str
    completed_steps: list[str] = Field(default_factory=list)
    steps: list[StepProgress] = Field(default_factory=list)
    last_deviation_message: str | None = None
    active_guidance_text: str | None = None
    mode: AssistanceMode = AssistanceMode.GUIDANCE
    timestamp: float = Field(default_factory=time.time)


class ConversationalQuery(BaseModel):
    """Operator question submitted to 'Ask ASTRA'."""
    question: str
    timestamp: float = Field(default_factory=time.time)


class ConversationalAnswer(BaseModel):
    """Grounded contextual answer from ASTRA conversational assistant."""
    answer: str
    current_step_id: str | None = None
    alignment_status: ProceduralAlignmentStatus = ProceduralAlignmentStatus.ALIGNED
    suggested_action: str = ""
    timestamp: float = Field(default_factory=time.time)
