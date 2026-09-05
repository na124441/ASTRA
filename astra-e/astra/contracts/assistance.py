"""Assistance subsystem contracts for GUI feedback and TTS voice alerts."""

from __future__ import annotations

from pydantic import Field
from astra.contracts.base import (
    AssistanceChannel,
    AssistancePriority,
    BaseMessage,
    current_timestamp,
)


class AssistanceEvent(BaseMessage):
    """
    Contract 09: AssistanceEvent.
    Translates procedure engine decisions and violations into human-oriented guidance.
    """
    type: str = Field(description="Assistance category: PROCEDURE_GUIDANCE, PROCEDURE_WARNING, PROCEDURE_COMPLETE")
    priority: AssistancePriority = Field(default=AssistancePriority.MEDIUM, description="Alert priority")
    message: str = Field(description="Natural language instruction or warning for the astronaut")
    channels: list[AssistanceChannel] = Field(
        default_factory=lambda: [AssistanceChannel.GUI, AssistanceChannel.TTS],
        description="Delivery channels to notify",
    )
    target_step_id: str | None = Field(default=None, description="Relevant procedure step ID")
    event_time: float = Field(default_factory=current_timestamp, description="Event occurrence time")
