"""Interaction subsystem contracts (HOI - Human-Object Interaction)."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from astra.contracts.base import BaseMessage, current_timestamp


class InteractionEvidence(BaseModel):
    """Geometric and spatial evidence backing an interaction hypothesis."""
    model_config = ConfigDict(frozen=True)

    hand_distance: float | None = Field(default=None, description="Euclidean distance between hand and object")
    relative_motion: float | None = Field(default=None, description="Normalized velocity co-movement score")
    details: dict[str, Any] = Field(default_factory=dict)


class InteractionEvent(BaseMessage):
    """
    Contract 03: InteractionEvent.
    Expresses human-object physical relations detected at a single timestep.
    """
    interaction_type: str = Field(description="APPROACH, TOUCH, GRASP, MOVE, PLACE, RELEASE, MANIPULATE")
    actor_id: str = Field(description="Interacting human identifier, e.g. human-01")
    hand_id: str | None = Field(default=None, description="Specific hand involved, e.g. hand-right")
    object_id: str | None = Field(default=None, description="Target object identifier, e.g. obj-red-01")
    target_id: str | None = Field(default=None, description="Destination target zone if relevant")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of interaction classification")
    event_time: float = Field(default_factory=current_timestamp, description="Timestamp of interaction occurrence")
    evidence: InteractionEvidence | None = Field(default=None)
