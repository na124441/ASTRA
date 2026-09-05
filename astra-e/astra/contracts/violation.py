"""Procedural violation contracts."""

from __future__ import annotations

from typing import Any
from pydantic import Field
from astra.contracts.base import BaseMessage, Severity, ViolationType, current_timestamp


class ViolationEvent(BaseMessage):
    """
    Contract 08: ViolationEvent.
    Created exclusively by the procedural reasoning/violation engine when an action
    deviates from the experiment procedure graph.
    """
    violation_type: ViolationType = Field(description="Standard category of the deviation")
    expected: dict[str, Any] = Field(description="Details of what the procedure expected")
    observed: dict[str, Any] = Field(description="Details of what was actually observed")
    severity: Severity = Field(default=Severity.WARNING, description="Severity: INFO, WARNING, CRITICAL")
    message: str = Field(description="Human-readable explanation of the deviation")
    event_time: float = Field(default_factory=current_timestamp, description="Timestamp of the violating action")
