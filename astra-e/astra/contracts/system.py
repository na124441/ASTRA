"""System events, health monitoring, and telemetry envelope contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from astra.contracts.base import current_timestamp, default_uuid


class EventTopic(str, Enum):
    """Standardized event bus publication topics."""
    ACTION_OBSERVED = "action.observed"
    ACTION_CONFIRMED = "action.confirmed"
    PROCEDURE_STARTED = "procedure.started"
    PROCEDURE_TRANSITIONED = "procedure.transitioned"
    PROCEDURE_COMPLETED = "procedure.completed"
    VIOLATION_DETECTED = "violation.detected"
    ASSISTANCE_ISSUED = "assistance.issued"
    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_COMPLETED = "experiment.completed"
    SYSTEM_ERROR = "system.error"
    SYSTEM_HEALTH = "system.health"


class ExperimentEvent(BaseModel):
    """
    Contract 10: ExperimentEvent.
    Universal structured event envelope stored in database logs and published to event bus.
    """
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=default_uuid, description="Unique event ID")
    event_type: str = Field(description="Event classification, e.g. ACTION_CONFIRMED, VIOLATION_DETECTED")
    timestamp: float = Field(default_factory=current_timestamp, description="Event generation timestamp")
    run_id: str = Field(description="Associated experiment run ID")
    source: str = Field(description="Component that generated event")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event content payload")


class SystemHealth(BaseModel):
    """
    Contract 11: SystemHealth.
    Periodic telemetry tracking subsystem statuses and hardware metrics.
    """
    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(default_factory=current_timestamp)
    system: str = Field(default="ASTRA-E")
    status: str = Field(default="OPERATIONAL", description="OPERATIONAL, DEGRADED, FAULT")
    components: dict[str, str] = Field(
        default_factory=lambda: {
            "camera": "HEALTHY",
            "perception": "HEALTHY",
            "activity": "HEALTHY",
            "procedure": "HEALTHY",
            "storage": "HEALTHY",
            "network": "OFFLINE_STANDALONE",
        }
    )
    metrics: dict[str, float] = Field(
        default_factory=lambda: {
            "fps": 30.0,
            "latency_ms": 15.0,
            "cpu_percent": 25.0,
            "memory_percent": 30.0,
        }
    )
