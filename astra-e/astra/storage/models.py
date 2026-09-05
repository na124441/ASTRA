"""Data models and records for ASTRA-E SQLite storage layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunRecord:
    """Record representing an experiment execution session."""
    run_id: str
    experiment_id: str
    procedure_id: str
    status: str
    start_time: float
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    """Record representing an immutable event in the ledger."""
    id: int | None
    message_id: str
    correlation_id: str
    topic: str
    source: str
    timestamp: float
    event_time: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViolationRecord:
    """Record representing a detected procedure deviation."""
    id: int | None
    run_id: str
    violation_type: str
    severity: str
    step_id: str | None
    message: str
    timestamp: float


@dataclass
class AssistanceRecord:
    """Record representing assistance provided to the astronaut."""
    id: int | None
    run_id: str
    assistance_id: str
    type: str
    priority: str
    message: str
    timestamp: float
    channels: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Complete audit summary of an experiment run for post-mission telemetry sync."""
    run_id: str
    experiment_id: str
    procedure_id: str
    status: str
    start_time: float
    end_time: float | None
    duration_seconds: float
    total_events: int
    total_confirmed_actions: int
    total_violations: int
    total_assistance_alerts: int
    events: list[dict[str, Any]] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    assistance: list[dict[str, Any]] = field(default_factory=list)
