"""Base contracts and core enumerations for ASTRA-E."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    """Recognized experiment primitive and composite actions."""
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    TOUCH = "TOUCH"
    GRASP = "GRASP"
    PICK = "PICK"
    MOVE = "MOVE"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    OPEN_CONTAINER = "OPEN_CONTAINER"
    CLOSE_CONTAINER = "CLOSE_CONTAINER"
    MANIPULATE = "MANIPULATE"


class Severity(str, Enum):
    """Severity levels for events, violations, and errors."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"


class DecisionType(str, Enum):
    """Procedure engine decision categories."""
    VALID = "VALID"
    INVALID = "INVALID"
    UNCERTAIN = "UNCERTAIN"
    IGNORED = "IGNORED"


class ViolationType(str, Enum):
    """Standardized procedural violation categories."""
    SKIPPED_STEP = "SKIPPED_STEP"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    WRONG_OBJECT = "WRONG_OBJECT"
    WRONG_TARGET = "WRONG_TARGET"
    UNAUTHORIZED_ACTION = "UNAUTHORIZED_ACTION"
    INCOMPLETE_ACTION = "INCOMPLETE_ACTION"
    REPEATED_ACTION = "REPEATED_ACTION"


class RunStatus(str, Enum):
    """Lifecycle status of an experiment run."""
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


class AssistancePriority(str, Enum):
    """Priority level for astronaut-facing assistance."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AssistanceChannel(str, Enum):
    """Delivery channels for guidance and alerts."""
    GUI = "GUI"
    TTS = "TTS"
    VISUAL_ALERT = "VISUAL_ALERT"
    GROUND_INTERFACE = "GROUND_INTERFACE"


def default_uuid() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def current_timestamp() -> float:
    """Generate high-precision current POSIX timestamp."""
    return time.time()


class BaseMessage(BaseModel):
    """
    Base contract for all internal runtime messages and telemetry.
    Immutable by default to guarantee traceability and thread safety.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str = Field(default_factory=default_uuid, description="Globally unique message ID")
    schema_version: str = Field(default="1.0", description="Contract schema version")
    timestamp: float = Field(default_factory=current_timestamp, description="Timestamp when message was produced")
    source: str = Field(description="Originating subsystem or module")
    correlation_id: str = Field(description="Run or session correlation ID for complete traceability")
