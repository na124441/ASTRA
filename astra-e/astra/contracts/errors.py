"""Standard error contracts for subsystem fault reporting."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from astra.contracts.base import Severity, current_timestamp, default_uuid


class ErrorEvent(BaseModel):
    """
    Standard error representation across all ASTRA-E subsystems.
    """
    model_config = ConfigDict(frozen=True)

    error_id: str = Field(default_factory=default_uuid, description="Unique error identifier")
    timestamp: float = Field(default_factory=current_timestamp, description="Timestamp of error occurrence")
    source: str = Field(description="Subsystem reporting the error")
    error_code: str = Field(description="Normalized error code, e.g. MODEL_INFERENCE_TIMEOUT")
    severity: Severity = Field(default=Severity.ERROR, description="Error severity")
    recoverable: bool = Field(default=True, description="Whether subsystem can self-recover")
    message: str = Field(description="Detailed error description")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic payload")
