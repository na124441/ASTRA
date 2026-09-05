"""Experiment run and lifecycle contracts."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from astra.contracts.base import RunStatus, current_timestamp


class ExperimentRun(BaseModel):
    """
    Contract 12: ExperimentRun.
    Lifecycle entity tracking a single run execution from start to finish.
    """
    model_config = ConfigDict(frozen=True)

    run_id: str = Field(description="Unique experiment execution run ID, e.g. RUN-2026-001")
    experiment_id: str = Field(description="Experiment identifier, e.g. EXP-001")
    procedure_version: str = Field(default="1.0", description="Version of procedure executed")
    started_at: float = Field(default_factory=current_timestamp, description="Run start timestamp")
    ended_at: float | None = Field(default=None, description="Run completion timestamp")
    status: RunStatus = Field(default=RunStatus.CREATED, description="Current run lifecycle status")
    completed_steps: list[str] = Field(default_factory=list, description="Ordered completed step IDs")
    total_violations: int = Field(default=0, description="Count of detected procedural violations")
    summary: dict[str, Any] = Field(default_factory=dict, description="Final run summary statistics")
