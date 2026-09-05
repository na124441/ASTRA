"""Configuration loaders for experiments and procedure YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from astra.contracts.procedure import ProcedureDefinition, ProcedureStep


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load a generic YAML file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_procedure_from_yaml(file_path: str | Path) -> ProcedureDefinition:
    """
    Load a ProcedureDefinition from an experiment procedure.yaml file.
    """
    data = load_yaml(file_path)
    proc_dict = data.get("procedure", data)

    raw_steps = proc_dict.get("steps", [])
    steps: list[ProcedureStep] = []
    for s in raw_steps:
        steps.append(
            ProcedureStep(
                id=s["id"],
                action=s["action"],
                object=s.get("object"),
                target=s.get("target"),
                description=s.get("description", ""),
                allowed_next=s.get("allowed_next", []),
                optional=s.get("optional", False),
                repeatable=s.get("repeatable", False),
            )
        )

    return ProcedureDefinition(
        id=proc_dict.get("id", "PROC-001"),
        experiment_id=proc_dict["experiment_id"],
        name=proc_dict.get("name", ""),
        version=str(proc_dict.get("version", "1.0")),
        objects=proc_dict.get("objects", []),
        targets=proc_dict.get("targets", []),
        steps=steps,
        initial_step_id=proc_dict.get("initial_step_id"),
        terminal_step_ids=proc_dict.get("terminal_step_ids", []),
    )
