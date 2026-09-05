"""Validation logic for experiment and procedure definitions."""

from __future__ import annotations

from astra.contracts.procedure import ProcedureDefinition


class ProcedureValidationError(ValueError):
    """Raised when a procedure definition violates graph integrity constraints."""
    pass


class ProcedureValidator:
    """Validates procedure definitions for structural integrity, reachability, and consistency."""

    @staticmethod
    def validate(definition: ProcedureDefinition) -> list[str]:
        """
        Validates procedure definition. Returns list of warning messages, or raises
        ProcedureValidationError if structural integrity is violated.
        """
        warnings: list[str] = []

        if not definition.steps:
            raise ProcedureValidationError(f"Procedure '{definition.id}' must contain at least one step.")

        step_ids = [step.id for step in definition.steps]
        if len(step_ids) != len(set(step_ids)):
            duplicates = [sid for sid in step_ids if step_ids.count(sid) > 1]
            raise ProcedureValidationError(f"Duplicate step IDs found: {set(duplicates)}")

        step_id_set = set(step_ids)

        # Validate initial step
        if definition.initial_step_id and definition.initial_step_id not in step_id_set:
            raise ProcedureValidationError(
                f"initial_step_id '{definition.initial_step_id}' does not exist in steps."
            )

        # Validate terminal steps
        for term_id in definition.terminal_step_ids:
            if term_id not in step_id_set:
                raise ProcedureValidationError(f"terminal_step_id '{term_id}' does not exist in steps.")

        # Validate transition references
        for step in definition.steps:
            for next_id in step.allowed_next:
                if next_id not in step_id_set:
                    raise ProcedureValidationError(
                        f"Step '{step.id}' references nonexistent allowed_next '{next_id}'."
                    )

            # Check object declarations if objects list is defined
            if definition.objects and step.object and step.object not in definition.objects:
                warnings.append(
                    f"Step '{step.id}' uses undeclared object '{step.object}'. "
                    f"Declared: {definition.objects}"
                )

            # Check target declarations if targets list is defined
            if definition.targets and step.target and step.target not in definition.targets:
                warnings.append(
                    f"Step '{step.id}' uses undeclared target '{step.target}'. "
                    f"Declared: {definition.targets}"
                )

        return warnings
