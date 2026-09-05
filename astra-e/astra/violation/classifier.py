"""Violation classification and severity determination."""

from __future__ import annotations

from astra.contracts.base import Severity, ViolationType


class ViolationClassifier:
    """Classifies severity and priority of detected procedural violations."""

    @staticmethod
    def classify_severity(
        violation_type: ViolationType,
        context: dict | None = None,
    ) -> Severity:
        """
        Determine severity of a violation.
        Aerospace rule: Do not make every violation CRITICAL.
        """
        if violation_type == ViolationType.UNAUTHORIZED_ACTION:
            return Severity.CRITICAL

        if violation_type in (
            ViolationType.SKIPPED_STEP,
            ViolationType.OUT_OF_ORDER,
            ViolationType.WRONG_OBJECT,
            ViolationType.WRONG_TARGET,
        ):
            return Severity.WARNING

        if violation_type == ViolationType.REPEATED_ACTION:
            return Severity.INFO

        return Severity.WARNING
