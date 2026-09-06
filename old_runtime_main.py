"""ASTRA-E Runtime simulation application (Sprint 01 Demo)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure astra package is importable when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astra.contracts.activity import ConfirmationMetadata, ConfirmedAction
from astra.contracts.base import DecisionType
from astra.contracts.procedure import ProcedureDefinition, ProcedureStep
from astra.contracts.system import EventTopic
from astra.contracts.violation import ViolationEvent
from astra.events.bus import EventBus
from astra.procedure.engine import ProcedureEngine
from astra.violation.detector import ViolationDetector
from astra.assistance.manager import AssistanceManager


def create_demo_procedure() -> ProcedureDefinition:
    """
    Creates the benchmark sprint demo procedure:
    S01: PICK RED
    S02: PLACE RED -> TARGET_A
    S03: PICK BLUE
    S04: PLACE BLUE -> TARGET_B
    """
    return ProcedureDefinition(
        id="PROC-SPRINT-01",
        experiment_id="EXP001",
        name="Sprint 01 Dual Component Sorter",
        version="1.0",
        objects=["RED", "BLUE"],
        targets=["TARGET_A", "TARGET_B"],
        initial_step_id="S01",
        terminal_step_ids=["S04"],
        steps=[
            ProcedureStep(
                id="S01",
                action="PICK",
                object="RED",
                description="Pick the red component",
                allowed_next=["S02"],
            ),
            ProcedureStep(
                id="S02",
                action="PLACE",
                object="RED",
                target="TARGET_A",
                description="Place the red component into Target A",
                allowed_next=["S03"],
            ),
            ProcedureStep(
                id="S03",
                action="PICK",
                object="BLUE",
                description="Pick the blue component",
                allowed_next=["S04"],
            ),
            ProcedureStep(
                id="S04",
                action="PLACE",
                object="BLUE",
                target="TARGET_B",
                description="Place the blue component into Target B",
                allowed_next=[],
            ),
        ],
    )


def run_simulation(verbose: bool = True) -> dict[str, int]:
    """
    Executes the non-negotiable Sprint 01 mock simulation sequence:
    START
    -> PICK RED
    -> PLACE RED
    -> PICK BLUE
    -> WRONG PLACE (inject violation)
    -> VIOLATION DETECTED
    -> CORRECTION
    -> COMPLETE
    """
    run_id = "RUN0001"
    event_bus = EventBus()
    violation_detector = ViolationDetector(suppression_cooldown_seconds=0.1)
    engine = ProcedureEngine(event_bus=event_bus, violation_detector=violation_detector)
    assistance_mgr = AssistanceManager(event_bus=event_bus)

    procedure = create_demo_procedure()
    assistance_mgr.set_procedure(procedure)
    engine.start(run_id=run_id, procedure=procedure)

    # Listeners for real-time display
    last_violation: list[ViolationEvent] = []
    def on_violation(v: ViolationEvent) -> None:
        last_violation.append(v)
    event_bus.subscribe(EventTopic.VIOLATION_DETECTED, on_violation)

    if verbose:
        print("\n" + "=" * 66)
        print("║" + "ASTRA-E AUTONOMOUS RUNTIME (BAS)".center(64) + "║")
        print("=" * 66)
        print(f"Experiment: {procedure.experiment_id} | Run: {run_id} | Mode: OFFLINE STANDALONE")
        print("[START] Procedure initialized\n")

    # Sequence of simulated actions including intentional violation and recovery
    simulated_stream = [
        # Step 1: PICK RED (Valid)
        ConfirmedAction(
            message_id="act-001",
            source="confidence-manager",
            correlation_id=run_id,
            action="PICK",
            object_id="RED",
            confidence=0.98,
            confirmation=ConfirmationMetadata(
                stable_frames=15,
                minimum_confidence=0.92,
                temporal_consistency=0.96,
            ),
        ),
        # Step 2: PLACE RED -> TARGET_A (Valid)
        ConfirmedAction(
            message_id="act-002",
            source="confidence-manager",
            correlation_id=run_id,
            action="PLACE",
            object_id="RED",
            target_id="TARGET_A",
            confidence=0.96,
            confirmation=ConfirmationMetadata(
                stable_frames=14,
                minimum_confidence=0.90,
                temporal_consistency=0.95,
            ),
        ),
        # Step 3: PICK BLUE (Valid)
        ConfirmedAction(
            message_id="act-003",
            source="confidence-manager",
            correlation_id=run_id,
            action="PICK",
            object_id="BLUE",
            confidence=0.94,
            confirmation=ConfirmationMetadata(
                stable_frames=12,
                minimum_confidence=0.88,
                temporal_consistency=0.92,
            ),
        ),
        # Step 4 (Fault Injection): PLACE BLUE -> TARGET_A (WRONG TARGET!)
        ConfirmedAction(
            message_id="act-004",
            source="confidence-manager",
            correlation_id=run_id,
            action="PLACE",
            object_id="BLUE",
            target_id="TARGET_A",
            confidence=0.97,
            confirmation=ConfirmationMetadata(
                stable_frames=16,
                minimum_confidence=0.91,
                temporal_consistency=0.98,
            ),
        ),
        # Step 5 (Correction): PLACE BLUE -> TARGET_B (Valid)
        ConfirmedAction(
            message_id="act-005",
            source="confidence-manager",
            correlation_id=run_id,
            action="PLACE",
            object_id="BLUE",
            target_id="TARGET_B",
            confidence=0.99,
            confirmation=ConfirmationMetadata(
                stable_frames=18,
                minimum_confidence=0.95,
                temporal_consistency=0.99,
            ),
        ),
    ]

    valid_count = 0
    violation_count = 0

    for action in simulated_stream:
        target_str = f" -> {action.target_id}" if action.target_id else ""
        obj_str = f" {action.object_id}" if action.object_id else ""
        action_label = f"{action.action}{obj_str}{target_str}"

        if verbose:
            print(f"[OBS] {action_label}")
            print(f"[CONF] {action.confidence:.2f}")

        decision = engine.process(action)

        if decision.decision == DecisionType.VALID:
            valid_count += 1
            curr = decision.current_state or "START"
            nxt = decision.next_state or "COMPLETE"
            if verbose:
                print(f"[PROC] {curr} -> {nxt} ✓")
                # Show next expected step guidance if not complete
                if not engine.is_completed:
                    snapshot = engine.state
                    next_steps = snapshot.next_expected
                    if next_steps:
                        next_step_obj = engine.graph.get_step(next_steps[0])
                        if next_step_obj:
                            desc = next_step_obj.description or f"{next_step_obj.action} {next_step_obj.object or ''}".strip()
                            print(f"[GUIDE] Next: {desc}")
                print()
        elif decision.decision == DecisionType.INVALID:
            violation_count += 1
            v = decision.violation
            if verbose and v:
                print(f"[VIOLATION] Type: {v.violation_type.value} ({v.severity.value})")
                print(f"            Expected: {v.expected.get('target') or v.expected.get('action')}")
                print(f"            Observed: {v.observed.get('target') or v.observed.get('action')}")
                # Most recent assistance alert
                if assistance_mgr.history:
                    latest_assist = assistance_mgr.history[-1]
                    print(f"[ASSIST] {latest_assist.message}")
                print()

    if verbose:
        print("─" * 66)
        if engine.is_completed:
            print(f"[PROC] → COMPLETE ✓")
            print(f"[RUN] {procedure.experiment_id} completed successfully.")
            print(f"      Valid transitions: {valid_count} | Violations caught & resolved: {violation_count}")
        else:
            print(f"[RUN] {procedure.experiment_id} ended incomplete.")
        print("=" * 66 + "\n")

    return {
        "valid_count": valid_count,
        "violation_count": violation_count,
        "completed": 1 if engine.is_completed else 0,
    }


def main() -> None:
    """Entry point for the ASTRA-E mock runtime."""
    parser = argparse.ArgumentParser(description="ASTRA-E Autonomous Space Task Runtime Simulation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose terminal output")
    args = parser.parse_args()

    results = run_simulation(verbose=not args.quiet)
    if results["completed"] != 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
