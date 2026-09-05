"""Unit tests for SQLiteLedger and database storage."""

import time
import pytest
from astra.contracts.activity import ConfirmationMetadata, ConfirmedAction
from astra.contracts.assistance import AssistanceChannel, AssistanceEvent, AssistancePriority
from astra.contracts.base import Severity, ViolationType
from astra.contracts.system import EventTopic
from astra.contracts.violation import ViolationEvent
from astra.events.bus import EventBus
from astra.storage.sqlite import SQLiteLedger


@pytest.fixture
def temp_ledger(tmp_path):
    db_file = tmp_path / "test_astra.db"
    bus = EventBus()
    ledger = SQLiteLedger(db_path=db_file, event_bus=bus, async_writes=False)
    yield ledger, bus
    ledger.close()


def test_schema_and_run_lifecycle(temp_ledger):
    ledger, _ = temp_ledger
    ledger.start_run("RUN-TEST-01", "EXP001", "PROC-01", metadata={"subject": "Astronaut-A"})
    
    run = ledger.get_run("RUN-TEST-01")
    assert run is not None
    assert run.run_id == "RUN-TEST-01"
    assert run.status == "RUNNING"
    assert run.metadata.get("subject") == "Astronaut-A"
    assert run.end_time is None

    ledger.end_run("RUN-TEST-01", status="COMPLETED")
    updated = ledger.get_run("RUN-TEST-01")
    assert updated.status == "COMPLETED"
    assert updated.end_time is not None
    assert updated.end_time >= updated.start_time


def test_event_recording_and_relational_tables(temp_ledger):
    ledger, _ = temp_ledger
    run_id = "RUN-TEST-02"
    ledger.start_run(run_id, "EXP001", "PROC-01")

    # 1. Confirmed action event
    act = ConfirmedAction(
        message_id="conf-001",
        source="confirmation-engine",
        correlation_id=run_id,
        action="PICK",
        object_id="RED",
        confidence=0.95,
        confirmation=ConfirmationMetadata(stable_frames=5, minimum_confidence=0.9, temporal_consistency=0.95),
    )
    ledger.record_event(EventTopic.ACTION_CONFIRMED, act)

    # 2. Violation event
    viol = ViolationEvent(
        message_id="viol-001",
        source="violation-detector",
        correlation_id=run_id,
        violation_type=ViolationType.WRONG_TARGET,
        severity=Severity.CRITICAL,
        observed={"action": "PLACE", "target": "TARGET_A"},
        expected={"action": "PLACE", "target": "TARGET_B", "step_id": "S04"},
        message="Wrong target used",
    )
    ledger.record_event(EventTopic.VIOLATION_DETECTED, viol)

    # 3. Assistance event
    assist = AssistanceEvent(
        message_id="assist-001",
        source="assistance-manager",
        correlation_id=run_id,
        type="PROCEDURE_WARNING",
        priority=AssistancePriority.HIGH,
        message="Warning: Please use Target B",
        channels=[AssistanceChannel.GUI, AssistanceChannel.TTS],
    )
    ledger.record_event(EventTopic.ASSISTANCE_ISSUED, assist)

    # Verify query methods
    events = ledger.get_events(run_id)
    assert len(events) == 3

    violations = ledger.get_violations(run_id)
    assert len(violations) == 1
    assert violations[0].violation_type == ViolationType.WRONG_TARGET.value
    assert violations[0].severity == "CRITICAL"
    assert violations[0].step_id == "S04"

    assistance = ledger.get_assistance(run_id)
    assert len(assistance) == 1
    assert assistance[0].priority == "HIGH"
    assert "TTS" in assistance[0].channels


def test_event_bus_automatic_recording(temp_ledger):
    ledger, bus = temp_ledger
    run_id = "RUN-AUTO-01"
    ledger.start_run(run_id, "EXP001", "PROC-01")

    act = ConfirmedAction(
        message_id="conf-auto-1",
        source="confirmation",
        correlation_id=run_id,
        action="APPROACH",
        confidence=0.9,
    )
    bus.publish(EventTopic.ACTION_CONFIRMED, act)

    # Event should be in ledger
    events = ledger.get_events(run_id, topic=EventTopic.ACTION_CONFIRMED)
    assert len(events) == 1
    assert events[0].message_id == "conf-auto-1"


def test_audit_report_export(temp_ledger):
    ledger, _ = temp_ledger
    run_id = "RUN-AUDIT-01"
    ledger.start_run(run_id, "EXP001", "PROC-DEMO")
    
    act = ConfirmedAction(
        message_id="conf-aud-1",
        source="confirmation",
        correlation_id=run_id,
        action="PICK",
        object_id="RED",
        confidence=0.92,
    )
    ledger.record_event(EventTopic.ACTION_CONFIRMED, act)
    time.sleep(0.05)
    ledger.end_run(run_id, status="COMPLETED")

    report = ledger.export_audit_report(run_id)
    assert report.run_id == run_id
    assert report.status == "COMPLETED"
    assert report.total_events == 1
    assert report.total_confirmed_actions == 1
    assert report.duration_seconds >= 0.04
