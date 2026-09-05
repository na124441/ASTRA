"""SQLiteLedger: Offline append-only event store and audit trail for ASTRA-E (FR-023, NFR-006)."""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from astra.contracts.activity import ConfirmedAction
from astra.contracts.assistance import AssistanceEvent
from astra.contracts.base import BaseMessage
from astra.contracts.system import EventTopic
from astra.contracts.violation import ViolationEvent
from astra.events.bus import EventBus
from astra.storage.database import get_connection, init_schema
from astra.storage.models import AssistanceRecord, AuditReport, EventRecord, RunRecord, ViolationRecord

logger = logging.getLogger("astra.storage.sqlite")


class SQLiteLedger:
    """
    Offline-first append-only execution ledger and event store.
    Features:
      - Asynchronous write queue to ensure zero-latency impact on real-time vision pipelines.
      - Immutable audit trail of every confirmed action, procedure decision, and violation.
      - Automated subscription to system EventBus.
      - Full audit report exporter for post-mission telemetry sync.
    """

    def __init__(
        self,
        db_path: str | Path = "data/runs/astra_runtime.db",
        event_bus: EventBus | None = None,
        async_writes: bool = True,
    ) -> None:
        self.db_path = db_path
        self.event_bus = event_bus
        self.async_writes = async_writes

        self._conn = get_connection(self.db_path)
        init_schema(self._conn)

        self._queue: queue.Queue[tuple[str, tuple[Any, ...]] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        if self.async_writes:
            self._worker_thread = threading.Thread(
                target=self._write_worker,
                name="astra-sqlite-writer",
                daemon=True,
            )
            self._worker_thread.start()

        if self.event_bus:
            self._subscribe_all()

    def _subscribe_all(self) -> None:
        """Subscribe to all operational and safety topics on the event bus."""
        assert self.event_bus is not None
        topics = [
            EventTopic.ACTION_CONFIRMED,
            EventTopic.PROCEDURE_TRANSITIONED,
            EventTopic.PROCEDURE_COMPLETED,
            EventTopic.VIOLATION_DETECTED,
            EventTopic.ASSISTANCE_ISSUED,
            EventTopic.EXPERIMENT_STARTED,
            EventTopic.EXPERIMENT_COMPLETED,
            EventTopic.SYSTEM_ERROR,
        ]
        for topic in topics:
            self.event_bus.subscribe(topic, lambda msg, t=topic: self.record_event(t, msg))

    def _write_worker(self) -> None:
        """Background thread executing batched database writes without blocking caller."""
        # Thread has its own connection for thread-safety in SQLite
        worker_conn = get_connection(self.db_path)
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    self._queue.task_done()
                    break

                query, params = item
                try:
                    worker_conn.execute(query, params)
                except Exception as e:
                    logger.error(f"Error executing ledger write: {e}", exc_info=True)
                finally:
                    self._queue.task_done()
        finally:
            worker_conn.close()

    def _execute(self, query: str, params: tuple[Any, ...]) -> None:
        """Execute a query synchronously or enqueue for background execution."""
        if self.async_writes:
            self._queue.put((query, params))
        else:
            self._conn.execute(query, params)

    def start_run(
        self,
        run_id: str,
        experiment_id: str,
        procedure_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record the start of an experiment session."""
        now = time.time()
        meta_json = json.dumps(metadata or {})
        query = """
            INSERT OR REPLACE INTO experiment_runs 
            (run_id, experiment_id, procedure_id, status, start_time, end_time, metadata_json)
            VALUES (?, ?, ?, 'RUNNING', ?, NULL, ?)
        """
        self._execute(query, (run_id, experiment_id, procedure_id, now, meta_json))
        logger.info(f"Recorded run start in ledger: {run_id} ({procedure_id})")

    def end_run(self, run_id: str, status: str = "COMPLETED") -> None:
        """Record the termination or completion of an experiment session."""
        now = time.time()
        query = "UPDATE experiment_runs SET status = ?, end_time = ? WHERE run_id = ?"
        self._execute(query, (status, now, run_id))
        logger.info(f"Recorded run {status} in ledger: {run_id}")

    def record_event(self, topic: str, message: Any) -> None:
        """
        Record a typed message or event dictionary to the immutable ledger.
        Extracts structured fields and writes to specific indexes.
        """
        now = time.time()
        msg_dict: dict[str, Any]

        if isinstance(message, BaseMessage):
            msg_dict = message.model_dump(mode="json")
            message_id = message.message_id
            correlation_id = message.correlation_id or "UNKNOWN"
            source = message.source
            timestamp = message.timestamp
            event_time = message.event_time
        elif isinstance(message, dict):
            msg_dict = message
            message_id = str(msg_dict.get("message_id", f"msg-{int(now * 1000)}"))
            correlation_id = str(msg_dict.get("correlation_id") or msg_dict.get("run_id") or "UNKNOWN")
            source = str(msg_dict.get("source", "unknown"))
            timestamp = float(msg_dict.get("timestamp", now))
            event_time = float(msg_dict.get("event_time", timestamp))
        else:
            msg_dict = {"raw": str(message)}
            message_id = f"msg-{int(now * 1000)}"
            correlation_id = "UNKNOWN"
            source = "unknown"
            timestamp = now
            event_time = now

        payload_json = json.dumps(msg_dict)
        query = """
            INSERT OR IGNORE INTO events_ledger 
            (message_id, correlation_id, topic, source, timestamp, event_time, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self._execute(query, (message_id, correlation_id, topic, source, timestamp, event_time, payload_json))

        # Specialized relational table writes
        if isinstance(message, ViolationEvent) or topic == EventTopic.VIOLATION_DETECTED:
            v_type = msg_dict.get("violation_type", "UNKNOWN")
            severity = msg_dict.get("severity", "MEDIUM")
            step_id = msg_dict.get("expected", {}).get("step_id")
            text = msg_dict.get("message", "")
            v_query = """
                INSERT INTO violations_log (run_id, violation_type, severity, step_id, message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            self._execute(v_query, (correlation_id, v_type, severity, step_id, text, timestamp))

        elif isinstance(message, AssistanceEvent) or topic == EventTopic.ASSISTANCE_ISSUED:
            a_id = message_id
            a_type = msg_dict.get("type", "GUIDANCE")
            priority = msg_dict.get("priority", "LOW")
            text = msg_dict.get("message", "")
            channels_json = json.dumps(msg_dict.get("channels", []))
            a_query = """
                INSERT OR IGNORE INTO assistance_log (run_id, assistance_id, type, priority, message, channels_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            self._execute(a_query, (correlation_id, a_id, a_type, priority, text, channels_json, timestamp))

    def flush(self) -> None:
        """Block until all queued writes have been executed."""
        if self.async_writes and self._worker_thread and self._worker_thread.is_alive():
            self._queue.join()

    def close(self) -> None:
        """Flush remaining items and close database resources."""
        self.flush()
        if self._worker_thread and self._worker_thread.is_alive():
            self._stop_event.set()
            self._queue.put(None)
            self._worker_thread.join(timeout=2.0)
        self._conn.close()

    def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve execution run metadata."""
        self.flush()
        row = self._conn.execute(
            "SELECT run_id, experiment_id, procedure_id, status, start_time, end_time, metadata_json FROM experiment_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return RunRecord(
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            procedure_id=row["procedure_id"],
            status=row["status"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def list_runs(self) -> list[RunRecord]:
        """List all experiment runs sorted by start time descending."""
        self.flush()
        rows = self._conn.execute(
            "SELECT run_id, experiment_id, procedure_id, status, start_time, end_time, metadata_json FROM experiment_runs ORDER BY start_time DESC"
        ).fetchall()
        return [
            RunRecord(
                run_id=r["run_id"],
                experiment_id=r["experiment_id"],
                procedure_id=r["procedure_id"],
                status=r["status"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                metadata=json.loads(r["metadata_json"] or "{}"),
            )
            for r in rows
        ]

    def get_events(self, run_id: str, topic: str | None = None) -> list[EventRecord]:
        """Retrieve all events recorded for a given run."""
        self.flush()
        if topic:
            rows = self._conn.execute(
                "SELECT id, message_id, correlation_id, topic, source, timestamp, event_time, payload_json FROM events_ledger WHERE correlation_id = ? AND topic = ? ORDER BY id ASC",
                (run_id, topic),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, message_id, correlation_id, topic, source, timestamp, event_time, payload_json FROM events_ledger WHERE correlation_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()

        return [
            EventRecord(
                id=r["id"],
                message_id=r["message_id"],
                correlation_id=r["correlation_id"],
                topic=r["topic"],
                source=r["source"],
                timestamp=r["timestamp"],
                event_time=r["event_time"],
                payload=json.loads(r["payload_json"]),
            )
            for r in rows
        ]

    def get_violations(self, run_id: str) -> list[ViolationRecord]:
        """Retrieve all detected deviations for a run."""
        self.flush()
        rows = self._conn.execute(
            "SELECT id, run_id, violation_type, severity, step_id, message, timestamp FROM violations_log WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [
            ViolationRecord(
                id=r["id"],
                run_id=r["run_id"],
                violation_type=r["violation_type"],
                severity=r["severity"],
                step_id=r["step_id"],
                message=r["message"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    def get_assistance(self, run_id: str) -> list[AssistanceRecord]:
        """Retrieve all assistance alerts emitted during a run."""
        self.flush()
        rows = self._conn.execute(
            "SELECT id, run_id, assistance_id, type, priority, message, channels_json, timestamp FROM assistance_log WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [
            AssistanceRecord(
                id=r["id"],
                run_id=r["run_id"],
                assistance_id=r["assistance_id"],
                type=r["type"],
                priority=r["priority"],
                message=r["message"],
                channels=json.loads(r["channels_json"]),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    def export_audit_report(self, run_id: str) -> AuditReport:
        """
        Generate complete, structured forensic audit report for ground control downlink.
        """
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run ID '{run_id}' not found in ledger.")

        events = self.get_events(run_id)
        violations = self.get_violations(run_id)
        assistance = self.get_assistance(run_id)

        duration = (run.end_time or time.time()) - run.start_time
        confirmed_actions = [e for e in events if e.topic == EventTopic.ACTION_CONFIRMED]

        return AuditReport(
            run_id=run.run_id,
            experiment_id=run.experiment_id,
            procedure_id=run.procedure_id,
            status=run.status,
            start_time=run.start_time,
            end_time=run.end_time,
            duration_seconds=round(duration, 2),
            total_events=len(events),
            total_confirmed_actions=len(confirmed_actions),
            total_violations=len(violations),
            total_assistance_alerts=len(assistance),
            events=[e.payload for e in events],
            violations=[
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "step_id": v.step_id,
                    "message": v.message,
                    "timestamp": v.timestamp,
                }
                for v in violations
            ],
            assistance=[
                {
                    "id": a.assistance_id,
                    "type": a.type,
                    "priority": a.priority,
                    "message": a.message,
                    "channels": a.channels,
                    "timestamp": a.timestamp,
                }
                for a in assistance
            ],
        )
