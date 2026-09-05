"""Immutable audit logging for ASTRA Collector uploads."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
from .schemas import AuditRecord

logger = logging.getLogger("astra.collector.audit")


class CollectorAuditLogger:
    """Thread-safe append-only audit logger for tracking raw video uploads."""

    def __init__(self, log_dir: Path | str | None = None) -> None:
        if log_dir is None:
            if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
                import tempfile
                self.log_dir = Path(tempfile.gettempdir()) / "collector_audit"
            else:
                self.log_dir = Path(__file__).resolve().parent.parent.parent / "data" / "collector_audit"
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "audit_log.jsonl"
        self._lock = threading.Lock()

    def record_verified_upload(self, record: AuditRecord) -> None:
        """Append an immutable audit entry."""
        line = json.dumps(record.model_dump(), sort_keys=True) + "\n"
        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line)
        logger.info(
            "Audit recorded: upload_id=%s, run_id=%s, sha256=%s, status=%s",
            record.upload_id,
            record.run_id,
            record.sha256[:12],
            record.status,
        )

    def query_records(
        self,
        collector_id: str | None = None,
        run_id: str | None = None,
        upload_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        """Query existing audit entries matching filters."""
        if not self.log_file.exists():
            return []

        results: list[AuditRecord] = []
        with self._lock:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        rec = AuditRecord(**data)
                        if collector_id and rec.collector_id != collector_id:
                            continue
                        if run_id and rec.run_id != run_id:
                            continue
                        if upload_id and rec.upload_id != upload_id:
                            continue
                        results.append(rec)
                    except Exception as e:
                        logger.warning("Error parsing audit log line: %s", e)

        return results[-limit:]
