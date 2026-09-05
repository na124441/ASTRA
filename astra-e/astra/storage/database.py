"""Database connection factory and schema initializer for ASTRA-E SQLite persistence."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("astra.storage.database")

SCHEMA_SQL = """
-- Experiment Execution Runs
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    procedure_id TEXT NOT NULL,
    status TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    metadata_json TEXT
);

-- Immutable Event Ledger (Audit Trail)
CREATE TABLE IF NOT EXISTS events_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    correlation_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    source TEXT NOT NULL,
    timestamp REAL NOT NULL,
    event_time REAL NOT NULL,
    payload_json TEXT NOT NULL
);

-- Detected Violations Log
CREATE TABLE IF NOT EXISTS violations_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    step_id TEXT,
    message TEXT NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
);

-- Astronaut Assistance Log
CREATE TABLE IF NOT EXISTS assistance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    assistance_id TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    priority TEXT NOT NULL,
    message TEXT NOT NULL,
    channels_json TEXT NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_ledger_correlation ON events_ledger(correlation_id);
CREATE INDEX IF NOT EXISTS idx_ledger_topic ON events_ledger(topic);
CREATE INDEX IF NOT EXISTS idx_ledger_timestamp ON events_ledger(timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_run ON violations_log(run_id);
CREATE INDEX IF NOT EXISTS idx_assistance_run ON assistance_log(run_id);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Creates and configures an edge-ready SQLite connection.
    Enables Write-Ahead Logging (WAL) for high concurrency and zero locking.
    """
    path_obj = Path(db_path)
    if str(path_obj) != ":memory:":
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path),
        timeout=10.0,
        isolation_level=None,  # Autocommit mode
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row

    # Performance and concurrency pragmas
    if str(path_obj) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA foreign_keys = ON;")

    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Executes idempotent schema migration."""
    conn.executescript(SCHEMA_SQL)
    logger.info("Initialized ASTRA-E SQLite schema successfully.")
