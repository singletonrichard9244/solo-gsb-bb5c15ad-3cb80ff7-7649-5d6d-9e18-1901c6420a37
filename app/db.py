"""SQLite persistence layer."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable

DB_PATH = os.environ.get("SENSOR_DB_PATH", "/data/sensor.db")
# When running locally without docker, fall back to ./data next to the repo.
if DB_PATH == "/data/sensor.db" and not os.path.isdir("/data"):
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sensor.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    imported_at REAL NOT NULL,
    total_rows INTEGER NOT NULL DEFAULT 0,
    inserted_rows INTEGER NOT NULL DEFAULT 0,
    overwritten_rows INTEGER NOT NULL DEFAULT 0,
    skipped_rows INTEGER NOT NULL DEFAULT 0,
    -- 1 for batches imported with pre-import key snapshots (undoable);
    -- 0 for batches created before the undo feature existed.
    snapshot_complete INTEGER NOT NULL DEFAULT 0,
    revoked_at REAL
);

-- Pre-import snapshot of every (device, metric, ts) key a batch touched for
-- the first time within that batch. event_id NULL means the key did not exist
-- before the batch; otherwise the row carries the overwritten event's value,
-- raw content and original batch ownership for restore-on-undo.
CREATE TABLE IF NOT EXISTS import_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    import_id INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    ts REAL NOT NULL,
    event_id INTEGER,
    value REAL,
    raw TEXT,
    origin_import_id INTEGER,
    UNIQUE (import_id, device, metric, ts)
);
CREATE INDEX IF NOT EXISTS idx_import_snapshots_batch ON import_snapshots(import_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    ts REAL NOT NULL,
    value REAL,
    raw TEXT,
    import_id INTEGER REFERENCES imports(id) ON DELETE SET NULL,
    UNIQUE (device, metric, ts)
);
CREATE INDEX IF NOT EXISTS idx_events_channel ON events(device, metric, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    logic TEXT NOT NULL DEFAULT 'AND',
    duration REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    op TEXT NOT NULL,
    threshold REAL NOT NULL,
    reset_threshold REAL,
    ord INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fault_intervals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    duration_required REAL NOT NULL,
    confirmed_at REAL NOT NULL,
    evidence_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_faults_rule ON fault_intervals(rule_id);
CREATE INDEX IF NOT EXISTS idx_faults_ts ON fault_intervals(start_ts, end_ts);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    label TEXT NOT NULL,
    color TEXT,
    note TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_annot_ts ON annotations(start_ts, end_ts);

CREATE TABLE IF NOT EXISTS maintenance_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device TEXT NOT NULL,
    name TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    note TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_maint_device ON maintenance_windows(device, start_ts, end_ts);
CREATE INDEX IF NOT EXISTS idx_maint_ts ON maintenance_windows(start_ts, end_ts);

CREATE TABLE IF NOT EXISTS suppressed_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    maintenance_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_supp_rule ON suppressed_segments(rule_id);
CREATE INDEX IF NOT EXISTS idx_supp_ts ON suppressed_segments(start_ts, end_ts);

CREATE TABLE IF NOT EXISTS channel_ttls (
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    ttl_seconds REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (device, metric)
);
"""


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


_conn: sqlite3.Connection | None = None
_write_lock = threading.RLock()


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = connect()
    return _conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    # Migration for databases created before the optional reset threshold:
    # CREATE TABLE IF NOT EXISTS does not add columns to an existing table.
    condition_cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(rule_conditions)").fetchall()
    }
    if "reset_threshold" not in condition_cols:
        conn.execute("ALTER TABLE rule_conditions ADD COLUMN reset_threshold REAL")
    # Migration for databases created before import-batch undo: the imports
    # table gains the snapshot flag and the revocation timestamp.
    import_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(imports)").fetchall()
    }
    if "overwritten_rows" not in import_cols:
        conn.execute("ALTER TABLE imports ADD COLUMN overwritten_rows INTEGER NOT NULL DEFAULT 0")
    if "snapshot_complete" not in import_cols:
        # Pre-feature batches have no snapshots and must be refused on undo.
        conn.execute(
            "ALTER TABLE imports ADD COLUMN snapshot_complete INTEGER NOT NULL DEFAULT 0"
        )
    if "revoked_at" not in import_cols:
        conn.execute("ALTER TABLE imports ADD COLUMN revoked_at REAL")
    conn.commit()


@contextmanager
def transaction():
    """Serialize writers; sqlite3 connections are shared across event-loop
    threads and a single connection cannot run two transactions at once."""
    conn = get_conn()
    with _write_lock:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return get_conn().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    return get_conn().execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
    conn = get_conn()
    with _write_lock:
        cur = conn.execute(sql, tuple(params))
        conn.commit()
    return cur


def reset_all() -> None:
    conn = get_conn()
    for table in (
        "fault_intervals",
        "suppressed_segments",
        "rule_conditions",
        "annotations",
        "maintenance_windows",
        "rules",
        "events",
        "import_snapshots",
        "imports",
        "channel_ttls",
    ):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def fetch_rule(rule_id: int) -> dict | None:
    row = query_one("SELECT * FROM rules WHERE id = ?", (rule_id,))
    if row is None:
        return None
    conds = query(
        "SELECT * FROM rule_conditions WHERE rule_id = ? ORDER BY ord, id",
        (rule_id,),
    )
    return {
        "id": row["id"],
        "name": row["name"],
        "logic": row["logic"],
        "duration": row["duration"],
        "created_at": row["created_at"],
        "conditions": [
            {
                "id": c["id"],
                "device": c["device"],
                "metric": c["metric"],
                "op": c["op"],
                "threshold": c["threshold"],
                "reset_threshold": c["reset_threshold"],
            }
            for c in conds
        ],
    }


def fetch_rules() -> list[dict]:
    rows = query("SELECT id FROM rules ORDER BY id")
    return [r for r in (fetch_rule(row["id"]) for row in rows) if r]


def upsert_event(
    conn: sqlite3.Connection,
    device: str,
    metric: str,
    ts: float,
    value: float | None,
    raw: str,
    import_id: int | None,
) -> bool:
    """Insert an event, or replace value/raw on (device, metric, ts) collision.

    Returns True when a new row was inserted, False on overwrite.
    """
    existing = conn.execute(
        "SELECT id FROM events WHERE device = ? AND metric = ? AND ts = ?",
        (device, metric, ts),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO events (device, metric, ts, value, raw, import_id) VALUES (?,?,?,?,?,?)",
            (device, metric, ts, value, raw, import_id),
        )
        return True
    conn.execute(
        "UPDATE events SET value = ?, raw = ?, import_id = ? WHERE id = ?",
        (value, raw, import_id, existing["id"]),
    )
    return False


def snapshot_import_key(
    conn: sqlite3.Connection,
    import_id: int,
    seen_keys: set[tuple[str, str, float]],
    device: str,
    metric: str,
    ts: float,
) -> None:
    """Record the pre-import state of a (device, metric, ts) key, once per batch.

    The caller tracks keys already snapshotted in *seen_keys*: repeated keys
    within the same batch snapshot only their first encounter even though the
    last valid row still wins the overwrite.  An event row existing before the
    batch is archived with its value, raw content and original batch ownership;
    a missing key is stored with ``event_id IS NULL`` so undo deletes the event
    the batch went on to create.
    """
    key = (device, metric, ts)
    if key in seen_keys:
        return
    seen_keys.add(key)
    existing = conn.execute(
        "SELECT id, value, raw, import_id FROM events"
        " WHERE device = ? AND metric = ? AND ts = ?",
        (device, metric, ts),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO import_snapshots (import_id, device, metric, ts, event_id)"
            " VALUES (?,?,?,?,?)",
            (import_id, device, metric, ts, None),
        )
    else:
        conn.execute(
            "INSERT INTO import_snapshots"
            " (import_id, device, metric, ts, event_id, value, raw, origin_import_id)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                import_id,
                device,
                metric,
                ts,
                existing["id"],
                existing["value"],
                existing["raw"],
                existing["import_id"],
            ),
        )


def fetch_maintenance_window(window_id: int) -> dict | None:
    row = query_one("SELECT * FROM maintenance_windows WHERE id = ?", (window_id,))
    if row is None:
        return None
    return {
        "id": row["id"],
        "device": row["device"],
        "name": row["name"],
        "start": row["start_ts"],
        "end": row["end_ts"],
        "note": row["note"],
        "created_at": row["created_at"],
    }


def fetch_maintenance_windows(
    start: float | None = None,
    end: float | None = None,
    device: str | None = None,
) -> list[dict]:
    """Windows overlapping the half-open filter window [start, end)."""
    where = []
    params: list = []
    if start is not None:
        where.append("end_ts > ?")
        params.append(start)
    if end is not None:
        where.append("start_ts < ?")
        params.append(end)
    if device:
        where.append("device = ?")
        params.append(device)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = query(
        f"SELECT id FROM maintenance_windows{clause} ORDER BY start_ts, device, id",
        params,
    )
    return [w for w in (fetch_maintenance_window(r["id"]) for r in rows) if w]


def faults_for_export() -> list[dict]:
    rows = query(
        """
        SELECT f.*, r.name AS rule_name, r.logic, r.duration
        FROM fault_intervals f JOIN rules r ON r.id = f.rule_id
        ORDER BY f.start_ts, f.rule_id
        """
    )
    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "rule_id": row["rule_id"],
                "rule_name": row["rule_name"],
                "logic": row["logic"],
                "start": row["start_ts"],
                "end": row["end_ts"],
                "duration_required": row["duration_required"],
                "confirmed_at": row["confirmed_at"],
                "evidence": json.loads(row["evidence_json"]),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Per-channel sample validity periods (TTL)
# --------------------------------------------------------------------------- #


def fetch_channel_ttl(device: str, metric: str) -> dict | None:
    row = query_one(
        "SELECT * FROM channel_ttls WHERE device = ? AND metric = ?",
        (device, metric),
    )
    if row is None:
        return None
    return {
        "device": row["device"],
        "metric": row["metric"],
        "ttl_seconds": row["ttl_seconds"],
        "updated_at": row["updated_at"],
    }


def fetch_channel_ttls(
    device: str | None = None, metric: str | None = None
) -> list[dict]:
    """All configured sample validity periods; channels without a row are absent
    (they keep the legacy hold-to-next-sample semantics)."""
    where = []
    params: list = []
    if device is not None:
        where.append("device = ?")
        params.append(device)
    if metric is not None:
        where.append("metric = ?")
        params.append(metric)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = query(
        f"SELECT device, metric, ttl_seconds, updated_at FROM channel_ttls{clause}"
        " ORDER BY device, metric",
        params,
    )
    return [dict(r) for r in rows]


def upsert_channel_ttl(device: str, metric: str, ttl_seconds: float) -> dict:
    """Set (or replace) a channel's positive sample validity period."""
    execute(
        """
        INSERT INTO channel_ttls (device, metric, ttl_seconds, updated_at)
        VALUES (?,?,?,?)
        ON CONFLICT(device, metric) DO UPDATE SET
            ttl_seconds = excluded.ttl_seconds,
            updated_at = excluded.updated_at
        """,
        (device, metric, float(ttl_seconds), time.time()),
    )
    return fetch_channel_ttl(device, metric)


def delete_channel_ttl(device: str, metric: str) -> bool:
    """Clear a channel's validity period; returns True when a row was removed."""
    cur = execute(
        "DELETE FROM channel_ttls WHERE device = ? AND metric = ?",
        (device, metric),
    )
    return cur.rowcount > 0
