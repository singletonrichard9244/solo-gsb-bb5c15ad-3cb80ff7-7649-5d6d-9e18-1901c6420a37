"""Import batch undo: pre-import snapshots, single-transaction undo of the
latest batch (delete inserted events, restore overwritten value/raw/batch
attribution, mark the batch undone), rejection of non-latest or
snapshot-less batches, fault re-evaluation and reset cleanup.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import to_epoch


def at(minute: int, second: int = 0) -> float:
    return to_epoch(f"2026-09-23T00:{minute:02d}:{second:02d}Z")


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


def _import(pairs, device="d", metric="m", filename=None):
    """Import [(minute, value), ...] as one batch; returns the result dict."""
    rows = [
        {
            "device": device,
            "metric": metric,
            "ts": f"2026-09-23T00:{m:02d}:00Z",
            "value": v,
        }
        for m, v in pairs
    ]
    return services.import_events(
        "\n".join(json.dumps(r) for r in rows), filename=filename
    )


def _event(device="d", metric="m", minute=0):
    return db.query_one(
        "SELECT * FROM events WHERE device = ? AND metric = ? AND ts = ?",
        (device, metric, at(minute)),
    )


def _snapshots(import_id):
    return db.query(
        "SELECT * FROM import_event_snapshots WHERE import_id = ? ORDER BY id",
        (import_id,),
    )


# --------------------------------------------------------------------------- #
# snapshot recording during import
# --------------------------------------------------------------------------- #


def test_snapshots_record_pre_import_state(scratch_db):
    first = _import([(0, 1.0), (1, 2.0)])
    second = _import([(1, 22.0), (2, 3.0)])  # overwrite minute 1, insert minute 2

    snaps = _snapshots(second["import_id"])
    assert len(snaps) == 2
    by_minute = {int((s["ts"] - at(0)) // 60): s for s in snaps}

    # overwritten key: snapshot carries the pre-import value/raw/attribution
    overwritten = by_minute[1]
    assert overwritten["event_id"] is not None
    assert overwritten["value"] == 2.0
    assert json.loads(overwritten["raw"])["value"] == 2.0
    assert overwritten["prev_import_id"] == first["import_id"]

    # brand-new key: all event fields NULL (undo will delete the event)
    inserted = by_minute[2]
    assert inserted["event_id"] is None
    assert inserted["value"] is None
    assert inserted["raw"] is None
    assert inserted["prev_import_id"] is None


def test_duplicate_keys_in_batch_snapshot_once_last_value_wins(scratch_db):
    _import([(0, 1.0)])
    # minute 0 appears three times in one batch: first encounter snapshots the
    # pre-import value, the last valid value still wins
    result = _import([(0, 10.0), (0, 20.0), (0, 30.0), (1, 5.0), (1, 6.0)])

    snaps = _snapshots(result["import_id"])
    assert len(snaps) == 2  # one per distinct key, not per occurrence
    snap0 = [s for s in snaps if s["ts"] == at(0)][0]
    assert snap0["value"] == 1.0  # pre-import state, not an intermediate value
    assert _event(minute=0)["value"] == 30.0
    assert _event(minute=1)["value"] == 6.0

    # undo restores the pre-import value, not an intermediate one
    outcome = services.undo_import(result["import_id"])
    assert outcome["deleted_events"] == 1
    assert outcome["restored_events"] == 1
    assert _event(minute=0)["value"] == 1.0
    assert _event(minute=1) is None


def test_skipped_rows_are_not_snapshotted(scratch_db):
    text = "\n".join(
        [
            json.dumps({"device": "d", "metric": "m", "ts": "2026-09-23T00:00:00Z", "value": 1}),
            json.dumps({"device": "d", "metric": "m", "ts": "2026-09-23T00:01:00Z", "value": "abc"}),
            json.dumps({"device": "d", "ts": "2026-09-23T00:02:00Z", "value": 3}),  # missing metric
        ]
    )
    result = services.import_events(text)
    assert result["skipped"] == 2
    assert len(_snapshots(result["import_id"])) == 1


def test_nested_metrics_snapshot_each_sub_key(scratch_db):
    text = json.dumps(
        {"device": "d", "ts": "2026-09-23T00:00:00Z", "value": {"temp": 80, "vib": 2}}
    )
    result = services.import_events(text)
    snaps = _snapshots(result["import_id"])
    assert {(s["device"], s["metric"]) for s in snaps} == {("d", "temp"), ("d", "vib")}


# --------------------------------------------------------------------------- #
# undo behaviour
# --------------------------------------------------------------------------- #


def test_undo_deletes_inserts_and_restores_overwrites(scratch_db):
    first = _import([(0, 1.0), (1, 2.0)], filename="a.jsonl")
    second = _import([(1, 22.0), (2, 3.0)], filename="b.jsonl")

    outcome = services.undo_import(second["import_id"])
    assert outcome["deleted_events"] == 1  # minute 2 was inserted by batch 2
    assert outcome["restored_events"] == 1  # minute 1 was overwritten by batch 2
    assert outcome["evaluation"]["rules_evaluated"] == 0

    # minute 1 restored: value, raw content and original batch attribution
    restored = _event(minute=1)
    assert restored["value"] == 2.0
    assert json.loads(restored["raw"])["value"] == 2.0
    assert restored["import_id"] == first["import_id"]
    # minute 2 (batch-2 insert) is gone; minute 0 (untouched by batch 2) stays
    assert _event(minute=2) is None
    assert _event(minute=0)["value"] == 1.0

    # the batch is marked undone and keeps its snapshots for audit
    row = db.query_one("SELECT undone_at FROM imports WHERE id = ?", (second["import_id"],))
    assert row["undone_at"] is not None
    assert len(_snapshots(second["import_id"])) == 2


def test_undo_is_lifo(scratch_db):
    first = _import([(0, 1.0)])
    second = _import([(1, 2.0)])

    # the older batch is not the latest: rejected
    with pytest.raises(services.UndoConflict, match="最近一次"):
        services.undo_import(first["import_id"])

    services.undo_import(second["import_id"])
    # after undoing the newer batch, the older one becomes undoable
    outcome = services.undo_import(first["import_id"])
    assert outcome["deleted_events"] == 1
    assert db.query_one("SELECT COUNT(*) AS c FROM events")["c"] == 0


def test_undo_rejects_already_undone(scratch_db):
    result = _import([(0, 1.0)])
    services.undo_import(result["import_id"])
    with pytest.raises(services.UndoConflict, match="已撤销"):
        services.undo_import(result["import_id"])


def test_undo_rejects_unknown_batch(scratch_db):
    with pytest.raises(KeyError):
        services.undo_import(999)


def test_undo_rejects_batch_without_snapshots(scratch_db):
    """Batches imported before the snapshot feature are explicitly rejected."""
    # simulate legacy data: an import row plus events, no snapshot rows
    legacy_id = db.execute(
        "INSERT INTO imports (filename, imported_at, total_rows, inserted_rows)"
        " VALUES ('old.jsonl', 1.0, 1, 1)"
    ).lastrowid
    db.execute(
        "INSERT INTO events (device, metric, ts, value, raw, import_id)"
        " VALUES ('d', 'm', ?, 5.0, '{}', ?)",
        (at(0), legacy_id),
    )
    with pytest.raises(services.UndoConflict, match="快照"):
        services.undo_import(legacy_id)
    # nothing was changed
    assert _event(minute=0)["value"] == 5.0
    assert db.query_one("SELECT undone_at AS u FROM imports WHERE id = ?", (legacy_id,))["u"] is None


def test_undo_rolls_back_everything_on_failure(scratch_db):
    first = _import([(0, 1.0)])
    second = _import([(0, 10.0), (1, 2.0)])

    # force the DELETE of the inserted event to fail mid-transaction
    conn = db.get_conn()
    conn.execute(
        "CREATE TRIGGER fail_event_delete BEFORE DELETE ON events"
        " BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    conn.commit()
    try:
        with pytest.raises(Exception, match="boom"):
            services.undo_import(second["import_id"])
    finally:
        conn.execute("DROP TRIGGER fail_event_delete")
        conn.commit()

    # full rollback: overwrite not restored, insert not deleted, batch not marked
    assert _event(minute=0)["value"] == 10.0
    assert _event(minute=0)["import_id"] == second["import_id"]
    assert _event(minute=1)["value"] == 2.0
    row = db.query_one("SELECT undone_at AS u FROM imports WHERE id = ?", (second["import_id"],))
    assert row["u"] is None

    # and the batch can still be undone afterwards
    outcome = services.undo_import(second["import_id"])
    assert (outcome["deleted_events"], outcome["restored_events"]) == (1, 1)
    assert _event(minute=0)["import_id"] == first["import_id"]


def test_undo_reevaluates_faults(scratch_db):
    db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES ('r', 'AND', 0, 0.0)"
    )
    db.execute(
        "INSERT INTO rule_conditions (rule_id, device, metric, op, threshold, ord)"
        " VALUES (1, 'd', 'm', '>=', 80, 0)"
    )
    result = _import([(0, 90.0), (1, 90.0)])  # auto re-evaluation confirms a fault
    assert len(services.query_faults(None, None)) == 1

    outcome = services.undo_import(result["import_id"])
    assert outcome["evaluation"]["rules_evaluated"] == 1
    assert outcome["evaluation"]["fault_intervals"] == 0
    assert services.query_faults(None, None) == []


def test_reset_clears_snapshots(scratch_db):
    _import([(0, 1.0)])
    _import([(0, 2.0)])
    assert db.query_one("SELECT COUNT(*) AS c FROM import_event_snapshots")["c"] == 2
    db.reset_all()
    assert db.query_one("SELECT COUNT(*) AS c FROM import_event_snapshots")["c"] == 0
    assert db.query_one("SELECT COUNT(*) AS c FROM imports")["c"] == 0


def test_schema_migration_adds_undone_at_to_old_db(scratch_db):
    # simulate a pre-feature database: imports without the undone_at column
    # and no snapshot table at all
    conn = db.get_conn()
    conn.execute("ALTER TABLE imports DROP COLUMN undone_at")
    conn.execute("DROP TABLE import_event_snapshots")
    conn.execute("INSERT INTO imports (filename, imported_at, total_rows) VALUES ('x', 1.0, 0)")
    conn.commit()
    db.init_db()  # a fresh process would run migrations on the old schema
    row = db.query_one("SELECT * FROM imports WHERE id = 1")
    assert row["undone_at"] is None
    assert db.query_one(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'import_event_snapshots'"
    ) is not None
    # and new imports still work on the migrated schema
    result = _import([(0, 1.0)])
    assert result["import_id"] == 2
    assert len(_snapshots(result["import_id"])) == 1


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def _post_text(client, pairs, device="d", metric="m"):
    return client.post(
        "/api/import/text",
        json={
            "text": "\n".join(
                json.dumps(
                    {
                        "device": device,
                        "metric": metric,
                        "ts": f"2026-09-23T00:{m:02d}:00Z",
                        "value": v,
                    }
                )
                for m, v in pairs
            )
        },
    )


def test_api_lists_batches_with_undo_flags(client):
    _post_text(client, [(0, 1.0)])
    _post_text(client, [(0, 2.0), (1, 3.0)])

    batches = client.get("/api/imports").json()
    assert [b["id"] for b in batches] == [2, 1]  # newest first
    latest, older = batches
    assert latest["undoable"] is True
    assert latest["snapshot_complete"] is True
    assert latest["snapshot_count"] == 2
    assert latest["snapshot_inserts"] == 1
    assert latest["snapshot_overwrites"] == 1
    assert older["undoable"] is False
    assert older["undone"] is False


def test_api_undo_roundtrip(client):
    _post_text(client, [(0, 1.0)])
    _post_text(client, [(0, 2.0), (1, 3.0)])

    r = client.post("/api/imports/2/undo")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted_events"] == 1
    assert body["restored_events"] == 1
    assert "evaluation" in body

    events = client.get("/api/events").json()["events"]
    assert [(e["ts"], e["value"]) for e in events] == [(at(0), 1.0)]

    batches = {b["id"]: b for b in client.get("/api/imports").json()}
    assert batches[2]["undone"] is True
    assert batches[2]["undoable"] is False
    assert batches[1]["undoable"] is True  # LIFO: batch 1 is now the latest active


def test_api_undo_rejections(client):
    _post_text(client, [(0, 1.0)])
    _post_text(client, [(1, 2.0)])

    # unknown batch -> 404
    r = client.post("/api/imports/999/undo")
    assert r.status_code == 404

    # not the latest batch -> 409
    r = client.post("/api/imports/1/undo")
    assert r.status_code == 409
    assert "最近一次" in r.json()["detail"]

    # undo the latest, then again -> already undone -> 409
    assert client.post("/api/imports/2/undo").status_code == 200
    r = client.post("/api/imports/2/undo")
    assert r.status_code == 409
    assert "已撤销" in r.json()["detail"]


def test_api_undo_rejects_snapshotless_legacy_batch(client):
    legacy_id = db.execute(
        "INSERT INTO imports (filename, imported_at, total_rows, inserted_rows)"
        " VALUES ('old.jsonl', 1.0, 1, 1)"
    ).lastrowid
    db.execute(
        "INSERT INTO events (device, metric, ts, value, raw, import_id)"
        " VALUES ('d', 'm', ?, 5.0, '{}', ?)",
        (at(0), legacy_id),
    )
    # listed as not undoable with an incomplete snapshot set
    batch = client.get("/api/imports").json()[0]
    assert batch["undoable"] is False
    assert batch["snapshot_complete"] is False
    # and explicitly rejected by the undo endpoint
    r = client.post(f"/api/imports/{legacy_id}/undo")
    assert r.status_code == 409
    assert "快照" in r.json()["detail"]


def test_api_reset_clears_batches_and_snapshots(client):
    _post_text(client, [(0, 1.0)])
    assert len(client.get("/api/imports").json()) == 1
    client.post("/api/reset")
    assert client.get("/api/imports").json() == []
    assert db.query_one("SELECT COUNT(*) AS c FROM import_event_snapshots")["c"] == 0
