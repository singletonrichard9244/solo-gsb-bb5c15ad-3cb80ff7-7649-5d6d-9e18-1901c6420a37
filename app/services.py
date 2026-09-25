"""Import, rule evaluation and timeline services built on the engine + db."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict

from . import db
from .engine import (
    Condition,
    Event,
    Interval,
    MaintenanceWindow,
    cluster_intervals,
    daily_fault_ledger,
    diff_faults,
    evaluate_rule,
    to_epoch,
    to_number,
    union,
)

DEFAULT_DEVICE_FIELD = "device"
DEFAULT_METRIC_FIELD = "metric"
DEFAULT_TS_FIELD = "ts"
DEFAULT_VALUE_FIELD = "value"


def parse_jsonl(text: str) -> list[dict]:
    """Parse JSONL; a top-level JSON array is also accepted."""
    text = text.strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON body must be an array of event objects")
        return [r for r in data if isinstance(r, dict)]

    rows: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"line {lineno}: event must be a JSON object")
        rows.append(obj)
    return rows


def _extract_metrics(
    row: dict, device_field: str, metric_field: str, ts_field: str, value_field: str
) -> list[tuple[str, str, object]]:
    """Turn one raw row into (device, metric, raw_value) triples.

    Two shapes are supported:
    1. flat  - {"device": "d", "metric": "m", "ts": ..., "value": 3.1}
    2. nested - {"device": "d", "ts": ..., "value": {"temp": 1, "vib": 0.2}}
       (one event per key of the value object; the metric column is ignored)
    """
    device = row.get(device_field)
    if device is None:
        raise ValueError("missing device")
    raw_ts = row.get(ts_field)
    if raw_ts is None:
        raise ValueError("missing timestamp")

    value = row.get(value_field) if value_field in row else row.get(DEFAULT_VALUE_FIELD)
    if isinstance(value, dict):
        if not value:
            raise ValueError("empty nested metric object")
        return [(str(device), str(sub_metric), sub_value) for sub_metric, sub_value in value.items()]

    metric = row.get(metric_field)
    if metric is None:
        raise ValueError("missing metric")
    return [(str(device), str(metric), value)]


def import_events(
    text: str,
    filename: str | None = None,
    device_field: str = DEFAULT_DEVICE_FIELD,
    metric_field: str = DEFAULT_METRIC_FIELD,
    ts_field: str = DEFAULT_TS_FIELD,
    value_field: str = DEFAULT_VALUE_FIELD,
    reevaluate: bool = True,
) -> dict:
    rows = parse_jsonl(text)
    now = time.time()

    inserted = 0
    overwritten = 0
    skipped: list[dict] = []
    seen_keys: set[tuple[str, str, float]] = set()
    with db.transaction() as tx:
        cur = tx.execute(
            "INSERT INTO imports (filename, imported_at, total_rows, snapshot_complete)"
            " VALUES (?,?,?,1)",
            (filename, now, len(rows)),
        )
        import_id = cur.lastrowid
        for index, row in enumerate(rows):
            try:
                triples = _extract_metrics(
                    row, device_field, metric_field, ts_field, value_field
                )
                ts = to_epoch(row[ts_field])
            except (ValueError, KeyError, TypeError) as exc:
                skipped.append({"line": index + 1, "reason": str(exc)})
                continue
            for device, metric, raw_value in triples:
                num = to_number(raw_value)
                if num is None:
                    skipped.append(
                        {"line": index + 1, "reason": f"non-numeric value for {device}/{metric}"}
                    )
                    continue
                # Snapshot the pre-import state on this batch's first encounter
                # of the key; later duplicates in the same batch still overwrite
                # (the last valid value wins) but must not re-snapshot.
                db.snapshot_import_key(tx, import_id, seen_keys, device, metric, ts)
                is_new = db.upsert_event(
                    tx, device, metric, ts, num, json.dumps(row, ensure_ascii=False), import_id
                )
                if is_new:
                    inserted += 1
                else:
                    overwritten += 1
        tx.execute(
            "UPDATE imports SET inserted_rows = ?, overwritten_rows = ?, skipped_rows = ?"
            " WHERE id = ?",
            (inserted, overwritten, len(skipped), import_id),
        )

    result = {
        "import_id": import_id,
        "total_rows": len(rows),
        "inserted": inserted,
        "overwritten": overwritten,
        "skipped": len(skipped),
        "skipped_details": skipped[:100],
    }
    # New or back-filled samples — including older timestamps — change held
    # intervals and can bridge or open TTL gaps, so stored faults are
    # recomputed immediately (maintenance windows keep participating).
    if reevaluate:
        evaluation = evaluate_all_rules()
        result["evaluation"] = {
            "rules_evaluated": evaluation["rules_evaluated"],
            "fault_intervals": evaluation["fault_intervals"],
            "suppressed_segments": evaluation["suppressed_segments"],
        }
    return result


class UndoNotAllowed(Exception):
    """Raised when an import batch cannot be revoked (409 at the API)."""


def list_import_batches() -> list[dict]:
    """All import batches, newest first, with their undo eligibility."""
    latest = db.query_one(
        "SELECT MAX(id) AS id FROM imports WHERE revoked_at IS NULL"
    )
    latest_id = latest["id"] if latest is not None else None
    rows = db.query(
        """
        SELECT i.*,
               (SELECT COUNT(*) FROM import_snapshots s WHERE s.import_id = i.id) AS snapshot_rows,
               (SELECT SUM(CASE WHEN s.event_id IS NOT NULL THEN 1 ELSE 0 END)
                  FROM import_snapshots s WHERE s.import_id = i.id) AS overwritten_keys
        FROM imports i
        ORDER BY i.id DESC
        """
    )
    batches: list[dict] = []
    for row in rows:
        batch = {
            "id": row["id"],
            "filename": row["filename"],
            "imported_at": row["imported_at"],
            "total_rows": row["total_rows"],
            "inserted_rows": row["inserted_rows"],
            "overwritten_rows": row["overwritten_rows"],
            "skipped_rows": row["skipped_rows"],
            "snapshot_keys": row["snapshot_rows"],
            "overwritten_keys": row["overwritten_keys"] or 0,
            "snapshot_complete": bool(row["snapshot_complete"]),
            "revoked": row["revoked_at"] is not None,
            "revoked_at": row["revoked_at"],
            "can_undo": False,
            "undo_reason": None,
        }
        if batch["revoked"]:
            batch["undo_reason"] = "该批次已撤销"
        elif not batch["snapshot_complete"]:
            # Batch imported before the undo feature: nothing to restore from.
            batch["undo_reason"] = "旧批次缺少导入前快照，无法撤销"
        elif row["id"] != latest_id:
            batch["undo_reason"] = "仅允许撤销最近一次未撤销的导入批次"
        else:
            batch["can_undo"] = True
        batches.append(batch)
    return batches


def undo_import(import_id: int, reevaluate: bool = True) -> dict:
    """Revoke one import batch atomically.

    Only the most recent non-revoked batch whose pre-import snapshots are
    complete may be revoked.  In a single transaction the batch's newly
    inserted events are deleted, while events it overwritten get back their
    previous value, raw content and original batch ownership; the batch is
    then marked revoked.  Any failure rolls the whole operation back.
    """
    row = db.query_one("SELECT * FROM imports WHERE id = ?", (import_id,))
    if row is None:
        raise KeyError(f"import batch {import_id} not found")
    if row["revoked_at"] is not None:
        raise UndoNotAllowed("该导入批次已撤销，不能重复撤销")
    if not row["snapshot_complete"]:
        raise UndoNotAllowed("该批次导入时未记录快照（旧数据），无法撤销")
    latest = db.query_one(
        "SELECT MAX(id) AS id FROM imports WHERE revoked_at IS NULL"
    )
    if latest is None or latest["id"] != import_id:
        raise UndoNotAllowed("仅允许撤销最近一次导入批次，请先撤销其后的导入")

    with db.transaction() as tx:
        snapshots = tx.execute(
            "SELECT * FROM import_snapshots WHERE import_id = ?"
            " ORDER BY id",
            (import_id,),
        ).fetchall()
        deleted = 0
        restored = 0
        for snap in snapshots:
            if snap["event_id"] is None:
                # The batch created this key: delete the current event, but
                # only if it still belongs to the batch being revoked.
                cur = tx.execute(
                    "DELETE FROM events"
                    " WHERE device = ? AND metric = ? AND ts = ? AND import_id = ?",
                    (snap["device"], snap["metric"], snap["ts"], import_id),
                )
                if cur.rowcount == 0:
                    raise RuntimeError(
                        f"snapshot consistency broken for new key"
                        f" {snap['device']}/{snap['metric']}@{snap['ts']}"
                    )
                deleted += cur.rowcount
            else:
                # The batch overwrote a pre-existing event: put its value,
                # raw content and original batch ownership back.
                cur = tx.execute(
                    "UPDATE events SET value = ?, raw = ?, import_id = ?"
                    " WHERE id = ? AND import_id = ?",
                    (
                        snap["value"],
                        snap["raw"],
                        snap["origin_import_id"],
                        snap["event_id"],
                        import_id,
                    ),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(
                        f"snapshot consistency broken for event {snap['event_id']}"
                    )
                restored += 1
        tx.execute(
            "UPDATE imports SET revoked_at = ? WHERE id = ?",
            (time.time(), import_id),
        )

    result = {
        "import_id": import_id,
        "deleted_events": deleted,
        "restored_events": restored,
    }
    # Faults derive from the current events, so the existing evaluation flow
    # refreshes them after the data is back to its pre-batch state.
    if reevaluate:
        evaluation = evaluate_all_rules()
        result["evaluation"] = {
            "rules_evaluated": evaluation["rules_evaluated"],
            "fault_intervals": evaluation["fault_intervals"],
            "suppressed_segments": evaluation["suppressed_segments"],
        }
    return result


def _load_events_index() -> tuple[dict[tuple[str, str], list[Event]], float | None]:
    events_by_channel: dict[tuple[str, str], list[Event]] = defaultdict(list)
    horizon: float | None = None
    for row in db.query("SELECT id, device, metric, ts, value FROM events ORDER BY ts, id"):
        if row["value"] is None:
            continue
        ev = Event(
            id=row["id"],
            device=row["device"],
            metric=row["metric"],
            ts=row["ts"],
            value=row["value"],
        )
        events_by_channel[(ev.device, ev.metric)].append(ev)
        horizon = ev.ts if horizon is None else max(horizon, ev.ts)
    return events_by_channel, horizon


def _load_maintenance_windows() -> list[MaintenanceWindow]:
    """All stored planned-maintenance windows as engine inputs."""
    return [
        MaintenanceWindow(
            id=w["id"],
            device=w["device"],
            start=w["start"],
            end=w["end"],
            name=w["name"],
        )
        for w in db.fetch_maintenance_windows()
    ]


def _load_channel_ttls() -> dict[tuple[str, str], float]:
    """Configured per-channel sample validity periods as engine input.

    Only channels with a positive configured TTL appear here; every other
    channel keeps the legacy hold-to-next-sample / hold-to-horizon semantics.
    """
    return {
        (row["device"], row["metric"]): row["ttl_seconds"]
        for row in db.fetch_channel_ttls()
    }


def evaluate_all_rules() -> dict:
    """Re-evaluate every rule and replace its stored fault intervals.

    Planned-maintenance windows are merged per device and subtracted from the
    truth segments before the duration check, so a maintenance interruption
    resets the sustained duration and re-arms confirmation after the window.
    Per-channel sample validity periods cap each sample's held interval, so
    expired stretches satisfy no condition and likewise reset the duration.
    The suppressed pieces are stored alongside for the timeline/export.
    """
    rules = db.fetch_rules()
    events_by_channel, horizon = _load_events_index()
    if horizon is None:
        horizon = 0.0
    maintenance = _load_maintenance_windows()
    ttl_by_channel = _load_channel_ttls()

    total = 0
    total_suppressed = 0
    per_rule: list[dict] = []
    with db.transaction() as tx:
        for rule in rules:
            conditions = [Condition.from_dict(c) for c in rule["conditions"]]
            result = evaluate_rule(
                rule_id=rule["id"],
                logic=rule["logic"],
                duration_required=rule["duration"],
                conditions=conditions,
                events_by_channel=events_by_channel,
                horizon=horizon,
                maintenance=maintenance,
                ttl_by_channel=ttl_by_channel,
            )
            tx.execute("DELETE FROM fault_intervals WHERE rule_id = ?", (rule["id"],))
            tx.execute("DELETE FROM suppressed_segments WHERE rule_id = ?", (rule["id"],))
            for f in result.faults:
                tx.execute(
                    """
                    INSERT INTO fault_intervals
                        (rule_id, start_ts, end_ts, duration_required, confirmed_at, evidence_json)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        f.rule_id,
                        f.start,
                        f.end,
                        f.duration_required,
                        f.confirmed_at,
                        json.dumps([e.to_dict() for e in f.evidence], ensure_ascii=False),
                    ),
                )
            for s in result.suppressed:
                tx.execute(
                    """
                    INSERT INTO suppressed_segments
                        (rule_id, device, metric, start_ts, end_ts, maintenance_ids)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        rule["id"],
                        s.device,
                        s.metric,
                        s.interval.start,
                        s.interval.end,
                        json.dumps(s.maintenance_ids),
                    ),
                )
            total += len(result.faults)
            total_suppressed += len(result.suppressed)
            per_rule.append(
                {
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "faults": len(result.faults),
                    "suppressed": len(result.suppressed),
                }
            )

    return {
        "rules_evaluated": len(rules),
        "fault_intervals": total,
        "suppressed_segments": total_suppressed,
        "maintenance_windows": len(maintenance),
        "per_rule": per_rule,
    }


def _fault_outline(fault) -> dict:
    """Compact fault payload for the impact preview (evidence is irrelevant)."""
    return {
        "rule_id": fault.rule_id,
        "start": fault.start,
        "end": fault.end,
        "duration_required": fault.duration_required,
        "confirmed_at": fault.confirmed_at,
    }


def preview_rule(rule_id: int | None, draft: dict) -> dict:
    """Evaluate a not-yet-saved rule draft against the live data, read-only.

    The baseline is the rule *as currently stored* (or nothing for a new
    rule), the candidate is the unsaved draft.  Both are evaluated right now
    with the current events, the currently stored planned-maintenance windows
    and the configured per-channel sample validity periods — exactly what a
    save + re-evaluation would see — but nothing is written to the rules or
    fault tables.  The two confirmed fault sets are compared as half-open
    interval coverings (see :func:`diff_faults`).
    """
    events_by_channel, horizon = _load_events_index()
    if horizon is None:
        horizon = 0.0
    maintenance = _load_maintenance_windows()
    ttl_by_channel = _load_channel_ttls()

    baseline_rule = db.fetch_rule(rule_id) if rule_id is not None else None
    if rule_id is not None and baseline_rule is None:
        raise KeyError(f"rule {rule_id} not found")

    def run(rule: dict, rid: int):
        return evaluate_rule(
            rule_id=rid,
            logic=rule["logic"],
            duration_required=rule["duration"],
            conditions=[Condition.from_dict(c) for c in rule["conditions"]],
            events_by_channel=events_by_channel,
            horizon=horizon,
            maintenance=maintenance,
            ttl_by_channel=ttl_by_channel,
        )

    baseline_result = (
        run(baseline_rule, baseline_rule["id"]) if baseline_rule is not None else None
    )
    draft_result = run(draft, rule_id if rule_id is not None else -1)

    baseline_faults = baseline_result.faults if baseline_result is not None else []
    segments = diff_faults(baseline_faults, draft_result.faults)
    counts = {"added": 0, "removed": 0, "overlap_changed": 0, "overlap_unchanged": 0}
    for seg in segments:
        counts[seg.kind] += 1

    return {
        "rule_id": rule_id,
        "mode": "edit" if baseline_rule is not None else "create",
        "baseline": {
            "name": baseline_rule["name"] if baseline_rule is not None else None,
            "faults": [_fault_outline(f) for f in baseline_faults],
        },
        "draft": {
            "name": draft["name"],
            "faults": [_fault_outline(f) for f in draft_result.faults],
        },
        "maintenance_windows": len(maintenance),
        "channel_ttls": len(ttl_by_channel),
        "diff": {
            "segments": [seg.to_dict() for seg in segments],
            "counts": counts,
            "changed": counts["added"] + counts["removed"] + counts["overlap_changed"] > 0,
        },
    }


def list_channels() -> list[dict]:
    rows = db.query(
        """
        SELECT device, metric, COUNT(*) AS n, MIN(ts) AS min_ts, MAX(ts) AS max_ts,
               MIN(value) AS min_value, MAX(value) AS max_value
        FROM events WHERE value IS NOT NULL
        GROUP BY device, metric ORDER BY device, metric
        """
    )
    return [dict(r) for r in rows]


def list_channel_ttls() -> list[dict]:
    """Configured per-channel sample validity periods (positive seconds).

    Channels absent from the list have no TTL and keep legacy hold semantics.
    """
    return db.fetch_channel_ttls()


def set_channel_ttl(device: str, metric: str, ttl_seconds: float) -> dict:
    """Set/replace a channel's sample validity period and re-evaluate rules.

    Changing the TTL changes how every rule using the channel expands its
    samples, so the stored fault intervals are recomputed immediately.
    Planned maintenance keeps participating in the evaluation as usual.
    """
    ttl_seconds = float(ttl_seconds)
    if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be a positive number of seconds")
    saved = db.upsert_channel_ttl(device, metric, ttl_seconds)
    evaluation = evaluate_all_rules()
    return {
        **saved,
        "evaluation": {
            "rules_evaluated": evaluation["rules_evaluated"],
            "fault_intervals": evaluation["fault_intervals"],
            "suppressed_segments": evaluation["suppressed_segments"],
        },
    }


def clear_channel_ttl(device: str, metric: str) -> dict:
    """Remove a channel's validity period (back to legacy semantics) and re-evaluate."""
    existed = db.fetch_channel_ttl(device, metric)
    removed = db.delete_channel_ttl(device, metric)
    if not removed:
        raise KeyError(f"no sample validity configured for channel {device}/{metric}")
    evaluation = evaluate_all_rules()
    return {
        "cleared": {"device": device, "metric": metric, "had_ttl_seconds": existed["ttl_seconds"]},
        "evaluation": {
            "rules_evaluated": evaluation["rules_evaluated"],
            "fault_intervals": evaluation["fault_intervals"],
            "suppressed_segments": evaluation["suppressed_segments"],
        },
    }


def list_events(
    start: float | None = None,
    end: float | None = None,
    device: str | None = None,
    metric: str | None = None,
    limit: int = 2000,
    offset: int = 0,
) -> tuple[list[dict], int]:
    where = []
    params: list = []
    # Point events: inclusive range on both ends.
    if start is not None:
        where.append("ts >= ?")
        params.append(start)
    if end is not None:
        where.append("ts <= ?")
        params.append(end)
    if device:
        where.append("device = ?")
        params.append(device)
    if metric:
        where.append("metric = ?")
        params.append(metric)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    total = db.query_one(f"SELECT COUNT(*) AS c FROM events{clause}", params)["c"]
    rows = db.query(
        f"SELECT id, device, metric, ts, value, import_id FROM events{clause} ORDER BY ts, id LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )
    return [dict(r) for r in rows], total


def query_faults(
    start: float | None = None, end: float | None = None
) -> list[dict]:
    where = []
    params: list = []
    if start is not None:
        where.append("end_ts > ?")  # half-open overlap: [start,end) ∩ range
        params.append(start)
    if end is not None:
        where.append("start_ts < ?")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.query(
        f"""
        SELECT f.*, r.name AS rule_name, r.logic
        FROM fault_intervals f JOIN rules r ON r.id = f.rule_id
        {clause} ORDER BY f.start_ts, f.rule_id
        """,
        params,
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


def _overlaps_window(start: float, end: float, win_start: float | None, win_end: float | None) -> bool:
    """Half-open overlap with the filter window [win_start, win_end)."""
    if win_start is not None and end <= win_start:
        return False
    if win_end is not None and start >= win_end:
        return False
    return True


def _fault_member(fault: dict, in_window: bool) -> dict:
    """One member fault as embedded inside a fault event.

    The payload is a copy of the original stored fault (id, interval and
    trigger evidence untouched) plus the window-membership flag; the event
    view never rewrites fault results.
    """
    return {
        "fault_id": fault["id"],
        "rule_id": fault["rule_id"],
        "rule_name": fault["rule_name"],
        "logic": fault["logic"],
        "start": fault["start"],
        "end": fault["end"],
        "duration_required": fault["duration_required"],
        "confirmed_at": fault["confirmed_at"],
        "in_window": in_window,
        "evidence": fault["evidence"],
    }


def build_fault_events(
    faults: list[dict], start: float | None = None, end: float | None = None
) -> list[dict]:
    """Group faults from different rules into fault *events*.

    Two faults (typically produced by different rules) belong to the same
    event when their ``[start,end)`` intervals overlap or touch end-to-end;
    the relation is transitive, so chaining links are merged as well.  An
    event spans ``[min member start, max member end)``.

    Components are first computed over *all* faults (so chains crossing the
    filter window stay intact), then only events overlapping the
    half-open filter window are returned.  Members outside the window are
    still listed, flagged ``in_window: false``.
    """
    groups = cluster_intervals([Interval(f["start"], f["end"]) for f in faults])
    events: list[dict] = []
    for members in groups:
        member_faults = [faults[i] for i in members]
        ev_start = min(f["start"] for f in member_faults)
        ev_end = max(f["end"] for f in member_faults)
        if not _overlaps_window(ev_start, ev_end, start, end):
            continue
        member_payload = [
            _fault_member(f, _overlaps_window(f["start"], f["end"], start, end))
            for f in sorted(member_faults, key=lambda f: (f["start"], f["end"], f["id"]))
        ]
        rule_ids = sorted({f["rule_id"] for f in member_faults})
        events.append(
            {
                "start": ev_start,
                "end": ev_end,
                "rule_count": len(rule_ids),
                "fault_count": len(member_payload),
                "member_count_in_window": sum(1 for m in member_payload if m["in_window"]),
                "rule_ids": rule_ids,
                "rule_names": sorted({f["rule_name"] for f in member_faults}),
                "members": member_payload,
            }
        )
    events.sort(key=lambda e: (e["start"], e["end"]))
    # Sequential display id, stable within the queried time range.
    for index, event in enumerate(events, start=1):
        event["id"] = f"E{index}"
    return events


def query_fault_events(
    start: float | None = None, end: float | None = None
) -> list[dict]:
    """Fault events for the filter window; clustering sees every stored fault."""
    return build_fault_events(query_faults(None, None), start, end)


def query_fault_daily_ledger(
    rule_ids: list[int], start: float, end: float
) -> dict:
    """Read-only per-day fault ledger for a combination of selected rules.

    The *already stored* confirmed fault intervals of the given rules are
    read verbatim (rule changes keep being recomputed by the existing
    evaluation flow), merged across rules as half-open intervals —
    overlapping or touching pieces become one — then split at UTC midnights
    within the half-open query window ``[start, end)``.  See
    :func:`app.engine.daily_fault_ledger` for the per-day metrics.

    Raises ``ValueError`` on an empty/inverted window and ``KeyError`` for
    unknown rule ids.
    """
    start = float(start)
    end = float(end)
    if end <= start:
        raise ValueError("ledger end must be greater than start")
    if not rule_ids:
        raise ValueError("at least one rule is required")

    seen: set[int] = set()
    unique_ids: list[int] = []
    for rid in rule_ids:
        rid = int(rid)
        if rid not in seen:
            seen.add(rid)
            unique_ids.append(rid)

    selected_rules: list[dict] = []
    missing: list[int] = []
    for rid in unique_ids:
        rule = db.fetch_rule(rid)
        if rule is None:
            missing.append(rid)
        else:
            selected_rules.append(rule)
    if missing:
        raise KeyError(f"rule(s) not found: {', '.join(str(i) for i in missing)}")

    faults = [
        f
        for f in query_faults(None, None)
        if f["rule_id"] in seen and f["end"] > start and f["start"] < end
    ]
    intervals = [Interval(f["start"], f["end"]) for f in faults]
    merged = [iv for iv in union(intervals) if iv.end > start and iv.start < end]
    days = daily_fault_ledger(intervals, start, end)

    total_fault_seconds = sum(d["fault_seconds"] for d in days)
    total_queried_seconds = sum(d["queried_seconds"] for d in days)
    return {
        "start": start,
        "end": end,
        "rules": [
            {"id": r["id"], "name": r["name"], "logic": r["logic"], "duration": r["duration"]}
            for r in selected_rules
        ],
        "merged_segments": [[iv.start, iv.end] for iv in merged],
        "days": days,
        "totals": {
            "fault_seconds": total_fault_seconds,
            "queried_seconds": total_queried_seconds,
            "new_segments": sum(d["new_segments"] for d in days),
            "continuing_segments": sum(d["continuing_segments"] for d in days),
            "merged_segments": len(merged),
            "coverage": (total_fault_seconds / total_queried_seconds) if total_queried_seconds else 0.0,
        },
    }


def query_suppressed(
    start: float | None = None, end: float | None = None
) -> list[dict]:
    """Suppressed segments (truth lost to maintenance), half-open filtered.

    Each segment is linked to its rule and to the maintenance windows that
    suppressed it (ids resolved to names for display/export).
    """
    where = []
    params: list = []
    if start is not None:
        where.append("s.end_ts > ?")
        params.append(start)
    if end is not None:
        where.append("s.start_ts < ?")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.query(
        f"""
        SELECT s.*, r.name AS rule_name
        FROM suppressed_segments s JOIN rules r ON r.id = s.rule_id
        {clause} ORDER BY s.start_ts, s.rule_id, s.id
        """,
        params,
    )
    windows = {w["id"]: w for w in db.fetch_maintenance_windows()}
    out = []
    for row in rows:
        linked = []
        for wid in json.loads(row["maintenance_ids"]):
            w = windows.get(wid)
            linked.append(
                {"id": wid, "name": w["name"], "device": w["device"]}
                if w
                else {"id": wid, "name": None, "device": row["device"]}
            )
        out.append(
            {
                "id": row["id"],
                "rule_id": row["rule_id"],
                "rule_name": row["rule_name"],
                "device": row["device"],
                "metric": row["metric"],
                "start": row["start_ts"],
                "end": row["end_ts"],
                "maintenance_ids": json.loads(row["maintenance_ids"]),
                "maintenance_windows": linked,
            }
        )
    return out


def query_annotations(start: float | None = None, end: float | None = None) -> list[dict]:
    where = []
    params: list = []
    if start is not None:
        where.append("end_ts > ?")
        params.append(start)
    if end is not None:
        where.append("start_ts < ?")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.query(
        f"SELECT * FROM annotations{clause} ORDER BY start_ts, id", params
    )
    return [
        {
            "id": r["id"],
            "start": r["start_ts"],
            "end": r["end_ts"],
            "label": r["label"],
            "color": r["color"],
            "note": r["note"],
        }
        for r in rows
    ]


def downsample(values: list[tuple[float, float, int]], max_points: int) -> list[tuple[float, float, int]]:
    """Stride downsample; first/last points are always kept.

    values: (ts, value, event_id) sorted by ts.  A stride (rather than bucket
    averaging) preserves the step-plot shape of held signals.
    """
    if len(values) <= max_points or max_points < 2:
        return values
    step = len(values) / max_points
    picked = [values[0]]
    index = step
    while index < len(values) - 1:
        picked.append(values[int(index)])
        index += step
    picked.append(values[-1])
    return picked


def timeline_data(start: float | None, end: float | None, max_points: int = 400) -> dict:
    channels = list_channels()
    channels_in_range = []
    for ch in channels:
        if start is not None and ch["max_ts"] < start:
            continue
        if end is not None and ch["min_ts"] > end:
            continue
        channels_in_range.append(ch)

    channel_payload = []
    for ch in channels_in_range:
        sql = (
            "SELECT ts, value, id FROM events WHERE device = ? AND metric = ? AND value IS NOT NULL"
        )
        params: list = [ch["device"], ch["metric"]]
        if start is not None:
            sql += " AND ts >= ?"
            params.append(start)
        if end is not None:
            sql += " AND ts <= ?"
            params.append(end)
        sql += " ORDER BY ts, id"
        rows = db.query(sql, params)
        values = [(r["ts"], r["value"], r["id"]) for r in rows]
        total = len(values)
        values = downsample(values, max_points)
        channel_payload.append(
            {
                "device": ch["device"],
                "metric": ch["metric"],
                "count": ch["n"],
                "shown": total,
                "min_ts": ch["min_ts"],
                "max_ts": ch["max_ts"],
                "min_value": ch["min_value"],
                "max_value": ch["max_value"],
                "values": [{"ts": t, "v": v, "id": i} for t, v, i in values],
            }
        )

    bounds = db.query_one("SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM events")
    return {
        "channels": channel_payload,
        "channel_ttls": db.fetch_channel_ttls(),
        "faults": query_faults(start, end),
        "annotations": query_annotations(start, end),
        "maintenance": db.fetch_maintenance_windows(start, end),
        "suppressed": query_suppressed(start, end),
        "data_start": bounds["lo"],
        "data_end": bounds["hi"],
    }
