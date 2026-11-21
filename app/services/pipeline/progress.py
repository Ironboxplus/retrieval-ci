"""Time-weighted progress and ETA for pipeline tasks on the task board.

`Task.progress/total` counts stages (3/5), so a pipeline parked in its 20-minute
embed stage and one a second from done look the same. Here each stage weighs
its typical duration (median of the most recent completed runs of that stage),
and the running stage contributes its own done/total fraction.

ETA covers work only: the running stage's rate-based remainder plus the typical
duration of every stage after it. A task whose next stage has not started is
waiting on a stage lock (GPU, Qdrant); lock order is not FIFO, so no ETA is
given for it rather than a number that ignores the queue.

The board-level ETA does account for the queue. Every stage holds a
single-lane lock (net, cpu-convert, sql, gpu, qdrant), so all running
pipelines finish roughly when the busiest lane drains its backlog and the last
task clears the stages after it: max over lanes of (backlog + tail).
"""
from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Any, Iterable, Optional

from sqlalchemy.orm import Session

from core.db.models import TaskSubtask, TaskSubtaskStatus

STAGE_TYPE = "pipeline_stage"
_DONE = {"completed", "skipped"}
# Recent runs only: an older run of the same stage may predate a speed-up
# (grouped encoding cut embed from 12.8 to 3.3 min per shard).
TYPICAL_SAMPLE = 5


def _status(sub: Any) -> str:
    st = sub.status
    return st.value if hasattr(st, "value") else str(st or "pending")


def estimate(stages: list[dict], typical: dict[str, float], now: datetime) -> Optional[dict]:
    """`stages` in pipeline order: key, name, status, progress, total, started_at."""
    if not stages:
        return None
    known = [typical[s["key"]] for s in stages if typical.get(s["key"])]
    fallback = median(known) if known else 1.0
    weights = [typical.get(s["key"]) or fallback for s in stages]

    current = next((i for i, s in enumerate(stages) if s["status"] not in _DONE), None)
    if current is None:
        return {"fraction": 1.0, "eta_seconds": 0.0, "stage": None, "waiting": False, "_backlog": {}}

    stage = stages[current]
    running = stage["status"] == "running" and stage.get("started_at") is not None
    elapsed = (now - stage["started_at"]).total_seconds() if running else 0.0
    done, total = stage.get("progress") or 0, stage.get("total") or 0
    typ = typical.get(stage["key"])

    if not running:
        frac = 0.0
    elif total > 0:
        frac = min(done / total, 1.0)
    elif typ:
        frac = min(elapsed / typ, 0.95)
    else:
        frac = 0.0

    earned = sum(weights[:current]) + weights[current] * frac
    fraction = earned / sum(weights)

    rest = stages[current + 1:]
    if running and total > 0 and done > 0:
        remaining: Optional[float] = elapsed * (total - done) / done
    elif typ:
        remaining = max(typ - elapsed, 0.0)
    else:
        remaining = None
    # Work left per stage lane, for the board-level ETA.
    backlog = {stage["key"]: remaining} | {s["key"]: typical.get(s["key"]) for s in rest}

    eta: Optional[float] = None
    if running and remaining is not None and all(typical.get(s["key"]) for s in rest):
        eta = remaining + sum(typical[s["key"]] for s in rest)

    return {
        "fraction": round(fraction, 4),
        "eta_seconds": round(eta) if eta is not None else None,
        "stage": {"key": stage["key"], "name": stage["name"], "status": stage["status"],
                  "progress": done, "total": total},
        "waiting": not running and stage["status"] == "pending",
        "_backlog": backlog,
    }


def board_eta(results: Iterable[dict], order: list[str], typical: dict[str, float]) -> Optional[float]:
    """Seconds until every listed pipeline is done; None if any lane is unknown."""
    lanes: dict[str, float] = {}
    for r in results:
        for key, secs in r["_backlog"].items():
            if secs is None:
                return None
            lanes[key] = lanes.get(key, 0.0) + secs
    if not lanes:
        return None
    ends = []
    for key, backlog in lanes.items():
        after = order[order.index(key) + 1:] if key in order else []
        if any(not typical.get(k) for k in after):
            return None
        ends.append(backlog + sum(typical[k] for k in after))
    return round(max(ends))


def typical_stage_seconds(db: Session, keys: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in set(keys):
        rows = (
            db.query(TaskSubtask.started_at, TaskSubtask.completed_at)
            .filter(
                TaskSubtask.subtask_type == STAGE_TYPE,
                TaskSubtask.subtask_key == key,
                TaskSubtask.status == TaskSubtaskStatus.completed,
                TaskSubtask.started_at.isnot(None),
                TaskSubtask.completed_at.isnot(None),
            )
            .order_by(TaskSubtask.completed_at.desc())
            .limit(TYPICAL_SAMPLE)
            .all()
        )
        spans = [(c - s).total_seconds() for s, c in rows if c >= s]
        if spans:
            out[key] = median(spans)
    return out


def pipeline_progress(db: Session, task_ids: list[int], now: Optional[datetime] = None) -> dict:
    """`{"tasks": {id: progress}, "eta_seconds": board ETA}` for the given
    pipelines. One query for their stages, one small query per stage key."""
    if not task_ids:
        return {"tasks": {}, "eta_seconds": None}
    subs = (
        db.query(TaskSubtask)
        .filter(TaskSubtask.task_id.in_(task_ids), TaskSubtask.subtask_type == STAGE_TYPE)
        .all()
    )
    by_task: dict[int, list[dict]] = {}
    for sub in subs:
        by_task.setdefault(sub.task_id, []).append({
            "key": sub.subtask_key, "name": sub.name or sub.subtask_key, "status": _status(sub),
            "progress": sub.progress or 0, "total": sub.total or 0, "started_at": sub.started_at,
            "order": (sub.metadata_json or {}).get("order", sub.id),
        })
    typical = typical_stage_seconds(db, {s["key"] for ss in by_task.values() for s in ss})
    now = now or datetime.now()  # stage timestamps are server-local, like datetime.now()
    tasks: dict[int, dict] = {}
    order: list[str] = []
    for tid, stages in by_task.items():
        stages.sort(key=lambda s: s["order"])
        if len(stages) > len(order):
            order = [s["key"] for s in stages]
        result = estimate(stages, typical, now)
        if result is not None:
            tasks[tid] = result
    eta = board_eta(tasks.values(), order, typical)
    for r in tasks.values():
        r.pop("_backlog", None)
    return {"tasks": tasks, "eta_seconds": eta}
