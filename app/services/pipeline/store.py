"""TaskStore — generic DB I/O over Task / TaskSubtask / TaskEvent.

This is the ONLY database boundary for the pipeline engine. It has zero
pipeline/stage knowledge: it just reads/writes the existing task tables. The
orchestrator talks to TaskStore in plain strings ("running"/"completed"/...);
TaskStore maps them to the SQLAlchemy enums. A session_factory is injected so
tests can bind a private in-memory DB; production uses get_db_session.

Each method opens its own short-lived session (the factory commits on exit), so
callers never hold a session across a blocking stage run.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger
from sqlalchemy import or_

from core.db.database import get_db_session
from core.db.models import (
    Task,
    TaskStatus,
    TaskSubtask,
    TaskSubtaskStatus,
    TaskEvent,
)

_TASK_STATUS = {s.value: s for s in TaskStatus}
_SUB_STATUS = {s.value: s for s in TaskSubtaskStatus}

_TERMINAL = {"completed", "failed", "cancelled", "timeout"}


class TaskStore:
    def __init__(self, session_factory: Callable[[], Any] = get_db_session):
        self._sf = session_factory

    # ------------------------------------------------------------------ task
    def create_task(
        self,
        *,
        task_type: str,
        params: Optional[dict] = None,
        total: int = 0,
        owner_user_id: Optional[str] = None,
    ) -> int:
        with self._sf() as s:
            t = Task(
                task_type=task_type,
                status=TaskStatus.pending,
                params_json=params or {},
                total=total,
                progress=0,
                owner_user_id=owner_user_id,
            )
            s.add(t)
            s.flush()
            return t.id

    def get_task(self, task_id: int) -> Optional[dict]:
        with self._sf() as s:
            t = s.get(Task, task_id)
            if t is None:
                return None
            return {
                "id": t.id,
                "task_type": t.task_type,
                "status": t.status.value if t.status else None,
                "progress": t.progress or 0,
                "total": t.total or 0,
                "params": t.params_json or {},
                "result": t.result_json,
                "error": t.error_message,
                "owner_user_id": t.owner_user_id,
                "worker_id": t.worker_id,
                "created_at": t.created_at.timestamp() if t.created_at else 0.0,
                "started_at": t.started_at.timestamp() if t.started_at else None,
                "finished_at": t.completed_at.timestamp() if t.completed_at else None,
            }

    def list_running(self, task_type: str) -> list[dict]:
        """All tasks of a type currently in `running` status (for conflict guard
        + startup resume). Returns lightweight dicts (id + params)."""
        with self._sf() as s:
            rows = (
                s.query(Task)
                .filter(Task.task_type == task_type, Task.status == TaskStatus.running)
                .order_by(Task.id.asc())
                .all()
            )
            return [{"id": t.id, "params": dict(t.params_json or {})} for t in rows]

    def list_active(self, task_type: str) -> list[dict]:
        """Pending/running durable work, ordered exactly as it was submitted."""
        with self._sf() as s:
            rows = (
                s.query(Task)
                .filter(
                    Task.task_type == task_type,
                    Task.status.in_((TaskStatus.pending, TaskStatus.running)),
                )
                .order_by(Task.id.asc())
                .all()
            )
            return [
                {
                    "id": task.id,
                    "status": task.status.value,
                    "params": dict(task.params_json or {}),
                }
                for task in rows
            ]

    def set_task_status(self, task_id: int, status: str, *, error: Optional[str] = None) -> None:
        with self._sf() as s:
            t = s.get(Task, task_id)
            if t is None:
                return
            t.status = _TASK_STATUS[status]
            now = datetime.now()
            if status == "running" and t.started_at is None:
                t.started_at = now
            if status in _TERMINAL:
                t.completed_at = now
            if error is not None:
                t.error_message = error
            t.last_heartbeat = now

    def compare_and_set_status(
        self,
        task_id: int,
        *,
        expect: set[str],
        new: str,
        error: Optional[str] = None,
    ) -> bool:
        """Move a task to `new` ONLY while it is still in one of `expect`.

        Returns True if this caller performed the transition, False if someone
        else already moved it (most importantly: cancelled it).

        `set_task_status` reads then writes through the ORM, so a cancel landing
        between the two silently loses: the writer overwrites `cancelled` with
        `running` and the job keeps going. This is a single conditional UPDATE and
        the rowcount tells the caller whether it won — the only safe way to start,
        finish or cancel durable work that another process can also touch.
        """
        with self._sf() as s:
            now = datetime.now()
            values: dict[str, Any] = {"status": _TASK_STATUS[new], "last_heartbeat": now}
            if new in _TERMINAL:
                values["completed_at"] = now
            if error is not None:
                values["error_message"] = error
            changed = (
                s.query(Task)
                .filter(
                    Task.id == task_id,
                    Task.status.in_([_TASK_STATUS[e] for e in expect]),
                )
                .update(values, synchronize_session=False)
            )
            if changed and new == "running":
                # started_at is first-write-wins; a resumed task keeps its original start.
                s.query(Task).filter(
                    Task.id == task_id, Task.started_at.is_(None),
                ).update({"started_at": now}, synchronize_session=False)
            return bool(changed)

    def claim_task(self, task_id: int, worker_id: str, *, lease_seconds: int = 120) -> bool:
        """Take cross-process ownership of one task, or return False.

        An in-process `set` of active ids cannot see another backend process. Two
        processes both running `resume_active()` at startup therefore both launch
        the same row, doubling GPU work and racing each other's writes. The lease
        lives in the columns the table already has (`worker_id` + `last_heartbeat`),
        so this needs no schema change: a row is claimable when nobody owns it, when
        we already own it (restart re-entrancy), or when the previous owner's
        heartbeat is older than the lease and is presumed dead.
        """
        with self._sf() as s:
            cutoff = datetime.fromtimestamp(datetime.now().timestamp() - lease_seconds)
            changed = (
                s.query(Task)
                .filter(
                    Task.id == task_id,
                    Task.status.in_((TaskStatus.pending, TaskStatus.running)),
                    or_(
                        Task.worker_id.is_(None),
                        Task.worker_id == worker_id,
                        Task.last_heartbeat.is_(None),
                        Task.last_heartbeat < cutoff,
                    ),
                )
                .update(
                    {"worker_id": worker_id, "last_heartbeat": datetime.now()},
                    synchronize_session=False,
                )
            )
            return bool(changed)

    def renew_lease(self, task_id: int, worker_id: str) -> bool:
        """Refresh our lease. False means we lost it — the caller must stop."""
        with self._sf() as s:
            changed = (
                s.query(Task)
                .filter(Task.id == task_id, Task.worker_id == worker_id)
                .update({"last_heartbeat": datetime.now()}, synchronize_session=False)
            )
            return bool(changed)

    def release_claim(self, task_id: int, worker_id: str) -> None:
        with self._sf() as s:
            s.query(Task).filter(
                Task.id == task_id, Task.worker_id == worker_id,
            ).update({"worker_id": None}, synchronize_session=False)

    def set_task_progress(self, task_id: int, progress: int, total: int) -> None:
        with self._sf() as s:
            t = s.get(Task, task_id)
            if t is None:
                return
            t.progress = progress
            t.total = total

    def set_task_result(self, task_id: int, result: dict) -> None:
        with self._sf() as s:
            task = s.get(Task, task_id)
            if task is None:
                return
            task.result_json = result
            task.status = TaskStatus.completed
            task.progress = task.total or 100
            task.completed_at = datetime.now()
            task.last_heartbeat = task.completed_at

    def publish_result_if_running(self, task_id: int, result: dict, worker_id: str) -> bool:
        """Publish a final result ONLY while we still own a running task.

        `set_task_result` flips to `completed` unconditionally, so a cancel that
        lands after the worker's last cancel check still ends up labelled
        `completed` with a result attached — which plan.md forbids ("任何 partial
        result 都不得标为 completed"). Conditioning the UPDATE on
        status=running AND worker_id=us makes a lost race a no-op instead.
        """
        with self._sf() as s:
            now = datetime.now()
            changed = (
                s.query(Task)
                .filter(
                    Task.id == task_id,
                    Task.status == TaskStatus.running,
                    Task.worker_id == worker_id,
                )
                .update(
                    {
                        "result_json": result,
                        "status": TaskStatus.completed,
                        "completed_at": now,
                        "last_heartbeat": now,
                    },
                    synchronize_session=False,
                )
            )
            if changed:
                s.query(Task).filter(Task.id == task_id).update(
                    {"progress": Task.total}, synchronize_session=False,
                )
            return bool(changed)

    def heartbeat(self, task_id: int) -> None:
        with self._sf() as s:
            t = s.get(Task, task_id)
            if t is not None:
                t.last_heartbeat = datetime.now()

    def get_state(self, task_id: int) -> dict:
        """Shared pipeline scratch state (chunk-id watermarks, shard ranges)
        lives in task.params_json['state'] — generic JSON, no schema change."""
        with self._sf() as s:
            t = s.get(Task, task_id)
            if t is None:
                return {}
            return dict((t.params_json or {}).get("state") or {})

    def update_state(self, task_id: int, **kv: Any) -> None:
        with self._sf() as s:
            t = s.get(Task, task_id)
            if t is None:
                return
            params = dict(t.params_json or {})
            state = dict(params.get("state") or {})
            state.update(kv)
            params["state"] = state
            t.params_json = params

    # --------------------------------------------------------------- subtask
    def ensure_subtask(
        self,
        task_id: int,
        key: str,
        *,
        subtask_type: str,
        name: str,
        order: int = 0,
        total: int = 0,
    ) -> int:
        """Idempotent on (task_id, key) — create on first run, reuse on resume."""
        with self._sf() as s:
            sub = (
                s.query(TaskSubtask)
                .filter_by(task_id=task_id, subtask_key=key)
                .first()
            )
            if sub is None:
                sub = TaskSubtask(
                    task_id=task_id,
                    subtask_key=key,
                    subtask_type=subtask_type,
                    name=name,
                    status=TaskSubtaskStatus.pending,
                    total=total,
                    metadata_json={"order": order},
                )
                s.add(sub)
                s.flush()
            return sub.id

    def get_subtask(self, task_id: int, key: str) -> Optional[dict]:
        with self._sf() as s:
            sub = (
                s.query(TaskSubtask)
                .filter_by(task_id=task_id, subtask_key=key)
                .first()
            )
            if sub is None:
                return None
            return {
                "id": sub.id,
                "key": sub.subtask_key,
                "status": sub.status.value if sub.status else None,
                "progress": sub.progress or 0,
                "total": sub.total or 0,
                "message": sub.message,
                "stage": sub.stage,
                "started_at": sub.started_at,
                "completed_at": sub.completed_at,
            }

    def start_subtask(self, task_id: int, key: str, *, message: Optional[str] = None, stage: Optional[str] = None) -> None:
        with self._sf() as s:
            sub = s.query(TaskSubtask).filter_by(task_id=task_id, subtask_key=key).first()
            if sub is None:
                return
            sub.status = TaskSubtaskStatus.running
            sub.started_at = datetime.now()
            if message:
                sub.message = message
            if stage:
                sub.stage = stage

    def update_progress(self, task_id: int, key: str, done: int, total: int, message: str = "") -> None:
        with self._sf() as s:
            sub = s.query(TaskSubtask).filter_by(task_id=task_id, subtask_key=key).first()
            if sub is None:
                return
            sub.progress = done
            sub.total = total
            if message:
                sub.message = message

    def complete_subtask(self, task_id: int, key: str, *, message: Optional[str] = None, result: Optional[dict] = None) -> None:
        with self._sf() as s:
            sub = s.query(TaskSubtask).filter_by(task_id=task_id, subtask_key=key).first()
            if sub is None:
                return
            sub.status = TaskSubtaskStatus.completed
            sub.completed_at = datetime.now()
            if sub.total and (sub.progress or 0) < sub.total:
                sub.progress = sub.total
            if message:
                sub.message = message
            if result is not None:
                sub.result_json = result

    def fail_subtask(self, task_id: int, key: str, reason: str) -> None:
        with self._sf() as s:
            sub = s.query(TaskSubtask).filter_by(task_id=task_id, subtask_key=key).first()
            if sub is None:
                return
            sub.status = TaskSubtaskStatus.failed
            sub.completed_at = datetime.now()
            sub.error_message = reason
            sub.message = reason

    def cancel_subtask(self, task_id: int, key: str, reason: str = "cancelled") -> None:
        with self._sf() as s:
            sub = s.query(TaskSubtask).filter_by(task_id=task_id, subtask_key=key).first()
            if sub is None:
                return
            sub.status = TaskSubtaskStatus.cancelled
            sub.completed_at = datetime.now()
            sub.message = reason
            sub.error_message = None

    def reset_pipeline_for_retry(self, task_id: int) -> bool:
        """Return a durable pipeline Task to pending without discarding its artifacts.

        Stages probe the manifest/artifacts on restart, so resetting task rows is
        safe: verified PDFs remain skipped and only unfinished/failed sources are
        retried.
        """
        with self._sf() as s:
            task = s.get(Task, task_id)
            if task is None:
                return False
            task.status = TaskStatus.pending
            task.progress = 0
            task.error_message = None
            task.result_json = None
            task.started_at = None
            task.completed_at = None
            task.worker_id = None
            task.last_heartbeat = None
            task.retry_count = (task.retry_count or 0) + 1
            params = dict(task.params_json or {})
            state = dict(params.get("state") or {})
            state.update({"retry_requested": True, "current_paper_id": None, "current_file_bytes": 0})
            params["state"] = state
            task.params_json = params
            for sub in s.query(TaskSubtask).filter(TaskSubtask.task_id == task_id).all():
                sub.status = TaskSubtaskStatus.pending
                sub.progress = 0
                sub.message = "Queued for retry"
                sub.error_message = None
                sub.started_at = None
                sub.completed_at = None
            return True

    # ----------------------------------------------------------------- event
    def emit_event(
        self,
        task_id: int,
        key: Optional[str],
        message: str,
        *,
        level: str = "info",
        event_type: str = "stage.log",
        stage: Optional[str] = None,
    ) -> None:
        with self._sf() as s:
            subtask_id = None
            if key is not None:
                sub = s.query(TaskSubtask).filter_by(task_id=task_id, subtask_key=key).first()
                if sub is not None:
                    subtask_id = sub.id
                    if stage is None:
                        stage = sub.stage
            s.add(
                TaskEvent(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    event_type=event_type,
                    level=level,
                    stage=stage or key,
                    message=message,
                )
            )
        if level in ("error", "warning"):
            logger.log(level.upper(), f"task={task_id} stage={key or '-'}: {message}")
