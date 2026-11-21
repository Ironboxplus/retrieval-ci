"""Mirror the in-process job queues onto `tasks` rows so the Tasks page shows them.

Journal bulk indexing, Code KB indexing and public-code jobs each run in their own
in-process queue; the Tasks page and the status bar only read the `tasks` table, so
those jobs were invisible there. This adapter writes one Task row per queue job
(created, running, counted progress, terminal state) and one `TaskRunner` per queue so
the generic Tasks API can cancel it and startup can settle rows a dead process left.

The queue stays the executor and the source of truth for execution. The row is an
observation surface: every write here is best-effort and a database failure is logged
once and never reaches the job. Request bodies are deliberately not persisted, so a
restart fails the row (clients resubmit) instead of resuming the job.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from loguru import logger

from app.services.pipeline.runner_registry import TaskNotCancellable
from app.services.pipeline.store import TaskStore

# Completed rows of one type kept after a job completes. Failed/cancelled rows are
# kept: they are the evidence of what went wrong.
QUEUE_TASK_KEEP_COMPLETED = 20
# One progress write per task per interval; per-batch ticks would otherwise be one
# SQLite commit per embed batch.
PROGRESS_WRITE_INTERVAL_SECONDS = 1.0
INTERRUPTED_ERROR = "后端重启，任务中断；客户端重新执行即可"

_ACTIVE = {"pending", "running"}


class QueueTasks:
    """Best-effort Task-row recorder shared by every in-process queue."""

    def __init__(
        self,
        store: Optional[TaskStore] = None,
        *,
        min_interval: float = PROGRESS_WRITE_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store or TaskStore()
        self._min_interval = min_interval
        self._clock = clock
        self._lock = threading.Lock()
        self._last_progress_write: dict[int, float] = {}
        self._last_heartbeat_write: dict[int, float] = {}
        self._failure_logged = False

    # ------------------------------------------------------------------ api
    def open(
        self, task_type: str, params: dict[str, Any], total: int, owner_user_id: Optional[str],
    ) -> Optional[int]:
        """Create the pending row. None when the row could not be written."""
        def write() -> int:
            task_id = self.store.create_task(
                task_type=task_type, params=params, total=total, owner_user_id=owner_user_id)
            self.store.emit_event(
                task_id, None, "Queued", event_type="task.created", stage="task.created")
            return task_id

        return self._guard(write)

    def started(self, task_id: Optional[int]) -> None:
        if task_id is None:
            return

        def write() -> None:
            if self.store.compare_and_set_status(task_id, expect={"pending"}, new="running"):
                self.store.emit_event(
                    task_id, None, "Started", event_type="task.started", stage="task.started")

        self._guard(write)

    def progress(self, task_id: Optional[int], done: int, total: int) -> None:
        if task_id is None:
            return
        now = self._clock()
        with self._lock:
            last = self._last_progress_write.get(task_id)
            # done >= total is the last tick before the job's own commit phase and
            # must land even inside the interval, or the bar would stop short.
            if (last is not None and now - last < self._min_interval
                    and not (total and done >= total)):
                return
            self._last_progress_write[task_id] = now
        # A tick is also the liveness signal the Tasks page's "heartbeat stale" check reads.
        self._guard(lambda: self.store.set_task_progress(
            task_id, done, total, heartbeat=True))

    def heartbeat(self, task_id: Optional[int]) -> None:
        """Refresh `last_heartbeat` only, for a phase that makes no countable progress."""
        if task_id is None:
            return
        now = self._clock()
        with self._lock:
            last = self._last_heartbeat_write.get(task_id)
            if last is not None and now - last < self._min_interval:
                return
            self._last_heartbeat_write[task_id] = now
        self._guard(lambda: self.store.heartbeat(task_id))

    def finished(
        self,
        task_id: Optional[int],
        status: str,
        *,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Move to a terminal state unless something else (a cancel) already did."""
        if task_id is None:
            return
        with self._lock:
            self._last_progress_write.pop(task_id, None)
            self._last_heartbeat_write.pop(task_id, None)

        def write() -> None:
            won = self.store.compare_and_set_status(
                task_id, expect=_ACTIVE, new=status, error=error, result=result,
                fill_progress=status == "completed")
            if not won:
                return
            row = self.store.get_task(task_id)
            self.store.emit_event(
                task_id, None, error or status.capitalize(),
                level="error" if status == "failed" else "info",
                event_type=f"task.{status}", stage=f"task.{status}")
            if status == "completed" and row is not None:
                self.store.prune_completed(row["task_type"], QUEUE_TASK_KEEP_COMPLETED)

        self._guard(write)

    # ------------------------------------------------------------- internals
    def _guard(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception:
            # One line per process: a locked/unavailable tasks table would otherwise
            # log once per progress tick for the whole job.
            if not self._failure_logged:
                self._failure_logged = True
                logger.opt(exception=True).warning(
                    "queue_tasks: task row write failed; the job continues unrecorded")
            return None


class QueueTaskRunner:
    """`TaskRunner` for rows written by `QueueTasks`.

    `cancel_job(job_id, owner_user_id)` is the queue's own cancel: it returns whether a
    cancellation was accepted, raises `TaskNotCancellable` for a job past its point of
    no return, and `KeyError` for a job the queue does not know.
    """

    def __init__(
        self,
        task_type: str,
        cancel_job: Callable[[str, Optional[str]], bool],
        *,
        store: Optional[TaskStore] = None,
    ) -> None:
        self.task_type = task_type
        self._cancel_job = cancel_job
        self.store = store or TaskStore()

    def start(self, params: dict) -> dict:
        raise NotImplementedError("该任务由对应的索引队列创建，不能从任务页启动")

    def retry(self, task_id: int) -> dict:
        raise NotImplementedError("由客户端重新执行")

    def cancel(self, task_id: int) -> bool:
        row = self.store.get_task(task_id)
        if row is None:
            return False
        job_id = str((row["params"] or {}).get("job_id") or "")
        try:
            return bool(self._cancel_job(job_id, row["owner_user_id"]))
        except KeyError:
            # The row is active but its queue has no such job (history trimmed, or a
            # row a previous process left). Nothing can run it, so settle the row.
            won = self.store.compare_and_set_status(
                task_id, expect=_ACTIVE, new="cancelled")
            if won:
                self.store.emit_event(
                    task_id, None, "Task cancelled by API request",
                    level="warning", event_type="task.cancelled", stage="task.cancelled")
            return won

    def resume_running(self) -> list[int]:
        """Fail rows left active by a previous process; the queues start empty.

        Nothing is relaunched: the request bodies were never persisted, and clients
        resubmit (Journal sync only records a job after it completed).
        """
        for task in self.store.list_active(self.task_type):
            task_id = int(task["id"])
            if self.store.compare_and_set_status(
                    task_id, expect=_ACTIVE, new="failed", error=INTERRUPTED_ERROR):
                self.store.emit_event(
                    task_id, None, INTERRUPTED_ERROR, level="error",
                    event_type="task.failed", stage="task.failed")
        return []


__all__ = [
    "INTERRUPTED_ERROR",
    "PROGRESS_WRITE_INTERVAL_SECONDS",
    "QUEUE_TASK_KEEP_COMPLETED",
    "QueueTaskRunner",
    "QueueTasks",
    "TaskNotCancellable",
]
