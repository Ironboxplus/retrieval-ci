"""Registry of the runners that drive each durable Task type.

A runner owns one `Task.task_type`: it creates the task row, drives it on a worker
thread, can be told to cancel or retry, and re-attaches to tasks a previous process
left `running`. Registering it here lets every caller reach it by task type instead
of naming the class: the Tasks API (cancel/retry), the scheduler (`start`) and the
startup hook (`resume_running`) are all generic over `all_runners()`.

Registration happens at import of the runner's module. `BUILTIN_RUNNER_MODULES`
lists those modules; they are imported lazily on first lookup so this module stays
free of runner dependencies and nothing imports a runner just to cancel a task.
"""
from __future__ import annotations

import importlib
import threading
from typing import Callable, Optional, Protocol

from loguru import logger

from app.services.pipeline.store import TaskStore

BUILTIN_RUNNER_MODULES: tuple[str, ...] = (
    "app.services.pipeline.service",
    "app.services.pipeline.upload_service",
    "app.services.pipeline.reembed_pipeline",
    "app.services.medical_corpus.service",
    "app.services.journal.jobs",
    "app.services.code_kb.jobs",
    "app.services.code_kb.analysis",
    "app.services.public_code.jobs",
)


class TaskNotCancellable(Exception):
    """`TaskRunner.cancel` refuses a task that is still active, with a reason the
    caller can show (the Tasks API answers 409). Distinct from `False`, which means
    the task had already finished."""


class TaskRunner(Protocol):
    task_type: str

    def start(self, params: dict) -> dict:
        """Create and launch a task. -> {"status": "started"|"conflict"|..., "task_id": int|None, "error"?: str}"""

    def cancel(self, task_id: int) -> bool:
        """Cancel a pending/running task. False when it was already finished."""

    def retry(self, task_id: int) -> dict:
        """Re-drive a failed/cancelled task in place. Same result shape as start()."""

    def resume_running(self) -> list[int]:
        """Relaunch tasks a previous process left `running`. -> the relaunched task ids."""


_getters: dict[str, Callable[[], TaskRunner]] = {}
_builtins_loaded = False
_load_lock = threading.Lock()


def register_runner(task_type: str, getter: Callable[[], TaskRunner]) -> None:
    """`getter` returns the process singleton; it is only called on lookup."""
    _getters[task_type] = getter


def _load_builtins() -> None:
    global _builtins_loaded
    with _load_lock:
        if _builtins_loaded:
            return
        for module in BUILTIN_RUNNER_MODULES:
            try:
                importlib.import_module(module)
            except Exception as exc:  # one broken runner must not hide the others
                logger.warning(f"runner_registry.load_failed module={module}: {exc}")
        _builtins_loaded = True


def get_runner(task_type: str) -> Optional[TaskRunner]:
    _load_builtins()
    getter = _getters.get(task_type)
    return getter() if getter is not None else None


def all_runners() -> dict[str, Callable[[], TaskRunner]]:
    """task_type -> getter. Getters, not instances: constructing a runner can fail
    (the medical runner validates its environment) and callers isolate that per type."""
    _load_builtins()
    return dict(_getters)


def cancel_run(store: TaskStore, task_id: int, abort: Optional[threading.Event]) -> bool:
    """Cancel a task: win the status race first, then stop the worker.

    pending/running -> cancelled is one conditional UPDATE, so exactly one of the
    cancel path and the orchestrator's own terminal write decides the final state
    (the orchestrator's writes are compare-and-set too). `abort` is set even when
    the task had already finished, which is harmless and makes repeated cancels
    idempotent. Returns True only for the caller that performed the transition.
    """
    won = store.compare_and_set_status(task_id, expect={"pending", "running"}, new="cancelled")
    if won:
        store.emit_event(
            task_id, None, "Task cancelled by API request",
            level="warning", event_type="task.cancelled", stage="task.cancelled",
        )
    if abort is not None:
        abort.set()
    return won


class CancellableRunner:
    """Mixin for runners that drive one orchestrator thread per task.

    Needs `self.store`, `self._lock` and `self._aborts: dict[int, Event]`. `_launch`
    registers the Event before starting the worker thread and drops it when the
    thread ends; `cancel()` signals it.
    """

    store: TaskStore
    _lock: threading.Lock
    _aborts: dict[int, threading.Event]

    def _register_abort(self, task_id: int) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._aborts[task_id] = event
        return event

    def _drop_abort(self, task_id: int) -> None:
        with self._lock:
            self._aborts.pop(task_id, None)

    def cancel(self, task_id: int) -> bool:
        with self._lock:
            abort = self._aborts.get(task_id)
        return cancel_run(self.store, task_id, abort)

    def _fail_crashed(self, task_id: int, exc: BaseException) -> None:
        """Mark a task whose orchestrator thread died, unless it was already cancelled."""
        try:
            self.store.compare_and_set_status(
                task_id, expect={"pending", "running"}, new="failed",
                error=f"orchestrator crash: {exc}",
            )
        except Exception:
            logger.exception(f"runner.crash.status_write_failed task={task_id}")
