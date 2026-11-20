"""Recurring-job engine: schedules live in the `schedules` table, APScheduler keeps time.

Division of labour (decided after reading the APScheduler 3.11.3 sources):

* The `schedules` / `schedule_ticks` tables are the source of truth. APScheduler
  runs with an in-memory job store and the jobs are rebuilt from the table at
  startup (`reconcile`). The SQLAlchemyJobStore is not used: its `_get_jobs`
  silently deletes rows it cannot unpickle, and `add_job(replace_existing=True)`
  discards runs missed while the process was down and un-pauses the job
  (schedulers/base.py `_real_add_job`); pickled blobs cannot be shown in a UI either.
* Catch-up after downtime is APScheduler's own logic, fed the right starting point.
  `reconcile` sets `next_run_time = trigger.get_next_fire_time(last_tick, now)`; when
  that is in the past, `_process_jobs` runs `job._get_run_times(now)`, which walks
  every missed slot, `coalesce` keeps only the latest, and `executors/base.py
  run_job` fires it if `now - slot <= misfire_grace_time`, else emits
  EVENT_JOB_MISSED. The grace default of 1 s would silently drop every late run, so
  each job gets one explicitly (row value, else the kind's default).
* Every trigger is written to `schedule_ticks` from the EVENT_JOB_EXECUTED /
  EVENT_JOB_ERROR / EVENT_JOB_MISSED events (events.py `JobExecutionEvent.
  scheduled_run_time`), keyed (schedule, slot, trigger) so a slot is recorded once.
* `trigger.get_next_fire_time(previous, now)` (triggers/cron/__init__.py) also
  drives the "next 5 fire times" preview, chained exactly as `_get_run_times` does.
* Single instance: APScheduler documents that several processes sharing one job
  source run every job once per process. A non-blocking file lock decides which
  backend process runs the engine.

Retries: a task started by a schedule that ends `failed` is re-driven through its
runner (`runner.retry`, which resumes at the failed stage) after the kind's
exponential backoff, by a one-shot date job.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    JobExecutionEvent,
)
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.base import ConflictingIdError
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from filelock import FileLock, Timeout
from loguru import logger
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.services.pipeline.runner_registry import get_runner
from app.services.pipeline.store import TaskStore
from app.services.scheduler.kinds import FireResult, JobKind, all_kinds
from app.services.scheduler.triggers import build_trigger, fire_times_from, next_fire_times, parse_trigger
from core.db.database import get_db_session
from core.db.models import Schedule, ScheduleTick
from core.timestamps import from_epoch, from_utc_naive, rfc3339

CRON_JOB_PREFIX = "schedule:"
RETRY_JOB_PREFIX = "retry:"
# A one-shot retry that the event loop could not start within this long is dropped
# (and recorded as missed) rather than run at a surprising time.
RETRY_MISFIRE_GRACE_S = 300
CATCHUP_MODES = ("latest", "skip")
# APScheduler logs "Running job" / "executed successfully" at INFO for every run, which
# a 30 s polling job turns into ~5,800 journal lines a day. schedule_ticks is the run
# record; only warnings (misfires, errors, max-instances skips) are worth a log line.
APSCHEDULER_QUIET_LOGGERS = ("apscheduler.executors.default", "apscheduler.scheduler")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _naive_utc(moment: datetime) -> datetime:
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def _aware_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc)


def _iso(moment: Optional[datetime]) -> Optional[str]:
    """A naive-UTC column (CURRENT_TIMESTAMP default, or written via _naive_utc)."""
    return rfc3339(from_utc_naive(moment))


class SchedulerService:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = get_db_session,
        store: Optional[TaskStore] = None,
        lock_path: Optional[str] = None,
        kinds: Callable[[], dict[str, JobKind]] = all_kinds,
    ) -> None:
        self._sf = session_factory
        self._store = store or TaskStore(session_factory)
        self._lock_path = lock_path
        self._kinds = kinds
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._file_lock: Optional[FileLock] = None

    # ------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def start(self) -> bool:
        """Start the engine inside the running event loop. False = not started
        (another process holds the lock)."""
        if self.running:
            return True
        if self._lock_path is not None:
            Path(self._lock_path).parent.mkdir(parents=True, exist_ok=True)
            lock = FileLock(self._lock_path)
            try:
                lock.acquire(timeout=0)
            except Timeout:
                logger.warning(
                    f"scheduler.lock.held path={self._lock_path}: another process runs the "
                    "scheduler; this one only serves the schedule tables"
                )
                return False
            self._file_lock = lock
        for name in APSCHEDULER_QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        try:
            self.ensure_seeds()
            self._loop = asyncio.get_running_loop()
            self._scheduler = AsyncIOScheduler(
                timezone="UTC",
                jobstores={"default": MemoryJobStore()},
                executors={"default": AsyncIOExecutor()},
                job_defaults={"coalesce": True, "max_instances": 1},
            )
            self._scheduler.add_listener(
                self._on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
            )
            self._scheduler.start()
            TaskStore.add_terminal_listener(self._on_task_terminal)
            self.reconcile()
        except BaseException:
            self.shutdown()
            raise
        logger.info("scheduler.started")
        return True

    def shutdown(self) -> None:
        TaskStore.remove_terminal_listener(self._on_task_terminal)
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._scheduler = None
        if self._file_lock is not None:
            self._file_lock.release()
            self._file_lock = None

    # ---------------------------------------------------------------- seeds
    def ensure_seeds(self) -> None:
        """Insert each kind's built-in schedules that are missing. Existing rows
        (which the user may have edited) are never touched."""
        for kind in self._kinds().values():
            for seed in kind.defaults:
                with self._sf() as s:
                    if s.query(Schedule.id).filter(Schedule.key == seed.key).first() is not None:
                        continue
                    trigger, params = self._validated(kind, seed.trigger, seed.params)
                    s.add(Schedule(
                        key=seed.key, kind=kind.name, title=seed.title, enabled=seed.enabled,
                        system=True, trigger_json=trigger, params_json=params,
                        catchup=seed.catchup, misfire_grace_s=seed.misfire_grace_s,
                    ))
                    logger.info(f"scheduler.seed key={seed.key}")

    # ----------------------------------------------------------- reconcile
    def reconcile(self) -> None:
        """Rebuild every job from the table: enabled rows get a job, others none."""
        for row in self._rows():
            self._sync_job(row)

    def _sync_job(self, row: dict) -> None:
        """Make the engine match one row (idempotent). The job is removed and re-added
        rather than modified: `modify_job(trigger=...)` does not recompute next_run_time."""
        scheduler = self._scheduler
        if scheduler is None or not scheduler.running:
            return
        job_id = f"{CRON_JOB_PREFIX}{row['id']}"
        if scheduler.get_job(job_id) is not None:
            scheduler.remove_job(job_id)
        if not row["enabled"]:
            return
        kind = self._kinds().get(row["kind"])
        if kind is None:
            logger.warning(f"scheduler.reconcile.unknown_kind schedule={row['key']} kind={row['kind']}")
            return
        trigger = build_trigger(parse_trigger(row["trigger_json"]))
        now = _utcnow()
        last = self._last_cron_slot(row["id"]) if row["catchup"] == "latest" else None
        next_run = trigger.get_next_fire_time(last, now)
        if next_run is None:
            logger.warning(f"scheduler.reconcile.no_next_fire schedule={row['key']}")
            return
        grace = row["misfire_grace_s"] if row["misfire_grace_s"] is not None else kind.default_misfire_grace_s
        scheduler.add_job(
            self._run_cron_job, trigger, id=job_id, name=row["key"],
            kwargs={"schedule_id": row["id"]},
            next_run_time=next_run, misfire_grace_time=grace,
        )

    def _last_cron_slot(self, schedule_id: int) -> Optional[datetime]:
        with self._sf() as s:
            last = (
                s.query(ScheduleTick.scheduled_for)
                .filter(ScheduleTick.schedule_id == schedule_id, ScheduleTick.trigger == "cron")
                .order_by(ScheduleTick.scheduled_for.desc())
                .limit(1)
                .scalar()
            )
        return _aware_utc(last) if last is not None else None

    # --------------------------------------------------------------- firing
    def _run_cron_job(self, schedule_id: int) -> dict:
        """APScheduler job body (runs in the event loop's thread pool)."""
        return self._fire(self._existing_row(schedule_id), "cron", 0).as_dict()

    def _run_retry_job(self, schedule_id: int, task_id: int, attempt: int) -> dict:
        """One-shot job: re-drive a failed task through its runner."""
        task = self._store.get_task(task_id)
        if task is None:
            return FireResult("skipped", task_id, "task no longer exists").as_dict()
        runner = get_runner(task["task_type"])
        if runner is None:
            return FireResult("skipped", task_id, f"no runner for {task['task_type']}").as_dict()
        result = runner.retry(task_id)
        if result.get("status") == "started":
            return FireResult("enqueued", task_id, f"retry {attempt}").as_dict()
        return FireResult("skipped", result.get("task_id"), result.get("error") or "retry not started").as_dict()

    def _existing_row(self, schedule_id: int) -> dict:
        row = self._row(schedule_id)
        if row is None:
            raise LookupError(f"schedule {schedule_id} not found")
        return row

    def _fire(self, row: dict, trigger: str, attempt: int) -> FireResult:
        kind = self._kinds().get(row["kind"])
        if kind is None:
            raise LookupError(f"unknown schedule kind {row['kind']!r}")
        params = kind.params_model.model_validate(row["params_json"])
        # `scheduled_for` here is when the trigger fired; the slot APScheduler was
        # due for is only known to its event and is what schedule_ticks records.
        origin = {
            "schedule_id": row["id"], "trigger": trigger,
            "scheduled_for": rfc3339(_utcnow()), "attempt": attempt,
        }
        return kind.fire(params, origin)

    def run_now(self, schedule_id: int) -> FireResult:
        """Fire a schedule immediately, whatever its trigger or enabled flag. Blocking:
        call from a worker thread."""
        row = self._existing_row(schedule_id)
        slot = _utcnow()
        try:
            result = self._fire(row, "manual", 0)
        except Exception as exc:
            self._write_tick(schedule_id, slot, "manual", "error", f"{type(exc).__name__}: {exc}", None, 0)
            raise
        self._record_result(schedule_id, slot, "manual", result, 0)
        return result

    # ---------------------------------------------------------------- ticks
    def _on_job_event(self, event: JobExecutionEvent) -> None:
        """EVENT_JOB_* listener. Runs in the event loop; the DB write goes to the
        thread pool so a locked SQLite file (busy_timeout is 30 s) cannot stall it."""
        if self._loop is not None:
            self._loop.run_in_executor(None, self._tick_from_event, event)

    def _tick_from_event(self, event: JobExecutionEvent) -> None:
        try:
            if event.job_id.startswith(RETRY_JOB_PREFIX):
                _, schedule_id, _task, attempt = event.job_id.split(":")
                trigger, attempt = "retry", int(attempt)
            elif event.job_id.startswith(CRON_JOB_PREFIX):
                schedule_id, trigger, attempt = event.job_id[len(CRON_JOB_PREFIX):], "cron", 0
            else:
                return
            schedule_id = int(schedule_id)
            slot = _naive_utc(event.scheduled_run_time)
            if event.code == EVENT_JOB_MISSED:
                self._write_tick(schedule_id, slot, trigger, "missed", "misfire grace time exceeded", None, attempt)
            elif event.code == EVENT_JOB_ERROR:
                self._write_tick(
                    schedule_id, slot, trigger, "error",
                    f"{type(event.exception).__name__}: {event.exception}"[:2000], None, attempt,
                )
            else:
                retval = event.retval or {}
                self._record_result(
                    schedule_id, slot,
                    trigger, FireResult(retval.get("status", "noop"), retval.get("task_id"), retval.get("reason")),
                    attempt,
                )
        except Exception:
            logger.exception(f"scheduler.tick.write_failed job={event.job_id}")

    def _record_result(self, schedule_id: int, slot: datetime, trigger: str, result: FireResult, attempt: int) -> None:
        if result.status == "noop":
            row = self._row(schedule_id)
            kind = self._kinds().get(row["kind"]) if row else None
            if kind is not None and not kind.record_idle_ticks:
                return
        status = "enqueued" if result.status == "enqueued" else "skipped"
        reason = result.reason if result.status != "noop" else (result.reason or "nothing to do")
        self._write_tick(schedule_id, slot, trigger, status, reason, result.task_id, attempt)

    def _write_tick(
        self, schedule_id: int, slot: datetime, trigger: str, status: str,
        reason: Optional[str], task_id: Optional[int], attempt: int,
    ) -> None:
        slot = _naive_utc(slot) if slot.tzinfo is not None else slot
        stmt = sqlite_insert(ScheduleTick).values(
            schedule_id=schedule_id, scheduled_for=slot, fired_at=_naive_utc(_utcnow()),
            trigger=trigger, status=status, reason=reason, task_id=task_id, attempt=attempt,
        ).on_conflict_do_nothing(index_elements=["schedule_id", "scheduled_for", "trigger"])
        with self._sf() as s:
            s.execute(stmt)

    # -------------------------------------------------------------- retries
    def _on_task_terminal(self, task_id: int, task_type: str, status: str) -> None:
        """TaskStore terminal listener (orchestrator thread). A failed task that a
        schedule started is retried while its kind allows it."""
        if status != "failed" or not self.running:
            return
        task = self._store.get_task(task_id)
        origin = (task or {}).get("params", {}).get("origin") or {}
        schedule_id = origin.get("schedule_id")
        if schedule_id is None:
            return
        row = self._row(int(schedule_id))
        kind = self._kinds().get(row["kind"]) if row else None
        if kind is None:
            return
        # Retrying re-drives this same task, so `origin` never changes; the number
        # of retries already made is the task's own retry_count.
        made = task["retry_count"]
        if made >= kind.retry.max_retries:
            return
        delay = kind.retry.delay_s(made)
        attempt = made + 1
        try:
            self._scheduler.add_job(
                self._run_retry_job, DateTrigger(run_date=_utcnow() + timedelta(seconds=delay)),
                id=f"{RETRY_JOB_PREFIX}{schedule_id}:{task_id}:{attempt}",
                kwargs={"schedule_id": int(schedule_id), "task_id": task_id, "attempt": attempt},
                misfire_grace_time=RETRY_MISFIRE_GRACE_S,
            )
        except ConflictingIdError:
            return
        logger.info(f"scheduler.retry.queued schedule={schedule_id} task={task_id} attempt={attempt} in={delay:g}s")

    # ------------------------------------------------------------------ CRUD
    def _validated(self, kind: JobKind, trigger: dict, params: dict) -> tuple[dict, dict]:
        """Normalised trigger and params; raises ValueError (pydantic's included)."""
        return (
            parse_trigger(trigger).model_dump(),
            kind.params_model.model_validate(params).model_dump(mode="json"),
        )

    def _kind(self, name: str) -> JobKind:
        kind = self._kinds().get(name)
        if kind is None:
            raise ValueError(f"unknown schedule kind {name!r}")
        return kind

    @staticmethod
    def _as_dict(row: Schedule) -> dict:
        return {
            "id": row.id, "key": row.key, "kind": row.kind, "title": row.title,
            "enabled": bool(row.enabled), "system": bool(row.system),
            "trigger_json": dict(row.trigger_json or {}), "params_json": dict(row.params_json or {}),
            "catchup": row.catchup, "misfire_grace_s": row.misfire_grace_s,
            "created_at": _iso(row.created_at), "updated_at": _iso(row.updated_at),
        }

    def _rows(self) -> list[dict]:
        with self._sf() as s:
            return [self._as_dict(r) for r in s.query(Schedule).order_by(Schedule.id).all()]

    def _row(self, schedule_id: int) -> Optional[dict]:
        with self._sf() as s:
            row = s.get(Schedule, schedule_id)
            return self._as_dict(row) if row is not None else None

    def get(self, schedule_id: int) -> Optional[dict]:
        return self._row(schedule_id)

    def create(
        self, *, key: str, kind: str, title: str, trigger: dict, params: Optional[dict] = None,
        enabled: bool = True, catchup: str = "latest", misfire_grace_s: Optional[int] = None,
    ) -> dict:
        job_kind = self._kind(kind)
        if catchup not in CATCHUP_MODES:
            raise ValueError(f"catchup must be one of {CATCHUP_MODES}")
        trigger, params = self._validated(job_kind, trigger, params or {})
        with self._sf() as s:
            if s.query(Schedule.id).filter(Schedule.key == key).first() is not None:
                raise KeyError(f"schedule key {key!r} already exists")
            row = Schedule(
                key=key, kind=kind, title=title, enabled=enabled, system=False,
                trigger_json=trigger, params_json=params, catchup=catchup,
                misfire_grace_s=misfire_grace_s,
            )
            s.add(row)
            s.flush()
            created = self._as_dict(row)
        self._sync_job(created)
        return created

    def update(self, schedule_id: int, changes: dict) -> dict:
        """Apply `changes` (any of title, enabled, trigger, params, catchup,
        misfire_grace_s) and re-sync the job. Raises LookupError / ValueError."""
        with self._sf() as s:
            row = s.get(Schedule, schedule_id)
            if row is None:
                raise LookupError(f"schedule {schedule_id} not found")
            kind = self._kind(row.kind)
            trigger = changes.get("trigger", row.trigger_json)
            params = changes.get("params", row.params_json)
            row.trigger_json, row.params_json = self._validated(kind, trigger, params)
            if "title" in changes:
                row.title = changes["title"]
            if "enabled" in changes:
                row.enabled = bool(changes["enabled"])
            if "catchup" in changes:
                if changes["catchup"] not in CATCHUP_MODES:
                    raise ValueError(f"catchup must be one of {CATCHUP_MODES}")
                row.catchup = changes["catchup"]
            if "misfire_grace_s" in changes:
                row.misfire_grace_s = changes["misfire_grace_s"]
            s.flush()
            updated = self._as_dict(row)
        self._sync_job(updated)
        return updated

    def delete(self, schedule_id: int) -> None:
        with self._sf() as s:
            row = s.get(Schedule, schedule_id)
            if row is None:
                raise LookupError(f"schedule {schedule_id} not found")
            if row.system:
                raise PermissionError("built-in schedules cannot be deleted; disable them instead")
            s.delete(row)
        self._sync_job({"id": schedule_id, "enabled": False})

    # --------------------------------------------------------------- reading
    def list_schedules(self) -> list[dict]:
        """Rows plus live state: engine next run, preview, last tick, running task, gauges."""
        out = []
        for row in self._rows():
            kind = self._kinds().get(row["kind"])
            spec = parse_trigger(row["trigger_json"])
            job = self._scheduler.get_job(f"{CRON_JOB_PREFIX}{row['id']}") if self.running else None
            if job is not None and job.next_run_time is not None:
                # Chain from the live job so the preview starts at its real next run.
                upcoming = fire_times_from(job.trigger, job.next_run_time, 5)
            else:
                upcoming = next_fire_times(spec, 5) if row["enabled"] else []
            running = self._store.list_by_origin(row["id"], 1, statuses=("pending", "running"))
            last = self._last_tick(row["id"])
            out.append({
                **row,
                "next_run_time": job.next_run_time.isoformat() if job is not None and job.next_run_time else None,
                "next_fire_times": [t.isoformat() for t in upcoming],
                "last_tick": self._with_tasks([last])[0] if last else None,
                "running_task": self._running_brief(running[0]) if running else None,
                "metrics": self._metrics(kind),
            })
        return out

    @staticmethod
    def _metrics(kind: Optional[JobKind]) -> list[dict]:
        if kind is None or kind.status is None:
            return []
        try:
            return [{"name": m.name, "value": m.value, "unit": m.unit} for m in kind.status()]
        except Exception as exc:  # a gauge failing must not take the listing down
            logger.warning(f"scheduler.metrics.failed kind={kind.name}: {exc}")
            return []

    @staticmethod
    def _tick_dict(tick: ScheduleTick) -> dict:
        return {
            "id": tick.id, "schedule_id": tick.schedule_id,
            "scheduled_for": _iso(tick.scheduled_for), "fired_at": _iso(tick.fired_at),
            "trigger": tick.trigger, "status": tick.status, "reason": tick.reason,
            "task_id": tick.task_id, "attempt": tick.attempt,
        }

    def _last_tick(self, schedule_id: int) -> Optional[dict]:
        with self._sf() as s:
            tick = (
                s.query(ScheduleTick).filter(ScheduleTick.schedule_id == schedule_id)
                .order_by(ScheduleTick.id.desc()).first()
            )
            return self._tick_dict(tick) if tick is not None else None

    def list_ticks(self, schedule_id: int, limit: int = 50, before_id: Optional[int] = None) -> list[dict]:
        """Newest-first ticks, each with the task it started (status, times, result)."""
        with self._sf() as s:
            q = s.query(ScheduleTick).filter(ScheduleTick.schedule_id == schedule_id)
            if before_id is not None:
                q = q.filter(ScheduleTick.id < before_id)
            ticks = [self._tick_dict(t) for t in q.order_by(ScheduleTick.id.desc()).limit(limit).all()]
        return self._with_tasks(ticks)

    def _with_tasks(self, ticks: list[dict]) -> list[dict]:
        """Attach the Task each tick started: {id, status, error, started_at, completed_at, result_json}."""
        tasks = self._store.get_tasks([t["task_id"] for t in ticks if t["task_id"] is not None])
        for tick in ticks:
            task = tasks.get(tick["task_id"])
            tick["task"] = None if task is None else {
                "id": task["id"], "status": task["status"], "error": task["error"],
                "started_at": _stamp(task["started_at"]), "completed_at": _stamp(task["finished_at"]),
                "result_json": task["result"],
            }
        return ticks

    @staticmethod
    def _running_brief(task: dict) -> dict:
        return {
            "id": task["id"], "task_type": task["task_type"], "status": task["status"],
            "progress": task["progress"], "total": task["total"], "started_at": _stamp(task["started_at"]),
        }

    def kinds_info(self) -> list[dict]:
        return [
            {
                "name": k.name, "title": k.title, "params_schema": k.params_model.model_json_schema(),
                "defaults": [
                    {"key": d.key, "title": d.title, "trigger": d.trigger, "params": d.params,
                     "catchup": d.catchup, "misfire_grace_s": d.misfire_grace_s, "enabled": d.enabled}
                    for d in k.defaults
                ],
                "retry": {
                    "max_retries": k.retry.max_retries, "base_delay_s": k.retry.base_delay_s,
                    "backoff": k.retry.backoff, "max_delay_s": k.retry.max_delay_s,
                },
            }
            for k in self._kinds().values()
        ]


def _stamp(epoch: Optional[float]) -> Optional[str]:
    """TaskStore dicts carry epoch seconds (already converted per column convention)."""
    return rfc3339(from_epoch(epoch))


@lru_cache(maxsize=1)
def get_scheduler_service() -> SchedulerService:
    from config import settings

    return SchedulerService(lock_path=settings.scheduler.lock_path)
