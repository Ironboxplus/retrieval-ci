"""Background duplicate-code analysis, run by the server after code indexing.

Why it exists: a duplicated design is cheapest to remove right after it is written, and
nobody calls `find_duplicate_code` right after writing it. The index job that just
finished is the one moment the server knows which files changed, so it tells this
scheduler, and the scheduler checks those files against the rest of the scope.

What it decides, and where each decision lives:

* *Did anything change?* ``index_catalog.content_changed_at`` / ``deleted_at`` against the
  scope's persisted watermark (analysis_store.py). Nothing is remembered in process, so a
  restart loses nothing, many index runs merge into one question, and an index that
  changed no content never runs an analysis or leaves a Tasks row.
* *Is it safe to look now?* Only when the scope's durable snapshot marker is completed.
  The scan itself holds the owner fence per query batch only and is discarded if the
  scope was written to meanwhile (CodeKBService.scan_guard), so it never blocks a search.
* *How often?* Two analyses of one scope are at least ``min_interval_seconds`` apart; index
  runs inside the interval coalesce into one later run, started by a timer. The last run
  is read back from SQLite, so the interval also holds across a restart. Nothing is
  scanned at startup: the next index completion covers any backlog through the watermark.
* *How much?* Only the chunks of the changed files are queried (`scan_changed_files`), and
  a change above ``max_chunks`` is reported as too large instead of being scanned.

Each analysis is a ``code_duplicates`` Task row (observation only, like the index and
journal queues) with the full pair lists, so the Tasks page shows what a short summary
could not. The summary text is stored with a version for later delivery.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from app.services.code_kb.analysis_store import (
    CodeAnalysisStore, ChangedFiles, pair_to_json)
from app.services.code_kb.duplicates import ScopeChangedDuringScan, scan_changed_files
from app.services.code_kb.service import CodeKBService, get_code_kb_service
from app.services.index_catalog.service import IndexCatalogService, get_index_catalog_service
from app.services.pipeline.queue_tasks import QueueTaskRunner, QueueTasks
from app.services.pipeline.runner_registry import TaskNotCancellable, register_runner
from app.services.pipeline.store import TaskStore
from app.services.scope_paths import canonicalize_scope_path, is_windows_scope
from config import settings
from core.code_kb.duplicate_rollup import DuplicateFilePair

logger = logging.getLogger("code_kb.analysis")

TASK_TYPE = "code_duplicates"

# Pairs the summary names; the rest are counted and left to the Tasks page.
SUMMARY_PAIRS = 3
# Pairs kept in a Tasks row's result; a large change must not make one row enormous.
RESULT_PAIRS = 100

_ScopeKey = tuple[str, str]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _rank(pair: DuplicateFilePair) -> tuple:
    return (-pair.shared_chunks, -pair.max_score, pair.path_a, pair.path_b)


def _pair_line(pair: DuplicateFilePair) -> str:
    detail = f"{pair.shared_chunks} shared chunks, max similarity {pair.max_score:.2f}"
    if not pair.examples:
        return f"{pair.path_a} <-> {pair.path_b} ({detail})"
    a, b = pair.examples[0].a, pair.examples[0].b
    return (f"{a.path}:{a.start_line}-{a.end_line} <-> "
            f"{b.path}:{b.start_line}-{b.end_line} ({detail})")


def summarize(
    *, first_analysis: bool, new_pairs: Sequence[DuplicateFilePair], gone_pairs: int,
    changed_files: int, task_id: int | None,
) -> str | None:
    """The English summary of one analysis, or None when there is nothing worth saying.

    The first analysis of a scope has no baseline, so what it finds is "found", never
    "new". Later ones report the pairs that did not stand before and the number of
    pairs that stopped standing."""
    if not new_pairs and not gone_pairs:
        return None
    ordered = sorted(new_pairs, key=_rank)
    lines: list[str] = []
    if ordered:
        noun = f"{len(ordered)} file pair{'s' if len(ordered) != 1 else ''}"
        if first_analysis:
            lines.append(
                f"Duplicate code found in the {changed_files} file(s) changed by the last "
                f"index: {noun}.")
        else:
            lines.append(
                f"New duplicate code after the last index ({changed_files} file(s) "
                f"changed): {noun}.")
        lines.extend(f"- {_pair_line(pair)}" for pair in ordered[:SUMMARY_PAIRS])
        if len(ordered) > SUMMARY_PAIRS:
            more = len(ordered) - SUMMARY_PAIRS
            lines.append(f"... {more} more" + (
                f" on Tasks page #{task_id}" if task_id is not None else ""))
    if gone_pairs:
        lines.append(
            f"{gone_pairs} earlier duplicate file pair{'s' if gone_pairs != 1 else ''} "
            f"no longer {'exists' if gone_pairs == 1 else 'exist'}.")
    return "\n".join(lines)


@dataclass
class _Job:
    job_id: str
    owner_user_id: str
    project: str
    task_id: int | None
    cancelled: bool = False

    @property
    def key(self) -> _ScopeKey:
        return (self.owner_user_id, self.project)


class CodeAnalysisScheduler:
    """Throttled, coalescing, single-worker duplicate analysis of code scopes."""

    def __init__(
        self,
        *,
        service: CodeKBService,
        catalog_service: IndexCatalogService,
        store: CodeAnalysisStore | None = None,
        tasks: QueueTasks | None = None,
        min_interval_seconds: float | None = None,
        max_chunks: int | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        config = settings.code_analysis
        self.service = service
        self.catalog_service = catalog_service
        self.store = store or CodeAnalysisStore()
        # Direct constructions (tests) record nothing unless they pass one.
        self.tasks = tasks
        self.min_interval_seconds = (
            config.min_interval_seconds if min_interval_seconds is None else min_interval_seconds)
        self.max_chunks = config.max_chunks if max_chunks is None else max_chunks
        self._clock = clock
        # How many index completions told us about a scope; lets a caller wait for one.
        self.notifications = 0
        self._lock = threading.Lock()
        self._state_changed = threading.Condition(self._lock)
        self._scheduled: dict[_ScopeKey, threading.Timer] = {}
        self._queued: dict[_ScopeKey, _Job] = {}
        self._running: dict[_ScopeKey, _Job] = {}
        self._dirty: set[_ScopeKey] = set()
        self._last_started: dict[_ScopeKey, float] = {}
        self._closed = False
        self._q: queue.Queue[_Job | None] = queue.Queue()
        self._worker = threading.Thread(
            target=self._work, name="code-analysis-worker", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ public

    def notify(self, owner_user_id: str, project: str) -> None:
        """An index job of this scope completed. Cheap and never raises into the caller:
        it only records that the scope may have changed; every database read happens on
        the timer/worker threads."""
        key = (owner_user_id, self._canonical(project))
        with self._lock:
            self.notifications += 1
            if self._closed:
                return
            if key in self._running:
                self._dirty.add(key)  # changes after this run read its watermark
            elif key not in self._scheduled and key not in self._queued:
                self._schedule_locked(key, 0.0)
            self._state_changed.notify_all()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until no analysis is scheduled, queued or running."""
        with self._state_changed:
            return self._state_changed.wait_for(self._idle_locked, timeout)

    def cancel(self, job_id: str, owner_user_id: str | None) -> bool:
        """Cancel an analysis that has not started. KeyError: no such queued/running job."""
        with self._lock:
            for key, job in list(self._queued.items()):
                if job.job_id == job_id and owner_user_id in (None, "", job.owner_user_id):
                    job.cancelled = True
                    del self._queued[key]
                    self._state_changed.notify_all()
                    break
            else:
                if any(job.job_id == job_id for job in self._running.values()):
                    raise TaskNotCancellable("重复代码检测已开始运行，无法取消")
                raise KeyError(job_id)
        self._finish(job, "cancelled")
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._lock:
            self._closed = True
            timers = list(self._scheduled.values())
            self._scheduled.clear()
            self._state_changed.notify_all()
        for timer in timers:
            timer.cancel()
        self._q.put(None)
        self._worker.join(timeout=timeout)

    # ----------------------------------------------------------------- internals

    @staticmethod
    def _canonical(project: str) -> str:
        try:
            return canonicalize_scope_path(project) or project
        except ValueError:
            return project  # not a path scope: use the name as given

    def _idle_locked(self) -> bool:
        return not (self._scheduled or self._queued or self._running)

    def _schedule_locked(self, key: _ScopeKey, delay: float) -> None:
        timer = threading.Timer(delay, self._dispatch, args=(key,))
        timer.daemon = True
        self._scheduled[key] = timer
        timer.start()

    def _aliases(self, owner_user_id: str, project: str) -> list[str]:
        try:
            return self.catalog_service.equivalent_project_scopes(
                owner_user_id=owner_user_id, project=project, kind="code")
        except Exception:  # noqa: BLE001 - a catalog hiccup must not stop the analysis
            logger.warning("code analysis scope-alias lookup failed project=%s", project,
                           exc_info=True)
            return [project]

    def _scope_completed(self, owner_user_id: str, aliases: Sequence[str]) -> bool:
        """The durable snapshot marker, the same authority search uses. A scope with no
        marker at all predates markers and is judged by its catalog rows alone."""
        reader = getattr(self.catalog_service, "code_scope_visibility", None)
        if reader is None:
            return True
        try:
            marker = reader(owner_user_id=owner_user_id, projects=list(aliases))
        except Exception:  # noqa: BLE001 - cannot prove it is complete, so do not look
            logger.warning("code analysis scope marker unreadable", exc_info=True)
            return False
        return marker is None or marker.get("state") == "completed"

    def _remaining_wait(self, key: _ScopeKey) -> float:
        with self._lock:
            last = self._last_started.get(key)
        if last is None:
            state = self.store.get_state(*key)
            if state is not None and state.last_run_at is not None:
                last = state.last_run_at.replace(tzinfo=timezone.utc).timestamp()
        if last is None:
            return 0.0
        return last + self.min_interval_seconds - self._clock()

    def _dispatch(self, key: _ScopeKey) -> None:
        """Timer thread: it is this scope's turn. Either wait out the interval, drop the
        request (nothing to do), or open the Tasks row and hand the job to the worker."""
        owner, project = key
        job: _Job | None = None
        try:
            wait = self._remaining_wait(key)
            if wait > 0:
                with self._lock:
                    if not self._closed and key in self._scheduled:
                        self._schedule_locked(key, wait)
                return
            aliases = self._aliases(owner, project)
            if self._scope_completed(owner, aliases):
                state = self.store.get_state(owner, project)
                changes = self.store.changed_files(
                    owner, aliases, state.watermark if state else None)
                if changes:
                    job_id = uuid.uuid4().hex[:16]
                    task_id = None
                    if self.tasks is not None:
                        task_id = self.tasks.open(
                            TASK_TYPE,
                            {"job_id": job_id, "project": project,
                             "changed_files": len(changes.changed),
                             "removed_files": len(changes.removed)},
                            0, owner)
                    job = _Job(job_id, owner, project, task_id)
        except Exception:  # noqa: BLE001 - one scope's failure must not kill the timer
            logger.exception("code analysis dispatch failed owner=%s project=%s", owner, project)
        with self._lock:
            self._scheduled.pop(key, None)
            if job is not None and not self._closed:
                self._queued[key] = job
                self._q.put(job)
            self._state_changed.notify_all()

    def _work(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            key = job.key
            with self._lock:
                if job.cancelled:
                    continue
                self._queued.pop(key, None)
                self._running[key] = job
                self._last_started[key] = self._clock()
            if self.tasks is not None:
                self.tasks.started(job.task_id)
            outcome: dict | None = None
            try:
                outcome = self._analyze(job)
                self._finish(job, "completed", result=outcome)
            except Exception as exc:  # noqa: BLE001 - record, never kill the worker
                logger.exception("code analysis failed owner=%s project=%s", *key)
                self._finish(job, "failed", error=str(exc))
            finally:
                with self._lock:
                    self._running.pop(key, None)
                    again = key in self._dirty or (
                        outcome is not None and outcome.get("status") == "discarded")
                    self._dirty.discard(key)
                    if again and not self._closed:
                        self._schedule_locked(key, 0.0)
                    self._state_changed.notify_all()

    def _finish(self, job: _Job, status: str, *, result: dict | None = None,
                error: str | None = None) -> None:
        if self.tasks is not None:
            self.tasks.finished(job.task_id, status, result=result, error=error)

    def _analyze(self, job: _Job) -> dict[str, Any]:
        owner, project = job.key
        aliases = self._aliases(owner, project)
        if not self._scope_completed(owner, aliases):
            return {"status": "skipped", "reason": "the scope is being indexed"}
        started = _utcnow()
        state = self.store.get_state(owner, project)
        changes = self.store.changed_files(owner, aliases, state.watermark if state else None)
        if not changes:
            return {"status": "no_changes"}
        first_analysis = state is None
        touched = [*changes.changed, *changes.removed]
        guard = self.service.scan_guard(
            owner_user_id=owner, projects=aliases,
            still_complete=lambda: self._scope_completed(owner, aliases))
        try:
            scan = scan_changed_files(
                self.service.vector_store, owner_user_id=owner, project=aliases,
                changed_paths=list(changes.changed), max_chunks=self.max_chunks,
                case_sensitive=not is_windows_scope(project), guard=guard)
        except ScopeChangedDuringScan as exc:
            logger.info("code analysis discarded owner=%s project=%s: %s", owner, project, exc)
            return {"status": "discarded", "reason": str(exc)}

        counts = {"changed_files": len(changes.changed),
                  "removed_files": len(changes.removed),
                  "scanned_chunks": scan.scanned_chunks,
                  "first_analysis": first_analysis}
        if scan.too_large:
            text = (
                "Duplicate-code check skipped: the last index change is too large (more "
                f"than {self.max_chunks} chunks) to check automatically. Use "
                "find_duplicate_code for a full scan of this scope.")
            # The removals still apply; the changed files' pairs cannot be recomputed.
            version = self._record(
                owner, project, touched=list(changes.removed), pairs=[], changes=changes,
                started=started, summary={"text": text, "status": "too_large"})
            return {"status": "too_large", "summary": text, "version": version, **counts}

        stored = self.store.pairs_touching(owner, project, touched)
        found = {(pair.path_a, pair.path_b): pair for pair in scan.pairs}
        new_pairs = sorted((pair for key, pair in found.items() if key not in stored), key=_rank)
        gone = sorted(key for key in stored if key not in found)
        text = summarize(
            first_analysis=first_analysis, new_pairs=new_pairs, gone_pairs=len(gone),
            changed_files=len(changes.changed), task_id=job.task_id)
        version = self._record(
            owner, project, touched=touched, pairs=scan.pairs, changes=changes,
            started=started,
            summary=({"text": text, "status": "analyzed", "new": len(new_pairs),
                      "gone": len(gone), "first_analysis": first_analysis,
                      "task_id": job.task_id} if text else None))
        return {
            "status": "analyzed", "summary": text, "version": version,
            "new_pairs": [pair_to_json(pair) for pair in new_pairs[:RESULT_PAIRS]],
            "new_pairs_total": len(new_pairs),
            "gone_pairs": [{"path_a": a, "path_b": b} for a, b in gone[:RESULT_PAIRS]],
            "gone_pairs_total": len(gone),
            **counts,
        }

    def _record(self, owner: str, project: str, *, touched: Sequence[str],
                pairs: Sequence[DuplicateFilePair], changes: ChangedFiles, started: datetime,
                summary: dict | None) -> int:
        return self.store.apply_run(
            owner, project, touched_paths=touched, pairs=pairs,
            watermark=changes.newest, summary=summary, now=started)


# ------------------------------------------------------------------ process singleton

_SCHEDULER: CodeAnalysisScheduler | None = None
_INIT_LOCK = threading.Lock()


def get_code_analysis_scheduler(
    *, service: CodeKBService | None = None,
    catalog_service: IndexCatalogService | None = None,
) -> CodeAnalysisScheduler:
    """The process scheduler. Dependencies only matter for the call that creates it."""
    global _SCHEDULER
    if _SCHEDULER is None:
        with _INIT_LOCK:
            if _SCHEDULER is None:
                _SCHEDULER = CodeAnalysisScheduler(
                    service=service or get_code_kb_service(),
                    catalog_service=catalog_service or get_index_catalog_service(),
                    tasks=QueueTasks())
    return _SCHEDULER


def _cancel_task_job(job_id: str, owner_user_id: str | None) -> bool:
    scheduler = _SCHEDULER  # never construct the scheduler just to cancel
    if scheduler is None:
        raise KeyError(job_id)
    return scheduler.cancel(job_id, owner_user_id)


def build_task_runner(store: TaskStore | None = None) -> QueueTaskRunner:
    return QueueTaskRunner(TASK_TYPE, _cancel_task_job, store=store)


_TASK_RUNNER: QueueTaskRunner | None = None


def get_task_runner() -> QueueTaskRunner:
    global _TASK_RUNNER
    if _TASK_RUNNER is None:
        _TASK_RUNNER = build_task_runner()
    return _TASK_RUNNER


register_runner(TASK_TYPE, get_task_runner)
