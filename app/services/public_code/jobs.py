"""Bounded background jobs for public-code checks and GPU indexing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import threading
import time
import uuid
from typing import Any, Callable, Protocol, Sequence

from app.services.public_code.index_service import PublicCodeIndexProgress
from app.services.public_code.source_catalog import PublicCodeSourceSelector


class PublicCodeJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PublicCodeJob:
    job_id: str
    operation: str
    source_ids: tuple[str, ...]
    total_sources: int
    status: PublicCodeJobStatus = PublicCodeJobStatus.QUEUED
    completed_sources: int = 0
    results: tuple[dict[str, Any], ...] = ()
    errors: dict[str, str] = field(default_factory=dict)
    phase: str = "queued"
    current_source_id: str | None = None
    current_path: str | None = None
    files_done: int = 0
    files_total: int = 0
    chunks_done: int = 0
    message: str | None = None
    source_progress: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def progress(self) -> float:
        if not self.total_sources:
            return 1.0
        completed_equivalent = float(self.completed_sources)
        current_progress = self.source_progress.get(self.current_source_id or "", {})
        if (
            self.status is PublicCodeJobStatus.RUNNING
            and current_progress.get("phase") not in {"completed", "failed"}
            and self.files_total > 0
        ):
            current_fraction = min(self.files_done / self.files_total, 0.99)
            completed_equivalent += max(0.0, current_fraction)
        return min(completed_equivalent / self.total_sources, 1.0)


class PublicCodeControlOperations(Protocol):
    def check_updates(
        self,
        *,
        source_ids: Sequence[str],
        domains: Sequence[str],
        progress_cb: Any = None,
    ) -> Sequence[Any]: ...


class PublicCodeIndexOperations(Protocol):
    def index_source(
        self, source_id: str, progress_cb: Any = None, *, refresh_mirror: bool = True,
        private_owner_user_id: str | None = None, private_project: str | None = None,
    ) -> Any: ...


class PublicCodeJobStore(Protocol):
    def save_job_snapshot(self, snapshot: dict[str, Any], *, max_retained_jobs: int) -> None: ...

    def get_job_snapshot(self, job_id: str) -> dict[str, Any] | None: ...

    def list_job_snapshots(self, *, limit: int) -> Sequence[dict[str, Any]]: ...

    def fail_incomplete_jobs(self, *, reason: str) -> None: ...


class PublicCodeJobService:
    """Return job receipts immediately while bounding network and GPU concurrency."""

    def __init__(
        self,
        *,
        source_registry: PublicCodeSourceSelector,
        control_service: PublicCodeControlOperations,
        index_service: PublicCodeIndexOperations,
        check_workers: int = 2,
        max_retained_jobs: int = 200,
        job_store: PublicCodeJobStore | None = None,
        persistence_interval_seconds: float = 1.0,
    ):
        if check_workers < 1:
            raise ValueError("check_workers must be at least 1")
        if max_retained_jobs < 1:
            raise ValueError("max_retained_jobs must be at least 1")
        if persistence_interval_seconds <= 0:
            raise ValueError("persistence_interval_seconds must be positive")
        self._source_registry = source_registry
        self._control_service = control_service
        self._index_service = index_service
        self._jobs: dict[str, PublicCodeJob] = {}
        self._lock = threading.Lock()
        self._max_retained_jobs = max_retained_jobs
        self._job_store = job_store
        self._persistence_interval_seconds = persistence_interval_seconds
        self._last_persisted_at: dict[str, float] = {}
        if self._job_store is not None:
            self._job_store.fail_incomplete_jobs(
                reason="backend restarted before the public-code job completed"
            )
            for snapshot in self._job_store.list_job_snapshots(limit=max_retained_jobs):
                job = _job_from_snapshot(snapshot)
                self._jobs[job.job_id] = job
        self._check_executor = ThreadPoolExecutor(
            max_workers=check_workers,
            thread_name_prefix="public-code-check-job",
        )
        self._index_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="public-code-index-job",
        )

    def submit_update_check(
        self,
        *,
        source_ids: Sequence[str] = (),
        domains: Sequence[str] = (),
    ) -> PublicCodeJob:
        selected_source_ids = self._resolve_source_ids(source_ids=source_ids, domains=domains)
        job = self._create_job("check_updates", selected_source_ids)
        self._check_executor.submit(self._run_update_check, job.job_id)
        return job

    def submit_index(
        self,
        *,
        source_ids: Sequence[str] = (),
        domains: Sequence[str] = (),
    ) -> PublicCodeJob:
        selected_source_ids = self._resolve_source_ids(source_ids=source_ids, domains=domains)
        job = self._create_job("index", selected_source_ids)
        self._index_executor.submit(self._run_index, job.job_id)
        return job

    def submit_import(
        self,
        *,
        source_id: str,
        prepare: Callable[[Callable[[str], None]], Any],
        private_owner_user_id: str | None = None,
        private_project: str | None = None,
    ) -> PublicCodeJob:
        """Queue a bundle import: `prepare` publishes the mirror (and registers a new
        source), then the source is indexed from that mirror without an upstream fetch.

        Shares the single index worker, so an import never overlaps GPU indexing."""

        job = self._create_job("import", (source_id,))
        self._index_executor.submit(
            self._run_import, job.job_id, prepare, private_owner_user_id, private_project)
        return job

    def get(self, job_id: str) -> PublicCodeJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return _copy_job(job)
        if self._job_store is None:
            return None
        snapshot = self._job_store.get_job_snapshot(job_id)
        return _job_from_snapshot(snapshot) if snapshot is not None else None

    def list_recent(self, *, limit: int = 50) -> tuple[PublicCodeJob, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("public-code job limit must be between 1 and 200")
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )[:limit]
            return tuple(_copy_job(job) for job in jobs)

    def has_active_index(self, source_id: str) -> bool:
        with self._lock:
            return any(
                job.operation in {"index", "import"}
                and source_id in job.source_ids
                and job.status in {PublicCodeJobStatus.QUEUED, PublicCodeJobStatus.RUNNING}
                for job in self._jobs.values()
            )

    def shutdown(self, *, wait: bool = True) -> None:
        self._check_executor.shutdown(wait=wait, cancel_futures=not wait)
        self._index_executor.shutdown(wait=wait, cancel_futures=not wait)

    def _resolve_source_ids(
        self,
        *,
        source_ids: Sequence[str],
        domains: Sequence[str],
    ) -> tuple[str, ...]:
        return tuple(
            source.source_id
            for source in self._source_registry.select(source_ids=source_ids, domains=domains)
        )

    def _create_job(self, operation: str, source_ids: tuple[str, ...]) -> PublicCodeJob:
        job = PublicCodeJob(
            job_id=uuid.uuid4().hex[:16],
            operation=operation,
            source_ids=source_ids,
            total_sources=len(source_ids),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._prune_completed_locked()
            self._persist_locked(job, force=True)
        return replace(job)

    def _run_update_check(self, job_id: str) -> None:
        self._start(job_id)
        try:
            job = self.get(job_id)
            if job is None:
                return
            checks = self._control_service.check_updates(
                source_ids=job.source_ids,
                domains=(),
                progress_cb=lambda check, completed, total: self._record_check_progress(
                    job_id,
                    check=check,
                    completed=completed,
                    total=total,
                ),
            )
            self._finish(
                job_id,
                results=tuple(asdict(check) for check in checks),
                completed_sources=len(job.source_ids),
            )
        except Exception as exc:
            self._finish(job_id, errors={"job": str(exc)})

    def _run_index(self, job_id: str) -> None:
        self._start(job_id)
        job = self.get(job_id)
        if job is None:
            return
        results: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        for completed_sources, source_id in enumerate(job.source_ids, start=1):
            try:
                results.append(asdict(self._index_service.index_source(
                    source_id,
                    progress_cb=lambda progress: self._record_index_progress(job_id, progress),
                )))
            except Exception as exc:
                errors[source_id] = str(exc)
                self._record_source_failure(job_id, source_id, str(exc))
            self._update_progress(
                job_id,
                completed_sources=completed_sources,
                results=results,
                errors=errors,
            )
        self._finish(
            job_id,
            results=tuple(results),
            errors=errors,
            completed_sources=len(job.source_ids),
        )

    def _run_import(
        self,
        job_id: str,
        prepare: Callable[[Callable[[str], None]], Any],
        private_owner_user_id: str | None = None,
        private_project: str | None = None,
    ) -> None:
        self._start(job_id)
        job = self.get(job_id)
        if job is None:
            return
        source_id = job.source_ids[0]
        try:
            prepare(lambda message: self._record_import_progress(job_id, source_id, message))
            reuse = ({"private_owner_user_id": private_owner_user_id,
                      "private_project": private_project} if private_project else {})
            result = self._index_service.index_source(
                source_id,
                progress_cb=lambda progress: self._record_index_progress(job_id, progress),
                refresh_mirror=False,
                **reuse,
            )
        except Exception as exc:
            self._record_source_failure(job_id, source_id, str(exc))
            self._finish(job_id, errors={source_id: str(exc)}, completed_sources=1)
            return
        self._finish(job_id, results=(asdict(result),), completed_sources=1)

    def _record_import_progress(self, job_id: str, source_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.phase = "importing_bundle"
            job.current_source_id = source_id
            job.message = message
            job.source_progress[source_id] = {"phase": "importing_bundle", "message": message}
            self._persist_locked(job, force=True)

    def _start(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = PublicCodeJobStatus.RUNNING
            job.phase = "running"
            job.started_at = time.time()
            self._persist_locked(job, force=True)

    def _record_check_progress(self, job_id: str, *, check: Any, completed: int, total: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.phase = "checking_remote_revisions"
            job.current_source_id = check.source_id
            job.completed_sources = completed
            job.message = f"checked {completed} of {total} source revisions"
            job.source_progress[check.source_id] = {
                "phase": "completed",
                "status": check.status,
                "message": check.message,
            }
            self._persist_locked(job)

    def _record_index_progress(self, job_id: str, progress: PublicCodeIndexProgress) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.phase = progress.phase
            job.current_source_id = progress.source_id
            job.current_path = progress.current_path
            job.files_done = progress.files_done
            job.files_total = progress.files_total
            job.chunks_done = progress.chunks_done
            job.message = progress.message
            job.source_progress[progress.source_id] = asdict(progress)
            self._persist_locked(job)

    def _record_source_failure(self, job_id: str, source_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            progress = dict(job.source_progress.get(source_id, {}))
            progress.update({"phase": "failed", "message": message})
            job.source_progress[source_id] = progress
            self._persist_locked(job)

    def _update_progress(
        self,
        job_id: str,
        *,
        completed_sources: int,
        results: Sequence[dict[str, Any]],
        errors: dict[str, str],
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.completed_sources = completed_sources
            job.results = tuple(results)
            job.errors = dict(errors)
            self._persist_locked(job)

    def _finish(
        self,
        job_id: str,
        *,
        results: tuple[dict[str, Any], ...] = (),
        errors: dict[str, str] | None = None,
        completed_sources: int | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.results = results
            job.errors = dict(errors or {})
            if completed_sources is not None:
                job.completed_sources = completed_sources
            job.finished_at = time.time()
            job.status = (
                PublicCodeJobStatus.FAILED
                if job.errors
                else PublicCodeJobStatus.COMPLETED
            )
            job.phase = "failed" if job.errors else "completed"
            job.current_path = None
            self._persist_locked(job, force=True)
            self._prune_completed_locked()

    def _prune_completed_locked(self) -> None:
        completed_jobs = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in {PublicCodeJobStatus.COMPLETED, PublicCodeJobStatus.FAILED}
            ),
            key=lambda job: job.finished_at or job.created_at,
            reverse=True,
        )
        for job in completed_jobs[self._max_retained_jobs:]:
            self._jobs.pop(job.job_id, None)
            self._last_persisted_at.pop(job.job_id, None)

    def _persist_locked(self, job: PublicCodeJob, *, force: bool = False) -> None:
        if self._job_store is None:
            return
        now = time.time()
        if not force and now - self._last_persisted_at.get(job.job_id, 0.0) < self._persistence_interval_seconds:
            return
        self._job_store.save_job_snapshot(
            _job_snapshot(job),
            max_retained_jobs=self._max_retained_jobs,
        )
        self._last_persisted_at[job.job_id] = now


def _copy_job(job: PublicCodeJob) -> PublicCodeJob:
    return replace(
        job,
        errors=dict(job.errors),
        source_progress={
            source_id: dict(progress)
            for source_id, progress in job.source_progress.items()
        },
    )


def _job_snapshot(job: PublicCodeJob) -> dict[str, Any]:
    snapshot = asdict(job)
    snapshot["status"] = job.status.value
    snapshot["source_ids"] = list(job.source_ids)
    return snapshot


def _job_from_snapshot(snapshot: dict[str, Any]) -> PublicCodeJob:
    values = dict(snapshot)
    values["source_ids"] = tuple(values.get("source_ids") or ())
    values["status"] = PublicCodeJobStatus(values["status"])
    values["results"] = tuple(values.get("results") or ())
    values["errors"] = dict(values.get("errors") or {})
    values["source_progress"] = {
        source_id: dict(progress)
        for source_id, progress in (values.get("source_progress") or {}).items()
    }
    return PublicCodeJob(**values)
