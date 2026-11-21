"""Async job queue for code-KB indexing.

POST /api/code/index used to embed+upsert inline; a large repo (5k+ chunks ≈ minutes
of BGE-M3 work) held the HTTP request open past client timeouts and read as a hang
(W156). This queue decouples submit from work: submit() returns a job immediately,
one daemon worker thread drains jobs FIFO — serializing embed work on the single warm
GPU embedder — and progress is pollable per owner.

The worker queue and its poll receipts remain in-process by design (deliberately not
Huey or the pipeline orchestrator): the embedder lives in the backend process and this
is a single-stage, per-user unit of work. Scope visibility is deliberately different:
before an in-place vector mutation, Index Catalog persists a visibility marker. A
worker/backend crash leaves that marker preparing or recovery_required, so scoped and
owner-wide search fails closed until a later full snapshot completes or an explicit
recovery succeeds. A vanished in-process receipt is therefore never visibility proof
and must not be presented as a job clients can poll. app/services/pipeline's
StageSpec/TaskStore engine remains the right tool for multi-stage resumable ingest.
"""
from __future__ import annotations

import logging
import hashlib
import json
import queue as _queue
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum

from app.services.code_kb.schemas import CodeIndexRequest
from app.services.code_kb.service import CodeKBService, get_code_kb_service
from app.services.index_catalog.schemas import IndexReportRequest
from app.services.index_catalog.service import IndexCatalogService, get_index_catalog_service

logger = logging.getLogger("code_kb.jobs")

MAX_JOBS_KEPT = 200  # finished-job history cap (per process, all owners)
_GENERATION_LOCK_STRIPES = 256
_CATALOG_WRITE_ATTEMPTS = 3
_CATALOG_RETRY_BASE_SECONDS = 0.1


def _hash_field(digest, value) -> None:
    """Feed one unambiguous UTF-8 field without building a whole-request buffer."""
    if value is None:
        encoded = b"n"
    elif isinstance(value, bool):
        encoded = b"b1" if value else b"b0"
    elif isinstance(value, int):
        encoded = b"i" + str(value).encode("ascii")
    else:
        encoded = b"s" + str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _request_fingerprint(request: CodeIndexRequest) -> str:
    """Stable request hash with bounded temporary memory for very large repositories."""
    digest = hashlib.sha256()
    for value in (request.project, request.host, request.root_dir, request.incremental):
        _hash_field(digest, value)
    _hash_field(digest, "<manifest>")
    if request.manifest is None:
        _hash_field(digest, None)
    else:
        for path in request.manifest:
            _hash_field(digest, path)
    _hash_field(digest, "<removed_paths>")
    for path in request.removed_paths:
        _hash_field(digest, path)
    for file in request.files:
        _hash_field(digest, "<file>")
        for value in (file.rel_path, file.language, file.content_hash):
            _hash_field(digest, value)
        for chunk in file.chunks:
            _hash_field(digest, "<chunk>")
            for value in (
                chunk.start_line, chunk.end_line, chunk.label, chunk.content, chunk.code,
            ):
                _hash_field(digest, value)
    _hash_field(digest, "<report>")
    if request.report is None:
        _hash_field(digest, None)
    else:
        _hash_field(digest, request.report.replace)
        for file in request.report.files:
            for value in (
                file.rel_path, file.kind, file.collection, file.language,
                file.chunk_count, file.content_hash, file.size_bytes,
                json.dumps(file.metadata, sort_keys=True, separators=(",", ":")),
            ):
                _hash_field(digest, value)
    return digest.hexdigest()


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CodeIndexCancelled(Exception):
    """Raised from the mutation-fence callback when a cancel request landed while this
    job was still RUNNING with no Qdrant mutation started. Never raised once
    ``mutation_started`` is True — see ``CodeIndexJobQueue.cancel``."""


class CodeIndexJobCommitting(Exception):
    """``cancel()`` refuses: the job is RUNNING and has already started its first Qdrant
    write. Interrupting now would leave a half-committed scope (AGENTS.md Qdrant
    invariants), so the caller must wait for it to finish instead."""


class CodeIndexJobAlreadyEnded(Exception):
    """``cancel()`` refuses: the job is already COMPLETED/FAILED/CANCELLED."""


@dataclass
class CodeIndexJob:
    job_id: str
    owner_user_id: str
    project: str
    host: str
    files_total: int
    chunks_total: int
    removed_total: int = 0
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0          # 0..1, embed+upsert fraction
    chunks_done: int = 0
    indexed: int = 0               # points actually upserted (set on completion)
    duplicates_updated: int = 0    # 方案2 cross-scope copies refreshed (set on completion)
    error: str | None = None
    queue_position: int = 0        # 0 = running/finished, 1 = next, ...
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    request_fingerprint: str = ""
    scope_generation: int = 0
    # Set under the service owner fence immediately before the first vector mutation.
    # A caller may search a previous completed snapshot while a refresh only embeds, but
    # must never read the in-place 256-point Qdrant batches once mutation begins.
    mutation_started: bool = False
    mutations_committed: bool = False
    # Set by cancel() while RUNNING and no mutation has started yet. The mutation fence
    # (before every batch's Qdrant write) is the only place that honors it — once it
    # observes mutation_started=True the flag is permanently moot for this job.
    cancel_requested: bool = False
    # Set atomically (same locked section as the cancel_requested check) the instant
    # the mutation fence decides to proceed to its first write — BEFORE releasing the
    # lock for the SQLite preparing-marker call. mutation_started itself only flips
    # after that write succeeds, so there is a real window where cancel_requested is
    # already moot but mutation_started still reads False; commit_claimed closes that
    # window for cancel() without pretending a marker-write failure ever touched
    # Qdrant (see _before_scope_mutation).
    commit_claimed: bool = False
    # Fan-out can mutate another project that shares one physical file.  These are
    # visibility scopes, not merely job input metadata: every member must be fenced.
    visibility_projects: set[str] = field(default_factory=set)
    fan_out_failed_projects: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CodeScopeSearchState:
    """In-process companion state for one Code KB scope's durable visibility marker.

    The SQLite marker is authoritative across backend restarts; this queue state makes
    a live writer visible before its terminal catalog transition.  It is a gate, not a
    revision system: Code KB has no immutable active-revision pointer yet, so after a
    write starts the only honest result is ``preparing`` rather than an accidental mix
    of old and newly-upserted point batches.
    """

    active_job_id: str
    mutation_started: bool
    previous_completed_snapshot_available: bool
    recovery_required: bool = False


@dataclass(frozen=True)
class _UnsafeScope:
    """Process-local fallback while the durable catalog marker remains authoritative."""

    job_id: str
    host: str


class CodeIndexJobQueue:
    """FIFO queue + single worker thread over CodeKBService.index."""

    def __init__(self, *, service: CodeKBService | None = None,
                 catalog_service: IndexCatalogService | None = None):
        self.service = service or get_code_kb_service()
        self.catalog_service = catalog_service or get_index_catalog_service()
        self._jobs: dict[str, CodeIndexJob] = {}   # insertion-ordered (submit order = FIFO order)
        self._job_aliases: dict[str, str] = {}
        self._requests: dict[str, CodeIndexRequest] = {}
        self._scope_generations: dict[tuple[str, str], int] = {}
        self._unsafe_scopes: dict[tuple[str, str], _UnsafeScope] = {}
        self._generation_locks = [
            threading.RLock() for _ in range(_GENERATION_LOCK_STRIPES)]
        self._lock = threading.Lock()
        self._q: _queue.Queue[str | None] = _queue.Queue()
        self._worker = threading.Thread(target=self._run, name="code-index-worker", daemon=True)
        self._worker.start()

    # ------------------------------- public API ------------------------------- #

    def submit(self, request: CodeIndexRequest, *, owner_user_id: str) -> CodeIndexJob:
        fingerprint = _request_fingerprint(request)
        with self._lock:
            scope_key = (owner_user_id, request.project)
            current_generation = self._scope_generations.get(scope_key, 0)
            if request.client_job_id:
                resolved_id = self._job_aliases.get(
                    request.client_job_id, request.client_job_id)
                existing_receipt = self._jobs.get(resolved_id)
                if existing_receipt is not None:
                    if (existing_receipt.owner_user_id == owner_user_id
                            and existing_receipt.project == request.project
                            and existing_receipt.request_fingerprint == fingerprint
                            and existing_receipt.scope_generation == current_generation):
                        return replace(existing_receipt, job_id=request.client_job_id)
                    raise ValueError("client_job_id is already bound to another index request")
            active = next((
                existing for existing in self._jobs.values()
                if existing.owner_user_id == owner_user_id
                and existing.project == request.project
                and existing.request_fingerprint == fingerprint
                and existing.scope_generation == current_generation
                and existing.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            ), None)
            if active is not None:
                if request.client_job_id:
                    self._job_aliases[request.client_job_id] = active.job_id
                    return replace(active, job_id=request.client_job_id)
                logger.info(
                    "code_kb job reused: %s owner=%s project=%s status=%s",
                    active.job_id, owner_user_id, request.project, active.status.value)
                return active

            job = CodeIndexJob(
                job_id=request.client_job_id or uuid.uuid4().hex[:16],
                owner_user_id=owner_user_id,
                project=request.project,
                host=request.host,
                files_total=len(request.files),
                chunks_total=sum(len(f.chunks) for f in request.files),
                removed_total=len(request.removed_paths),
                request_fingerprint=fingerprint,
                scope_generation=current_generation,
                visibility_projects={request.project},
            )
            self._jobs[job.job_id] = job
            self._requests[job.job_id] = request
            self._refresh_positions_locked()
            self._trim_history_locked()
        self._q.put(job.job_id)
        logger.info("code_kb job queued: %s owner=%s project=%s chunks=%d position=%d",
                    job.job_id, owner_user_id, job.project, job.chunks_total, job.queue_position)
        return job

    def get(self, job_id: str, *, owner_user_id: str | None = None) -> CodeIndexJob | None:
        with self._lock:
            resolved_id = self._job_aliases.get(job_id, job_id)
            job = self._jobs.get(resolved_id)
            if job is None:
                return None
            if owner_user_id is not None and job.owner_user_id != owner_user_id:
                return None  # cross-owner reads must look like a miss, not a 403 oracle
            return replace(job, job_id=job_id) if resolved_id != job_id else job

    def cancel(self, job_id: str, *, owner_user_id: str) -> CodeIndexJob:
        """Cancel one of the caller's jobs (F2-a).

        queued -> removed from the queue and marked cancelled immediately (the worker's
        existing ``request is None`` skip, already used by ``cancel_project``, discards it
        when its turn comes).
        running, no mutation started yet -> a flag is set; the mutation fence (called
        before every batch's Qdrant write, including the first) observes it and raises
        ``CodeIndexCancelled`` instead of writing, so the worker settles this job as
        cancelled with nothing ever committed and no recovery marker needed.
        running, mutation already started -> ``CodeIndexJobCommitting``: a partial write
        cannot be safely interrupted.
        already COMPLETED/FAILED/CANCELLED -> ``CodeIndexJobAlreadyEnded``.
        Unknown job_id or another owner's job -> ``KeyError`` (cross-owner must look like
        a miss, matching ``get()``, not reveal that the id belongs to someone else).
        """
        with self._lock:
            resolved_id = self._job_aliases.get(job_id, job_id)
            job = self._jobs.get(resolved_id)
            if job is None or job.owner_user_id != owner_user_id:
                raise KeyError(job_id)
            if job.status is JobStatus.QUEUED:
                self._requests.pop(resolved_id, None)
                job.status = JobStatus.CANCELLED
                job.finished_at = time.time()
                self._refresh_positions_locked()
                logger.info("code_kb job cancelled while queued: %s owner=%s project=%s",
                            resolved_id, owner_user_id, job.project)
                return replace(job, job_id=job_id) if resolved_id != job_id else job
            if job.status is JobStatus.RUNNING:
                # commit_claimed (not mutation_started) is the real fence: it flips in
                # the SAME locked section the mutation fence uses to check
                # cancel_requested, before the fence releases the lock for its SQLite
                # marker write. Checking mutation_started here instead would leave a
                # window where a cancel lands after the fence has already committed to
                # writing but before the marker write finishes — this call would then
                # report a cancel that never actually stopped the job.
                if job.commit_claimed:
                    raise CodeIndexJobCommitting(job_id)
                job.cancel_requested = True
                logger.info("code_kb job cancel requested: %s owner=%s project=%s",
                            resolved_id, owner_user_id, job.project)
                return replace(job, job_id=job_id) if resolved_id != job_id else job
            raise CodeIndexJobAlreadyEnded(job_id)

    def list_jobs(self, *, owner_user_id: str, limit: int = 20,
                  project: str | None = None, active_only: bool = False) -> list[CodeIndexJob]:
        with self._lock:
            mine: list[CodeIndexJob] = []
            for j in reversed(self._jobs.values()):  # newest first
                if (j.owner_user_id == owner_user_id
                        and (project is None or j.project == project)
                        and (not active_only or j.status in (JobStatus.QUEUED, JobStatus.RUNNING))):
                    mine.append(j)
                    if len(mine) >= limit:
                        break
            return mine

    def scope_search_state(
        self,
        *,
        owner_user_id: str,
        projects: Sequence[str],
    ) -> CodeScopeSearchState | None:
        """Return the active visibility state for one equivalent-project scope set.

        ``projects`` contains the canonical Windows spelling plus any catalog-proven
        legacy spelling.  A completed job from the same process is evidence that a prior
        snapshot existed; callers can supplement it with durable catalog evidence after
        a backend restart.
        """
        scope_set = {project for project in projects if project}
        if not scope_set:
            return None
        with self._lock:
            return self._search_state_locked(owner_user_id=owner_user_id, scope_set=scope_set)

    def owner_search_state(self, *, owner_user_id: str) -> CodeScopeSearchState | None:
        """Return an in-process visibility state for any scope in one owner's search.

        Owner-wide search has no project filter, so it can otherwise return a partial
        fan-out target.  The route takes the same owner fence before consuming this.
        """
        with self._lock:
            return self._search_state_locked(owner_user_id=owner_user_id, scope_set=None)

    def inventory_scope_states(
        self,
        *,
        owner_user_id: str | None = None,
        host: str | None = None,
        project: str | None = None,
    ) -> dict[tuple[str, str, str], dict[str, object]]:
        """Overlay active Code KB jobs onto durable inventory rows.

        Catalog rows are written only after a job succeeds, while a policy is registered
        before the client scan starts.  This small in-process overlay therefore prevents
        the inventory from conflating ``preparing`` with a completed vector snapshot.
        """
        with self._lock:
            grouped: dict[tuple[str, str, str], list[CodeIndexJob]] = {}
            for job in self._jobs.values():
                if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
                    continue
                if owner_user_id is not None and job.owner_user_id != owner_user_id:
                    continue
                if host is not None and job.host != host:
                    continue
                for visible_project in self._visibility_projects_locked(job):
                    if project is not None and visible_project != project:
                        continue
                    grouped.setdefault(
                        (job.owner_user_id, job.host, visible_project), []).append(job)

            states: dict[tuple[str, str, str], dict[str, object]] = {}
            for key, jobs in grouped.items():
                jobs.sort(key=self._active_job_sort_key)
                current = jobs[0]
                prior_complete = self._previous_complete_locked(
                    current.owner_user_id, {key[2]})
                states[key] = {
                    "active_index_job_id": current.job_id,
                    "previous_completed_snapshot_available": prior_complete,
                    "recovery_required": False,
                }

            # A failed in-place write is still a visibility event after the worker leaves
            # RUNNING.  The durable catalog marker is used by production routes after a
            # restart; retain this local mirror for queue status and lightweight fakes.
            for (unsafe_owner, unsafe_project), unsafe in self._unsafe_scopes.items():
                if owner_user_id is not None and unsafe_owner != owner_user_id:
                    continue
                if host is not None and unsafe.host != host:
                    continue
                if project is not None and unsafe_project != project:
                    continue
                states[(unsafe_owner, unsafe.host, unsafe_project)] = {
                    "active_index_job_id": unsafe.job_id,
                    "previous_completed_snapshot_available": self._previous_complete_locked(
                        unsafe_owner, {unsafe_project}),
                    "recovery_required": True,
                }
            return states

    @staticmethod
    def _active_job_sort_key(job: CodeIndexJob) -> tuple[int, float]:
        return (
            0 if job.status is JobStatus.RUNNING and job.mutation_started else
            1 if job.status is JobStatus.RUNNING else 2,
            job.created_at,
        )

    @staticmethod
    def _visibility_projects_locked(job: CodeIndexJob) -> set[str]:
        return job.visibility_projects or {job.project}

    def _previous_complete_locked(
        self, owner_user_id: str, scope_set: set[str] | None,
    ) -> bool:
        return any(
            job.owner_user_id == owner_user_id
            and job.status is JobStatus.COMPLETED
            and any(
                scope_set is None or visible_project in scope_set
                for visible_project in self._visibility_projects_locked(job)
                if visible_project not in job.fan_out_failed_projects
            )
            for job in self._jobs.values()
        )

    def _search_state_locked(
        self, *, owner_user_id: str, scope_set: set[str] | None,
    ) -> CodeScopeSearchState | None:
        unsafe = [
            (project, state)
            for (owner, project), state in self._unsafe_scopes.items()
            if owner == owner_user_id and (scope_set is None or project in scope_set)
        ]
        # A prior failed mutation wins over a newly queued recovery job.  Until that
        # recovery completes, both the old and new process must fail closed.
        if unsafe:
            project, state = sorted(unsafe, key=lambda item: item[0])[0]
            return CodeScopeSearchState(
                active_job_id=state.job_id,
                mutation_started=True,
                previous_completed_snapshot_available=self._previous_complete_locked(
                    owner_user_id, {project}),
                recovery_required=True,
            )

        active = [
            job for job in self._jobs.values()
            if job.owner_user_id == owner_user_id
            and job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            and any(
                scope_set is None or visible_project in scope_set
                for visible_project in self._visibility_projects_locked(job)
            )
        ]
        if not active:
            return None
        active.sort(key=self._active_job_sort_key)
        current = active[0]
        return CodeScopeSearchState(
            active_job_id=current.job_id,
            mutation_started=current.mutation_started,
            previous_completed_snapshot_available=self._previous_complete_locked(
                owner_user_id, scope_set),
        )

    def _mark_visibility_unsafe_locked(self, job: CodeIndexJob) -> None:
        for project in self._visibility_projects_locked(job):
            self._unsafe_scopes[(job.owner_user_id, project)] = _UnsafeScope(
                job_id=job.job_id, host=job.host)

    def _clear_recovered_visibility_locked(self, job: CodeIndexJob) -> None:
        for project in self._visibility_projects_locked(job):
            if project not in job.fan_out_failed_projects:
                self._unsafe_scopes.pop((job.owner_user_id, project), None)

    def cancel_project(self, *, owner_user_id: str, project: str) -> int:
        """Fence prior jobs for a scope before explicit deletion."""
        with self._generation_lock(owner_user_id, project):
            with self._lock:
                key = (owner_user_id, project)
                self._scope_generations[key] = self._scope_generations.get(key, 0) + 1
                self._unsafe_scopes.pop(key, None)
                cancelled = 0
                for job_id, job in self._jobs.items():
                    if (job.owner_user_id == owner_user_id and job.project == project
                            and job.status is JobStatus.QUEUED):
                        self._requests.pop(job_id, None)
                        job.error = "cancelled by project deletion"
                        job.finished_at = time.time()
                        job.status = JobStatus.FAILED
                        cancelled += 1
                self._refresh_positions_locked()
                return cancelled

    def shutdown(self, timeout: float = 10.0) -> None:
        self._q.put(None)
        self._worker.join(timeout=timeout)

    # ------------------------------- worker loop ------------------------------ #

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs.get(job_id)
                request = self._requests.pop(job_id, None)
                if job is None or request is None:
                    continue
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                self._refresh_positions_locked()

            def _progress(done: int, total: int) -> None:
                # The same callback is checked immediately before Qdrant mutation.
                # A project deletion increments the generation while an old job may
                # still be embedding; fail that job before it can resurrect the scope.
                with self._lock:
                    self._assert_generation_locked(job)
                    # A batch's progress_cb fires AFTER that batch's own mutation fence
                    # (see _MutationFence / CodeKBService.index), so by the time this
                    # runs for batch N, mutation_started is already True if N>=1. This
                    # check only ever fires for the (rare) case of a batch whose fence
                    # somehow didn't run — the real cancellation gate is the fence
                    # itself, in _before_scope_mutation below.
                    if job.cancel_requested and not job.mutation_started:
                        raise CodeIndexCancelled(job.job_id)
                    job.chunks_done = done
                    job.progress = done / total if total else 1.0

            def _before_mutation() -> None:
                self._before_scope_mutation(job, job.project)

            def _before_fan_out_target_mutation(target_project: str) -> None:
                # Fan-out writes are real mutations of the target scope.  Persist its
                # marker before the target upsert, not merely under the source project.
                self._before_scope_mutation(job, target_project)

            def _fan_out_target_mutation_failed(target_project: str) -> None:
                with self._lock:
                    job.visibility_projects.add(target_project)
                    job.fan_out_failed_projects.add(target_project)
                    self._unsafe_scopes[(job.owner_user_id, target_project)] = _UnsafeScope(
                        job_id=job.job_id, host=job.host)
                # The pre-write marker is already durable; upgrade it to the more useful
                # recovery state when SQLite is available.  A marker write failure here
                # must not erase the original preparing marker or allow a vector search.
                self._mark_scope_recovery_required(job, target_project)

            def _after_mutation() -> None:
                # Called while the generation fence is still held, after primary
                # upserts, stale pruning, and cross-scope fan-out all succeeded.
                self._write_catalog(request, owner_user_id=job.owner_user_id)
                with self._lock:
                    completed_projects = sorted(
                        self._visibility_projects_locked(job) - job.fan_out_failed_projects)
                for project in completed_projects:
                    self._mark_scope_completed(job, project)
                with self._lock:
                    self._assert_generation_locked(job)
                    job.mutations_committed = True
                    self._clear_recovered_visibility_locked(job)

            @contextmanager
            def _mutation_guard():
                # A project delete linearizes before all writes or after all writes,
                # including cross-scope fan-out. It can never fence a half-committed job.
                with self._generation_lock(job.owner_user_id, job.project):
                    with self._lock:
                        self._assert_generation_locked(job)
                    yield

            try:
                resp = self.service.index(
                    request, owner_user_id=job.owner_user_id, progress_cb=_progress,
                    before_mutation_cb=_before_mutation,
                    before_fan_out_target_mutation_cb=_before_fan_out_target_mutation,
                    fan_out_target_mutation_failed_cb=_fan_out_target_mutation_failed,
                    after_mutation_cb=_after_mutation,
                    mutation_guard=_mutation_guard)
                with self._lock:
                    self._assert_generation_locked(job)
                    job.indexed = resp.indexed
                    job.duplicates_updated = resp.duplicates_updated
                    job.chunks_done = job.chunks_total
                    job.progress = 1.0
                    # Timestamp before status: a reader that observes COMPLETED also sees
                    # the terminal timestamp and the already-published catalog report.
                    job.finished_at = time.time()
                    job.status = JobStatus.COMPLETED
                logger.info("code_kb job done: %s indexed=%d in %.1fs",
                            job_id, job.indexed, job.finished_at - (job.started_at or job.finished_at))
            except CodeIndexCancelled:
                # Only raised while mutation_started is still False (the fence's own
                # guard), so nothing was ever written — no recovery marker, no unsafe
                # scope, unlike the generic failure branch below.
                with self._lock:
                    job.finished_at = time.time()
                    job.status = JobStatus.CANCELLED
                logger.info("code_kb job cancelled: %s owner=%s project=%s",
                            job_id, job.owner_user_id, job.project)
            except Exception as exc:  # noqa: BLE001 — job must record, not kill the worker
                with self._lock:
                    visibility_projects: set[str] = set()
                    if job.mutation_started and not job.mutations_committed:
                        self._mark_visibility_unsafe_locked(job)
                        visibility_projects = set(self._visibility_projects_locked(job))
                    job.error = str(exc)
                    job.finished_at = time.time()
                    job.status = JobStatus.FAILED
                for project in visibility_projects:
                    self._mark_scope_recovery_required(job, project)
                logger.exception("code_kb job failed: %s owner=%s project=%s",
                                 job_id, job.owner_user_id, job.project)

    def _before_scope_mutation(self, job: CodeIndexJob, project: str) -> None:
        """Persist a fail-closed marker before this scope's first vector mutation.

        This callback runs while ``CodeKBService.scope_snapshot`` is held.  If SQLite
        cannot accept the marker, abort before Qdrant changes; after a process crash the
        marker remains and a rebuilt queue cannot mistake partial batches for complete.
        """
        with self._lock:
            self._assert_generation_locked(job)
            # Last chance to honor a cancel request: once this call proceeds past here,
            # cancel() will refuse with 409 (CodeIndexJobCommitting) instead. Subsequent
            # batches of the same job hit this function again, but mutation_started is
            # already True by then, so the flag is a no-op for them — cancellation only
            # ever stops a job before its very first write.
            if job.cancel_requested and not job.mutation_started:
                raise CodeIndexCancelled(job.job_id)
            # commit_claimed must flip in THIS same locked section, before the lock is
            # released for the SQLite call below — not after it succeeds. The marker
            # write can take real wall time; a cancel() landing in that gap must not
            # see mutation_started=False and report a cancel that never actually
            # stopped the job. mutation_started itself still only flips after the
            # write succeeds (below) — a marker-write failure must remain a clean
            # abort (nothing was ever written to Qdrant), not a visibility-unsafe scope.
            job.commit_claimed = True
        self._mark_scope_preparing(job, project)
        with self._lock:
            self._assert_generation_locked(job)
            job.visibility_projects.add(project)
            job.mutation_started = True

    def _set_durable_scope_visibility(
        self, method_name: str, job: CodeIndexJob, project: str,
    ) -> None:
        method = getattr(self.catalog_service, method_name, None)
        if method is None:
            # Small test fakes predate the durable control-plane contract. Production
            # IndexCatalogService always exposes these methods; their absence must not
            # change the in-process safety tests into a different service API.
            return
        method(
            owner_user_id=job.owner_user_id,
            host=job.host,
            project=project,
            job_id=job.job_id,
        )

    def _mark_scope_preparing(self, job: CodeIndexJob, project: str) -> None:
        self._set_durable_scope_visibility("mark_code_scope_preparing", job, project)

    def _mark_scope_completed(self, job: CodeIndexJob, project: str) -> None:
        self._set_durable_scope_visibility("mark_code_scope_completed", job, project)

    def _mark_scope_recovery_required(self, job: CodeIndexJob, project: str) -> None:
        # The process-local unsafe entry was installed before this best-effort durable
        # upgrade is attempted.  SQLite can fail precisely while the primary vector
        # mutation is already failing; letting that secondary error escape the worker
        # would strand all later jobs and erase the in-process fail-closed boundary.
        try:
            self._set_durable_scope_visibility(
                "mark_code_scope_recovery_required", job, project)
        except Exception as exc:  # noqa: BLE001 - retain unsafe state and keep FIFO alive
            logger.error(
                "code_kb recovery marker unavailable; retaining process-local unsafe "
                "scope owner=%s project=%s job=%s error_type=%s",
                job.owner_user_id,
                project,
                job.job_id,
                type(exc).__name__,
            )

    def _write_catalog(self, request: CodeIndexRequest, *, owner_user_id: str) -> None:
        """Persist bundled inventory after all vector mutations have succeeded."""
        if request.report is None:
            return
        report = IndexReportRequest(
            host=request.host,
            project=request.project,
            root_dir=request.root_dir,
            files=request.report.files,
            replace=request.report.replace,
        )
        delay = _CATALOG_RETRY_BASE_SECONDS
        for attempt in range(1, _CATALOG_WRITE_ATTEMPTS + 1):
            try:
                self.catalog_service.report(report, owner_user_id=owner_user_id)
                if request.removed_paths:
                    self.catalog_service.delete_files(
                        owner_user_id=owner_user_id,
                        project=request.project,
                        host=request.host,
                        kind="code",
                        rel_paths=request.removed_paths,
                    )
                return
            except Exception:
                if attempt == _CATALOG_WRITE_ATTEMPTS:
                    raise
                logger.warning(
                    "code_kb catalog report failed; retrying attempt=%d/%d owner=%s project=%s",
                    attempt,
                    _CATALOG_WRITE_ATTEMPTS,
                    owner_user_id,
                    request.project,
                    exc_info=True,
                )
                time.sleep(delay)
                delay *= 2

    # ------------------------------- internals -------------------------------- #

    def _assert_generation_locked(self, job: CodeIndexJob) -> None:
        if job.mutations_committed:
            return
        generation = self._scope_generations.get(
            (job.owner_user_id, job.project), 0)
        if generation != job.scope_generation:
            raise RuntimeError("index job superseded by project deletion")

    def _generation_lock(self, owner_user_id: str, project: str) -> threading.RLock:
        index = hash((owner_user_id, project)) % len(self._generation_locks)
        return self._generation_locks[index]

    def _refresh_positions_locked(self) -> None:
        # Queue order == submit order == dict insertion order; positions are derivable.
        pos = 0
        for job in self._jobs.values():
            if job.status is JobStatus.QUEUED:
                pos += 1
                job.queue_position = pos
            else:
                job.queue_position = 0

    def _trim_history_locked(self) -> None:
        # Drop oldest FINISHED jobs past the cap; never drop queued/running ones.
        while len(self._jobs) > MAX_JOBS_KEPT:
            victim = next((jid for jid, j in self._jobs.items()
                           if j.status in (JobStatus.COMPLETED, JobStatus.FAILED,
                                            JobStatus.CANCELLED)), None)
            if victim is None:
                return  # everything live — don't trim
            self._jobs.pop(victim)
            self._requests.pop(victim, None)
            stale_aliases = [
                alias for alias, target in self._job_aliases.items() if target == victim
            ]
            for alias in stale_aliases:
                self._job_aliases.pop(alias, None)


_QUEUE: CodeIndexJobQueue | None = None
# Index routes may reach the lazy singleton concurrently through FastAPI worker
# threads. One process owns one queue/worker because queue history and the owner fence
# are process-local; constructing two would split receipts and permit competing writes.
_QUEUE_INIT_LOCK = threading.Lock()


def get_code_index_queue() -> CodeIndexJobQueue:
    global _QUEUE
    if _QUEUE is None:
        with _QUEUE_INIT_LOCK:
            if _QUEUE is None:
                _QUEUE = CodeIndexJobQueue()
    return _QUEUE


def get_existing_code_index_queue() -> CodeIndexJobQueue | None:
    """Return the process queue only when an index route already initialized it.

    Read-only inventory/search diagnostics must not construct a Qdrant-backed queue solely
    to prove that there is no active job.
    """
    return _QUEUE
