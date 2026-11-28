"""In-process FIFO jobs for directory-level Journal indexing.

Each job is mirrored onto a ``journal_index`` Task row (app/services/pipeline/queue_tasks.py)
so the Tasks page shows it; the queue stays the executor.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum

from app.services.journal.schemas import JournalBulkIndexRequest
from app.services.journal.service import JournalService, get_journal_service
from app.services.index_catalog.schemas import IndexFileRecord, IndexReportRequest
from app.services.index_catalog.service import IndexCatalogService, get_index_catalog_service
from app.services.pipeline.queue_tasks import QueueTaskRunner, QueueTasks
from app.services.pipeline.runner_registry import TaskNotCancellable, register_runner
from app.services.pipeline.store import TaskStore

TASK_TYPE = "journal_index"
# Embedding is the long phase; the last slice of the 0..1 progress is the Qdrant write,
# catalog report and latest-wins sweep, so a job never reads 100% while still writing.
EMBED_PROGRESS_SHARE = 0.95
# The Qdrant write fills the rest of the bar up to here; only completion reports 1.0.
WRITE_PROGRESS_CAP = 0.99


class JournalJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JournalIndexCancelled(Exception):
    """Raised between embed batches (or just before the first Qdrant write) once a
    cancel was accepted. Nothing has been written to Qdrant when this is raised."""


class JournalIndexCommitting(Exception):
    """``cancel()`` refuses: the job already started writing vectors to Qdrant."""


class JournalIndexAlreadyEnded(Exception):
    """``cancel()`` refuses: the job is already completed, failed or cancelled."""


@dataclass
class JournalBulkIndexJob:
    job_id: str
    owner_user_id: str
    project_slug: str
    host: str
    files_total: int
    status: JournalJobStatus = JournalJobStatus.QUEUED
    files_done: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    indexed: int = 0
    progress: float = 0.0
    queue_position: int = 0
    error: str | None = None
    # Titles left out because their body is binary; the rest of the request was indexed.
    skipped_binary: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    task_id: int | None = None
    # Set by cancel() while RUNNING and nothing written yet; honoured by the embed
    # progress hook and by before_write, the only two places a job can stop cleanly.
    cancel_requested: bool = False
    # Set under the lock the instant before_write lets the first Qdrant write proceed;
    # from then on cancel() refuses.
    commit_claimed: bool = False


class JournalBulkIndexQueue:
    """One worker keeps background embeds from competing with interactive retrieval.

    ``tasks`` mirrors jobs onto Task rows; the process singleton passes one, direct
    constructions (tests) record nothing unless they ask.
    """
    def __init__(self, *, service: JournalService | None = None,
                 catalog_service: IndexCatalogService | None = None,
                 tasks: QueueTasks | None = None):
        self.service = service or get_journal_service()
        self.catalog_service = catalog_service or get_index_catalog_service()
        self.tasks = tasks
        self._jobs: dict[str, JournalBulkIndexJob] = {}
        self._requests: dict[str, JournalBulkIndexRequest] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="journal-index-worker", daemon=True)
        self._worker.start()

    def submit(self, request: JournalBulkIndexRequest, *, owner_user_id: str) -> JournalBulkIndexJob:
        with self._lock:
            job_id = request.client_job_id or uuid.uuid4().hex[:16]
            existing = self._jobs.get(job_id)
            if existing is not None:
                if existing.owner_user_id == owner_user_id and existing.project_slug == request.project_slug:
                    return replace(existing)
                raise ValueError("client_job_id is already bound to another journal index request")
            job = JournalBulkIndexJob(job_id=job_id, owner_user_id=owner_user_id,
                                      project_slug=request.project_slug, host=request.host,
                                      files_total=len(request.files))
            self._jobs[job_id] = job
            self._requests[job_id] = request
            self._positions_locked()
        if self.tasks is not None:
            # Chunk total is unknown until chunking ran inside the worker; the first
            # embed progress tick sets it.
            job.task_id = self.tasks.open(
                TASK_TYPE,
                {"job_id": job_id, "project": request.project_slug, "host": request.host,
                 "files_total": job.files_total},
                0, owner_user_id)
        self._q.put(job_id)
        return replace(job)

    def get(self, job_id: str, *, owner_user_id: str) -> JournalBulkIndexJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return replace(job) if job is not None and job.owner_user_id == owner_user_id else None

    def cancel(self, job_id: str, *, owner_user_id: str) -> JournalBulkIndexJob:
        """Cancel one of the caller's jobs.

        queued -> cancelled immediately. running before its first Qdrant write -> a flag
        the embed loop honours at the next batch boundary (the job settles as cancelled
        asynchronously). running after that -> ``JournalIndexCommitting``. ended ->
        ``JournalIndexAlreadyEnded``. Unknown or another owner's job -> ``KeyError``.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.owner_user_id != owner_user_id:
                raise KeyError(job_id)
            if job.status is JournalJobStatus.QUEUED:
                self._requests.pop(job_id, None)
                job.status = JournalJobStatus.CANCELLED
                job.finished_at = time.time()
                self._positions_locked()
            elif job.status is JournalJobStatus.RUNNING:
                if job.commit_claimed:
                    raise JournalIndexCommitting(job_id)
                job.cancel_requested = True
                return replace(job)
            else:
                raise JournalIndexAlreadyEnded(job_id)
            snapshot = replace(job)
        if self.tasks is not None:
            self.tasks.finished(snapshot.task_id, "cancelled")
        return snapshot

    def _positions_locked(self) -> None:
        pos = 0
        for job in self._jobs.values():
            if job.status is JournalJobStatus.QUEUED:
                pos += 1
                job.queue_position = pos
            else:
                job.queue_position = 0

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs[job_id]
                request = self._requests.pop(job_id, None)
                if request is None:  # cancelled while queued
                    continue
                job.status = JournalJobStatus.RUNNING
                job.started_at = time.time()
                self._positions_locked()
            if self.tasks is not None:
                self.tasks.started(job.task_id)

            def embed_progress(done: int, total: int, job=job) -> None:
                with self._lock:
                    if job.cancel_requested and not job.commit_claimed:
                        raise JournalIndexCancelled(job.job_id)
                    job.chunks_total = total
                    job.chunks_done = done
                    job.progress = EMBED_PROGRESS_SHARE * (done / total if total else 1.0)
                if self.tasks is not None:
                    self.tasks.progress(job.task_id, done, total)

            def before_write(job=job) -> None:
                with self._lock:
                    if job.cancel_requested:
                        raise JournalIndexCancelled(job.job_id)
                    job.commit_claimed = True

            def write_progress(written: int, total: int, job=job) -> None:
                with self._lock:
                    job.progress = min(
                        WRITE_PROGRESS_CAP,
                        EMBED_PROGRESS_SHARE
                        + (1.0 - EMBED_PROGRESS_SHARE) * (written / total if total else 1.0))
                if self.tasks is not None:
                    self.tasks.heartbeat(job.task_id)

            try:
                result = self.service.index_docs_bulk(
                    project_slug=request.project_slug, owner_user_id=job.owner_user_id,
                    files=request.files, client=request.client,
                    progress=embed_progress, before_write=before_write,
                    write_progress=write_progress)
                # Publish catalog and latest-wins sweep only after durable Qdrant
                # upserts.  A failed vector job leaves previous complete docs live.
                report_files = [dict(row) for row in request.report_files]
                chunk_counts = result.get("chunk_counts", {})
                for row in report_files:
                    if row.get("rel_path") in chunk_counts:
                        row["chunk_count"] = chunk_counts[row["rel_path"]]
                self.catalog_service.report(IndexReportRequest(
                    host=request.host, project=request.project_slug, root_dir=request.root_dir,
                    files=[IndexFileRecord.model_validate(row) for row in report_files],
                    replace=request.replace, patterns=request.patterns, max_depth=request.max_depth,
                ), owner_user_id=job.owner_user_id)
                if request.replace:
                    self.service.reconcile_doc_scope(
                        project_slug=request.project_slug, live_titles=request.live_titles,
                        owner_user_id=job.owner_user_id, max_depth=request.max_depth,
                        patterns=request.patterns)
                with self._lock:
                    job.files_done = result["files"]
                    job.chunks_total = result["chunks"]
                    job.chunks_done = result["chunks"]
                    job.indexed = result["indexed"]
                    job.skipped_binary = list(result.get("skipped_binary") or [])
                    job.progress = 1.0
                    job.finished_at = time.time()
                    job.status = JournalJobStatus.COMPLETED
                if self.tasks is not None:
                    self.tasks.finished(job.task_id, "completed", result={
                        "files": job.files_done, "chunks": job.chunks_total,
                        "indexed": job.indexed, "skipped_binary": job.skipped_binary})
            except JournalIndexCancelled:
                # Only raised while commit_claimed is False: nothing reached Qdrant.
                with self._lock:
                    job.finished_at = time.time()
                    job.status = JournalJobStatus.CANCELLED
                if self.tasks is not None:
                    self.tasks.finished(job.task_id, "cancelled")
            except Exception as exc:  # keep worker alive after a bad document/job
                with self._lock:
                    job.error = str(exc)
                    job.finished_at = time.time()
                    job.status = JournalJobStatus.FAILED
                if self.tasks is not None:
                    self.tasks.finished(job.task_id, "failed", error=str(exc))


_QUEUE: JournalBulkIndexQueue | None = None


def get_journal_index_queue() -> JournalBulkIndexQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = JournalBulkIndexQueue(tasks=QueueTasks())
    return _QUEUE


def _cancel_task_job(job_id: str, owner_user_id: str | None) -> bool:
    # Never construct the queue (and its Qdrant-backed service) just to cancel: a
    # process that never accepted a job has nothing to cancel.
    queue_ = _QUEUE
    if queue_ is None:
        raise KeyError(job_id)
    try:
        queue_.cancel(job_id, owner_user_id=owner_user_id or "")
    except JournalIndexCommitting as exc:
        raise TaskNotCancellable("Journal 索引已开始写入 Qdrant，无法取消") from exc
    except JournalIndexAlreadyEnded:
        return False
    return True


def build_task_runner(store: TaskStore | None = None) -> QueueTaskRunner:
    return QueueTaskRunner(TASK_TYPE, _cancel_task_job, store=store)


_TASK_RUNNER: QueueTaskRunner | None = None


def get_task_runner() -> QueueTaskRunner:
    global _TASK_RUNNER
    if _TASK_RUNNER is None:
        _TASK_RUNNER = build_task_runner()
    return _TASK_RUNNER


register_runner(TASK_TYPE, get_task_runner)
