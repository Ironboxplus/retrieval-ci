"""In-process FIFO jobs for directory-level Journal indexing."""
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


class JournalJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


class JournalBulkIndexQueue:
    """One worker keeps background embeds from competing with interactive retrieval."""
    def __init__(self, *, service: JournalService | None = None,
                 catalog_service: IndexCatalogService | None = None):
        self.service = service or get_journal_service()
        self.catalog_service = catalog_service or get_index_catalog_service()
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
        self._q.put(job_id)
        return replace(job)

    def get(self, job_id: str, *, owner_user_id: str) -> JournalBulkIndexJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return replace(job) if job is not None and job.owner_user_id == owner_user_id else None

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
                request = self._requests.pop(job_id)
                job.status = JournalJobStatus.RUNNING
                job.started_at = time.time()
                self._positions_locked()
            try:
                result = self.service.index_docs_bulk(
                    project_slug=request.project_slug, owner_user_id=job.owner_user_id,
                    files=request.files)
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
                    job.progress = 1.0
                    job.finished_at = time.time()
                    job.status = JournalJobStatus.COMPLETED
            except Exception as exc:  # keep worker alive after a bad document/job
                with self._lock:
                    job.error = str(exc)
                    job.finished_at = time.time()
                    job.status = JournalJobStatus.FAILED


_QUEUE: JournalBulkIndexQueue | None = None


def get_journal_index_queue() -> JournalBulkIndexQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = JournalBulkIndexQueue()
    return _QUEUE
