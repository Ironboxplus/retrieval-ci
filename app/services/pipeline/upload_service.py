"""UploadPipelineRunner — create, launch, and resume single-document upload pipelines.

The upload analog of PipelineRunner (service.py). It drives the SAME generic
PipelineOrchestrator over build_upload_pipeline()'s [extract, index] stages in
a daemon thread, keyed on doc_id instead of (venue, year).

Why a separate runner rather than overloading PipelineRunner: the venue runner's
launch/resume are coupled to (venue, year) params and a venue ctx-factory, and it
is HOT (resumes in-flight ingests on startup). Keeping uploads in their own small
runner avoids touching that path. The ONE thing they must share is the process
StageLockSet — so an upload's `index` stage (resource_key="qdrant") serializes
with a venue `ingest` on the same Qdrant lock (W58/W60 commit-charge safety).
get_upload_runner() wires that shared lock set from the venue singleton.
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional

from loguru import logger

from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.registry import UPLOAD_STAGE_KEYS, build_upload_pipeline
from app.services.pipeline.stage_lock import StageLockSet
from app.services.pipeline.store import TaskStore
from app.services.pipeline.upload_context import UploadStageContext

UPLOAD_PIPELINE = "upload_pipeline"


class UploadPipelineRunner:
    def __init__(
        self,
        *,
        store: Optional[TaskStore] = None,
        pipeline_builder: Callable[[UploadStageContext], list] = build_upload_pipeline,
        context_factory: Optional[Callable[..., UploadStageContext]] = None,
        stage_locks: Optional[StageLockSet] = None,
    ):
        self.store = store or TaskStore()
        self._build = pipeline_builder
        self._ctx_factory = context_factory or self._default_ctx
        self._active: set[int] = set()
        self._lock = threading.Lock()
        self.stage_locks = stage_locks or StageLockSet()

    @staticmethod
    def _default_ctx(*, doc_id, file_path, store, task_id) -> UploadStageContext:
        return UploadStageContext(
            doc_id=doc_id, file_path=file_path, store=store, task_id=task_id,
        )

    def start_upload(self, doc_id: int, file_path: str, *, wait: bool = False) -> dict:
        """Create an `upload_pipeline` task for one document and drive it.

        Guard: a second submission for the SAME doc_id while one is in-flight
        returns a conflict pointing at the existing task (duplicate work serves
        no purpose). Returns {status, task_id}.
        """
        for r in self.store.list_running(UPLOAD_PIPELINE):
            if (r.get("params") or {}).get("doc_id") == doc_id:
                logger.warning(f"upload.start.duplicate existing={r['id']} doc_id={doc_id}")
                return {
                    "status": "conflict",
                    "task_id": r["id"],
                    "error": f"an upload pipeline (#{r['id']}) is already running for doc {doc_id}",
                }

        params = {"doc_id": doc_id, "file_path": str(file_path)}
        task_id = self.store.create_task(
            task_type=UPLOAD_PIPELINE, params=params, total=len(UPLOAD_STAGE_KEYS),
        )
        logger.info(f"upload.start task={task_id} doc_id={doc_id} file={file_path}")
        self._launch(task_id, doc_id, str(file_path), wait=wait)
        return {"status": "started", "task_id": task_id}

    def resume_running(self, *, wait: bool = False) -> List[int]:
        """Relaunch orchestrators for running upload pipelines (startup hook, W61)."""
        resumed: List[int] = []
        try:
            running = self.store.list_running(UPLOAD_PIPELINE)
        except Exception:
            logger.exception("upload.resume.query_failed")
            return resumed

        for t in running:
            tid = t["id"]
            with self._lock:
                if tid in self._active:
                    continue
            params = t.get("params") or {}
            doc_id, file_path = params.get("doc_id"), params.get("file_path")
            if doc_id is None or not file_path:
                logger.warning(f"upload.resume.skip task={tid} missing doc_id/file_path")
                continue
            logger.info(f"upload.resume task={tid} doc_id={doc_id}")
            self._launch(tid, doc_id, file_path, wait=wait)
            resumed.append(tid)
        return resumed

    def _launch(self, task_id: int, doc_id: int, file_path: str, *, wait: bool = False) -> None:
        def target():
            with self._lock:
                self._active.add(task_id)
            try:
                ctx = self._ctx_factory(
                    doc_id=doc_id, file_path=file_path, store=self.store, task_id=task_id,
                )
                specs = self._build(ctx)
                PipelineOrchestrator(task_id, specs, ctx, self.store, locks=self.stage_locks).run()
            except Exception as e:
                logger.exception(f"upload.crash task={task_id}: {e}")
                try:
                    self.store.set_task_status(task_id, "failed", error=f"orchestrator crash: {e}")
                except Exception:
                    logger.exception("upload.crash.status_write_failed")
            finally:
                with self._lock:
                    self._active.discard(task_id)

        if wait:
            target()
        else:
            threading.Thread(target=target, daemon=True, name=f"upload-{task_id}").start()


_runner: Optional[UploadPipelineRunner] = None
_runner_lock = threading.Lock()


def get_upload_runner() -> UploadPipelineRunner:
    """Process-wide singleton. Shares the venue runner's StageLockSet so an
    upload's `index` stage serializes with venue `ingest` on the same `qdrant`
    resource lock (W58/W60 commit-charge safety)."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                from app.services.pipeline.service import get_pipeline_runner
                _runner = UploadPipelineRunner(stage_locks=get_pipeline_runner().stage_locks)
    return _runner
