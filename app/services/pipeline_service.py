"""
PipelineService — manages paper collection, extraction, and indexing pipelines.

All long-running operations return immediately with a task_id.
Progress is tracked via get_status() / get_task_status(task_id).
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import settings
from core.db.database import get_db_session
from core.db.models import Paper as DBPaper, Document as DBDocument, Chunk as DBChunk
from sqlalchemy import func


@dataclass
class TaskState:
    task_id: str
    task_type: str
    status: str = "pending"
    progress: float = 0.0
    processed: int = 0
    total: int = 0
    indexed: int = 0
    failed: int = 0
    started_at: float = 0.0
    finished_at: Optional[float] = None
    error: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {message}")
        if len(self.logs) > 500:
            self.logs = self.logs[-300:]


_active_tasks: dict[str, TaskState] = {}


class PipelineService:

    def __init__(self):
        self._corpus_root = Path(settings.data_dir) / "papers" / "ccf_a_main_conference_corpus"
        self._manifest_path = self._corpus_root / "manifests" / "paper_manifest.json"

    def get_task_status(self, task_id: str) -> Optional[dict[str, Any]]:
        task = _active_tasks.get(task_id)
        if task is None:
            return None
        elapsed = (task.finished_at or time.time()) - task.started_at
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "processed": task.processed,
            "total": task.total,
            "indexed": task.indexed,
            "failed": task.failed,
            "elapsed_seconds": round(elapsed, 1),
            "error": task.error,
            "details": task.details,
            "logs": task.logs[-50:],
        }

    def list_tasks(self) -> list[dict[str, Any]]:
        return [self.get_task_status(tid) for tid in _active_tasks]

    # `start_reindex` (the standalone /api/pipeline/reindex entry point) was
    # deleted in F1-a. `_run_reindex` stays: it is the shared background-thread
    # worker that `start_batch_index` below still uses to drive /api/papers/
    # batch-index, so deleting it would break the route the frontend refactor
    # plan explicitly keeps.
    def _run_reindex(self, task: TaskState, papers: list,
                     batch_size: int, force: bool):
        try:
            from pipeline.paper_crawler.integration.index_integration import PaperIndexer, PaperIndexConfig

            config = PaperIndexConfig(
                embedding_device="cuda",
                require_gpu=False,
            )
            task.log(f"Initializing PaperIndexer (device=cuda, batch_size={batch_size})")
            indexer = PaperIndexer(config=config)
            task.log(f"Starting reindex of {task.total} papers")

            for i in range(0, len(papers), batch_size):
                batch = papers[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(papers) + batch_size - 1) // batch_size
                try:
                    result = asyncio.run(indexer.batch_index_papers(
                        batch, skip_whoosh=True, per_batch_timeout_sec=600,
                    ))
                    batch_indexed = result.get("indexed", 0)
                    batch_failed = result.get("failed", 0)
                    task.indexed += batch_indexed
                    task.failed += batch_failed
                    task.log(f"Batch {batch_num}/{total_batches}: +{batch_indexed} indexed, {batch_failed} failed")

                    for r in result.get("results", []):
                        if r.get("status") == "failed":
                            task.log(f"  FAIL: {r.get('paper_id','?')} — {r.get('error','?')[:100]}")

                except Exception as e:
                    task.failed += len(batch)
                    task.details["last_error"] = str(e)
                    task.log(f"Batch {batch_num}/{total_batches} ERROR: {str(e)[:200]}")

                task.processed += len(batch)
                task.progress = round(task.processed / task.total * 100, 1) if task.total else 0

            task.status = "completed"
            task.log(f"Reindex completed: {task.indexed} indexed, {task.failed} failed")
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.log(f"FATAL: {str(e)[:300]}")
        finally:
            task.finished_at = time.time()

    def start_batch_index(self, *, venue: str = None, year: int = None,
                          paper_ids: list[str] = None, batch_size: int = 50,
                          force: bool = False, dry_run: bool = False) -> dict[str, Any]:
        """Batch index papers matching filter criteria (venue/year/IDs)."""
        from pipeline.paper_crawler.runtime_files import load_json_with_recovery
        from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper

        if not self._manifest_path.exists():
            return {"error": "Manifest not found", "status": "error"}

        records = load_json_with_recovery(self._manifest_path, default_value=[])

        if venue:
            vu = venue.upper()
            records = [r for r in records
                       if (r.get("conference") or r.get("venue") or "").upper() == vu]
        if year:
            records = [r for r in records if r.get("year") == year]
        if paper_ids:
            id_set = set(paper_ids)
            records = [r for r in records if r.get("paper_id") in id_set]

        papers = [CorpusPaper.from_manifest_record(r) for r in records]
        eligible = [p for p in papers if p.markdown_path and Path(p.markdown_path).exists()]

        if dry_run:
            return {
                "status": "dry_run",
                "total_matched": len(records),
                "eligible": len(eligible),
            }

        for task in _active_tasks.values():
            if task.task_type in ("batch_index", "reindex") and task.status == "running":
                return {"error": "Index task already running",
                        "task_id": task.task_id, "status": "conflict"}

        task_id = str(uuid.uuid4())[:8]
        task = TaskState(
            task_id=task_id, task_type="batch_index",
            status="running", total=len(eligible),
            started_at=time.time(),
        )
        _active_tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_reindex,
            args=(task, eligible, batch_size, force),
            daemon=True,
        )
        thread.start()

        return {
            "task_id": task_id,
            "status": "started",
            "total_matched": len(records),
            "eligible": len(eligible),
            "batch_size": batch_size,
        }
