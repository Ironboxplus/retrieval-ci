"""
任务运行时观测与细粒度控制辅助函数。

这层专注于：
1. 任务事件日志
2. 子任务状态
3. 进程级资源采样
4. crawl 任务聚合结果重建
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import psutil
except Exception:  # pragma: no cover - graceful fallback when psutil is missing
    psutil = None

from sqlalchemy import func

from core.db import get_db_session
from core.db.models import (
    Task,
    TaskStatus as DBTaskStatus,
    TaskEvent,
    TaskSubtask,
    TaskSubtaskStatus,
    TaskResourceSample,
    TaskWorkflowEdge,
    TaskWorkflowEdgeStatus,
    CrawlRecord,
)


def _db_task_id(task_id: Any) -> Optional[int]:
    try:
        if task_id is None:
            return None
        task_id_str = str(task_id)
        if not task_id_str.isdigit():
            return None
        return int(task_id_str)
    except Exception:
        return None


def _task_row_exists(db, task_id: int) -> bool:
    return db.query(Task.id).filter(Task.id == task_id).first() is not None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def record_task_event(
    task_id: Any,
    *,
    event_type: str,
    message: str = "",
    payload: Optional[Dict[str, Any]] = None,
    level: str = "info",
    stage: Optional[str] = None,
    subtask_id: Optional[int] = None,
) -> Optional[int]:
    db_task_id = _db_task_id(task_id)
    if db_task_id is None:
        return None

    with get_db_session() as db:
        if not _task_row_exists(db, db_task_id):
            return None
        event = TaskEvent(
            task_id=db_task_id,
            subtask_id=subtask_id,
            event_type=event_type,
            level=level,
            stage=stage,
            message=message,
            payload_json=_json_safe(payload) if payload else None,
        )
        db.add(event)
        db.flush()
        return event.id


def upsert_task_subtask(
    task_id: Any,
    *,
    subtask_key: str,
    subtask_type: str,
    name: Optional[str] = None,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    total: Optional[int] = None,
    retry_count: Optional[int] = None,
    max_retries: Optional[int] = None,
    message: Optional[str] = None,
    stage: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    db_task_id = _db_task_id(task_id)
    if db_task_id is None or not subtask_key:
        return None

    with get_db_session() as db:
        if not _task_row_exists(db, db_task_id):
            return None
        subtask = db.query(TaskSubtask).filter(
            TaskSubtask.task_id == db_task_id,
            TaskSubtask.subtask_key == subtask_key,
        ).first()

        if subtask is None:
            subtask = TaskSubtask(
                task_id=db_task_id,
                subtask_key=subtask_key,
                subtask_type=subtask_type,
            )
            db.add(subtask)
            db.flush()

        if name is not None:
            subtask.name = name
        if status is not None:
            subtask.status = TaskSubtaskStatus(status)
        if progress is not None:
            subtask.progress = progress
        if total is not None:
            subtask.total = total
        if retry_count is not None:
            subtask.retry_count = retry_count
        if max_retries is not None:
            subtask.max_retries = max_retries
        if message is not None:
            subtask.message = message
        if stage is not None:
            subtask.stage = stage
        if result is not None:
            subtask.result_json = _json_safe(result)
        if error is not None:
            subtask.error_message = error
        if metadata is not None:
            merged = dict(subtask.metadata_json or {})
            merged.update(_json_safe(metadata))
            subtask.metadata_json = merged

        now = datetime.now()
        if started_at is not None:
            subtask.started_at = started_at
        elif status == TaskSubtaskStatus.running.value and subtask.started_at is None:
            subtask.started_at = now

        if completed_at is not None:
            subtask.completed_at = completed_at
        elif status in {
            TaskSubtaskStatus.completed.value,
            TaskSubtaskStatus.failed.value,
            TaskSubtaskStatus.cancelled.value,
            TaskSubtaskStatus.skipped.value,
        }:
            subtask.completed_at = now

        db.flush()
        return {
            "id": subtask.id,
            "task_id": subtask.task_id,
            "subtask_key": subtask.subtask_key,
            "subtask_type": subtask.subtask_type,
            "status": subtask.status.value if subtask.status else None,
            "retry_count": subtask.retry_count or 0,
        }


def get_task_subtask(task_id: Any, subtask_id: int) -> Optional[TaskSubtask]:
    db_task_id = _db_task_id(task_id)
    if db_task_id is None:
        return None

    with get_db_session() as db:
        return db.query(TaskSubtask).filter(
            TaskSubtask.task_id == db_task_id,
            TaskSubtask.id == subtask_id,
        ).first()


def record_task_resource_sample(
    task_id: Any,
    *,
    phase: Optional[str] = None,
    subtask_id: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    if psutil is None:
        return None

    db_task_id = _db_task_id(task_id)
    if db_task_id is None:
        return None

    process = psutil.Process(os.getpid())
    try:
        memory = process.memory_info()
        open_file_count = None  # W49: psutil.open_files() blocks 30s+ on Windows (NtQuerySystemInformation)
    except Exception:
        memory = None
        open_file_count = None

    with get_db_session() as db:
        if not _task_row_exists(db, db_task_id):
            return None
        sample = TaskResourceSample(
            task_id=db_task_id,
            subtask_id=subtask_id,
            phase=phase,
            pid=process.pid,
            cpu_percent=process.cpu_percent(interval=None),
            rss_bytes=getattr(memory, "rss", None),
            vms_bytes=getattr(memory, "vms", None),
            thread_count=process.num_threads(),
            open_file_count=open_file_count,
            extra_json=_json_safe(extra) if extra else None,
        )
        db.add(sample)
        db.flush()
        return sample.id


def upsert_workflow_edge(
    task_id: Any,
    *,
    edge_key: str,
    from_key: str,
    to_key: str,
    status: str = "active",
    label: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    db_task_id = _db_task_id(task_id)
    if db_task_id is None or not edge_key:
        return None

    with get_db_session() as db:
        if not _task_row_exists(db, db_task_id):
            return None
        edge = db.query(TaskWorkflowEdge).filter(
            TaskWorkflowEdge.task_id == db_task_id,
            TaskWorkflowEdge.edge_key == edge_key,
        ).first()

        if edge is None:
            edge = TaskWorkflowEdge(
                task_id=db_task_id,
                edge_key=edge_key,
                from_key=from_key,
                to_key=to_key,
            )
            db.add(edge)
            db.flush()

        edge.from_key = from_key
        edge.to_key = to_key
        edge.status = TaskWorkflowEdgeStatus(status)
        if label is not None:
            edge.label = label
        if metadata is not None:
            merged = dict(edge.metadata_json or {})
            merged.update(_json_safe(metadata))
            edge.metadata_json = merged
        db.flush()
        return {
            "id": edge.id,
            "task_id": edge.task_id,
            "edge_key": edge.edge_key,
            "status": edge.status.value if edge.status else None,
            "from_key": edge.from_key,
            "to_key": edge.to_key,
        }


def get_task_workflow_graph(task_id: int) -> Dict[str, Any]:
    with get_db_session() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            return {"nodes": [], "edges": [], "summary": {}}

        subtasks = db.query(TaskSubtask).filter(
            TaskSubtask.task_id == task_id
        ).order_by(TaskSubtask.id.asc()).all()
        edges = db.query(TaskWorkflowEdge).filter(
            TaskWorkflowEdge.task_id == task_id
        ).order_by(TaskWorkflowEdge.id.asc()).all()

        nodes_payload: List[Dict[str, Any]] = []
        for subtask in subtasks:
            meta = dict(subtask.metadata_json or {})
            retriable = bool(meta.get("retriable", False))
            node_actions = ["inspect"]
            if retriable and subtask.status in {
                TaskSubtaskStatus.failed,
                TaskSubtaskStatus.completed,
                TaskSubtaskStatus.cancelled,
                TaskSubtaskStatus.skipped,
            }:
                node_actions.append("retry")
            if subtask.status in {TaskSubtaskStatus.pending, TaskSubtaskStatus.failed}:
                node_actions.append("skip")

            nodes_payload.append(
                {
                    "id": subtask.id,
                    "key": subtask.subtask_key,
                    "type": subtask.subtask_type,
                    "name": subtask.name or subtask.subtask_key,
                    "status": subtask.status.value if subtask.status else "pending",
                    "stage": subtask.stage,
                    "message": subtask.message,
                    "progress": subtask.progress or 0,
                    "total": subtask.total or 0,
                    "retry_count": subtask.retry_count or 0,
                    "max_retries": subtask.max_retries or 3,
                    "error_message": subtask.error_message,
                    "metadata": meta,
                    "result": subtask.result_json or {},
                    "allowed_actions": node_actions,
                    "created_at": subtask.created_at.isoformat() if subtask.created_at else None,
                    "updated_at": subtask.updated_at.isoformat() if subtask.updated_at else None,
                    "started_at": subtask.started_at.isoformat() if subtask.started_at else None,
                    "completed_at": subtask.completed_at.isoformat() if subtask.completed_at else None,
                }
            )

        edges_payload: List[Dict[str, Any]] = []
        for edge in edges:
            edge_actions = ["inspect"]
            if edge.status == TaskWorkflowEdgeStatus.active:
                edge_actions.append("disable")
            elif edge.status == TaskWorkflowEdgeStatus.disabled:
                edge_actions.append("enable")

            edges_payload.append(
                {
                    "id": edge.id,
                    "key": edge.edge_key,
                    "source": edge.from_key,
                    "target": edge.to_key,
                    "label": edge.label,
                    "status": edge.status.value if edge.status else "active",
                    "metadata": edge.metadata_json or {},
                    "allowed_actions": edge_actions,
                    "created_at": edge.created_at.isoformat() if edge.created_at else None,
                    "updated_at": edge.updated_at.isoformat() if edge.updated_at else None,
                }
            )

        summary = {
            "task_id": task.id,
            "task_type": task.task_type,
            "task_status": task.status.value if task.status else "unknown",
            "node_count": len(nodes_payload),
            "edge_count": len(edges_payload),
            "running_nodes": len([node for node in nodes_payload if node["status"] == "running"]),
            "failed_nodes": len([node for node in nodes_payload if node["status"] == "failed"]),
            "completed_nodes": len([node for node in nodes_payload if node["status"] == "completed"]),
        }
        return {"nodes": nodes_payload, "edges": edges_payload, "summary": summary}


def purge_task_runtime_rows(task_id: int, *, db) -> None:
    """Delete all task-scoped runtime rows explicitly.

    This protects task deletion even if SQLite foreign-key enforcement was
    previously disabled and old rows are still present.
    """
    db.query(TaskEvent).filter(TaskEvent.task_id == task_id).delete(synchronize_session=False)
    db.query(TaskResourceSample).filter(TaskResourceSample.task_id == task_id).delete(synchronize_session=False)
    db.query(TaskWorkflowEdge).filter(TaskWorkflowEdge.task_id == task_id).delete(synchronize_session=False)
    db.query(CrawlRecord).filter(CrawlRecord.task_id == task_id).delete(synchronize_session=False)
    db.query(TaskSubtask).filter(TaskSubtask.task_id == task_id).delete(synchronize_session=False)


def get_task_resource_summary(task_id: int) -> Dict[str, Any]:
    with get_db_session() as db:
        peak_rss, peak_cpu, samples = db.query(
            func.max(TaskResourceSample.rss_bytes),
            func.max(TaskResourceSample.cpu_percent),
            func.count(TaskResourceSample.id),
        ).filter(TaskResourceSample.task_id == task_id).one()

        latest = db.query(TaskResourceSample).filter(
            TaskResourceSample.task_id == task_id
        ).order_by(TaskResourceSample.created_at.desc(), TaskResourceSample.id.desc()).first()

        return {
            "sample_count": samples or 0,
            "peak_rss_bytes": peak_rss,
            "peak_cpu_percent": peak_cpu,
            "latest": None if latest is None else {
                "id": latest.id,
                "phase": latest.phase,
                "pid": latest.pid,
                "cpu_percent": latest.cpu_percent,
                "rss_bytes": latest.rss_bytes,
                "vms_bytes": latest.vms_bytes,
                "thread_count": latest.thread_count,
                "open_file_count": latest.open_file_count,
                "extra": latest.extra_json or {},
                "created_at": latest.created_at.isoformat() if latest.created_at else None,
            },
        }


def rebuild_crawl_task_result(task_id: int, *, db=None) -> Dict[str, Any]:
    session = db
    use_context = session is None

    def _build(session):
        subtasks = session.query(TaskSubtask).filter(
            TaskSubtask.task_id == task_id,
            TaskSubtask.subtask_type == "crawl_url",
        ).order_by(TaskSubtask.id.asc()).all()

        if not subtasks:
            records = session.query(CrawlRecord).filter(CrawlRecord.task_id == task_id).all()
            if records:
                s_ok = [r for r in records if r.status == DBCrawlStatus.success]
                s_fail = [r for r in records if r.status == DBCrawlStatus.failed]
                return {
                    "status": "success" if s_ok else ("failed" if s_fail else "pending"),
                    "total": len(records),
                    "success_count": len(s_ok),
                    "failed_count": len(s_fail),
                    "blocked_count": 0, "captcha_count": 0, "index_failed_count": 0,
                    "terminal_count": len(s_ok) + len(s_fail),
                    "success_results": [{"url": r.url, "title": r.title, "content_length": r.content_length, "doc_id": r.doc_id} for r in s_ok],
                    "failed_results": [{"url": r.url, "error": r.error_message} for r in s_fail],
                    "blocked_results": [], "captcha_results": [], "index_failed_results": [],
                    "indexed_docs": None,
                }

        success_results: List[Dict[str, Any]] = []
        failed_results: List[Dict[str, Any]] = []
        blocked_results: List[Dict[str, Any]] = []
        captcha_results: List[Dict[str, Any]] = []
        indexed_docs: List[Dict[str, Any]] = []
        index_failed_results: List[Dict[str, Any]] = []

        for subtask in subtasks:
            result = dict(subtask.result_json or {})
            status = result.get("status")

            if subtask.status == TaskSubtaskStatus.completed:
                success_entry = {
                    "url": result.get("url") or subtask.subtask_key,
                    "title": result.get("title"),
                    "content_length": result.get("content_length"),
                    "doc_id": result.get("doc_id"),
                    "indexed": result.get("indexed"),
                    "index_error": result.get("index_error"),
                    "chunks_count": result.get("chunks_count"),
                    "crawled_at": result.get("crawled_at"),
                    "retry_count": result.get("retry_count"),
                    "crawl_time": result.get("crawl_time"),
                    "status_code": result.get("status_code"),
                    "proxy_used": result.get("proxy_used"),
                }
                success_results.append(success_entry)
                if result.get("indexed") and result.get("doc_id"):
                    indexed_docs.append(
                        {
                            "doc_id": result.get("doc_id"),
                            "url": success_entry["url"],
                            "chunks_count": result.get("chunks_count"),
                        }
                    )
                elif result.get("index_error"):
                    index_failed_results.append(
                        {
                            "url": success_entry["url"],
                            "doc_id": result.get("doc_id"),
                            "error": result.get("index_error"),
                            "chunks_count": result.get("chunks_count"),
                        }
                    )
                continue

            failure_entry = {
                "url": result.get("url") or subtask.subtask_key,
                "error": subtask.error_message or result.get("error"),
                "retry_count": result.get("retry_count"),
                "crawl_time": result.get("crawl_time"),
                "status_code": result.get("status_code"),
            }
            if status == "blocked":
                blocked_results.append(failure_entry)
            elif status == "captcha":
                failure_entry["captcha_type"] = result.get("captcha_type")
                captcha_results.append(failure_entry)
            elif subtask.status in {TaskSubtaskStatus.failed, TaskSubtaskStatus.cancelled, TaskSubtaskStatus.skipped}:
                failed_results.append(failure_entry)

        total = len(subtasks)
        terminal_count = sum(
            1
            for subtask in subtasks
            if subtask.status in {
                TaskSubtaskStatus.completed,
                TaskSubtaskStatus.failed,
                TaskSubtaskStatus.cancelled,
                TaskSubtaskStatus.skipped,
            }
        )

        result_status = "success"
        if failed_results or blocked_results or captcha_results or index_failed_results:
            result_status = "partial_success" if success_results else "failed"

        return {
            "status": result_status,
            "total": total,
            "success_count": len(success_results),
            "failed_count": len(failed_results),
            "blocked_count": len(blocked_results),
            "captcha_count": len(captcha_results),
            "index_failed_count": len(index_failed_results),
            "terminal_count": terminal_count,
            "success_results": success_results,
            "failed_results": failed_results,
            "blocked_results": blocked_results,
            "captcha_results": captcha_results,
            "index_failed_results": index_failed_results,
            "indexed_docs": indexed_docs or None,
        }

    if use_context:
        with get_db_session() as session:
            return _build(session)
    return _build(session)


def sync_crawl_task_rollup(task_id: int, *, status: Optional[str] = None, db=None) -> Dict[str, Any]:
    session = db
    use_context = session is None

    def _sync(session):
        task = session.query(Task).filter(Task.id == task_id).first()
        if task is None:
            return {}

        result = rebuild_crawl_task_result(task_id, db=session)
        total = result.get("total", 0) or 0
        terminal_count = result.get("terminal_count", 0) or 0

        task.total = total
        task.progress = terminal_count
        task.result_json = result
        if status is not None:
            task.status = DBTaskStatus(status)
            if status in {
                DBTaskStatus.completed.value,
                DBTaskStatus.failed.value,
                DBTaskStatus.cancelled.value,
                DBTaskStatus.timeout.value,
            }:
                task.completed_at = datetime.now()
        session.flush()
        return result

    if use_context:
        with get_db_session() as session:
            return _sync(session)
    return _sync(session)
