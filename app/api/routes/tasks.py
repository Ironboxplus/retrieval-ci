"""
任务管理路由。

在原有任务 CRUD 基础上，新增：
1. 任务详情聚合视图
2. 任务事件列表
3. 任务资源采样列表
4. 子任务列表/详情
5. SSE 流式任务进度
6. crawl_urls 的单子任务重试
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from core.db import get_db_session
from core.db.database import get_db
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
    CrawlStatus as DBCrawlStatus,
)
from app.services.task_runtime import (
    get_task_resource_summary,
    rebuild_crawl_task_result,
    sync_crawl_task_rollup,
    get_task_workflow_graph,
    purge_task_runtime_rows,
)
from app.services.pipeline.progress import pipeline_progress
from core.timestamps import from_local_naive, from_utc_naive, local_naive_iso, utc_naive_iso
from app.services.pipeline.runner_registry import TaskNotCancellable, get_runner
from app.tasks.base import cancel_task, TASKS_DIR
from app.tasks.crawl_tasks import enqueue_crawl_urls

router = APIRouter()

# Server-Sent Events, WHATWG HTML Living Standard section 9.2.
SSE_RETRY_MS = 3000          # `retry:` field, 9.2.5: reconnection delay the client should use
SSE_KEEPALIVE_SECONDS = 15.0  # comment lines keep idle proxies/EventSource from timing out


def _sse_start_event_id(after_event_id: int, last_event_id: Optional[str]) -> int:
    """First task-event id to send after.

    A reconnecting EventSource sends the last `id:` it saw as the `Last-Event-ID`
    request header (9.2.4). It is client-controlled text: anything that is not a
    non-negative integer is ignored rather than rejected, and the explicit
    `after_event_id` query parameter still applies.
    """
    try:
        header_id = int(last_event_id) if last_event_id is not None else 0
    except ValueError:
        header_id = 0
    return max(after_event_id, header_id, 0)


def _sse_message(event: str, data: Any, event_id: Optional[int] = None) -> str:
    """One event block. `id:` is only set for task-event: an event without an `id`
    field leaves the client's last-event-ID unchanged (9.2.6), so snapshot / resource
    / final never move the resume cursor."""
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _enqueue_retry(task: Task, retry_urls: List[str]) -> None:
    """Re-enqueue supported failed tasks using their existing DB task id."""
    params = task.params_json or {}

    enqueue_crawl_urls(
        retry_urls,
        use_proxy=bool(params.get("use_proxy", False)),
        auto_index=bool(params.get("auto_index", False)),
        max_concurrent=int(params.get("max_concurrent", 5) or 5),
        task_id=str(task.id),
    )


def _retry_urls_for_task(task: Task) -> List[str]:
    """Return the URL subset that should be retried for supported task types."""
    if task.task_type != "crawl_urls":
        raise HTTPException(400, f"Retry not supported for task type: {task.task_type}")

    params = task.params_json or {}
    if task.status == DBTaskStatus.failed:
        urls = params.get("urls") or []
    elif task.status == DBTaskStatus.completed:
        failed_results = (task.result_json or {}).get("failed_results") or []
        blocked_results = (task.result_json or {}).get("blocked_results") or []
        captcha_results = (task.result_json or {}).get("captcha_results") or []
        urls = [
            item.get("url")
            for item in [*failed_results, *blocked_results, *captcha_results]
            if isinstance(item, dict) and item.get("url")
        ]
    else:
        urls = []

    if not isinstance(urls, list) or not urls:
        raise HTTPException(400, "No retryable crawl_urls entries found for this task")

    return urls


def _task_duration_seconds(task: Task) -> Optional[float]:
    if task.started_at is None:
        return None
    end_time = task.completed_at or datetime.now()
    return round((end_time - task.started_at).total_seconds(), 3)


def _task_health(task: Task) -> Dict[str, Any]:
    now = datetime.now()
    heartbeat_age = None
    issues: List[str] = []
    is_terminal = task.status in {
        DBTaskStatus.completed,
        DBTaskStatus.failed,
        DBTaskStatus.cancelled,
        DBTaskStatus.timeout,
    }
    if task.last_heartbeat is not None:
        heartbeat_age = round((now - task.last_heartbeat).total_seconds(), 3)
        if task.status == DBTaskStatus.running and heartbeat_age > 120:
            issues.append(f"heartbeat stale: {heartbeat_age}s")

    duration = _task_duration_seconds(task)
    if task.status == DBTaskStatus.running and task.started_at is not None and duration is not None:
        if task.timeout_seconds and duration > task.timeout_seconds:
            issues.append(f"task exceeded timeout budget: {duration}s > {task.timeout_seconds}s")

    return {
        "is_terminal": is_terminal,
        "heartbeat_age_seconds": heartbeat_age,
        "duration_seconds": duration,
        "is_healthy": len(issues) == 0,
        "issues": issues,
    }


def _subtask_summary(task_id: int, db: Session) -> Dict[str, Any]:
    subtasks = db.query(TaskSubtask).filter(TaskSubtask.task_id == task_id).all()
    counts = {
        "total": len(subtasks),
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "skipped": 0,
    }
    for subtask in subtasks:
        if subtask.status:
            counts[subtask.status.value] = counts.get(subtask.status.value, 0) + 1
    return counts


class TaskEventResponse(BaseModel):
    id: int
    task_id: int
    subtask_id: Optional[int] = None
    event_type: str
    level: str
    stage: Optional[str] = None
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    created_at: datetime


class TaskResourceSampleResponse(BaseModel):
    id: int
    task_id: int
    subtask_id: Optional[int] = None
    phase: Optional[str] = None
    pid: Optional[int] = None
    cpu_percent: Optional[float] = None
    rss_bytes: Optional[int] = None
    vms_bytes: Optional[int] = None
    thread_count: Optional[int] = None
    open_file_count: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None
    created_at: datetime


class TaskSubtaskResponse(BaseModel):
    id: int
    task_id: int
    subtask_key: str
    subtask_type: str
    name: Optional[str] = None
    status: str
    progress: int
    total: int
    retry_count: int
    max_retries: int
    message: Optional[str] = None
    stage: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskResponse(BaseModel):
    """任务响应模型"""
    id: int
    task_type: str
    status: str
    priority: int
    progress: int
    total: int
    error_message: Optional[str] = None
    retry_count: int
    max_retries: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    last_heartbeat: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    params: Optional[dict] = None
    result: Optional[dict] = None
    # Running ingest pipelines only (list view): time-weighted fraction, ETA,
    # current stage. See app/services/pipeline/progress.py.
    pipeline: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class TaskDetailResponse(TaskResponse):
    health: Dict[str, Any]
    resource_summary: Dict[str, Any]
    subtask_summary: Dict[str, Any]
    recent_events: List[TaskEventResponse]


class TaskListResponse(BaseModel):
    """任务列表响应"""
    total: int
    page: int
    page_size: int
    tasks: List[TaskResponse]
    # All running ingest pipelines, not just this page: seconds until the last
    # one is done (stage-lane backlog, see app/services/pipeline/progress.py).
    pipeline_eta_seconds: Optional[float] = None


class TaskEventListResponse(BaseModel):
    total: int
    events: List[TaskEventResponse]


class TaskSubtaskListResponse(BaseModel):
    total: int
    subtasks: List[TaskSubtaskResponse]


class TaskResourceListResponse(BaseModel):
    total: int
    samples: List[TaskResourceSampleResponse]


class TaskActionResponse(BaseModel):
    """任务操作响应"""
    success: bool
    message: str
    task_id: int


class WorkflowEdgeResponse(BaseModel):
    id: int
    key: str
    source: str
    target: str
    label: Optional[str] = None
    status: str
    metadata: Dict[str, Any]
    allowed_actions: List[str]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TaskWorkflowResponse(BaseModel):
    task_id: int
    task_type: str
    task_status: str
    summary: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[WorkflowEdgeResponse]


def _event_to_response(event: TaskEvent) -> TaskEventResponse:
    return TaskEventResponse(
        id=event.id,
        task_id=event.task_id,
        subtask_id=event.subtask_id,
        event_type=event.event_type,
        level=event.level,
        stage=event.stage,
        message=event.message,
        payload=event.payload_json,
        created_at=from_utc_naive(event.created_at),
    )


def _resource_to_response(sample: TaskResourceSample) -> TaskResourceSampleResponse:
    return TaskResourceSampleResponse(
        id=sample.id,
        task_id=sample.task_id,
        subtask_id=sample.subtask_id,
        phase=sample.phase,
        pid=sample.pid,
        cpu_percent=sample.cpu_percent,
        rss_bytes=sample.rss_bytes,
        vms_bytes=sample.vms_bytes,
        thread_count=sample.thread_count,
        open_file_count=sample.open_file_count,
        extra=sample.extra_json,
        created_at=from_utc_naive(sample.created_at),
    )


def _subtask_to_response(subtask: TaskSubtask) -> TaskSubtaskResponse:
    return TaskSubtaskResponse(
        id=subtask.id,
        task_id=subtask.task_id,
        subtask_key=subtask.subtask_key,
        subtask_type=subtask.subtask_type,
        name=subtask.name,
        status=subtask.status.value if subtask.status else "pending",
        progress=subtask.progress or 0,
        total=subtask.total or 0,
        retry_count=subtask.retry_count or 0,
        max_retries=subtask.max_retries or 3,
        message=subtask.message,
        stage=subtask.stage,
        error_message=subtask.error_message,
        metadata=subtask.metadata_json,
        result=subtask.result_json,
        created_at=from_utc_naive(subtask.created_at),
        updated_at=from_utc_naive(subtask.updated_at),
        started_at=from_local_naive(subtask.started_at),
        completed_at=from_local_naive(subtask.completed_at),
    )


def _task_to_response(task: Task) -> TaskResponse:
    return TaskResponse(
        id=task.id,
        task_type=task.task_type,
        status=task.status.value if task.status else "unknown",
        priority=task.priority or 0,
        progress=task.progress or 0,
        total=task.total or 0,
        error_message=task.error_message,
        retry_count=task.retry_count or 0,
        max_retries=task.max_retries or 3,
        created_at=from_utc_naive(task.created_at),
        started_at=from_local_naive(task.started_at),
        completed_at=from_local_naive(task.completed_at),
        worker_id=task.worker_id,
        last_heartbeat=from_local_naive(task.last_heartbeat),
        duration_seconds=_task_duration_seconds(task),
        params=task.params_json,
        result=task.result_json,
    )


def _pipeline_progress() -> Dict[str, Any]:
    with get_db_session() as db:
        ids = [row.id for row in db.query(Task.id).filter(
            Task.task_type == "ingest_pipeline", Task.status == DBTaskStatus.running,
        )]
        return pipeline_progress(db, ids)


def _resource_summary(task_id: int) -> Dict[str, Any]:
    summary = get_task_resource_summary(task_id)
    if summary.get("latest"):
        summary["latest"]["created_at"] = utc_naive_iso(summary["latest"].get("created_at"))
    return summary


def _graph_with_offsets(graph: Dict[str, Any]) -> Dict[str, Any]:
    """task_runtime serialises the naive columns with isoformat(); give them offsets."""
    for node in graph.get("nodes", []):
        for key in ("created_at", "updated_at"):
            node[key] = utc_naive_iso(node.get(key))
        for key in ("started_at", "completed_at"):
            node[key] = local_naive_iso(node.get(key))
    for edge in graph.get("edges", []):
        for key in ("created_at", "updated_at"):
            edge[key] = utc_naive_iso(edge.get(key))
    return graph


def _task_to_detail(task: Task, db: Session) -> TaskDetailResponse:
    recent_events = db.query(TaskEvent).filter(
        TaskEvent.task_id == task.id
    ).order_by(desc(TaskEvent.created_at), desc(TaskEvent.id)).limit(20).all()

    return TaskDetailResponse(
        **_task_to_response(task).model_dump(),
        health=_task_health(task),
        resource_summary=_resource_summary(task.id),
        subtask_summary=_subtask_summary(task.id, db),
        recent_events=[_event_to_response(event) for event in recent_events],
    )


def _workflow_response(task_id: int, task_type: str, task_status: str) -> TaskWorkflowResponse:
    graph = _graph_with_offsets(get_task_workflow_graph(task_id))
    return TaskWorkflowResponse(
        task_id=task_id,
        task_type=task_type,
        task_status=task_status,
        summary=graph.get("summary", {}),
        nodes=graph.get("nodes", []),
        edges=[WorkflowEdgeResponse(**edge) for edge in graph.get("edges", [])],
    )


def _task_detail_payload(task: Task, db: Session) -> Dict[str, Any]:
    return _task_to_detail(task, db).model_dump(mode="json")


def _workflow_payload(task: Task) -> Dict[str, Any]:
    return _workflow_response(
        task.id,
        task.task_type,
        task.status.value if task.status else "unknown",
    ).model_dump(mode="json")


@router.get("", response_model=TaskListResponse)
def list_tasks(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="状态筛选: pending|running|completed|failed|cancelled"),
    task_type: Optional[str] = Query(None, description="任务类型筛选"),
    db: Session = Depends(get_db)
):
    """
    获取任务列表。

    支持分页和状态筛选。
    """
    query = db.query(Task)

    if status:
        try:
            status_enum = DBTaskStatus(status)
            query = query.filter(Task.status == status_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    if task_type:
        query = query.filter(Task.task_type == task_type)

    total = query.count()
    tasks = query.order_by(desc(Task.created_at)).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    responses = [_task_to_response(task) for task in tasks]
    board = _pipeline_progress()
    for resp in responses:
        resp.pipeline = board["tasks"].get(resp.id)

    return TaskListResponse(
        total=total,
        page=page,
        page_size=page_size,
        tasks=responses,
        pipeline_eta_seconds=board["eta_seconds"],
    )


@router.get("/{task_id}", response_model=TaskDetailResponse)
def get_task(
    task_id: int,
    db: Session = Depends(get_db)
):
    """
    获取单个任务详情。
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")

    if task.task_type == "crawl_urls":
        sync_crawl_task_rollup(task.id, status=task.status.value, db=db)
        db.refresh(task)

    return _task_to_detail(task, db)


@router.get("/{task_id}/events", response_model=TaskEventListResponse)
def get_task_events(
    task_id: int,
    after_id: Optional[int] = Query(None, ge=1, description="只返回该事件ID之后的事件"),
    limit: int = Query(100, ge=1, le=500, description="返回的最大事件数"),
    tail: bool = Query(False, description="返回最新的 limit 条（仍按 ID 升序），而不是最早的"),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")

    query = db.query(TaskEvent).filter(TaskEvent.task_id == task_id)
    if after_id is not None:
        query = query.filter(TaskEvent.id > after_id)

    total = query.count()
    if tail:
        events = list(reversed(query.order_by(TaskEvent.id.desc()).limit(limit).all()))
    else:
        events = query.order_by(TaskEvent.id.asc()).limit(limit).all()
    return TaskEventListResponse(
        total=total,
        events=[_event_to_response(event) for event in events],
    )


@router.get("/{task_id}/resources", response_model=TaskResourceListResponse)
def get_task_resources(
    task_id: int,
    after_id: Optional[int] = Query(None, ge=1, description="只返回该采样ID之后的资源样本"),
    limit: int = Query(100, ge=1, le=500, description="返回的最大资源样本数"),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")

    query = db.query(TaskResourceSample).filter(TaskResourceSample.task_id == task_id)
    if after_id is not None:
        query = query.filter(TaskResourceSample.id > after_id)

    total = query.count()
    samples = query.order_by(TaskResourceSample.id.asc()).limit(limit).all()
    return TaskResourceListResponse(
        total=total,
        samples=[_resource_to_response(sample) for sample in samples],
    )


@router.get("/{task_id}/subtasks", response_model=TaskSubtaskListResponse)
def get_task_subtasks(
    task_id: int,
    status: Optional[str] = Query(None, description="子任务状态筛选"),
    limit: int = Query(500, ge=1, le=1000, description="最大子任务数"),
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")

    query = db.query(TaskSubtask).filter(TaskSubtask.task_id == task_id)
    if status is not None:
        try:
            query = query.filter(TaskSubtask.status == TaskSubtaskStatus(status))
        except ValueError:
            raise HTTPException(400, f"Invalid subtask status: {status}")

    total = query.count()
    subtasks = query.order_by(TaskSubtask.id.asc()).limit(limit).all()
    return TaskSubtaskListResponse(
        total=total,
        subtasks=[_subtask_to_response(subtask) for subtask in subtasks],
    )


@router.get("/{task_id}/subtasks/{subtask_id}", response_model=TaskSubtaskResponse)
def get_task_subtask(
    task_id: int,
    subtask_id: int,
    db: Session = Depends(get_db),
):
    subtask = db.query(TaskSubtask).filter(
        TaskSubtask.task_id == task_id,
        TaskSubtask.id == subtask_id,
    ).first()
    if not subtask:
        raise HTTPException(404, "Subtask not found")
    return _subtask_to_response(subtask)


@router.get("/{task_id}/workflow", response_model=TaskWorkflowResponse)
def get_task_workflow(
    task_id: int,
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        raise HTTPException(404, "Task not found")

    if task.task_type == "crawl_urls":
        sync_crawl_task_rollup(task.id, status=task.status.value, db=db)
        db.refresh(task)

    return _workflow_response(task.id, task.task_type, task.status.value if task.status else "unknown")


def _require_task(task_id: int) -> None:
    with get_db_session() as db:
        if db.query(Task).filter(Task.id == task_id).first() is None:
            raise HTTPException(404, "Task not found")


@router.get("/{task_id}/stream")
async def stream_task(
    task_id: int,
    request: Request,
    after_event_id: int = Query(0, ge=0),
    after_resource_id: int = Query(0, ge=0),
    last_event_id_header: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """
    以 SSE 形式流式输出任务快照、事件和资源数据（WHATWG HTML 9.2）。

    `task-event` 带 `id:`（= TaskEvent.id），断线重连时浏览器自动带 `Last-Event-ID`，
    只补发 id 更大的事件；`snapshot` / `resource` / `final` 不带 id。流首发 `retry:`，
    空闲每 15 s 发注释行保活。任务进入终态后发 `final` 并由服务器结束连接：
    **客户端收到 `final` 必须调用 `EventSource.close()`**，否则浏览器会按 `retry:` 自动重连，
    每次都再收到一份 snapshot + final。
    """
    await asyncio.to_thread(_require_task, task_id)

    async def event_generator():
        last_event_id = _sse_start_event_id(after_event_id, last_event_id_header)
        last_resource_id = after_resource_id
        first_snapshot = True
        last_output = time.monotonic()

        def poll_once() -> tuple[List[str], bool, bool]:
            """One DB pass on a worker thread: (messages in send order, drained, finished)."""
            nonlocal last_event_id, last_resource_id, first_snapshot
            messages: List[str] = []
            with get_db_session() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task is None:
                    messages.append("event: error\ndata: {\"message\":\"Task not found\"}\n\n")
                    return messages, False, True

                if task.task_type == "crawl_urls":
                    sync_crawl_task_rollup(task.id, status=task.status.value, db=db)
                    db.refresh(task)

                if first_snapshot:
                    snapshot = _task_detail_payload(task, db)
                    messages.append(_sse_message("snapshot", snapshot))
                    first_snapshot = False

                events = db.query(TaskEvent).filter(
                    TaskEvent.task_id == task_id,
                    TaskEvent.id > last_event_id,
                ).order_by(TaskEvent.id.asc()).limit(200).all()
                for event in events:
                    payload = _event_to_response(event).model_dump(mode="json")
                    last_event_id = event.id
                    messages.append(_sse_message("task-event", payload, event.id))

                samples = db.query(TaskResourceSample).filter(
                    TaskResourceSample.task_id == task_id,
                    TaskResourceSample.id > last_resource_id,
                ).order_by(TaskResourceSample.id.asc()).limit(100).all()
                for sample in samples:
                    payload = _resource_to_response(sample).model_dump(mode="json")
                    last_resource_id = sample.id
                    messages.append(_sse_message("resource", payload))

                # A full batch means more may be waiting: poll again at once instead
                # of sleeping, and only announce the end once nothing is left to send.
                drained = bool(events or samples)
                if not drained and task.status in {
                    DBTaskStatus.completed,
                    DBTaskStatus.failed,
                    DBTaskStatus.cancelled,
                    DBTaskStatus.timeout,
                }:
                    final_snapshot = _task_detail_payload(task, db)
                    messages.append(_sse_message("final", final_snapshot))
                    return messages, drained, True
            return messages, drained, False

        yield f"retry: {SSE_RETRY_MS}\n\n"

        while True:
            if await request.is_disconnected():
                break
            messages, drained, finished = await asyncio.to_thread(poll_once)
            for message in messages:
                yield message
            if messages:
                last_output = time.monotonic()
            if finished:
                break
            if drained:
                continue
            await asyncio.sleep(1.0)
            if time.monotonic() - last_output >= SSE_KEEPALIVE_SECONDS:
                yield ": keepalive\n\n"
                last_output = time.monotonic()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{task_id}/workflow/stream")
async def stream_task_workflow(
    task_id: int,
    request: Request,
    after_event_id: int = Query(0, ge=0),
    after_resource_id: int = Query(0, ge=0),
    last_event_id_header: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """
    以 SSE 形式流式输出任务详情、工作流图、事件和资源数据。

    重连、`id:`、保活与结束语义同 `/{task_id}/stream`：收到 `final` 后客户端必须 `close()`。
    """
    await asyncio.to_thread(_require_task, task_id)

    async def workflow_event_generator():
        last_event_id = _sse_start_event_id(after_event_id, last_event_id_header)
        last_resource_id = after_resource_id
        last_workflow_signature = ""
        first_snapshot = True
        last_output = time.monotonic()

        def poll_once() -> tuple[List[str], bool, bool]:
            """One DB pass on a worker thread: (messages in send order, drained, finished)."""
            nonlocal last_event_id, last_resource_id, last_workflow_signature, first_snapshot
            messages: List[str] = []
            with get_db_session() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task is None:
                    messages.append("event: error\ndata: {\"message\":\"Task not found\"}\n\n")
                    return messages, False, True

                if task.task_type == "crawl_urls":
                    sync_crawl_task_rollup(task.id, status=task.status.value, db=db)
                    db.refresh(task)

                task_payload = _task_detail_payload(task, db)
                workflow_payload = _workflow_payload(task)
                workflow_signature = json.dumps(workflow_payload, ensure_ascii=False, sort_keys=True)

                if first_snapshot:
                    snapshot = {"task": task_payload, "workflow": workflow_payload}
                    messages.append(_sse_message("snapshot", snapshot))
                    last_workflow_signature = workflow_signature
                    first_snapshot = False
                elif workflow_signature != last_workflow_signature:
                    messages.append(_sse_message("workflow", workflow_payload))
                    last_workflow_signature = workflow_signature

                events = db.query(TaskEvent).filter(
                    TaskEvent.task_id == task_id,
                    TaskEvent.id > last_event_id,
                ).order_by(TaskEvent.id.asc()).limit(200).all()
                for event in events:
                    payload = _event_to_response(event).model_dump(mode="json")
                    last_event_id = event.id
                    messages.append(_sse_message("task-event", payload, event.id))

                samples = db.query(TaskResourceSample).filter(
                    TaskResourceSample.task_id == task_id,
                    TaskResourceSample.id > last_resource_id,
                ).order_by(TaskResourceSample.id.asc()).limit(100).all()
                for sample in samples:
                    payload = _resource_to_response(sample).model_dump(mode="json")
                    last_resource_id = sample.id
                    messages.append(_sse_message("resource", payload))

                # A full batch means more may be waiting: poll again at once instead
                # of sleeping, and only announce the end once nothing is left to send.
                drained = bool(events or samples)
                if not drained and task.status in {
                    DBTaskStatus.completed,
                    DBTaskStatus.failed,
                    DBTaskStatus.cancelled,
                    DBTaskStatus.timeout,
                }:
                    final_snapshot = {"task": task_payload, "workflow": workflow_payload}
                    messages.append(_sse_message("final", final_snapshot))
                    return messages, drained, True
            return messages, drained, False

        yield f"retry: {SSE_RETRY_MS}\n\n"

        while True:
            if await request.is_disconnected():
                break
            messages, drained, finished = await asyncio.to_thread(poll_once)
            for message in messages:
                yield message
            if messages:
                last_output = time.monotonic()
            if finished:
                break
            if drained:
                continue
            await asyncio.sleep(1.0)
            if time.monotonic() - last_output >= SSE_KEEPALIVE_SECONDS:
                yield ": keepalive\n\n"
                last_output = time.monotonic()

    return StreamingResponse(workflow_event_generator(), media_type="text/event-stream")


@router.post("/{task_id}/cancel", response_model=TaskActionResponse)
def cancel_task_endpoint(
    task_id: int,
    db: Session = Depends(get_db)
):
    """
    取消任务。

    只能取消 pending 或 running 状态的任务。
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")

    if task.status not in (DBTaskStatus.pending, DBTaskStatus.running):
        raise HTTPException(
            400,
            f"Cannot cancel task in status: {task.status.value}"
        )

    # A task type with a registered runner is cancelled by that runner: it flips
    # the status atomically AND stops the worker. Anything else (crawl_urls)
    # falls through to the legacy status-only path below.
    runner = get_runner(task.task_type)
    if runner is not None:
        try:
            cancelled = runner.cancel(task_id)
        except NotImplementedError as exc:
            raise HTTPException(400, str(exc)) from exc
        except TaskNotCancellable as exc:
            raise HTTPException(409, str(exc)) from exc
        if not cancelled:
            raise HTTPException(409, "Task already finished")
        return TaskActionResponse(success=True, message="Task cancelled successfully", task_id=task_id)

    task.status = DBTaskStatus.cancelled
    task.completed_at = datetime.now()
    db.add(
        TaskEvent(
            task_id=task.id,
            event_type="task.cancelled",
            level="warning",
            stage="task.cancelled",
            message="Task cancelled by API request",
            payload_json={"task_id": task.id},
        )
    )
    db.commit()

    try:
        cancel_task(str(task_id))
    except Exception:
        pass

    return TaskActionResponse(
        success=True,
        message="Task cancelled successfully",
        task_id=task_id
    )


@router.post("/{task_id}/retry", response_model=TaskActionResponse)
def retry_task(
    task_id: int,
    db: Session = Depends(get_db)
):
    """
    重试失败的任务。

    crawl_urls 额外支持“已完成但内部存在失败 URL”的重试。
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")

    runner = get_runner(task.task_type)
    if runner is not None:
        # Runner-driven tasks are executed by an in-process orchestrator, not a
        # queue: a row set back to pending would never run. The runner re-drives
        # it in place, from the stage that stopped.
        try:
            result = runner.retry(task_id)
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, detail=str(exc)) from exc
        if result.get("status") != "started":
            raise HTTPException(409, result.get("error") or "retry conflict")
        return TaskActionResponse(success=True, message="Task re-driven from its failed stage",
                                  task_id=task_id)

    retry_urls: List[str] = []
    if task.task_type == "crawl_urls" and task.status in (DBTaskStatus.failed, DBTaskStatus.completed):
        retry_urls = _retry_urls_for_task(task)

    retryable_completed_crawl = (
        task.task_type == "crawl_urls"
        and task.status == DBTaskStatus.completed
        and bool(retry_urls)
    )

    if task.status != DBTaskStatus.failed and not retryable_completed_crawl:
        raise HTTPException(
            400,
            f"Cannot retry task in status: {task.status.value}"
        )

    if task.retry_count >= task.max_retries:
        raise HTTPException(
            400,
            f"Task has exceeded max retries ({task.max_retries})"
        )

    task.status = DBTaskStatus.pending
    task.retry_count = (task.retry_count or 0) + 1
    task.progress = 0
    task.error_message = None
    task.result_json = None
    task.started_at = None
    task.completed_at = None
    task.worker_id = None
    task.last_heartbeat = None

    if task.task_type == "crawl_urls":
        auto_index = bool((task.params_json or {}).get("auto_index", False))
        db.query(CrawlRecord).filter(
            CrawlRecord.task_id == task_id,
            CrawlRecord.url.in_(retry_urls),
        ).update(
            {
                CrawlRecord.status: DBCrawlStatus.pending,
                CrawlRecord.title: None,
                CrawlRecord.content_length: None,
                CrawlRecord.doc_id: None,
                CrawlRecord.error_message: None,
            },
            synchronize_session=False,
        )
        db.query(TaskSubtask).filter(
            TaskSubtask.task_id == task_id,
            TaskSubtask.subtask_key.in_(retry_urls),
        ).update(
            {
                TaskSubtask.status: TaskSubtaskStatus.pending,
                TaskSubtask.progress: 0,
                TaskSubtask.message: "Queued for retry",
                TaskSubtask.stage: "retry.pending",
                TaskSubtask.error_message: None,
                TaskSubtask.started_at: None,
                TaskSubtask.completed_at: None,
            },
            synchronize_session=False,
        )
        if auto_index:
            index_keys = [f"index::{url}" for url in retry_urls]
            db.query(TaskSubtask).filter(
                TaskSubtask.task_id == task_id,
                TaskSubtask.subtask_key.in_(index_keys),
            ).update(
                {
                    TaskSubtask.status: TaskSubtaskStatus.pending,
                    TaskSubtask.progress: 0,
                    TaskSubtask.message: "Waiting for crawl result",
                    TaskSubtask.stage: "workflow.index.pending",
                    TaskSubtask.error_message: None,
                    TaskSubtask.started_at: None,
                    TaskSubtask.completed_at: None,
                },
                synchronize_session=False,
            )
            db.query(TaskWorkflowEdge).filter(
                TaskWorkflowEdge.task_id == task_id,
                TaskWorkflowEdge.from_key.in_(retry_urls),
                TaskWorkflowEdge.to_key.in_(index_keys),
            ).update(
                {TaskWorkflowEdge.status: TaskWorkflowEdgeStatus.active},
                synchronize_session=False,
            )
            db.query(TaskWorkflowEdge).filter(
                TaskWorkflowEdge.task_id == task_id,
                TaskWorkflowEdge.from_key.in_(index_keys),
                TaskWorkflowEdge.to_key == "workflow.finalize",
            ).update(
                {TaskWorkflowEdge.status: TaskWorkflowEdgeStatus.active},
                synchronize_session=False,
            )
        else:
            db.query(TaskWorkflowEdge).filter(
                TaskWorkflowEdge.task_id == task_id,
                TaskWorkflowEdge.from_key.in_(retry_urls),
                TaskWorkflowEdge.to_key == "workflow.finalize",
            ).update(
                {TaskWorkflowEdge.status: TaskWorkflowEdgeStatus.active},
                synchronize_session=False,
            )
        finalize_node = db.query(TaskSubtask).filter(
            TaskSubtask.task_id == task_id,
            TaskSubtask.subtask_key == "workflow.finalize",
        ).first()
        if finalize_node is not None:
            finalize_node.status = TaskSubtaskStatus.pending
            finalize_node.progress = 0
            finalize_node.message = "Waiting for upstream actions"
            finalize_node.stage = "workflow.finalize.pending"
            finalize_node.error_message = None
            finalize_node.started_at = None
            finalize_node.completed_at = None

    db.add(
        TaskEvent(
            task_id=task.id,
            event_type="task.retry_requested",
            level="warning",
            stage="task.retry",
            message=f"Retry requested for task {task.id}",
            payload_json={"urls": retry_urls, "retry_count": task.retry_count},
        )
    )
    db.commit()

    try:
        _enqueue_retry(task, retry_urls)
    except HTTPException:
        db.refresh(task)
        task.status = DBTaskStatus.failed
        task.error_message = "Retry dispatch rejected: unsupported or invalid task payload"
        db.commit()
        raise
    except Exception as e:
        db.refresh(task)
        task.status = DBTaskStatus.failed
        task.error_message = f"Retry dispatch failed: {e}"
        db.commit()
        raise HTTPException(500, f"Failed to re-enqueue task: {e}") from e

    return TaskActionResponse(
        success=True,
        message=f"Task queued for retry (attempt {task.retry_count}/{task.max_retries})",
        task_id=task_id
    )


@router.post("/{task_id}/subtasks/{subtask_id}/retry", response_model=TaskActionResponse)
def retry_task_subtask(
    task_id: int,
    subtask_id: int,
    db: Session = Depends(get_db),
):
    """
    重试单个子任务。

    当前仅支持 crawl_urls 的 URL 级子任务。
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        raise HTTPException(404, "Task not found")

    if task.task_type != "crawl_urls":
        raise HTTPException(400, "Subtask retry currently only supports crawl_urls tasks")

    if task.status == DBTaskStatus.running:
        raise HTTPException(400, "Cannot retry a subtask while the parent task is running")

    subtask = db.query(TaskSubtask).filter(
        TaskSubtask.task_id == task_id,
        TaskSubtask.id == subtask_id,
    ).first()
    if subtask is None:
        raise HTTPException(404, "Subtask not found")

    if subtask.retry_count >= subtask.max_retries:
        raise HTTPException(400, f"Subtask has exceeded max retries ({subtask.max_retries})")

    metadata = subtask.metadata_json or {}
    if metadata.get("workflow_action_type") != "crawl_url":
        raise HTTPException(400, "Subtask retry currently only supports crawl_url workflow nodes")

    retry_url = metadata.get("target_url") or metadata.get("url") or subtask.subtask_key
    params = task.params_json or {}
    auto_index = bool(params.get("auto_index", False))

    subtask.retry_count = (subtask.retry_count or 0) + 1
    subtask.status = TaskSubtaskStatus.pending
    subtask.progress = 0
    subtask.message = "Queued for manual retry"
    subtask.stage = "retry.pending"
    subtask.error_message = None
    subtask.started_at = None
    subtask.completed_at = None

    if auto_index:
        index_key = f"index::{retry_url}"
        index_subtask = db.query(TaskSubtask).filter(
            TaskSubtask.task_id == task_id,
            TaskSubtask.subtask_key == index_key,
        ).first()
        if index_subtask is not None:
            index_subtask.status = TaskSubtaskStatus.pending
            index_subtask.progress = 0
            index_subtask.message = "Waiting for crawl result"
            index_subtask.stage = "workflow.index.pending"
            index_subtask.error_message = None
            index_subtask.started_at = None
            index_subtask.completed_at = None
        db.query(TaskWorkflowEdge).filter(
            TaskWorkflowEdge.task_id == task_id,
            TaskWorkflowEdge.from_key == retry_url,
            TaskWorkflowEdge.to_key == index_key,
        ).update(
            {TaskWorkflowEdge.status: TaskWorkflowEdgeStatus.active},
            synchronize_session=False,
        )
        db.query(TaskWorkflowEdge).filter(
            TaskWorkflowEdge.task_id == task_id,
            TaskWorkflowEdge.from_key == index_key,
            TaskWorkflowEdge.to_key == "workflow.finalize",
        ).update(
            {TaskWorkflowEdge.status: TaskWorkflowEdgeStatus.active},
            synchronize_session=False,
        )
    else:
        db.query(TaskWorkflowEdge).filter(
            TaskWorkflowEdge.task_id == task_id,
            TaskWorkflowEdge.from_key == retry_url,
            TaskWorkflowEdge.to_key == "workflow.finalize",
        ).update(
            {TaskWorkflowEdge.status: TaskWorkflowEdgeStatus.active},
            synchronize_session=False,
        )

    finalize_node = db.query(TaskSubtask).filter(
        TaskSubtask.task_id == task_id,
        TaskSubtask.subtask_key == "workflow.finalize",
    ).first()
    if finalize_node is not None:
        finalize_node.status = TaskSubtaskStatus.pending
        finalize_node.progress = 0
        finalize_node.message = "Waiting for upstream actions"
        finalize_node.stage = "workflow.finalize.pending"
        finalize_node.error_message = None
        finalize_node.started_at = None
        finalize_node.completed_at = None

    crawl_record = db.query(CrawlRecord).filter(
        CrawlRecord.task_id == task_id,
        CrawlRecord.url == retry_url,
    ).first()
    if crawl_record is not None:
        crawl_record.status = DBCrawlStatus.pending
        crawl_record.title = None
        crawl_record.content_length = None
        crawl_record.doc_id = None
        crawl_record.error_message = None

    task.status = DBTaskStatus.pending
    task.started_at = None
    task.completed_at = None
    task.worker_id = None
    task.last_heartbeat = None
    task.error_message = None
    task.result_json = None
    db.add(
        TaskEvent(
            task_id=task.id,
            subtask_id=subtask.id,
            event_type="task.subtask.retry_requested",
            level="warning",
            stage="retry.pending",
            message=f"Manual retry requested for {retry_url}",
            payload_json={"url": retry_url, "subtask_retry_count": subtask.retry_count},
        )
    )
    db.commit()

    try:
        enqueue_crawl_urls(
            [retry_url],
            use_proxy=bool(params.get("use_proxy", False)),
            auto_index=bool(params.get("auto_index", False)),
            max_concurrent=int(params.get("max_concurrent", 5) or 5),
            task_id=str(task.id),
        )
    except Exception as e:
        db.refresh(task)
        task.status = DBTaskStatus.failed
        task.error_message = f"Subtask retry dispatch failed: {e}"
        db.commit()
        raise HTTPException(500, f"Failed to enqueue subtask retry: {e}") from e

    return TaskActionResponse(
        success=True,
        message=f"Subtask queued for retry ({subtask.retry_count}/{subtask.max_retries})",
        task_id=task_id,
    )


@router.post("/{task_id}/workflow/nodes/{subtask_id}/actions/{action}", response_model=TaskActionResponse)
def workflow_node_action(
    task_id: int,
    subtask_id: int,
    action: str,
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        raise HTTPException(404, "Task not found")

    subtask = db.query(TaskSubtask).filter(
        TaskSubtask.task_id == task_id,
        TaskSubtask.id == subtask_id,
    ).first()
    if subtask is None:
        raise HTTPException(404, "Subtask not found")

    action = action.lower()
    if action == "retry":
        return retry_task_subtask(task_id, subtask_id, db)

    if action != "skip":
        raise HTTPException(400, f"Unsupported node action: {action}")

    if task.status == DBTaskStatus.running:
        raise HTTPException(400, "Cannot skip a node while the parent task is running")

    subtask_meta = subtask.metadata_json or {}
    subtask.status = TaskSubtaskStatus.skipped
    subtask.progress = 0
    subtask.message = "Skipped by workflow action"
    subtask.stage = "workflow.skipped"
    subtask.completed_at = datetime.now()
    db.add(
        TaskEvent(
            task_id=task_id,
            subtask_id=subtask.id,
            event_type="workflow.node.skipped",
            level="warning",
            stage="workflow.skipped",
            message=f"Skipped node {subtask.subtask_key}",
            payload_json={"node_id": subtask.id, "node_key": subtask.subtask_key},
        )
    )
    if (
        task.task_type == "crawl_urls"
        and subtask_meta.get("workflow_action_type") == "crawl_url"
        and bool((task.params_json or {}).get("auto_index", False))
    ):
        index_key = f"index::{subtask_meta.get('target_url') or subtask.subtask_key}"
        index_subtask = db.query(TaskSubtask).filter(
            TaskSubtask.task_id == task_id,
            TaskSubtask.subtask_key == index_key,
        ).first()
        if index_subtask is not None:
            index_subtask.status = TaskSubtaskStatus.skipped
            index_subtask.progress = 0
            index_subtask.message = "Skipped because upstream crawl node was skipped"
            index_subtask.stage = "workflow.index.skipped"
            index_subtask.completed_at = datetime.now()
            merged_meta = dict(index_subtask.metadata_json or {})
            merged_meta.update({"blocked_by": subtask.subtask_key, "skip_reason": "workflow_skip"})
            index_subtask.metadata_json = merged_meta

        edge = db.query(TaskWorkflowEdge).filter(
            TaskWorkflowEdge.task_id == task_id,
            TaskWorkflowEdge.from_key == subtask.subtask_key,
            TaskWorkflowEdge.to_key == index_key,
        ).first()
        if edge is not None:
            edge.status = TaskWorkflowEdgeStatus.blocked
            merged_edge_meta = dict(edge.metadata_json or {})
            merged_edge_meta.update({"reason": "workflow_skip"})
            edge.metadata_json = merged_edge_meta

    if task.task_type == "crawl_urls":
        target_url = subtask_meta.get("target_url") or subtask.subtask_key
        crawl_record = db.query(CrawlRecord).filter(
            CrawlRecord.task_id == task_id,
            CrawlRecord.url == target_url,
        ).first()
        if crawl_record is not None:
            crawl_record.status = DBCrawlStatus.skipped
            crawl_record.error_message = "Skipped by workflow action"
        sync_crawl_task_rollup(task_id, status=task.status.value, db=db)

    db.commit()
    return TaskActionResponse(success=True, message="Node skipped", task_id=task_id)


@router.post("/{task_id}/workflow/edges/{edge_id}/actions/{action}", response_model=TaskActionResponse)
def workflow_edge_action(
    task_id: int,
    edge_id: int,
    action: str,
    db: Session = Depends(get_db),
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        raise HTTPException(404, "Task not found")

    edge = db.query(TaskWorkflowEdge).filter(
        TaskWorkflowEdge.task_id == task_id,
        TaskWorkflowEdge.id == edge_id,
    ).first()
    if edge is None:
        raise HTTPException(404, "Workflow edge not found")

    action = action.lower()
    if action == "disable":
        edge.status = TaskWorkflowEdgeStatus.disabled
    elif action == "enable":
        edge.status = TaskWorkflowEdgeStatus.active
    else:
        raise HTTPException(400, f"Unsupported edge action: {action}")

    db.add(
        TaskEvent(
            task_id=task_id,
            event_type="workflow.edge.updated",
            level="warning",
            stage="workflow.edge",
            message=f"Edge {edge.edge_key} -> {edge.status.value}",
            payload_json={
                "edge_id": edge.id,
                "edge_key": edge.edge_key,
                "status": edge.status.value,
            },
        )
    )
    db.commit()
    return TaskActionResponse(
        success=True,
        message=f"Edge action applied: {action}",
        task_id=task_id,
    )


@router.delete("/{task_id}", response_model=TaskActionResponse)
def delete_task(
    task_id: int,
    db: Session = Depends(get_db)
):
    """
    删除任务记录。

    注意：只能删除已完成、失败或取消的任务。
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task not found")

    if task.status in (DBTaskStatus.pending, DBTaskStatus.running):
        raise HTTPException(
            400,
            f"Cannot delete task in status: {task.status.value}. Cancel it first."
        )

    purge_task_runtime_rows(task_id, db=db)
    db.delete(task)
    db.commit()

    task_file = TASKS_DIR / f"{task_id}.json"
    if task_file.exists():
        try:
            task_file.unlink()
        except Exception:
            pass

    return TaskActionResponse(
        success=True,
        message="Task deleted successfully",
        task_id=task_id
    )
