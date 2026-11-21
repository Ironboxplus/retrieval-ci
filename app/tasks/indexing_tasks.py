"""
索引任务 - 文档索引相关的异步任务
"""
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.tasks.task_queue import huey
from app.tasks.base import TaskContext, run_with_context, create_task_record, TaskStatus


def _normalize_source_type(source_type: str) -> str:
    """Map legacy task source types onto the repo-wide retrieval vocabulary."""
    normalized = (source_type or "").lower()
    if normalized in {"webpage", "paper", "upload"}:
        return normalized
    if normalized in {"bookmark", "pdf", "txt"}:
        return "upload"
    return "upload"


def _normalize_data_source(metadata: Optional[Dict[str, Any]]) -> str:
    """Default task-generated content to user data unless metadata says otherwise."""
    raw = str((metadata or {}).get("data_source", "user")).lower()
    return "system" if raw == "system" else "user"


def _index_task_document(
    title: str,
    content: str,
    source: str,
    source_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist task-indexed content through the canonical DB + index write path."""
    from core.db import get_db_session
    from core.db.models import Document as DBDocument, SourceType, DataSource, DocumentStatus
    from app.services.index_service import get_index_service

    normalized_source_type = _normalize_source_type(source_type)
    normalized_data_source = _normalize_data_source(metadata)
    source_type_enum = {
        "webpage": SourceType.webpage,
        "paper": SourceType.paper,
        "upload": SourceType.upload,
    }[normalized_source_type]
    data_source_enum = DataSource.system if normalized_data_source == "system" else DataSource.user

    with get_db_session() as db:
        doc = DBDocument(
            title=title or "Untitled",
            source=source or "",
            source_type=source_type_enum,
            data_source=data_source_enum,
            url=source if normalized_source_type == "webpage" else None,
            status=DocumentStatus.pending,
            metadata_json=metadata or {},
        )
        db.add(doc)
        db.flush()

        chunker_kind = "html" if normalized_source_type == "webpage" else None
        index_result = get_index_service(chunker_type=chunker_kind).index_document(
            doc_id=doc.id,
            content=content or "",
            title=title or "Untitled",
            source=source or "",
            metadata=metadata or {},
            db=db,
            source_type=normalized_source_type,
            data_source=normalized_data_source,
        )

        doc.status = DocumentStatus.indexed if index_result.get("success") else DocumentStatus.failed

        return {
            "doc_id": doc.id,
            "title": doc.title,
            "chunks_count": index_result.get("chunk_count", 0),
            "index_result": index_result,
            "source_type": normalized_source_type,
            "data_source": normalized_data_source,
        }


def _set_edge_status(
    ctx: TaskContext,
    from_key: str,
    to_key: str,
    *,
    status: str,
    label: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    ctx.upsert_edge(
        edge_key=f"{from_key}->{to_key}",
        from_key=from_key,
        to_key=to_key,
        status=status,
        label=label,
        metadata=metadata,
    )


@huey.task()
@run_with_context("index_document")
def index_document_task(ctx: TaskContext, title: str, content: str,
                        source: str, source_type: str,
                        metadata: Optional[Dict[str, Any]] = None):
    """
    单文档索引任务

    Args:
        ctx: 任务上下文
        title: 文档标题
        content: 文档内容
        source: 来源(文件名或URL)
        source_type: 来源类型 (bookmark, pdf, txt)
        metadata: 额外元数据

    Returns:
        索引结果
    """
    subtask_key = source or title or "document"
    ctx.upsert_subtask(
        subtask_key="workflow.prepare",
        subtask_type="workflow_action",
        name="Prepare Index Task",
        status="completed",
        progress=1,
        total=1,
        message="Workflow prepared",
        stage="workflow.prepare.completed",
        metadata={
            "workflow_action_type": "prepare",
            "depends_on": [],
            "retriable": False,
            "node_group": "system",
        },
    )
    ctx.upsert_subtask(
        subtask_key=subtask_key,
        subtask_type="document_index",
        name=title or source or "Untitled",
        status="pending",
        progress=0,
        total=1,
        message="Queued",
        stage="index.pending",
        metadata={
            "source": source,
            "source_type": source_type,
            "data_source": _normalize_data_source(metadata),
            "workflow_action_type": "document_index",
            "depends_on": ["workflow.prepare"],
            "retriable": False,
            "node_group": "index",
        },
    )
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        name="Finalize Index Task",
        status="pending",
        progress=0,
        total=1,
        message="Waiting for index result",
        stage="workflow.finalize.pending",
        metadata={
            "workflow_action_type": "finalize",
            "depends_on": [subtask_key],
            "retriable": False,
            "node_group": "system",
        },
    )
    ctx.upsert_edge(
        edge_key="workflow.prepare->document",
        from_key="workflow.prepare",
        to_key=subtask_key,
        status="active",
        label="prepare",
        metadata={"interactive": True},
    )
    ctx.upsert_edge(
        edge_key="document->workflow.finalize",
        from_key=subtask_key,
        to_key="workflow.finalize",
        status="active",
        label="finalize",
        metadata={"interactive": True},
    )
    ctx.emit_event(
        "index.document.queued",
        f"Queued index task for {title or source or 'document'}",
        stage="index.queued",
        payload={"source": source, "source_type": source_type},
    )
    ctx.update_progress(0.1, "准备统一索引写路径...")
    _set_edge_status(
        ctx,
        "workflow.prepare",
        subtask_key,
        status="traversed",
        label="prepare",
        metadata={"interactive": True},
    )

    if ctx.is_cancelled():
        ctx.upsert_subtask(
            subtask_key="workflow.finalize",
            subtask_type="workflow_action",
            status="cancelled",
            progress=0,
            total=1,
            message="Workflow cancelled before indexing",
            stage="workflow.finalize.cancelled",
        )
        return {"status": "cancelled"}

    ctx.upsert_subtask(
        subtask_key=subtask_key,
        subtask_type="document_index",
        status="running",
        progress=0,
        total=1,
        message="Preparing index write path",
        stage="index.preparing",
    )
    ctx.update_progress(0.4, "写入数据库与索引...")
    ctx.emit_event(
        "index.document.started",
        f"Indexing {title or source or 'document'}",
        stage="index.running",
        payload={"source": source, "source_type": source_type},
    )

    if ctx.is_cancelled():
        return {"status": "cancelled"}

    indexed = _index_task_document(
        title=title,
        content=content,
        source=source,
        source_type=source_type,
        metadata=metadata,
    )

    ctx.update_progress(1.0, "索引完成")

    if not indexed["index_result"].get("success"):
        _set_edge_status(
            ctx,
            subtask_key,
            "workflow.finalize",
            status="traversed",
            label="finalize",
            metadata={"interactive": True},
        )
        ctx.upsert_subtask(
            subtask_key="workflow.finalize",
            subtask_type="workflow_action",
            status="completed",
            progress=1,
            total=1,
            message="Workflow finalized with failure",
            stage="workflow.finalize.completed",
            result={"status": "failed"},
        )
        ctx.upsert_subtask(
            subtask_key=subtask_key,
            subtask_type="document_index",
            status="failed",
            progress=0,
            total=1,
            message=indexed["index_result"].get("error", "index failed"),
            stage="index.failed",
            error=indexed["index_result"].get("error", "index failed"),
            result={
                "source_type": indexed["source_type"],
                "data_source": indexed["data_source"],
            },
        )
        ctx.emit_event(
            "index.document.failed",
            indexed["index_result"].get("error", "index failed"),
            stage="index.failed",
            level="error",
            payload={
                "source": source,
                "source_type": indexed["source_type"],
                "data_source": indexed["data_source"],
            },
        )
        return {
            "status": "failed",
            "error": indexed["index_result"].get("error", "index failed"),
            "source_type": indexed["source_type"],
            "data_source": indexed["data_source"],
        }

    ctx.upsert_subtask(
        subtask_key=subtask_key,
        subtask_type="document_index",
        status="completed",
        progress=1,
        total=1,
        message="Indexed successfully",
        stage="index.completed",
        result={
            "doc_id": indexed["doc_id"],
            "title": indexed["title"],
            "chunks_count": indexed["chunks_count"],
            "source_type": indexed["source_type"],
            "data_source": indexed["data_source"],
        },
    )
    _set_edge_status(
        ctx,
        subtask_key,
        "workflow.finalize",
        status="traversed",
        label="finalize",
        metadata={"interactive": True},
    )
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        status="completed",
        progress=1,
        total=1,
        message="Workflow finalized",
        stage="workflow.finalize.completed",
        result={"status": "success", "doc_id": indexed["doc_id"]},
    )
    ctx.emit_event(
        "index.document.completed",
        f"Indexed {indexed['title']}",
        stage="index.completed",
        payload={
            "doc_id": indexed["doc_id"],
            "chunks_count": indexed["chunks_count"],
            "source_type": indexed["source_type"],
            "data_source": indexed["data_source"],
        },
    )
    return {
        "status": "success",
        "doc_id": indexed["doc_id"],
        "title": indexed["title"],
        "chunks_count": indexed["chunks_count"],
        "source_type": indexed["source_type"],
        "data_source": indexed["data_source"],
    }


@huey.task()
@run_with_context("batch_index")
def batch_index_task(ctx: TaskContext, documents_data: List[Dict[str, Any]]):
    """
    批量索引任务

    Args:
        ctx: 任务上下文
        documents_data: 文档数据列表，每个元素包含:
            - title: 标题
            - content: 内容
            - source: 来源
            - source_type: 类型
            - metadata: 可选元数据

    Returns:
        批量索引结果
    """
    total = len(documents_data)
    if total == 0:
        return {"status": "success", "indexed": 0, "documents": []}

    ctx.upsert_subtask(
        subtask_key="workflow.prepare",
        subtask_type="workflow_action",
        name="Prepare Batch Index",
        status="completed",
        progress=1,
        total=1,
        message="Workflow prepared",
        stage="workflow.prepare.completed",
        metadata={
            "workflow_action_type": "prepare",
            "depends_on": [],
            "retriable": False,
            "node_group": "system",
        },
    )
    for i, doc_data in enumerate(documents_data):
        subtask_key = doc_data.get("source") or f"document-{i}"
        ctx.upsert_subtask(
            subtask_key=subtask_key,
            subtask_type="document_index",
            name=doc_data.get("title") or doc_data.get("source") or f"Document {i + 1}",
            status="pending",
            progress=0,
            total=1,
            message="Queued",
            stage="index.pending",
            metadata={
                "batch_index": True,
                "source": doc_data.get("source"),
                "source_type": doc_data.get("source_type", "upload"),
                "workflow_action_type": "document_index",
                "depends_on": ["workflow.prepare"],
                "retriable": False,
                "node_group": "index",
            },
        )
        ctx.upsert_edge(
            edge_key=f"workflow.prepare->{subtask_key}",
            from_key="workflow.prepare",
            to_key=subtask_key,
            status="active",
            label="prepare",
            metadata={"interactive": True},
        )
        ctx.upsert_edge(
            edge_key=f"{subtask_key}->workflow.finalize",
            from_key=subtask_key,
            to_key="workflow.finalize",
            status="active",
            label="finalize",
            metadata={"interactive": True},
        )
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        name="Finalize Batch Index",
        status="pending",
        progress=0,
        total=1,
        message="Waiting for all document actions",
        stage="workflow.finalize.pending",
        metadata={
            "workflow_action_type": "finalize",
            "depends_on": [
                (doc_data.get("source") or f"document-{i}")
                for i, doc_data in enumerate(documents_data)
            ],
            "retriable": False,
            "node_group": "system",
        },
    )
    ctx.emit_event(
        "index.batch.queued",
        f"Queued batch index for {total} documents",
        stage="index.queued",
        payload={"total": total},
    )
    ctx.update_progress(0.1, f"准备统一索引写路径, 共 {total} 个文档...")
    for i, doc_data in enumerate(documents_data):
        subtask_key = doc_data.get("source") or f"document-{i}"
        _set_edge_status(
            ctx,
            "workflow.prepare",
            subtask_key,
            status="traversed",
            label="prepare",
            metadata={"interactive": True},
        )

    indexed_docs = []
    failed_docs = []
    for i, doc_data in enumerate(documents_data):
        if ctx.is_cancelled():
            ctx.upsert_subtask(
                subtask_key="workflow.finalize",
                subtask_type="workflow_action",
                status="cancelled",
                progress=0,
                total=1,
                message="Workflow cancelled during batch indexing",
                stage="workflow.finalize.cancelled",
            )
            return {
                "status": "cancelled",
                "indexed": len(indexed_docs),
                "total": total
            }

        subtask_key = doc_data.get("source") or f"document-{i}"
        ctx.upsert_subtask(
            subtask_key=subtask_key,
            subtask_type="document_index",
            status="running",
            progress=0,
            total=1,
            message="Indexing document",
            stage="index.running",
        )
        indexed = _index_task_document(
            title=doc_data.get("title", "Untitled"),
            content=doc_data.get("content", ""),
            source=doc_data.get("source", ""),
            source_type=doc_data.get("source_type", "upload"),
            metadata=doc_data.get("metadata", {}),
        )

        if indexed["index_result"].get("success"):
            subtask_result = {
                "doc_id": indexed["doc_id"],
                "title": indexed["title"],
                "chunks_count": indexed["chunks_count"],
                "source_type": indexed["source_type"],
                "data_source": indexed["data_source"],
            }
            indexed_docs.append({
                "doc_id": indexed["doc_id"],
                "title": indexed["title"],
                "chunks_count": indexed["chunks_count"],
                "source_type": indexed["source_type"],
                "data_source": indexed["data_source"],
            })
            ctx.upsert_subtask(
                subtask_key=subtask_key,
                subtask_type="document_index",
                status="completed",
                progress=1,
                total=1,
                message="Indexed successfully",
                stage="index.completed",
                result=subtask_result,
            )
            _set_edge_status(
                ctx,
                subtask_key,
                "workflow.finalize",
                status="traversed",
                label="finalize",
                metadata={"interactive": True},
            )
        else:
            subtask_error = indexed["index_result"].get("error", "index failed")
            failed_docs.append({
                "title": doc_data.get("title", "Untitled"),
                "error": subtask_error,
            })
            ctx.upsert_subtask(
                subtask_key=subtask_key,
                subtask_type="document_index",
                status="failed",
                progress=0,
                total=1,
                message=subtask_error,
                stage="index.failed",
                error=subtask_error,
                result={"title": doc_data.get("title", "Untitled")},
            )
            _set_edge_status(
                ctx,
                subtask_key,
                "workflow.finalize",
                status="traversed",
                label="finalize",
                metadata={"interactive": True},
            )
            ctx.emit_event(
                "index.batch.document_failed",
                subtask_error,
                stage="index.failed",
                level="error",
                payload={"title": doc_data.get("title", "Untitled"), "source": doc_data.get("source", "")},
            )

        progress = 0.1 + (0.9 * (i + 1) / total)
        ctx.update_progress(progress, f"已处理文档 {i + 1}/{total}...")

    ctx.emit_event(
        "index.batch.completed",
        "Batch indexing finished",
        stage="index.completed",
        payload={
            "indexed": len(indexed_docs),
            "failed": len(failed_docs),
            "total": total,
        },
    )
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        status="completed",
        progress=1,
        total=1,
        message="Workflow finalized",
        stage="workflow.finalize.completed",
        result={
            "indexed": len(indexed_docs),
            "failed": len(failed_docs),
            "total": total,
        },
    )
    return {
        "status": "success" if not failed_docs else "partial_success",
        "indexed": len(indexed_docs),
        "failed": len(failed_docs),
        "documents": indexed_docs,
        "failed_documents": failed_docs,
    }


def enqueue_index_document(title: str, content: str, source: str,
                           source_type: str = "txt",
                           metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    入队单文档索引任务

    Args:
        title: 文档标题
        content: 文档内容
        source: 来源
        source_type: 来源类型
        metadata: 可选元数据

    Returns:
        任务ID
    """
    task_id = str(uuid.uuid4())

    # 创建任务记录
    create_task_record("index_document", task_id)

    # 入队任务
    index_document_task(task_id, title, content, source, source_type, metadata)

    return task_id


def enqueue_batch_index(documents_data: List[Dict[str, Any]]) -> str:
    """
    入队批量索引任务

    Args:
        documents_data: 文档数据列表

    Returns:
        任务ID
    """
    task_id = str(uuid.uuid4())

    # 创建任务记录
    create_task_record("batch_index", task_id)

    # 入队任务
    batch_index_task(task_id, documents_data)

    return task_id
