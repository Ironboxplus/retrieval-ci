"""
爬虫任务 - URL爬取相关的异步任务。

当前实现除了保留原有的队列语义，还额外补了：
1. URL 级子任务快照
2. 任务事件流
3. 进程资源采样
4. 子任务级重试可追踪数据
"""
import uuid
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings
from bs4 import BeautifulSoup

from app.tasks.task_queue import huey
from app.tasks.base import TaskContext, run_with_context, create_task_record


def _extracted_text_length(content: str) -> int:
    """Character count of extracted plain text, stripping HTML tags."""
    if not content:
        return 0
    return len(BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True))


def _db_task_id(task_id: str = None) -> Optional[int]:
    """Convert queue task ids to the DB task id shape used by crawl records."""
    if task_id and str(task_id).isdigit():
        return int(task_id)
    return None


def _update_crawl_record(
    *,
    task_id: str = None,
    url: str,
    status,
    title: str = None,
    content_length: int = None,
    doc_id: int = None,
    error_message: str = None,
    status_code: int = None,
    retry_count: int = None,
    crawl_time: float = None,
    proxy_used: str = None,
    db=None,
) -> None:
    """Update the existing crawl record when the API pre-created one for this task."""
    db_task_id = _db_task_id(task_id)
    if db_task_id is None:
        return

    from core.db.models import CrawlRecord

    def _apply(session):
        record = session.query(CrawlRecord).filter(
            CrawlRecord.task_id == db_task_id,
            CrawlRecord.url == url,
        ).first()
        if not record:
            return

        record.status = status
        record.title = title
        record.content_length = content_length
        record.doc_id = doc_id
        record.error_message = error_message
        # W41: diagnostic fields. Only overwrite when caller provides a value
        # so partial updates (e.g. status change without re-running crawl)
        # don't wipe earlier diagnostics.
        if status_code is not None:
            record.status_code = status_code
        if retry_count is not None:
            record.retry_count = retry_count
        if crawl_time is not None:
            record.crawl_time = crawl_time
        if proxy_used is not None:
            record.proxy_used = proxy_used

    if db is not None:
        _apply(db)
        return

    from core.db import get_db_session

    with get_db_session() as session:
        _apply(session)


def _build_enhanced_crawler(use_proxy: bool, max_concurrent: int):
    from pipeline.crawler.enhanced_crawler import EnhancedCrawler, CrawlerConfig
    from pipeline.crawler.retry_handler import RetryConfig, BackoffStrategy

    config = CrawlerConfig(
        headless=settings.crawler.headless,
        timeout=settings.crawler.timeout,
        max_concurrent=max_concurrent,
        delay_between_requests=settings.crawler.delay_between_requests,
        retry_config=RetryConfig(
            max_retries=settings.crawler.retry_times,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            initial_delay=2.0,
            max_delay=30.0,
        ),
    )
    config.proxy_config.enabled = use_proxy
    if use_proxy and settings.proxy.http_proxy:
        configured = [
            url for url in {settings.proxy.http_proxy, settings.proxy.https_proxy} if url
        ]
        config.proxy_config.proxies = configured
    return EnhancedCrawler(config)


def _subtask_status_from_crawl_status(status_value: str) -> str:
    mapping = {
        "pending": "pending",
        "crawling": "running",
        "success": "completed",
        "failed": "failed",
        "captcha": "failed",
        "blocked": "failed",
        "cancelled": "cancelled",
    }
    return mapping.get((status_value or "").lower(), "failed")


def _db_crawl_status(status_value: str):
    from core.db.models import CrawlStatus as DBCrawlStatus

    mapping = {
        "success": DBCrawlStatus.success,
        "cancelled": DBCrawlStatus.skipped,
    }
    return mapping.get((status_value or "").lower(), DBCrawlStatus.failed)


def _result_payload(result) -> Dict[str, Any]:
    status_value = result.status.value if hasattr(result.status, "value") else result.status
    payload = {
        "url": result.url,
        "title": result.title,
        "content_length": _extracted_text_length(result.content),
        "status": status_value,
        "status_code": getattr(result, "status_code", None),
        "error": result.error,
        "retry_count": getattr(result, "retry_count", 0),
        "crawl_time": getattr(result, "crawl_time", 0.0),
        "proxy_used": getattr(result, "proxy_used", None),
        "captcha_detected": getattr(result, "captcha_detected", False),
        "captcha_type": (
            result.captcha_type.value
            if getattr(result, "captcha_type", None) is not None and hasattr(result.captcha_type, "value")
            else None
        ),
        "crawled_at": result.crawled_at.isoformat() if getattr(result, "crawled_at", None) else None,
    }
    return payload


def _index_crawled_result(result, task_id: str = None) -> Dict[str, Any]:
    """Persist crawled content through the canonical DB + index write path.

    Uses the shared `get_index_service(chunker_type='section_l1')` singleton —
    same fast L1+L2 schema as Stage R ICLR ingest (SemanticSectionChunker with
    semantic_inner_split=False). Webpages emit:
      - 1 L1 title_summary chunk (title + first 500 bge-m3 tokens)
      - N L2 section chunks (size-bounded heading-aware split)
    Aligns webpage retrieval with paper retrieval so the L1 dedup_by_paper
    + reranker assumptions hold across source types (W42).

    `content_type='html'` triggers HTMLNodeParser tag-strip + flatten before
    chunking so result.content (raw HTML from EnhancedCrawler) is reduced to
    visible text first.
    """
    from core.db import get_db_session
    from core.db.models import (
        Document as DBDocument,
        CrawlStatus as DBCrawlStatus,
        SourceType,
        DataSource,
        DocumentStatus,
    )
    from app.services.index_service import get_index_service

    metadata = {
        "crawled_at": result.crawled_at.isoformat() if result.crawled_at else None,
        "retry_count": getattr(result, "retry_count", 0),
        "crawl_time": getattr(result, "crawl_time", 0.0),
        "status_code": getattr(result, "status_code", None),
        "proxy_used": getattr(result, "proxy_used", None),
    }

    # W49: split into two DB sessions so the long embed+upsert doesn't
    # hold a SQLite write lock. Session 1 creates the doc row and commits
    # immediately; session 2 (inside index_document) writes chunks after
    # embedding finishes. This prevents "database is locked" on concurrent
    # task_subtask updates that happen while embedding runs.
    with get_db_session() as db:
        doc = DBDocument(
            title=result.title or result.url,
            source=result.url,
            source_type=SourceType.webpage,
            data_source=DataSource.user,
            url=result.url,
            status=DocumentStatus.pending,
            metadata_json=metadata,
        )
        db.add(doc)
        db.flush()
        doc_id = doc.id

    # W48: result.content is plain text (innerText); result.html is raw
    # HTML. Chunker needs raw HTML for heading-aware L2 splits.
    raw_html = getattr(result, "html", None)
    if raw_html:
        chunk_content = raw_html
        chunk_content_type = "html"
    else:
        chunk_content = result.content
        chunk_content_type = "text"

    index_result = get_index_service(chunker_type="section_l1").index_document(
        doc_id=doc_id,
        content=chunk_content,
        title=result.title or result.url,
        source=result.url,
        metadata=metadata,
        source_type="webpage",
        data_source="user",
        content_type=chunk_content_type,
    )

    with get_db_session() as db:
        doc = db.get(DBDocument, doc_id)
        doc.status = DocumentStatus.indexed if index_result.get("success") else DocumentStatus.failed
        _update_crawl_record(
            task_id=task_id,
            url=result.url,
            status=DBCrawlStatus.success,
            title=result.title,
            content_length=_extracted_text_length(result.content),
            doc_id=doc_id,
            error_message=None,
            status_code=getattr(result, "status_code", None),
            retry_count=getattr(result, "retry_count", None),
            crawl_time=getattr(result, "crawl_time", None),
            proxy_used=getattr(result, "proxy_used", None),
            db=db,
        )

        return {
            "doc_id": doc_id,
            "chunks_count": index_result.get("chunk_count", 0),
            "index_result": index_result,
        }


def _emit_subtask_runtime_event(ctx: TaskContext, url: str, status_value: str, message: str):
    subtask = ctx.upsert_subtask(
        subtask_key=url,
        subtask_type="crawl_url",
        name=url,
        status=_subtask_status_from_crawl_status(status_value),
        progress=1 if status_value == "success" else 0,
        total=1,
        message=message,
        stage=f"crawl.{status_value}",
        metadata={
            "url": url,
            "target_url": url,
            "last_runtime_status": status_value,
            "workflow_action_type": "crawl_url",
            "depends_on": ["workflow.prepare"],
            "retriable": True,
            "node_group": "crawl",
        },
    )
    ctx.emit_event(
        "crawl.subtask.status",
        message or f"{url} -> {status_value}",
        stage=f"crawl.{status_value}",
        payload={"url": url, "status": status_value},
        subtask_id=subtask["id"] if subtask else None,
    )


def _set_edge_status(
    ctx: TaskContext,
    from_key: str,
    to_key: str,
    *,
    status: str,
    label: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Update a workflow edge in-place so graph state follows runtime state."""
    ctx.upsert_edge(
        edge_key=f"{from_key}->{to_key}",
        from_key=from_key,
        to_key=to_key,
        status=status,
        label=label,
        metadata=metadata,
    )


def _sync_rollup_if_possible(ctx: TaskContext, status: str) -> Optional[Dict[str, Any]]:
    from app.services.task_runtime import sync_crawl_task_rollup

    db_task_id = _db_task_id(getattr(ctx, "task_id", None))
    if db_task_id is None:
        return None
    return sync_crawl_task_rollup(db_task_id, status=status)


@huey.task()
@run_with_context("crawl_urls")
def crawl_urls_task(ctx: TaskContext, urls: List[str],
                    use_proxy: bool = False,
                    auto_index: bool = False,
                    max_concurrent: int = 5):
    """
    URL爬取任务。

    现在使用增强版爬虫以便暴露 URL 级状态、重试次数、耗时等信息。
    """
    total = len(urls)
    if total == 0:
        return {"status": "success", "crawled": 0, "results": []}

    finalize_dependencies = urls if not auto_index else [f"index::{url}" for url in urls]
    ctx.upsert_subtask(
        subtask_key="workflow.prepare",
        subtask_type="workflow_action",
        name="Prepare Crawl Job",
        status="running",
        progress=0,
        total=1,
        message="Preparing workflow",
        stage="workflow.prepare.running",
        metadata={
            "workflow_action_type": "prepare",
            "depends_on": [],
            "retriable": False,
            "node_group": "system",
            "is_phantom": True,
        },
    )
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        name="Finalize Crawl Job",
        status="pending",
        progress=0,
        total=1,
        message="Waiting for upstream actions",
        stage="workflow.finalize.pending",
        metadata={
            "workflow_action_type": "finalize",
            "depends_on": finalize_dependencies,
            "retriable": False,
            "node_group": "system",
            "is_phantom": True,
        },
    )
    for url in urls:
        ctx.upsert_subtask(
            subtask_key=url,
            subtask_type="crawl_url",
            name=url,
            status="pending",
            progress=0,
            total=1,
            message="Queued",
            stage="crawl.pending",
            metadata={
                "url": url,
                "target_url": url,
                "use_proxy": use_proxy,
                "auto_index": auto_index,
                "max_concurrent": max_concurrent,
                "workflow_action_type": "crawl_url",
                "depends_on": ["workflow.prepare"],
                "retriable": True,
                "node_group": "crawl",
            },
        )
        ctx.upsert_edge(
            edge_key=f"workflow.prepare->{url}",
            from_key="workflow.prepare",
            to_key=url,
            status="active",
            label="prepare",
            metadata={"interactive": True},
        )
        if auto_index:
            ctx.upsert_subtask(
                subtask_key=f"index::{url}",
                subtask_type="index_url",
                name=f"Index {url}",
                status="pending",
                progress=0,
                total=1,
                message="Waiting for crawl result",
                stage="workflow.index.pending",
                metadata={
                    "workflow_action_type": "index_document",
                    "depends_on": [url],
                    "target_url": url,
                    "retriable": False,
                    "node_group": "index",
                },
            )
            ctx.upsert_edge(
                edge_key=f"{url}->index::{url}",
                from_key=url,
                to_key=f"index::{url}",
                status="active",
                label="index",
                metadata={"interactive": True},
            )
            ctx.upsert_edge(
                edge_key=f"index::{url}->workflow.finalize",
                from_key=f"index::{url}",
                to_key="workflow.finalize",
                status="active",
                label="finalize",
                metadata={"interactive": True},
            )
        else:
            ctx.upsert_edge(
                edge_key=f"{url}->workflow.finalize",
                from_key=url,
                to_key="workflow.finalize",
                status="active",
                label="finalize",
                metadata={"interactive": True},
            )

    ctx.emit_event(
        "crawl.batch.queued",
        f"Queued crawl job for {total} URLs",
        stage="crawl.queued",
        payload={
            "total": total,
            "use_proxy": use_proxy,
            "auto_index": auto_index,
            "max_concurrent": max_concurrent,
        },
    )
    ctx.update_progress(0.05, f"初始化增强爬虫, 共 {total} 个URL...")
    ctx.upsert_subtask(
        subtask_key="workflow.prepare",
        subtask_type="workflow_action",
        status="completed",
        progress=1,
        total=1,
        message="Workflow prepared",
        stage="workflow.prepare.completed",
    )
    for url in urls:
        _set_edge_status(
            ctx,
            "workflow.prepare",
            url,
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
            message="Workflow cancelled before crawl start",
            stage="workflow.finalize.cancelled",
        )
        return {"status": "cancelled"}

    crawler = _build_enhanced_crawler(use_proxy=use_proxy, max_concurrent=max_concurrent)

    def status_callback(url: str, status, message: str = ""):
        status_value = status.value if hasattr(status, "value") else str(status)
        _emit_subtask_runtime_event(ctx, url, status_value, message)

    crawler.set_status_callback(status_callback)
    ctx.update_progress(0.15, "开始增强爬取...")
    ctx.sample_resources(phase="crawl.started", extra={"url_count": total})

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        results = loop.run_until_complete(
            crawler.crawl_urls(urls, max_concurrent=max_concurrent)
        )
    finally:
        # W45: cancel pending tasks before close so playwright browsers
        # don't leak when crawl_urls raises mid-flight or the worker is
        # interrupted. asyncio.all_tasks(loop) requires the loop attribute,
        # not the running event loop (which is already torn down here).
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()

    if ctx.is_cancelled():
        crawler.cancel()
        ctx.upsert_subtask(
            subtask_key="workflow.finalize",
            subtask_type="workflow_action",
            status="cancelled",
            progress=0,
            total=1,
            message="Workflow cancelled during crawl",
            stage="workflow.finalize.cancelled",
        )
        _sync_rollup_if_possible(ctx, "cancelled")
        return {"status": "cancelled", "crawled": len(results)}

    ctx.update_progress(0.65, "爬取完成, 汇总并写入子任务结果...")
    ctx.emit_event(
        "crawl.batch.results",
        "Received crawl results",
        stage="crawl.results",
        payload={"total_results": len(results), "stats": crawler.get_stats()},
    )

    success_results = []
    failed_results = []
    captcha_results = []
    blocked_results = []
    indexed_docs = []

    processed_count = 0
    for result in results:
        result_data = _result_payload(result)
        subtask = ctx.upsert_subtask(
            subtask_key=result.url,
            subtask_type="crawl_url",
            name=result.url,
            status=_subtask_status_from_crawl_status(result_data["status"]),
            progress=1 if result_data["status"] == "success" else 0,
            total=1,
            message=result_data.get("error") or result_data["status"],
            stage=f"crawl.{result_data['status']}",
            result=result_data,
            error=result_data.get("error"),
            metadata={
                "status_code": result_data.get("status_code"),
                "proxy_used": result_data.get("proxy_used"),
                "crawler_retry_count": result_data.get("retry_count", 0),
            },
        )

        ctx.emit_event(
            "crawl.subtask.result",
            f"{result.url} -> {result_data['status']}",
            stage=f"crawl.{result_data['status']}",
            payload=result_data,
            subtask_id=subtask["id"] if subtask else None,
            level="error" if result_data["status"] in {"failed", "blocked", "captcha"} else "info",
        )
        if subtask:
            ctx.sample_resources(
                phase=f"crawl.{result_data['status']}",
                subtask_id=subtask["id"],
                extra={"url": result.url},
            )

        if result_data["status"] == "success":
            _update_crawl_record(
                task_id=getattr(ctx, "task_id", None),
                url=result.url,
                status=_db_crawl_status(result_data["status"]),
                title=result.title,
                content_length=_extracted_text_length(result.content),
                error_message=None,
                status_code=result_data.get("status_code"),
                retry_count=result_data.get("retry_count"),
                crawl_time=result_data.get("crawl_time"),
                proxy_used=result_data.get("proxy_used"),
            )

            if auto_index:
                if result.content:
                    _set_edge_status(
                        ctx,
                        result.url,
                        f"index::{result.url}",
                        status="traversed",
                        label="index",
                        metadata={"interactive": True},
                    )
                    ctx.upsert_subtask(
                        subtask_key=f"index::{result.url}",
                        subtask_type="index_url",
                        status="running",
                        progress=0,
                        total=1,
                        message="Indexing crawled document",
                        stage="workflow.index.running",
                    )
                    ctx.emit_event(
                        "crawl.subtask.indexing",
                        f"Indexing {result.url}",
                        stage="crawl.indexing",
                        payload={"url": result.url},
                        subtask_id=subtask["id"] if subtask else None,
                    )
                    try:
                        indexed = _index_crawled_result(result, getattr(ctx, "task_id", None))
                        result_data["doc_id"] = indexed["doc_id"]
                        result_data["chunks_count"] = indexed["chunks_count"]
                        result_data["indexed"] = indexed["index_result"].get("success", False)
                        if result_data["indexed"]:
                            indexed_docs.append(
                                {
                                    "doc_id": indexed["doc_id"],
                                    "url": result.url,
                                    "chunks_count": indexed["chunks_count"],
                                }
                            )
                            ctx.upsert_subtask(
                                subtask_key=result.url,
                                subtask_type="crawl_url",
                                status="completed",
                                result=result_data,
                                metadata={
                                    "indexed": True,
                                    "doc_id": indexed["doc_id"],
                                    "chunks_count": indexed["chunks_count"],
                                },
                            )
                            ctx.upsert_subtask(
                                subtask_key=f"index::{result.url}",
                                subtask_type="index_url",
                                status="completed",
                                progress=1,
                                total=1,
                                message="Index completed",
                                stage="workflow.index.completed",
                                result={
                                    "target_url": result.url,
                                    "doc_id": indexed["doc_id"],
                                    "chunks_count": indexed["chunks_count"],
                                },
                                metadata={"target_url": result.url},
                            )
                        else:
                            index_error = indexed["index_result"].get("error") or "Indexing failed"
                            result_data["index_error"] = index_error
                            # Crawl succeeded; only the downstream index call failed.
                            # Keep CrawlRecord.status=success so the records table
                            # accurately reports the crawl outcome and surface the
                            # index failure via error_message + indexed_docs bucket.
                            _update_crawl_record(
                                task_id=getattr(ctx, "task_id", None),
                                url=result.url,
                                status=_db_crawl_status("success"),
                                title=result.title,
                                content_length=_extracted_text_length(result.content),
                                doc_id=indexed["doc_id"],
                                error_message=f"Index failed: {index_error}",
                            )
                            ctx.upsert_subtask(
                                subtask_key=result.url,
                                subtask_type="crawl_url",
                                status="completed",
                                result=result_data,
                                metadata={
                                    "indexed": False,
                                    "doc_id": indexed["doc_id"],
                                    "chunks_count": indexed["chunks_count"],
                                },
                            )
                            ctx.upsert_subtask(
                                subtask_key=f"index::{result.url}",
                                subtask_type="index_url",
                                status="failed",
                                progress=0,
                                total=1,
                                message=index_error,
                                stage="workflow.index.failed",
                                error=index_error,
                                result={
                                    "target_url": result.url,
                                    "doc_id": indexed["doc_id"],
                                    "chunks_count": indexed["chunks_count"],
                                },
                                metadata={"target_url": result.url},
                            )
                            ctx.emit_event(
                                "crawl.subtask.index_failed",
                                index_error,
                                stage="crawl.index_failed",
                                payload={"url": result.url, "error": index_error},
                                subtask_id=subtask["id"] if subtask else None,
                                level="error",
                            )
                    except Exception as e:
                        result_data["indexed"] = False
                        result_data["index_error"] = str(e)
                        # Crawl succeeded (we have result.content); only index
                        # raised. Keep status=success and report index failure
                        # via error_message — see partner block above.
                        _update_crawl_record(
                            task_id=getattr(ctx, "task_id", None),
                            url=result.url,
                            status=_db_crawl_status("success"),
                            title=result.title,
                            content_length=_extracted_text_length(result.content),
                            error_message=f"Index failed: {e}",
                        )
                        ctx.upsert_subtask(
                            subtask_key=result.url,
                            subtask_type="crawl_url",
                            status="completed",
                            result=result_data,
                            metadata={"indexed": False},
                        )
                        ctx.upsert_subtask(
                            subtask_key=f"index::{result.url}",
                            subtask_type="index_url",
                            status="failed",
                            progress=0,
                            total=1,
                            message=str(e),
                            stage="workflow.index.failed",
                            error=str(e),
                            metadata={"target_url": result.url},
                        )
                        ctx.emit_event(
                            "crawl.subtask.index_failed",
                            str(e),
                            stage="crawl.index_failed",
                            payload={"url": result.url, "error": str(e)},
                            subtask_id=subtask["id"] if subtask else None,
                            level="error",
                        )
                else:
                    ctx.upsert_subtask(
                        subtask_key=f"index::{result.url}",
                        subtask_type="index_url",
                        status="skipped",
                        progress=0,
                        total=1,
                        message="Skipped because crawl content was empty",
                        stage="workflow.index.skipped",
                        result={
                            "target_url": result.url,
                            "reason": "empty_content",
                        },
                        metadata={"target_url": result.url, "blocked_by": result.url},
                    )
                    _set_edge_status(
                        ctx,
                        result.url,
                        f"index::{result.url}",
                        status="blocked",
                        label="index",
                        metadata={
                            "interactive": True,
                            "reason": "empty_content",
                        },
                    )
            else:
                _set_edge_status(
                    ctx,
                    result.url,
                    "workflow.finalize",
                    status="traversed",
                    label="finalize",
                    metadata={"interactive": True},
                )
            success_results.append(result_data)
        else:
            if auto_index:
                ctx.upsert_subtask(
                    subtask_key=f"index::{result.url}",
                    subtask_type="index_url",
                    status="skipped",
                    progress=0,
                    total=1,
                    message="Skipped because crawl step failed",
                    stage="workflow.index.skipped",
                    result={
                        "target_url": result.url,
                        "reason": "crawl_failed",
                        "crawl_status": result_data["status"],
                    },
                    metadata={"target_url": result.url, "blocked_by": result.url},
                )
                _set_edge_status(
                    ctx,
                    result.url,
                    f"index::{result.url}",
                    status="blocked",
                    label="index",
                    metadata={
                        "interactive": True,
                        "reason": "crawl_failed",
                        "crawl_status": result_data["status"],
                    },
                )
            else:
                _set_edge_status(
                    ctx,
                    result.url,
                    "workflow.finalize",
                    status="traversed",
                    label="finalize",
                    metadata={"interactive": True},
                )
            _update_crawl_record(
                task_id=getattr(ctx, "task_id", None),
                url=result.url,
                status=_db_crawl_status(result_data["status"]),
                error_message=result_data.get("error"),
                status_code=result_data.get("status_code"),
                retry_count=result_data.get("retry_count"),
                crawl_time=result_data.get("crawl_time"),
                proxy_used=result_data.get("proxy_used"),
            )
            failure_entry = {
                "url": result.url,
                "error": result_data.get("error"),
                "retry_count": result_data.get("retry_count"),
                "crawl_time": result_data.get("crawl_time"),
                "status_code": result_data.get("status_code"),
            }
            if result_data["status"] == "captcha":
                failure_entry["captcha_type"] = result_data.get("captcha_type")
                captcha_results.append(failure_entry)
            elif result_data["status"] == "blocked":
                blocked_results.append(failure_entry)
            else:
                failed_results.append(failure_entry)

        processed_count += 1
        progress = 0.65 + (0.30 * processed_count / total)
        ctx.update_progress(progress, f"已处理 URL {processed_count}/{total}")

    ctx.update_progress(0.98, "重建任务聚合结果...")
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        status="running",
        progress=0,
        total=1,
        message="Rebuilding rollup",
        stage="workflow.finalize.running",
    )
    for dependency_key in finalize_dependencies:
        _set_edge_status(
            ctx,
            dependency_key,
            "workflow.finalize",
            status="traversed",
            label="finalize",
            metadata={"interactive": True},
        )
    any_index_failed = auto_index and any(
        r.get("indexed") is False for r in success_results
    )
    rollup = _sync_rollup_if_possible(
        ctx, "failed" if any_index_failed else "completed",
    )
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        status="completed",
        progress=1,
        total=1,
        message="Workflow finalized",
        stage="workflow.finalize.completed",
        result=rollup,
    )
    ctx.emit_event(
        "crawl.batch.completed",
        "Crawl batch completed",
        stage="crawl.completed",
        payload=rollup or {
            "total": total,
            "success_count": len(success_results),
            "failed_count": len(failed_results),
            "blocked_count": len(blocked_results),
            "captcha_count": len(captcha_results),
            "stats": crawler.get_stats(),
        },
    )
    ctx.update_progress(1.0, "任务完成")

    return rollup or {
        "status": "partial" if any_index_failed else "success",
        "total": total,
        "success_count": len(success_results),
        "failed_count": len(failed_results),
        "blocked_count": len(blocked_results),
        "captcha_count": len(captcha_results),
        "success_results": success_results,
        "failed_results": failed_results,
        "blocked_results": blocked_results,
        "captcha_results": captcha_results,
        "indexed_docs": indexed_docs if auto_index else None,
        "stats": crawler.get_stats(),
    }


@huey.task()
@run_with_context("crawl_single")
def crawl_single_task(ctx: TaskContext, url: str,
                      use_proxy: bool = False,
                      auto_index: bool = False):
    """
    单URL爬取任务。
    """
    from pipeline.crawler.enhanced_crawler import CrawlStatusType

    finalize_dependencies = [url] if not auto_index else [f"index::{url}"]
    ctx.upsert_subtask(
        subtask_key="workflow.prepare",
        subtask_type="workflow_action",
        name="Prepare Crawl Job",
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
        subtask_key=url,
        subtask_type="crawl_url",
        name=url,
        status="pending",
        progress=0,
        total=1,
        message="Queued",
        stage="crawl.pending",
        metadata={
            "url": url,
            "target_url": url,
            "use_proxy": use_proxy,
            "auto_index": auto_index,
            "workflow_action_type": "crawl_url",
            "depends_on": ["workflow.prepare"],
            "retriable": True,
            "node_group": "crawl",
        },
    )
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        name="Finalize Crawl Job",
        status="pending",
        progress=0,
        total=1,
        message="Waiting for upstream actions",
        stage="workflow.finalize.pending",
        metadata={
            "workflow_action_type": "finalize",
            "depends_on": finalize_dependencies,
            "retriable": False,
            "node_group": "system",
        },
    )
    ctx.upsert_edge(
        edge_key=f"workflow.prepare->{url}",
        from_key="workflow.prepare",
        to_key=url,
        status="traversed",
        label="prepare",
        metadata={"interactive": True},
    )
    if auto_index:
        ctx.upsert_subtask(
            subtask_key=f"index::{url}",
            subtask_type="index_url",
            name=f"Index {url}",
            status="pending",
            progress=0,
            total=1,
            message="Waiting for crawl result",
            stage="workflow.index.pending",
            metadata={
                "workflow_action_type": "index_document",
                "depends_on": [url],
                "target_url": url,
                "retriable": False,
                "node_group": "index",
            },
        )
        ctx.upsert_edge(
            edge_key=f"{url}->index::{url}",
            from_key=url,
            to_key=f"index::{url}",
            status="active",
            label="index",
            metadata={"interactive": True},
        )
        ctx.upsert_edge(
            edge_key=f"index::{url}->workflow.finalize",
            from_key=f"index::{url}",
            to_key="workflow.finalize",
            status="active",
            label="finalize",
            metadata={"interactive": True},
        )
    else:
        ctx.upsert_edge(
            edge_key=f"{url}->workflow.finalize",
            from_key=url,
            to_key="workflow.finalize",
            status="active",
            label="finalize",
            metadata={"interactive": True},
        )
    ctx.emit_event(
        "crawl.single.queued",
        f"Queued crawl job for {url}",
        stage="crawl.queued",
        payload={"url": url, "use_proxy": use_proxy, "auto_index": auto_index},
    )
    ctx.update_progress(0.1, "初始化增强爬虫...")

    if ctx.is_cancelled():
        ctx.upsert_subtask(
            subtask_key="workflow.finalize",
            subtask_type="workflow_action",
            status="cancelled",
            progress=0,
            total=1,
            message="Workflow cancelled before crawl start",
            stage="workflow.finalize.cancelled",
        )
        return {"status": "cancelled"}

    crawler = _build_enhanced_crawler(use_proxy=use_proxy, max_concurrent=1)

    def status_callback(callback_url: str, status, message: str = ""):
        status_value = status.value if hasattr(status, "value") else str(status)
        _emit_subtask_runtime_event(ctx, callback_url, status_value, message)

    crawler.set_status_callback(status_callback)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(crawler.crawl_single(url))
    finally:
        # W45: cancel pending tasks before close to prevent playwright leak.
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()

    result_data = _result_payload(result)
    subtask = ctx.upsert_subtask(
        subtask_key=url,
        subtask_type="crawl_url",
        name=url,
        status=_subtask_status_from_crawl_status(result_data["status"]),
        progress=1 if result_data["status"] == "success" else 0,
        total=1,
        message=result_data.get("error") or result_data["status"],
        stage=f"crawl.{result_data['status']}",
        result=result_data,
        error=result_data.get("error"),
        metadata={"crawler_retry_count": result_data.get("retry_count", 0)},
    )

    if result.status != CrawlStatusType.SUCCESS:
        if auto_index:
            ctx.upsert_subtask(
                subtask_key=f"index::{url}",
                subtask_type="index_url",
                status="skipped",
                progress=0,
                total=1,
                message="Skipped because crawl step failed",
                stage="workflow.index.skipped",
                result={"target_url": url, "reason": "crawl_failed", "crawl_status": result_data["status"]},
                metadata={"target_url": url, "blocked_by": url},
            )
            _set_edge_status(
                ctx,
                url,
                f"index::{url}",
                status="blocked",
                label="index",
                metadata={"interactive": True, "reason": "crawl_failed", "crawl_status": result_data["status"]},
            )
        else:
            _set_edge_status(
                ctx,
                url,
                "workflow.finalize",
                status="traversed",
                label="finalize",
                metadata={"interactive": True},
            )
        ctx.upsert_subtask(
            subtask_key="workflow.finalize",
            subtask_type="workflow_action",
            status="running",
            progress=0,
            total=1,
            message="Finalizing failed crawl",
            stage="workflow.finalize.running",
        )
        for dependency_key in finalize_dependencies:
            _set_edge_status(
                ctx,
                dependency_key,
                "workflow.finalize",
                status="traversed",
                label="finalize",
                metadata={"interactive": True},
            )
        _update_crawl_record(
            task_id=getattr(ctx, "task_id", None),
            url=result.url or url,
            status=_db_crawl_status(result_data["status"]),
            error_message=result.error,
            status_code=result_data.get("status_code"),
            retry_count=result_data.get("retry_count"),
            crawl_time=result_data.get("crawl_time"),
            proxy_used=result_data.get("proxy_used"),
        )
        ctx.emit_event(
            "crawl.single.failed",
            result.error or "爬取失败",
            stage=f"crawl.{result_data['status']}",
            payload=result_data,
            subtask_id=subtask["id"] if subtask else None,
            level="error",
        )
        ctx.upsert_subtask(
            subtask_key="workflow.finalize",
            subtask_type="workflow_action",
            status="completed",
            progress=1,
            total=1,
            message="Workflow finalized with failure",
            stage="workflow.finalize.completed",
            result={
                "status": "failed",
                "url": url,
                "error": result.error,
            },
        )
        ctx.set_error(result.error or "爬取失败")
        return {
            "status": "failed",
            "url": url,
            "error": result.error,
            "retry_count": result_data.get("retry_count"),
            "crawl_time": result_data.get("crawl_time"),
        }

    ctx.update_progress(0.7, "爬取完成")
    _update_crawl_record(
        task_id=getattr(ctx, "task_id", None),
        url=result.url,
        status=_db_crawl_status(result_data["status"]),
        title=result.title,
        content_length=_extracted_text_length(result.content),
        error_message=None,
        status_code=result_data.get("status_code"),
        retry_count=result_data.get("retry_count"),
        crawl_time=result_data.get("crawl_time"),
        proxy_used=result_data.get("proxy_used"),
    )

    if auto_index and result.content:
        _set_edge_status(
            ctx,
            url,
            f"index::{url}",
            status="traversed",
            label="index",
            metadata={"interactive": True},
        )
        ctx.upsert_subtask(
            subtask_key=f"index::{url}",
            subtask_type="index_url",
            status="running",
            progress=0,
            total=1,
            message="Indexing crawled document",
            stage="workflow.index.running",
        )
        ctx.update_progress(0.85, "索引页面内容...")
        try:
            indexed = _index_crawled_result(result, getattr(ctx, "task_id", None))
            result_data["indexed"] = True
            result_data["doc_id"] = indexed["doc_id"]
            result_data["chunks_count"] = indexed["chunks_count"]
            ctx.upsert_subtask(
                subtask_key=url,
                subtask_type="crawl_url",
                status="completed",
                result=result_data,
                metadata={"indexed": True, "doc_id": indexed["doc_id"]},
            )
            ctx.upsert_subtask(
                subtask_key=f"index::{url}",
                subtask_type="index_url",
                status="completed",
                progress=1,
                total=1,
                message="Index completed",
                stage="workflow.index.completed",
                result={
                    "target_url": url,
                    "doc_id": indexed["doc_id"],
                    "chunks_count": indexed["chunks_count"],
                },
                metadata={"target_url": url},
            )
        except Exception as e:
            result_data["indexed"] = False
            result_data["index_error"] = str(e)
            ctx.upsert_subtask(
                subtask_key=url,
                subtask_type="crawl_url",
                status="completed",
                result=result_data,
                metadata={"indexed": False},
            )
            ctx.upsert_subtask(
                subtask_key=f"index::{url}",
                subtask_type="index_url",
                status="failed",
                progress=0,
                total=1,
                message=str(e),
                stage="workflow.index.failed",
                error=str(e),
                metadata={"target_url": url},
            )
            ctx.emit_event(
                "crawl.subtask.index_failed",
                str(e),
                stage="crawl.index_failed",
                payload={"url": url, "error": str(e)},
                subtask_id=subtask["id"] if subtask else None,
                level="error",
            )
    elif auto_index:
        ctx.upsert_subtask(
            subtask_key=f"index::{url}",
            subtask_type="index_url",
            status="skipped",
            progress=0,
            total=1,
            message="Skipped because crawl content was empty",
            stage="workflow.index.skipped",
            result={"target_url": url, "reason": "empty_content"},
            metadata={"target_url": url, "blocked_by": url},
        )
        _set_edge_status(
            ctx,
            url,
            f"index::{url}",
            status="blocked",
            label="index",
            metadata={"interactive": True, "reason": "empty_content"},
        )
    else:
        _set_edge_status(
            ctx,
            url,
            "workflow.finalize",
            status="traversed",
            label="finalize",
            metadata={"interactive": True},
        )

    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        status="running",
        progress=0,
        total=1,
        message="Finalizing crawl",
        stage="workflow.finalize.running",
    )
    for dependency_key in finalize_dependencies:
        _set_edge_status(
            ctx,
            dependency_key,
            "workflow.finalize",
            status="traversed",
            label="finalize",
            metadata={"interactive": True},
        )

    ctx.emit_event(
        "crawl.single.completed",
        f"Crawled {url}",
        stage="crawl.completed",
        payload=result_data,
        subtask_id=subtask["id"] if subtask else None,
    )
    ctx.upsert_subtask(
        subtask_key="workflow.finalize",
        subtask_type="workflow_action",
        status="completed",
        progress=1,
        total=1,
        message="Workflow finalized",
        stage="workflow.finalize.completed",
        result=result_data,
    )
    if auto_index and result_data.get("indexed") is False:
        ctx.set_error(f"Index failed: {result_data.get('index_error', 'unknown')}")
    ctx.update_progress(1.0, "任务完成")
    return result_data


def enqueue_crawl_urls(urls: List[str], use_proxy: bool = False,
                       auto_index: bool = False,
                       max_concurrent: int = 5,
                       task_id: str = None) -> str:
    """
    调度URL爬取任务。

    使用后台线程 + ``crawl_urls_task.call_local`` 绕过 huey 队列，避免在没有
    Huey worker 时任务永久 pending。同步逻辑仍然走 ``@run_with_context``
    装饰器，保证 subtask/event/workflow/资源采样全部写入。
    """
    import threading

    if task_id is None:
        task_id = str(uuid.uuid4())

    create_task_record("crawl_urls", task_id)

    def _runner():
        try:
            crawl_urls_task.call_local(
                task_id, urls, use_proxy, auto_index, max_concurrent
            )
        except Exception as e:
            from core.logging import get_logger
            get_logger("crawl_thread").error(
                f"crawl_urls inline run failed: task={task_id}, error={e}",
                exc_info=True,
            )

    threading.Thread(target=_runner, daemon=True).start()
    return task_id


def enqueue_crawl_single(url: str, use_proxy: bool = False,
                         auto_index: bool = False) -> str:
    """
    调度单URL爬取任务（同 enqueue_crawl_urls：线程 + call_local）。
    """
    import threading

    task_id = str(uuid.uuid4())
    create_task_record("crawl_single", task_id)

    def _runner():
        try:
            crawl_single_task.call_local(task_id, url, use_proxy, auto_index)
        except Exception as e:
            from core.logging import get_logger
            get_logger("crawl_thread").error(
                f"crawl_single inline run failed: task={task_id}, error={e}",
                exc_info=True,
            )

    threading.Thread(target=_runner, daemon=True).start()
    return task_id
