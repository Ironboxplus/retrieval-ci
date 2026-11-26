"""TDD: crawl_urls_task inline-execution observability contract.

Regression test for the 2026-05-15 bug where API-side crawl tasks ran via an
ad-hoc threading + BrowserCrawler path that bypassed all subtask / event /
workflow / resource writes — task.status flipped to "completed" but the task
detail screens were empty (subtasks=0, events=0, edges=0).

The contract this test pins down: every code path that creates a DB-backed
crawl task MUST drive the @run_with_context-decorated huey task (directly or
via call_local), so that all runtime observability tables are populated.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from core.db import get_db_session
from core.db.models import (
    CrawlRecord,
    CrawlStatus as DBCrawlStatus,
    Task,
    TaskEvent,
    TaskStatus,
    TaskSubtask,
    TaskWorkflowEdge,
)
from pipeline.crawler.enhanced_crawler import (
    CrawlStatusType,
    EnhancedCrawlResult,
)


class _FakeEnhancedCrawler:
    """Stand-in for EnhancedCrawler used to keep tests offline and deterministic."""

    def __init__(self, fixtures: dict):
        self._fixtures = fixtures
        self._cb = None

    def set_status_callback(self, cb):
        self._cb = cb

    async def crawl_urls(self, urls, max_concurrent=5):
        results = []
        for url in urls:
            payload = self._fixtures.get(
                url, {"status": CrawlStatusType.FAILED, "error": "no fixture"}
            )
            if self._cb:
                self._cb(url, CrawlStatusType.CRAWLING, "Starting crawl")
            result = EnhancedCrawlResult(
                url=url,
                title=payload.get("title", ""),
                content=payload.get("content", ""),
                html=payload.get("html", ""),
                status=payload["status"],
                error=payload.get("error"),
                crawled_at=datetime.now(),
                retry_count=0,
                crawl_time=0.01,
            )
            if self._cb:
                self._cb(url, payload["status"], "Crawl finished")
            results.append(result)
        return results

    async def crawl_single(self, url):
        return (await self.crawl_urls([url]))[0]

    def get_stats(self):
        return {"success": 1, "failed": 0}

    def cancel(self):
        pass


def _make_test_task(url: str) -> int:
    with get_db_session() as sess:
        task = Task(
            task_type="crawl_urls",
            status=TaskStatus.pending,
            total=1,
            params_json={
                "urls": [url],
                "use_proxy": False,
                "auto_index": False,
                "max_concurrent": 1,
            },
        )
        sess.add(task)
        sess.flush()
        sess.add(CrawlRecord(task_id=task.id, url=url, status=DBCrawlStatus.pending))
        sess.commit()
        return task.id


def _cleanup(task_id: int) -> None:
    with get_db_session() as sess:
        sess.query(TaskWorkflowEdge).filter(TaskWorkflowEdge.task_id == task_id).delete()
        sess.query(TaskEvent).filter(TaskEvent.task_id == task_id).delete()
        sess.query(TaskSubtask).filter(TaskSubtask.task_id == task_id).delete()
        sess.query(CrawlRecord).filter(CrawlRecord.task_id == task_id).delete()
        sess.query(Task).filter(Task.id == task_id).delete()
        sess.commit()


@pytest.fixture
def success_fixture():
    return {
        "https://example.test/page": {
            "status": CrawlStatusType.SUCCESS,
            "title": "Example Page",
            "content": "hello world body",
            "html": "<p>hello world body</p>",
        }
    }


@pytest.fixture
def failure_fixture():
    return {
        "https://example.test/will-fail": {
            "status": CrawlStatusType.FAILED,
            "error": "network unreachable",
        }
    }


def test_call_local_populates_all_observability_tables(success_fixture):
    """Successful crawl through call_local must write subtasks/events/edges and a rich result_json."""
    url = next(iter(success_fixture))
    task_id = _make_test_task(url)

    try:
        from app.tasks.crawl_tasks import crawl_urls_task

        with patch(
            "app.tasks.crawl_tasks._build_enhanced_crawler",
            return_value=_FakeEnhancedCrawler(success_fixture),
        ):
            result = crawl_urls_task.call_local(str(task_id), [url], False, False, 1)

        assert isinstance(result, dict)
        for key in ("success_count", "failed_count", "total"):
            assert key in result, f"result_json missing {key}"
        assert result["success_count"] == 1
        assert result["total"] == 1

        with get_db_session() as sess:
            task = sess.query(Task).filter(Task.id == task_id).first()
            assert task is not None
            assert task.status == TaskStatus.completed
            assert task.started_at is not None and task.completed_at is not None

            subtasks = sess.query(TaskSubtask).filter(TaskSubtask.task_id == task_id).all()
            events = sess.query(TaskEvent).filter(TaskEvent.task_id == task_id).all()
            edges = sess.query(TaskWorkflowEdge).filter(TaskWorkflowEdge.task_id == task_id).all()

            assert len(subtasks) >= 3, (
                f"BUG regression: ad-hoc thread path skipped subtask writes "
                f"(got {len(subtasks)}, expected ≥3)"
            )
            assert len(events) >= 5, (
                f"BUG regression: event stream missing "
                f"(got {len(events)}, expected ≥5)"
            )
            assert len(edges) >= 1, "workflow edges must be written"

            keys = {s.subtask_key for s in subtasks}
            assert "workflow.prepare" in keys
            assert "workflow.finalize" in keys
            assert url in keys

            rec = (
                sess.query(CrawlRecord)
                .filter(CrawlRecord.task_id == task_id, CrawlRecord.url == url)
                .first()
            )
            assert rec.status == DBCrawlStatus.success
            assert rec.title == "Example Page"
            assert rec.content_length and rec.content_length > 0
    finally:
        _cleanup(task_id)


def test_call_local_failure_still_writes_observability(failure_fixture):
    """Failed crawl path must NOT regress to silent task with empty detail."""
    url = next(iter(failure_fixture))
    task_id = _make_test_task(url)

    try:
        from app.tasks.crawl_tasks import crawl_urls_task

        with patch(
            "app.tasks.crawl_tasks._build_enhanced_crawler",
            return_value=_FakeEnhancedCrawler(failure_fixture),
        ):
            result = crawl_urls_task.call_local(str(task_id), [url], False, False, 1)

        assert result["failed_count"] == 1
        assert result["success_count"] == 0

        with get_db_session() as sess:
            assert (
                sess.query(TaskSubtask).filter(TaskSubtask.task_id == task_id).count()
                >= 3
            )
            assert (
                sess.query(TaskEvent).filter(TaskEvent.task_id == task_id).count() >= 5
            )

            rec = (
                sess.query(CrawlRecord)
                .filter(CrawlRecord.task_id == task_id, CrawlRecord.url == url)
                .first()
            )
            assert rec.status == DBCrawlStatus.failed
            assert rec.error_message and "network" in rec.error_message
    finally:
        _cleanup(task_id)


def test_api_route_uses_call_local_not_browsercrawler():
    """The /api/crawl/urls handler must dispatch via crawl_urls_task.call_local.

    Catches the exact ad-hoc-thread regression: if someone reintroduces a path
    that pumps results into the DB without going through the huey task body,
    crawl_urls_task.call_local will not be invoked and this test fails.
    """
    from fastapi.testclient import TestClient
    import main as main_module

    invocations = []

    def _fake_call_local(task_id, urls, use_proxy, auto_index, max_concurrent):
        invocations.append(
            {
                "task_id": task_id,
                "urls": urls,
                "use_proxy": use_proxy,
                "auto_index": auto_index,
                "max_concurrent": max_concurrent,
            }
        )
        return {
            "status": "success",
            "total": len(urls),
            "success_count": len(urls),
            "failed_count": 0,
        }

    client = TestClient(main_module.app)
    target_url = "https://example.test/api-route"

    with patch(
        "app.tasks.crawl_tasks.crawl_urls_task.call_local",
        side_effect=_fake_call_local,
    ):
        resp = client.post(
            "/api/crawl/urls",
            json={
                "urls": [target_url],
                "use_proxy": False,
                "auto_index": False,
                "max_concurrent": 1,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        task_id_str = data["task_id"]

        # Background thread is daemon; wait a few seconds for call_local
        import time

        deadline = time.time() + 5.0
        while time.time() < deadline and not invocations:
            time.sleep(0.05)

    try:
        assert invocations, (
            "API handler did NOT invoke crawl_urls_task.call_local — the inline "
            "path was bypassed, likely regression to ad-hoc BrowserCrawler thread"
        )
        call = invocations[0]
        assert call["task_id"] == task_id_str
        assert call["urls"] == [target_url]
        assert call["max_concurrent"] == 1
    finally:
        try:
            _cleanup(int(task_id_str))
        except Exception:
            pass
