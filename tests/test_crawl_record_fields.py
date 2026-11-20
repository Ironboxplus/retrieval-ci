"""TDD: CrawlRecord must surface diagnostic fields (status_code / retry_count /
crawl_time / proxy_used) so the records page can show *why* a crawl failed and
which proxy was used.

Bug discovered 2026-05-20: EnhancedCrawler captures these fields,
crawl_tasks._result_payload propagates them in-memory, but neither the DB
column nor the API response model carry them. UI is blind to retry count,
HTTP status code, and proxy used.

Contract:
1. `CrawlRecord` ORM has columns: status_code (int?), retry_count (int?),
   crawl_time (float?), proxy_used (str?).
2. `_update_crawl_record` accepts + persists these fields.
3. `GET /api/crawl/records` returns them in `CrawlRecordResponse`.
4. Defaults are None — old rows pre-migration must still serialize.
"""
from __future__ import annotations

import pytest


def test_crawl_record_orm_has_diagnostic_columns():
    """ORM-level: CrawlRecord must declare 4 new columns."""
    from core.db.models import CrawlRecord

    columns = {c.name for c in CrawlRecord.__table__.columns}
    required = {"status_code", "retry_count", "crawl_time", "proxy_used"}
    missing = required - columns
    assert not missing, (
        f"CrawlRecord ORM missing diagnostic columns: {missing}. "
        f"Current columns: {sorted(columns)}"
    )


def test_update_crawl_record_persists_diagnostic_fields(tmp_path):
    """`_update_crawl_record` must write the 4 fields when given."""
    from core.db.database import init_db, get_db_session
    from core.db.models import CrawlRecord, CrawlStatus, Task, TaskStatus
    from app.tasks.crawl_tasks import _update_crawl_record

    init_db()

    with get_db_session() as db:
        task = Task(
            task_type="crawl_urls",
            status=TaskStatus.pending,
            total=1,
            params_json={"urls": ["https://example.test/diag-fields"]},
        )
        db.add(task)
        db.flush()
        record = CrawlRecord(
            task_id=task.id,
            url="https://example.test/diag-fields",
            status=CrawlStatus.pending,
        )
        db.add(record)
        db.flush()
        task_id = task.id
        record_id = record.id

    _update_crawl_record(
        task_id=str(task_id),
        url="https://example.test/diag-fields",
        status=CrawlStatus.failed,
        title=None,
        content_length=0,
        doc_id=None,
        error_message="proxy_timeout",
        status_code=502,
        retry_count=3,
        crawl_time=12.5,
        proxy_used="http://127.0.0.1:10808",
    )

    with get_db_session() as db:
        row = db.query(CrawlRecord).filter(CrawlRecord.id == record_id).first()
        assert row is not None
        assert row.status == CrawlStatus.failed
        assert row.error_message == "proxy_timeout"
        assert row.status_code == 502
        assert row.retry_count == 3
        assert row.crawl_time == pytest.approx(12.5)
        assert row.proxy_used == "http://127.0.0.1:10808"


def test_crawl_record_response_includes_diagnostic_fields():
    """The pydantic response model exposes the 4 fields to clients."""
    from app.api.routes.crawl import CrawlRecordResponse

    fields = set(CrawlRecordResponse.model_fields.keys())
    required = {"status_code", "retry_count", "crawl_time", "proxy_used"}
    missing = required - fields
    assert not missing, (
        f"CrawlRecordResponse missing fields: {missing}. Current: {sorted(fields)}"
    )
