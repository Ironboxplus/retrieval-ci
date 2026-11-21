"""The `arxiv_delta` schedule kind: the daily arXiv update.

A trigger starts one ingest_pipeline task for DELTA_KEY (arxiv_stages: enumerate,
download, convert, preindex, embed, ingest, cleanup). The task is the work; this
module only decides when to start one and what the schedule card shows. A run
still going when the next trigger fires (the first one catches up a backlog of
days) makes the new trigger a `skipped` tick naming it, via the same
per-(venue, year) guard every pipeline start goes through.

This host never downloads: the downloading host (cfscan) does, and the download stage imports the
packages it leaves in the 115 volume (``source``, "volume", is the only one).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.services.pipeline.arxiv_stages import (
    DEFAULT_VOLUME,
    DOWNLOAD_MAX_ATTEMPTS,
    VOLUME_DEADLINE_HOURS,
)
from app.services.scheduler.kinds import (
    FireResult,
    JobKind,
    Metric,
    RetryPolicy,
    ScheduleSeed,
    register_kind,
)
from pipeline.paper_crawler.arxiv_corpus import DELTA_KEY, harvest_root, slice_sql
from pipeline.paper_crawler.arxiv_html import (
    CONTENT_VERSION_DATE_SQL,
    NO_HTML_RETRY_DAYS,
)

KIND = "arxiv_delta"
SEED_KEY = "arxiv_delta_daily"


class ArxivDeltaParams(BaseModel):
    """What one run may be told; the form of the schedule's parameters in the UI."""

    html_concurrency: int = Field(3, ge=1, le=6, description="Parallel HTML downloads")
    no_html_retry_days: int = Field(
        NO_HTML_RETRY_DAYS, ge=0, le=14,
        description="Retry papers without HTML while their newest content is this many days old")
    limit: int = Field(0, ge=0, description="Download at most this many papers; 0 = all queued")
    source: Literal["volume"] = Field(
        "volume",
        description="volume: import the packages the downloading host (cfscan) left in the 115 volume "
                    "(this host never downloads)")
    receive_deadline_hours: float = Field(
        VOLUME_DEADLINE_HOURS, gt=0, le=48,
        description="volume: wait this long for a new package, then fail")
    volume: str = Field(DEFAULT_VOLUME, description="volume: the Cfs volume on this host's OpenList node")


def fire_arxiv_delta(params: ArxivDeltaParams, origin: dict) -> FireResult:
    from app.services.pipeline.service import get_pipeline_runner

    result = get_pipeline_runner().start_arxiv_delta(options=params.model_dump(), origin=origin)
    task_id = result.get("task_id")
    if result.get("status") == "started":
        return FireResult("enqueued", task_id)
    if result.get("status") == "conflict":
        return FireResult("skipped", task_id, f"already running #{task_id}")
    raise RuntimeError(f"arxiv delta did not start: {result}")


def arxiv_delta_metrics() -> list[Metric]:
    """Gauges of the card. A missing harvest database leaves the harvest ones empty."""
    from core.db import get_db_session
    from core.db.models import Paper as DBPaper
    from sqlalchemy import func

    with get_db_session() as db:
        corpus = db.query(func.count(DBPaper.paper_id)).filter(DBPaper.paper_id.like("arxiv-26%")).scalar()
        withdrawn = db.query(func.count(DBPaper.paper_id)).filter(
            DBPaper.paper_id.like("arxiv-%"), DBPaper.withdrawn.is_(True)).scalar()
    watermark = queued = retryable = None
    db_file = harvest_root() / "arxiv_cs.sqlite3"
    if db_file.exists():
        where, params = slice_sql(DELTA_KEY)
        harvest = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            row = harvest.execute("SELECT value FROM harvest_state WHERE key='oai_from'").fetchone()
            watermark = row[0] if row else None
            queued = harvest.execute(
                f"SELECT COUNT(*) FROM papers WHERE {where} AND (status='pending' OR (status='error' AND attempts < ?))",
                [*params, DOWNLOAD_MAX_ATTEMPTS]).fetchone()[0]
            since = (datetime.now(timezone.utc) - timedelta(days=NO_HTML_RETRY_DAYS)).isoformat()
            retryable = harvest.execute(
                f"SELECT COUNT(*) FROM papers WHERE {where} AND status='no_html' AND {CONTENT_VERSION_DATE_SQL} >= ?",
                [*params, since]).fetchone()[0]
        finally:
            harvest.close()
    return [
        Metric("OAI 水位", watermark, ""),
        Metric("待下载", queued, "篇"),
        Metric("HTML 待重试", retryable, "篇"),
        Metric("语料", corpus, "篇"),
        Metric("撤稿", withdrawn, "篇"),
    ]


register_kind(JobKind(
    name=KIND,
    title="arXiv daily update",
    params_model=ArxivDeltaParams,
    fire=fire_arxiv_delta,
    # 05:00 missed by a stopped backend is still worth running within the day.
    default_misfire_grace_s=24 * 3600,
    retry=RetryPolicy(max_retries=2, base_delay_s=1800),
    status=arxiv_delta_metrics,
    defaults=(ScheduleSeed(
        key=SEED_KEY,
        title="arXiv daily update",
        trigger={"type": "daily", "time": "05:00", "timezone": "Asia/Shanghai"},
        params=ArxivDeltaParams().model_dump(),
    ),),
))
