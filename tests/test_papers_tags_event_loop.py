"""`/api/papers/tags` must not stall the event loop while its SQL runs.

Production 2026-11-09: the route was an `async def` running its GROUP BY queries inline;
during its 11 s a 30 ms `/api/dashboard/system` also waited 11 s. Here a real SQLite
database behind a real FastAPI app gets a deliberately slow synchronous session open, and a
cheap concurrent request is timed.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager

import httpx
from fastapi import FastAPI

from core.db import database
from core.db.models import Document, Paper, SourceType
from pipeline.paper_crawler.integration.api_routes import create_paper_router
from tests.sqlite_task_db import make_session_factory

SLOW_SECONDS = 0.8


def _seed(factory) -> None:
    with factory() as db:
        for number, (venue, year) in enumerate([("ICLR", 2024), ("ICLR", 2025), ("ACL", 2025)]):
            doc = Document(title=f"t{number}", source="s", source_type=SourceType.paper)
            db.add(doc)
            db.flush()
            db.add(Paper(paper_id=f"p{number}", doc_id=doc.id, title=f"t{number}", venue=venue, year=year))


def test_slow_tags_query_does_not_delay_a_concurrent_cheap_request(tmp_path, monkeypatch):
    factory = make_session_factory(tmp_path / "tags.db")
    _seed(factory)

    @contextmanager
    def slow_session():
        time.sleep(SLOW_SECONDS)  # a synchronous DB call that takes a long time
        with factory() as db:
            yield db

    monkeypatch.setattr(database, "get_db_session", slow_session)

    app = FastAPI()
    app.include_router(create_paper_router(prefix="/api/papers"))

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            tags_started = time.monotonic()
            tags_task = asyncio.create_task(client.get("/api/papers/tags"))
            # Timed from before the ping is scheduled: a loop blocked inside /tags cannot even
            # start the ping task, so the wait shows up in the measurement.
            ping_started = time.monotonic()
            ping = await asyncio.create_task(client.get("/ping"))
            ping_seconds = time.monotonic() - ping_started
            tags = await tags_task
            return ping, ping_seconds, tags, time.monotonic() - tags_started

    ping, ping_seconds, tags, tags_seconds = asyncio.run(scenario())

    assert tags.status_code == 200
    assert tags.json()["venue_counts"] == {"ICLR": 2, "ACL": 1}
    assert tags_seconds >= SLOW_SECONDS  # the slow call really ran during the ping
    assert ping.status_code == 200
    assert ping_seconds < 0.3, f"cheap request waited {ping_seconds:.2f}s behind /tags"
