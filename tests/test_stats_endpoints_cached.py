"""WebUI statistics endpoints are served from the shared stale-while-revalidate cache.

Real SQLite files with the application's tables; the cache clock is the only seam, so
"60 s later" and "10 minutes later" are exact. The scans these endpoints run are 2-19 s on
production tables (plan.md, 2026-11-09), so a repeat call must not re-run them.
"""
from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import dashboard
from app.models.schemas import Chunk as VectorChunk
from core.cache import DISTRIBUTION_TTL_SECONDS, TOTAL_COUNT_TTL_SECONDS, stats_cache
from core.db import database
from core.db.models import Chunk, Document, Paper, SourceType
from pipeline.paper_crawler.integration import api_routes, index_integration
from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig, PaperIndexer
from tests.qdrant_scratch import drop, scratch_names
from tests.sqlite_task_db import make_session_factory


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(stats_cache, "_clock", clock)
    return clock


def settle() -> None:
    """Wait for the background refresh threads the cache started."""
    for thread in threading.enumerate():
        if thread.name == "stats-cache-refresh":
            thread.join(10)


def add_paper(factory, paper_id: str, venue: str = "ICLR", year: int = 2025, chunk_levels=(1, 2)) -> None:
    with factory() as db:
        doc = Document(title=paper_id, source="s", source_type=SourceType.paper)
        db.add(doc)
        db.flush()
        db.add(Paper(paper_id=paper_id, doc_id=doc.id, title=paper_id, venue=venue, year=year))
        for index, level in enumerate(chunk_levels):
            db.add(Chunk(doc_id=doc.id, content="c", level=level, chunk_index=index))


def test_ttl_constants():
    assert DISTRIBUTION_TTL_SECONDS == 600.0
    assert TOTAL_COUNT_TTL_SECONDS == 60.0


def test_dashboard_distribution_is_cached_then_refreshed_in_the_background(tmp_path, monkeypatch, clock):
    factory = make_session_factory(tmp_path / "dist.db")
    add_paper(factory, "p1")
    add_paper(factory, "p2")
    monkeypatch.setattr(dashboard, "DB_PATH", str(tmp_path / "dist.db"))

    first = dashboard.distribution()
    assert first["venues"] == [{"venue": "ICLR", "papers": 2}]
    assert first["chunk_levels"] == {"1": 2, "2": 2}
    assert set(first) == {"chunk_levels", "venues", "years", "venue_year", "data_sources"}

    add_paper(factory, "p3", venue="ACL")
    assert dashboard.distribution() == first  # inside the TTL: the table scans are not repeated

    clock.advance(DISTRIBUTION_TTL_SECONDS + 1)
    assert dashboard.distribution() == first  # stale value answers at once ...
    settle()
    refreshed = dashboard.distribution()  # ... and the background refresh has replaced it
    assert {v["venue"] for v in refreshed["venues"]} == {"ICLR", "ACL"}


def test_dashboard_overview_counts_refresh_after_sixty_seconds(tmp_path, monkeypatch, clock):
    factory = make_session_factory(tmp_path / "overview.db")
    add_paper(factory, "p1")
    monkeypatch.setattr(dashboard, "DB_PATH", str(tmp_path / "overview.db"))
    monkeypatch.setattr(dashboard, "QDRANT_URL", "http://127.0.0.1:1")  # only the SQL counts are under test

    assert dashboard.overview()["sql"] == {"papers": 1, "documents": 1, "chunks": 2}
    add_paper(factory, "p2")
    assert dashboard.overview()["sql"] == {"papers": 1, "documents": 1, "chunks": 2}

    clock.advance(TOTAL_COUNT_TTL_SECONDS + 1)
    assert dashboard.overview()["sql"]["papers"] == 1
    settle()
    assert dashboard.overview()["sql"] == {"papers": 2, "documents": 2, "chunks": 4}


def test_dashboard_sqlite_row_counts_use_the_distribution_ttl(tmp_path, monkeypatch, clock):
    factory = make_session_factory(tmp_path / "sqlite.db")
    add_paper(factory, "p1")
    monkeypatch.setattr(dashboard, "DB_PATH", str(tmp_path / "sqlite.db"))

    def rows(body):
        return {t["name"]: t["rows"] for t in body["tables"]}

    first = dashboard.sqlite_detail()
    assert rows(first)["papers"] == 1
    assert first["db_size_bytes"] > 0 and first["indexes"] and first["pragma"]["page_size"]
    add_paper(factory, "p2")

    clock.advance(TOTAL_COUNT_TTL_SECONDS + 1)
    dashboard.sqlite_detail()
    settle()
    assert rows(dashboard.sqlite_detail())["papers"] == 1  # per-table counts are 10-minute data

    clock.advance(DISTRIBUTION_TTL_SECONDS)
    dashboard.sqlite_detail()
    settle()
    assert rows(dashboard.sqlite_detail())["papers"] == 2


def test_papers_tags_are_cached(tmp_path, monkeypatch, clock):
    factory = make_session_factory(tmp_path / "tags.db")
    add_paper(factory, "p1", venue="ICLR", year=2024)
    monkeypatch.setattr(database, "get_db_session", factory)
    app = FastAPI()
    app.include_router(api_routes.create_paper_router(prefix="/api/papers"))
    client = TestClient(app)

    first = client.get("/api/papers/tags").json()
    assert first["venue_counts"] == {"ICLR": 1} and first["total_papers"] == 1
    assert set(first) == {"total_papers", "venues", "years", "venue_tiers", "track_types",
                          "impact_levels", "venue_counts", "year_counts"}
    add_paper(factory, "p2", venue="ACL", year=2025)
    assert client.get("/api/papers/tags").json() == first

    clock.advance(DISTRIBUTION_TTL_SECONDS + 1)
    assert client.get("/api/papers/tags").json() == first
    settle()
    assert client.get("/api/papers/tags").json()["venue_counts"] == {"ICLR": 1, "ACL": 1}


@pytest.fixture()
def scratch_store():
    """A real Qdrant scratch collection holding three points."""
    from qdrant_client import QdrantClient

    from core.storage.qdrant_store import VECTOR_DIM, QdrantStore

    try:
        client = QdrantClient(url="http://127.0.0.1:6333", timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant: {exc}")
    name = scratch_names("statscache", roles=("papers",))["papers"]
    store = QdrantStore(collection_name=name)
    store.add_chunks(
        [VectorChunk(id=f"1-{n}", doc_id="1", content=f"text {n}", level=2, start_idx=0, end_idx=6,
                     metadata={"paper_id": "p1", "venue": "ICLR", "year": 2025, "venue_tier": "CCF-A"},
                     embedding=[0.5] * VECTOR_DIM)
         for n in range(3)],
        source_type="paper",
    )
    yield store
    drop(client, [name])


def count_statements(engine, needle: str):
    from sqlalchemy import event

    seen: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def record(conn, cursor, statement, parameters, context, executemany):
        if needle in statement.lower():
            seen.append(statement)

    return seen


def test_papers_stats_totals_refresh_at_sixty_seconds_and_levels_at_ten_minutes(
        tmp_path, monkeypatch, clock, scratch_store):
    factory = make_session_factory(tmp_path / "pstats.db")
    add_paper(factory, "p1")
    monkeypatch.setattr(index_integration, "get_db_session", factory)
    indexer = PaperIndexer(PaperIndexConfig(collection_name="papers"))
    indexer._chroma_store = scratch_store
    level_scans = count_statements(factory.engine, "group by")
    count_scans = count_statements(factory.engine, "count(")

    first = indexer.get_stats()
    settle()
    first = indexer.get_stats()  # the cold level histogram / vector count arrive in the background
    assert (first["total_papers"], first["total_documents"], first["total_chunks"]) == (1, 1, 2)
    assert first["chunk_levels"] == {1: 1, 2: 1}
    assert first["vector_index_size"] == 3
    assert first["bm25_index_size"] == 0

    scans_before = (len(level_scans), len(count_scans))
    add_paper(factory, "p2")
    assert indexer.get_stats() == first
    assert (len(level_scans), len(count_scans)) == scans_before  # no SQL at all inside the TTL

    clock.advance(TOTAL_COUNT_TTL_SECONDS + 1)
    stale = indexer.get_stats()
    assert stale == first
    settle()
    refreshed = indexer.get_stats()
    assert refreshed["total_papers"] == 2 and refreshed["total_chunks"] == 4
    assert refreshed["chunk_levels"] == {1: 1, 2: 1}  # the 10-minute histogram was not rescanned
    assert len(level_scans) == scans_before[0]

    clock.advance(DISTRIBUTION_TTL_SECONDS)
    indexer.get_stats()
    settle()
    assert indexer.get_stats()["chunk_levels"] == {1: 2, 2: 2}


def test_vector_count_and_levels_are_never_computed_on_the_request_path(
        tmp_path, monkeypatch, clock, scratch_store):
    """A Qdrant probe can block for a minute and the level GROUP BY for 19 s: a cold stats call
    answers at once with the empty defaults and the real values arrive in the background."""
    factory = make_session_factory(tmp_path / "cold.db")
    add_paper(factory, "p1")
    monkeypatch.setattr(index_integration, "get_db_session", factory)
    indexer = PaperIndexer(PaperIndexConfig(collection_name="papers"))
    indexer._chroma_store = scratch_store

    cold = indexer.get_stats()
    assert cold["vector_index_size"] == 0 and cold["chunk_levels"] == {}
    settle()
    warm = indexer.get_stats()
    assert warm["vector_index_size"] == 3 and warm["chunk_levels"] == {1: 1, 2: 1}


def test_api_stats_counts_are_cached_for_sixty_seconds(tmp_path, monkeypatch, clock):
    import main

    factory = make_session_factory(tmp_path / "apistats.db")
    add_paper(factory, "p1")
    monkeypatch.setattr(database, "get_db_session", factory)

    first = main._get_stats_sync()
    assert (first["total_papers"], first["total_documents"], first["total_chunks"]) == (1, 1, 2)
    assert {"vector_index_size", "coverage_ratio", "embedding_model", "embedding_dimension",
            "supports_sparse"} <= set(first)
    add_paper(factory, "p2")
    assert main._get_stats_sync() == first

    clock.advance(TOTAL_COUNT_TTL_SECONDS + 1)
    assert main._get_stats_sync() == first
    settle()
    assert main._get_stats_sync()["total_papers"] == 2


def test_index_coverage_uses_the_shared_cache(tmp_path, monkeypatch, clock):
    factory = make_session_factory(tmp_path / "cov.db")
    add_paper(factory, "p1", venue="ICLR", year=2024)
    monkeypatch.setattr(database, "get_db_session", factory)

    first = api_routes._index_coverage_cached()
    assert first["venues"][0]["venue"] == "ICLR"
    add_paper(factory, "p2", venue="ACL", year=2024)
    assert api_routes._index_coverage_cached() is first

    clock.advance(DISTRIBUTION_TTL_SECONDS + 1)
    assert api_routes._index_coverage_cached() is first
    settle()
    assert {v["venue"] for v in api_routes._index_coverage_cached()["venues"]} == {"ICLR", "ACL"}
