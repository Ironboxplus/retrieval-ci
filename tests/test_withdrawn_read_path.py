"""Withdrawn read path outside Qdrant: search routes, hit enrichment, browse, detail.

Real SQLite file with the application's tables (tests/sqlite_task_db.py), the real
FastAPI paper router, and the real ``main`` enrichment function. SQL is the authority
for ``withdrawn`` / ``source_version``: the Qdrant payload flag can lag a withdrawal or a
reinstatement, so the hit metadata is overridden from the ``papers`` row.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.db.models import DataSource, Document, DocumentStatus, Paper, SourceType
from tests.sqlite_task_db import make_session_factory

import main  # noqa: E402
from app.models.schemas import RetrievalMethod, SearchRequest, SearchResult  # noqa: E402
from pipeline.paper_crawler.integration.api_routes import create_paper_router  # noqa: E402

CONFERENCE = "iclr-2025-1"
LIVE = "arxiv-2611.00001"
WITHDRAWN = "arxiv-2611.00002"


def _seed(db, paper_id, *, venue, tier, title, withdrawn=False, version=None, withdrawn_at=None):
    doc = Document(title=title, source=f"https://example.com/{paper_id}",
                   source_type=SourceType.paper, data_source=DataSource.system,
                   status=DocumentStatus.indexed)
    db.add(doc)
    db.flush()
    db.add(Paper(paper_id=paper_id, doc_id=doc.id, title=title, abstract="abstract text",
                 authors_json=["A. Author"], year=2026, venue=venue, venue_tier=tier,
                 withdrawn=withdrawn, withdrawn_at=withdrawn_at, source_version=version))
    db.flush()
    return doc.id


@pytest.fixture
def papers(tmp_path, monkeypatch):
    factory = make_session_factory(tmp_path / "retrieval.db")
    monkeypatch.setattr("core.db.database.get_db_session", factory)
    ids = {}
    with factory() as db:
        ids[CONFERENCE] = _seed(db, CONFERENCE, venue="ICLR", tier="CCF-A", title="Conference paper")
        ids[LIVE] = _seed(db, LIVE, venue="arXiv", tier="preprint", title="Live preprint",
                          version="v2")
        ids[WITHDRAWN] = _seed(db, WITHDRAWN, venue="arXiv", tier="preprint",
                               title="Withdrawn preprint", withdrawn=True, version="v1",
                               withdrawn_at="2026-11-01T03:00:00")
    return ids


@pytest.fixture
def client(papers):
    app = FastAPI()
    app.include_router(create_paper_router(prefix="/api/papers"))
    return TestClient(app)


# ---- hit enrichment ----

def _hit(doc_id: int, **metadata) -> SearchResult:
    return SearchResult(doc_id=str(doc_id), chunk_id=f"c{doc_id}", content="text", score=0.5,
                        doc_title="t", source="s", metadata=metadata)


def test_hits_carry_withdrawn_and_source_version_from_sql(papers):
    hits = [_hit(papers[LIVE], paper_id=LIVE), _hit(papers[WITHDRAWN], paper_id=WITHDRAWN,
                                                    withdrawn=True),
            _hit(papers[CONFERENCE], paper_id=CONFERENCE)]
    live, withdrawn, conference = main._enrich_results_sync(hits, SearchRequest(query="q"))
    assert (live.metadata["withdrawn"], live.metadata["source_version"]) == (False, "v2")
    assert (withdrawn.metadata["withdrawn"], withdrawn.metadata["source_version"]) == (True, "v1")
    # A conference paper has no source version: the key is absent, not None.
    assert conference.metadata["withdrawn"] is False
    assert "source_version" not in conference.metadata


def test_sql_overrides_a_stale_qdrant_flag(papers):
    # Reinstated in SQL but the point still says withdrawn.
    (hit,) = main._enrich_results_sync(
        [_hit(papers[LIVE], paper_id=LIVE, withdrawn=True)], SearchRequest(query="q"))
    assert hit.metadata["withdrawn"] is False


def test_non_paper_documents_are_untouched(papers, tmp_path):
    from core.db.database import get_db_session

    with get_db_session() as db:
        page = Document(title="page", source="web", source_type=SourceType.webpage,
                        data_source=DataSource.user, status=DocumentStatus.indexed)
        db.add(page)
        db.flush()
        page_id = page.id
    (hit,) = main._enrich_results_sync([_hit(page_id)], SearchRequest(query="q"))
    assert "withdrawn" not in hit.metadata and "source_version" not in hit.metadata


# ---- browse ----

def _browse(client, **params):
    resp = client.get("/api/papers/browse", params={"page_size": 50, **params})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_browse_leaves_withdrawn_out_by_default(client):
    body = _browse(client)
    assert {r["paper_id"] for r in body["results"]} == {CONFERENCE, LIVE}
    assert body["total"] == 2


def test_browse_include_withdrawn_lists_them_with_the_flag(client):
    body = _browse(client, include_withdrawn="true")
    assert body["total"] == 3
    by_id = {r["paper_id"]: r for r in body["results"]}
    assert by_id[WITHDRAWN]["withdrawn"] is True
    assert by_id[WITHDRAWN]["source_version"] == "v1"
    assert by_id[LIVE]["withdrawn"] is False and by_id[LIVE]["source_version"] == "v2"
    assert by_id[CONFERENCE]["withdrawn"] is False and by_id[CONFERENCE]["source_version"] is None


def test_browse_filters_still_compose_with_the_withdrawn_default(client):
    body = _browse(client, venue="arXiv")
    assert [r["paper_id"] for r in body["results"]] == [LIVE]
    body = _browse(client, venue="arXiv", include_withdrawn="true")
    assert {r["paper_id"] for r in body["results"]} == {LIVE, WITHDRAWN}


# ---- detail ----

def test_detail_reports_withdrawn_and_version_even_without_the_switch(client):
    meta = client.get(f"/api/papers/{WITHDRAWN}").json()["metadata"]
    assert meta["withdrawn"] is True and meta["source_version"] == "v1"
    assert meta["withdrawn_at"] == "2026-11-01T03:00:00"
    live = client.get(f"/api/papers/{LIVE}").json()["metadata"]
    assert live["withdrawn"] is False and live["source_version"] == "v2"
    assert live["withdrawn_at"] is None


# ---- search routes forward the flag ----

class _RecordingPipeline:
    """Stands in for the Qdrant-backed pipeline only to record the request the route
    built; the filters it would produce are covered against a real Qdrant in
    tests/test_withdrawn_filter_real_qdrant.py."""

    def __init__(self):
        self.requests = []

    def search(self, req):
        from core.retrieval.pipeline import PipelineMetrics, SearchResponse

        self.requests.append(req)
        return SearchResponse(query=req.query, results=[], mode=req.mode, total=0,
                              metrics=PipelineMetrics())

    def search_stream(self, req):
        self.requests.append(req)
        yield {"stage": "results", "results": [], "metrics": {}}


@pytest.mark.parametrize("flag", [False, True])
def test_search_and_stream_routes_forward_include_withdrawn(monkeypatch, flag):
    pipeline = _RecordingPipeline()
    monkeypatch.setattr(main, "search_pipeline", pipeline)
    request = SearchRequest(query="q", method=RetrievalMethod.HYBRID, include_withdrawn=flag)

    async def run():
        await main.search(request)
        response = await main.search_stream(request)
        async for _ in response.body_iterator:
            pass
        await main.search_arxiv(request)

    asyncio.run(run())
    assert [r.include_withdrawn for r in pipeline.requests] == [flag, flag, flag]
    assert pipeline.requests[2].filters["venue_tier"] == "preprint"
