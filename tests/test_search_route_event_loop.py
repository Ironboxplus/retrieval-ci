"""Regression: /api/search must not block the asyncio event loop.

Hang forensics 2026-07-25 (dlm_explore session): concurrent retrieval__search_papers
calls left one tool (masked-diffusion unmasking-order query) without a completed
result for a long wall-clock time while the other finished in ~7s. Direct
POST /api/search for the same query returns ~1s when the GPU is free, but the
async route ran search_pipeline.search + SQL enrichment INLINE — same class of
bug W60 fixed for /api/search/stream and code_kb search routes.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


async def _assert_route_yields_to_event_loop(coro):
    task = asyncio.create_task(coro)
    await asyncio.sleep(0.02)
    event_loop_remained_responsive = not task.done()
    await task
    return event_loop_remained_responsive


def test_main_search_route_offloads_blocking_pipeline(monkeypatch):
    import main
    from app.models.schemas import RetrievalMethod, SearchRequest
    from core.retrieval.pipeline import PipelineMetrics, SearchResponse as PipeResp

    class SlowPipeline:
        def search(self, req):
            time.sleep(0.1)
            return PipeResp(
                query=req.query, results=[], mode=req.mode, total=0,
                metrics=PipelineMetrics(),
            )

    monkeypatch.setattr(main, "search_pipeline", SlowPipeline())
    monkeypatch.setattr(main, "_enrich_results_sync", lambda results, request: results)
    monkeypatch.setattr(main, "_attach_paper_chunk_positions", lambda results, request: results)
    monkeypatch.setattr(main, "group_into_papers", lambda enriched, keep_best_only=True: [])
    monkeypatch.setattr(main, "_distributions", lambda chunks: ([], []))
    monkeypatch.setattr(main, "build_qdrant_filters", lambda request: {})

    req = SearchRequest(
        query="masked diffusion language model unmasking order",
        top_k=20,
        method=RetrievalMethod.HYBRID,
        dedup_by_paper=True,
    )
    assert asyncio.run(_assert_route_yields_to_event_loop(main.search(req)))


def test_main_search_source_uses_to_thread():
    source = open("main.py", encoding="utf-8").read()
    # The non-stream search path must offload pipeline + enrichment.
    assert "await asyncio.to_thread(search_pipeline.search" in source or (
        "await asyncio.to_thread(" in source
        and "search_pipeline.search" in source[source.find("@app.post(\"/api/search\""):source.find("@app.post(\"/api/search/stream\"")]
    )
    segment = source[source.find("@app.post(\"/api/search\""):source.find("@app.post(\"/api/search/stream\"")]
    assert "asyncio.to_thread" in segment
    assert "search_pipeline.search" in segment
