"""Regression: /api/match-doc must not block the asyncio event loop.

2026-09-06 现场：一次 MCP `match_paper` 卡住时，并发的 `POST /api/search` 也 120 秒
不返回，后端对所有请求无响应。根因与 2026-07-25 的 `/api/search` hang 同类——async
路由里直接调用同步的 `service.match()`，没有 offload。

match_doc 的放大比 search 大得多：`MatchDocService.match()` 按 `max_query_chunks`
（默认 40）逐块**串行**调用 SearchPipeline，一次请求就是几十次检索。这些全部跑在
event loop 上，所以它不只是自己慢，而是把整个 backend 挂起几十秒到几分钟。

AGENTS.md 的验证门写明「async 路由中的同步 I/O 必须 offload」；`/api/search` 已按
该门修过，`/api/match-doc` 当时漏掉了。
"""
from __future__ import annotations

import asyncio
import time

import pytest


async def _assert_route_yields_to_event_loop(coro):
    task = asyncio.create_task(coro)
    await asyncio.sleep(0.02)
    event_loop_remained_responsive = not task.done()
    await task
    return event_loop_remained_responsive


def test_match_doc_route_offloads_blocking_service(monkeypatch):
    import main
    import app.services.match_doc.service as match_service
    from app.services.match_doc.schemas import MatchDocRequest, MatchDocResponse

    class SlowMatchService:
        """Stands in for the real 40-chunk serial retrieval loop."""

        def match(self, req: MatchDocRequest) -> MatchDocResponse:
            time.sleep(0.1)
            return MatchDocResponse(
                input_type=req.input_type, query_chunks=[], papers=[],
                metrics={"query_chunks": 0, "total_ms": 100.0},
            )

    monkeypatch.setattr(main, "search_pipeline", object())
    monkeypatch.setattr(
        match_service, "get_match_doc_service", lambda pipeline=None: SlowMatchService()
    )

    req = MatchDocRequest(
        input_type="text",
        text="We propose a retrieval-augmented method that reranks passages.",
        top_k=20,
    )
    assert asyncio.run(_assert_route_yields_to_event_loop(main.match_doc(req)))


def test_match_doc_source_uses_to_thread():
    source = open("main.py", encoding="utf-8").read()
    start = source.find('@app.post("/api/match-doc"')
    assert start != -1, "match-doc route not found in main.py"
    end = source.find("@app.post(", start + 1)
    segment = source[start:end if end != -1 else len(source)]
    assert "asyncio.to_thread" in segment, (
        "match_doc must offload the synchronous service.match() off the event loop"
    )
    assert "service.match" in segment
