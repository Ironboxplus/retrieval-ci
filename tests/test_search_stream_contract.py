"""TDD: /api/search/stream NDJSON contract — real HTTP, no mocks.

Locks the streaming contract introduced 2026-05-16:
  - Endpoint POST /api/search/stream takes the same SearchRequest as /api/search
  - Response media type is application/x-ndjson
  - One JSON object per line, terminated by \\n
  - Stages arrive IN ORDER: embed -> retrieve -> rerank -> enrich -> complete
  - Final 'complete' event contains a SearchResponse-shaped payload under .response
"""
from __future__ import annotations

import json
import os

import pytest
import requests


os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

API = os.environ.get("RETRIVAL_API", "http://127.0.0.1:8000")
TIMEOUT = 60


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


def _stream_events(payload: dict) -> list[dict]:
    """Open the stream, consume to EOF, return parsed events in arrival order."""
    events: list[dict] = []
    with _session().post(
        f"{API}/api/search/stream",
        json=payload,
        timeout=TIMEOUT,
        stream=True,
    ) as r:
        assert r.status_code == 200, f"{r.status_code}: {r.text[:300]}"
        assert "application/x-ndjson" in r.headers.get("content-type", "")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            events.append(json.loads(line))
    return events


def test_stages_arrive_in_order():
    events = _stream_events({"query": "attention", "top_k": 5, "method": "hybrid"})
    stages = [e["stage"] for e in events]
    # Must contain all expected stages, in this relative order
    expected_order = ["embed", "retrieve", "rerank", "enrich", "complete"]
    seen_indices = []
    for s in expected_order:
        idx = stages.index(s) if s in stages else -1
        assert idx >= 0, f"missing stage {s!r} in stream events {stages}"
        seen_indices.append(idx)
    assert seen_indices == sorted(seen_indices), (
        f"stages out of order: {stages}"
    )


def test_embed_event_has_embed_ms():
    events = _stream_events({"query": "language model", "top_k": 3, "method": "hybrid"})
    embed = next(e for e in events if e["stage"] == "embed")
    assert isinstance(embed["embed_ms"], (int, float))
    assert embed["embed_ms"] >= 0


def test_retrieve_event_has_candidate_count():
    events = _stream_events({"query": "language model", "top_k": 5, "method": "hybrid"})
    retrieve = next(e for e in events if e["stage"] == "retrieve")
    assert "candidates" in retrieve
    assert isinstance(retrieve["candidates"], int)
    assert retrieve["candidates"] >= 0


def test_rerank_event_reports_before_after():
    events = _stream_events({"query": "transformer", "top_k": 5, "method": "hybrid"})
    rerank = next(e for e in events if e["stage"] == "rerank")
    assert "candidates_before" in rerank
    assert "candidates_after" in rerank
    # rerank must not multiply candidates
    assert rerank["candidates_after"] <= rerank["candidates_before"]


def test_complete_event_carries_search_response_shape():
    events = _stream_events({"query": "transformer", "top_k": 5, "method": "hybrid"})
    complete = next(e for e in events if e["stage"] == "complete")
    resp = complete["response"]
    assert resp is not None
    # Required SearchResponse fields
    for key in ("query", "method", "total_results", "results", "pipeline_metrics",
                "venue_distribution", "year_distribution"):
        assert key in resp, f"complete.response missing {key!r}"
    assert isinstance(resp["results"], list)
    assert resp["total_results"] == len(resp["results"])


def test_stream_response_matches_legacy_endpoint_payload():
    """The full SearchResponse delivered via 'complete' must be structurally identical
    to what /api/search returns for the same request.
    """
    payload = {"query": "neural", "top_k": 3, "method": "hybrid"}
    one = _session().post(f"{API}/api/search", json=payload, timeout=TIMEOUT).json()
    events = _stream_events(payload)
    streamed = next(e for e in events if e["stage"] == "complete")["response"]
    # Same shape — but DON'T compare results 1:1 (rerank is non-deterministic across runs).
    assert set(one.keys()) == set(streamed.keys())
    assert one["method"] == streamed["method"]
    assert one["query"] == streamed["query"]


@pytest.mark.parametrize("mode", ["hybrid", "vector", "sparse"])
def test_stream_supports_all_three_modes(mode: str):
    events = _stream_events({"query": "graph", "top_k": 3, "method": mode})
    stages = [e["stage"] for e in events]
    assert "complete" in stages, f"mode={mode!r} did not finish: {stages}"
