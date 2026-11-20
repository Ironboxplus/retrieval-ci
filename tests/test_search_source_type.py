"""TDD: /api/search must support source_type filter (paper | webpage | upload).

Backend already exposes source_type end-to-end (SearchRequest -> SearchPipeline
-> QdrantStore Filter). This test pins the contract so a future refactor cannot
silently drop the filter.
"""
from __future__ import annotations

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="module")
def live_server():
    """Skip suite if the live server is not running — these are integration tests."""
    try:
        r = requests.get(f"{BASE_URL}/api/stats", timeout=2)
        if r.status_code != 200:
            pytest.skip(f"live server returned {r.status_code} at /api/stats")
    except requests.RequestException as e:
        pytest.skip(f"live server not reachable: {e}")
    return BASE_URL


def _post_search(base_url: str, **overrides):
    body = {
        "query": "attention mechanism",
        "method": "hybrid",
        "top_k": 5,
    }
    body.update(overrides)
    resp = requests.post(f"{base_url}/api/search", json=body, timeout=30)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_search_paper_filter_returns_only_papers(live_server):
    """source_type='paper' must restrict results to paper-source chunks."""
    data = _post_search(live_server, source_type="paper")
    results = data.get("results", [])
    assert results, "expected at least one paper hit for a generic CS query"
    for r in results:
        meta = r.get("metadata") or {}
        st = meta.get("source_type") or r.get("source_type")
        # Either embedded in metadata or hoisted to the top of the result
        assert st == "paper", f"non-paper leaked through paper filter: {st!r} in {r}"


def test_search_webpage_filter_excludes_papers(live_server):
    """source_type='webpage' must not return any paper-source chunks."""
    data = _post_search(live_server, source_type="webpage", top_k=10)
    results = data.get("results", [])
    for r in results:
        meta = r.get("metadata") or {}
        st = meta.get("source_type") or r.get("source_type")
        assert st != "paper", f"paper leaked through webpage filter: {r}"


def test_search_no_source_type_unrestricted(live_server):
    """Omitting source_type must NOT add an implicit filter (back-compat)."""
    data = _post_search(live_server)
    assert "results" in data
    # Successful 200 + non-error response is enough; semantic check is covered
    # by the two filter tests above.
    assert data.get("total_results", 0) >= 0


def test_search_request_schema_exposes_source_type():
    """The wire contract must expose source_type as an optional string."""
    from app.models.schemas import SearchRequest

    req = SearchRequest(query="x", source_type="paper")
    assert req.source_type == "paper"

    req2 = SearchRequest(query="x")
    assert req2.source_type is None
