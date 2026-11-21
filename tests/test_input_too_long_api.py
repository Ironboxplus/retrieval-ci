"""A query longer than the embedding model's limit is a 422 that states the token count, not a 500.

The over-long input is rejected by the real ``encode`` guard with the real BGE-M3 tokenizer,
before any model would be touched; the pipelines below only stand in for the retrieval stack
behind the route.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.indexing.embedding_profile import get_profile, load_tokenizer
from tests.embedding_model_support import bge_m3_snapshot, text_of_tokens


@pytest.fixture(scope="module", autouse=True)
def snapshot():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    yield path
    mp.undo()


@pytest.fixture(scope="module")
def over_long_query(snapshot):
    return text_of_tokens(load_tokenizer(get_profile()), get_profile().max_seq_len + 5)


def _embed_like_the_backend(text: str):
    from core.indexing.embedders.bge_m3_core import encode

    profile = get_profile()
    return encode(SimpleNamespace(tokenizer=load_tokenizer(profile)), [text], profile)


def _assert_422_with_count(response):
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["input_tokens"] == 8197 and body["token_limit"] == 8192
    assert "8197" in body["detail"] and "8192" in body["detail"]


def test_api_search_returns_422_with_the_token_count(monkeypatch, over_long_query):
    import main

    class Pipeline:
        def search(self, request):
            return _embed_like_the_backend(request.query)

    monkeypatch.setattr(main, "search_pipeline", Pipeline())
    monkeypatch.setattr(main, "build_qdrant_filters", lambda request: {})
    response = TestClient(main.app).post("/api/search", json={"query": over_long_query, "top_k": 5})
    _assert_422_with_count(response)


def test_match_doc_returns_422_not_400(monkeypatch, over_long_query):
    import main
    import app.services.match_doc.service as match_service

    class Service:
        def match(self, request):
            return _embed_like_the_backend(over_long_query)

    monkeypatch.setattr(main, "search_pipeline", object())
    monkeypatch.setattr(match_service, "get_match_doc_service", lambda pipeline: Service())
    response = TestClient(main.app).post(
        "/api/match-doc", json={"input_type": "text", "text": "x"})
    _assert_422_with_count(response)


def test_journal_search_returns_422(monkeypatch, over_long_query):
    import main
    import app.api.routes.journal as journal_routes

    class Service:
        def search(self, body):
            return _embed_like_the_backend(body.query)

    monkeypatch.setattr(journal_routes, "get_journal_service", lambda: Service())
    response = TestClient(main.app).post(
        "/api/journal/search", json={"query": over_long_query, "project_slug": "p"},
        headers={"X-Retrival-User-Id": "user-7"})
    _assert_422_with_count(response)
