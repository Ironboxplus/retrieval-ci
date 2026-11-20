"""A search through the real ``SearchPipeline`` and the real ``VllmReranker`` reports which backend
scored it in its metrics (``rerank_backend``): online / local / mixed / skipped.

The embedder and the store are plain stand-ins that return fixed candidates (no vector database is
involved); the reranker is the real client talking to two real threaded HTTP servers.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.retrieval.pipeline import SearchPipeline, SearchRequest
from core.retrieval.reranker import VllmReranker
from tests.test_reranker_online_first import LOCAL_BASE, ONLINE_BASE, FakeRerankServer, _config

RAW = [
    {"id": f"c{i}", "content": f"passage {i}", "score": 0.9 - i / 100,
     "metadata": {"paper_id": f"P{i}", "doc_id": str(i)}}
    for i in range(4)
]


class _Embedding:
    dense = np.array([[0.1] * 8], dtype=np.float32)
    sparse = [{"42": 0.5}]


class _Embedder:
    def embed_texts(self, texts, **kwargs):
        return _Embedding()


class _Store:
    def hybrid_search(self, **kwargs):
        return list(RAW)

    sparse_search = hybrid_search

    def search(self, *args, **kwargs):
        return list(RAW)


@pytest.fixture()
def online():
    with FakeRerankServer(ONLINE_BASE) as server:
        yield server


@pytest.fixture()
def local():
    with FakeRerankServer(LOCAL_BASE, reported_tokens=None) as server:
        yield server


def _pipeline(local_server, online_server, *, batch_size=32) -> SearchPipeline:
    reranker = VllmReranker(base_url=local_server.url, online=_config(online_server),
                            batch_size=batch_size, in_flight=1)
    return SearchPipeline(embedder=_Embedder(), store=_Store(), reranker=reranker)


def _request() -> SearchRequest:
    return SearchRequest(query="a query", mode="hybrid", top_k=4)


def test_search_reports_the_online_backend(online, local):
    response = _pipeline(local, online).search(_request())

    assert response.metrics.rerank_backend == "online"
    assert [r.chunk_id for r in response.results] == ["c3", "c2", "c1", "c0"]   # ordered by the online scores
    assert local.count == 0


def test_search_reports_local_when_the_online_service_fails(online, local):
    online.mode = "500"

    response = _pipeline(local, online).search(_request())

    assert response.metrics.rerank_backend == "local"
    assert response.results[0].score == pytest.approx(LOCAL_BASE + 0.003)


def test_search_reports_mixed_when_only_some_batches_went_online(online, local):
    online.script.extend(["ok", "500"])

    response = _pipeline(local, online, batch_size=2).search(_request())

    assert response.metrics.rerank_backend == "mixed"


def test_search_without_a_reranker_reports_skipped():
    pipeline = SearchPipeline(embedder=_Embedder(), store=_Store(), reranker=None)

    assert pipeline.search(_request()).metrics.rerank_backend == "skipped"


def test_the_streaming_search_reports_it_in_the_rerank_event_and_the_final_metrics(online, local):
    events = list(_pipeline(local, online).search_stream(_request()))

    rerank = [e for e in events if e.get("stage") == "rerank"][-1]
    final = [e for e in events if e.get("stage") == "results"][-1]
    assert rerank["rerank_backend"] == "online"
    assert final["metrics"]["rerank_backend"] == "online"


def test_the_api_metrics_model_accepts_the_new_field_and_stays_optional():
    from app.models.schemas import PipelineMetricsResponse

    assert PipelineMetricsResponse().rerank_backend is None
    assert PipelineMetricsResponse(rerank_backend="mixed").rerank_backend == "mixed"
