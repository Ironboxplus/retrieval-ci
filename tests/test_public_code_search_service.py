"""Public code search uses public source scopes, never private Code KB ownership."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.public_code.search_service import (
    PublicCodeSearchRequest,
    PublicCodeSearchService,
)
from app.services.public_code.repository import PublicCodeSearchScope
from app.services.public_code.source_catalog import PublicCodeSource, PublicCodeSourceCatalog


def catalog() -> PublicCodeSourceCatalog:
    return PublicCodeSourceCatalog(
        catalog_id="fixture",
        sources=(
            PublicCodeSource("ffmpeg", "multimedia", "multimedia/ffmpeg", "https://example.invalid/ffmpeg.git"),
            PublicCodeSource("pytorch", "machine-learning", "machine-learning/pytorch", "https://example.invalid/pytorch.git"),
        ),
    )


@dataclass
class FakeRepository:
    revisions: dict[str, str]
    received_source_ids: tuple[str, ...] = ()
    received_domains: tuple[str, ...] = ()

    def resolve_search_scope(self, *, source_ids, domains):
        selected = catalog().select(source_ids=source_ids, domains=domains)
        self.received_source_ids = tuple(source.source_id for source in selected)
        self.received_domains = tuple(domains)
        return PublicCodeSearchScope(
            source_ids=self.received_source_ids,
            active_revisions={
                source_id: self.revisions[source_id]
                for source_id in self.received_source_ids
                if source_id in self.revisions
            },
        )


class FakeVectorStore:
    supports_sparse = True

    def __init__(self):
        self.arguments = None

    def search(self, **kwargs):
        self.arguments = kwargs
        return [
            {
                "score": 0.8,
                "source_id": "ffmpeg",
                "revision": "ffmpeg-commit",
                "domain": "multimedia",
                "path": "libavcodec/encode.c",
                "language": "c",
                "start_line": 20,
                "end_line": 34,
                "label": "encode_frame",
                "content": "# path: libavcodec/encode.c\nint encode_frame(void);",
                "code": "int encode_frame(void);",
            }
        ]


class CapturingVectorStore(FakeVectorStore):
    """Return a deterministic retrieval order and retain the service search contract."""

    def __init__(self, hits, *, supports_sparse=True):
        super().__init__()
        self._hits = list(hits)
        self.supports_sparse = supports_sparse

    def search(self, **kwargs):
        self.arguments = kwargs
        return self._hits[:kwargs["top_k"]]


def _raw_hit(
    *,
    score: float,
    path: str,
    start_line: int,
    code: str,
    source_id: str = "ffmpeg",
) -> dict[str, object]:
    return {
        "score": score,
        "source_id": source_id,
        "revision": "ffmpeg-commit",
        "domain": "multimedia",
        "path": path,
        "language": "python",
        "start_line": start_line,
        "end_line": start_line + 3,
        "label": f"chunk_{start_line}",
        "content": f"# path: {path}\n{code}",
        "code": code,
    }


def test_exact_source_search_only_queries_that_source_active_revision():
    repository = FakeRepository({"ffmpeg": "ffmpeg-commit", "pytorch": "pytorch-commit"})
    vector_store = FakeVectorStore()
    service = PublicCodeSearchService(
        repository=repository,
        vector_store=vector_store,
        embed_fn=lambda texts: [[float(len(texts[0]))]],
        sparse_embed_fn=lambda _texts: [{"1": 1.0}],
    )

    response = service.search(
        PublicCodeSearchRequest(query="H.264 frame encoding", source_ids=("ffmpeg",), mode="hybrid")
    )

    assert repository.received_source_ids == ("ffmpeg",)
    assert vector_store.arguments["active_revisions"] == {"ffmpeg": "ffmpeg-commit"}
    assert vector_store.arguments["source_ids"] == ("ffmpeg",)
    assert response.hits[0].source_id == "ffmpeg"


def test_domain_search_flattens_all_selected_repositories_into_one_qdrant_call():
    repository = FakeRepository({"ffmpeg": "ffmpeg-commit", "pytorch": "pytorch-commit"})
    vector_store = FakeVectorStore()
    service = PublicCodeSearchService(
        repository=repository,
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
    )

    service.search(PublicCodeSearchRequest(query="kernel implementation", domains=("machine-learning", "multimedia")))

    assert repository.received_source_ids == ("ffmpeg", "pytorch")
    assert vector_store.arguments["source_ids"] == ("ffmpeg", "pytorch")
    assert vector_store.arguments["domains"] == ()


def test_explicit_source_and_domain_preserve_catalog_union_in_vector_scope():
    repository = FakeRepository({"ffmpeg": "ffmpeg-commit", "pytorch": "pytorch-commit"})
    vector_store = FakeVectorStore()
    service = PublicCodeSearchService(
        repository=repository,
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
    )

    service.search(PublicCodeSearchRequest(
        query="kernel implementation",
        source_ids=("ffmpeg",),
        domains=("machine-learning",),
    ))

    assert repository.received_source_ids == ("ffmpeg", "pytorch")
    assert vector_store.arguments["source_ids"] == ("ffmpeg", "pytorch")
    assert vector_store.arguments["domains"] == ()


def test_unknown_source_is_rejected_before_embedding_or_qdrant_work():
    service = PublicCodeSearchService(
        repository=FakeRepository({}),
        vector_store=FakeVectorStore(),
        embed_fn=lambda _texts: pytest.fail("embedding must not run"),
    )

    with pytest.raises(ValueError, match="Unknown public-code source IDs: missing"):
        service.search(PublicCodeSearchRequest(query="x", source_ids=("missing",)))


def test_reranking_reorders_public_code_hits_without_loading_another_model_instance():
    class Reranker:
        def score_pairs(self, _query, _contents):
            return [0.99]

    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=FakeVectorStore(),
        embed_fn=lambda _texts: [[1.0]],
        reranker=Reranker(),
    )

    response = service.search(
        PublicCodeSearchRequest(query="encode", source_ids=("ffmpeg",), rerank=True)
    )

    assert response.hits[0].score == 0.99
    assert response.hits[0].retrieval_score == 0.8
    assert response.hits[0].reranker_score == 0.99


def test_rerank_fetches_an_independent_candidate_pool_and_returns_only_requested_hits():
    class PreferenceReranker:
        def __init__(self):
            self.contents = []

        def score_pairs(self, _query, contents):
            self.contents = list(contents)
            return [0.99 if "TARGET Depends" in content else 0.1 for content in contents]

    lifecycle_first = _raw_hit(
        score=0.99,
        path="api/lifecycle.py",
        start_line=10,
        code="async def lifespan(app): pass",
    )
    raw_hits = [
        lifecycle_first,
        dict(lifecycle_first),  # exact Qdrant duplicate must not consume reranker budget
        _raw_hit(
            score=0.98,
            path="api/lifecycle.py",
            start_line=20,
            code="async def startup(app): pass",
        ),
        _raw_hit(
            score=0.97,
            path="api/lifecycle.py",
            start_line=30,
            code="async def shutdown(app): pass",
        ),
    ]
    raw_hits.extend(
        _raw_hit(
            score=0.9 - index / 100,
            path=f"examples/example_{index}.py",
            start_line=index * 10,
            code=(
                "TARGET Depends dependency injection"
                if index == 19
                else f"unrelated example {index}"
            ),
        )
        for index in range(4, 20)
    )
    reranker = PreferenceReranker()
    vector_store = CapturingVectorStore(raw_hits)
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
        sparse_embed_fn=lambda _texts: [{"1": 1.0}],
        reranker=reranker,
    )

    response = service.search(PublicCodeSearchRequest(
        query="FastAPI dependency injection",
        source_ids=("ffmpeg",),
        top_k=5,
        rerank=True,
    ))

    assert vector_store.arguments["top_k"] == 20
    assert len(reranker.contents) == 19
    assert reranker.contents.count(lifecycle_first["content"]) == 1
    assert sum("# path: api/lifecycle.py" in content for content in reranker.contents) == 3
    assert any("TARGET Depends" in content for content in reranker.contents)
    assert len(response.hits) == 5
    assert response.hits[0].code == "TARGET Depends dependency injection"
    assert response.hits[0].retrieval_score == pytest.approx(0.71)
    assert response.hits[0].reranker_score == pytest.approx(0.99)
    assert response.hits[0].score == pytest.approx(0.99)
    assert response.requested_mode == "hybrid"
    assert response.effective_mode == "hybrid"
    assert response.notice is None
    assert response.rerank_applied is True
    assert response.candidate_count == 20
    assert response.rerank_candidate_count == 19


def test_rerank_scores_every_single_file_overflow_candidate_in_a_narrow_scope():
    class RecordingReranker:
        def __init__(self):
            self.contents = []

        def score_pairs(self, _query, contents):
            self.contents = list(contents)
            return [float(index) for index, _ in enumerate(contents)]

    raw_hits = [
        _raw_hit(
            score=1.0 - index / 100,
            path="api/lifecycle.py",
            start_line=(index + 1) * 10,
            code=f"single-file chunk {index}",
        )
        for index in range(20)
    ]
    reranker = RecordingReranker()
    vector_store = CapturingVectorStore(raw_hits)
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
        reranker=reranker,
    )

    response = service.search(PublicCodeSearchRequest(
        query="lifecycle", source_ids=("ffmpeg",), top_k=5, rerank=True
    ))

    assert vector_store.arguments["top_k"] == 20
    assert reranker.contents == [hit["content"] for hit in raw_hits]
    assert response.hits[0].code == "single-file chunk 19"
    assert response.candidate_count == 20
    assert response.rerank_candidate_count == 20


def test_rerank_candidate_selection_keeps_two_chunks_per_file_before_overflow():
    class RecordingReranker:
        def __init__(self):
            self.contents = []

        def score_pairs(self, _query, contents):
            self.contents = list(contents)
            return [0.1 for _ in contents]

    first = _raw_hit(score=0.9, path="api/lifecycle.py", start_line=10, code="first")
    second = _raw_hit(score=0.8, path="api/lifecycle.py", start_line=20, code="second")
    third = _raw_hit(score=0.7, path="api/lifecycle.py", start_line=30, code="third")
    other = _raw_hit(score=0.6, path="api/dependencies.py", start_line=10, code="other")
    reranker = RecordingReranker()
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=CapturingVectorStore([first, dict(first), second, third, other]),
        embed_fn=lambda _texts: [[1.0]],
        reranker=reranker,
    )

    service.search(PublicCodeSearchRequest(
        query="dependency", source_ids=("ffmpeg",), top_k=1, rerank=True
    ))

    assert reranker.contents == [
        first["content"],
        second["content"],
        other["content"],
        third["content"],
    ]


def test_non_reranked_public_code_hit_keeps_retrieval_score_and_does_not_expand_fetch():
    vector_store = FakeVectorStore()
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
    )

    response = service.search(PublicCodeSearchRequest(
        query="encode", source_ids=("ffmpeg",), top_k=5, rerank=False
    ))

    assert vector_store.arguments["top_k"] == 5
    assert response.hits[0].score == pytest.approx(0.8)
    assert response.hits[0].retrieval_score == pytest.approx(0.8)
    assert response.hits[0].reranker_score is None


def test_rerank_failure_falls_back_to_the_original_retrieval_top_k_in_original_order():
    class FailingReranker:
        def __init__(self):
            self.contents = []

        def score_pairs(self, _query, contents):
            self.contents = list(contents)
            raise RuntimeError("cross-encoder unavailable")

    first = _raw_hit(score=0.9, path="api/lifecycle.py", start_line=10, code="first")
    second = _raw_hit(score=0.8, path="api/lifecycle.py", start_line=20, code="second")
    third = _raw_hit(score=0.7, path="api/lifecycle.py", start_line=30, code="third")
    other = _raw_hit(score=0.6, path="api/dependencies.py", start_line=10, code="other")
    raw_hits = [first, dict(first), second, third, other]
    raw_hits.extend(
        _raw_hit(
            score=0.5 - index / 100,
            path=f"api/overflow_{index}.py",
            start_line=index * 10,
            code=f"overflow {index}",
        )
        for index in range(5, 20)
    )
    reranker = FailingReranker()
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=CapturingVectorStore(raw_hits),
        embed_fn=lambda _texts: [[1.0]],
        sparse_embed_fn=lambda _texts: [{"1": 1.0}],
        reranker=reranker,
    )

    response = service.search(PublicCodeSearchRequest(
        query="dependency", source_ids=("ffmpeg",), top_k=3, rerank=True
    ))

    assert [hit.code for hit in response.hits] == ["first", "first", "second"]
    assert [hit.score for hit in response.hits] == [0.9, 0.9, 0.8]
    assert all(hit.reranker_score is None for hit in response.hits)
    assert len(reranker.contents) == 19
    assert response.rerank_applied is False
    assert response.candidate_count == 20
    assert response.rerank_candidate_count == 19
    assert response.requested_mode == "hybrid"
    assert response.effective_mode == "hybrid"
    assert response.notice is None


def test_rerank_exception_fails_open_without_logging_exception_payload(caplog):
    class PayloadLeakingReranker:
        def score_pairs(self, _query, _contents):
            raise RuntimeError("PRIVATE_EXCEPTION_PAYLOAD")

    first = _raw_hit(score=0.9, path="api/first.py", start_line=10, code="first")
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=CapturingVectorStore([first]),
        embed_fn=lambda _texts: [[1.0]],
        reranker=PayloadLeakingReranker(),
    )

    response = service.search(PublicCodeSearchRequest(
        query="PRIVATE_QUERY_MARKER", source_ids=("ffmpeg",), top_k=1, rerank=True
    ))

    assert [hit.code for hit in response.hits] == ["first"]
    assert "PRIVATE_QUERY_MARKER" not in caplog.text
    assert "PRIVATE_EXCEPTION_PAYLOAD" not in caplog.text


def test_rerank_score_count_mismatch_falls_back_to_the_original_retrieval_top_k():
    class MismatchedReranker:
        def score_pairs(self, _query, _contents):
            return [0.99]

    first = _raw_hit(score=0.9, path="api/lifecycle.py", start_line=10, code="first")
    second = _raw_hit(score=0.8, path="api/lifecycle.py", start_line=20, code="second")
    third = _raw_hit(score=0.7, path="api/lifecycle.py", start_line=30, code="third")
    other = _raw_hit(score=0.6, path="api/dependencies.py", start_line=10, code="other")
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=CapturingVectorStore([first, dict(first), second, third, other]),
        embed_fn=lambda _texts: [[1.0]],
        reranker=MismatchedReranker(),
    )

    response = service.search(PublicCodeSearchRequest(
        query="dependency", source_ids=("ffmpeg",), top_k=3, rerank=True
    ))

    assert [hit.code for hit in response.hits] == ["first", "first", "second"]
    assert all(hit.reranker_score is None for hit in response.hits)


def test_equal_reranker_scores_keep_higher_retrieval_score_ahead_of_location_order():
    class TiedReranker:
        def score_pairs(self, _query, contents):
            return [0.5 for _ in contents]

    low_retrieval = _raw_hit(score=0.1, path="a.py", start_line=1, code="low retrieval")
    high_retrieval = _raw_hit(score=0.9, path="z.py", start_line=1, code="high retrieval")
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=CapturingVectorStore([low_retrieval, high_retrieval]),
        embed_fn=lambda _texts: [[1.0]],
        reranker=TiedReranker(),
    )

    response = service.search(PublicCodeSearchRequest(
        query="tie", source_ids=("ffmpeg",), top_k=2, rerank=True
    ))

    assert [hit.code for hit in response.hits] == ["high retrieval", "low retrieval"]


@pytest.mark.parametrize(
    "invalid_scores",
    [
        None,
        0.9,
        object(),
        "12",
        [float("nan"), 0.2],
        [float("inf"), 0.2],
        [float("-inf"), 0.2],
    ],
    ids=[
        "none",
        "scalar",
        "non_iterable",
        "equal_length_string",
        "nan",
        "positive_infinity",
        "negative_infinity",
    ],
)
def test_invalid_reranker_scores_fail_open_without_logging_query_content(invalid_scores, caplog):
    class InvalidScoreReranker:
        def score_pairs(self, _query, _contents):
            return invalid_scores

    first = _raw_hit(score=0.9, path="api/first.py", start_line=10, code="first")
    second = _raw_hit(score=0.8, path="api/second.py", start_line=10, code="second")
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=CapturingVectorStore([first, second]),
        embed_fn=lambda _texts: [[1.0]],
        reranker=InvalidScoreReranker(),
    )

    response = service.search(PublicCodeSearchRequest(
        query="PRIVATE_QUERY_MARKER", source_ids=("ffmpeg",), top_k=2, rerank=True
    ))

    assert [hit.code for hit in response.hits] == ["first", "second"]
    assert [hit.score for hit in response.hits] == [0.9, 0.8]
    assert all(hit.reranker_score is None for hit in response.hits)
    assert all("PRIVATE_QUERY_MARKER" not in record.getMessage() for record in caplog.records)


def test_rerank_active_scope_with_no_vector_hits_returns_empty_without_loading_reranker():
    class UnexpectedReranker:
        calls = 0

        def score_pairs(self, _query, _contents):
            self.calls += 1
            pytest.fail("reranker must not run when retrieval returns no hits")

    vector_store = CapturingVectorStore([])
    reranker = UnexpectedReranker()
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
        sparse_embed_fn=lambda _texts: [{"1": 1.0}],
        reranker=reranker,
    )

    response = service.search(PublicCodeSearchRequest(
        query="empty", source_ids=("ffmpeg",), top_k=5, rerank=True
    ))

    assert response.hits == ()
    assert vector_store.arguments["top_k"] == 20
    assert reranker.calls == 0
    assert response.requested_mode == "hybrid"
    assert response.effective_mode == "hybrid"
    assert response.notice is None
    assert response.rerank_applied is False
    assert response.candidate_count == 0
    assert response.rerank_candidate_count == 0


def test_rerank_with_no_active_revision_returns_empty_before_embedding_or_vector_search():
    service = PublicCodeSearchService(
        repository=FakeRepository({}),
        vector_store=FakeVectorStore(),
        embed_fn=lambda _texts: pytest.fail("embedding must not run without active revisions"),
        reranker=object(),
    )

    response = service.search(PublicCodeSearchRequest(
        query="empty active revision", source_ids=("ffmpeg",), top_k=5, rerank=True
    ))

    assert response.hits == ()
    assert response.requested_mode == "hybrid"
    assert response.effective_mode == "hybrid"
    assert response.notice is None
    assert response.rerank_applied is False
    assert response.candidate_count == 0
    assert response.rerank_candidate_count == 0


def test_rerank_returns_all_available_candidates_when_fewer_than_requested_top_k():
    class RecordingReranker:
        def __init__(self):
            self.contents = []

        def score_pairs(self, _query, contents):
            self.contents = list(contents)
            return [0.1, 0.9]

    first = _raw_hit(score=0.9, path="api/first.py", start_line=10, code="first")
    second = _raw_hit(score=0.8, path="api/second.py", start_line=10, code="second")
    vector_store = CapturingVectorStore([first, second])
    reranker = RecordingReranker()
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
        reranker=reranker,
    )

    response = service.search(PublicCodeSearchRequest(
        query="short candidate pool", source_ids=("ffmpeg",), top_k=5, rerank=True
    ))

    assert vector_store.arguments["top_k"] == 20
    assert reranker.contents == [first["content"], second["content"]]
    assert [hit.code for hit in response.hits] == ["second", "first"]
    assert len(response.hits) == 2
    assert response.rerank_applied is True
    assert response.candidate_count == 2
    assert response.rerank_candidate_count == 2


@pytest.mark.parametrize(
    ("supports_sparse", "sparse_vector"),
    [
        (False, {"1": 1.0}),
        (True, {}),
    ],
    ids=["vector_store_without_sparse", "empty_sparse_embedding"],
)
def test_keyword_search_fails_explicitly_when_sparse_retrieval_is_unavailable(
    supports_sparse,
    sparse_vector,
):
    vector_store = CapturingVectorStore([], supports_sparse=supports_sparse)
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
        sparse_embed_fn=lambda _texts: [sparse_vector],
    )

    response = service.search(PublicCodeSearchRequest(
        query="Depends", source_ids=("ffmpeg",), mode="keyword", rerank=True
    ))

    assert vector_store.arguments is None
    assert response.hits == ()
    assert response.requested_mode == "keyword"
    assert response.effective_mode == "unavailable"
    assert response.notice is not None
    assert "does not substitute dense" in response.notice
    assert response.rerank_applied is False
    assert response.candidate_count == 0
    assert response.rerank_candidate_count == 0


@pytest.mark.parametrize(
    ("supports_sparse", "sparse_vector"),
    [
        (False, {"1": 1.0}),
        (True, {}),
    ],
    ids=["vector_store_without_sparse", "empty_sparse_embedding"],
)
def test_hybrid_search_reports_dense_fallback_when_sparse_retrieval_is_unavailable(
    supports_sparse,
    sparse_vector,
):
    hit = _raw_hit(score=0.8, path="api/dependencies.py", start_line=10, code="Depends")
    vector_store = CapturingVectorStore([hit], supports_sparse=supports_sparse)
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=vector_store,
        embed_fn=lambda _texts: [[1.0]],
        sparse_embed_fn=lambda _texts: [sparse_vector],
    )

    response = service.search(PublicCodeSearchRequest(
        query="Depends", source_ids=("ffmpeg",), mode="hybrid", rerank=False
    ))

    assert vector_store.arguments["mode"] == "dense"
    assert response.requested_mode == "hybrid"
    assert response.effective_mode == "dense"
    assert response.notice is not None
    assert "dense-only fallback" in response.notice
    assert response.rerank_applied is False
    assert response.candidate_count == 1
    assert response.rerank_candidate_count == 0


def test_default_search_embedding_uses_the_shared_dense_and_sparse_model_once(monkeypatch):
    class SharedEmbedder:
        calls = 0

        def embed_texts(self, _texts, *, priority):
            self.calls += 1
            assert priority == "interactive"
            return type(
                "EmbeddingResult",
                (),
                {"dense": [type("Vector", (), {"tolist": lambda self: [1.0]})()], "sparse": [{"1": 1.0}]},
            )()

    shared = SharedEmbedder()
    monkeypatch.setattr("core.indexing.embedders.get_embedder", lambda: shared)
    repository = FakeRepository({"ffmpeg": "ffmpeg-commit"})
    vector_store = FakeVectorStore()
    service = PublicCodeSearchService(
        repository=repository,
        vector_store=vector_store,
    )

    service.search(PublicCodeSearchRequest(query="encode", source_ids=("ffmpeg",), mode="hybrid"))

    assert shared.calls == 1
    assert vector_store.arguments["sparse_vector"] == {"1": 1.0}


def test_shared_dense_only_embedder_degrades_hybrid_search_to_dense(monkeypatch):
    class DenseOnlyEmbedder:
        def embed_texts(self, _texts, *, priority):
            assert priority == "interactive"
            return type(
                "EmbeddingResult",
                (),
                {
                    "dense": [type("Vector", (), {"tolist": lambda self: [0.1, 0.2]})()],
                    "sparse": None,
                },
            )()

    vector_store = FakeVectorStore()
    monkeypatch.setattr("core.indexing.embedders.get_embedder", lambda: DenseOnlyEmbedder())
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=vector_store,
    )

    service.search(PublicCodeSearchRequest(query="decoder", source_ids=("ffmpeg",)))

    assert vector_store.arguments["mode"] == "dense"
    assert vector_store.arguments["sparse_vector"] is None


def test_default_rerank_uses_the_process_shared_reranker(monkeypatch):
    class SharedReranker:
        calls = 0

        def score_pairs(self, _query, _contents):
            self.calls += 1
            return [0.97]

    shared = SharedReranker()
    monkeypatch.setattr("core.retrieval.reranker.get_shared_reranker", lambda: shared)
    service = PublicCodeSearchService(
        repository=FakeRepository({"ffmpeg": "ffmpeg-commit"}),
        vector_store=FakeVectorStore(),
        embed_fn=lambda _texts: [[1.0]],
    )

    response = service.search(PublicCodeSearchRequest(
        query="encode", source_ids=("ffmpeg",), rerank=True
    ))

    assert shared.calls == 1
    assert response.hits[0].score == 0.97
