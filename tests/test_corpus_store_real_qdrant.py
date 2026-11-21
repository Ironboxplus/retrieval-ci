"""Paper search split across papers / arxiv / reviews collections, on a real Qdrant.

The filter dicts come from the production builders (``build_qdrant_filters``,
``arxiv_scope``, ``retrieve_filters``) and run through ``CorpusStore`` against three
isolated collections, so what is verified is the routing actually reaching the
engine:

* default search reads only ``papers`` and sends it no preprint/review filter once
  ``papers`` is clean; while it still holds legacy preprints/reviews (the migration
  window) those are excluded there so nothing comes back twice;
* ``include_preprints`` / ``include_reviews`` add the other collections and merge by
  score; ``arxiv_scope`` and ``source_type="review"`` read one collection only, and
  the queries they send carry no condition that is always true there;
* id-scoped requests (in-paper search) find the paper wherever it lives, and a
  caller's own ``__exclude__`` (match_doc's query paper) survives routing;
* dense, sparse, hybrid and the batched hybrid path all route the same way.

Skipped unless a Qdrant is reachable on loopback. Never touches production names.
"""
from __future__ import annotations

import time
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.models.schemas import SearchRequest as HttpSearchRequest  # noqa: E402
from app.services.search_grouping import arxiv_scope, build_qdrant_filters  # noqa: E402
from core.retrieval.pipeline import SearchRequest, retrieve_filters  # noqa: E402
from core.storage.corpus_layout import CorpusLayout, current_layout, write_layout  # noqa: E402
from core.storage.corpus_store import CorpusStore, route  # noqa: E402
from core.storage.qdrant_store import QdrantStore, sparse_dict_to_qdrant  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
PAPERS, ARXIV, REVIEWS = "corpus_papers_test", "corpus_arxiv_test", "corpus_reviews_test"
DIM = 8

FORMAL = {"paper_id": "iclr-2025-1", "venue": "ICLR", "year": 2025, "venue_tier": "CCF-A",
          "source_type": "paper", "level": 2}
PREPRINT = {"paper_id": "arxiv-2609.08019", "venue": "arXiv", "year": 2026,
            "venue_tier": "preprint", "source_type": "paper", "level": 2}
REVIEW = {"paper_id": "iclr-2025-1", "venue": "ICLR", "year": 2025, "venue_tier": "CCF-A",
          "source_type": "review", "level": 2}
UPLOAD = {"source_type": "upload", "level": 2}  # uploads carry no venue_tier

# id -> (collection, payload, similarity to the query). 2 and 3 are the legacy copies
# still inside `papers` during the migration window.
POINTS = {
    1: (PAPERS, FORMAL, 0.70),
    4: (PAPERS, UPLOAD, 0.40),
    2: (PAPERS, PREPRINT, 0.95),
    3: (PAPERS, REVIEW, 0.90),
    12: (ARXIV, PREPRINT, 0.80),
    13: (REVIEWS, REVIEW, 0.60),
}
QUERY = [1.0] + [0.0] * (DIM - 1)
SPARSE_QUERY = {"7": 1.0}


def _vector(similarity: float) -> list[float]:
    return [similarity, (1 - similarity ** 2) ** 0.5] + [0.0] * (DIM - 2)


def _drop(client: QdrantClient, attempts: int) -> Exception | None:
    last: Exception | None = None
    for name in (PAPERS, ARXIV, REVIEWS):
        for attempt in range(attempts):
            try:
                client.delete_collection(name)
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.5 * (attempt + 1))
    return last


def _store(client: QdrantClient, name: str) -> QdrantStore:
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = name
    inner.client = client
    return inner


@pytest.fixture(scope="module")
def client():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    _drop(client, attempts=4)
    for name in (PAPERS, ARXIV, REVIEWS):
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )
        for key in ("venue_tier", "source_type", "paper_id", "venue"):
            client.create_payload_index(name, key, qmodels.PayloadSchemaType.KEYWORD, wait=True)
    for pid, (name, payload, similarity) in POINTS.items():
        client.upsert(name, wait=True, points=[qmodels.PointStruct(
            id=pid,
            vector={"dense": _vector(similarity),
                    "sparse": qmodels.SparseVector(indices=[7], values=[similarity])},
            payload={**payload, "content": f"point {pid}"},
        )])
    try:
        yield client
    finally:
        failure = _drop(client, attempts=3)
        if failure is not None:
            warnings.warn(f"could not drop corpus test collections ({failure})", stacklevel=1)


def _corpus(client: QdrantClient, *, papers_legacy_mixed: bool) -> CorpusStore:
    return CorpusStore(papers=_store(client, PAPERS), arxiv=_store(client, ARXIV),
                       reviews=_store(client, REVIEWS),
                       layout=CorpusLayout(split=True, papers_legacy_mixed=papers_legacy_mixed))


@pytest.fixture(params=[True, False], ids=["legacy_mixed", "clean"])
def corpus(request, client):
    if not request.param:
        # A clean `papers` has no legacy preprint/review copies.
        client.delete(PAPERS, points_selector=qmodels.PointIdsList(points=[2, 3]), wait=True)
    try:
        yield _corpus(client, papers_legacy_mixed=request.param)
    finally:
        if not request.param:
            for pid in (2, 3):
                name, payload, similarity = POINTS[pid]
                client.upsert(name, wait=True, points=[qmodels.PointStruct(
                    id=pid,
                    vector={"dense": _vector(similarity),
                            "sparse": qmodels.SparseVector(indices=[7], values=[similarity])},
                    payload={**payload, "content": f"point {pid}"},
                )])


def _filters(http: HttpSearchRequest, **pipeline_kwargs) -> dict:
    pipe = SearchRequest(query=http.query, filters=build_qdrant_filters(http),
                         source_type=http.source_type, include_preprints=http.include_preprints,
                         **pipeline_kwargs)
    return retrieve_filters(pipe)


def _ids(rows: list[dict]) -> list[int]:
    return [int(r["content"].split()[1]) for r in rows]


def _dense(corpus: CorpusStore, http: HttpSearchRequest, **kw) -> list[int]:
    return _ids(corpus.search(QUERY, top_k=10, filter_dict=_filters(http, **kw)))


def test_default_search_reads_formal_papers_only(corpus):
    assert _dense(corpus, HttpSearchRequest(query="q")) == [1, 4]


def test_include_preprints_merges_arxiv_by_score_without_duplicates(corpus):
    assert _dense(corpus, HttpSearchRequest(query="q", include_preprints=True)) == [12, 1, 4]


def test_include_reviews_merges_the_reviews_collection(corpus):
    assert _dense(corpus, HttpSearchRequest(query="q"), include_reviews=True) == [1, 13, 4]


def test_arxiv_scope_reads_arxiv_only_and_cannot_be_widened(corpus):
    assert _dense(corpus, arxiv_scope(HttpSearchRequest(query="q"))) == [12]
    widened = HttpSearchRequest(query="q", venue="ICLR", venue_tier="CCF-A")
    assert _dense(corpus, arxiv_scope(widened)) == [12]


def test_review_only_reads_the_reviews_collection(corpus):
    assert _dense(corpus, HttpSearchRequest(query="q", source_type="review")) == [13]


def test_id_scoped_requests_find_the_paper_wherever_it_lives(corpus):
    assert _dense(corpus, HttpSearchRequest(query="q", paper_ids=["arxiv-2609.08019"])) == [12]
    assert _dense(corpus, HttpSearchRequest(query="q", paper_ids=["iclr-2025-1"])) == [1]
    assert _dense(corpus, HttpSearchRequest(query="q", venue="arXiv")) == [12]
    assert _dense(corpus, HttpSearchRequest(query="q", venue_tier="CCF-A")) == [1]


def test_caller_exclusions_survive_routing(corpus):
    filters = _filters(HttpSearchRequest(query="q", include_preprints=True))
    filters["__exclude__"]["paper_id"] = ["arxiv-2609.08019"]
    assert _ids(corpus.search(QUERY, top_k=10, filter_dict=filters)) == [1, 4]


def test_sparse_hybrid_and_batched_paths_route_the_same_way(corpus):
    sparse = sparse_dict_to_qdrant(SPARSE_QUERY)
    default = _filters(HttpSearchRequest(query="q"))
    mixed = _filters(HttpSearchRequest(query="q", include_preprints=True))
    assert set(_ids(corpus.sparse_search(sparse, top_k=10, filter_dict=default))) == {1, 4}
    assert set(_ids(corpus.hybrid_search(QUERY, sparse, top_k=10, filter_dict=mixed))) == {1, 4, 12}
    batched = corpus.hybrid_search_many_exact([(QUERY, sparse), (QUERY, sparse)], top_k=10,
                                              filter_dicts=[default, mixed])
    assert [set(_ids(rows)) for rows in batched] == [{1, 4}, {1, 4, 12}]


def test_queries_to_arxiv_and_reviews_carry_no_always_true_condition():
    arxiv = route(retrieve_filters(SearchRequest(
        query="q", filters=build_qdrant_filters(arxiv_scope(HttpSearchRequest(query="q"))))),
        papers_legacy_mixed=False)
    assert list(arxiv) == ["arxiv"]
    assert "venue_tier" not in arxiv["arxiv"] and "venue" not in arxiv["arxiv"]
    assert "venue_tier" not in arxiv["arxiv"].get("__exclude__", {})

    reviews = route(retrieve_filters(SearchRequest(query="q", source_type="review")),
                    papers_legacy_mixed=False)
    assert list(reviews) == ["reviews"]
    assert "source_type" not in reviews["reviews"]

    default = route(retrieve_filters(SearchRequest(query="q")), papers_legacy_mixed=False)
    assert default == {"papers": {}}
    legacy = route(retrieve_filters(SearchRequest(query="q")), papers_legacy_mixed=True)
    assert legacy == {"papers": {"__exclude__": {"source_type": ["review"],
                                                 "venue_tier": ["preprint"]}}}


def test_mixed_layout_serves_papers_exactly_as_before(client):
    mixed = CorpusStore(papers=_store(client, PAPERS), layout=CorpusLayout(split=False))
    # The legacy single collection still applies the default exclusions itself.
    assert _dense(mixed, HttpSearchRequest(query="q")) == [1, 4]
    assert _dense(mixed, HttpSearchRequest(query="q", include_preprints=True)) == [2, 1, 4]
    assert _dense(mixed, arxiv_scope(HttpSearchRequest(query="q"))) == [2]


def test_layout_file_switches_without_a_restart(client, tmp_path, monkeypatch):
    path = tmp_path / "corpus_layout.json"
    monkeypatch.setenv("QDRANT_CORPUS_LAYOUT_FILE", str(path))
    live = CorpusStore(papers=_store(client, PAPERS), arxiv=_store(client, ARXIV),
                       reviews=_store(client, REVIEWS))
    arxiv_only = arxiv_scope(HttpSearchRequest(query="q"))
    assert current_layout() == CorpusLayout(split=False)  # no file yet: mixed
    assert _dense(live, arxiv_only) == [2]
    write_layout(CorpusLayout(split=True, papers_legacy_mixed=True), path)
    assert _dense(live, arxiv_only) == [12]
    # An invalid edit keeps the last good layout instead of guessing.
    path.write_text('{"layout": "spilt"}', encoding="utf-8")
    import os
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
    assert current_layout() == CorpusLayout(split=True, papers_legacy_mixed=True)


def test_split_collection_not_created_yet_falls_back_to_papers(client, monkeypatch):
    monkeypatch.setattr("core.storage.corpus_store.paper_collections",
                        lambda: {"papers": PAPERS, "arxiv": "corpus_missing_test",
                                 "reviews": "corpus_missing_test"})
    from core.storage import qdrant_store

    monkeypatch.setattr(qdrant_store, "_get_shared_client", lambda _dir: client)
    if client.collection_exists("corpus_missing_test"):
        client.delete_collection("corpus_missing_test")
    split = CorpusStore(papers=_store(client, PAPERS),
                        layout=CorpusLayout(split=True, papers_legacy_mixed=True))
    assert _dense(split, arxiv_scope(HttpSearchRequest(query="q"))) == [2]
    assert not client.collection_exists("corpus_missing_test")
