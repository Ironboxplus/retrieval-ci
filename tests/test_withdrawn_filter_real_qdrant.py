"""Withdrawn arXiv papers stay out of search unless asked for, on a real Qdrant.

The filter dicts come from the production builders (``build_qdrant_filters``,
``arxiv_scope``, ``retrieve_filters``) and run through ``CorpusStore`` against three
private scratch collections. Only the ``arxiv`` collection carries the optional
``withdrawn`` payload key (``true`` on withdrawn papers only) and its BOOL index, as in
production; ``papers`` and ``reviews`` have neither, so the exclusion must never be
sent to them. Verified here:

* default arXiv search and ``include_preprints`` search leave withdrawn points out,
  ``include_withdrawn`` brings them back;
* an id-scoped request (``paper_ids`` / ``doc_ids``) still finds a withdrawn paper;
* the routed filter for ``papers`` / ``reviews`` carries no ``withdrawn`` condition and
  the one for ``arxiv`` carries ``must_not withdrawn == true`` as a bool ``MatchValue``;
* under the unsplit layout (``papers`` only, no ``withdrawn`` index) nothing is sent.

Skipped unless a Qdrant is reachable on loopback. Never touches production names.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.models.schemas import SearchRequest as HttpSearchRequest  # noqa: E402
from app.services.search_grouping import arxiv_scope, build_qdrant_filters  # noqa: E402
from core.retrieval.pipeline import SearchRequest, retrieve_filters  # noqa: E402
from core.storage.corpus_layout import CorpusLayout  # noqa: E402
from core.storage.corpus_store import CorpusStore, route  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
DIM = 8

FORMAL = {"paper_id": "iclr-2025-1", "doc_id": "1", "venue": "ICLR", "year": 2025,
          "venue_tier": "CCF-A", "source_type": "paper", "level": 2}


def _preprint(n: int, **extra) -> dict:
    return {"paper_id": f"arxiv-2611.0000{n}", "doc_id": f"10{n}", "venue": "arXiv",
            "year": 2026, "venue_tier": "preprint", "source_type": "paper", "level": 2,
            **extra}


# point id -> (role, payload, similarity to the query). 12 is the withdrawn preprint and
# is the best match, so a missing exclusion is visible in the ranking too.
POINTS = {
    1: ("papers", FORMAL, 0.50),
    11: ("arxiv", _preprint(1), 0.70),
    12: ("arxiv", _preprint(2, withdrawn=True), 0.90),
    13: ("arxiv", _preprint(3), 0.60),
}
QUERY = [1.0] + [0.0] * (DIM - 1)


def _vector(similarity: float) -> list[float]:
    return [similarity, (1 - similarity ** 2) ** 0.5] + [0.0] * (DIM - 2)


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
    return client


@pytest.fixture(scope="module")
def names(client):
    names = scratch_names("withdrawn")
    drop(client, names.values())
    for role, name in names.items():
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )
        for key in ("venue_tier", "source_type", "paper_id", "doc_id", "venue"):
            client.create_payload_index(name, key, qmodels.PayloadSchemaType.KEYWORD, wait=True)
    # Production: the BOOL index exists on `arxiv` only.
    client.create_payload_index(names["arxiv"], "withdrawn", qmodels.PayloadSchemaType.BOOL,
                                wait=True)
    for pid, (role, payload, similarity) in POINTS.items():
        client.upsert(names[role], wait=True, points=[qmodels.PointStruct(
            id=pid,
            vector={"dense": _vector(similarity),
                    "sparse": qmodels.SparseVector(indices=[7], values=[similarity])},
            payload={**payload, "content": f"point {pid}"},
        )])
    try:
        yield names
    finally:
        drop(client, names.values())


@pytest.fixture
def corpus(client, names) -> CorpusStore:
    return CorpusStore(papers=_store(client, names["papers"]),
                       arxiv=_store(client, names["arxiv"]),
                       reviews=_store(client, names["reviews"]),
                       layout=CorpusLayout(split=True, papers_legacy_mixed=False))


def _pipe(http: HttpSearchRequest) -> SearchRequest:
    return SearchRequest(query=http.query, filters=build_qdrant_filters(http),
                         source_type=http.source_type, include_preprints=http.include_preprints,
                         include_withdrawn=http.include_withdrawn)


def _ids(rows: list[dict]) -> list[int]:
    return [int(r["content"].split()[1]) for r in rows]


def _hits(corpus: CorpusStore, http: HttpSearchRequest) -> list[int]:
    return _ids(corpus.search(QUERY, top_k=10, filter_dict=retrieve_filters(_pipe(http))))


def test_arxiv_search_leaves_withdrawn_out_by_default(corpus):
    assert _hits(corpus, arxiv_scope(HttpSearchRequest(query="q"))) == [11, 13]


def test_arxiv_search_with_include_withdrawn_brings_them_back(corpus):
    http = arxiv_scope(HttpSearchRequest(query="q", include_withdrawn=True))
    assert _hits(corpus, http) == [12, 11, 13]


def test_mixed_search_excludes_withdrawn_only_from_the_preprint_side(corpus):
    assert _hits(corpus, HttpSearchRequest(query="q", include_preprints=True)) == [11, 13, 1]
    both = HttpSearchRequest(query="q", include_preprints=True, include_withdrawn=True)
    assert _hits(corpus, both) == [12, 11, 13, 1]


def test_default_search_is_unchanged(corpus):
    assert _hits(corpus, HttpSearchRequest(query="q")) == [1]


def test_id_scoped_requests_still_find_a_withdrawn_paper(corpus):
    by_paper = HttpSearchRequest(query="q", paper_ids=["arxiv-2611.00002"])
    assert _hits(corpus, by_paper) == [12]
    assert _hits(corpus, arxiv_scope(by_paper)) == [12]
    by_doc = HttpSearchRequest(query="q", doc_ids=["102"])
    assert _hits(corpus, by_doc) == [12]


def test_hit_payload_carries_the_withdrawn_flag(corpus):
    rows = corpus.search(QUERY, top_k=10, filter_dict=retrieve_filters(
        _pipe(arxiv_scope(HttpSearchRequest(query="q", include_withdrawn=True)))))
    flags = {int(r["content"].split()[1]): r["metadata"].get("withdrawn") for r in rows}
    assert flags == {12: True, 11: None, 13: None}


def test_sparse_hybrid_and_batched_paths_apply_the_exclusion(corpus):
    from core.storage.qdrant_store import sparse_dict_to_qdrant

    sparse = sparse_dict_to_qdrant({"7": 1.0})
    filters = retrieve_filters(_pipe(arxiv_scope(HttpSearchRequest(query="q"))))
    assert set(_ids(corpus.sparse_search(sparse, top_k=10, filter_dict=filters))) == {11, 13}
    assert set(_ids(corpus.hybrid_search(QUERY, sparse, top_k=10, filter_dict=filters))) == {11, 13}
    batched = corpus.hybrid_search_many_exact([(QUERY, sparse)], top_k=10,
                                              filter_dicts=[filters])
    assert [set(_ids(rows)) for rows in batched] == [{11, 13}]


# ---- what is sent to which collection ----

def _plan(http: HttpSearchRequest, *, papers_legacy_mixed: bool = False) -> dict:
    return route(retrieve_filters(_pipe(http)), papers_legacy_mixed=papers_legacy_mixed)


def test_only_the_arxiv_role_receives_the_withdrawn_exclusion():
    plan = _plan(HttpSearchRequest(query="q", include_preprints=True))
    assert set(plan) == {"papers", "arxiv"}
    assert plan["arxiv"]["__exclude__"] == {"withdrawn": True}
    assert "withdrawn" not in plan["papers"].get("__exclude__", {})
    assert plan["papers"] == {}


def test_reviews_role_never_receives_it():
    pipe = SearchRequest(query="q", include_reviews=True, include_preprints=True)
    plan = route(retrieve_filters(pipe), papers_legacy_mixed=False)
    assert set(plan) == {"papers", "arxiv", "reviews"}
    for role in ("papers", "reviews"):
        assert "withdrawn" not in plan[role].get("__exclude__", {}), role
    assert plan["arxiv"]["__exclude__"] == {"withdrawn": True}


def test_papers_role_stays_unfiltered_while_it_holds_legacy_copies():
    plan = _plan(HttpSearchRequest(query="q", include_preprints=True), papers_legacy_mixed=True)
    assert "withdrawn" not in plan["papers"]["__exclude__"]
    assert plan["papers"]["__exclude__"] == {"source_type": ["review"], "venue_tier": ["preprint"]}


def test_plain_default_search_sends_no_withdrawn_condition_anywhere():
    for legacy in (True, False):
        plan = _plan(HttpSearchRequest(query="q"), papers_legacy_mixed=legacy)
        assert list(plan) == ["papers"]
        assert "withdrawn" not in plan["papers"].get("__exclude__", {})


def test_arxiv_filter_is_a_bool_match_value_must_not(client, names):
    plan = _plan(arxiv_scope(HttpSearchRequest(query="q")))
    assert list(plan) == ["arxiv"]
    built = _store(client, names["arxiv"])._build_filter(plan["arxiv"])
    assert built.must is None
    assert built.must_not == [qmodels.FieldCondition(
        key="withdrawn", match=qmodels.MatchValue(value=True))]
    assert built.must_not[0].match.value is True


def test_include_withdrawn_sends_no_condition_at_all():
    plan = _plan(arxiv_scope(HttpSearchRequest(query="q", include_withdrawn=True)))
    assert plan == {"arxiv": {}}


def test_id_scoped_request_sends_no_withdrawn_condition():
    plan = _plan(HttpSearchRequest(query="q", paper_ids=["arxiv-2611.00002"]))
    for residual in plan.values():
        assert "withdrawn" not in residual.get("__exclude__", {})


def test_unsplit_layout_sends_no_withdrawn_condition(client, names):
    # `papers` alone (unsplit) has no `withdrawn` index and no such payload key: a
    # must_not there would be an unindexed condition over every visited point.
    unsplit = CorpusStore(papers=_store(client, names["arxiv"]), layout=CorpusLayout(split=False))
    # The unsplit store here is pointed at the arxiv scratch collection purely so a
    # withdrawn point exists to be (not) excluded: if the condition were sent, 12 would vanish.
    http = arxiv_scope(HttpSearchRequest(query="q"))
    assert _hits(unsplit, http) == [12, 11, 13]
