"""arXiv preprints share `papers` but stay out of default search, on a real Qdrant.

The filter dicts come from the production builders (``build_qdrant_filters``,
``arxiv_scope``, ``retrieve_filters``) and are executed by the production
``QdrantStore.search`` against a real Qdrant, so what is verified is the
must_not actually reaching the engine: default search drops preprints and
reviews but keeps points with no venue_tier, ``include_preprints`` mixes them
in, /api/arxiv/search returns only preprints, and an in-paper search of an
arXiv paper is not emptied by the default exclusion.

Skipped unless a Qdrant is reachable on loopback. Never touches `papers`.
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
from core.storage.qdrant_store import QdrantStore  # noqa: E402

# Loopback only and hardcoded, never read from settings/QDRANT_URL -- see
# tests/test_code_duplicates_real_qdrant.py for why.
LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "papers_preprint_exclusion_test"
DIM = 8

POINTS = {
    # id: payload (the papers 8-field profile, minus content/doc_id noise)
    1: {"paper_id": "iclr-2025-1", "venue": "ICLR", "year": 2025, "venue_tier": "CCF-A",
        "source_type": "paper", "level": 2},
    2: {"paper_id": "arxiv-2609.08019", "venue": "arXiv", "year": 2026, "venue_tier": "preprint",
        "source_type": "paper", "level": 2},
    3: {"paper_id": "iclr-2025-1", "venue": "ICLR", "year": 2025, "venue_tier": "CCF-A",
        "source_type": "review", "level": 2},
    4: {"source_type": "upload", "level": 2},  # uploads carry no venue_tier
}


def _drop(client: QdrantClient, attempts: int) -> Exception | None:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            client.delete_collection(COLLECTION)
            return None
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.5 * (attempt + 1))
    return last


@pytest.fixture(scope="module")
def store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    _drop(client, attempts=4)

    from config import settings
    named = settings.embedding.supports_sparse
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(collection_name=COLLECTION,
                             vectors_config={"dense": params} if named else params)
    for key in ("venue_tier", "source_type", "paper_id", "venue"):
        client.create_payload_index(COLLECTION, key, qmodels.PayloadSchemaType.KEYWORD, wait=True)
    vector = [1.0] + [0.0] * (DIM - 1)
    client.upsert(COLLECTION, wait=True, points=[
        qmodels.PointStruct(id=pid, vector={"dense": vector} if named else vector,
                            payload={**payload, "content": f"point {pid}"})
        for pid, payload in POINTS.items()
    ])

    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = COLLECTION
    inner.client = client
    try:
        yield inner
    finally:
        failure = _drop(client, attempts=3)
        if failure is not None:
            warnings.warn(f"could not drop `{COLLECTION}` ({failure})", stacklevel=1)


def _hits(store: QdrantStore, http: HttpSearchRequest) -> set[int]:
    """Run the HTTP request's filters the way /api/search does, return point ids."""
    pipe = SearchRequest(query=http.query, filters=build_qdrant_filters(http),
                         source_type=http.source_type, include_preprints=http.include_preprints)
    rows = store.search([1.0] + [0.0] * (DIM - 1), top_k=10, filter_dict=retrieve_filters(pipe))
    return {int(r["content"].split()[1]) for r in rows}


def test_default_search_leaves_preprints_and_reviews_out(store):
    assert _hits(store, HttpSearchRequest(query="q")) == {1, 4}


def test_include_preprints_mixes_them_in(store):
    assert _hits(store, HttpSearchRequest(query="q", include_preprints=True)) == {1, 2, 4}


def test_arxiv_search_returns_only_preprints(store):
    assert _hits(store, arxiv_scope(HttpSearchRequest(query="q"))) == {2}
    # A venue/venue_tier the caller passes cannot widen the arXiv scope.
    widened = HttpSearchRequest(query="q", venue="ICLR", venue_tier="CCF-A")
    assert _hits(store, arxiv_scope(widened)) == {2}


def test_scoped_requests_are_not_narrowed(store):
    # In-paper search of an arXiv paper (search_within_paper) still finds it.
    assert _hits(store, HttpSearchRequest(query="q", paper_ids=["arxiv-2609.08019"])) == {2}
    assert _hits(store, HttpSearchRequest(query="q", venue="arXiv")) == {2}
    assert _hits(store, HttpSearchRequest(query="q", venue_tier="CCF-A")) == {1}


def test_retrieve_filters_keeps_caller_exclusions():
    # match_doc passes its own __exclude__ (the query paper); both must survive.
    item = retrieve_filters(SearchRequest(query="q", filters={"__exclude__": {"paper_id": ["p1"]}}))
    assert item["__exclude__"] == {"paper_id": ["p1"], "source_type": ["review"], "venue_tier": ["preprint"],
                                 "withdrawn": True}
