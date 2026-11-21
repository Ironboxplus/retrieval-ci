"""The Qdrant count behind document index-status, on a real Qdrant.

Historical paper imports carry both integer and string ``doc_id`` payloads, so a
paper document is counted by its canonical ``paper_id``; a non-paper document by
its string ``doc_id``. Under the split layout each is counted where its points
live. Skipped unless a Qdrant is reachable on loopback.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from core.storage.corpus_layout import CorpusLayout  # noqa: E402
from core.storage.corpus_store import CorpusStore  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
PAPERS, ARXIV, REVIEWS = scratch_names("doccount").values()

POINTS = {
    PAPERS: [
        # One paper whose chunks were imported in two batches: int and str doc_id.
        (1, {"doc_id": 102394, "paper_id": "iclr-2024-wg8NPfeMF9", "venue_tier": "CCF-A",
             "source_type": "paper"}),
        (2, {"doc_id": "102394", "paper_id": "iclr-2024-wg8NPfeMF9", "venue_tier": "CCF-A",
             "source_type": "paper"}),
        (3, {"doc_id": "555", "source_type": "upload"}),
        # Legacy copy of the preprint below, still in papers during the migration.
        (4, {"doc_id": "777", "paper_id": "arxiv-2609.1", "venue_tier": "preprint",
             "source_type": "paper"}),
    ],
    ARXIV: [
        (4, {"doc_id": "777", "paper_id": "arxiv-2609.1", "venue_tier": "preprint",
             "source_type": "paper"}),
    ],
    REVIEWS: [],
}


@pytest.fixture(scope="module")
def client():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    for name, points in POINTS.items():
        client.create_collection(
            name, vectors_config={"dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)})
        for key in ("paper_id", "doc_id", "venue_tier", "source_type"):
            client.create_payload_index(name, key, qmodels.PayloadSchemaType.KEYWORD, wait=True)
        if points:
            client.upsert(name, wait=True, points=[
                qmodels.PointStruct(id=pid, vector={"dense": [0.5] * 4}, payload=payload)
                for pid, payload in points])
    try:
        yield client
    finally:
        drop(client, POINTS)


def _store(client, name):
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = name
    store.client = client
    return store


def _corpus(client, layout):
    return CorpusStore(papers=_store(client, PAPERS), arxiv=_store(client, ARXIV),
                       reviews=_store(client, REVIEWS), layout=layout)


def test_paper_document_counts_every_batch_by_paper_id(client):
    from app.api.routes.documents import _count_qdrant_points_for_document

    assert _count_qdrant_points_for_document(
        _corpus(client, CorpusLayout()), doc_id=102394, paper_id="iclr-2024-wg8NPfeMF9",
        source_type="paper", venue_tier="CCF-A") == 2


def test_nonpaper_document_counts_by_string_doc_id(client):
    from app.api.routes.documents import _count_qdrant_points_for_document

    assert _count_qdrant_points_for_document(
        _corpus(client, CorpusLayout()), doc_id=555, paper_id=None,
        source_type="upload") == 1


@pytest.mark.parametrize("layout", [CorpusLayout(),
                                    CorpusLayout(split=True, papers_legacy_mixed=True)])
def test_preprint_document_is_counted_once_in_either_layout(client, layout):
    from app.api.routes.documents import _count_qdrant_points_for_document

    assert _count_qdrant_points_for_document(
        _corpus(client, layout), doc_id=777, paper_id="arxiv-2609.1",
        source_type="paper", venue_tier="preprint") == 1
