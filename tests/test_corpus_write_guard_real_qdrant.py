"""QdrantStore.add_chunks refuses a paper point bound for the wrong split collection
before anything is written, and projects what it does write, on a real Qdrant.

The scratch collection stands in for `arxiv` by patching the configured name, so the
guard and projector run exactly as they would on the real collection. Skipped unless
a Qdrant is reachable on loopback; never touches production names.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.models.schemas import Chunk  # noqa: E402
from config import settings  # noqa: E402
from core.storage import corpus_layout  # noqa: E402
from core.storage.corpus_layout import MisroutedPaperPoint  # noqa: E402
from core.storage.papers_payload import PAPER_PROFILE_KEYS  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "corpus_arxiv_guard_test"
DIM = 4


@pytest.fixture()
def arxiv_store(monkeypatch):
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    monkeypatch.setattr(corpus_layout, "QDRANT_ARXIV_COLLECTION", COLLECTION)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    named = settings.embedding.supports_sparse
    client.create_collection(COLLECTION, vectors_config={"dense": params} if named else params)
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = COLLECTION
    store.client = client
    try:
        yield store
    finally:
        client.delete_collection(COLLECTION)


def _chunk(chunk_id: str, **metadata) -> Chunk:
    return Chunk(id=chunk_id, doc_id="901", content=f"text {chunk_id}", level=2,
                 start_idx=0, end_idx=8, embedding=[0.5] * DIM,
                 metadata={"paper_id": "p", "venue": "arXiv", "year": 2026, **metadata})


def test_misrouted_batch_is_refused_before_any_point_is_written(arxiv_store):
    batch = [_chunk("a", venue_tier="preprint"), _chunk("b", venue_tier="CCF-A")]
    with pytest.raises(MisroutedPaperPoint):
        arxiv_store.add_chunks(batch, source_type="paper")
    assert arxiv_store.client.count(COLLECTION, exact=True).count == 0


def test_correct_points_are_written_through_the_projector(arxiv_store):
    arxiv_store.add_chunks([_chunk("a", venue_tier="preprint", internal_note="x")],
                           source_type="paper")
    points, _ = arxiv_store.client.scroll(COLLECTION, limit=10, with_payload=True)
    assert len(points) == 1
    assert set(points[0].payload) <= set(PAPER_PROFILE_KEYS)
    assert points[0].payload["venue_tier"] == "preprint"
