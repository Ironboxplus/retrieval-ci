"""Per-call corpus paths must not count a whole collection.

``QdrantStore.delete_by_doc_id`` used to run an exact unfiltered ``count`` before and
after the delete just to report ``before - after`` (two whole-collection counts per
collection per document, up to three collections), and ``CorpusStore.point_counts``
(behind ``/api/stats`` and the dashboard overview poll) did an exact unfiltered count of
every serving collection. On the ~24M-point corpus both are O(points); the document's
own point count (payload-indexed ``doc_id``) and the collection's ``points_count`` are
the cheap, equally accurate sources.

Real Qdrant on loopback with scratch collections; the real client's ``count`` is wrapped
to record calls and always forwards.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.models.schemas import Chunk  # noqa: E402
from core.storage import corpus_layout  # noqa: E402
from core.storage.corpus_layout import CorpusLayout, write_layout  # noqa: E402
from core.storage.corpus_store import paper_store  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
NAMES: dict[str, str] = {}
DIM = 8
FORMAL = {"paper_id": "iclr-2025-1", "venue": "ICLR", "year": 2025, "venue_tier": "CCF-A"}
PREPRINT = {"paper_id": "arxiv-2609.1", "venue": "arXiv", "year": 2026,
            "venue_tier": "preprint"}


def _chunk(doc_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(id=f"{doc_id}-{text}", doc_id=doc_id, content=text, level=2,
                 start_idx=0, end_idx=len(text), metadata=dict(meta),
                 embedding=[0.5] * DIM)


@pytest.fixture()
def counts(monkeypatch, tmp_path):
    """Yields (client, calls); ``calls`` lists (collection, filtered?, exact) per count."""
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    NAMES.update(scratch_names("cheapcount"))
    monkeypatch.setattr(corpus_layout, "QDRANT_PAPERS_COLLECTION", NAMES["papers"])
    monkeypatch.setattr(corpus_layout, "QDRANT_ARXIV_COLLECTION", NAMES["arxiv"])
    monkeypatch.setattr(corpus_layout, "QDRANT_REVIEWS_COLLECTION", NAMES["reviews"])
    monkeypatch.setenv("QDRANT_CORPUS_LAYOUT_FILE", str(tmp_path / "corpus_layout.json"))
    for name in NAMES.values():
        client.create_collection(
            name,
            vectors_config={"dense": qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )
        for key in ("venue_tier", "source_type", "paper_id", "doc_id"):
            client.create_payload_index(name, key, qmodels.PayloadSchemaType.KEYWORD, wait=True)

    calls: list[tuple[str, bool, bool]] = []
    original = QdrantClient.count

    def recording_count(self, collection_name, count_filter=None, exact=True, **kwargs):
        calls.append((collection_name, count_filter is not None, exact))
        return original(self, collection_name, count_filter=count_filter, exact=exact, **kwargs)

    monkeypatch.setattr(QdrantClient, "count", recording_count)
    try:
        yield client, calls
    finally:
        monkeypatch.undo()
        drop(client, NAMES.values())


def _write_three(store) -> None:
    store.add_chunks([_chunk("1", "formal-a", FORMAL), _chunk("1", "formal-b", FORMAL)],
                     source_type="paper")
    store.add_chunks([_chunk("2", "preprint", PREPRINT)], source_type="paper")
    store.add_chunks([_chunk("3", "review", FORMAL)], source_type="review")


def _unfiltered(calls):
    return [c for c in calls if not c[1]]


def test_delete_by_doc_id_reports_the_documents_points_without_a_collection_count(counts):
    client, calls = counts
    store = paper_store(NAMES["papers"])
    _write_three(store)
    calls.clear()

    assert store.delete_by_doc_id("1") == 2

    assert _unfiltered(calls) == []
    assert client.count(NAMES["papers"], exact=True).count == 2  # other documents intact
    assert store.delete_by_doc_id("1") == 0  # gone: nothing to delete, nothing reported
    assert store.delete_by_doc_id("404") == 0


def test_delete_by_doc_id_sums_the_migration_copies_in_every_collection(counts, tmp_path):
    client, calls = counts
    QdrantStore(collection_name=NAMES["papers"]).add_chunks(
        [_chunk("2", "preprint", PREPRINT)], source_type="paper")
    QdrantStore(collection_name=NAMES["arxiv"]).add_chunks(
        [_chunk("2", "preprint", PREPRINT), _chunk("2", "preprint-2", PREPRINT)],
        source_type="paper")
    calls.clear()

    assert paper_store(NAMES["papers"]).delete_by_doc_id("2") == 3
    assert _unfiltered(calls) == []


def test_point_counts_use_collection_metadata_not_an_exact_count(counts, tmp_path):
    client, calls = counts
    legacy = QdrantStore(collection_name=NAMES["papers"])
    _write_three(legacy)
    QdrantStore(collection_name=NAMES["arxiv"]).add_chunks(
        [_chunk("2", "preprint", PREPRINT)], source_type="paper")
    QdrantStore(collection_name=NAMES["reviews"]).add_chunks(
        [_chunk("3", "review", FORMAL)], source_type="review")
    store = paper_store(NAMES["papers"])
    write_layout(CorpusLayout(split=True, papers_legacy_mixed=True), tmp_path / "corpus_layout.json")
    calls.clear()

    assert store.point_counts() == {"papers": 4, "arxiv": 1, "reviews": 1, "legacy_copies": 2}
    assert store.count() == 4  # 4 in papers + 1 + 1 minus 2 legacy copies
    assert _unfiltered(calls) == []
