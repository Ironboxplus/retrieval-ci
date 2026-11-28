"""Writes, deletes and counts over papers / arxiv / reviews, on a real Qdrant.

`paper_store()` is what IndexService and the crawler integration write through. With
the layout file switched per test (no restart, as in production) this checks:

* mixed layout: every chunk lands in `papers`, as before the split;
* split layout: each chunk lands in the collection its payload belongs to, and a
  missing target collection refuses the whole write before anything lands;
* deletes reach every paper collection that exists, so a migration copy cannot
  outlive its document;
* counts read where points are served from and skip the legacy copies still in
  `papers`, so nothing is counted twice during the migration window.

Scratch collections stand in for the three by patching the configured names.
Skipped unless a Qdrant is reachable on loopback.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.models.schemas import Chunk  # noqa: E402
from core.storage import corpus_layout  # noqa: E402
from core.storage.corpus_layout import CorpusLayout, write_layout  # noqa: E402
from core.storage.corpus_store import CorpusStore, paper_store, role_for_document  # noqa: E402
from core.storage.qdrant_store import QdrantCollectionMissing, QdrantStore  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
NAMES: dict[str, str] = {}  # fresh per test (see tests/qdrant_scratch.py)
DIM = 8

FORMAL = {"paper_id": "iclr-2025-1", "venue": "ICLR", "year": 2025, "venue_tier": "CCF-A"}
PREPRINT = {"paper_id": "arxiv-2609.1", "venue": "arXiv", "year": 2026,
            "venue_tier": "preprint"}


def _chunk(doc_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(id=f"{doc_id}-{text}", doc_id=doc_id, content=text, level=2,
                 start_idx=0, end_idx=len(text), metadata=dict(meta),
                 embedding=[0.5] * DIM)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    NAMES.update(scratch_names("writes"))
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
    try:
        yield client
    finally:
        drop(client, NAMES.values())


def _set(tmp_path, **kwargs) -> None:
    write_layout(CorpusLayout(**kwargs), tmp_path / "corpus_layout.json")


def _docs(client, role) -> list[str]:
    points, _ = client.scroll(NAMES[role], limit=100, with_payload=True)
    return sorted(f"{p.payload['doc_id']}:{p.payload['source_type']}" for p in points)


def _write_three(store) -> None:
    store.add_chunks([_chunk("1", "formal", FORMAL)], source_type="paper")
    store.add_chunks([_chunk("2", "preprint", PREPRINT)], source_type="paper")
    store.add_chunks([_chunk("3", "review", FORMAL)], source_type="review")


def test_papers_name_gets_the_routed_corpus_and_other_names_do_not(client):
    assert isinstance(paper_store(NAMES["papers"]), CorpusStore)
    assert type(paper_store(NAMES["arxiv"])) is QdrantStore


def test_mixed_layout_writes_everything_to_papers(client):
    _write_three(paper_store(NAMES["papers"]))
    assert _docs(client, "papers") == ["1:paper", "2:paper", "3:review"]
    assert _docs(client, "arxiv") == [] and _docs(client, "reviews") == []


def test_split_layout_writes_each_chunk_where_it_belongs(client, tmp_path):
    _set(tmp_path, split=True, papers_legacy_mixed=False)
    _write_three(paper_store(NAMES["papers"]))
    assert _docs(client, "papers") == ["1:paper"]
    assert _docs(client, "arxiv") == ["2:paper"]
    assert _docs(client, "reviews") == ["3:review"]


def test_split_write_refuses_when_its_collection_is_missing(client, tmp_path):
    _set(tmp_path, split=True, papers_legacy_mixed=False)
    client.delete_collection(NAMES["arxiv"])
    store = paper_store(NAMES["papers"])
    with pytest.raises(QdrantCollectionMissing):
        store.add_chunks([_chunk("1", "formal", FORMAL), _chunk("2", "preprint", PREPRINT)],
                         source_type="paper")
    assert _docs(client, "papers") == []
    assert not client.collection_exists(NAMES["arxiv"])  # never auto-created bare


def test_delete_reaches_the_migration_copy_in_either_layout(client, tmp_path):
    # Mixed layout while the copy runs: the preprint is in papers and in arxiv.
    QdrantStore(collection_name=NAMES["papers"]).add_chunks(
        [_chunk("2", "preprint", PREPRINT)], source_type="paper")
    QdrantStore(collection_name=NAMES["arxiv"]).add_chunks(
        [_chunk("2", "preprint", PREPRINT)], source_type="paper")
    store = paper_store(NAMES["papers"])
    assert store.delete_by_doc_id("2") == 2
    assert _docs(client, "papers") == [] and _docs(client, "arxiv") == []


def test_counts_skip_the_legacy_copies_still_in_papers(client, tmp_path):
    # Migration window: papers still holds copies of the preprint and the review.
    legacy = QdrantStore(collection_name=NAMES["papers"])
    _write_three(legacy)
    QdrantStore(collection_name=NAMES["arxiv"]).add_chunks(
        [_chunk("2", "preprint", PREPRINT)], source_type="paper")
    QdrantStore(collection_name=NAMES["reviews"]).add_chunks(
        [_chunk("3", "review", FORMAL)], source_type="review")
    store = paper_store(NAMES["papers"])

    assert store.count() == 3  # mixed: all of papers
    assert store.count_points("paper_id", "iclr-2025-1") == 2  # paper + its review

    _set(tmp_path, split=True, papers_legacy_mixed=True)
    assert store.point_counts() == {"papers": 3, "arxiv": 1, "reviews": 1, "legacy_copies": 2}
    assert store.count() == 3  # 1 formal + 1 arxiv + 1 review, copies not doubled
    assert store.count_by_doc_id("2") == 1 and store.exists_by_doc_id("3")
    formal = [role_for_document(source_type="paper", venue_tier="CCF-A")]
    assert store.count_points("paper_id", "iclr-2025-1", roles=formal) == 1
    preprint = [role_for_document(source_type="paper", venue_tier="preprint")]
    assert store.count_points("paper_id", "arxiv-2609.1", roles=preprint) == 1

    _set(tmp_path, split=True, papers_legacy_mixed=False)  # copies deleted
    for doc_id in ("2", "3"):
        legacy.client.delete(NAMES["papers"], points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(must=[qmodels.FieldCondition(
                key="doc_id", match=qmodels.MatchValue(value=doc_id))])), wait=True)
    assert store.count() == 3


@pytest.mark.parametrize("source_type,venue_tier,role", [
    ("review", "CCF-A", "reviews"),
    ("paper", "preprint", "arxiv"),
    ("paper", "CCF-A", "papers"),
    ("upload", None, "papers"),
])
def test_role_for_document(source_type, venue_tier, role):
    assert role_for_document(source_type=source_type, venue_tier=venue_tier) == role
