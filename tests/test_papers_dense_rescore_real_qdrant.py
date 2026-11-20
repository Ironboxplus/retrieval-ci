"""`papers` dense retrieval rescores oversampled candidates against the originals.

The 2-bit TurboQuant copy that keeps `papers` resident ranks coarsely on its own;
QdrantStore therefore asks for rescore + 2x oversampling on papers dense queries
(plain and the dense leg of hybrid). What has to hold on a real Qdrant:

* the returned dense scores are the exact cosine of the original vectors (rescored),
  and the ranking matches exact search;
* hybrid search accepts the params on its dense prefetch and still fuses normally;
* other collections (code, journal: no quantization) get no params at all.

Skipped unless a local Qdrant is reachable on loopback.
"""
from __future__ import annotations

import random
import time

import numpy as np
import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models  # noqa: E402

from core.storage import qdrant_store  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "papers_dense_rescore_test"
POINTS = 1500
DIM = 1024
SEGMENTS = 16


def _drop(client: QdrantClient, attempts: int = 4) -> None:
    for attempt in range(attempts):
        try:
            client.delete_collection(COLLECTION)
            return
        except Exception:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


@pytest.fixture(scope="module")
def corpus():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=60)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    _drop(client)
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.COSINE,
                                                     on_disk=True)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
        quantization_config=models.TurboQuantization(turbo=models.TurboQuantQuantizationConfig(
            bits=models.TurboQuantBitSize.BITS2, always_ram=True)),
        # Qdrant's default segment count is the host's CPU count, and a 2-bit quantized
        # candidate set depends on how the points are split: a 2-vCPU runner and a
        # 16-thread desktop disagree on the lowest ranks. Pin the layout.
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1000, default_segment_number=SEGMENTS),
    )
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(POINTS, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    terms = random.Random(5)
    points = []
    for i, vec in enumerate(vectors):
        ids = sorted(terms.sample(range(1, 80), 5))
        points.append(models.PointStruct(
            id=i + 1,
            vector={"dense": vec.tolist(),
                    "sparse": models.SparseVector(indices=ids, values=[1.0] * len(ids))},
            payload={"chunk_id": f"chunk-{i + 1}", "content": f"c{i}"}))
    for start in range(0, POINTS, 300):
        client.upsert(COLLECTION, points=points[start:start + 300], wait=True)
    for _ in range(120):
        if client.get_collection(COLLECTION).status == models.CollectionStatus.GREEN:
            break
        time.sleep(0.5)
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = COLLECTION
    store.client = client
    yield store, vectors
    _drop(client)


@pytest.fixture()
def as_papers(monkeypatch):
    monkeypatch.setattr(qdrant_store, "QDRANT_PAPERS_COLLECTION", COLLECTION)


def test_papers_dense_search_returns_exact_rescored_scores_and_ranking(corpus, as_papers):
    store, vectors = corpus
    rng = np.random.default_rng(9)
    for qi in rng.choice(POINTS, size=10, replace=False):
        query = vectors[qi] + 0.05 * rng.normal(size=DIM).astype(np.float32)
        query /= np.linalg.norm(query)
        hits = store.search(query.tolist(), top_k=10)
        exact = np.argsort(-(vectors @ query))[:10] + 1
        assert [h["id"] for h in hits] == [f"chunk-{i}" for i in exact]
        for hit in hits:
            pid = int(hit["id"].split("-")[1])
            assert hit["score"] == pytest.approx(float(vectors[pid - 1] @ query), abs=1e-4)


def test_papers_hybrid_search_carries_the_params_on_its_dense_leg(corpus, as_papers):
    store, vectors = corpus
    hits = store.hybrid_search(
        dense_vector=vectors[41].tolist(),
        sparse_vector=models.SparseVector(indices=[1, 2, 3], values=[1.0, 1.0, 1.0]),
        top_k=10)
    # RRF may put a point strong on both legs first; the dense #1 must survive fusion.
    assert len(hits) == 10 and "chunk-42" in [h["id"] for h in hits]


def test_only_papers_gets_rescore_params(corpus, monkeypatch):
    store, _ = corpus
    monkeypatch.setattr(qdrant_store, "QDRANT_PAPERS_COLLECTION", COLLECTION)
    params = store._dense_search_params()
    assert params.quantization.rescore is True and params.quantization.oversampling == 2.0

    monkeypatch.setattr(qdrant_store, "QDRANT_PAPERS_COLLECTION", "papers")
    assert store._dense_search_params() is None  # e.g. code_chunks / journal_chunks

    monkeypatch.setattr(qdrant_store, "QDRANT_PAPERS_COLLECTION", COLLECTION)
    monkeypatch.setattr(qdrant_store, "QDRANT_PAPERS_DENSE_OVERSAMPLING", 0.0)
    assert store._dense_search_params() is None
