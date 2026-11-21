"""local_ingest routes parquet points to papers / arxiv / reviews, on a real Qdrant.

A real parquet shard in the dump_chunks schema is ingested with the default
`--collection auto`. Under the mixed layout everything lands in `papers` (today's
behaviour); under the split layout each point lands in the collection its payload
belongs to; an explicit target refuses the whole shard if any point belongs
elsewhere, before anything is written. Scratch collections stand in for the three
by patching the configured names. Skipped unless a Qdrant is reachable on loopback.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("qdrant_client")
pq = pytest.importorskip("pyarrow.parquet")
import pyarrow as pa  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from core.storage import corpus_layout  # noqa: E402
from core.storage.corpus_layout import CorpusLayout, MisroutedPaperPoint  # noqa: E402
from scripts.remote_embed import local_ingest  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
NAMES: dict[str, str] = {}  # fresh per test (see tests/qdrant_scratch.py)
DIM = 4

ROWS = [
    (101, {"doc_id": "1", "paper_id": "iclr-2025-1", "venue": "ICLR", "venue_tier": "CCF-A",
           "source_type": "paper", "level": 2, "content": "formal"}),
    (102, {"doc_id": "2", "paper_id": "arxiv-2609.1", "venue": "arXiv", "venue_tier": "preprint",
           "source_type": "paper", "level": 2, "content": "preprint"}),
    (103, {"doc_id": "3", "paper_id": "iclr-2025-1", "venue": "ICLR", "venue_tier": "CCF-A",
           "source_type": "review", "level": 2, "content": "review"}),
]


@pytest.fixture()
def client(monkeypatch):
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    NAMES.update(scratch_names("ingest"))
    monkeypatch.setattr(corpus_layout, "QDRANT_PAPERS_COLLECTION", NAMES["papers"])
    monkeypatch.setattr(corpus_layout, "QDRANT_ARXIV_COLLECTION", NAMES["arxiv"])
    monkeypatch.setattr(corpus_layout, "QDRANT_REVIEWS_COLLECTION", NAMES["reviews"])
    for name in NAMES.values():
        client.create_collection(
            name,
            vectors_config={"dense": qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )
    try:
        yield client
    finally:
        drop(client, NAMES.values())


@pytest.fixture()
def shard(tmp_path):
    path = tmp_path / "vectors_0001.parquet"
    pq.write_table(pa.table({
        "point_id": [pid for pid, _ in ROWS],
        "dense": [[0.5] * DIM for _ in ROWS],
        "sparse_indices": [[3] for _ in ROWS],
        "sparse_values": [[0.2] for _ in ROWS],
        "payload_json": [json.dumps(payload) for _, payload in ROWS],
    }), path)
    return path


def _ids(client, role):
    points, _ = client.scroll(NAMES[role], limit=10)
    return sorted(p.id for p in points)


def test_mixed_layout_writes_every_point_to_papers(client, shard):
    sent = local_ingest.upsert_shard(client, "auto", shard, batch_size=2,
                                     layout=CorpusLayout(split=False))
    assert sent == 3
    assert _ids(client, "papers") == [101, 102, 103]
    assert _ids(client, "arxiv") == [] and _ids(client, "reviews") == []


def test_split_layout_routes_each_point_by_payload(client, shard):
    done = []
    sent = local_ingest.upsert_shard(client, "auto", shard, batch_size=2,
                                     layout=CorpusLayout(split=True),
                                     progress_callback=done.append)
    assert sent == 3 and done[-1] == 3
    assert _ids(client, "papers") == [101]
    assert _ids(client, "arxiv") == [102]
    assert _ids(client, "reviews") == [103]


def test_explicit_target_refuses_a_misrouted_shard_before_writing(client, shard):
    with pytest.raises(MisroutedPaperPoint):
        local_ingest.upsert_shard(client, NAMES["arxiv"], shard, batch_size=10,
                                  layout=CorpusLayout(split=True))
    assert _ids(client, "arxiv") == []
