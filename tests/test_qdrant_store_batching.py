"""add_chunks must batch the Qdrant upsert.

W: a whole-book upload chunked into ~7000 points and add_chunks sent them in a
single REST upsert → 70MB JSON payload → Qdrant 400 "JSON payload larger than
allowed (limit 33554432 bytes)" → index_document failed, 0 points landed.
Fix: split the upsert into bounded batches (UPSERT_BATCH_SIZE) so no single
payload approaches the 32MB REST limit, each batch wait=True + W14 retry.
"""
import pytest

from core.storage.qdrant_store import QdrantStore, UPSERT_BATCH_SIZE
from app.models.schemas import Chunk


class _RecordingClient:
    """Fake Qdrant client that records every upsert's point batch."""

    def __init__(self):
        self.batches = []

    def upsert(self, collection_name, points, wait=True):
        # record a copy of the batch sizes / ids
        self.batches.append(list(points))


class _FlakyClient(_RecordingClient):
    """Fails the first `fail_times` upsert attempts, then succeeds."""

    def __init__(self, fail_times=1):
        super().__init__()
        self.fail_times = fail_times
        self.attempts = 0

    def upsert(self, collection_name, points, wait=True):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(f"transient 503 (attempt {self.attempts})")
        super().upsert(collection_name, points, wait=wait)


class _CollectionClient:
    def __init__(self):
        self.create_args = None
        self.create_kwargs = None

    def collection_exists(self, _collection_name):
        return False

    def create_collection(self, *args, **kwargs):
        self.create_args = args
        self.create_kwargs = kwargs


def _bare_store(client):
    """Build a QdrantStore without touching a real Qdrant connection.

    Collection name is deliberately NOT "papers" — these tests exercise
    generic upsert batching/retry, not the papers payload contract (which
    fail-closed rejects any payload without a registered `source_type`,
    see core/storage/papers_payload.py). Papers-specific payload behavior is
    covered by tests/test_papers_write_paths_contract.py.
    """
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "code_chunks"
    store.client = client
    return store


def _make_chunks(n):
    return [
        Chunk(
            id=f"c{i}",
            doc_id="102394",
            content=f"chunk text number {i}",
            level=2,
            start_idx=i,
            end_idx=i + 1,
            embedding=[0.0] * 8,
            metadata={},
        )
        for i in range(n)
    ]


def test_add_chunks_splits_large_input_into_bounded_batches():
    client = _RecordingClient()
    store = _bare_store(client)
    n = UPSERT_BATCH_SIZE * 2 + 5  # forces 3 batches

    store.add_chunks(_make_chunks(n))

    assert len(client.batches) == 3, "should split into ceil(n / batch) upserts"
    for batch in client.batches:
        assert len(batch) <= UPSERT_BATCH_SIZE, "no batch may exceed the cap"
    # every point delivered exactly once — batching must not drop data
    assert sum(len(b) for b in client.batches) == n


def test_add_chunks_small_input_is_single_batch():
    client = _RecordingClient()
    store = _bare_store(client)

    store.add_chunks(_make_chunks(3))

    assert len(client.batches) == 1
    assert len(client.batches[0]) == 3


def test_add_chunks_empty_is_noop():
    client = _RecordingClient()
    store = _bare_store(client)

    store.add_chunks([])

    assert client.batches == []


def test_add_chunks_retries_transient_upsert_failure(monkeypatch):
    # don't actually sleep through the backoff
    import core.storage.qdrant_store as qs
    monkeypatch.setattr(qs.time, "sleep", lambda *_a, **_k: None)

    client = _FlakyClient(fail_times=1)
    store = _bare_store(client)

    store.add_chunks(_make_chunks(3))  # single batch, fails once then succeeds

    assert client.attempts == 2, "should retry the failed batch"
    assert len(client.batches) == 1, "retried batch lands exactly once"
    assert len(client.batches[0]) == 3


@pytest.mark.parametrize("prefer_grpc", [True, False])
def test_server_client_honors_grpc_transport_configuration(monkeypatch, prefer_grpc):
    import core.storage.qdrant_store as qs

    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(qs, "QdrantClient", fake_client)
    monkeypatch.setattr(qs, "QDRANT_PREFER_GRPC", prefer_grpc)
    monkeypatch.setattr(qs, "QDRANT_GRPC_PORT", 7444)

    qs._build_server_client("http://qdrant.internal:6333")

    assert captured == {
        "url": "http://qdrant.internal:6333",
        "timeout": 30,
        "prefer_grpc": prefer_grpc,
        "grpc_port": 7444,
    }


def test_low_memory_collection_creation_puts_dense_sparse_hnsw_and_payload_on_disk(
    monkeypatch,
):
    import core.storage.qdrant_store as qs

    client = _CollectionClient()
    store = _bare_store(client)
    store.collection_name = "code_chunks"
    monkeypatch.setattr(QdrantStore, "supports_sparse", property(lambda _self: True))
    monkeypatch.setattr(qs, "QDRANT_COLLECTION_ON_DISK", True, raising=False)
    monkeypatch.setattr(qs, "QDRANT_COLLECTION_MEMMAP_THRESHOLD", 5000, raising=False)
    monkeypatch.setattr(qs, "QDRANT_COLLECTION_MAX_OPTIMIZATION_THREADS", 1, raising=False)

    store._ensure_collection()

    assert client.create_args == ("code_chunks",)
    options = client.create_kwargs
    dense = options["vectors_config"]["dense"]
    sparse = options["sparse_vectors_config"]["sparse"]
    assert dense.on_disk is True
    assert dense.hnsw_config.on_disk is True
    assert sparse.index.on_disk is True
    assert options["on_disk_payload"] is True
    assert options["hnsw_config"].on_disk is True
    assert options["optimizers_config"].memmap_threshold == 5000
    assert options["optimizers_config"].max_optimization_threads == 1


def test_default_collection_creation_keeps_existing_in_memory_contract(monkeypatch):
    import core.storage.qdrant_store as qs

    client = _CollectionClient()
    store = _bare_store(client)
    monkeypatch.setattr(QdrantStore, "supports_sparse", property(lambda _self: True))
    monkeypatch.setattr(qs, "QDRANT_COLLECTION_ON_DISK", False, raising=False)

    store._ensure_collection()

    options = client.create_kwargs
    assert options["vectors_config"]["dense"].on_disk is None
    assert options["sparse_vectors_config"]["sparse"].index.on_disk is None
    assert "on_disk_payload" not in options
    assert "hnsw_config" not in options
    assert "optimizers_config" not in options
