"""The journal collection indexes the payload keys its searches filter on.

``journal_chunks`` had no payload index at all, so a subdirectory search (``entry_id``
MatchAny OR NOT ``source_kind=doc``) made Qdrant scan the whole collection. Real Qdrant on
loopback, a test-private scratch collection. The only stand-in is a thin client wrapper that
delegates every call to the real server and can fail ``create_payload_index`` on demand.
"""
from __future__ import annotations

import time
import uuid
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.services.journal import vector_store as vector_store_module  # noqa: E402
from app.services.journal.vector_store import JournalVectorStore  # noqa: E402
from core.storage.qdrant_store import QDRANT_URL, QdrantStore  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
DIM = 8
FIELDS = {"project_slug", "owner_user_id", "entry_id", "source_kind", "client", "embed_fp"}


class _Client:
    """The real client, recording index creations and failing the chosen fields."""

    def __init__(self, real, *, fail_fields=()):
        self._real = real
        self.fail_fields = set(fail_fields)
        self.created: list[str] = []
        self.attempted: list[str] = []

    def create_payload_index(self, *args, **kwargs):
        field = kwargs["field_name"]
        self.attempted.append(field)
        if field in self.fail_fields:
            raise ConnectionError(f"qdrant refused the {field} index")
        self.created.append(field)
        return self._real.create_payload_index(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _drop(client, name):
    for attempt in range(3):
        try:
            client.delete_collection(name)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                warnings.warn(f"could not drop scratch collection {name}: {exc}")
            time.sleep(0.5 * (attempt + 1))


@pytest.fixture()
def real_client():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=60)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    return client


def _create_collection(client, name):
    from config import settings

    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params)


@pytest.fixture()
def collection(real_client):
    name = f"journal_payload_idx_{uuid.uuid4().hex[:8]}_test"
    _create_collection(real_client, name)
    try:
        yield name
    finally:
        _drop(real_client, name)


def _store(real_client, name, **client_options):
    client = _Client(real_client, **client_options)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = name
    inner.client = client
    return JournalVectorStore(collection_name=name, store=inner), client


def _schema(real_client, name):
    return real_client.get_collection(name).payload_schema or {}


@pytest.fixture()
def fast_retries(monkeypatch):
    """Same retry policy, without the real backoff sleeps."""
    real = vector_store_module.retry_qdrant_operation
    monkeypatch.setattr(
        vector_store_module, "retry_qdrant_operation",
        lambda operation, run, *, max_attempts: real(
            operation, run, max_attempts=max_attempts, initial_delay_seconds=0.0))


def _seed(real_client, name):
    from config import settings

    named = settings.embedding.supports_sparse
    rows = [
        ("doc-in", "doc", "p", "o"), ("doc-out", "doc", "p", "o"),
        ("memory", "memory", "p", "o"), ("handoff", "handoff", "p", "o"),
        ("memory-other-project", "memory", "q", "o"), ("memory-other-owner", "memory", "p", "x"),
    ]
    real_client.upsert(name, wait=True, points=[
        qmodels.PointStruct(
            id=index + 1,
            vector={"dense": [1.0, 0.1 * index] + [0.0] * (DIM - 2)} if named
            else [1.0, 0.1 * index] + [0.0] * (DIM - 2),
            payload={"entry_id": entry, "source_kind": kind, "project_slug": project,
                     "owner_user_id": owner, "client": "claude", "chunk_id": entry,
                     "content": entry})
        for index, (entry, kind, project, owner) in enumerate(rows)])


def _subdirectory_search(store):
    hits = store.search(
        dense_vector=[1.0, 0.0] + [0.0] * (DIM - 2), sparse_vector=None, mode="dense",
        top_k=20,
        filters={"project_slug": "p", "owner_user_id": "o",
                 "__should__": [{"entry_id": ["doc-in"]}, {"__exclude__": {"source_kind": "doc"}}]})
    return sorted(hit["metadata"]["entry_id"] for hit in hits)


def test_missing_indexes_are_created_on_an_existing_collection(real_client, collection):
    store, client = _store(real_client, collection)
    assert _schema(real_client, collection) == {}

    store.ensure_search_payload_indexes()

    schema = _schema(real_client, collection)
    assert set(schema) == FIELDS
    assert all(str(info.data_type).lower().endswith("keyword") for info in schema.values())
    assert set(client.created) == FIELDS
    assert store._missing_search_indexes == set()


def test_a_second_run_is_a_no_op(real_client, collection):
    store, client = _store(real_client, collection)
    store.ensure_search_payload_indexes()
    before = _schema(real_client, collection)
    client.attempted.clear()

    store.ensure_search_payload_indexes()
    again, _ = _store(real_client, collection)  # a restarted process finds them too
    again.ensure_search_payload_indexes()

    assert client.attempted == []
    assert _schema(real_client, collection) == before


def test_only_the_missing_indexes_are_created(real_client, collection):
    real_client.create_payload_index(
        collection, field_name="project_slug",
        field_schema=qmodels.PayloadSchemaType.KEYWORD, wait=True)
    store, client = _store(real_client, collection)

    store.ensure_search_payload_indexes()

    assert set(client.created) == FIELDS - {"project_slug"}
    assert set(_schema(real_client, collection)) == FIELDS


def test_a_failed_index_is_retried_by_a_later_search_and_never_blocks_it(
        real_client, collection, fast_retries, monkeypatch):
    _seed(real_client, collection)
    store, client = _store(real_client, collection, fail_fields={"entry_id"})

    store.ensure_search_payload_indexes()  # must not raise

    assert store._missing_search_indexes == {"entry_id"}
    assert set(_schema(real_client, collection)) == FIELDS - {"entry_id"}
    attempts = list(client.attempted)

    # A search inside the back-off window neither retries nor fails.
    assert _subdirectory_search(store) == ["doc-in", "handoff", "memory"]
    assert client.attempted == attempts

    # Once the window has passed and Qdrant accepts the index, the next search builds it.
    client.fail_fields.clear()
    store._next_payload_index_retry_at = 0.0
    assert _subdirectory_search(store) == ["doc-in", "handoff", "memory"]
    assert client.attempted == attempts + ["entry_id"]
    assert set(_schema(real_client, collection)) == FIELDS
    assert store._missing_search_indexes == set()

    client.attempted.clear()
    _subdirectory_search(store)
    assert client.attempted == []


def test_an_unreadable_collection_does_not_raise_and_is_retried(
        real_client, collection, fast_retries):
    store, client = _store(real_client, collection)

    class Unreachable(_Client):
        def get_collection(self, *_args, **_kwargs):
            raise ConnectionError("qdrant unreachable")

    store.store.client = Unreachable(real_client)
    store.ensure_search_payload_indexes()  # must not raise
    assert store._missing_search_indexes == FIELDS

    store.store.client = client
    store._next_payload_index_retry_at = 0.0
    store._retry_missing_search_indexes()
    assert set(_schema(real_client, collection)) == FIELDS


def test_search_results_do_not_change_when_the_indexes_appear(real_client, collection):
    _seed(real_client, collection)
    store, _client = _store(real_client, collection)
    before = {
        "subdirectory": _subdirectory_search(store),
        "scope": sorted(h["metadata"]["entry_id"] for h in store.search(
            dense_vector=[1.0, 0.0] + [0.0] * (DIM - 2), sparse_vector=None, mode="dense",
            top_k=20, filters={"project_slug": "p", "owner_user_id": "o"})),
        "docs": sorted(h["metadata"]["entry_id"] for h in store.search(
            dense_vector=[1.0, 0.0] + [0.0] * (DIM - 2), sparse_vector=None, mode="dense",
            top_k=20, filters={"project_slug": "p", "source_kind": "doc", "client": "claude"})),
    }
    assert before["subdirectory"] == ["doc-in", "handoff", "memory"]

    store.ensure_search_payload_indexes()

    after = {
        "subdirectory": _subdirectory_search(store),
        "scope": sorted(h["metadata"]["entry_id"] for h in store.search(
            dense_vector=[1.0, 0.0] + [0.0] * (DIM - 2), sparse_vector=None, mode="dense",
            top_k=20, filters={"project_slug": "p", "owner_user_id": "o"})),
        "docs": sorted(h["metadata"]["entry_id"] for h in store.search(
            dense_vector=[1.0, 0.0] + [0.0] * (DIM - 2), sparse_vector=None, mode="dense",
            top_k=20, filters={"project_slug": "p", "source_kind": "doc", "client": "claude"})),
    }
    assert after == before


def test_a_store_the_service_builds_indexes_its_collection_at_startup(
        real_client, monkeypatch):
    if "127.0.0.1" not in QDRANT_URL and "localhost" not in QDRANT_URL:
        pytest.skip(f"default Qdrant {QDRANT_URL} is not the local test server")
    name = f"journal_payload_idx_{uuid.uuid4().hex[:8]}_test"
    monkeypatch.setenv("JOURNAL_QDRANT_COLLECTION", name)
    try:
        store = JournalVectorStore()
        assert store.collection_name == name
        assert set(_schema(real_client, name)) == FIELDS
    finally:
        _drop(real_client, name)
