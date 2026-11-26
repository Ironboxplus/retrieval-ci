"""``QdrantStore._build_filter`` ``__should__``: an OR over independent sub-filters.

Journal subdirectory search needs "doc entries in this id set OR any non-doc entry", which
neither ``must`` (AND) nor ``__exclude__`` (NOT) can say. Real Qdrant on loopback, a
test-private scratch collection.
"""
from __future__ import annotations

import time
import uuid
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from core.storage.qdrant_store import QdrantStore  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
DIM = 4


@pytest.fixture()
def store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    from config import settings

    name = f"should_filter_{uuid.uuid4().hex[:8]}_test"
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    named = settings.embedding.supports_sparse
    client.create_collection(
        collection_name=name, vectors_config={"dense": params} if named else params)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = name
    inner.client = client
    rows = [
        ("doc-in", "doc", "p"), ("doc-out", "doc", "p"), ("memory", "memory", "p"),
        ("handoff", "handoff", "p"), ("memory-other-project", "memory", "q"),
    ]
    vector = [1.0, 0.0, 0.0, 0.0]
    client.upsert(name, wait=True, points=[
        qmodels.PointStruct(
            id=index + 1, vector={"dense": vector} if named else vector,
            payload={"entry_id": entry, "source_kind": kind, "project_slug": project})
        for index, (entry, kind, project) in enumerate(rows)])
    try:
        yield inner
    finally:
        for attempt in range(3):
            try:
                client.delete_collection(name)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    warnings.warn(f"could not drop scratch collection {name}: {exc}")
                time.sleep(0.5 * (attempt + 1))


def _entries(store, filter_dict):
    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=20, filter_dict=filter_dict)
    return sorted(hit["metadata"]["entry_id"] for hit in hits)


def test_should_is_an_or_between_sub_filters_and_and_with_the_rest(store):
    assert _entries(store, {
        "project_slug": "p",
        "__should__": [{"entry_id": ["doc-in"]}, {"__exclude__": {"source_kind": "doc"}}],
    }) == ["doc-in", "handoff", "memory"]


def test_should_with_one_branch_behaves_like_that_branch(store):
    assert _entries(store, {
        "project_slug": "p", "__should__": [{"entry_id": ["doc-in", "doc-out"]}],
    }) == ["doc-in", "doc-out"]


def test_without_should_the_filter_is_unchanged(store):
    assert _entries(store, {"project_slug": "p", "source_kind": "doc"}) == [
        "doc-in", "doc-out"]


def test_should_alone_is_a_complete_filter(store):
    assert _entries(store, {
        "__should__": [{"entry_id": ["memory-other-project"]}, {"entry_id": ["doc-out"]}],
    }) == ["doc-out", "memory-other-project"]


def test_an_empty_should_list_adds_no_condition(store):
    assert store._build_filter({"__should__": []}) is None
    assert _entries(store, {"project_slug": "p", "__should__": []}) == [
        "doc-in", "doc-out", "handoff", "memory"]
