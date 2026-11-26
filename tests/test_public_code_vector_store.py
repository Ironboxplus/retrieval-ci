"""Public-code vector storage stays a thin scoped wrapper over Core primitives."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.services.public_code.vector_store import PublicCodeVectorStore
from core.code_kb.vector_records import CodeChunkVectorRecord


def record() -> CodeChunkVectorRecord:
    return CodeChunkVectorRecord(
        rel_path="libavcodec/encode.c",
        language="c",
        start_line=20,
        end_line=34,
        label="encode_frame",
        content="# path: libavcodec/encode.c\nint encode_frame(void);",
        code="int encode_frame(void);",
    )


class FakeStore:
    supports_sparse = True

    def __init__(self):
        self.batches = []
        self.query_filter = None

    def _upsert_batch(self, points, *, wait):
        self.batches.append((points, wait))

    def hybrid_search(self, _dense, _sparse, *, top_k, prefetch_limit, query_filter):
        self.query_filter = query_filter
        return [
            {
                "score": 0.91,
                "content": "# path: libavcodec/encode.c\nint encode_frame(void);",
                "metadata": {
                    "source_id": "ffmpeg",
                    "revision": "ffmpeg-commit",
                    "domain": "multimedia",
                    "path": "libavcodec/encode.c",
                    "language": "c",
                    "start_line": 20,
                    "end_line": 34,
                    "label": "encode_frame",
                    "code": "int encode_frame(void);",
                },
            }
        ]


class MutationClient:
    def __init__(self, *, count=3, count_error=None):
        self.count_value = count
        self.count_error = count_error
        self.count_filters = []
        self.deletes = []

    def count(self, *, collection_name, count_filter, exact):
        if self.count_error is not None:
            raise self.count_error
        self.count_filters.append((collection_name, count_filter, exact))
        return SimpleNamespace(count=self.count_value)

    def delete(self, *, collection_name, points_selector, wait):
        self.deletes.append((collection_name, points_selector, wait))


class MutationStore:
    supports_sparse = True

    def __init__(self, client):
        self.client = client

    def _build_filter(self, filters):
        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filters.items()
            ]
        )


def test_upsert_uses_public_scope_and_wait_true_without_private_owner_fields():
    inner = FakeStore()
    store = PublicCodeVectorStore(store=inner)

    inserted = store.upsert(
        source_id="ffmpeg",
        revision="ffmpeg-commit",
        domain="multimedia",
        items=[record()],
        dense_vectors=[[0.1, 0.2]],
        sparse_vectors=[{"1": 2.0}],
    )

    assert inserted == 1
    points, wait = inner.batches[0]
    assert wait is True
    assert points[0].payload["source_id"] == "ffmpeg"
    assert points[0].payload["revision"] == "ffmpeg-commit"
    assert points[0].payload["domain"] == "multimedia"
    assert "owner_user_id" not in points[0].payload
    assert points[0].payload["path"] == "libavcodec/encode.c"


def test_hybrid_search_pushes_active_source_revision_pairs_into_qdrant():
    inner = FakeStore()
    store = PublicCodeVectorStore(store=inner)

    results = store.search(
        dense_vector=[0.1, 0.2],
        sparse_vector={"1": 2.0},
        mode="hybrid",
        top_k=5,
        active_revisions={"ffmpeg": "ffmpeg-commit", "pytorch": "pytorch-commit"},
        domains=("multimedia",),
        source_ids=("ffmpeg",),
        language="c",
    )

    assert results[0]["source_id"] == "ffmpeg"
    assert results[0]["revision"] == "ffmpeg-commit"
    assert len(inner.query_filter.should) == 1
    assert {condition.key for condition in inner.query_filter.must} == {
        "domain",
        "language",
    }


def test_search_with_no_active_revisions_returns_empty_without_querying_qdrant():
    inner = FakeStore()
    store = PublicCodeVectorStore(store=inner)

    assert store.search(
        dense_vector=[0.1],
        mode="dense",
        top_k=5,
        active_revisions={},
    ) == []
    assert inner.query_filter is None


def test_search_never_runs_payload_index_maintenance(monkeypatch):
    inner = FakeStore()
    store = PublicCodeVectorStore(store=inner)
    store._missing_search_indexes = {"source_id"}
    monkeypatch.setattr(
        store,
        "ensure_search_payload_indexes",
        lambda: (_ for _ in ()).throw(AssertionError("search performed schema maintenance")),
    )

    store.search(
        dense_vector=[0.1, 0.2],
        sparse_vector={"1": 2.0},
        mode="hybrid",
        top_k=5,
        active_revisions={"ffmpeg": "ffmpeg-commit"},
    )

    assert inner.query_filter is not None


def test_payload_index_maintenance_retries_until_all_indexes_exist():
    store = PublicCodeVectorStore(store=FakeStore(), maintenance_retry_seconds=0)
    attempts = []

    def ensure_indexes():
        attempts.append(len(attempts) + 1)
        store._missing_search_indexes = {"source_id"} if len(attempts) == 1 else set()

    store.ensure_search_payload_indexes = ensure_indexes

    store._maintain_payload_indexes()

    assert attempts == [1, 2]


def test_revision_count_is_exact_and_delete_waits_for_durable_completion():
    client = MutationClient(count=3)
    store = PublicCodeVectorStore(store=MutationStore(client))

    assert store.count_revision("ffmpeg", "commit-a") == 3
    assert store.delete_revision("ffmpeg", "commit-a") == 3

    collection_name, query_filter, exact = client.count_filters[0]
    assert collection_name == "public_code_chunks"
    assert exact is True
    assert {(condition.key, condition.match.value) for condition in query_filter.must} == {
        ("source_id", "ffmpeg"),
        ("revision", "commit-a"),
    }
    assert client.deletes[0][0] == "public_code_chunks"
    assert {
        (condition.key, condition.match.value)
        for condition in client.deletes[0][1].filter.must
    } == {
        ("source_id", "ffmpeg"),
        ("revision", "commit-a"),
    }
    assert client.deletes[0][2] is True


def test_revision_count_failure_is_not_silently_treated_as_zero(monkeypatch):
    store = PublicCodeVectorStore(
        store=MutationStore(MutationClient(count_error=RuntimeError("qdrant unavailable")))
    )
    monkeypatch.setattr("core.storage.qdrant_operations.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        store.count_revision("ffmpeg", "commit-a")
