"""Qdrant-backed vector index for ACE journal chunks."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import models
from qdrant_client.models import PointStruct

from app.services.journal.chunker import JournalTextChunk
from core.storage.qdrant_store import QdrantStore, sparse_dict_to_qdrant


def _point_id(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


@dataclass(frozen=True)
class JournalChunkUpsert:
    """One scope-local vector payload group within a multi-scope write.

    Chunks in different groups may share the same dense/sparse vector objects when
    they came from one physical file, but their entry/chunk IDs and Qdrant payload
    fields remain scope-local.  Keeping that distinction at the vector boundary
    makes a fan-out write one durable Qdrant batch rather than N independent calls.
    """

    chunks: list[JournalTextChunk]
    project_id: int
    project_slug: str
    entry_id: str
    owner_user_id: str | None
    client: str
    source_kind: str
    tags: list[str]
    dense_vectors: list[list[float]]
    sparse_vectors: list[dict[str, float]] | None = None


class JournalVectorStore:
    """Journal-specific wrapper that keeps payloads out of the paper schema."""

    def __init__(self, collection_name: str | None = None, store: QdrantStore | None = None):
        # Production remains on the established collection.  A named environment
        # override exists solely for isolated lifecycle/maintenance processes; it
        # prevents a real test server from ever opening the normal Journal collection.
        self.collection_name = collection_name or os.environ.get(
            "JOURNAL_QDRANT_COLLECTION", "journal_chunks")
        self.store = store or QdrantStore(collection_name=self.collection_name)

    def upsert_chunks(
        self,
        chunks: list[JournalTextChunk],
        *,
        project_id: int,
        project_slug: str,
        entry_id: str,
        owner_user_id: str | None,
        client: str,
        source_kind: str,
        tags: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[str, float]] | None = None,
    ) -> int:
        return self.upsert_chunk_groups([JournalChunkUpsert(
            chunks=chunks,
            project_id=project_id,
            project_slug=project_slug,
            entry_id=entry_id,
            owner_user_id=owner_user_id,
            client=client,
            source_kind=source_kind,
            tags=tags,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )])

    def upsert_chunk_groups(self, groups: Sequence[JournalChunkUpsert]) -> int:
        """Durably upsert many scope-local groups in Qdrant batches.

        A Journal bulk fan-out embeds one physical source once, then supplies the
        resulting vectors to several entries.  We flatten those entries here so the
        real Qdrant boundary performs 256-point ``wait=True`` batches across the
        whole operation, while every point still carries its own scope/entry payload.
        Any failure propagates to the service before latest-wins promotion runs.
        """
        points: list[PointStruct] = []
        for group in groups:
            points.extend(self._points_for_group(group))
        for start in range(0, len(points), 256):
            self.store._upsert_batch(points[start:start + 256], wait=True)
        return len(points)

    def _points_for_group(self, group: JournalChunkUpsert) -> list[PointStruct]:
        points: list[PointStruct] = []
        chunks = group.chunks
        for i, chunk in enumerate(chunks):
            if i >= len(group.dense_vectors):
                continue
            payload = {
                "project_id": group.project_id,
                "project_slug": group.project_slug,
                "owner_user_id": group.owner_user_id or "",
                "entry_id": group.entry_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "source_kind": group.source_kind,
                "client": group.client,
                "level": chunk.level,
                "heading": chunk.heading or "",
                "token_count": chunk.token_count,
                "tags": group.tags,
                "content": chunk.content,
            }
            if (group.sparse_vectors and i < len(group.sparse_vectors)
                    and group.sparse_vectors[i]):
                vector: Any = {
                    "dense": group.dense_vectors[i],
                    "sparse": sparse_dict_to_qdrant(group.sparse_vectors[i]),
                }
            elif self.store.supports_sparse:
                vector = {"dense": group.dense_vectors[i]}
            else:
                vector = group.dense_vectors[i]
            points.append(PointStruct(id=_point_id(chunk.chunk_id), vector=vector, payload=payload))
        return points

    def search(
        self,
        *,
        dense_vector: list[float],
        sparse_vector: dict[str, float] | None,
        mode: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if mode == "sparse" and sparse_vector:
            return self.store.sparse_search(
                sparse_dict_to_qdrant(sparse_vector),
                top_k=top_k,
                filter_dict=filters,
            )
        if mode == "hybrid" and sparse_vector:
            return self.store.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_dict_to_qdrant(sparse_vector),
                top_k=top_k,
                prefetch_limit=max(top_k * 5, 20),
                filter_dict=filters,
            )
        return self.store.search(dense_vector, top_k=top_k, filter_dict=filters)

    def delete_by_entry_id(self, entry_id: str) -> int:
        return self.delete_by_entry_ids([entry_id])

    def delete_by_entry_ids(self, entry_ids: list[str]) -> int:
        """One durable MatchAny delete for reconciliation/latest-wins batches."""
        if not entry_ids:
            return 0
        # Count matches with the SAME filter we delete by, before deleting: accurate
        # even without wait= and unaffected by concurrent writes to other entries, and
        # O(matched) instead of 2×O(collection) (mirrors CodeVectorStore.delete_project).
        try:
            qf = self.store._build_filter({"entry_id": entry_ids})
            removed = self.store.client.count(self.collection_name, count_filter=qf).count
            delay = 0.1
            for attempt in range(1, 4):
                try:
                    self.store.client.delete(
                        self.collection_name,
                        points_selector=models.FilterSelector(filter=qf), wait=True,
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(delay)
                    delay *= 2
            return removed
        except Exception:
            return 0
