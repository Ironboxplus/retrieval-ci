"""Qdrant-backed vector index for ACE journal chunks."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from qdrant_client import models
from qdrant_client.models import PointStruct

from app.services.journal.chunker import JournalTextChunk
from core.indexing.embedding_profile import current_embed_fp
from core.storage.qdrant_operations import (
    ensure_keyword_payload_indexes,
    retry_qdrant_operation,
)
from core.storage.qdrant_store import (
    QDRANT_COLLECTION_ON_DISK,
    QdrantStore,
    sparse_dict_to_qdrant,
)

logger = logging.getLogger("journal.vector_store")

# Every field a journal search filters on (service.py::_search_in_scope, delete_by_entry_ids).
# Without these indexes Qdrant answers a filtered search by scanning the collection: a
# subdirectory search (entry_id MatchAny OR NOT source_kind=doc) took 3.7 s on 180K points.
# ``embed_fp`` is not a search filter: "which points are stale" is a must_not on it.
_SEARCH_FILTER_FIELDS = ("project_slug", "owner_user_id", "entry_id", "source_kind", "client", "embed_fp")
_PAYLOAD_INDEX_RETRY_SECONDS = 60.0
_MAX_RETRIES = 5


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


def journal_points(group: JournalChunkUpsert) -> list[tuple[int, dict[str, Any], str]]:
    """``(point id, payload, embedded text)`` of every chunk of ``group``, in chunk order.

    The one place a journal chunk becomes a Qdrant point: the live write path
    (``JournalVectorStore._points_for_group``) and the re-embed pipeline's journal rebuild both
    take identity and payload from here, so a rebuilt point is the point a fresh write produces.
    """
    fingerprint = current_embed_fp()
    points = []
    for chunk in group.chunks:
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
            "embed_fp": fingerprint,
        }
        points.append((_point_id(chunk.chunk_id), payload, chunk.content))
    return points


class JournalVectorStore:
    """Journal-specific wrapper that keeps payloads out of the paper schema."""

    def __init__(self, collection_name: str | None = None, store: QdrantStore | None = None):
        # Production remains on the established collection.  A named environment
        # override exists solely for isolated lifecycle/maintenance processes; it
        # prevents a real test server from ever opening the normal Journal collection.
        self.collection_name = collection_name or os.environ.get(
            "JOURNAL_QDRANT_COLLECTION", "journal_chunks")
        self.store = store or QdrantStore(collection_name=self.collection_name)
        self._payload_index_lock = threading.Lock()
        self._missing_search_indexes: set[str] = set()
        self._next_payload_index_retry_at = 0.0
        if store is None:
            self.ensure_search_payload_indexes()

    def ensure_search_payload_indexes(self) -> None:
        """Create the keyword payload indexes the journal filters use, if missing.

        Never raises: a failure is logged and the missing fields are retried by later
        searches (at most once per ``_PAYLOAD_INDEX_RETRY_SECONDS``), so neither startup
        nor a search depends on Qdrant accepting the schema change. An index that already
        exists is left untouched, which makes a repeat call a no-op.
        """
        with self._payload_index_lock:
            try:
                schema = self.store.client.get_collection(
                    self.collection_name).payload_schema or {}
            except Exception:
                self._missing_search_indexes = set(_SEARCH_FILTER_FIELDS)
                self._next_payload_index_retry_at = (
                    time.monotonic() + _PAYLOAD_INDEX_RETRY_SECONDS)
                logger.exception("[journal] could not inspect search payload indexes")
                return

            failed = ensure_keyword_payload_indexes(
                client=self.store.client,
                collection_name=self.collection_name,
                field_names=_SEARCH_FILTER_FIELDS,
                retry=lambda what, fn: retry_qdrant_operation(
                    what, fn, max_attempts=_MAX_RETRIES),
                existing_schema=schema,
                on_disk=QDRANT_COLLECTION_ON_DISK,
            )
            self._missing_search_indexes = failed
            self._next_payload_index_retry_at = (
                time.monotonic() + _PAYLOAD_INDEX_RETRY_SECONDS if failed else 0.0)

    def _retry_missing_search_indexes(self) -> None:
        if (self._missing_search_indexes
                and time.monotonic() >= self._next_payload_index_retry_at):
            self.ensure_search_payload_indexes()

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

    def upsert_chunk_groups(
        self,
        groups: Sequence[JournalChunkUpsert],
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> int:
        """Durably upsert many scope-local groups in Qdrant batches.

        A Journal bulk fan-out embeds one physical source once, then supplies the
        resulting vectors to several entries.  We flatten those entries here so the
        real Qdrant boundary performs 256-point ``wait=True`` batches across the
        whole operation, while every point still carries its own scope/entry payload.
        Any failure propagates to the service before latest-wins promotion runs.
        ``on_batch(written, total)`` runs after each durable batch so a long write
        can report liveness.
        """
        points: list[PointStruct] = []
        for group in groups:
            points.extend(self._points_for_group(group))
        for start in range(0, len(points), 256):
            batch = points[start:start + 256]
            self.store._upsert_batch(batch, wait=True)
            if on_batch is not None:
                on_batch(start + len(batch), len(points))
        return len(points)

    def _points_for_group(self, group: JournalChunkUpsert) -> list[PointStruct]:
        points: list[PointStruct] = []
        for i, (point_id, payload, _text) in enumerate(journal_points(group)):
            if i >= len(group.dense_vectors):
                continue
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
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
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
        self._retry_missing_search_indexes()
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

    @property
    def supports_sparse(self) -> bool:
        return self.store.supports_sparse

    def reusable_vectors_by_content(
        self, entry_ids: Sequence[str], texts: Sequence[str], *, require_sparse: bool,
    ) -> dict[str, tuple[list[float], dict[str, float] | None]]:
        """Stored vectors of ``entry_ids`` that can stand in for embedding ``texts``.

        A point qualifies only when its payload ``content`` equals a text byte for byte and
        its ``embed_fp`` is the current embedding profile's, so a vector from an edited chunk
        or an older model is never copied forward. Payload is scrolled first (no vectors);
        vectors are fetched, at most 256 at a time, only for the one matching point per text
        (the first seen). A point without a dense vector - or without a sparse one when
        ``require_sparse`` - is dropped. Returns ``{text: (dense, sparse | None)}``.
        Qdrant failures propagate after the usual retries: an outage is never a cache miss.
        """
        wanted = set(texts)
        ids = list(dict.fromkeys(entry_ids))
        if not ids or not wanted:
            return {}
        fingerprint = current_embed_fp()
        matched: dict[str, Any] = {}
        for start in range(0, len(ids), 256):
            qf = self.store._build_filter({"entry_id": ids[start:start + 256]})
            offset: Any = None
            while True:
                points, offset = retry_qdrant_operation(
                    f"journal_reuse_scroll(collection={self.collection_name})",
                    lambda offset=offset, qf=qf: self.store.client.scroll(
                        self.collection_name, scroll_filter=qf, limit=256, offset=offset,
                        with_payload=["content", "embed_fp"], with_vectors=False),
                    max_attempts=_MAX_RETRIES)
                for point in points:
                    payload = point.payload or {}
                    content = payload.get("content")
                    if (payload.get("embed_fp") != fingerprint or content not in wanted
                            or content in matched):
                        continue
                    matched[content] = point.id
                if offset is None:
                    break
        by_point_id = {point_id: text for text, point_id in matched.items()}
        found: dict[str, tuple[list[float], dict[str, float] | None]] = {}
        point_ids = list(by_point_id)
        for start in range(0, len(point_ids), 256):
            batch = point_ids[start:start + 256]
            points = retry_qdrant_operation(
                f"journal_reuse_retrieve(collection={self.collection_name} points={len(batch)})",
                lambda batch=batch: self.store.client.retrieve(
                    self.collection_name, ids=batch, with_payload=False, with_vectors=True),
                max_attempts=_MAX_RETRIES)
            for point in points:
                vector = point.vector
                dense = vector.get("dense") if isinstance(vector, dict) else vector
                sparse = vector.get("sparse") if isinstance(vector, dict) else None
                if not dense or (require_sparse and sparse is None):
                    continue
                lexical = ({str(index): value for index, value in zip(sparse.indices, sparse.values)}
                           if sparse is not None else None)
                found[by_point_id[point.id]] = (list(dense), lexical)
        return found

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
