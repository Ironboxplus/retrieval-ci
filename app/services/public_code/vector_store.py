"""Thin Qdrant wrapper for the shared public-code collection."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client import models
from qdrant_client.models import PointStruct

from core.code_kb.vector_records import (
    CodeChunkVectorRecord,
    build_code_chunk_payload,
    code_chunk_point_id,
)
from core.storage.qdrant_operations import (
    ensure_keyword_payload_indexes,
    retry_qdrant_operation,
)
from core.storage.qdrant_scope_filters import build_scope_pair_filter
from core.storage.qdrant_store import (
    QDRANT_COLLECTION_ON_DISK,
    QdrantStore,
    sparse_dict_to_qdrant,
)


PUBLIC_CODE_COLLECTION = "public_code_chunks"
# ``embed_fp`` is not a search filter: "which points are stale" is a must_not on it.
_SEARCH_FILTER_FIELDS = ("source_id", "revision", "domain", "path", "language", "embed_fp")


class PublicCodeVectorStore:
    """Store and retrieve public code without private owner/project semantics."""

    def __init__(
        self,
        collection_name: str = PUBLIC_CODE_COLLECTION,
        store: QdrantStore | None = None,
        maintenance_retry_seconds: float = 60.0,
    ):
        self.collection_name = collection_name
        self.store = store or QdrantStore(collection_name=collection_name)
        self._missing_search_indexes: set[str] = set()
        self._payload_index_maintenance_lock = threading.Lock()
        self._payload_index_maintenance_started = False
        self._maintenance_retry_seconds = maintenance_retry_seconds

    @property
    def supports_sparse(self) -> bool:
        return self.store.supports_sparse

    def ensure_search_payload_indexes(self) -> None:
        """Ensure every public search filter has a keyword payload index."""

        try:
            schema = self.store.client.get_collection(self.collection_name).payload_schema or {}
            self._missing_search_indexes = ensure_keyword_payload_indexes(
                client=self.store.client,
                collection_name=self.collection_name,
                field_names=_SEARCH_FILTER_FIELDS,
                retry=retry_qdrant_operation,
                existing_schema=schema,
                on_disk=QDRANT_COLLECTION_ON_DISK,
            )
        except Exception:
            self._missing_search_indexes = set(_SEARCH_FILTER_FIELDS)

    def _maintain_payload_indexes(self) -> None:
        """Retry schema maintenance in the background until every filter is indexed."""

        while True:
            self.ensure_search_payload_indexes()
            if not self._missing_search_indexes:
                return
            time.sleep(self._maintenance_retry_seconds)

    def start_payload_index_maintenance(self) -> None:
        """Start one best-effort schema-maintenance thread outside the search path."""

        with self._payload_index_maintenance_lock:
            if self._payload_index_maintenance_started:
                return
            self._payload_index_maintenance_started = True
        threading.Thread(
            target=self._maintain_payload_indexes,
            name="public-code-payload-index-maintenance",
            daemon=True,
        ).start()

    def upsert(
        self,
        *,
        source_id: str,
        revision: str,
        domain: str,
        items: Sequence[CodeChunkVectorRecord],
        dense_vectors: Sequence[list[float]],
        sparse_vectors: Sequence[dict[str, float]] | None = None,
    ) -> int:
        """Durably upsert one candidate or active source revision with ``wait=True``."""

        if len(items) != len(dense_vectors):
            raise ValueError("public code items and dense vectors must have equal lengths")

        scope = {"source_id": source_id, "revision": revision, "domain": domain}
        namespace = f"public:{source_id}:{revision}"
        points: list[PointStruct] = []
        for index, (item, dense_vector) in enumerate(zip(items, dense_vectors)):
            vector: Any = dense_vector
            if self.supports_sparse:
                vector = {"dense": dense_vector}
                if sparse_vectors and index < len(sparse_vectors) and sparse_vectors[index]:
                    vector["sparse"] = sparse_dict_to_qdrant(sparse_vectors[index])
            points.append(
                PointStruct(
                    id=code_chunk_point_id(namespace, item),
                    vector=vector,
                    payload=build_code_chunk_payload(item, scope=scope),
                )
            )

        for start in range(0, len(points), 256):
            self.store._upsert_batch(points[start:start + 256], wait=True)
        return len(points)

    def count_revision(self, source_id: str, revision: str) -> int:
        """Return the exact durable point count for one candidate revision."""

        query_filter = self.store._build_filter(
            {"source_id": source_id, "revision": revision}
        )
        result = retry_qdrant_operation(
            f"count_public_code_revision(source={source_id} revision={revision})",
            lambda: self.store.client.count(
                collection_name=self.collection_name,
                count_filter=query_filter,
                exact=True,
            ),
        )
        return int(result.count)

    def delete_revision(self, source_id: str, revision: str) -> int:
        """Delete one candidate revision durably without touching active revisions."""

        query_filter = self.store._build_filter(
            {"source_id": source_id, "revision": revision}
        )
        existing_count = self.count_revision(source_id, revision)
        if existing_count == 0:
            return 0
        retry_qdrant_operation(
            f"delete_public_code_revision(source={source_id} revision={revision})",
            lambda: self.store.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True,
            ),
        )
        return existing_count

    def search(
        self,
        *,
        dense_vector: list[float],
        top_k: int,
        active_revisions: Mapping[str, str],
        sparse_vector: dict[str, float] | None = None,
        mode: str = "hybrid",
        source_ids: Sequence[str] = (),
        domains: Sequence[str] = (),
        language: str | None = None,
        paths: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search only active source/revision pairs with all filters pushed to Qdrant."""

        selected_revisions = dict(active_revisions)
        if source_ids:
            selected_source_ids = set(source_ids)
            selected_revisions = {
                source_id: revision
                for source_id, revision in selected_revisions.items()
                if source_id in selected_source_ids
            }
        if not selected_revisions:
            return []

        payload_filters: dict[str, Any] = {}
        if domains:
            payload_filters["domain"] = list(domains)
        if language:
            payload_filters["language"] = language
        if paths is not None:
            payload_filters["path"] = list(paths)
        query_filter = build_scope_pair_filter(
            tuple(
                {"source_id": source_id, "revision": revision}
                for source_id, revision in sorted(selected_revisions.items())
            ),
            payload_filters=payload_filters,
        )

        if mode == "hybrid" and sparse_vector is not None:
            raw = self.store.hybrid_search(
                dense_vector,
                sparse_dict_to_qdrant(sparse_vector),
                top_k=top_k,
                prefetch_limit=max(top_k * 4, 50),
                query_filter=query_filter,
            )
        elif mode == "keyword" and sparse_vector is not None:
            raw = self.store.sparse_search(
                sparse_dict_to_qdrant(sparse_vector),
                top_k=top_k,
                query_filter=query_filter,
            )
        else:
            raw = self.store.search(
                dense_vector,
                top_k=top_k,
                query_filter=query_filter,
            )
        return [_public_code_result(item) for item in raw]


def _public_code_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    metadata = raw.get("metadata", {})
    return {
        "score": raw.get("score", 0.0),
        "source_id": metadata.get("source_id", ""),
        "revision": metadata.get("revision", ""),
        "domain": metadata.get("domain", ""),
        "path": metadata.get("path", ""),
        "language": metadata.get("language", ""),
        "start_line": metadata.get("start_line", 0),
        "end_line": metadata.get("end_line", 0),
        "label": metadata.get("label", ""),
        "content": raw.get("content", ""),
        "code": metadata.get("code", ""),
    }
