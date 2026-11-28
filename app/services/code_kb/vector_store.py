"""Qdrant-backed vector index for code-KB chunks (4090 `code_chunks` collection).

Mirrors app/services/journal/vector_store.JournalVectorStore: a thin wrapper over
the shared QdrantStore that keeps code payloads out of the paper schema and scopes
every operation to (owner_user_id, project) for multi-tenant isolation. `host` is
carried in the payload for display only — never a scoping key, so re-indexing the
same project from a renamed/second machine cleanly replaces it instead of leaving
orphans or cross-host duplicates (W148).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Sequence

from qdrant_client import models
from qdrant_client.models import PointStruct

from core.code_kb.vector_records import (
    CodeChunkVectorRecord,
    build_code_chunk_payload,
    code_chunk_identity_id,
)
from core.storage.qdrant_operations import (
    ensure_keyword_payload_indexes,
    retry_qdrant_operation,
)
from core.storage.qdrant_store import QdrantStore, sparse_dict_to_qdrant

logger = logging.getLogger("code_kb.vector_store")

# Deletes and scrolls get the SAME durability treatment as upserts (W4/W14): retry with
# backoff, then RAISE on permanent failure. Silently returning 0/∅ on a failed delete
# would let index() proceed to upsert and leave the old chunks of a changed/shrunk file
# as permanent orphans — the exact "index drifts from truth after an append" failure the
# incremental path must never cause (W157-inc). Failing loud makes the job FAILED instead.
_MAX_RETRIES = 5
_PATH_FACET_LIMIT = 100_000
_SEARCH_FILTER_FIELDS = ("owner_user_id", "project", "path", "language")
_PAYLOAD_INDEX_RETRY_SECONDS = 60.0


def _with_retry(what: str, fn: Callable[[], Any]) -> Any:
    return retry_qdrant_operation(what, fn, max_attempts=_MAX_RETRIES)


# Compatibility alias for the existing private Code KB service contract. The
# canonical chunk record lives in Core so public-code indexing uses the same
# path/line/content semantics without importing this private module.
CodeChunkPoint = CodeChunkVectorRecord


def point_id(
    owner: str,
    project: str,
    path: str,
    start: int,
    label: str,
    chunk_ordinal: int = 0,
) -> int:
    """Deterministic Qdrant point id for one chunk — a public cross-module contract
    (the fan-out sync in code_kb/service.py recomputes it to diff stale points)."""
    # host is intentionally NOT part of the id: the same chunk indexed from any
    # machine collapses to one point, so re-indexing overwrites in place (W148).
    return code_chunk_identity_id(
        f"{owner}:{project}",
        rel_path=path,
        start_line=start,
        label=label,
        chunk_ordinal=chunk_ordinal,
    )


class CodeVectorStore:
    def __init__(self, collection_name: str = "code_chunks", store: QdrantStore | None = None):
        self.collection_name = collection_name
        self.store = store or QdrantStore(collection_name=collection_name)
        self._payload_index_lock = threading.Lock()
        self._missing_search_indexes: set[str] = set()
        self._next_payload_index_retry_at = 0.0
        if store is None:
            self.ensure_search_payload_indexes()

    def ensure_search_payload_indexes(self) -> None:
        """Ensure code-search filters and path facets use payload indexes.

        The code collection is separate from the paper collection, so these indexes do
        not alter the paper payload contract. Existing indexes are left untouched.
        """
        with self._payload_index_lock:
            try:
                schema = self.store.client.get_collection(
                    self.collection_name).payload_schema or {}
            except Exception:
                self._missing_search_indexes = set(_SEARCH_FILTER_FIELDS)
                self._next_payload_index_retry_at = (
                    time.monotonic() + _PAYLOAD_INDEX_RETRY_SECONDS)
                logger.exception("[code_kb] could not inspect search payload indexes")
                return

            failed = ensure_keyword_payload_indexes(
                client=self.store.client,
                collection_name=self.collection_name,
                field_names=_SEARCH_FILTER_FIELDS,
                retry=_with_retry,
                existing_schema=schema,
            )
            self._missing_search_indexes = failed
            self._next_payload_index_retry_at = (
                time.monotonic() + _PAYLOAD_INDEX_RETRY_SECONDS if failed else 0.0)

    def _retry_missing_search_indexes(self) -> None:
        if (self._missing_search_indexes
                and time.monotonic() >= self._next_payload_index_retry_at):
            self.ensure_search_payload_indexes()

    @property
    def supports_sparse(self) -> bool:
        return self.store.supports_sparse

    def reusable_vectors(
        self, *, owner_user_id: str, project: str,
        items: Sequence[CodeChunkPoint], require_sparse: bool,
    ) -> list[tuple[list[float], dict[str, float] | None] | None]:
        """Read a bounded batch of exact chunk matches from one private scope: the
        private index reuses them for unchanged chunks, public import for its copies.

        Point identity alone is insufficient: edits overwrite the same id. Comparing
        every canonical payload field prevents publishing vectors for private edits.
        Lookup failures propagate so an upstream outage is never a false cache miss.
        """
        if not owner_user_id or not project:
            raise ValueError("private vector reuse requires both owner and project")
        if not items:
            return []
        ids = [point_id(owner_user_id, project, item.rel_path, item.start_line,
                        item.label, item.chunk_ordinal) for item in items]
        points = _with_retry(
            "read_private_vectors_for_public_import",
            lambda: self.store.client.retrieve(
                collection_name=self.collection_name, ids=ids,
                with_payload=True, with_vectors=True),
        )
        by_id = {point.id: point for point in points}
        result = []
        for item, identity in zip(items, ids):
            point = by_id.get(identity)
            expected = build_code_chunk_payload(
                item, scope={"owner_user_id": owner_user_id, "project": project})
            if point is None or any((point.payload or {}).get(key) != value
                                    for key, value in expected.items()):
                result.append(None)
                continue
            vector = point.vector
            dense = vector.get("dense") if isinstance(vector, dict) else vector
            sparse = vector.get("sparse") if isinstance(vector, dict) else None
            if not dense or (require_sparse and sparse is None):
                result.append(None)
                continue
            lexical = ({str(index): value for index, value
                        in zip(sparse.indices, sparse.values)} if sparse is not None else None)
            result.append((list(dense), lexical))
        return result

    def upsert(self, *, owner_user_id: str, project: str, host: str,
               items: Sequence[CodeChunkPoint], dense_vectors: list[list[float]],
               sparse_vectors: Sequence[dict[str, float]] | None = None) -> int:
        points: list[PointStruct] = []
        for i, (it, vec) in enumerate(zip(items, dense_vectors)):
            payload = build_code_chunk_payload(
                it,
                scope={
                    "owner_user_id": owner_user_id,
                    "project": project,
                    "host": host,
                },
            )
            if self.store.supports_sparse:
                vector: Any = {"dense": vec}
                if sparse_vectors and i < len(sparse_vectors) and sparse_vectors[i]:
                    vector["sparse"] = sparse_dict_to_qdrant(sparse_vectors[i])
            else:
                vector = vec
            points.append(PointStruct(
                id=point_id(
                    owner_user_id, project, it.rel_path, it.start_line, it.label,
                    it.chunk_ordinal,
                ),
                vector=vector, payload=payload,
            ))
        for start in range(0, len(points), 256):
            self.store._upsert_batch(points[start:start + 256], wait=True)
        return len(points)

    def _count_then_delete(self, qf) -> int:
        """Count what a filter matches, then delete it (wait=True), returning the count —
        measured with the exact delete filter so it's accurate under concurrent writers
        and O(matched), not 2×O(collection) (W149). Wrap in _with_retry at the call site."""
        removed = self.store.client.count(self.collection_name, count_filter=qf).count
        self.store.client.delete(
            self.collection_name, points_selector=models.FilterSelector(filter=qf), wait=True)
        return removed

    def delete_project(self, *, owner_user_id: str, project: str) -> int:
        """Remove every point for (owner, project). Retries then RAISES on permanent
        failure (never a silent partial wipe)."""
        qf = self.store._build_filter({"owner_user_id": owner_user_id, "project": project})
        return _with_retry(f"delete_project(owner={owner_user_id} project={project})",
                           lambda: self._count_then_delete(qf))

    def list_paths(self, *, owner_user_id: str, project: str | Sequence[str]) -> set[str]:
        """The distinct file paths currently indexed for (owner, project) — used by
        incremental indexing to find files that were deleted from the repo (W157-inc).
        Uses Qdrant's path facet so work scales with files, not chunks. Missing facet
        support fails visibly instead of hiding a full chunk scan inside a request."""
        qf = self.store._build_filter({"owner_user_id": owner_user_id, "project": project})

        return self._facet_paths(
            qf,
            operation=f"list_paths(owner={owner_user_id} project={project})",
        )

    def list_owner_paths(self, *, owner_user_id: str) -> set[str]:
        """Distinct relative paths across an owner's projects for cross-project search."""
        qf = self.store._build_filter({"owner_user_id": owner_user_id})
        return self._facet_paths(
            qf,
            operation=f"list_owner_paths(owner={owner_user_id})",
        )

    def _facet_paths(self, qf, *, operation: str) -> set[str]:
        self._retry_missing_search_indexes()
        return self._facet_values("path", qf, operation=operation)

    def _facet_values(self, key: str, qf, *, operation: str) -> set[str]:
        def _do() -> set[str]:
            response = self.store.client.facet(
                collection_name=self.collection_name,
                key=key,
                facet_filter=qf,
                limit=_PATH_FACET_LIMIT,
                exact=True,
            )
            values = {str(hit.value) for hit in response.hits if hit.value}
            if len(values) >= _PATH_FACET_LIMIT:
                raise RuntimeError(
                    f"{operation} reached its {_PATH_FACET_LIMIT}-value safety limit")
            return values

        return _with_retry(operation, _do)

    def list_owner_files(self, *, owner_user_id: str) -> set[tuple[str, str]]:
        """Distinct (project, path) pairs indexed for this owner across ALL projects —
        the cross-scope fan-out (方案2) uses these to find the same physical file
        (matched by absolute path = project/path) indexed under another scope. It facets
        projects and then paths per project, never scanning code chunks."""
        self._retry_missing_search_indexes()
        owner_filter = self.store._build_filter({"owner_user_id": owner_user_id})
        projects = self._facet_values(
            "project", owner_filter,
            operation=f"list_owner_projects(owner={owner_user_id})")
        out: set[tuple[str, str]] = set()
        for project in projects:
            project_filter = self.store._build_filter(
                {"owner_user_id": owner_user_id, "project": project})
            paths = self._facet_values(
                "path", project_filter,
                operation=(f"list_owner_project_paths(owner={owner_user_id} "
                           f"project={project})"))
            out.update((project, path) for path in paths)
        return out

    def delete_paths(self, *, owner_user_id: str, project: str, paths: set[str]) -> int:
        """Remove points for specific files within (owner, project). Returns count removed.
        No-op on an empty path set (never a whole-project wipe by accident). Retries then
        RAISES on permanent failure so a failed purge can't be mistaken for success."""
        if not paths:
            return 0
        qf = self.store._build_filter(
            {"owner_user_id": owner_user_id, "project": project, "path": list(paths)})
        return _with_retry(
            f"delete_paths(owner={owner_user_id} project={project} n={len(paths)})",
            lambda: self._count_then_delete(qf))

    def list_point_ids(self, *, owner_user_id: str, project: str, path: str) -> set[int]:
        """The Qdrant point ids currently stored for one file (owner, project, path) — used
        by the cross-scope fan-out to prune only the now-stale points AFTER upserting the
        new ones (upsert-first, so a mid-op failure can never data-loss-empty the file).
        Retries then RAISES on permanent failure."""
        return self.list_point_ids_for_paths(
            owner_user_id=owner_user_id,
            project=project,
            paths={path},
        )

    def list_point_ids_for_paths(
        self,
        *,
        owner_user_id: str,
        project: str,
        paths: set[str] | None = None,
    ) -> set[int]:
        """Return current point ids for a project snapshot or a selected path set.

        ``paths=None`` means the whole project. An empty set is always a no-op. Indexing
        uses this write-path snapshot to upsert first and prune stale ids only after the
        new batch succeeds, so a failed upsert cannot empty the previous index.
        """
        if paths == set():
            return set()
        filters: dict[str, Any] = {
            "owner_user_id": owner_user_id,
            "project": project,
        }
        if paths is not None:
            filters["path"] = list(paths)
        qf = self.store._build_filter(filters)

        def _do() -> set[int]:
            ids: set[int] = set()
            offset = None
            while True:
                rows, offset = self.store.client.scroll(
                    self.collection_name, scroll_filter=qf, limit=1000,
                    offset=offset, with_payload=False, with_vectors=False)
                for r in rows:
                    ids.add(r.id)
                if offset is None:
                    return ids
        scope = "all" if paths is None else str(len(paths))
        return _with_retry(
            f"list_point_ids_for_paths(owner={owner_user_id} project={project} paths={scope})",
            _do,
        )

    def delete_point_ids(self, ids: set[int]) -> int:
        """Delete specific points by id (no-op on empty). Retries then RAISES on permanent
        failure. Used by the fan-out to drop only the stale points of a resynced file."""
        if not ids:
            return 0
        id_list = list(ids)

        def _do() -> int:
            self.store.client.delete(
                self.collection_name,
                points_selector=models.PointIdsList(points=id_list), wait=True)
            return len(id_list)
        return _with_retry(f"delete_point_ids(n={len(id_list)})", _do)

    def search(self, *, dense_vector: list[float], top_k: int, owner_user_id: str,
               project: str | Sequence[str] | None = None, language: str | None = None,
               sparse_vector: dict[str, float] | None = None,
               mode: str = "dense", paths: list[str] | None = None,
               excluded_projects: Sequence[str] | None = None) -> list[dict]:
        filters: dict[str, Any] = {"owner_user_id": owner_user_id}
        if project:
            filters["project"] = project
        if language:
            filters["language"] = language  # exact keyword match, indexed payload field
        if paths is not None:
            # path_glob push-down: the service resolved the glob to this exact path set,
            # pushed here as a MatchAny on `path` (a list → MatchAny in _build_filter) so
            # top_k fills ONLY from matching files inside Qdrant — never a post-filter that
            # could drop a match ranked below the fetch cap. Empty list = match nothing.
            filters["path"] = paths
        if excluded_projects:
            # Federated-search decision 10 (per-scope exclusion): a `must_not` on
            # `project`, not an allow-list, because a scope indexed before the catalog
            # marker existed has no durable row and would be silently dropped by a
            # positive filter. QdrantStore._build_filter's reserved "__exclude__" key
            # is the same mechanism `papers` already uses to drop review points.
            filters["__exclude__"] = {"project": list(dict.fromkeys(excluded_projects))}
        # mode routes to the shared QdrantStore's dense / RRF-hybrid / sparse retrievers.
        # The service already downgrades to "dense" when no sparse query vector exists,
        # so here hybrid/keyword can assume sparse_vector is present.
        if mode == "hybrid" and sparse_vector is not None:
            raw = self.store.hybrid_search(
                dense_vector, sparse_dict_to_qdrant(sparse_vector), top_k=top_k,
                prefetch_limit=max(top_k * 4, 50), filter_dict=filters)
        elif mode == "keyword" and sparse_vector is not None:
            raw = self.store.sparse_search(
                sparse_dict_to_qdrant(sparse_vector), top_k=top_k, filter_dict=filters)
        else:
            raw = self.store.search(dense_vector, top_k=top_k, filter_dict=filters)
        # QdrantStore returns {content, metadata:{...payload}, score}; flatten to the
        # same shape the service (and the in-memory test store) consume.
        out: list[dict] = []
        for r in raw:
            md = r.get("metadata", {})
            out.append({
                "score": r.get("score", 0.0),
                "path": md.get("path", ""), "language": md.get("language"),
                "start_line": md.get("start_line", 0), "end_line": md.get("end_line", 0),
                "label": md.get("label", ""), "project": md.get("project", ""),
                "host": md.get("host", ""), "code": md.get("code", ""),
            })
        return out

    def count(self, *, owner_user_id: str | None = None, project: str | None = None) -> int:
        filters: dict[str, Any] = {}
        if owner_user_id:
            filters["owner_user_id"] = owner_user_id
        if project:
            filters["project"] = project
        try:
            qf = self.store._build_filter(filters) if filters else None
            return self.store.client.count(self.collection_name, count_filter=qf).count
        except Exception:
            logger.exception("[code_kb] count failed")
            return 0
