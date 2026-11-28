"""
Qdrant 向量存储

替代 ChromaDB，解决并发读写和崩溃损坏问题。
接口与 ChromaStore 完全兼容。
"""
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Callable
import hashlib
import logging
import os
import time

from qdrant_client import QdrantClient, models

_logger = logging.getLogger(__name__)


class QdrantRetrievalUpstreamError(RuntimeError):
    """A Qdrant transport/query failure; never equivalent to a true no-match."""

    def __init__(self, mode: str, cause: Exception):
        super().__init__(f"Qdrant {mode} retrieval upstream failure")
        self.mode = mode
        self.cause = cause


class QdrantRetrievalCancelledError(RuntimeError):
    """The owning request/job was cancelled between bounded Qdrant RPCs."""


from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny,
    SparseVector, SparseVectorParams, SparseIndexParams,
    Prefetch, Fusion, FusionQuery, QuantizationSearchParams, SearchParams,
)

from config import settings
from app.models.schemas import Chunk
from core.storage.corpus_layout import (
    QDRANT_ARXIV_COLLECTION,
    check_paper_point,
    is_paper_collection,
    paper_collections,
)
from core.storage.papers_payload import project_papers_payload
from core.storage.qdrant_operations import retry_qdrant_operation


VECTOR_DIM = settings.embedding.dimension
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
QDRANT_PREFER_GRPC = os.environ.get(
    "QDRANT_PREFER_GRPC", "true").strip().lower() not in {"0", "false", "no", "off"}
QDRANT_COLLECTION_ON_DISK = os.environ.get(
    "QDRANT_COLLECTION_ON_DISK", "false"
).strip().lower() in {"1", "true", "yes", "on"}
QDRANT_COLLECTION_MEMMAP_THRESHOLD = int(os.environ.get(
    "QDRANT_COLLECTION_MEMMAP_THRESHOLD", "5000"
))
QDRANT_COLLECTION_MAX_OPTIMIZATION_THREADS = int(os.environ.get(
    "QDRANT_COLLECTION_MAX_OPTIMIZATION_THREADS", "1"
))
QDRANT_PAPERS_COLLECTION = os.environ.get("QDRANT_PAPERS_COLLECTION", "papers").strip() or "papers"
# `papers` dense search rescoring. Its 2-bit TurboQuant copy (2026-10-14, 264 B/point
# so it can stay resident) ranks coarsely on its own; rescoring 2x oversampled
# candidates against the original vectors measured Recall@10/@100 = 0.923/0.826 on
# 288 real search_papers queries, vs 0.846/0.739 for the former int8 default
# (journal 2026-10-14). 0 disables it and restores Qdrant's per-method default.
QDRANT_PAPERS_DENSE_OVERSAMPLING = float(os.environ.get(
    "QDRANT_PAPERS_DENSE_OVERSAMPLING", "2.0"))

# Qdrant rejects any single REST upsert whose JSON payload exceeds 32MB
# ("Payload error: JSON payload (... bytes) is larger than allowed (limit:
# 33554432 bytes)"). A whole-book upload chunks into thousands of dense+sparse
# points; sent as one upsert they overflow that limit. Batch the upsert so no
# single payload approaches it. At ~10-30KB/point (dense 1024d + sparse +
# content) 256 points stays well under 32MB with comfortable margin.
UPSERT_BATCH_SIZE = int(os.environ.get("QDRANT_UPSERT_BATCH", "256"))
# Iron law (W4/W14): every upsert wait=True + exponential-backoff retry.
_UPSERT_MAX_RETRIES = 5

_shared_client: QdrantClient | None = None
_shared_client_key: str | None = None
# Track when the shared client was last successfully validated. W25: long-lived
# uvicorn workers occasionally returned 0 candidates from query_points without
# raising — symptom is consistent with a stale httpx connection pool after the
# Qdrant side completed segment merges. Re-validating + recreating the client
# on suspicion gives us a recovery hook.
_shared_client_validated_at: float = 0.0
_VALIDATION_INTERVAL_S = 60.0


def _build_server_client(url: str) -> QdrantClient:
    # Large MatchAny path filters can stall concurrent Qdrant REST requests even
    # while equivalent gRPC requests remain consistently sub-100ms. Prefer the
    # native transport for query/upsert traffic while retaining an explicit opt-out
    # for deployments that expose only port 6333.
    return QdrantClient(
        url=url,
        timeout=30,
        prefer_grpc=QDRANT_PREFER_GRPC,
        grpc_port=QDRANT_GRPC_PORT,
    )


def _validate_or_rebuild(reason: str = "stale") -> bool:
    """Ping current shared client; rebuild from scratch if validation fails.

    Returns True if a rebuild happened. Caller can use this to retry a search.
    """
    global _shared_client, _shared_client_key, _shared_client_validated_at
    if _shared_client is None or _shared_client_key is None:
        return False
    url = getattr(settings, "qdrant_url", None) or QDRANT_URL
    expected_key = f"server:{url}"
    if _shared_client_key != expected_key:
        # local-mode client; can't usefully rebuild from here.
        return False
    try:
        _shared_client.get_collections()
        _shared_client_validated_at = time.time()
        return False
    except Exception as e:
        _logger.warning(
            "[qdrant] shared client validation failed (%s): %s — rebuilding",
            reason, e,
        )
    try:
        try:
            _shared_client.close()
        except Exception:
            pass
        _shared_client = _build_server_client(url)
        _shared_client.get_collections()
        _shared_client_validated_at = time.time()
        _logger.info("[qdrant] shared client rebuilt successfully")
        return True
    except Exception as e:
        _logger.error("[qdrant] shared client rebuild failed: %s", e, exc_info=True)
        return False


def _get_shared_client(qdrant_dir: str) -> QdrantClient:
    """Connect to Qdrant server (preferred) or fall back to local mode."""
    global _shared_client, _shared_client_key, _shared_client_validated_at
    url = getattr(settings, "qdrant_url", None) or QDRANT_URL
    key = f"server:{url}"
    if _shared_client is not None and _shared_client_key == key:
        # Periodic re-validation to catch dropped connections proactively.
        if time.time() - _shared_client_validated_at > _VALIDATION_INTERVAL_S:
            _validate_or_rebuild(reason="periodic")
        return _shared_client
    if _shared_client is not None:
        try:
            _shared_client.close()
        except Exception:
            pass
    try:
        client = _build_server_client(url)
        client.get_collections()
        _shared_client = client
        _shared_client_key = key
        _shared_client_validated_at = time.time()
        return _shared_client
    except Exception as e:
        _logger.error(
            "[qdrant] server connection failed (%s), returning degraded client "
            "(local fallback disabled — 82GB collection would OOM the process)",
            e,
        )
        client = _build_server_client(url)
        _shared_client = client
        _shared_client_key = key
        _shared_client_validated_at = 0.0
        return _shared_client


def generate_chunk_id(doc_id: str, content: str, level: int, start_idx: int = 0) -> str:
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    return f"{doc_id}_L{level}_{start_idx}_{content_hash}"


def _chunk_id_to_int(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:15], 16)


def sparse_dict_to_qdrant(sparse: dict[str, float]) -> SparseVector:
    """Convert {token_id_str: weight} dict to Qdrant SparseVector."""
    indices = [int(k) for k in sparse.keys()]
    values = [float(v) for v in sparse.values()]
    return SparseVector(indices=indices, values=values)


class QdrantCollectionMissing(RuntimeError):
    """The collection does not exist and this store must not create it."""


class QdrantStore:
    """Qdrant 向量存储，接口兼容 ChromaStore"""

    def __init__(self, collection_name: str | None = None, *, create_if_missing: bool = True):
        self.collection_name = collection_name or QDRANT_PAPERS_COLLECTION
        self.qdrant_dir = getattr(settings, 'qdrant_dir', os.path.join(settings.index_dir, 'qdrant'))
        client = _get_shared_client(self.qdrant_dir)
        # Follow the module's shared client rather than keep this one: see `client`.
        self._pinned_client = None if client is _shared_client else client
        self._ensure_collection(create_if_missing=create_if_missing)

    @property
    def client(self) -> QdrantClient:
        """The current shared client, unless one was assigned to this store.

        `_validate_or_rebuild` closes the shared client and builds a new one. A store
        that kept the client it was built with then failed every call with "Cannot
        invoke RPC on closed channel!" until the process restarted (2026-11-28 on .88:
        a rebuild at 06:00 after an oomd kill of Qdrant left /api/search answering 503
        for seven hours against a healthy Qdrant). Reading the module slot on each use
        lets one rebuild reach every store."""
        pinned = self.__dict__.get("_pinned_client")
        if pinned is not None:
            return pinned
        if _shared_client is not None:
            return _shared_client
        return _get_shared_client(self.qdrant_dir)

    @client.setter
    def client(self, value: QdrantClient) -> None:
        self._pinned_client = value

    @property
    def supports_sparse(self) -> bool:
        return settings.embedding.supports_sparse

    def _dense_search_params(self) -> Optional[SearchParams]:
        """Query-time quantization params for dense retrieval: the TurboQuant paper
        collections (`papers`, and `arxiv`, which copies its config) only, since the
        code/journal collections carry no quantization."""
        rescored = {QDRANT_PAPERS_COLLECTION, QDRANT_ARXIV_COLLECTION}
        if self.collection_name not in rescored or QDRANT_PAPERS_DENSE_OVERSAMPLING <= 0:
            return None
        return SearchParams(quantization=QuantizationSearchParams(
            rescore=True, oversampling=QDRANT_PAPERS_DENSE_OVERSAMPLING))

    def _ensure_collection(self, *, create_if_missing: bool = True):
        if self.client.collection_exists(self.collection_name):
            return
        # The split paper collections are created with their full production config
        # (quantization, on-disk placement, payload indexes) by
        # scripts/split_papers_corpus.py; a bare auto-created one would silently
        # serve empty results and later block that script.
        split_names = {paper_collections()[role] for role in ("arxiv", "reviews")}
        if not create_if_missing or self.collection_name in split_names:
            raise QdrantCollectionMissing(
                f"Qdrant collection `{self.collection_name}` does not exist")
        collection_options: dict[str, Any] = {}
        dense_options: dict[str, Any] = {}
        sparse_index_options: dict[str, Any] = {}
        if QDRANT_COLLECTION_ON_DISK:
            # Low-memory deployments must create every future collection in the same
            # storage mode used by the migrated collections. This does not rewrite an
            # existing collection; that remains an explicit, separately verified
            # maintenance operation because Qdrant rebuilds its segments.
            dense_options = {
                "on_disk": True,
                "hnsw_config": models.HnswConfigDiff(on_disk=True),
            }
            sparse_index_options = {"on_disk": True}
            collection_options = {
                "on_disk_payload": True,
                "hnsw_config": models.HnswConfigDiff(on_disk=True),
                "optimizers_config": models.OptimizersConfigDiff(
                    memmap_threshold=QDRANT_COLLECTION_MEMMAP_THRESHOLD,
                    max_optimization_threads=(
                        QDRANT_COLLECTION_MAX_OPTIMIZATION_THREADS
                    ),
                ),
            }
        if self.supports_sparse:
            self.client.create_collection(
                self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=VECTOR_DIM,
                        distance=Distance.COSINE,
                        **dense_options,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(**sparse_index_options)
                    )
                },
                **collection_options,
            )
        else:
            self.client.create_collection(
                self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                    **dense_options,
                ),
                **collection_options,
            )

    def add_chunks(self, chunks: List[Chunk],
                   source_type: str = None,
                   data_source: str = None,
                   doc_title: str = None,
                   doc_source: str = None,
                   use_upsert: bool = True,
                   wait: bool = True,
                   sparse_vectors: Optional[List[Dict[str, float]]] = None):
        self.upsert_points(self.build_points(
            chunks, source_type=source_type, data_source=data_source,
            doc_title=doc_title, doc_source=doc_source, sparse_vectors=sparse_vectors,
        ), wait=wait)

    def build_points(self, chunks: List[Chunk],
                     source_type: str = None,
                     data_source: str = None,
                     doc_title: str = None,
                     doc_source: str = None,
                     sparse_vectors: Optional[List[Dict[str, float]]] = None,
                     ) -> List[PointStruct]:
        """The points `add_chunks` writes, payload already projected for a paper
        collection; nothing is written."""
        points = []
        for i, chunk in enumerate(chunks or []):
            if chunk.embedding is None:
                continue

            chunk_id = generate_chunk_id(
                doc_id=chunk.doc_id, content=chunk.content,
                level=chunk.level, start_idx=chunk.start_idx,
            )

            payload = {
                "chunk_id": chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "level": chunk.level,
                "start_idx": chunk.start_idx,
                "end_idx": chunk.end_idx,
                **chunk.metadata,
            }
            if doc_title:
                payload["title"] = doc_title
            elif "title" not in payload:
                payload["title"] = chunk.metadata.get("title", "")
            if doc_source:
                payload["source"] = doc_source
            elif "source" not in payload:
                payload["source"] = chunk.metadata.get("source", "")
            if source_type:
                payload["source_type"] = source_type
            if data_source:
                payload["data_source"] = data_source

            # The paper collections (`papers`, `arxiv`, `reviews`) share the one
            # schema contract (D1-D3, docs/papers_payload_contract_plan.md); route
            # every point through the projector so `**chunk.metadata` above can
            # never leak an arbitrary key into it; `upsert_points` then refuses a
            # point that belongs to another paper collection before writing. Other
            # collections (journal_chunks, code_chunks, public_code_chunks) own
            # their payload shape elsewhere and are intentionally NOT constrained
            # here (AGENTS.md 2.1: "禁止跨 collection 复用不匹配的 payload projector").
            if is_paper_collection(self.collection_name):
                payload = project_papers_payload(payload)

            if sparse_vectors and i < len(sparse_vectors) and sparse_vectors[i]:
                vector = {
                    "dense": chunk.embedding,
                    "sparse": sparse_dict_to_qdrant(sparse_vectors[i]),
                }
            elif self.supports_sparse:
                vector = {"dense": chunk.embedding}
            else:
                vector = chunk.embedding

            points.append(PointStruct(
                id=_chunk_id_to_int(chunk_id),
                vector=vector,
                payload=payload,
            ))
        return points

    def upsert_points(self, points: List[PointStruct], wait: bool = True) -> None:
        """Write points in bounded batches. A paper collection first checks every
        point belongs to it, so a misrouted batch is refused before any write."""
        if not points:
            return
        if is_paper_collection(self.collection_name):
            for point in points:
                check_paper_point(self.collection_name, point.payload or {})

        n_batches = (len(points) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
        if n_batches > 1:
            _logger.info(
                "[qdrant] upserting %d points in %d batches of <=%d",
                len(points), n_batches, UPSERT_BATCH_SIZE,
            )
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            self._upsert_batch(points[start:start + UPSERT_BATCH_SIZE], wait=wait)

    def _upsert_batch(self, points: list, wait: bool = True):
        """Upsert one batch with exponential-backoff retry (W4 wait=True, W14 retry x5).

        Qdrant can silently drop a batch under write pressure if wait=False (W4),
        and transiently 5xx during segment merges; retry keeps the write durable.
        Point IDs are content-derived (idempotent) so a retry after a partial
        server-side apply overwrites rather than duplicates.
        """
        try:
            retry_qdrant_operation(
                f"upsert(collection={self.collection_name} points={len(points)})",
                lambda: self.client.upsert(self.collection_name, points=points, wait=wait),
                max_attempts=_UPSERT_MAX_RETRIES,
            )
        except Exception:
            _logger.error(
                "[qdrant] upsert batch permanently failed after %d attempts (%d pts)",
                _UPSERT_MAX_RETRIES, len(points),
            )
            raise

    def _build_filter(self, filter_dict: Dict[str, Any] = None,
                      level: int = None, source_type: str = None,
                      data_source: str = None) -> Optional[Filter]:
        # Treat all named kwargs as just payload-equality sugar over filter_dict.
        merged: Dict[str, Any] = dict(filter_dict or {})
        # Reserved key: must_not exclusions, e.g. {"source_type": ["review"]}.
        # Lets the default search drop review points WITHOUT touching any other
        # point (a positive MatchAny would silently exclude rows with a null/odd
        # source_type — must_not only removes the named values).
        exclude: Dict[str, Any] = merged.pop("__exclude__", None) or {}
        # Reserved key: an OR over independent sub-filters, each a filter_dict of its own
        # (e.g. [{"entry_id": [...]}, {"__exclude__": {"source_kind": "doc"}}] = "in this
        # id set, or not a doc"). It ANDs with every other condition, like ``must``.
        should_groups: List[Dict[str, Any]] = merged.pop("__should__", None) or []
        if level is not None:
            merged["level"] = level
        if source_type is not None:
            merged["source_type"] = source_type
        if data_source is not None:
            merged["data_source"] = data_source
        if not merged and not exclude and not should_groups:
            return None
        conditions = []
        for k, v in merged.items():
            if isinstance(v, list):
                conditions.append(FieldCondition(key=k, match=MatchAny(any=v)))
            else:
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
        # A list is MatchAny, a scalar MatchValue. The distinction matters for bool:
        # qdrant_client's MatchAny.any is List[StrictStr] | List[StrictInt] (it rejects
        # [True] at validation), while MatchValue.value is bool | int | str. So
        # {"withdrawn": True} -> MatchValue(True), {"source_type": ["review"]} -> MatchAny.
        # Cost of a must_not: Qdrant estimates it as total - estimate(condition)
        # (lib/segment/src/index/query_estimator.rs::invert_estimation: min=total-max,
        # exp=total-exp, max=total-min): for a rare value that is nearly every point,
        # so the condition is checked on each visited point instead of narrowing the
        # candidate set. It is only cheap when the key is payload-indexed in this
        # collection, so a must_not on an unindexed key must not go to a large
        # collection (see core/storage/corpus_layout.py ROLE_ONLY_EXCLUSIONS).
        must_not = []
        for k, v in exclude.items():
            if isinstance(v, list):
                must_not.append(FieldCondition(key=k, match=MatchAny(any=v)))
            else:
                must_not.append(FieldCondition(key=k, match=MatchValue(value=v)))
        should = [self._build_filter(group) for group in should_groups]
        return Filter(
            must=conditions or None, must_not=must_not or None, should=should or None)

    @staticmethod
    def _format_points(points) -> List[Dict]:
        formatted = []
        for point in points:
            score = point.score if point.score is not None else 0.0
            formatted.append({
                "id": point.payload.get("chunk_id", str(point.id)),
                "content": point.payload.get("content", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "content"},
                "distance": 1 - score,
                "score": score,
            })
        return formatted

    def search(self, query_embedding: List[float],
               top_k: int = 10,
               filter_dict: Dict[str, Any] = None,
               query_filter: Filter | None = None,
               level: int = None,
               source_type: str = None,
               data_source: str = None) -> List[Dict]:
        if top_k <= 0:
            return []

        resolved_filter = query_filter if query_filter is not None else self._build_filter(
            filter_dict, level, source_type, data_source
        )
        kwargs = {}
        if self.supports_sparse:
            kwargs["using"] = "dense"
        dense_params = self._dense_search_params()
        if dense_params is not None:
            kwargs["search_params"] = dense_params

        def _do() -> Optional[List]:
            try:
                results = self.client.query_points(
                    self.collection_name,
                    query=query_embedding,
                    query_filter=resolved_filter,
                    limit=top_k,
                    with_payload=True,
                    **kwargs,
                )
                return results.points
            except Exception as exc:
                _logger.exception("[qdrant] dense search raised")
                raise QdrantRetrievalUpstreamError("dense", exc) from exc

        points = _do()
        if not points and _validate_or_rebuild(reason="empty dense search"):
            _logger.info("[qdrant] dense search retrying after client rebuild")
            points = _do()
        if not points:
            _logger.warning(
                "[qdrant] dense search returned 0 points (filter=%s top_k=%d)",
                filter_dict, top_k,
            )
            return []
        return self._format_points(points)

    def hybrid_search(self, dense_vector: List[float],
                      sparse_vector: SparseVector,
                      top_k: int = 10,
                      prefetch_limit: int = 100,
                      filter_dict: Dict[str, Any] = None,
                      query_filter: Filter | None = None,
                      level: int = None,
                      source_type: str = None,
                      data_source: str = None) -> List[Dict]:
        """RRF fusion of dense + sparse retrieval."""
        if top_k <= 0:
            return []

        resolved_filter = query_filter if query_filter is not None else self._build_filter(
            filter_dict, level, source_type, data_source
        )
        dense_params = self._dense_search_params()

        def _do() -> Optional[List]:
            try:
                results = self.client.query_points(
                    self.collection_name,
                    prefetch=[
                        Prefetch(query=dense_vector, using="dense",
                                 limit=prefetch_limit, filter=resolved_filter,
                                 params=dense_params),
                        Prefetch(query=sparse_vector, using="sparse",
                                 limit=prefetch_limit, filter=resolved_filter),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
                return results.points
            except Exception as exc:
                _logger.exception("[qdrant] hybrid_search raised")
                raise QdrantRetrievalUpstreamError("hybrid", exc) from exc

        points = _do()
        # W25 defense: empty result on a presumed-healthy collection often means
        # the shared httpx pool went stale (Stage F ingest closed connections).
        # Probe + rebuild client, retry once before returning empty.
        if not points and _validate_or_rebuild(reason="empty hybrid_search"):
            _logger.info("[qdrant] hybrid_search retrying after client rebuild")
            points = _do()
        if not points:
            _logger.warning(
                "[qdrant] hybrid_search returned 0 points (filter=%s top_k=%d prefetch=%d)",
                filter_dict, top_k, prefetch_limit,
            )
            return []
        return self._format_points(points)

    def hybrid_search_many_exact(
        self,
        queries: list[tuple[List[float], SparseVector]],
        *,
        top_k: int,
        filter_dicts: list[Dict[str, Any] | None] | None = None,
        max_concurrent: int = 2,
        runtime_metrics: dict[str, int] | None = None,
        progress_callback: Callable[[int, int, dict[str, int]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[List[Dict]]:
        """Run legacy-identical hybrid requests with bounded parallelism.

        Qdrant's QueryBatch endpoint has a single deadline for the whole RPC and
        timed out even for a cold singleton in production.  Separate
        ``query_points`` calls preserve the established candidate semantics;
        running at most ``max_concurrent`` of them lets a 4-thread Qdrant keep
        capacity for interactive search while a long match progresses.
        """
        if not queries or top_k <= 0:
            return [[] for _ in queries]
        if filter_dicts is None:
            filter_dicts = [None] * len(queries)
        if len(filter_dicts) != len(queries):
            raise ValueError("filter_dicts must match queries")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")

        output: list[List[Dict]] = []
        completed = 0

        def run_one(item):
            (dense, sparse), filters = item
            if should_cancel is not None and should_cancel():
                raise QdrantRetrievalCancelledError(
                    "Qdrant exact hybrid search cancelled",
                )
            return self.hybrid_search(
                dense_vector=dense,
                sparse_vector=sparse,
                top_k=top_k,
                filter_dict=filters,
            )

        paired = list(zip(queries, filter_dicts))
        for offset in range(0, len(paired), max_concurrent):
            if should_cancel is not None and should_cancel():
                raise QdrantRetrievalCancelledError(
                    "Qdrant exact hybrid search cancelled between groups",
                )
            group = paired[offset:offset + max_concurrent]
            with ThreadPoolExecutor(max_workers=len(group)) as executor:
                group_results = list(executor.map(run_one, group))
            output.extend(group_results)
            completed += len(group_results)
            metrics = {
                "qdrant_batches": completed,
                "qdrant_fallback_splits": 0,
            }
            if progress_callback is not None:
                progress_callback(completed, len(queries), metrics)

        if runtime_metrics is not None:
            runtime_metrics.update({
                "qdrant_batches": completed,
                "qdrant_fallback_splits": 0,
            })
        return output

    def sparse_search(self, sparse_vector: SparseVector,
                      top_k: int = 10,
                      filter_dict: Dict[str, Any] = None,
                      query_filter: Filter | None = None,
                      level: int = None,
                      source_type: str = None,
                      data_source: str = None) -> List[Dict]:
        """Sparse-only (lexical) retrieval."""
        if top_k <= 0:
            return []

        resolved_filter = query_filter if query_filter is not None else self._build_filter(
            filter_dict, level, source_type, data_source
        )

        def _do() -> Optional[List]:
            try:
                results = self.client.query_points(
                    self.collection_name,
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k,
                    query_filter=resolved_filter,
                    with_payload=True,
                )
                return results.points
            except Exception as exc:
                _logger.exception("[qdrant] sparse search raised")
                raise QdrantRetrievalUpstreamError("sparse", exc) from exc

        points = _do()
        if not points and _validate_or_rebuild(reason="empty sparse search"):
            _logger.info("[qdrant] sparse search retrying after client rebuild")
            points = _do()
        if not points:
            _logger.warning(
                "[qdrant] sparse search returned 0 points (filter=%s top_k=%d)",
                filter_dict, top_k,
            )
            return []
        return self._format_points(points)

    def search_by_level(self, query_embedding: List[float],
                        levels: List[int],
                        top_k_per_level: int = 100,
                        source_type: str = None,
                        data_source: str = None) -> Dict[int, List[Dict]]:
        results = {}
        for level in levels:
            results[level] = self.search(
                query_embedding=query_embedding, top_k=top_k_per_level,
                level=level, source_type=source_type, data_source=data_source,
            )
        return results

    def get_by_doc_id(self, doc_id: str) -> List[Dict]:
        try:
            results, _ = self.client.scroll(
                self.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                limit=10000, with_payload=True, with_vectors=True,
            )
        except Exception:
            return []

        return [{
            "id": p.payload.get("chunk_id", str(p.id)),
            "content": p.payload.get("content", ""),
            "metadata": {k: v for k, v in p.payload.items() if k != "content"},
            "embedding": p.vector,
        } for p in results]

    def delete_chunks(self, chunk_ids: List[str]):
        if chunk_ids:
            int_ids = [_chunk_id_to_int(cid) for cid in chunk_ids]
            self.client.delete(self.collection_name, points_selector=models.PointIdsList(points=int_ids))

    def delete_by_doc_id(self, doc_id: str) -> int:
        try:
            # Count the document's own points (``doc_id`` is payload-indexed, the same
            # index the delete filter uses) instead of differencing two whole-collection
            # counts, which are O(points) on the ~24M-point corpus.
            doc_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            deleted = self.client.count(
                self.collection_name, count_filter=doc_filter, exact=True).count
            self.client.delete(
                self.collection_name, points_selector=models.FilterSelector(filter=doc_filter))
            return deleted
        except Exception as e:
            print(f"Warning: Error deleting chunks for doc_id={doc_id}: {e}")
            return 0

    def exists_by_doc_id(self, doc_id: str) -> bool:
        try:
            results, _ = self.client.scroll(
                self.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                limit=1, with_payload=False,
            )
            return len(results) > 0
        except Exception:
            return False

    def count_by_doc_id(self, doc_id: str) -> int:
        try:
            results = self.client.count(
                self.collection_name,
                count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            )
            return results.count
        except Exception:
            return 0

    def get_all(self, include: List[str] = None) -> Dict[str, Any]:
        """兼容 ChromaDB collection.get() 接口"""
        ids, documents, metadatas = [], [], []
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                self.collection_name, limit=1000,
                offset=offset, with_payload=True, with_vectors=False,
            )
            if not results:
                break
            for p in results:
                ids.append(p.payload.get("chunk_id", str(p.id)))
                documents.append(p.payload.get("content", ""))
                metadatas.append({k: v for k, v in p.payload.items() if k != "content"})
            offset = next_offset
            if offset is None:
                break
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def count(self) -> int:
        try:
            return self.client.count(self.collection_name).count
        except Exception:
            return 0

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
