"""
SearchPipeline — orchestrates embed → retrieve → rerank → return.

Single entry point for all search operations, replacing the scattered
retriever classes with a unified pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import logging
import time
from copy import deepcopy
from config import settings

from app.models.schemas import SearchResult
from core.storage.qdrant_store import (
    QdrantRetrievalCancelledError,
    sparse_dict_to_qdrant,
)

_logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    query: str
    top_k: int = 20
    mode: str = "hybrid"
    source_type: Optional[str] = None
    level: Optional[int] = None  # Qdrant payload.level filter (1=paper title chunk)
    # OpenReview review chunks (source_type=review) share the `papers` collection.
    # Default search EXCLUDES them (they're a different content type); set
    # include_reviews=True to mix them in, or source_type="review" for review-only.
    include_reviews: bool = False
    filters: dict[str, Any] = field(default_factory=dict)
    # True (default): collapse multi-chunk hits from one paper to its single
    # best-scoring chunk (W3 anti-flooding). False ("chunk mode"): return the
    # raw top-k chunks ranked by score — multiple chunks per paper allowed.
    dedup_by_paper: bool = True


@dataclass
class PipelineMetrics:
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    candidates_before_rerank: int = 0
    candidates_after_rerank: int = 0
    mode: str = "hybrid"
    embedding_batches: int = 0
    qdrant_batches: int = 0
    qdrant_fallback_splits: int = 0
    rerank_model_calls: int = 0
    rerank_batches: int | None = 0
    rerank_pairs: int = 0


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    mode: str
    total: int = 0
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)


class SearchPipeline:
    """Orchestrates: embed query → retrieve from Qdrant → optional rerank → return."""

    def __init__(self, embedder=None, store=None, reranker=None):
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    def search(self, request: SearchRequest) -> SearchResponse:
        if self._embedder is None or self._store is None:
            return SearchResponse(
                query=request.query, results=[], mode=request.mode, total=0,
            )

        t0 = time.perf_counter()

        embed_result = self._embedder.embed_texts([request.query])
        query_dense = embed_result.dense[0].tolist()
        query_sparse = embed_result.sparse[0] if embed_result.sparse else None
        t_embed = time.perf_counter()

        # Merge all retrieve-stage filters (source_type, level, free-form) into a
        # single payload-equality dict consumed by QdrantStore._build_filter.
        filter_dict = dict(request.filters or {})
        if request.source_type:
            filter_dict["source_type"] = request.source_type
        elif not request.include_reviews:
            # Keep review chunks out of normal results (they share the collection).
            filter_dict.setdefault("__exclude__", {})["source_type"] = ["review"]
        if request.level is not None:
            filter_dict["level"] = request.level

        retrieve_k = request.top_k * 5 if self._reranker else request.top_k

        if request.mode == "sparse" and query_sparse is not None:
            raw = self._store.sparse_search(
                sparse_vector=sparse_dict_to_qdrant(query_sparse),
                top_k=retrieve_k,
                filter_dict=filter_dict,
            )
        elif request.mode == "hybrid" and query_sparse is not None:
            raw = self._store.hybrid_search(
                dense_vector=query_dense,
                sparse_vector=sparse_dict_to_qdrant(query_sparse),
                top_k=retrieve_k,
                filter_dict=filter_dict,
            )
        else:
            raw = self._store.search(
                query_dense, top_k=retrieve_k, filter_dict=filter_dict,
            )
        t_retrieve = time.perf_counter()

        results = self._to_search_results(raw, request.mode)
        candidates_before = len(results)

        rerank_ms = 0.0
        if self._reranker and results:
            # Rerank entire candidate pool — early truncation here starves the
            # paper-level dedup below, since a single high-relevance paper can
            # legitimately occupy many top chunks (BGE behavior, not a bug).
            results = self._reranker.rerank(request.query, results, top_k=len(results))
            rerank_ms = (time.perf_counter() - t_retrieve) * 1000

        if request.dedup_by_paper:
            results = self._dedup_by_paper(results)
        results = results[:request.top_k]

        t_end = time.perf_counter()
        _logger.info(
            "search.done mode=%s dedup_by_paper=%s candidates=%d returned=%d rerank_ms=%.0f",
            request.mode, request.dedup_by_paper, candidates_before,
            len(results), rerank_ms,
        )

        metrics = PipelineMetrics(
            embed_ms=round((t_embed - t0) * 1000, 1),
            retrieve_ms=round((t_retrieve - t_embed) * 1000, 1),
            rerank_ms=round(rerank_ms, 1),
            total_ms=round((t_end - t0) * 1000, 1),
            candidates_before_rerank=candidates_before,
            candidates_after_rerank=len(results),
            mode=request.mode,
        )

        return SearchResponse(
            query=request.query,
            results=results,
            mode=request.mode,
            total=len(results),
            metrics=metrics,
        )

    def search_many(
        self,
        requests: list[SearchRequest],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[SearchResponse]:
        """Batch only Qdrant transport while preserving legacy model execution.

        FP16 outputs can depend on activation batch composition.  Embedding and
        reranking therefore stay one-query-at-a-time, exactly like repeated
        ``search()`` calls; only the already-computed vectors share bounded,
        ordered Qdrant RPCs.
        """
        if not requests:
            return []
        if should_cancel is not None and should_cancel():
            raise QdrantRetrievalCancelledError("search batch cancelled before embedding")
        if self._embedder is None or self._store is None:
            return [SearchResponse(query=r.query, results=[], mode=r.mode) for r in requests]

        def serial_fallback() -> list[SearchResponse]:
            responses = []
            for index, request in enumerate(requests):
                if should_cancel is not None and should_cancel():
                    raise QdrantRetrievalCancelledError(
                        "search batch cancelled during serial fallback",
                    )
                responses.append(self.search(request))
                if progress_callback is not None:
                    progress_callback({
                        "phase": "serial_fallback",
                        "completed": index + 1,
                        "total": len(requests),
                    })
            return responses

        if any(request.mode != "hybrid" for request in requests):
            return serial_fallback()

        started = time.perf_counter()
        if progress_callback is not None:
            progress_callback({"phase": "embedding", "completed": 0, "total": len(requests)})
        embedded_queries = []
        for index, request in enumerate(requests):
            if should_cancel is not None and should_cancel():
                raise QdrantRetrievalCancelledError(
                    "search batch cancelled during embedding",
                )
            embedded = self._embedder.embed_texts([request.query])
            if not embedded.sparse:
                return serial_fallback()
            embedded_queries.append((
                embedded.dense[0].tolist(),
                sparse_dict_to_qdrant(embedded.sparse[0]),
            ))
            if progress_callback is not None:
                progress_callback({
                    "phase": "embedding",
                    "completed": index + 1,
                    "total": len(requests),
                })
        embedded_at = time.perf_counter()

        filters = []
        for request in requests:
            item = deepcopy(request.filters or {})
            if request.source_type:
                item["source_type"] = request.source_type
            elif not request.include_reviews:
                item.setdefault("__exclude__", {})["source_type"] = ["review"]
            if request.level is not None:
                item["level"] = request.level
            filters.append(item)
        depths = [request.top_k * 5 if self._reranker else request.top_k
                  for request in requests]
        if len(set(depths)) != 1:
            return serial_fallback()

        qdrant_runtime: dict[str, int] = {}

        def on_qdrant_progress(completed: int, total: int, runtime: dict[str, int]) -> None:
            if progress_callback is not None:
                progress_callback({
                    "phase": "qdrant",
                    "completed": completed,
                    "total": total,
                    **runtime,
                })

        exact_many = getattr(self._store, "hybrid_search_many_exact", None)
        if exact_many is not None:
            raw_groups = exact_many(
                embedded_queries,
                top_k=depths[0],
                filter_dicts=filters,
                max_concurrent=settings.retrieval.match_doc_qdrant_concurrency,
                runtime_metrics=qdrant_runtime,
                progress_callback=on_qdrant_progress,
                should_cancel=should_cancel,
            )
        else:
            raw_groups = self._store.hybrid_search_many(
                embedded_queries,
                top_k=depths[0],
                filter_dicts=filters,
                max_queries_per_batch=settings.retrieval.match_doc_qdrant_concurrency,
                runtime_metrics=qdrant_runtime,
                progress_callback=on_qdrant_progress,
                should_cancel=should_cancel,
            )
        retrieved_at = time.perf_counter()
        if len(raw_groups) != len(requests):
            raise RuntimeError("Qdrant batch response count did not match query requests")
        result_groups = [
            self._to_search_results(raw, request.mode)
            for request, raw in zip(requests, raw_groups)
        ]
        rerank_pairs = sum(len(group) for group in result_groups)
        rerank_model_calls = 0
        rerank_completed = 0
        if progress_callback is not None:
            progress_callback({"phase": "rerank", "completed": 0, "total": rerank_pairs})

        output = []
        for request, results in zip(requests, result_groups):
            candidates_before = len(results)
            if self._reranker and results:
                if should_cancel is not None and should_cancel():
                    raise QdrantRetrievalCancelledError(
                        "search batch cancelled during reranking",
                    )
                # Pass the predicate down: the check above only fires BETWEEN
                # queries, and one query's rerank is the single longest
                # uninterruptible span in the whole match (all 5x candidates in one
                # call). Inside, it is re-checked every batch_size pairs.
                results = self._reranker.rerank(
                    request.query,
                    results,
                    top_k=len(results),
                    should_cancel=should_cancel,
                )
                rerank_model_calls += 1
            rerank_completed += candidates_before
            if progress_callback is not None:
                progress_callback({
                    "phase": "rerank",
                    "completed": rerank_completed,
                    "total": rerank_pairs,
                })
            if request.dedup_by_paper:
                results = self._dedup_by_paper(results)
            results = results[:request.top_k]
            output.append(SearchResponse(
                query=request.query,
                results=results,
                mode=request.mode,
                total=len(results),
                metrics=PipelineMetrics(
                    embed_ms=round((embedded_at - started) * 1000, 1),
                    retrieve_ms=round((retrieved_at - embedded_at) * 1000, 1),
                    rerank_ms=round((time.perf_counter() - retrieved_at) * 1000, 1),
                    total_ms=round((time.perf_counter() - started) * 1000, 1),
                    candidates_before_rerank=candidates_before,
                    candidates_after_rerank=len(results),
                    mode=request.mode,
                    embedding_batches=len(requests),
                    qdrant_batches=qdrant_runtime.get("qdrant_batches", 0),
                    qdrant_fallback_splits=qdrant_runtime.get("qdrant_fallback_splits", 0),
                    rerank_model_calls=rerank_model_calls,
                    rerank_batches=(None if self._reranker and rerank_pairs else 0),
                    rerank_pairs=rerank_pairs,
                ),
            ))
        return output

    def search_stream(self, request: SearchRequest):
        """Generator variant of search(): yields per-stage events for live UI streaming.

        Yields dicts with key `stage` in {embed, retrieve, rerank, results}. Final
        event (stage='results') carries the SearchResult list + cumulative metrics.
        Callers (e.g. /api/search/stream) wrap each event as one NDJSON line.
        """
        if self._embedder is None or self._store is None:
            yield {"stage": "results", "results": [], "metrics": {
                "embed_ms": 0.0, "retrieve_ms": 0.0, "rerank_ms": 0.0,
                "total_ms": 0.0, "candidates_before_rerank": 0,
                "candidates_after_rerank": 0, "mode": request.mode,
            }}
            return

        t0 = time.perf_counter()

        embed_result = self._embedder.embed_texts([request.query])
        query_dense = embed_result.dense[0].tolist()
        query_sparse = embed_result.sparse[0] if embed_result.sparse else None
        t_embed = time.perf_counter()
        embed_ms = round((t_embed - t0) * 1000, 1)
        yield {"stage": "embed", "embed_ms": embed_ms}

        filter_dict = dict(request.filters or {})
        if request.source_type:
            filter_dict["source_type"] = request.source_type
        elif not request.include_reviews:
            # Keep review chunks out of normal results (they share the collection).
            filter_dict.setdefault("__exclude__", {})["source_type"] = ["review"]
        if request.level is not None:
            filter_dict["level"] = request.level

        retrieve_k = request.top_k * 5 if self._reranker else request.top_k

        if request.mode == "sparse" and query_sparse is not None:
            raw = self._store.sparse_search(
                sparse_vector=sparse_dict_to_qdrant(query_sparse),
                top_k=retrieve_k, filter_dict=filter_dict,
            )
        elif request.mode == "hybrid" and query_sparse is not None:
            raw = self._store.hybrid_search(
                dense_vector=query_dense,
                sparse_vector=sparse_dict_to_qdrant(query_sparse),
                top_k=retrieve_k, filter_dict=filter_dict,
            )
        else:
            raw = self._store.search(
                query_dense, top_k=retrieve_k, filter_dict=filter_dict,
            )
        t_retrieve = time.perf_counter()
        retrieve_ms = round((t_retrieve - t_embed) * 1000, 1)

        results = self._to_search_results(raw, request.mode)
        candidates_before = len(results)
        yield {"stage": "retrieve", "retrieve_ms": retrieve_ms,
               "candidates": candidates_before, "mode": request.mode}

        rerank_ms = 0.0
        if self._reranker and results:
            # Rerank entire candidate pool (no early truncation) so the
            # paper-level dedup below has full headroom.
            results = self._reranker.rerank(request.query, results, top_k=len(results))
            rerank_ms = round((time.perf_counter() - t_retrieve) * 1000, 1)
            yield {"stage": "rerank", "rerank_ms": rerank_ms,
                   "candidates_before": candidates_before,
                   "candidates_after": len(results)}
        else:
            yield {"stage": "rerank", "rerank_ms": 0.0,
                   "candidates_before": candidates_before,
                   "candidates_after": len(results),
                   "skipped": True}

        if request.dedup_by_paper:
            results = self._dedup_by_paper(results)
        results = results[:request.top_k]

        t_end = time.perf_counter()
        _logger.info(
            "search_stream.done mode=%s dedup_by_paper=%s candidates=%d returned=%d",
            request.mode, request.dedup_by_paper, candidates_before, len(results),
        )
        metrics = {
            "embed_ms": embed_ms, "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
            "total_ms": round((t_end - t0) * 1000, 1),
            "candidates_before_rerank": candidates_before,
            "candidates_after_rerank": len(results),
            "mode": request.mode,
        }
        yield {"stage": "results", "results": results,
               "metrics": metrics}

    @staticmethod
    def _dedup_by_paper(results: list[SearchResult]) -> list[SearchResult]:
        """Collapse multi-chunk hits from one paper to its best-scoring chunk.

        Key precedence: payload.paper_id → doc_id. Iteration order is preserved,
        so when called after rerank the first occurrence is the top-scoring
        chunk for that paper and subsequent same-paper chunks are dropped.
        """
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for r in results:
            paper_id = (r.metadata or {}).get("paper_id") if r.metadata else None
            key = str(paper_id) if paper_id else str(r.doc_id or r.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped

    @staticmethod
    def _to_search_results(raw: list[dict], mode: str) -> list[SearchResult]:
        if not isinstance(raw, list):
            return []
        # Fields hoisted to top-level SearchResult; rest of payload is exposed
        # under .metadata so the frontend can render source/venue/year etc.
        _HOISTED = {"content", "chunk_id", "doc_id", "title", "source"}
        results: list[SearchResult] = []
        for r in raw:
            payload = r.get("metadata") or {}
            meta = {k: v for k, v in payload.items() if k not in _HOISTED}
            meta["method"] = mode
            results.append(
                SearchResult(
                    doc_id=str(payload.get("doc_id", r.get("doc_id", ""))),
                    chunk_id=str(r.get("id", r.get("chunk_id", ""))),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    doc_title=payload.get("title", ""),
                    source=payload.get("source", ""),
                    metadata=meta,
                )
            )
        return results
