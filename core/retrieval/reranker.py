"""
Reranker — re-scores candidate results using a cross-encoder model.

Takes top-N candidates from retrieval and re-ranks them to top-K
with much higher precision.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from config import settings
from app.models.schemas import SearchResult
from core.gpu_memory import (
    CudaAllocatorSnapshot,
    capture_cuda_allocator_snapshot,
    configure_cuda_memory_fraction,
    gpu_model_activity,
    is_cuda_device,
    trim_cuda_cache_if_idle,
    trim_cuda_cache_and_capture_if_idle,
)

_logger = logging.getLogger(__name__)
_MIB = 1024 * 1024


class RerankCancelledError(RuntimeError):
    """Rerank stopped between micro-batches because the caller cancelled.

    Distinct from QdrantRetrievalCancelledError so a caller can tell which phase
    it died in; core.retrieval must not import the storage layer's exception.
    """


@dataclass(frozen=True)
class RerankerCudaTelemetry:
    """Outcome of one allocator-maintenance decision, without request content."""

    before: CudaAllocatorSnapshot | None
    after: CudaAllocatorSnapshot | None
    post_trim: CudaAllocatorSnapshot | None
    trim_reason: str | None
    trim_executed: bool


class BaseReranker:
    """Abstract base for rerankers."""

    def rerank(self, query: str, results: list[SearchResult],
               top_k: int = 10, *,
               should_cancel: Callable[[], bool] | None = None) -> list[SearchResult]:
        raise NotImplementedError

    def score_pairs(
        self,
        query: str,
        texts: list[str],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[float]:
        """Cross-encoder relevance score for each text against the query.

        The text-level primitive under rerank(): callers that don't model results as
        SearchResult (e.g. the code KB) can rescore raw snippets without adapting to
        that schema. Same order as the input."""
        raise NotImplementedError


class BgeReranker(BaseReranker):
    """BGE reranker using FlagEmbedding's FlagReranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str | None = None, *, batch_size: int | None = None,
                 idle_cache_limit_mib: int | None = None):
        self._model_name = model_name
        self._device = device or settings.embedding.device
        self._use_fp16 = is_cuda_device(self._device)
        self._batch_size = (
            settings.reranker.batch_size if batch_size is None else batch_size
        )
        idle_cache_limit = (
            settings.reranker.idle_cache_limit_mib
            if idle_cache_limit_mib is None else idle_cache_limit_mib
        )
        if not 1 <= self._batch_size <= 128:
            raise ValueError("batch_size must be between 1 and 128")
        if idle_cache_limit < 0:
            raise ValueError("idle_cache_limit_mib must not be negative")
        self._idle_cache_limit_bytes = _MIB * idle_cache_limit
        self._model = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._last_cuda_telemetry: RerankerCudaTelemetry | None = None

    @property
    def last_cuda_telemetry(self) -> RerankerCudaTelemetry | None:
        """Most recent aggregate CUDA maintenance state for a diagnostic caller."""
        with self._telemetry_lock:
            return self._last_cuda_telemetry

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                import gc, os, torch
                from FlagEmbedding import FlagReranker
                # Deployment override: load from a local dir (e.g. modelscope cache on a
                # HF-blocked host) while keeping ``_model_name`` as the repo-id.
                model_path = os.environ.get("BGE_RERANKER_PATH") or self._model_name
                if settings.reranker.cuda_memory_fraction is not None and is_cuda_device(self._device):
                    if not configure_cuda_memory_fraction(
                        self._device,
                        settings.reranker.cuda_memory_fraction,
                    ):
                        raise RuntimeError(
                            "configured Retrieval CUDA memory fraction could not be applied",
                        )
                with gpu_model_activity(self._device):
                    model = FlagReranker(
                        model_path,
                        use_fp16=self._use_fp16,
                        devices=[self._device],
                        batch_size=self._batch_size,
                    )
                    if is_cuda_device(self._device):
                        model.compute_score([["warmup", "warmup"]])
                        gc.collect()
                self._model = model
                if is_cuda_device(self._device):
                    trim_cuda_cache_if_idle(self._device)
        return self._model

    def score_pairs(
        self,
        query: str,
        texts: list[str],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[float]:
        if not texts:
            return []
        model = self._load_model()
        # FlagReranker uses shared model/device state. Serialize inference calls so
        # concurrent API requests reuse one GPU instance without racing its internals.
        before: CudaAllocatorSnapshot | None = None
        after: CudaAllocatorSnapshot | None = None
        elapsed_ms = 0.0
        try:
            with self._inference_lock:
                with gpu_model_activity(self._device):
                    before = capture_cuda_allocator_snapshot(self._device)
                    started = time.perf_counter()
                    try:
                        pairs = [[query, text] for text in texts]
                        if should_cancel is None:
                            # Every ordinary caller: one call, FlagEmbedding forms its
                            # own batches.
                            scores = model.compute_score(pairs, normalize=True)
                        else:
                            # A durable match reranks its candidates in one
                            # uninterruptible call, so a cancel cannot land until it
                            # finishes. Drive the batches here instead and check
                            # between them.
                            #
                            # This is NOT score-identical to the single call, and that
                            # is a deliberate, measured trade. FlagEmbedding groups
                            # pairs by length to limit padding rather than taking them
                            # in input order, so pre-chunking changes each batch's
                            # composition and FP16 activations move with it.
                            #
                            # Measured on 394 real `papers` chunks x 3 queries
                            # (scripts/smoke/rerank_cancel_equivalence_real_gate.py):
                            # max |score delta| 3.47e-05; top-10 order identical for
                            # every query; top-50 SET identical for every query, with
                            # one query swapping 2 adjacent near-tied positions. So
                            # recall is unchanged and the visible ranking is stable —
                            # which is the bar, not bit-exactness. Cancellation
                            # latency drops from the whole candidate set to one batch.
                            scores = []
                            for offset in range(0, len(pairs), self._batch_size):
                                if should_cancel():
                                    raise RerankCancelledError(
                                        f"rerank cancelled after {offset}/{len(pairs)} pairs",
                                    )
                                chunk = model.compute_score(
                                    pairs[offset:offset + self._batch_size],
                                    normalize=True,
                                )
                                if isinstance(chunk, (int, float)):
                                    chunk = [chunk]
                                scores.extend(chunk)
                    finally:
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        after = capture_cuda_allocator_snapshot(self._device)
        finally:
            trim_reason = self._trim_reason(before, after)
            trim_executed, post_trim = self._maintain_cuda_cache(trim_reason)
            self._log_cuda_request(
                candidate_count=len(texts),
                elapsed_ms=elapsed_ms,
                before=before,
                after=after,
                post_trim=post_trim,
                trim_reason=trim_reason,
                trim_executed=trim_executed,
            )
            with self._telemetry_lock:
                self._last_cuda_telemetry = RerankerCudaTelemetry(
                    before=before,
                    after=after,
                    post_trim=post_trim,
                    trim_reason=trim_reason,
                    trim_executed=trim_executed,
                )
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(s) for s in scores]

    def _maintain_cuda_cache(
        self, trim_reason: str | None,
    ) -> tuple[bool, CudaAllocatorSnapshot | None]:
        if trim_reason is None:
            return False, None
        try:
            return trim_cuda_cache_and_capture_if_idle(self._device)
        except Exception:
            # Defend the search contract even if a future maintenance helper
            # regresses its own fail-open guarantee.
            _logger.warning("reranker CUDA cache maintenance failed", exc_info=True)
            return False, None

    def _trim_reason(
        self,
        before: CudaAllocatorSnapshot | None,
        after: CudaAllocatorSnapshot | None,
    ) -> str | None:
        if before is None or after is None:
            return None
        if after.num_ooms > before.num_ooms:
            return "oom"
        if after.num_alloc_retries > before.num_alloc_retries:
            return "allocator_retry"
        if after.unused_cache_bytes > self._idle_cache_limit_bytes:
            return "idle_cache_budget"
        return None

    def _log_cuda_request(
        self,
        *,
        candidate_count: int,
        elapsed_ms: float,
        before: CudaAllocatorSnapshot | None,
        after: CudaAllocatorSnapshot | None,
        post_trim: CudaAllocatorSnapshot | None,
        trim_reason: str | None,
        trim_executed: bool,
    ) -> None:
        if before is None or after is None:
            return
        _logger.info(
            "reranker.cuda candidates=%d batch_size=%d elapsed_ms=%.1f "
            "allocated_before_mib=%.1f reserved_before_mib=%.1f "
            "inactive_split_before_mib=%.1f ooms_before=%d retries_before=%d "
            "allocated_after_mib=%.1f reserved_after_mib=%.1f "
            "inactive_split_after_mib=%.1f ooms_after=%d retries_after=%d "
            "allocated_post_trim_mib=%.1f reserved_post_trim_mib=%.1f "
            "inactive_split_post_trim_mib=%.1f "
            "trim_reason=%s trim_executed=%s",
            candidate_count,
            self._batch_size,
            elapsed_ms,
            before.allocated_bytes / _MIB,
            before.reserved_bytes / _MIB,
            before.inactive_split_bytes / _MIB,
            before.num_ooms,
            before.num_alloc_retries,
            after.allocated_bytes / _MIB,
            after.reserved_bytes / _MIB,
            after.inactive_split_bytes / _MIB,
            after.num_ooms,
            after.num_alloc_retries,
            (post_trim or after).allocated_bytes / _MIB,
            (post_trim or after).reserved_bytes / _MIB,
            (post_trim or after).inactive_split_bytes / _MIB,
            trim_reason or "none",
            trim_executed,
        )

    def rerank(self, query: str, results: list[SearchResult],
               top_k: int = 10, *,
               should_cancel: Callable[[], bool] | None = None) -> list[SearchResult]:
        if not results:
            return []

        scores = self.score_pairs(
            query, [r.content for r in results], should_cancel=should_cancel,
        )

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored[:top_k]:
            reranked.append(SearchResult(
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                content=result.content,
                score=float(score),
                doc_title=result.doc_title,
                source=result.source,
                metadata={**result.metadata, "reranker_score": float(score)},
            ))
        return reranked


def create_reranker(model_name: Optional[str] = None) -> BaseReranker:
    """Factory for creating reranker instances."""
    name = model_name or "BAAI/bge-reranker-v2-m3"
    return BgeReranker(model_name=name)


_SHARED_RERANKER: BaseReranker | None = None
_SHARED_RERANKER_LOCK = threading.Lock()


def get_shared_reranker(model_name: Optional[str] = None) -> BaseReranker:
    """Return the process-wide reranker instance.

    The cross-encoder is large and may hold GPU memory after warmup. Search,
    match-doc, and journal rerank paths must share this object instead of each
    service constructing another model wrapper.
    """
    global _SHARED_RERANKER
    if _SHARED_RERANKER is None:
        with _SHARED_RERANKER_LOCK:
            if _SHARED_RERANKER is None:
                _SHARED_RERANKER = create_reranker(model_name)
    return _SHARED_RERANKER
