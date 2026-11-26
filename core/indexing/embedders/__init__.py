"""
Embedder abstraction layer.

Provides a unified interface for embedding models that may output
dense vectors, sparse vectors, or both.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ContextManager, Dict, Iterator, Literal, Optional, TYPE_CHECKING

import numpy as np

from config import settings, EMBEDDING_MODELS
from core.gpu_memory import (
    configure_cuda_memory_fraction,
    is_cuda_device,
)

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.models.schemas import Chunk


EmbeddingPriority = Literal["interactive", "background"]


class _EmbeddingScheduler:
    """Serialize access to one shared model while giving waiting queries priority.

    Background indexers release the slot after every batch. If an interactive request
    arrived during that batch, the next background batch waits until all already-waiting
    interactive work has run. This prevents long indexing jobs from starving search
    without loading a second model instance.
    """

    def __init__(self, *, max_interactive_burst: int = 8) -> None:
        if max_interactive_burst < 1:
            raise ValueError("max_interactive_burst must be at least 1")
        self._condition = threading.Condition()
        self._active = False
        self._interactive_waiters = 0
        self._background_waiters = 0
        self._interactive_streak = 0
        self._max_interactive_burst = max_interactive_burst

    @contextmanager
    def acquire(self, priority: EmbeddingPriority) -> Iterator[None]:
        if priority not in {"interactive", "background"}:
            raise ValueError(f"unknown embedding priority: {priority}")
        interactive = priority == "interactive"
        with self._condition:
            if interactive:
                self._interactive_waiters += 1
            else:
                self._background_waiters += 1
            try:
                while self._active or (
                    interactive
                    and self._background_waiters > 0
                    and self._interactive_streak >= self._max_interactive_burst
                ) or (
                    not interactive
                    and self._interactive_waiters > 0
                    and self._interactive_streak < self._max_interactive_burst
                ):
                    self._condition.wait()
                self._active = True
                if interactive:
                    if self._background_waiters > 0:
                        self._interactive_streak += 1
                    else:
                        self._interactive_streak = 0
                else:
                    self._interactive_streak = 0
            finally:
                if interactive:
                    self._interactive_waiters -= 1
                else:
                    self._background_waiters -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


_EMBEDDING_SCHEDULER = _EmbeddingScheduler()


def embedding_slot(priority: EmbeddingPriority = "interactive") -> ContextManager[None]:
    """Reserve the process-wide shared embedder for one encode call."""
    return _EMBEDDING_SCHEDULER.acquire(priority)


@dataclass
class EmbeddingResult:
    """Output of an embedding model."""
    dense: np.ndarray
    sparse: Optional[list[dict[str, float]]] = None


class BaseEmbedder(ABC):
    """Abstract base for all embedding models."""

    @abstractmethod
    def embed_texts(
        self, texts: list[str], *, priority: EmbeddingPriority = "interactive",
    ) -> EmbeddingResult:
        """Embed one request and release its temporary CUDA cache before return."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    def supports_sparse(self) -> bool:
        return False

    _EMBED_BATCH = 8

    def embed_chunks(self, chunks: list["Chunk"]) -> tuple[list["Chunk"], Optional[list[dict[str, float]]]]:
        """Embed chunks in place, batched to limit peak memory.

        W49: on 32 GB machines running Qdrant + BGE-M3 concurrently, a
        single large encode call can OOM.  Splitting into batches of
        ``_EMBED_BATCH`` and releasing intermediate tensors between
        batches keeps peak allocation bounded.
        """
        import gc
        texts = [c.content for c in chunks]
        if len(texts) <= self._EMBED_BATCH:
            result = self.embed_texts(texts, priority="background")
            for i, chunk in enumerate(chunks):
                chunk.embedding = result.dense[i].tolist()
            return chunks, result.sparse

        all_dense = []
        all_sparse: list[dict[str, float]] | None = [] if self.supports_sparse else None
        for start in range(0, len(texts), self._EMBED_BATCH):
            batch = texts[start : start + self._EMBED_BATCH]
            result = self.embed_texts(batch, priority="background")
            all_dense.append(result.dense)
            if all_sparse is not None and result.sparse is not None:
                all_sparse.extend(result.sparse)
            del result
            gc.collect()

        dense = np.concatenate(all_dense, axis=0)
        for i, chunk in enumerate(chunks):
            chunk.embedding = dense[i].tolist()
        return chunks, all_sparse


def create_embedder(model_name: str | None = None, **kwargs) -> BaseEmbedder:
    """Factory: create the right embedder based on model name.

    Prefer ``get_embedder(model_name)`` for callers that want the
    process-wide shared instance (cheaper, hot in GPU memory). Use
    ``create_embedder`` directly only when a one-shot fresh instance is
    actually required (tests / device probes).
    """
    name = model_name or settings.embedding.model_name
    model_info = EMBEDDING_MODELS.get(name, {})

    device = kwargs.get("device") or settings.embedding.device
    if settings.reranker.cuda_memory_fraction is not None and is_cuda_device(device):
        if not configure_cuda_memory_fraction(
            device,
            settings.reranker.cuda_memory_fraction,
        ):
            raise RuntimeError("configured Retrieval CUDA memory fraction could not be applied")

    if model_info.get("supports_sparse"):
        from core.indexing.embedders.bge_m3 import BgeM3Embedder
        return BgeM3Embedder(model_name=name, **kwargs)

    from core.indexing.embedders.sentence_transformer import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model_name=name, **kwargs)


# W46 (2026-05-20): process-wide cached embedder. Background:
#   - IndexService.new_embedder was per-instance lazy → with get_index_service
#     keyed by chunker_type (W36), each chunker variant held its own BGE-M3
#     copy in GPU memory.
#   - SemanticSectionChunker's _BgeM3LlamaAdapter built yet another copy when
#     semantic_inner_split=True.
#   - Each fresh process / first-touch paid the ~5-10s tokenizer + ~6.4GB
#     checkpoint load cost.
# Fix: route ALL in-process callers through ``get_embedder()`` so the
# instance is created exactly once per process (per model_name key) and
# hot in GPU memory for the lifetime of the worker. uvicorn startup pays
# the load cost up front; subsequent /api/search / crawl indexing / paper
# reindex all hit the shared instance instantly.
_SHARED_EMBEDDERS: Dict[str, BaseEmbedder] = {}
_SHARED_LOCK = threading.Lock()


def get_embedder(model_name: str | None = None, **kwargs) -> BaseEmbedder:
    """Return the process-wide shared embedder for ``model_name``.

    Double-checked locking so concurrent first-touch (e.g. crawl worker
    + search request hitting cold start together) doesn't duplicate the
    expensive model load. ``kwargs`` are only honored on first call;
    subsequent calls ignore them and return the cached instance — this
    matches the singleton contract (kwargs must not vary across callers
    for the same model_name).
    """
    name = model_name or settings.embedding.model_name
    cached = _SHARED_EMBEDDERS.get(name)
    if cached is not None:
        return cached
    with _SHARED_LOCK:
        cached = _SHARED_EMBEDDERS.get(name)
        if cached is not None:
            return cached
        _logger.info("embedder.get_or_create model=%s cold_load_start", name)
        instance = create_embedder(model_name=name, **kwargs)
        _SHARED_EMBEDDERS[name] = instance
        _logger.info(
            "embedder.get_or_create model=%s ready supports_sparse=%s dim=%d",
            name, instance.supports_sparse, instance.dimension,
        )
        return instance


def embed_texts(
    texts: list[str],
    *,
    priority: EmbeddingPriority = "interactive",
) -> EmbeddingResult:
    """Embed text with the process-wide shared model instance."""

    return get_embedder().embed_texts(texts, priority=priority)


def reset_embedder_cache() -> None:
    """Drop all cached embedders. Used by tests to force a fresh load."""
    with _SHARED_LOCK:
        _SHARED_EMBEDDERS.clear()


__all__ = [
    "BaseEmbedder",
    "EmbeddingPriority",
    "EmbeddingResult",
    "create_embedder",
    "embedding_slot",
    "embed_texts",
    "get_embedder",
    "reset_embedder_cache",
]
