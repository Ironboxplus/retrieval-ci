"""
Embedder abstraction layer.

Provides a unified interface for embedding models that may output
dense vectors, sparse vectors, or both. Every model runs as a vLLM service
(``EMBEDDING__SERVICE_URL``); ``VllmBgeM3Embedder`` is the backend's client of it, built on
``vllm_client.encode``. The in-process embedders are retired and raise when constructed.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ContextManager, Dict, Iterator, Literal, Optional, TYPE_CHECKING

import numpy as np

from core.indexing.embedders.vllm_client import ServiceError

# ``config`` is imported where it is used, not here: ``bge_m3_core`` and ``vllm_client`` live in
# this package and the standalone embed worker imports them, so importing the package must not
# pull in the project's settings layer (pydantic_settings).

_logger = logging.getLogger(__name__)

SERVICE_POLL_S = 1.0

if TYPE_CHECKING:
    from app.models.schemas import Chunk


EmbeddingPriority = Literal["interactive", "background"]

# vLLM's priority scale: the lower value runs first (the service is started with
# ``--scheduling-policy priority``). Business requests are 0, background indexing 5.
SERVICE_PRIORITY: Dict[str, int] = {"interactive": 0, "background": 5}

# Background work resumes this long after the last interactive call in this process (user
# decision 2026-11-14: foreground = business requests, background starts again 10 s after the
# foreground goes idle).
BACKGROUND_QUIET_S = 10.0


class ForegroundGate:
    """Keeps background embedding out of the way of business requests in this process.

    An interactive call marks the gate busy while it runs and stamps the time it ended. A
    background call waits until no interactive call is running and ``quiet_s`` seconds have passed
    since the last one ended. Waiting costs nothing but time: the service orders whatever reaches it
    by priority, this only keeps a background stream from starting while a person is searching.
    """

    def __init__(self, quiet_s: float = BACKGROUND_QUIET_S, clock=time.monotonic) -> None:
        if quiet_s < 0:
            raise ValueError("quiet_s must not be negative")
        self._quiet_s = quiet_s
        self._clock = clock
        self._condition = threading.Condition()
        self._running = 0
        self._last_end: Optional[float] = None

    @contextmanager
    def acquire(self, priority: EmbeddingPriority) -> Iterator[None]:
        if priority not in SERVICE_PRIORITY:
            raise ValueError(f"unknown embedding priority: {priority}")
        if priority == "interactive":
            with self._condition:
                self._running += 1
            try:
                yield
            finally:
                with self._condition:
                    self._running -= 1
                    self._last_end = self._clock()
                    self._condition.notify_all()
            return
        self._wait_until_quiet()
        yield

    def _wait_until_quiet(self) -> float:
        """Block until background work may run; returns the seconds waited."""
        started = self._clock()
        with self._condition:
            while True:
                if self._running:
                    self._condition.wait(timeout=1.0)
                    continue
                remaining = 0.0 if self._last_end is None else self._last_end + self._quiet_s - self._clock()
                if remaining <= 0:
                    return self._clock() - started
                self._condition.wait(timeout=remaining)


_FOREGROUND_GATE = ForegroundGate()


def embedding_slot(priority: EmbeddingPriority = "interactive") -> ContextManager[None]:
    """Mark one encode call of the shared embedder: interactive calls hold background back for
    ``BACKGROUND_QUIET_S``, background calls wait for that quiet. The model itself is the service."""
    return _FOREGROUND_GATE.acquire(priority)


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
        """Embed one request.

        The token limit belongs to the model's profile, not the call: a text over it
        raises instead of being cut.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    def supports_sparse(self) -> bool:
        return False

    def embed_chunks(self, chunks: list["Chunk"]) -> tuple[list["Chunk"], Optional[list[dict[str, float]]]]:
        """Embed chunks in place as background work: one call, the client batches it."""
        result = embed_in_pieces(self, [c.content for c in chunks], "background")
        for i, chunk in enumerate(chunks):
            chunk.embedding = result.dense[i].tolist()
        return chunks, result.sparse


class VllmBgeM3Embedder(BaseEmbedder):
    """BGE-M3 (dense + sparse) from the vLLM embedding service.

    Texts are tokenized here with the profile's tokenizer and only token ids are sent
    (``vllm_client.encode``), under the profile's limits: a text over the model limit raises
    ``InputTooLong`` before a request leaves. ``priority`` becomes the request's top-level vLLM
    priority; a background call also waits for the foreground to be idle (``ForegroundGate``).
    Constructing one does not contact the service.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", *, base_url: str | None = None):
        from config import settings
        from core.indexing.embedding_profile import get_profile

        self._model_name = model_name
        self._profile = get_profile(model_name)
        self._base_url = base_url or settings.embedding.service_url

    @property
    def dimension(self) -> int:
        return self._profile.dimension

    @property
    def supports_sparse(self) -> bool:
        return self._profile.sparse

    def check_service(self, *, wait_s: float = 0.0) -> bool:
        """True once the service answers and serves this profile (revision, ``max_model_len``).

        A service that answers with something else raises ``ServiceError`` at once: its vectors would
        carry this profile's fingerprint without being this profile's. One that does not answer is
        waited for up to ``wait_s``; if it never does, that is logged and False is returned (the
        backend then runs, and every embedding call raises until the service is up).
        """
        from core.indexing.embedders import vllm_client

        deadline = time.monotonic() + wait_s
        while True:
            try:
                vllm_client.check_service(self._profile, self._base_url)
                return True
            except vllm_client.ServiceError as exc:
                if exc.__cause__ is None:        # only "not answering" is raised from an underlying error
                    raise
                if time.monotonic() >= deadline:
                    _logger.error("embedding service not reachable after %.0f s: %s", wait_s, exc)
                    return False
            time.sleep(min(SERVICE_POLL_S, max(0.0, deadline - time.monotonic())))

    def embed_texts(
        self, texts: list[str], *, priority: EmbeddingPriority = "interactive",
    ) -> EmbeddingResult:
        from core.indexing.embedders import vllm_client

        with embedding_slot(priority):
            dense, sparse = vllm_client.encode(
                texts, self._profile, base_url=self._base_url, priority=SERVICE_PRIORITY[priority])
        return EmbeddingResult(
            dense=dense,
            sparse=[{str(token_id): weight for token_id, weight in weights.items()} for weights in sparse],
        )


def embed_in_pieces(
    embedder: BaseEmbedder, texts: list[str], priority: EmbeddingPriority,
) -> EmbeddingResult:
    """Embed ``texts`` in one call.

    Kept as the name callers import. It used to cut background work into 8-text pieces so a waiting
    query could take the in-process model between them; the model is a service that schedules by
    priority now and the client batches, so a request is never split here.
    """
    return embedder.embed_texts(texts, priority=priority)


def create_embedder(model_name: str | None = None, **kwargs) -> BaseEmbedder:
    """Factory: create the right embedder based on model name.

    Prefer ``get_embedder(model_name)`` for callers that want the process-wide shared
    instance. A fresh one costs nothing (it is a client), but only BGE-M3 is served: any other
    model name reaches the retired in-process embedder, which raises.
    """
    from config import settings, EMBEDDING_MODELS

    name = model_name or settings.embedding.model_name
    model_info = EMBEDDING_MODELS.get(name, {})

    if model_info.get("supports_sparse"):
        return VllmBgeM3Embedder(model_name=name, **kwargs)

    from core.indexing.embedders.sentence_transformer import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model_name=name, **kwargs)


# W46 (2026-05-20): process-wide cached embedder: every in-process caller goes through
# ``get_embedder()`` so the instance is created once per process (per model_name key).
_SHARED_EMBEDDERS: Dict[str, BaseEmbedder] = {}
_SHARED_LOCK = threading.Lock()


def get_embedder(model_name: str | None = None, **kwargs) -> BaseEmbedder:
    """Return the process-wide shared embedder for ``model_name``.

    Double-checked locking so concurrent first-touch doesn't build two instances. ``kwargs``
    are only honored on first call; subsequent calls ignore them and return the cached
    instance — this matches the singleton contract (kwargs must not vary across callers
    for the same model_name).
    """
    from config import settings

    name = model_name or settings.embedding.model_name
    cached = _SHARED_EMBEDDERS.get(name)
    if cached is not None:
        return cached
    with _SHARED_LOCK:
        cached = _SHARED_EMBEDDERS.get(name)
        if cached is not None:
            return cached
        instance = create_embedder(model_name=name, **kwargs)
        _SHARED_EMBEDDERS[name] = instance
        _logger.info(
            "embedder.get_or_create model=%s ready supports_sparse=%s dim=%d",
            name, instance.supports_sparse, instance.dimension,
        )
        return instance


def require_embedding_service(*, wait_s: float | None = None) -> bool:
    """Start-up check of the shared embedder's service: ``VllmBgeM3Embedder.check_service``.

    ``wait_s`` defaults to ``settings.embedding.service_wait_s``.

    Raises ``ServiceError`` for a service that serves another revision or length limit; returns
    False (after logging) for one that stays silent.
    """
    from config import settings

    return get_embedder().check_service(wait_s=settings.embedding.service_wait_s if wait_s is None else wait_s)


def embed_texts(
    texts: list[str],
    *,
    priority: EmbeddingPriority = "interactive",
) -> EmbeddingResult:
    """Embed with the shared embedder (the vLLM service); background calls wait for the foreground."""
    return embed_in_pieces(get_embedder(), texts, priority)


def reset_embedder_cache() -> None:
    """Drop all cached embedders. Used by tests to force a fresh instance."""
    with _SHARED_LOCK:
        _SHARED_EMBEDDERS.clear()


__all__ = [
    "BACKGROUND_QUIET_S",
    "BaseEmbedder",
    "EmbeddingPriority",
    "EmbeddingResult",
    "ForegroundGate",
    "SERVICE_PRIORITY",
    "ServiceError",
    "VllmBgeM3Embedder",
    "create_embedder",
    "embedding_slot",
    "embed_in_pieces",
    "embed_texts",
    "get_embedder",
    "require_embedding_service",
    "reset_embedder_cache",
]
