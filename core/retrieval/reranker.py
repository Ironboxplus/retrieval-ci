"""
Reranker — re-scores candidate results using a cross-encoder model.

Takes top-N candidates from retrieval and re-ranks them to top-K
with much higher precision. The cross-encoder runs as a vLLM service
(``RERANKER__SERVICE_URL``) and ``VllmReranker`` is its client. The in-process torch
``BgeReranker`` is retired and raises when constructed.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

from config import RERANK_QUERY_TOKENS, settings
from core.net.token_bucket import TokenBucket
from app.models.schemas import SearchResult
from core.inference_retired import RERANKER_INSTEAD, RetiredInferencePath
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


# ``served-model-name`` of the reranking service (deploy/compute88/vllm_reranker.yaml); vLLM
# routes the request by it.
RERANKER_SERVED_MODEL = "bge-reranker-v2-m3"
# Pairs per request. A cancel can land between requests, so this is also the cancel latency.
RERANK_REQUEST_PAIRS = 32
# Requests in flight at once: the service batches across them (4 measured as fast as 16).
RERANK_IN_FLIGHT = 4
# Scoring is business traffic: the top of vLLM's priority scale (lower runs first).
RERANK_PRIORITY = 0
RERANK_TIMEOUT_S = 120.0


class RerankerServiceError(RuntimeError):
    """The reranking service did not answer, or answered something that is not a score list."""


@dataclass(frozen=True)
class _Endpoint:
    """Where one scoring request goes and what it carries: the one scoring path serves the
    local vLLM service and the hosted reranker, which speak the same ``/rerank`` shape."""

    base_url: str
    model: str
    headers: dict = field(default_factory=dict, repr=False)   # holds the key online: never repr'd
    extra_body: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineRerankConfig:
    """The hosted reranker (SiliconFlow) and the limits around it. ``from_settings`` returns None
    when no key is configured: the online path is then off and scoring is local only."""

    base_url: str
    model: str
    api_key: str = field(repr=False)
    rpm: int
    tpm: int
    max_wait_s: float
    connect_timeout_s: float
    timeout_s: float
    failures_to_open: int
    cooldown_s: float

    @classmethod
    def from_settings(cls) -> "OnlineRerankConfig | None":
        cfg = settings.reranker
        key = cfg.online_api_key.get_secret_value().strip()
        if not key:
            return None
        return cls(
            base_url=cfg.online_url.rstrip("/"), model=cfg.online_model, api_key=key,
            rpm=cfg.online_rpm, tpm=cfg.online_tpm, max_wait_s=cfg.online_max_wait_s,
            connect_timeout_s=cfg.online_connect_timeout_s, timeout_s=cfg.online_timeout_s,
            failures_to_open=cfg.online_failures_to_open, cooldown_s=cfg.online_cooldown_s,
        )


class _Circuit:
    """Consecutive-failure breaker for the online path. After ``failures_to_open`` failures in a
    row it is open for ``cooldown_s``; then exactly one trial request is let through (half-open):
    its success closes the circuit, its failure re-opens it for another cooldown."""

    def __init__(self, failures_to_open: int, cooldown_s: float):
        self._threshold = failures_to_open
        self._cooldown = cooldown_s
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0
        self._probing = False

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def allow(self) -> bool:
        with self._lock:
            if self._failures < self._threshold:
                return True
            if not self._probing and time.monotonic() >= self._open_until:
                self._probing = True
                return True
            return False

    def abandon(self) -> None:
        """An allowed request was never sent (no budget): it is neither a success nor a failure."""
        with self._lock:
            self._probing = False

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._probing = False

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._probing = False
            if self._failures >= self._threshold:
                self._open_until = time.monotonic() + self._cooldown


# Tokens of a (query, passage) pair are estimated as characters / 3 before the request: no
# tokenizer on the request path, and the estimate is settled to the response's real usage.
_CHARS_PER_TOKEN_ESTIMATE = 3
BACKEND_ONLINE = "online"
BACKEND_LOCAL = "local"
BACKEND_MIXED = "mixed"


class VllmReranker(BaseReranker):
    """Cross-encoder scores from the vLLM reranking service (bge-reranker-v2-m3).

    With an API key configured (``OnlineRerankConfig``) each request of pairs goes to the hosted
    reranker first and falls back to the service for that request when the online budget refuses
    it or the call fails; ``last_backend`` tells which scored the calling thread's latest
    ``score_pairs``.

    ``score_pairs`` returns the service's default (sigmoid) score of each (query, text) pair, in
    input order: the scale ``FlagReranker(normalize=True)`` had. A pair is cut, not refused, as
    before:

    * the query is clipped here to ``RERANK_QUERY_TOKENS`` tokens (FlagReranker's
      ``query_max_length``) with the production tokenizer, which the reranker's vocabulary shares;
    * the service cuts the whole pair to ``settings.reranker.max_length`` tokens
      (``truncate_prompt_tokens``, a top-level request field: inside the body it is ignored), so
      only the end of a long passage is lost.

    Texts go out in requests of ``RERANK_REQUEST_PAIRS`` pairs, ``RERANK_IN_FLIGHT`` at a time.
    ``should_cancel`` is polled before each request is sent, so a cancelled rerank stops after
    at most the requests already in flight.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *, base_url: str | None = None,
                 batch_size: int | None = None, max_length: int | None = None,
                 in_flight: int = RERANK_IN_FLIGHT, timeout_s: float = RERANK_TIMEOUT_S,
                 client: httpx.Client | None = None,
                 online: OnlineRerankConfig | None | bool = True):
        self._model_name = model_name
        self._base_url = (base_url or settings.reranker.service_url).rstrip("/")
        self._batch_size = RERANK_REQUEST_PAIRS if batch_size is None else batch_size
        self._max_length = settings.reranker.max_length if max_length is None else max_length
        if self._batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if in_flight < 1:
            raise ValueError("in_flight must be at least 1")
        if self._max_length < 64:
            raise ValueError("max_length must be at least 64")
        self._in_flight = in_flight
        # trust_env=False: the host's HTTP proxy must never carry loopback traffic.
        self._client = client or httpx.Client(timeout=timeout_s, trust_env=False)
        self._local = _Endpoint(
            self._base_url, RERANKER_SERVED_MODEL,
            extra_body={"truncate_prompt_tokens": self._max_length, "priority": RERANK_PRIORITY},
        )
        # ``online=True``: from settings (None without a key); None/False: local only.
        online_cfg = OnlineRerankConfig.from_settings() if online is True else (online or None)
        self._online_cfg = online_cfg
        self._online: _Endpoint | None = None
        self._online_client: httpx.Client | None = None
        self._bucket: TokenBucket | None = None
        self._circuit: _Circuit | None = None
        if online_cfg is not None:
            self._online = _Endpoint(
                online_cfg.base_url, online_cfg.model,
                headers={"Authorization": f"Bearer {online_cfg.api_key}"},
                extra_body={"return_documents": False},
            )
            # One pooled keep-alive client: a cold connect to the provider took 11.4 s on .88.
            self._online_client = httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(online_cfg.timeout_s, connect=online_cfg.connect_timeout_s),
            )
            self._bucket = TokenBucket(online_cfg.tpm, online_cfg.rpm)
            self._circuit = _Circuit(online_cfg.failures_to_open, online_cfg.cooldown_s)
        self._tls = threading.local()

    @property
    def online_enabled(self) -> bool:
        return self._online is not None

    @property
    def online_failures(self) -> int:
        """Consecutive online failures so far (0 without an online path)."""
        return 0 if self._circuit is None else self._circuit.failures

    def last_backend(self) -> str | None:
        """``online`` / ``local`` / ``mixed`` for the latest ``score_pairs`` of the calling
        thread, None before any."""
        return getattr(self._tls, "backend", None)

    def _load_model(self) -> "VllmReranker":
        """Startup hook kept for ``main.py``: there is no model to load, the service is the model."""
        return self

    @staticmethod
    def _clip_query(query: str) -> str:
        from core.indexing.embedding_profile import ascii_within_limit, get_profile
        from core.indexing.token_budget import SEGMENT_CHARS, token_offsets

        if ascii_within_limit(query, RERANK_QUERY_TOKENS):
            return query
        offsets = token_offsets(query[:SEGMENT_CHARS], get_profile())
        return query if len(offsets) <= RERANK_QUERY_TOKENS else query[:offsets[RERANK_QUERY_TOKENS - 1][1]]

    @staticmethod
    def _post(endpoint: _Endpoint, client: httpx.Client, query: str,
              texts: list[str]) -> tuple[list[float], int | None]:
        """One ``/rerank`` request: the scores in input order and the provider's reported input
        tokens (None when it reports none)."""
        body = {"model": endpoint.model, "query": query, "documents": texts, **endpoint.extra_body}
        try:
            response = client.post(f"{endpoint.base_url}/rerank", json=body, headers=endpoint.headers)
        except httpx.HTTPError as exc:
            raise RerankerServiceError(f"reranker service {endpoint.base_url} unreachable: {exc!r}") from exc
        if response.status_code != 200:
            raise RerankerServiceError(
                f"reranker service {endpoint.base_url} answered {response.status_code}: {response.text[:300]}")
        try:
            payload = response.json()
            scores: list[float | None] = [None] * len(texts)
            for item in payload["results"]:
                scores[int(item["index"])] = float(item["relevance_score"])
            tokens = ((payload.get("meta") or {}).get("tokens") or {}).get("input_tokens")
            tokens = int(tokens) if tokens is not None else None
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise RerankerServiceError(
                f"reranker service {endpoint.base_url} answered a malformed body: {exc!r}") from exc
        if any(score is None for score in scores):
            raise RerankerServiceError(
                f"reranker service {endpoint.base_url} returned scores for "
                f"{sum(s is not None for s in scores)} of {len(texts)} documents")
        return scores, tokens  # type: ignore[return-value]

    @staticmethod
    def _estimate_tokens(query: str, texts: list[str]) -> int:
        chars = sum(len(query) + len(text) for text in texts)
        return max(1, -(-chars // _CHARS_PER_TOKEN_ESTIMATE))

    def _score_online(self, query: str, texts: list[str]) -> list[float] | None:
        """Scores from the hosted reranker, or None when this request must go local: the circuit is
        open, the budget would make it wait past ``max_wait_s`` (not a failure), or the call failed
        (a failure, counted)."""
        assert self._online is not None and self._bucket is not None and self._circuit is not None
        assert self._online_client is not None and self._online_cfg is not None
        if not self._circuit.allow():
            return None
        # From here the circuit may be half-open with its one trial taken: every path out must
        # end in success(), failure() or abandon(), whatever the exception.
        try:
            estimate = self._estimate_tokens(query, texts)
            entry = self._bucket.try_acquire(estimate, self._online_cfg.max_wait_s)
            if entry is None:
                self._circuit.abandon()
                return None
            scores, used = self._post(self._online, self._online_client, query, texts)
            self._bucket.settle(entry, used if used is not None else estimate)
        except RerankerServiceError as exc:
            self._circuit.failure()
            _logger.warning("online rerank failed, this request goes to the local service: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 -- any online fault must fall back, never fail the search
            self._circuit.failure()
            _logger.warning("online rerank failed (%s at %s), this request goes to the local service",
                            type(exc).__name__, self._online.base_url)
            return None
        self._circuit.success()
        return scores

    def _score_request(self, query: str, texts: list[str]) -> tuple[list[float], str]:
        """Scores of one request and which backend produced them."""
        if self._online is not None:
            scores = self._score_online(query, texts)
            if scores is not None:
                return scores, BACKEND_ONLINE
        return self._post(self._local, self._client, query, texts)[0], BACKEND_LOCAL

    def score_pairs(
        self,
        query: str,
        texts: list[str],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[float]:
        if not texts:
            return []
        query = self._clip_query(query)
        step = self._batch_size
        scores: list[float] = []
        backends: list[str] = []
        pending: list[Future] = []
        pool = ThreadPoolExecutor(max_workers=self._in_flight, thread_name_prefix="rerank")

        def collect() -> None:
            batch_scores, backend = pending.pop(0).result()
            scores.extend(batch_scores)
            backends.append(backend)

        try:
            for offset in range(0, len(texts), step):
                if should_cancel is not None and should_cancel():
                    raise RerankCancelledError(f"rerank cancelled after {offset}/{len(texts)} pairs")
                pending.append(pool.submit(self._score_request, query, texts[offset:offset + step]))
                if len(pending) >= self._in_flight:
                    collect()
            while pending:
                collect()
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False)
        self._record_backends(backends)
        return scores

    def _record_backends(self, backends: list[str]) -> None:
        online = backends.count(BACKEND_ONLINE)
        local = backends.count(BACKEND_LOCAL)
        self._tls.backend = BACKEND_MIXED if online and local else (BACKEND_ONLINE if online else BACKEND_LOCAL)
        if self._online is not None:
            _logger.info("rerank.backend=%s online_batches=%d local_batches=%d",
                         self._tls.backend, online, local)

    def rerank(self, query: str, results: list[SearchResult],
               top_k: int = 10, *,
               should_cancel: Callable[[], bool] | None = None) -> list[SearchResult]:
        return _rerank_by_scores(self, query, results, top_k, should_cancel)


class BgeReranker(BaseReranker):
    """RETIRED: BGE reranker loaded into this process with FlagEmbedding's FlagReranker.

    Kept for reference only; constructing it raises. Scoring goes through the vLLM reranking
    service (``get_shared_reranker()`` / ``VllmReranker``).
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str | None = None, *, batch_size: int | None = None,
                 idle_cache_limit_mib: int | None = None):
        raise RetiredInferencePath("BgeReranker (in-process FlagReranker)", RERANKER_INSTEAD)
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
                        max_length=settings.reranker.max_length,
                        query_max_length=RERANK_QUERY_TOKENS,
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
        return _rerank_by_scores(self, query, results, top_k, should_cancel)


def _rerank_by_scores(reranker: BaseReranker, query: str, results: list[SearchResult], top_k: int,
                      should_cancel: Callable[[], bool] | None) -> list[SearchResult]:
    """The top ``top_k`` of ``results`` by ``reranker.score_pairs``, each carrying its score."""
    if not results:
        return []

    scores = reranker.score_pairs(
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


def rerank_backend_of(reranker) -> str | None:
    """Which backend scored the calling thread's latest rerank on ``reranker``: ``online`` /
    ``local`` / ``mixed``, or None when it has no online path to report (a stub, the retired
    torch class). Call it on the thread that called ``rerank()``, right after it returns."""
    report = getattr(reranker, "last_backend", None)
    return report() if callable(report) else None


def create_reranker(model_name: Optional[str] = None) -> BaseReranker:
    """Factory for creating reranker instances: the client of the reranking service."""
    name = model_name or "BAAI/bge-reranker-v2-m3"
    return VllmReranker(model_name=name)


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
