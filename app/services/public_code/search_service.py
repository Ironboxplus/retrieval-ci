"""Application search service for flat and source-scoped public code retrieval."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
from typing import Any, Callable, Protocol, Sequence

from app.services.public_code.repository import PublicCodeSearchScope
from app.services.public_code.vector_store import PublicCodeVectorStore


logger = logging.getLogger(__name__)


# Reranking needs a wider retrieval pool than the result size. The capped Qdrant
# retrieval pool is also the cross-encoder's upper bound: every exact-deduplicated
# candidate must be scored so a relevant tail hit cannot be hidden by an artificial
# rerank budget.
_RERANK_RETRIEVAL_MULTIPLIER = 4
_MIN_RERANK_RETRIEVAL_CANDIDATES = 20
_MAX_RERANK_RETRIEVAL_CANDIDATES = 50
_MAX_RERANK_CANDIDATES_PER_FILE = 2


class PublicCodeSearchScopeReader(Protocol):
    def resolve_search_scope(
        self,
        *,
        source_ids: Sequence[str],
        domains: Sequence[str],
    ) -> PublicCodeSearchScope: ...


@dataclass(frozen=True, slots=True)
class PublicCodeSearchRequest:
    query: str
    top_k: int = 10
    mode: str = "hybrid"
    source_ids: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    language: str | None = None
    rerank: bool = False

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("public code search query must not be empty")
        if not 1 <= self.top_k <= 50:
            raise ValueError("public code search top_k must be between 1 and 50")
        if self.mode not in {"dense", "hybrid", "keyword"}:
            raise ValueError(f"unsupported public code search mode: {self.mode}")


@dataclass(frozen=True, slots=True)
class PublicCodeSearchHit:
    score: float
    retrieval_score: float
    reranker_score: float | None
    source_id: str
    revision: str
    domain: str
    path: str
    language: str
    start_line: int
    end_line: int
    label: str
    content: str
    code: str


@dataclass(frozen=True, slots=True)
class PublicCodeSearchResponse:
    query: str
    selected_source_ids: tuple[str, ...]
    hits: tuple[PublicCodeSearchHit, ...]
    requested_mode: str
    effective_mode: str
    notice: str | None
    rerank_applied: bool
    candidate_count: int
    rerank_candidate_count: int


class PublicCodeSearchService:
    """Thin orchestration over shared embedding/retrieval infrastructure."""

    def __init__(
        self,
        *,
        repository: PublicCodeSearchScopeReader,
        vector_store: PublicCodeVectorStore,
        embed_fn: Callable[[Sequence[str]], list[list[float]]] | None = None,
        sparse_embed_fn: Callable[[Sequence[str]], list[dict[str, float]]] | None = None,
        reranker: Any | None = None,
    ):
        self._repository = repository
        self._vector_store = vector_store
        self._embed_fn = embed_fn
        self._sparse_embed_fn = sparse_embed_fn
        self._reranker = reranker
        self._use_shared_dual_embedder = embed_fn is None and sparse_embed_fn is None

    def search(self, request: PublicCodeSearchRequest) -> PublicCodeSearchResponse:
        """Search selected source scopes in one Qdrant request, then optionally rerank."""

        scope = self._repository.resolve_search_scope(
            source_ids=request.source_ids,
            domains=request.domains,
        )
        if not scope.active_revisions:
            return PublicCodeSearchResponse(
                query=request.query,
                selected_source_ids=scope.source_ids,
                hits=(),
                requested_mode=request.mode,
                effective_mode=request.mode,
                notice=None,
                rerank_applied=False,
                candidate_count=0,
                rerank_candidate_count=0,
            )

        dense_vector, sparse_vector = self._embed(request.query, mode=request.mode)
        sparse_available = bool(sparse_vector and self._vector_store.supports_sparse)
        # A semantic neighbour can help a behaviour query, but it is not an honest
        # result for an exact token search. Keep keyword strict instead of relabeling
        # a dense fallback as lexical retrieval.
        if request.mode == "keyword" and not sparse_available:
            return PublicCodeSearchResponse(
                query=request.query,
                selected_source_ids=scope.source_ids,
                hits=(),
                requested_mode=request.mode,
                effective_mode="unavailable",
                notice=(
                    "keyword search is unavailable because sparse lexical retrieval is "
                    "unavailable for this query or index; it does not substitute dense "
                    "results. Re-index this scope to write sparse vectors, then retry "
                    "mode='keyword'."
                ),
                rerank_applied=False,
                candidate_count=0,
                rerank_candidate_count=0,
            )

        effective_mode = request.mode if sparse_available else "dense"
        notice = None
        if request.mode == "hybrid" and effective_mode == "dense":
            notice = (
                "hybrid search used dense-only fallback because sparse lexical retrieval "
                "is unavailable for this query or index. Re-index this scope to restore "
                "hybrid lexical fusion."
            )
        retrieval_limit = (
            _rerank_retrieval_limit(request.top_k)
            if request.rerank
            else request.top_k
        )
        raw_hits = self._vector_store.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector if sparse_available else None,
            mode=effective_mode,
            top_k=retrieval_limit,
            active_revisions=scope.active_revisions,
            source_ids=scope.source_ids,
            domains=(),
            language=request.language,
        )
        hits = tuple(_public_code_hit_from_result(hit) for hit in raw_hits)
        rerank_candidates: tuple[PublicCodeSearchHit, ...] = ()
        rerank_applied = False
        if request.rerank and hits:
            rerank_candidates = _select_rerank_candidates(hits)
            reranked_hits = self._rerank(request.query, rerank_candidates)
            if reranked_hits is not None:
                hits = reranked_hits
                rerank_applied = True
        return PublicCodeSearchResponse(
            query=request.query,
            selected_source_ids=scope.source_ids,
            hits=hits[:request.top_k],
            requested_mode=request.mode,
            effective_mode=effective_mode,
            notice=notice,
            rerank_applied=rerank_applied,
            candidate_count=len(raw_hits),
            rerank_candidate_count=len(rerank_candidates),
        )

    def _embed(
        self,
        query: str,
        *,
        mode: str,
    ) -> tuple[list[float], dict[str, float] | None]:
        if self._use_shared_dual_embedder:
            from core.indexing.embedders import embed_texts

            result = embed_texts([query], priority="interactive")
            dense_vector = result.dense[0].tolist()
            sparse_vector = (
                result.sparse[0]
                if mode != "dense" and result.sparse is not None
                else None
            )
            return dense_vector, sparse_vector

        if self._embed_fn is None:
            raise RuntimeError("public code search has no embedding function")
        dense_vector = self._embed_fn([query])[0]
        if mode == "dense" or self._sparse_embed_fn is None:
            return dense_vector, None
        return dense_vector, self._sparse_embed_fn([query])[0]

    def _rerank(
        self,
        query: str,
        hits: tuple[PublicCodeSearchHit, ...],
    ) -> tuple[PublicCodeSearchHit, ...] | None:
        try:
            raw_scores = self._reranker_or_default().score_pairs(
                query,
                [hit.content for hit in hits],
            )
        except Exception as exc:
            logger.warning(
                "public-code rerank raised %s; preserving original retrieval order",
                type(exc).__name__,
            )
            return None
        if isinstance(raw_scores, (str, bytes, bytearray)):
            logger.warning(
                "public-code rerank returned a text/binary score scalar; preserving original retrieval order"
            )
            return None
        try:
            scores = list(raw_scores)
        except Exception:
            logger.warning(
                "public-code rerank returned a non-iterable score output; preserving original retrieval order"
            )
            return None
        if len(scores) != len(hits):
            logger.warning(
                "public-code rerank returned %d scores for %d candidates; preserving original retrieval order",
                len(scores),
                len(hits),
            )
            return None
        try:
            finite_scores = [float(score) for score in scores]
        except Exception:
            logger.warning(
                "public-code rerank returned non-numeric scores; preserving original retrieval order"
            )
            return None
        if not all(math.isfinite(score) for score in finite_scores):
            logger.warning(
                "public-code rerank returned non-finite scores; preserving original retrieval order"
            )
            return None
        reranked = [
            replace(hit, score=score, reranker_score=score)
            for hit, score in zip(hits, finite_scores)
        ]
        return tuple(sorted(reranked, key=_reranked_hit_sort_key))

    def _reranker_or_default(self):
        if self._reranker is None:
            from core.retrieval.reranker import get_shared_reranker

            self._reranker = get_shared_reranker()
        return self._reranker


def _rerank_retrieval_limit(top_k: int) -> int:
    """Bound the wider Qdrant pool independently from the caller's result size."""

    return min(
        max(top_k * _RERANK_RETRIEVAL_MULTIPLIER, _MIN_RERANK_RETRIEVAL_CANDIDATES),
        _MAX_RERANK_RETRIEVAL_CANDIDATES,
    )


def _public_code_hit_from_result(result: dict[str, Any]) -> PublicCodeSearchHit:
    """Attach the retrieval score before a reranker is allowed to replace ``score``."""

    raw = dict(result)
    score = float(raw.pop("score", 0.0))
    retrieval_score = float(raw.pop("retrieval_score", score))
    reranker_score = raw.pop("reranker_score", None)
    return PublicCodeSearchHit(
        score=score,
        retrieval_score=retrieval_score,
        reranker_score=(float(reranker_score) if reranker_score is not None else None),
        **raw,
    )


def _select_rerank_candidates(
    hits: Sequence[PublicCodeSearchHit],
) -> tuple[PublicCodeSearchHit, ...]:
    """Remove exact duplicates and softly order same-file chunks before reranking.

    The per-file cap is ordering-only: every exact-deduplicated candidate remains in
    the cross-encoder batch, including a narrow scope's overflow chunks. Input order
    is preserved inside each tier.
    """

    seen_chunks: set[tuple[str, str, str, int, int, str, str, str]] = set()
    chunks_per_file: dict[tuple[str, str, str], int] = {}
    diverse: list[PublicCodeSearchHit] = []
    same_file_overflow: list[PublicCodeSearchHit] = []
    for hit in hits:
        chunk_identity = (
            hit.source_id,
            hit.revision,
            hit.path,
            hit.start_line,
            hit.end_line,
            hit.label,
            hit.content,
            hit.code,
        )
        if chunk_identity in seen_chunks:
            continue
        seen_chunks.add(chunk_identity)
        file_identity = (hit.source_id, hit.revision, hit.path)
        file_count = chunks_per_file.get(file_identity, 0)
        if file_count < _MAX_RERANK_CANDIDATES_PER_FILE:
            diverse.append(hit)
            chunks_per_file[file_identity] = file_count + 1
        else:
            same_file_overflow.append(hit)
    return tuple(diverse + same_file_overflow)


def _reranked_hit_sort_key(
    hit: PublicCodeSearchHit,
) -> tuple[float, float, str, str, str, int, int, str]:
    """Prefer stronger retrieval evidence before using source location as a tie-break."""

    return (
        -hit.score,
        -hit.retrieval_score,
        hit.source_id,
        hit.revision,
        hit.path,
        hit.start_line,
        hit.end_line,
        hit.label,
    )
