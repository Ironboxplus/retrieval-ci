"""Pure fusion for the federated private+public code search route.

No I/O and no Qdrant import here by design (docs/federated_code_search_plan.md A2):
`fuse_code_hits` merges two ALREADY-RETRIEVED candidate pools (one per plane, each
`rerank=False` at the vector-store level) with Reciprocal Rank Fusion, optionally
re-scored by the shared cross-encoder reranker. The route layer is responsible for
running both legs' retrieval (concurrently, via asyncio.gather + to_thread) and for
converting this module's ``FusedResult`` into ``FederatedCodeSearchResponse``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

# RRF constant per the plan: score = 1 / (60 + rank), rank counted from 1 within each
# leg independently. This is the same constant used across the codebase's other RRF
# fusion (dense+sparse hybrid search) for consistent score magnitudes.
_RRF_K = 60
_ORIGIN_SORT_RANK = {"private": 0, "public": 1}  # ties prefer private, per the plan


@dataclass(frozen=True, slots=True)
class CodeFederationCandidate:
    """One leg's retrieval hit, before fusion.

    `score` is that leg's own retrieval score (dense/hybrid/keyword) — it is NOT
    comparable across legs; RRF only uses each candidate's RANK (its position in the
    list passed to `fuse_code_hits`), and `score` only breaks ties between equal RRF
    scores.
    """

    origin: Literal["private", "public"]
    path: str
    start_line: int
    end_line: int
    score: float
    label: str = ""
    language: str | None = None
    code: str = ""
    project: str | None = None    # private only
    host: str | None = None       # private only
    source_id: str | None = None  # public only
    revision: str | None = None   # public only
    domain: str | None = None     # public only


@dataclass(frozen=True, slots=True)
class FusedCodeHit:
    """One fused result. `score` is the final sort score: the reranker score when
    `fusion="rerank"`, else the RRF score (same value as `fusion_rank_score`)."""

    origin: Literal["private", "public"]
    path: str
    start_line: int
    end_line: int
    score: float
    retrieval_score: float
    fusion_rank_score: float
    reranker_score: float | None
    label: str = ""
    language: str | None = None
    code: str = ""
    project: str | None = None
    host: str | None = None
    source_id: str | None = None
    revision: str | None = None
    domain: str | None = None


@dataclass(frozen=True, slots=True)
class FusedResult:
    hits: tuple[FusedCodeHit, ...]
    fusion: Literal["rrf", "rerank"]
    rerank_applied: bool
    # F2-b timing instrumentation only — does not affect ranking. fuse_ms covers the
    # RRF/dedup/sort step, which always runs; rerank_ms covers only the cross-encoder
    # call and is None whenever no reranker was given or there was nothing to score.
    fuse_ms: float = 0.0
    rerank_ms: float | None = None


class CodeFederationReranker(Protocol):
    def score_pairs(self, query: str, texts: list[str]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    candidate: CodeFederationCandidate
    retrieval_score: float
    fusion_rank_score: float


def _rrf_leg(hits: Sequence[CodeFederationCandidate]) -> list[_RankedCandidate]:
    return [
        _RankedCandidate(
            candidate=hit,
            retrieval_score=hit.score,
            fusion_rank_score=1.0 / (_RRF_K + rank),
        )
        for rank, hit in enumerate(hits, start=1)
    ]


def _dedup_key(candidate: CodeFederationCandidate) -> tuple:
    # Exact-dedup key. Origin AND the owning scope (private project / public
    # source+revision) are part of it: the same chunk in two repositories is two
    # results, exactly as /api/code/search returns it. Dropping the scope made
    # owner-wide federated search lose cross-scope copies the old route keeps.
    return (candidate.origin, candidate.project, candidate.source_id, candidate.revision,
            candidate.path, candidate.start_line, candidate.end_line, candidate.code)


def _dedupe(ranked: list[_RankedCandidate]) -> list[_RankedCandidate]:
    seen: set[tuple] = set()
    deduped: list[_RankedCandidate] = []
    for item in ranked:
        key = _dedup_key(item.candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _rrf_sort_key(item: _RankedCandidate):
    c = item.candidate
    return (
        -item.fusion_rank_score,
        -item.retrieval_score,
        _ORIGIN_SORT_RANK[c.origin],
        c.path,
        c.start_line,
    )


def _to_fused_hit(
    item: _RankedCandidate, *, score: float, reranker_score: float | None,
) -> FusedCodeHit:
    c = item.candidate
    return FusedCodeHit(
        origin=c.origin, path=c.path, start_line=c.start_line, end_line=c.end_line,
        score=score, retrieval_score=item.retrieval_score,
        fusion_rank_score=item.fusion_rank_score, reranker_score=reranker_score,
        label=c.label, language=c.language, code=c.code,
        project=c.project, host=c.host,
        source_id=c.source_id, revision=c.revision, domain=c.domain,
    )


def _rrf_result(
    ranked_sorted: list[_RankedCandidate], top_k: int, *, fuse_ms: float,
) -> FusedResult:
    hits = tuple(
        _to_fused_hit(item, score=item.fusion_rank_score, reranker_score=None)
        for item in ranked_sorted[:top_k]
    )
    return FusedResult(hits=hits, fusion="rrf", rerank_applied=False, fuse_ms=fuse_ms)


def _validated_scores(raw_scores, expected_len: int) -> list[float] | None:
    """None on ANY malformed reranker output — the caller falls back to RRF order."""
    try:
        scores = [float(s) for s in raw_scores]
    except (TypeError, ValueError):
        return None
    if len(scores) != expected_len:
        return None
    if not all(math.isfinite(s) for s in scores):
        return None
    return scores


def fuse_code_hits(
    private_hits: Sequence[CodeFederationCandidate],
    public_hits: Sequence[CodeFederationCandidate],
    *,
    query: str,
    top_k: int,
    reranker: CodeFederationReranker | None = None,
) -> FusedResult:
    """Fuse two already-retrieved (rerank=False) candidate pools into one ranking.

    Each leg is RRF-scored independently (rank 1..len(leg)), then the union is exact-
    deduplicated and truncated to `top_k` AFTER the global sort — never by first taking
    each leg's own top_k. With `reranker` given, the full deduplicated union (not the
    RRF top_k) is scored once; any reranker failure (exception, wrong score count, a
    non-finite score) falls back to the RRF order with `rerank_applied=False`, matching
    `PublicCodeSearchService._rerank`'s fail-open contract.
    """
    fuse_start = time.perf_counter()
    ranked = _dedupe(_rrf_leg(private_hits) + _rrf_leg(public_hits))
    rrf_sorted = sorted(ranked, key=_rrf_sort_key)
    fuse_ms = (time.perf_counter() - fuse_start) * 1000.0

    if reranker is None or not ranked:
        return _rrf_result(rrf_sorted, top_k, fuse_ms=fuse_ms)

    texts = [item.candidate.code for item in ranked]
    rerank_start = time.perf_counter()
    try:
        raw_scores = reranker.score_pairs(query, texts)
    except Exception:  # noqa: BLE001 - fail open to RRF order, matching public-code search
        return _rrf_result(rrf_sorted, top_k, fuse_ms=fuse_ms)
    rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
    scores = _validated_scores(raw_scores, len(ranked))
    if scores is None:
        return _rrf_result(rrf_sorted, top_k, fuse_ms=fuse_ms)

    reranked = sorted(
        zip(ranked, scores),
        key=lambda pair: (
            -pair[1],
            -pair[0].fusion_rank_score,
            _ORIGIN_SORT_RANK[pair[0].candidate.origin],
            pair[0].candidate.path,
            pair[0].candidate.start_line,
        ),
    )
    hits = tuple(
        _to_fused_hit(item, score=rerank_score, reranker_score=rerank_score)
        for item, rerank_score in reranked[:top_k]
    )
    return FusedResult(
        hits=hits, fusion="rerank", rerank_applied=True,
        fuse_ms=fuse_ms, rerank_ms=rerank_ms)
