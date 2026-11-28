"""Paper-grouped search helpers (W87).

Two pure, DB-free transforms shared by /api/search and /api/search/stream:

  - build_qdrant_filters(request): collapse the HTTP SearchRequest's payload
    filters (venue / year / venue_tier / data_source / paper_ids) into the
    single dict QdrantStore._build_filter consumes. paper_ids → payload field
    `paper_id` (indexed, MatchAny) so search can be restricted to specific
    papers server-side.

  - group_into_papers(results): collapse a flat, score-desc chunk list into
    paper cards. Granularity is the paper: each card is ranked by its single
    best (max) chunk score and carries the chunks that matched. keep_best_only
    governs only how many chunks a card shows (1 vs all) — never the grouping
    or the ordering.

Kept free of DB / Qdrant / pipeline imports so the unit tests stay fast and
deterministic.
"""
from __future__ import annotations

from typing import Any, Optional

from app.models.schemas import PaperGroup, SearchRequest, SearchResult


def normalize_year_filter(year: Any) -> Any:
    """Normalize SearchRequest.year for the payload filter.

    Accepts None / int / list[int]. Returns the value to push into the filter,
    or None if the field should be skipped. An empty list maps to None so we
    never emit MatchAny([]) (which matches nothing).
    """
    if year is None:
        return None
    if isinstance(year, list):
        return year if year else None
    return year


def build_qdrant_filters(request: SearchRequest) -> dict[str, Any]:
    """Build the Qdrant payload-equality filter dict from a SearchRequest.

    Only includes keys the caller actually set; empty lists are dropped so they
    can't degrade into match-nothing conditions. `paper_ids` maps to the indexed
    `paper_id` payload field, restricting retrieval to the given papers.
    `doc_ids` maps to the indexed `doc_id` field, which is the only way to scope
    a search to an upload or a webpage — those profiles carry no `paper_id` at
    all (core/storage/papers_payload.py), so a paper_ids filter over a book
    returns 0 hits rather than an error. SearchRequest rejects both at once.

    doc_id values are stringified: Qdrant's KEYWORD index holds only the string
    form, and an int-valued condition falls back to a payload scan that times
    out on this collection (measured 2026-09-21 on production `papers`: str
    "114854" returned instantly, int 114854 hit the 60s scroll timeout).
    """
    filters: dict[str, Any] = {}
    if request.data_source:
        filters["data_source"] = request.data_source
    if request.venue:
        filters["venue"] = request.venue
    year_val = normalize_year_filter(request.year)
    if year_val is not None:
        filters["year"] = year_val
    if request.venue_tier:
        filters["venue_tier"] = request.venue_tier
    if request.paper_ids:
        filters["paper_id"] = list(request.paper_ids)
    doc_ids = getattr(request, "doc_ids", None)
    if doc_ids:
        filters["doc_id"] = [str(d) for d in doc_ids]
    return filters


def attach_chunk_positions(
    results: list[SearchResult],
    content_to_index: dict[str, int],
) -> list[SearchResult]:
    """Stamp each result with its document position (chunk_index) by content.

    The Qdrant payload omits positional fields (8-field whitelist), but `content`
    is whitelisted and identical to the SQL chunk text — so an exact content
    match recovers the SQL chunk_index. Used for single-paper in-paper search so
    the UI can offer a document-order view. Results whose content isn't in the
    map are left untouched (no chunk_index key added).
    """
    for r in results:
        idx = content_to_index.get(r.content)
        if idx is not None:
            r.metadata = {**(r.metadata or {}), "chunk_index": idx}
    return results


def _paper_key(r: SearchResult) -> tuple[str, str]:
    """(paper_id, doc_id) for a result. paper_id may be '' for non-paper docs.

    Group key precedence mirrors the pipeline's dedup: business paper_id first,
    falling back to doc_id so webpages/uploads (no paper_id) still group sanely.
    """
    paper_id = ""
    if r.metadata:
        pid = r.metadata.get("paper_id")
        if pid:
            paper_id = str(pid)
    doc_id = str(r.doc_id or "")
    return paper_id, doc_id


def group_into_papers(
    results: list[SearchResult],
    keep_best_only: bool = False,
    max_papers: Optional[int] = None,
) -> list[PaperGroup]:
    """Collapse a flat chunk list into paper cards ranked by max chunk score.

    Args:
        results: chunks (typically already score-desc from the pipeline).
        keep_best_only: if True each card keeps only its single best chunk
            (the "保留最高分" / data-light view); else it keeps every matched
            chunk for that paper.
        max_papers: optional cap on the number of paper cards returned.

    Returns paper cards sorted by `max_score` descending. Within a card the
    chunks are sorted by score descending.
    """
    groups: dict[str, dict[str, Any]] = {}
    for r in results:
        paper_id, doc_id = _paper_key(r)
        key = paper_id or doc_id or (r.chunk_id or "")
        g = groups.get(key)
        if g is None:
            # First (highest-scoring, given score-desc input) chunk seeds the
            # paper-level identity + metadata.
            g = {
                "paper_id": paper_id,
                "doc_id": doc_id,
                "doc_title": r.doc_title or "",
                "source": r.source or "",
                "metadata": dict(r.metadata or {}),
                "max_score": r.score,
                "chunks": [],
            }
            groups[key] = g
        g["chunks"].append(r)
        if r.score > g["max_score"]:
            g["max_score"] = r.score
            # Keep paper-level metadata anchored to the actual best chunk even
            # if input wasn't perfectly score-sorted.
            g["doc_title"] = r.doc_title or g["doc_title"]
            g["source"] = r.source or g["source"]
            g["metadata"] = dict(r.metadata or {})

    paper_list = sorted(groups.values(), key=lambda g: g["max_score"], reverse=True)
    if max_papers is not None:
        paper_list = paper_list[:max_papers]

    out: list[PaperGroup] = []
    for g in paper_list:
        chunks = sorted(g["chunks"], key=lambda c: c.score, reverse=True)
        if keep_best_only:
            chunks = chunks[:1]
        out.append(
            PaperGroup(
                paper_id=g["paper_id"],
                doc_id=g["doc_id"],
                doc_title=g["doc_title"],
                source=g["source"],
                max_score=g["max_score"],
                chunk_count=len(chunks),
                metadata=g["metadata"],
                chunks=chunks,
            )
        )
    return out
