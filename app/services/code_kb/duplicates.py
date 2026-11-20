"""Find duplicated code inside one indexed scope.

The scan asks Qdrant, for every chunk in scope, "which other chunks are you a
near-duplicate of", then rolls the answers up into file pairs
(core/code_kb/duplicate_rollup explains why the file pair is the reporting unit
and why `min_shared_chunks` matters more than the threshold).

Cost shape, measured on this repository's own index on 2026-09-01: 8,624 chunks
scanned in 28.7s with zero errors. That is fine for an explicitly invoked tool
and far too slow to run after every edit, which is why this is a tool the caller
asks for rather than a background watcher.

Two things keep the cost honest rather than merely bounded:

* excluded paths are dropped BEFORE their chunks are queried, so vendored and
  generated trees cost nothing at all rather than costing a query each;
* `max_chunks` truncation is reported in ``notice`` and ``truncated``. A silently
  truncated scan is indistinguishable from a clean bill of health, which is the
  worst possible failure for a tool whose output is "nothing found".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from qdrant_client.http import models as qmodels

from app.services.code_kb.vector_store import CodeVectorStore
from core.code_kb.duplicate_rollup import (
    DEFAULT_EXCLUDE_GLOBS,
    ChunkMatch,
    ChunkRef,
    DuplicateFilePair,
    is_excluded,
    roll_up_file_pairs,
)

logger = logging.getLogger("code_kb.duplicates")

DEFAULT_THRESHOLD = 0.92
DEFAULT_MIN_SHARED_CHUNKS = 2
DEFAULT_MAX_PAIRS = 50
DEFAULT_MAX_EXAMPLES = 3
DEFAULT_NEIGHBORS = 10
DEFAULT_MAX_CHUNKS = 20_000
_QUERY_BATCH = 64
_SCROLL_PAGE = 1_000
_PAYLOAD_FIELDS = ["path", "start_line", "end_line", "label", "project"]


@dataclass
class DuplicateScanResult:
    pairs: list[DuplicateFilePair] = field(default_factory=list)
    scanned_chunks: int = 0
    scanned_files: int = 0
    truncated: bool = False
    notice: str | None = None


def _collect_chunks(
    vector_store: CodeVectorStore,
    *,
    scope_filter: Any,
    exclude_globs: Sequence[str],
    case_sensitive: bool,
    max_chunks: int,
) -> tuple[list[ChunkRef], bool]:
    """Scroll the scope's chunks, skipping excluded files before they cost a query.

    Excluded chunks do not consume the ``max_chunks`` budget: spending the budget
    on `storage/` segments and then truncating before the real source would make
    the cap actively misleading.

    Chunks are also deduplicated ACROSS scope spellings. ``scope_filter`` may
    cover a Windows scope plus its historical case variants, which is correct for
    search but means one physical file is present once per spelling here. Left
    alone, every cross-file match multiplies by (spellings)^2: measured on `.88`
    on 2026-09-02, this repository was indexed under three spellings and a pair
    the probe measured at 65 shared chunks was reported as 484, with all three
    `examples` printing the same citation. Worse, the one-shared-chunk noise that
    ``min_shared_chunks=2`` exists to suppress counted as nine and cleared the gate.

    The de-dup key is the span, not the point id, and it is only applied when the
    SAME span turns up under a DIFFERENT ``project``. Within one spelling, a
    repeated span is a genuinely distinct chunk — a single-line minified file
    yields many chunks all attributed to that one line (`methviz/index.html:55-55`
    holds 147 of them in production) — and collapsing those would under-report.
    """
    refs: list[ChunkRef] = []
    offset = None
    truncated = False
    # span -> the scope spelling that claimed it first.
    span_owner: dict[tuple[str, int, int, str], str] = {}
    while True:
        rows, offset = vector_store.store.client.scroll(
            vector_store.collection_name, scroll_filter=scope_filter,
            limit=_SCROLL_PAGE, offset=offset,
            with_payload=_PAYLOAD_FIELDS, with_vectors=False)
        for row in rows:
            payload = row.payload or {}
            path = str(payload.get("path") or "")
            if not path or is_excluded(path, exclude_globs, case_sensitive=case_sensitive):
                continue
            start_line = int(payload.get("start_line") or 0)
            end_line = int(payload.get("end_line") or 0)
            label = str(payload.get("label") or "")
            project = str(payload.get("project") or "")
            span = (path, start_line, end_line, label)
            claimed_by = span_owner.get(span)
            if claimed_by is not None and claimed_by != project:
                continue  # the same file under another spelling of this scope
            if claimed_by is None:
                span_owner[span] = project
            if len(refs) >= max_chunks:
                truncated = True
                break
            refs.append(ChunkRef(
                point_id=row.id, path=path,
                start_line=start_line, end_line=end_line, label=label,
            ))
        if truncated or offset is None:
            return refs, truncated


def _neighbour_matches(
    vector_store: CodeVectorStore,
    refs: Sequence[ChunkRef],
    *,
    scope_filter: Any,
    threshold: float,
    neighbors: int,
) -> list[ChunkMatch]:
    by_id = {ref.point_id: ref for ref in refs}
    # `supports_sparse` is what QdrantStore actually exposes, and what it uses
    # itself to decide between a named "dense" vector and a bare one
    # (core/storage/qdrant_store.py:203). An earlier version read
    # `dense_is_named`, which no real store has — it existed only because the
    # test fixture set it, so the tests passed and production answered HTTP 500.
    using = "dense" if vector_store.store.supports_sparse else None
    matches: list[ChunkMatch] = []
    for start in range(0, len(refs), _QUERY_BATCH):
        window = refs[start:start + _QUERY_BATCH]
        requests = [
            qmodels.QueryRequest(
                query=ref.point_id,
                using=using,
                filter=scope_filter,
                # +1 because a point is always its own nearest neighbour at 1.0.
                limit=neighbors + 1,
                score_threshold=threshold,
                with_payload=False,
                with_vector=False,
            )
            for ref in window
        ]
        responses = vector_store.store.client.query_batch_points(
            collection_name=vector_store.collection_name, requests=requests)
        for ref, response in zip(window, responses):
            for point in response.points:
                if point.id == ref.point_id:
                    continue
                target = by_id.get(point.id)
                if target is None:
                    # An excluded or beyond-the-cap chunk. It is a real neighbour
                    # in Qdrant but not part of this report's universe.
                    continue
                matches.append(ChunkMatch(source=ref, target=target, score=point.score))
    return matches


def scan_duplicates(
    vector_store: CodeVectorStore,
    *,
    owner_user_id: str,
    project: str | Sequence[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_shared_chunks: int = DEFAULT_MIN_SHARED_CHUNKS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    exclude_globs: Sequence[str] = DEFAULT_EXCLUDE_GLOBS,
    neighbors_per_chunk: int = DEFAULT_NEIGHBORS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    case_sensitive: bool = True,
) -> DuplicateScanResult:
    """Report file pairs in this scope that share ``min_shared_chunks`` near-duplicates."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")

    filters: dict[str, Any] = {"owner_user_id": owner_user_id}
    if project:
        filters["project"] = project
    scope_filter = vector_store.store._build_filter(filters)

    refs, truncated = _collect_chunks(
        vector_store, scope_filter=scope_filter, exclude_globs=exclude_globs,
        case_sensitive=case_sensitive, max_chunks=max_chunks)
    result = DuplicateScanResult(
        scanned_chunks=len(refs),
        scanned_files=len({ref.path for ref in refs}),
        truncated=truncated,
    )
    if truncated:
        result.notice = (
            f"scan stopped at the max_chunks={max_chunks} cap, so this report covers "
            f"{result.scanned_files} file(s) and is NOT complete for this scope. Narrow "
            "the scope or raise max_chunks before reading an empty result as 'no duplicates'."
        )
    if not refs:
        return result

    matches = _neighbour_matches(
        vector_store, refs, scope_filter=scope_filter,
        threshold=threshold, neighbors=neighbors_per_chunk)
    # Exclusion already happened at collection time; passing () here avoids
    # re-running pathspec over every match for no benefit.
    result.pairs = roll_up_file_pairs(
        matches, min_shared_chunks=min_shared_chunks, max_pairs=max_pairs,
        max_examples=max_examples, exclude_globs=(), case_sensitive=case_sensitive)
    logger.info(
        "[code_kb] duplicate scan owner=%s project=%s chunks=%d files=%d "
        "matches=%d pairs=%d truncated=%s",
        owner_user_id, project, result.scanned_chunks, result.scanned_files,
        len(matches), len(result.pairs), truncated)
    return result
