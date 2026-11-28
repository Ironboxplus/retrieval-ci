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

`scan_changed_files` is the incremental counterpart used by the background analysis
(app/services/code_kb/analysis.py): it queries the neighbours of the chunks of a
given set of files only, so its cost follows the change instead of the scope. Both
entry points take a ``ScanGuard`` so a long scan holds the owner fence per query batch
rather than for its whole length, and discards its result if the scope was written to
in between.
"""
from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

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
# Paths per MatchAny condition of the incremental scan: a change can touch thousands of
# files and one filter naming all of them would be an unreasonable request.
_PATH_GROUP = 200
_PAYLOAD_FIELDS = ["path", "start_line", "end_line", "label", "project"]


class ScopeChangedDuringScan(Exception):
    """The scanned scope was written to (or stopped being a completed snapshot) while the
    scan was between query batches, so what it read may mix old and new chunks."""


class ScanGuard(Protocol):
    """Serialises ONE query batch against the scope's writers.

    A scan used to hold the owner fence for its whole length, which blocked every search
    and index write of that owner for minutes on a large scope. It now takes the fence per
    batch (``batch``) and asks once at the end whether anything was written in between
    (``finish``), raising ``ScopeChangedDuringScan`` if so."""

    def batch(self) -> AbstractContextManager[None]: ...

    def finish(self) -> None: ...


class _Unguarded:
    """For callers that already hold the fence or are not sharing the scope with writers."""

    def batch(self) -> AbstractContextManager[None]:
        return nullcontext()

    def finish(self) -> None:
        return None


UNGUARDED: ScanGuard = _Unguarded()


@dataclass
class DuplicateScanResult:
    pairs: list[DuplicateFilePair] = field(default_factory=list)
    scanned_chunks: int = 0
    scanned_files: int = 0
    truncated: bool = False
    notice: str | None = None


@dataclass
class IncrementalScanResult:
    """File pairs that involve at least one of the scanned (changed) files.

    ``too_large`` means the changed files hold more chunks than the budget; nothing was
    queried, so ``pairs`` says nothing about those files."""
    pairs: list[DuplicateFilePair] = field(default_factory=list)
    scanned_chunks: int = 0
    scanned_files: int = 0
    too_large: bool = False


class _SpanClaims:
    """Which scope spelling first claimed a (path, lines, label) span.

    See `_collect_chunks` for why the same span under another spelling is one physical
    chunk and must be counted once."""

    def __init__(self) -> None:
        self._claimed_by: dict[tuple[str, int, int, str], str] = {}

    def admit(self, span: tuple[str, int, int, str], project: str) -> bool:
        return self._claimed_by.setdefault(span, project) == project


class _ChunkCollector:
    """Accumulates the chunks to scan from one or more filtered scrolls.

    One budget (``max_chunks``) and one set of span claims are shared by every scroll,
    so the incremental scan can read its changed paths in filter-sized groups without
    spending the budget or counting a spelling twice per group."""

    def __init__(
        self,
        vector_store: CodeVectorStore,
        *,
        exclude_globs: Sequence[str],
        case_sensitive: bool,
        max_chunks: int,
        guard: ScanGuard,
    ) -> None:
        self._vector_store = vector_store
        self._exclude_globs = exclude_globs
        self._case_sensitive = case_sensitive
        self._max_chunks = max_chunks
        self._guard = guard
        self._claims = _SpanClaims()
        self.refs: list[ChunkRef] = []
        self.truncated = False

    @property
    def claims(self) -> _SpanClaims:
        return self._claims

    def scroll(self, scope_filter: Any) -> None:
        offset = None
        while not self.truncated:
            with self._guard.batch():
                rows, offset = self._vector_store.store.client.scroll(
                    self._vector_store.collection_name, scroll_filter=scope_filter,
                    limit=_SCROLL_PAGE, offset=offset,
                    with_payload=_PAYLOAD_FIELDS, with_vectors=False)
            for row in rows:
                payload = row.payload or {}
                path = str(payload.get("path") or "")
                if not path or is_excluded(
                        path, self._exclude_globs, case_sensitive=self._case_sensitive):
                    continue
                start_line = int(payload.get("start_line") or 0)
                end_line = int(payload.get("end_line") or 0)
                label = str(payload.get("label") or "")
                project = str(payload.get("project") or "")
                if not self._claims.admit((path, start_line, end_line, label), project):
                    continue  # the same file under another spelling of this scope
                if len(self.refs) >= self._max_chunks:
                    self.truncated = True
                    break
                self.refs.append(ChunkRef(
                    point_id=row.id, path=path,
                    start_line=start_line, end_line=end_line, label=label,
                ))
            if offset is None:
                return


def _collect_chunks(
    vector_store: CodeVectorStore,
    *,
    scope_filter: Any,
    exclude_globs: Sequence[str],
    case_sensitive: bool,
    max_chunks: int,
    guard: ScanGuard = UNGUARDED,
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
    collector = _ChunkCollector(
        vector_store, exclude_globs=exclude_globs, case_sensitive=case_sensitive,
        max_chunks=max_chunks, guard=guard)
    collector.scroll(scope_filter)
    return collector.refs, collector.truncated


def _neighbour_matches(
    vector_store: CodeVectorStore,
    refs: Sequence[ChunkRef],
    *,
    scope_filter: Any,
    threshold: float,
    neighbors: int,
    guard: ScanGuard = UNGUARDED,
    resolve_target: Callable[[Any, ChunkRef], ChunkRef | None] | None = None,
) -> list[ChunkMatch]:
    """Nearest neighbours of ``refs`` inside ``scope_filter``, as chunk matches.

    A neighbour outside ``refs`` is skipped (it is an excluded or beyond-the-cap chunk)
    unless ``resolve_target`` is given: the incremental scan queries only some files'
    chunks, so its neighbours are by design outside ``refs`` and are described by the
    payload each hit carries."""
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
                with_payload=_PAYLOAD_FIELDS if resolve_target is not None else False,
                with_vector=False,
            )
            for ref in window
        ]
        with guard.batch():
            responses = vector_store.store.client.query_batch_points(
                collection_name=vector_store.collection_name, requests=requests)
        for ref, response in zip(window, responses):
            for point in response.points:
                if point.id == ref.point_id:
                    continue
                target = by_id.get(point.id)
                if target is None and resolve_target is not None:
                    target = resolve_target(point, ref)
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
    guard: ScanGuard = UNGUARDED,
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
        case_sensitive=case_sensitive, max_chunks=max_chunks, guard=guard)
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
        guard.finish()
        return result

    matches = _neighbour_matches(
        vector_store, refs, scope_filter=scope_filter,
        threshold=threshold, neighbors=neighbors_per_chunk, guard=guard)
    # Exclusion already happened at collection time; passing () here avoids
    # re-running pathspec over every match for no benefit.
    result.pairs = roll_up_file_pairs(
        matches, min_shared_chunks=min_shared_chunks, max_pairs=max_pairs,
        max_examples=max_examples, exclude_globs=(), case_sensitive=case_sensitive)
    guard.finish()
    logger.info(
        "[code_kb] duplicate scan owner=%s project=%s chunks=%d files=%d "
        "matches=%d pairs=%d truncated=%s",
        owner_user_id, project, result.scanned_chunks, result.scanned_files,
        len(matches), len(result.pairs), truncated)
    return result


def scan_changed_files(
    vector_store: CodeVectorStore,
    *,
    owner_user_id: str,
    project: str | Sequence[str],
    changed_paths: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    min_shared_chunks: int = DEFAULT_MIN_SHARED_CHUNKS,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    exclude_globs: Sequence[str] = DEFAULT_EXCLUDE_GLOBS,
    neighbors_per_chunk: int = DEFAULT_NEIGHBORS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    case_sensitive: bool = True,
    guard: ScanGuard = UNGUARDED,
) -> IncrementalScanResult:
    """File pairs, across the whole scope, that involve at least one of ``changed_paths``.

    Only the changed files' chunks are queried (path MatchAny inside the scope filter);
    each neighbour comes back with its payload because it is, in general, a chunk of a
    file that was not scrolled. Exclusions, cross-spelling de-duplication and the roll-up
    are the same ones the full scan uses.

    Known difference from a full scan: neighbours are only looked up from the changed
    side, so an unchanged file whose chunk ranks a changed chunk outside its own top
    ``neighbors_per_chunk`` is not seen from there. A full scan can therefore find a few
    more shared chunks for a pair. Callers must not present the result as complete.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")

    projects = [project] if isinstance(project, str) else list(dict.fromkeys(project))
    if not projects:
        raise ValueError("project must name at least one scope spelling")
    scope: dict[str, Any] = {
        "owner_user_id": owner_user_id,
        "project": projects[0] if len(projects) == 1 else projects,
    }
    store = vector_store.store
    paths = list(dict.fromkeys(path for path in changed_paths if path))

    collector = _ChunkCollector(
        vector_store, exclude_globs=exclude_globs, case_sensitive=case_sensitive,
        max_chunks=max_chunks, guard=guard)
    for start in range(0, len(paths), _PATH_GROUP):
        collector.scroll(store._build_filter({**scope, "path": paths[start:start + _PATH_GROUP]}))
        if collector.truncated:
            break
    refs = collector.refs
    result = IncrementalScanResult(
        scanned_chunks=len(refs),
        scanned_files=len({ref.path for ref in refs}),
        too_large=collector.truncated,
    )
    if collector.truncated or not refs:
        guard.finish()
        return result

    excluded_paths: dict[str, bool] = {}

    def resolve_target(point: Any, source: ChunkRef) -> ChunkRef | None:
        payload = point.payload or {}
        path = str(payload.get("path") or "")
        if not path or path == source.path:
            return None  # adjacent chunks of one file are not duplicated design
        if path not in excluded_paths:
            excluded_paths[path] = is_excluded(
                path, exclude_globs, case_sensitive=case_sensitive)
        if excluded_paths[path]:
            return None
        start_line = int(payload.get("start_line") or 0)
        end_line = int(payload.get("end_line") or 0)
        label = str(payload.get("label") or "")
        if not collector.claims.admit(
                (path, start_line, end_line, label), str(payload.get("project") or "")):
            return None  # the same file under another spelling of this scope
        return ChunkRef(point_id=point.id, path=path, start_line=start_line,
                        end_line=end_line, label=label)

    matches = _neighbour_matches(
        vector_store, refs, scope_filter=store._build_filter(scope),
        threshold=threshold,
        # Every spelling holds its own copy of a neighbour and Qdrant returns all of
        # them, so one slot per spelling keeps the real neighbour count unchanged.
        neighbors=neighbors_per_chunk * len(projects),
        guard=guard, resolve_target=resolve_target)
    result.pairs = roll_up_file_pairs(
        matches, min_shared_chunks=min_shared_chunks, max_pairs=None,
        max_examples=max_examples, exclude_globs=(), case_sensitive=case_sensitive)
    guard.finish()
    logger.info(
        "[code_kb] incremental duplicate scan owner=%s project=%s changed=%d chunks=%d "
        "matches=%d pairs=%d", owner_user_id, project, len(paths), result.scanned_chunks,
        len(matches), len(result.pairs))
    return result
