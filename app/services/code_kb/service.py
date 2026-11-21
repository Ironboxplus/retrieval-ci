"""Server-side code KB service.

Client AST-chunks its repo locally and POSTs chunks here; this service embeds them
with the warm BGE-M3 singleton (no HTTP hop, runs in the backend) and stores them
in the 4090 Qdrant `code_chunks` collection, scoped per (owner_user_id, project);
`host` is display-only payload, never a scoping key (W148). Re-indexing a project
replaces its points (stale-orphan free); an empty enumeration never wipes an
existing index.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
from functools import partial

from app.services.scope_paths import file_abspath as _abspath
from app.services.scope_paths import is_windows_scope
from app.services.scope_paths import path_matches as _path_matches

from app.services.code_kb.duplicates import scan_duplicates
from app.services.code_kb.schemas import (
    CodeDeleteResponse,
    CodeDuplicateChunk,
    CodeDuplicateExample,
    CodeDuplicatePair,
    CodeDuplicateScanRequest,
    CodeDuplicateScanResponse,
    CodeFileIn,
    CodeHit,
    CodeIndexRequest,
    CodeIndexResponse,
    CodeSearchRequest,
    CodeSearchResponse,
    CodeSearchTimingsMs,
)
from app.services.code_kb.vector_store import CodeChunkPoint, CodeVectorStore, point_id
from core.code_kb.duplicate_rollup import DEFAULT_EXCLUDE_GLOBS, ChunkRef

logger = logging.getLogger("code_kb.service")
_OWNER_LOCK_STRIPES = 256

# embed_fn(texts) -> one dense vector per text
EmbedFn = Callable[[Sequence[str]], list[list[float]]]
# sparse_embed_fn(texts) -> one {token_id: weight} dict per text (BGE-M3 lexical)
SparseEmbedFn = Callable[[Sequence[str]], list[dict[str, float]]]
# progress_cb(chunks_done, chunks_total) — called after each embed batch
ProgressCb = Callable[[int, int], None]
MutationCb = Callable[[], None]
ScopeMutationCb = Callable[[str], None]
MutationGuard = Callable[[], AbstractContextManager[None]]

EMBED_BATCH = 128  # chunks per embed_fn call; small enough for steady progress ticks


def _elapsed_ms(start: float) -> float:
    """Wall time since ``start`` (a ``time.perf_counter()`` reading), in milliseconds."""
    return (time.perf_counter() - start) * 1000.0

# max_per_file caps hits per file and rerank rescores them — both discard/reorder AFTER
# retrieval, so pull this many × top_k from the vector store first to still fill top_k
# (capped so a huge top_k can't scan the whole collection). path_glob is NOT in this list:
# it's pushed down as a filter (see search()), so it never needs over-fetch.
OVERFETCH_FACTOR = 6
MAX_FETCH_K = 200


class _MutationFence:
    """Calls `before_mutation_cb` immediately before EVERY write, not just the first.

    That callback is not merely an in-progress marker: it is the job queue's
    delete/cancel fence and it RAISES when the project was deleted or the job
    superseded (`_assert_generation_locked`). Under the old design all writes
    happened in one burst after embedding, so checking once was checking always.

    Streaming spreads the writes across the whole embed, which opens a window the
    old code did not have: a delete landing between batch 3 and batch 4 would have
    its points wiped and then be half-rebuilt by the remaining batches — and the
    job would still mark the scope `completed`. Re-checking per batch closes it, at
    the cost of one idempotent `set_code_scope_visibility` upsert per batch (sub-ms
    against seconds of embedding).

    Hoisting the call above the embed loop instead is what broke
    tests/test_code_kb_jobs.py: it fires the fence before the delete can even be
    requested, and reports the previous snapshot as being overwritten while nothing
    has been written yet.
    """

    __slots__ = ("_cb",)

    def __init__(self, cb: MutationCb | None) -> None:
        self._cb = cb

    def __call__(self) -> None:
        if self._cb is not None:
            self._cb()


def _last_flagged(items: Iterator) -> Iterator[tuple[object, bool]]:
    """Yield each item paired with whether it is the final one.

    Costs one extra item held at a time — for the embed stream that is a second
    batch of vectors (~10 MiB at dim=1024), not a second repository. What it buys
    is that the last batch's writes and the commit that follows can share ONE
    fence: a project delete must not land between "everything is written" and "it
    is committed", which would fail a job whose work actually landed and leave a
    recovery_required marker behind for a scope that no longer exists.
    """
    pending: object = None
    have_pending = False
    for item in items:
        if have_pending:
            yield pending, False
        pending, have_pending = item, True
    if have_pending:
        yield pending, True


def _duplicate_chunk(ref: ChunkRef) -> CodeDuplicateChunk:
    return CodeDuplicateChunk(path=ref.path, start_line=ref.start_line,
                              end_line=ref.end_line, label=ref.label)


@dataclass(frozen=True)
class _PathGlobResolution:
    """A path_glob's push-down match set plus bounded diagnostic counts.

    Candidate paths are read only when the caller supplied ``path_glob``. Keeping the
    counts on that already-required path preserves the normal search hot path while
    making a zero-result filter explainable without guessing that a scope is missing.
    """

    paths: list[str] | None
    indexed_file_count: int | None


def _default_embed(
    texts: Sequence[str], *, priority: str = "interactive",
) -> list[list[float]]:
    from core.indexing.embedders import embed_texts
    return [
        vector.tolist()
        for vector in embed_texts(list(texts), priority=priority).dense
    ]


def _cap_per_file(hits: list[CodeHit], max_per_file: int) -> list[CodeHit]:
    """Keep score order but allow at most max_per_file hits from any one file, so one
    file's many chunks can't monopolise the top-k (W-code #6). Keyed by (project, path)
    so two projects that share a rel_path (e.g. src/main.py) don't share one cap when
    searching across all projects (project=None)."""
    seen: dict[tuple[str, str], int] = {}
    out: list[CodeHit] = []
    for h in hits:
        key = (h.project, h.path)
        n = seen.get(key, 0)
        if n < max_per_file:
            out.append(h)
            seen[key] = n + 1
    return out


_LEXICAL_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _lexical_overlap(value: str, query_terms: set[str]) -> int:
    """Count exact query tokens in a business field without pretending to be BM25.

    Qdrant/RRF remains the retriever.  This only gives tied retrieval scores a stable,
    useful ordering: an exact identifier in a symbol label or path wins over an unrelated
    generated file with the same fused score.
    """
    if not query_terms:
        return 0
    return len({token.casefold() for token in _LEXICAL_TOKEN.findall(value)} & query_terms)


def _finite_retrieval_score(value: float) -> float:
    """Keep malformed upstream scores at the end of a deterministic tie-break."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return score if math.isfinite(score) else float("-inf")


def _stable_retrieval_order(query: str, hits: list[CodeHit]) -> list[CodeHit]:
    """Preserve retrieval score order, resolving only equal scores deterministically.

    Location is the final key so a repeated Qdrant/RRF tie cannot flip between requests.
    Exact tokens in label/path come first because a query such as ``Hysteria-CC-RX
    SendBPS`` is more likely to refer to that symbol than to a generated ``*.pb.go``
    implementation that happens to contain the final token too.
    """
    query_terms = {token.casefold() for token in _LEXICAL_TOKEN.findall(query)}
    return sorted(
        hits,
        key=lambda hit: (
            -_finite_retrieval_score(hit.score),
            -_lexical_overlap(hit.label, query_terms),
            -_lexical_overlap(hit.path, query_terms),
            -_lexical_overlap(hit.code, query_terms),
            hit.project.casefold(),
            hit.path.casefold(),
            hit.start_line,
            hit.end_line,
            hit.label.casefold(),
        ),
    )


def _chunk_points(files: Sequence[CodeFileIn]) -> list[CodeChunkPoint]:
    return [
        CodeChunkPoint(
            rel_path=file.rel_path, language=file.language,
            start_line=chunk.start_line, end_line=chunk.end_line, label=chunk.label,
            content=chunk.content, code=chunk.code, chunk_ordinal=chunk_ordinal,
        )
        for file in files for chunk_ordinal, chunk in enumerate(file.chunks)
    ]


class CodeKBService:
    def __init__(self, *, embed_fn: EmbedFn | None = None,
                 sparse_embed_fn: SparseEmbedFn | None = None,
                 vector_store: CodeVectorStore | None = None,
                 reranker=None):
        self.embed_fn = embed_fn
        self.sparse_embed_fn = sparse_embed_fn
        self.vector_store = vector_store or CodeVectorStore()
        self._reranker = reranker
        self._owner_locks = [threading.RLock() for _ in range(_OWNER_LOCK_STRIPES)]
        # Only the fully-default path can do a single BGE-M3 pass for dense+sparse.
        # If a caller injected either fn (tests), honour exactly what they gave us:
        # dense from embed_fn, sparse from sparse_embed_fn (may be absent → dense-only).
        self._dual = embed_fn is None and sparse_embed_fn is None

    def _owner_lock(self, owner_user_id: str) -> threading.RLock:
        index = hash(owner_user_id) % len(self._owner_locks)
        return self._owner_locks[index]

    @contextmanager
    def scope_snapshot(self, *, owner_user_id: str) -> Iterator[None]:
        """Serialize this owner's vector mutations and inventory reports.

        Fan-out can mutate another project for the same owner, so one owner-level fence
        avoids cross-project lock ordering. The API takes this fence before checking
        durable/runtime visibility and running a search; direct service callers remain
        responsible for applying the same visibility contract.
        """
        with self._owner_lock(owner_user_id):
            yield

    @contextmanager
    def _write_window(self, guard: MutationGuard,
                      owner_user_id: str) -> Iterator[None]:
        """Hold both write fences for ONE mutation step. Neither may span embedding.

        `mutation_guard` is the job queue's per-scope generation lock, which
        `CodeIndexJobQueue.cancel_project` also takes — wrapping the embed stream in
        it makes `DELETE /api/code` block for the entire index. `scope_snapshot` is
        the owner fence the API takes before every search, so spanning that freezes
        this owner's searches for the same duration. Under the old design embedding
        finished before either was acquired, and both facts were free.

        What streaming gives up in exchange: a concurrent delete now linearizes at a
        BATCH boundary instead of around the whole job. The generation check on entry
        then fails the job mid-stream, and the queue's failure path escalates the
        scope's already-durable `preparing` marker to `recovery_required` — a
        partially written scope is reported as partial rather than completed.
        """
        with guard():
            with self.scope_snapshot(owner_user_id=owner_user_id):
                yield

    def _reranker_or_default(self):
        if self._reranker is None:
            from core.retrieval.reranker import get_shared_reranker
            self._reranker = get_shared_reranker()
        return self._reranker

    def _embed(
        self, texts: Sequence[str], *, priority: str = "interactive",
    ) -> tuple[list[list[float]], list[dict[str, float]] | None]:
        """(dense, sparse|None) for a batch. Production does ONE BGE-M3 pass for both;
        an injected embed_fn (tests) yields dense-only unless a sparse_embed_fn is also
        injected. Hybrid then explicitly reports a dense fallback; strict keyword search
        reports unavailable rather than pretending a semantic hit is a literal match."""
        if self._dual:
            from core.indexing.embedders import embed_texts
            res = embed_texts(list(texts), priority=priority)
            return [v.tolist() for v in res.dense], res.sparse
        if self.embed_fn is None:
            dense = _default_embed(texts, priority=priority)
        else:
            dense = self.embed_fn(list(texts))
        sparse = self.sparse_embed_fn(list(texts)) if self.sparse_embed_fn else None
        return dense, sparse

    def _embed_stream(
        self, files: Sequence[CodeFileIn], progress_cb: ProgressCb | None, *,
        owner_user_id: str, project: str,
    ) -> Iterator[tuple[list[CodeFileIn], list[CodeChunkPoint],
                        list[list[float]], list[dict[str, float]] | None]]:
        """Yield (files, items, dense, sparse) one embed batch at a time.

        The caller upserts each batch and drops it, so the vectors in flight are
        bounded by the batch instead of by the repository. That bound is the whole
        point: ``_embed`` hands back ``[v.tolist() for v in res.dense]``, and a
        Python list of Python floats costs ~41KB per 1024-dim chunk against a
        4KB numpy row. Accumulating all of them cost 11.35GiB for the 295,000-chunk
        submission that OOM-killed the backend on 2026-09-04 (MemoryMax=5GiB, ~2.5GiB
        already resident in models). Streaming makes the array library irrelevant,
        because nothing survives the loop iteration.

        Batches never split a FILE. ``_fan_out_sync`` slices each path's vectors out
        of the batch it is handed and then prunes that path's stale point ids, so
        half a file would let the first batch's upsert delete the second batch's
        points in every duplicate scope. A single file with more chunks than
        EMBED_BATCH becomes its own oversized batch — the bound is then
        max(EMBED_BATCH, largest file), still never the repository.

        Progress still ticks per batch: streaming progress is what stopped a
        5k-chunk repo from reading as a hang (W156).
        """
        total = sum(len(f.chunks) for f in files)
        done = 0
        batch: list[CodeFileIn] = []
        batch_chunks = 0
        for file in files:
            batch.append(file)
            batch_chunks += len(file.chunks)
            if batch_chunks < EMBED_BATCH:
                continue
            done += batch_chunks
            yield self._embed_one_batch(batch, owner_user_id=owner_user_id, project=project)
            if progress_cb:
                progress_cb(done, total)
            batch, batch_chunks = [], 0
        if batch:
            done += batch_chunks
            yield self._embed_one_batch(batch, owner_user_id=owner_user_id, project=project)
            if progress_cb:
                progress_cb(done, total)

    def _embed_one_batch(
        self, files: list[CodeFileIn], *, owner_user_id: str, project: str,
    ) -> tuple[list[CodeFileIn], list[CodeChunkPoint], list[list[float]],
               list[dict[str, float] | None] | None]:
        """Embed one batch, keeping the stored vectors of chunks that did not change.

        Both index paths upsert before pruning, so the previous points are still there
        to read. A point is reused only when its id AND every canonical payload field
        (path, language, lines, label, content, code) equal the incoming chunk; it must
        also carry a sparse vector whenever this service would produce one. Re-indexing
        after the 2026-10-13 chunker fix resends every file, yet only the chunks whose
        text actually changed need the embedding model."""
        items = _chunk_points(files)
        reused = self.vector_store.reusable_vectors(
            owner_user_id=owner_user_id, project=project, items=items,
            require_sparse=self.vector_store.supports_sparse
            and (self._dual or self.sparse_embed_fn is not None))
        dense: list = [hit[0] if hit else None for hit in reused]
        sparse: list = [hit[1] if hit else None for hit in reused]
        missing = [index for index, hit in enumerate(reused) if hit is None]
        if missing:
            new_dense, new_sparse = self._embed(
                [items[index].content for index in missing], priority="background")
            aligned = bool(new_sparse) and len(new_sparse) == len(new_dense)
            for position, index in enumerate(missing):
                dense[index] = new_dense[position]
                sparse[index] = new_sparse[position] if aligned else None
        if len(missing) < len(items):
            logger.info("code_kb embed: owner=%s project=%s reused %d/%d chunk vectors",
                        owner_user_id, project, len(items) - len(missing), len(items))
        return files, items, dense, (sparse if any(sparse) else None)

    def _duplicate_candidates(
        self, *, owner_user_id: str, project: str,
    ) -> dict[str, list[tuple[str, str]]] | None:
        """abspath -> [(other_project, other_rel)] for this owner's OTHER scopes.

        Enumerates every file the owner has indexed under every scope — 150+ scopes
        in production — so it must be computed ONCE per index run and reused across
        batches. Returns None if enumeration fails; fan-out is then skipped, because
        the primary index has already succeeded and a partial candidate set would
        silently leave some duplicate scopes stale.
        """
        try:
            owned = self.vector_store.list_owner_files(owner_user_id=owner_user_id)
        except Exception:
            logger.exception("code_kb fan-out: list_owner_files failed — skipping cross-scope sync")
            return None
        by_abs: dict[str, list[tuple[str, str]]] = {}
        for proj2, rel2 in owned:
            if proj2 == project:
                continue
            by_abs.setdefault(_abspath(proj2, rel2), []).append((proj2, rel2))
        return by_abs

    def _fan_out_sync(self, *, owner_user_id: str, project: str, host: str,
                      files, items: list[CodeChunkPoint], dense: list[list[float]],
                      sparse: list[dict[str, float]] | None,
                      deleted_paths: set[str] | None = None,
                      owned_index: dict[str, list[tuple[str, str]]] | None = None,
                      before_target_mutation_cb: ScopeMutationCb | None = None,
                      target_mutation_failed_cb: ScopeMutationCb | None = None) -> int:
        """方案2 — cross-scope write-time sync. For every file just indexed under `project`,
        find the SAME physical file (by absolute path = project/rel_path) indexed under a
        DIFFERENT scope of this owner and refresh that copy with this run's content +
        vectors (embedded once here, reused — never re-embedded). Files removed from the
        source scope are removed from duplicate scopes too. Each target is updated
        upsert-first and stale ids are pruned only after success; every unrelated file is
        untouched, so BOTH scopes stay complete. Returns how many other-scope copies were
        refreshed or removed. Best-effort: if enumerating the owner's files fails, fan-out
        is skipped (the primary index already succeeded)."""
        deleted_paths = deleted_paths or set()
        if not files and not deleted_paths:
            return 0
        by_abs = (self._duplicate_candidates(owner_user_id=owner_user_id, project=project)
                  if owned_index is None else owned_index)
        if by_abs is None:  # enumeration failed; primary index already succeeded
            return 0
        if not by_abs:
            return 0
        desired_by_path: dict[
            str,
            tuple[list[CodeChunkPoint], list[list[float]], list[dict[str, float]] | None],
        ] = {}
        idx = 0
        for f in files:
            n = len(f.chunks)
            file_items = items[idx:idx + n]
            file_dense = dense[idx:idx + n]
            file_sparse = sparse[idx:idx + n] if sparse is not None else None
            idx += n
            desired_by_path[f.rel_path] = (file_items, file_dense, file_sparse)
        for rel_path in deleted_paths:
            desired_by_path.setdefault(rel_path, ([], [], None))

        updated = 0
        for source_rel_path, (file_items, file_dense, file_sparse) in desired_by_path.items():
            source_abs_path = _abspath(project, source_rel_path)
            targets = by_abs.get(source_abs_path)
            if not targets:
                continue
            for proj2, rel2 in targets:
                try:
                    with self.scope_snapshot(owner_user_id=owner_user_id):
                        # Discovery may be stale after a concurrent explicit delete. Re-check
                        # under the target scope lock and never recreate a file whose points
                        # are now gone.
                        before_ids = self.vector_store.list_point_ids(
                            owner_user_id=owner_user_id, project=proj2, path=rel2)
                        if not before_ids:
                            continue
                        # The queue persists the target scope marker while this same
                        # owner fence is held.  A scoped reader cannot slip between this
                        # callback and the first target upsert/delete.
                        if before_target_mutation_cb:
                            before_target_mutation_cb(proj2)
                        items2 = [replace(it, rel_path=rel2) for it in file_items]
                        if items2:
                            # Upsert-first, then prune stale ids. The failure callback
                            # marks this target recovery-required instead of presenting a
                            # partly refreshed cross-scope copy as complete.
                            self.vector_store.upsert(
                                owner_user_id=owner_user_id, project=proj2, host=host,
                                items=items2, dense_vectors=file_dense,
                                sparse_vectors=file_sparse)
                        new_ids = {
                            point_id(
                                owner_user_id, proj2, rel2, it.start_line, it.label,
                                it.chunk_ordinal,
                            )
                            for it in items2
                        }
                        self.vector_store.delete_point_ids(before_ids - new_ids)
                    updated += 1
                    logger.info("code_kb fan-out: synced %s -> project=%s rel=%s",
                                source_abs_path, proj2, rel2)
                except Exception:
                    if target_mutation_failed_cb:
                        target_mutation_failed_cb(proj2)
                    logger.exception("code_kb fan-out: failed syncing %s -> project=%s rel=%s",
                                     source_abs_path, proj2, rel2)
        return updated

    def index(self, request: CodeIndexRequest, *, owner_user_id: str,
              progress_cb: ProgressCb | None = None,
              before_mutation_cb: MutationCb | None = None,
              before_fan_out_target_mutation_cb: ScopeMutationCb | None = None,
              fan_out_target_mutation_failed_cb: ScopeMutationCb | None = None,
              after_mutation_cb: MutationCb | None = None,
              mutation_guard: MutationGuard | None = None) -> CodeIndexResponse:
        """Embed (dense + sparse) + upsert a project's chunks, reporting (done, total)
        after each batch.

        Full (default) wipes the project and rebuilds; incremental (W157-inc) only
        touches changed/removed files. Sparse lexical vectors are stored alongside dense
        so search can run hybrid/keyword (W157-hybrid)."""
        if request.incremental:
            return self._index_incremental(request, owner_user_id=owner_user_id,
                                           progress_cb=progress_cb,
                                           before_mutation_cb=before_mutation_cb,
                                           before_fan_out_target_mutation_cb=(
                                               before_fan_out_target_mutation_cb),
                                           fan_out_target_mutation_failed_cb=(
                                               fan_out_target_mutation_failed_cb),
                                           after_mutation_cb=after_mutation_cb,
                                           mutation_guard=mutation_guard)

        total_chunks = sum(len(file.chunks) for file in request.files)
        if not total_chunks:
            # An empty enumeration must NOT delete an existing project index.
            logger.warning("code_kb index: 0 chunks owner=%s project=%s — skipping (index intact)",
                           owner_user_id, request.project)
            return CodeIndexResponse(project=request.project, host=request.host, files=0, indexed=0)

        guard = mutation_guard or nullcontext
        n = 0
        dup = 0
        new_ids: set = set()
        fence = _MutationFence(before_mutation_cb)
        window = partial(self._write_window, guard, owner_user_id)
        with window():
            existing_paths = self.vector_store.list_paths(
                owner_user_id=owner_user_id, project=request.project)
            # Captured before any upsert of ours, or the prune below would see
            # this run's own points as pre-existing.
            before_ids = self.vector_store.list_point_ids_for_paths(
                owner_user_id=owner_user_id, project=request.project)
        # Once per run, never per batch: this enumerates every file in every scope
        # this owner has. Outside the fence — it is a read, and it is the expensive
        # part of fan-out.
        owned_index = self._duplicate_candidates(
            owner_user_id=owner_user_id, project=request.project)

        def commit() -> int:
            """Prune, reconcile removed files, publish. Called inside the FINAL
            batch's fence so the job's last write and its commit are indivisible."""
            # Deletes stay AFTER all embedding: a mid-embed failure can leave the
            # project partly refreshed (the visibility marker covers that), but
            # never deleted-and-not-rebuilt.
            fence()  # a delete is a mutation too
            self.vector_store.delete_point_ids(before_ids - new_ids)
            extra = 0
            gone = existing_paths - {file.rel_path for file in request.files}
            if gone:
                # Removed-file fan-out runs once, not per batch: it is keyed on
                # paths absent from the whole request, which no single batch knows.
                extra = self._fan_out_sync(
                    owner_user_id=owner_user_id, project=request.project,
                    host=request.host, files=[], items=[], dense=[], sparse=None,
                    deleted_paths=gone, owned_index=owned_index,
                    before_target_mutation_cb=before_fan_out_target_mutation_cb,
                    target_mutation_failed_cb=fan_out_target_mutation_failed_cb)
            if after_mutation_cb:
                after_mutation_cb()
            return extra

        # Each iteration embeds a batch with NO fence held, then takes both fences
        # only for that batch's writes. A delete can therefore linearize between
        # batches — i.e. while embedding, where failing the job is the right answer.
        committed = False
        for batch, is_last in _last_flagged(
                self._embed_stream(request.files, progress_cb,
                                   owner_user_id=owner_user_id,
                                   project=request.project)):
            batch_files, items, dense, sparse_arg = batch
            with window():
                fence()
                n += self.vector_store.upsert(
                    owner_user_id=owner_user_id, project=request.project,
                    host=request.host, items=items, dense_vectors=dense,
                    sparse_vectors=sparse_arg)
                new_ids.update(
                    point_id(owner_user_id, request.project, item.rel_path,
                             item.start_line, item.label, item.chunk_ordinal)
                    for item in items
                )
                dup += self._fan_out_sync(
                    owner_user_id=owner_user_id, project=request.project,
                    host=request.host, files=batch_files, items=items,
                    dense=dense, sparse=sparse_arg, owned_index=owned_index,
                    before_target_mutation_cb=before_fan_out_target_mutation_cb,
                    target_mutation_failed_cb=fan_out_target_mutation_failed_cb)
                if is_last:
                    dup += commit()
                    committed = True
        if not committed:  # no batches at all — nothing to share a fence with
            with window():
                dup += commit()
        logger.info("code_kb index: owner=%s project=%s files=%d chunks=%d dup_synced=%d",
                    owner_user_id, request.project, len(request.files), n, dup)
        return CodeIndexResponse(project=request.project, host=request.host,
                                 files=len(request.files), indexed=n, duplicates_updated=dup)

    def _index_incremental(self, request: CodeIndexRequest, *, owner_user_id: str,
                           progress_cb: ProgressCb | None,
                           before_mutation_cb: MutationCb | None,
                           before_fan_out_target_mutation_cb: ScopeMutationCb | None,
                           fan_out_target_mutation_failed_cb: ScopeMutationCb | None,
                           after_mutation_cb: MutationCb | None,
                           mutation_guard: MutationGuard | None) -> CodeIndexResponse:
        """Update only what changed. Correctness invariant (the one the user flagged):
        the result MUST equal a full reindex of the current tree. Guaranteed by —
          1. `manifest` = every path that currently exists; any indexed path absent from
             it is deleted (removed-file reconciliation). manifest=None → additive only.
          2. each changed file is upserted first, then stale point ids are pruned, so a
             failed upsert preserves the previous complete file while a successful shrink
             leaves no orphan chunks.
          3. embed happens BEFORE any delete, so a mid-embed failure changes nothing.
        """
        project = request.project
        # Files the client sent (changed/new) — purge their old points even if a file
        # became empty (produced no chunks), so emptied files don't linger.
        changed_paths = {f.rel_path for f in request.files}
        removed = set(request.removed_paths) - changed_paths
        if request.manifest is not None:
            keep = set(request.manifest)
            existing = self.vector_store.list_paths(owner_user_id=owner_user_id, project=project)
            # Delete-safety valve (like rsync --max-delete): an empty manifest, or one that
            # shares NO path with the existing index, is almost certainly a broken walk or a
            # wrong project scope — not a legitimate full-rename. Refuse rather than nuke the
            # whole project from a bad client (W157-inc). To intentionally clear a project use
            # DELETE /api/code?project=<abspath> (W165); to replace it, run a full
            # (non-incremental) reindex of the new tree (a full reindex of an EMPTY tree is a
            # no-op by design, so it won't clear a project either — deletion must be explicit).
            if existing and keep.isdisjoint(existing):
                raise ValueError(
                    f"incremental manifest ({len(keep)} paths) is empty or disjoint from the "
                    f"existing index ({len(existing)} paths) for project={project} — refusing to "
                    f"delete the whole project. Use DELETE /api/code?project={project} to clear "
                    f"it, or a full reindex of the new tree to replace it.")
            removed.update(existing - keep)
        if not changed_paths and not removed:
            logger.info("code_kb incremental: owner=%s project=%s — nothing changed", owner_user_id, project)
            return CodeIndexResponse(project=project, host=request.host, files=0, indexed=0)

        guard = mutation_guard or nullcontext
        n = 0
        dup = 0
        new_ids: set = set()
        fence = _MutationFence(before_mutation_cb)
        window = partial(self._write_window, guard, owner_user_id)
        affected_paths = changed_paths | removed
        with window():
            # Captured before any upsert of ours, or the prune would treat this
            # run's own points as pre-existing.
            before_ids = self.vector_store.list_point_ids_for_paths(
                owner_user_id=owner_user_id,
                project=project,
                paths=affected_paths,
            )
        # Once per run, never per batch — see index().
        owned_index = self._duplicate_candidates(
            owner_user_id=owner_user_id, project=project)

        def commit() -> int:
            """Runs inside the FINAL batch's fence — or in its own when a
            removal-only request produced no batches to share one with."""
            # Deletes stay after all embedding (invariant 3's surviving half).
            fence()
            self.vector_store.delete_point_ids(before_ids - new_ids)
            extra = 0
            if removed:
                # Keyed on paths absent from the whole request, so it runs once.
                extra = self._fan_out_sync(
                    owner_user_id=owner_user_id, project=project, host=request.host,
                    files=[], items=[], dense=[], sparse=None,
                    deleted_paths=removed, owned_index=owned_index,
                    before_target_mutation_cb=before_fan_out_target_mutation_cb,
                    target_mutation_failed_cb=fan_out_target_mutation_failed_cb)
            if after_mutation_cb:
                after_mutation_cb()
            return extra

        # Embedding stays outside both fences — see _write_window for why.
        committed = False
        for batch, is_last in _last_flagged(
                self._embed_stream(request.files, progress_cb,
                                   owner_user_id=owner_user_id,
                                   project=request.project)):
            batch_files, items, dense, sparse_arg = batch
            with window():
                fence()
                n += self.vector_store.upsert(
                    owner_user_id=owner_user_id, project=project, host=request.host,
                    items=items, dense_vectors=dense, sparse_vectors=sparse_arg)
                new_ids.update(
                    point_id(owner_user_id, project, item.rel_path,
                             item.start_line, item.label, item.chunk_ordinal)
                    for item in items
                )
                dup += self._fan_out_sync(
                    owner_user_id=owner_user_id, project=project, host=request.host,
                    files=batch_files, items=items, dense=dense, sparse=sparse_arg,
                    owned_index=owned_index,
                    before_target_mutation_cb=before_fan_out_target_mutation_cb,
                    target_mutation_failed_cb=fan_out_target_mutation_failed_cb)
                if is_last:
                    dup += commit()
                    committed = True
        if not committed:
            with window():
                dup += commit()
        logger.info("code_kb incremental: owner=%s project=%s changed_files=%d chunks=%d "
                    "removed=%d dup_synced=%d",
                    owner_user_id, project, len(request.files), n, len(removed), dup)
        return CodeIndexResponse(project=project, host=request.host,
                                 files=len(request.files), indexed=n, duplicates_updated=dup)

    @staticmethod
    def _project_filter(project: str | None,
                        project_aliases: Sequence[str] | None) -> str | list[str] | None:
        """Build one Qdrant project filter including legacy Windows case variants."""
        if not project:
            return None
        candidates = [project, *(project_aliases or ())]
        unique = list(dict.fromkeys(candidate for candidate in candidates if candidate))
        return unique[0] if len(unique) == 1 else unique

    def _resolve_path_glob_paths(
        self,
        request: CodeSearchRequest,
        *,
        owner_user_id: str,
        project_filter: str | list[str] | None,
    ) -> _PathGlobResolution:
        """Resolve a path_glob to an exact push-down path set plus diagnostic counts.

        ``paths=None`` means no filter and preserves the normal no-enumeration search
        path. An empty list means the supplied relative filter selected no indexed file;
        callers can explain that precisely instead of misdiagnosing a missing scope.
        """
        if not request.path_glob:
            return _PathGlobResolution(paths=None, indexed_file_count=None)
        if request.project:
            candidate = self.vector_store.list_paths(
                owner_user_id=owner_user_id, project=project_filter)
        else:
            candidate = self.vector_store.list_owner_paths(owner_user_id=owner_user_id)
        case_sensitive = not is_windows_scope(request.project or "")
        return _PathGlobResolution(
            paths=[p for p in candidate if _path_matches(
                p, request.path_glob, case_sensitive=case_sensitive)],
            indexed_file_count=len(candidate),
        )

    def search(self, request: CodeSearchRequest, *, owner_user_id: str,
               project_aliases: Sequence[str] | None = None,
               excluded_projects: Sequence[str] | None = None) -> CodeSearchResponse:
        """Search code with strict keyword semantics and bounded scope diagnostics.

        ``project_aliases`` comes from the compact catalog scope list only for legacy
        Windows casing variants. Ordinary searches still issue one scoped vector request
        and do not enumerate files unless the caller supplied ``path_glob``.

        ``excluded_projects`` (federated-search decision 10) pushes a Qdrant
        ``must_not`` on ``project`` instead of narrowing the positive filter, so a
        scope that predates the catalog marker (and so has no durable row to put on an
        allow-list) is still correctly excluded.
        """
        total_start = time.perf_counter()
        if not request.query.strip():
            return CodeSearchResponse(
                hits=[], total=0, requested_mode=request.mode, effective_mode=request.mode,
                timings_ms=CodeSearchTimingsMs(total=_elapsed_ms(total_start)))

        project_filter = self._project_filter(request.project, project_aliases)
        legacy_aliases = [
            alias for alias in dict.fromkeys(project_aliases or ())
            if alias and alias != request.project
        ]
        glob = self._resolve_path_glob_paths(
            request, owner_user_id=owner_user_id, project_filter=project_filter)
        if glob.paths is not None and not glob.paths:
            return CodeSearchResponse(
                hits=[], total=0, requested_mode=request.mode, effective_mode=request.mode,
                notice=(
                    f"path_glob {request.path_glob!r} matched 0 of "
                    f"{glob.indexed_file_count or 0} indexed file(s) in scope "
                    f"{request.project!r}. path_glob is scope-relative; use "
                    "'conversion/**', not an absolute repository path."
                ),
                path_glob_matched_files=0,
                indexed_file_count=glob.indexed_file_count,
                scope_aliases=legacy_aliases,
                timings_ms=CodeSearchTimingsMs(total=_elapsed_ms(total_start)),
            )

        embed_start = time.perf_counter()
        dense_q, sparse_list = self._embed([request.query], priority="interactive")
        embed_ms = _elapsed_ms(embed_start)
        qv = dense_q[0]
        sparse_q = sparse_list[0] if sparse_list else None
        sparse_available = bool(sparse_q and self.vector_store.supports_sparse)
        # A semantic fallback helps a behavior query but is not an honest result for an
        # exact symbol query. Keyword therefore fails explicitly rather than emitting an
        # unrelated low-score dense neighbour under a lexical-looking result heading.
        if request.mode == "keyword" and not sparse_available:
            return CodeSearchResponse(
                hits=[], total=0, requested_mode=request.mode, effective_mode="unavailable",
                notice=(
                    "keyword search is unavailable because sparse lexical retrieval is "
                    "unavailable for this query or index; it does not substitute dense "
                    "results. Re-index this scope to write sparse vectors, then retry "
                    "mode='keyword'."
                ),
                path_glob_matched_files=(len(glob.paths) if glob.paths is not None else None),
                indexed_file_count=glob.indexed_file_count,
                scope_aliases=legacy_aliases,
                timings_ms=CodeSearchTimingsMs(
                    embed=embed_ms, total=_elapsed_ms(total_start)),
            )

        mode = request.mode if sparse_available else "dense"
        notice = None
        if request.mode == "hybrid" and mode == "dense":
            notice = (
                "hybrid search used dense-only fallback because sparse lexical retrieval "
                "is unavailable for this query or index. Re-index this scope to restore "
                "hybrid lexical fusion."
            )
        # path_glob is pushed down as a MatchAny path set. Only max_per_file/rerank still
        # discard/reorder after retrieval, so only they need over-fetch.
        if request.max_per_file or request.rerank:
            fetch_k = min(request.top_k * OVERFETCH_FACTOR, MAX_FETCH_K)
        else:
            fetch_k = request.top_k
        retrieve_start = time.perf_counter()
        raw = self.vector_store.search(
            dense_vector=qv, top_k=fetch_k, owner_user_id=owner_user_id,
            project=project_filter, language=request.language,
            sparse_vector=sparse_q, mode=mode, paths=glob.paths,
            excluded_projects=excluded_projects)
        retrieve_ms = _elapsed_ms(retrieve_start)
        hits = [
            CodeHit(
                path=r.get("path", ""), start_line=r.get("start_line", 0),
                end_line=r.get("end_line", 0), label=r.get("label", ""),
                language=r.get("language"), project=r.get("project", ""),
                host=r.get("host", ""), score=r.get("score", 0.0), code=r.get("code", ""),
            )
            for r in raw
        ]
        hits = _stable_retrieval_order(request.query, hits)
        rerank_ms = None
        if request.rerank and hits:
            rerank_start = time.perf_counter()
            hits = self._rerank(request.query, hits)
            rerank_ms = _elapsed_ms(rerank_start)
        if request.max_per_file:
            hits = _cap_per_file(hits, request.max_per_file)
        hits = hits[:request.top_k]
        return CodeSearchResponse(
            hits=hits, total=len(hits), requested_mode=request.mode,
            effective_mode=mode, notice=notice,
            path_glob_matched_files=(len(glob.paths) if glob.paths is not None else None),
            indexed_file_count=glob.indexed_file_count,
            scope_aliases=legacy_aliases,
            timings_ms=CodeSearchTimingsMs(
                embed=embed_ms, retrieve=retrieve_ms, rerank=rerank_ms,
                total=_elapsed_ms(total_start)),
        )

    def _rerank(self, query: str, hits: list[CodeHit]) -> list[CodeHit]:
        """Cross-encoder rescore with all-or-nothing validation and stable fail-open.

        Reranker output is an external model boundary.  Do not mutate a single hit until
        every score has been materialised, length-checked, converted to a finite float,
        and accepted; otherwise returning a half-reranked list would conceal a model
        failure as retrieval truth.
        """
        try:
            raw_scores = self._reranker_or_default().score_pairs(query, [h.code for h in hits])
            if raw_scores is None or isinstance(raw_scores, (str, bytes, bytearray)):
                raise ValueError("reranker scores are not a score sequence")
            scores = list(raw_scores)
            if len(scores) != len(hits):
                raise ValueError("reranker score count does not match candidates")
            finite_scores: list[float] = []
            for score in scores:
                value = float(score)
                if not math.isfinite(value):
                    raise ValueError("reranker score is not finite")
                finite_scores.append(value)
        except Exception as exc:  # noqa: BLE001 - fail open is the search contract
            # Do not log the query or exception text: either can contain user source text.
            logger.warning(
                "code_kb rerank unavailable; preserving retrieval order candidate_count=%d "
                "error_type=%s",
                len(hits),
                type(exc).__name__,
            )
            return hits
        # ``hits`` already has the deterministic retrieval order.  Keep that order only
        # after the reranker score AND original retrieval score tie; do not let a location
        # sort erase a stronger retrieval candidate just because the model rounded equal.
        reranked = list(zip(hits, finite_scores, range(len(hits))))
        reranked.sort(
            key=lambda row: (-row[1], -_finite_retrieval_score(row[0].score), row[2]))
        for hit, score, _ in reranked:
            hit.score = score
        return [hit for hit, _, _ in reranked]

    def find_duplicates(
        self,
        request: CodeDuplicateScanRequest,
        *,
        owner_user_id: str,
        project_aliases: Sequence[str] | None = None,
    ) -> CodeDuplicateScanResponse:
        """Report file pairs in one scope that share near-duplicate chunks.

        Read-only, but it takes the same owner fence as search: an in-place scope
        rewrite would otherwise let the scan see a file's old and new chunks at
        once and report the file as a duplicate of itself under two paths.

        Windows scopes match paths case-insensitively for exclusions, matching the
        path_glob semantics in :meth:`search`.
        """
        project_filter = self._project_filter(request.project, project_aliases)
        globs = (DEFAULT_EXCLUDE_GLOBS if request.exclude_globs is None
                 else tuple(request.exclude_globs))
        with self.scope_snapshot(owner_user_id=owner_user_id):
            result = scan_duplicates(
                self.vector_store,
                owner_user_id=owner_user_id,
                project=project_filter,
                threshold=request.threshold,
                min_shared_chunks=request.min_shared_chunks,
                max_pairs=request.max_pairs,
                max_examples=request.max_examples,
                exclude_globs=globs,
                max_chunks=request.max_chunks,
                case_sensitive=not is_windows_scope(request.project),
            )
        aliases = [
            alias for alias in dict.fromkeys(project_aliases or ())
            if alias and alias != request.project
        ]
        return CodeDuplicateScanResponse(
            project=request.project,
            pairs=[
                CodeDuplicatePair(
                    path_a=pair.path_a, path_b=pair.path_b,
                    shared_chunks=pair.shared_chunks,
                    max_score=pair.max_score, mean_score=pair.mean_score,
                    examples=[
                        CodeDuplicateExample(
                            a=_duplicate_chunk(example.a),
                            b=_duplicate_chunk(example.b),
                            score=example.score,
                        )
                        for example in pair.examples
                    ],
                )
                for pair in result.pairs
            ],
            total=len(result.pairs),
            scanned_chunks=result.scanned_chunks,
            scanned_files=result.scanned_files,
            truncated=result.truncated,
            notice=result.notice,
            scope_aliases=aliases,
        )

    def delete_project(
        self,
        project: str,
        *,
        owner_user_id: str,
        project_aliases: Sequence[str] | None = None,
    ) -> CodeDeleteResponse:
        """Delete one canonical scope plus any proven legacy Windows spellings.

        The route obtains aliases from the compact inventory project list.  Deleting
        them together is required for case-insensitive Windows semantics: otherwise a
        mixed-case pre-normalisation Qdrant scope could remain searchable after an
        explicit delete of the same NTFS directory.
        """
        projects = list(dict.fromkeys(
            candidate for candidate in [project, *(project_aliases or ())] if candidate))
        with self.scope_snapshot(owner_user_id=owner_user_id):
            n = sum(
                self.vector_store.delete_project(owner_user_id=owner_user_id, project=scope)
                for scope in projects
            )
        aliases = [scope for scope in projects if scope != project]
        logger.info(
            "code_kb delete: owner=%s project=%s aliases=%d removed=%d",
            owner_user_id, project, len(aliases), n,
        )
        return CodeDeleteResponse(project=project, deleted=n, scope_aliases=aliases)


_SERVICE: CodeKBService | None = None
_SERVICE_INIT_LOCK = threading.Lock()


def get_code_kb_service() -> CodeKBService:
    global _SERVICE
    if _SERVICE is None:
        # Route threads can reach this lazy factory at the same time.  The warm service
        # owns both owner fences and shared model/vector dependencies, so two first-use
        # constructions would split the visibility contract as well as duplicate memory.
        with _SERVICE_INIT_LOCK:
            if _SERVICE is None:
                _SERVICE = CodeKBService()
    return _SERVICE
