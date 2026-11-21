"""Application service for ACE journal CRUD and search."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, NamedTuple, Sequence

from app.models.schemas import SearchResult
from app.services.scope_paths import (
    canonicalize_scope_path,
    file_abspath as _abspath,
    is_windows_scope,
    path_matches as _path_matches,
    stored_relative_path_identity,
)
from app.services.journal.chunker import JournalChunker, JournalTextChunk
from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import (
    JournalBulkDocItem,
    JournalChunkResponse,
    JournalDeleteResponse,
    JournalDocReconcileResponse,
    JournalDocScopeDeleteResponse,
    JournalEntryListResponse,
    JournalEntryResponse,
    JournalRecordRequest,
    JournalRecordResponse,
    JournalSearchHit,
    JournalSearchRequest,
    JournalSearchResponse,
)
from app.services.journal.vector_store import JournalChunkUpsert, JournalVectorStore
from core.db.models import JournalChunk, JournalEntry


logger = logging.getLogger("journal.service")

EmbedFn = Callable[[Sequence[str], bool], tuple[list[list[float]], list[dict[str, float]] | None]]
JOURNAL_EMBED_BATCH = 128


class _GlobResolution(NamedTuple):
    """Result of resolving a search's path_glob to matching entry_ids, carrying the scope's
    doc count + a sample title so an EMPTY match can be explained (see _resolve_path_glob_ids)."""
    ids: list[str] | None       # None ⇒ no path_glob → caller adds no filter
    scope_doc_count: int        # titled live docs in scope (0 when no path_glob)
    sample_title: str | None    # a representative title, for the empty-match notice


def _looks_binary(text: str) -> bool:
    """True when `text` is binary content, not prose — the server-side backstop for the
    "no images/binaries in the journal" invariant (W161). The client MCP tools filter by
    extension before upload, but that guard lives only in their two copies; a raw POST, a
    forked plugin, or an unlabeled binary can still reach `record()`. A NUL byte is the
    signal: real binaries (PNG/pickle/parquet/npy — and any UTF-16 text) carry them,
    legitimate prose never does. We deliberately do NOT reject on a high U+FFFD ratio —
    a GBK/Latin-1 doc read as UTF-8 is dense with replacement chars yet is legitimate,
    mis-encoded prose, and punishing it would 400 real user docs (W162 review CR1)."""
    return "\x00" in text


def _default_embed(
    texts: Sequence[str], include_sparse: bool, *, priority: str = "interactive",
) -> tuple[list[list[float]], list[dict[str, float]] | None]:
    from core.indexing.embedders import embed_texts
    result = embed_texts(list(texts), priority=priority)
    dense = [v.tolist() for v in result.dense]
    sparse = result.sparse if include_sparse else None
    return dense, sparse


class JournalService:
    def __init__(
        self,
        *,
        repository: JournalRepository | None = None,
        chunker: JournalChunker | None = None,
        vector_store: JournalVectorStore | None = None,
        embed_fn: EmbedFn | None = None,
        reranker: Any | None = None,
    ):
        self.repository = repository or JournalRepository()
        self.chunker = chunker or JournalChunker()
        self.vector_store = vector_store or JournalVectorStore()
        self._uses_default_embed = embed_fn is None
        self.embed_fn = embed_fn or _default_embed
        self._reranker = reranker

    def _project_scope_aliases(
        self, project_slug: str, *, owner_user_id: str | None,
    ) -> tuple[str, list[str]]:
        canonical = canonicalize_scope_path(project_slug)
        if not is_windows_scope(canonical):
            return canonical, [canonical]
        try:
            aliases = self.repository.equivalent_project_slugs(
                owner_user_id=owner_user_id, project_slug=canonical)
        except Exception:  # noqa: BLE001 - keep Journal reads available on SQL alias failure
            logger.exception(
                "journal scope-alias lookup failed for project_slug=%s", canonical)
            aliases = [canonical]
        return canonical, list(dict.fromkeys([canonical, *aliases]))

    def _embed(
        self, texts: Sequence[str], include_sparse: bool, *, priority: str,
    ) -> tuple[list[list[float]], list[dict[str, float]] | None]:
        if self._uses_default_embed:
            if priority == "background" and len(texts) > JOURNAL_EMBED_BATCH:
                dense: list[list[float]] = []
                sparse: list[dict[str, float]] = []
                sparse_complete = include_sparse
                for start in range(0, len(texts), JOURNAL_EMBED_BATCH):
                    batch = texts[start:start + JOURNAL_EMBED_BATCH]
                    # This is deliberately count-only observability: it proves the
                    # real shared model call shape without ever logging Journal text.
                    logger.info(
                        "journal.embed_batch index=%d size=%d total=%d priority=%s",
                        start // JOURNAL_EMBED_BATCH + 1, len(batch), len(texts), priority,
                    )
                    batch_dense, batch_sparse = self.embed_fn(  # type: ignore[call-arg]
                        batch,
                        include_sparse,
                        priority=priority,
                    )
                    dense.extend(batch_dense)
                    if batch_sparse is None:
                        sparse_complete = False
                    else:
                        sparse.extend(batch_sparse)
                return dense, (sparse if sparse_complete else None)
            logger.info(
                "journal.embed_batch index=1 size=%d total=%d priority=%s",
                len(texts), len(texts), priority,
            )
            return self.embed_fn(  # type: ignore[call-arg]
                texts, include_sparse, priority=priority)
        return self.embed_fn(texts, include_sparse)

    @staticmethod
    def _limit_returned_chunks(
        chunks: list[JournalChunkResponse],
        chunk_limit: int | None,
    ) -> list[JournalChunkResponse]:
        if chunk_limit is None:
            return chunks
        body_chunks = [chunk for chunk in chunks if chunk.chunk_index != 0]
        return (body_chunks or chunks)[:chunk_limit]

    def record(self, request: JournalRecordRequest, *, _fan_out: bool = True) -> JournalRecordResponse:
        body = request.body.strip()
        if not body:
            raise ValueError("journal body must be non-empty")
        if _looks_binary(body):
            raise ValueError("journal body appears to be binary, not text; refusing to index")

        entry, project, deduped = self.repository.create_entry(
            project_slug=request.project_slug,
            client=request.client,
            source_kind=request.source_kind,
            title=request.title,
            summary=request.summary,
            body=body,
            tags=request.tags,
            metadata=request.metadata,
            workspace_hint=request.workspace_hint,
            owner_user_id=request.owner_user_id,
            dedupe_policy=request.dedupe_policy,
        )
        if deduped:
            self._supersede_prior_doc_versions(request, entry.entry_id)
            dup = self._fan_out_doc(request) if _fan_out else 0
            return JournalRecordResponse(
                entry=self._entry_response(entry, include_body=False, include_chunks=True),
                chunk_count=len(entry.chunks),
                indexed=entry.indexed_at is not None,
                deduped=True,
                index_error=entry.index_error,
                duplicates_updated=dup,
            )

        chunks = self.chunker.chunk_entry(
            entry_id=entry.entry_id,
            title=request.title,
            summary=request.summary,
            body=body,
            tags=request.tags,
            source_kind=request.source_kind,
            client=request.client,
        )
        self.repository.replace_chunks(entry.entry_id, [self._chunk_to_dict(c) for c in chunks])

        indexed = False
        index_error = None
        try:
            dense, sparse = self._embed(
                [c.content for c in chunks], True, priority="background")
            self.vector_store.upsert_chunks(
                chunks,
                project_id=project.project_id,
                project_slug=project.slug,
                entry_id=entry.entry_id,
                owner_user_id=request.owner_user_id,
                client=request.client,
                source_kind=request.source_kind,
                tags=request.tags,
                dense_vectors=dense,
                sparse_vectors=sparse,
            )
            indexed = True
            self.repository.mark_index_result(entry.entry_id, success=True)
        except Exception as exc:  # noqa: BLE001 - SQL fact must survive index failures.
            index_error = str(exc)
            self.repository.mark_index_result(entry.entry_id, success=False, error=index_error)

        # A failed vector write leaves the SQL fact pending for recovery, but it
        # must not retire the prior searchable revision.  Promotion is only after
        # embed + durable Qdrant upsert + indexed marker succeeded.
        if indexed:
            self._supersede_prior_doc_versions(request, entry.entry_id)
        dup = self._fan_out_doc(request) if indexed and _fan_out else 0

        fresh = self.repository.get_entry(entry.entry_id, include_deleted=True) or entry
        return JournalRecordResponse(
            entry=self._entry_response(fresh, include_body=False, include_chunks=True),
            chunk_count=len(fresh.chunks),
            indexed=indexed,
            deduped=False,
            index_error=index_error,
            duplicates_updated=dup,
        )

    @staticmethod
    def _doc_scope_key(project_slug: str, title: str) -> tuple[str, str]:
        """A durable per-scope logical-file key, including Windows case semantics."""
        canonical_scope = canonicalize_scope_path(project_slug)
        case_sensitive = not is_windows_scope(canonical_scope)
        identity = stored_relative_path_identity(title, case_sensitive=case_sensitive)
        if identity is None:
            raise ValueError("journal document title must be a scope-relative path")
        return (canonical_scope.casefold() if not case_sensitive else canonical_scope, identity)

    @staticmethod
    def _physical_doc_key(project_slug: str, title: str) -> str:
        """Resolve one stored document safely; ``file_abspath`` rejects scope escape."""
        return _abspath(project_slug, title)

    def _owner_physical_doc_index(
        self, *, owner_user_id: str | None,
    ) -> dict[str, list[tuple[str, str]]]:
        """Read the owner's live path inventory once for an entire bulk request.

        The source of truth is physical identity, not content hash: two distinct
        files may have identical bytes, while a parent/child scope can address the
        same file with different relative titles.  Invalid legacy titles are kept out
        of the fan-out map rather than being allowed to escape a scope.
        """
        paths: dict[str, list[tuple[str, str]]] = defaultdict(list)
        seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for scope, title in self.repository.all_live_doc_files(owner_user_id=owner_user_id):
            try:
                physical = self._physical_doc_key(scope, title)
                scope_key = self._doc_scope_key(scope, title)
            except ValueError:
                logger.warning("journal bulk fan-out skipped unsafe stored path scope=%s title=%s", scope, title)
                continue
            if scope_key not in seen[physical]:
                paths[physical].append((scope, title))
                seen[physical].add(scope_key)
        return paths

    def index_docs_bulk(self, *, project_slug: str, owner_user_id: str | None,
                        files: Sequence[JournalBulkDocItem], client: str = "mcp") -> dict[str, int]:
        """Bulk-index changed documents and atomically promote every fan-out scope.

        A request reads the owner's physical-path inventory *once*.  Each changed
        document adds its pre-existing aliases (for example parent/child scopes) to
        the plan.  Chunk text is built once per physical source and embedded once;
        SQL then assigns every target a distinct entry/chunk identity, and Qdrant
        receives one grouped, ``wait=True`` batch.  Latest-wins deletion runs only
        after every new scope payload has been accepted, so a failed target can never
        retire any scope's previous searchable revision.
        """
        canonical_primary_scope = canonicalize_scope_path(project_slug)
        owner_paths = self._owner_physical_doc_index(owner_user_id=owner_user_id)

        # ``plans`` is keyed by the target scope/title, while ``template_key`` names
        # the one physical-source chunk/vector template that target reuses.
        plans: dict[tuple[str, str], dict[str, Any]] = {}
        templates: dict[str, list[JournalTextChunk]] = {}
        incoming_physical: dict[str, dict[str, Any]] = {}
        for item in files:
            if _looks_binary(item.body):
                raise ValueError(f"journal body appears binary: {item.title}")
            if not item.body.strip():
                continue
            source_key = self._doc_scope_key(canonical_primary_scope, item.title)
            if source_key in plans:
                raise ValueError(f"duplicate journal bulk document title: {item.title}")
            physical = self._physical_doc_key(canonical_primary_scope, item.title)
            # A legacy Windows project can differ only in casing/slashes.  It is the
            # same logical scope and must be written in place; otherwise a bulk
            # refresh would create a second project that merely *reads* as an alias.
            stored_scope = canonical_primary_scope
            matching_scopes = [scope for scope, stored_title in owner_paths.get(physical, [])
                               if self._doc_scope_key(scope, stored_title) == source_key]
            if matching_scopes:
                stored_scope = next(
                    (scope for scope in matching_scopes if scope == canonical_primary_scope),
                    matching_scopes[0],
                )
            source = {
                "scope": stored_scope,
                "title": item.title,
                "body": item.body,
                "content_hash": item.content_hash,
                "physical": physical,
                # A same-file request should normally share a hash.  Including the
                # hash protects a rare racing snapshot: two differing reads of the
                # same path never borrow each other's semantic vectors.
                "template_key": f"{physical}\0{item.content_hash}",
            }
            incoming_physical.setdefault(physical, source)
            plans[source_key] = dict(source)

        if not plans:
            return {"files": 0, "deduped": 0, "chunks": 0, "indexed": 0, "chunk_counts": {}}

        # Add aliases from the one owner-wide inventory.  If this request happens to
        # contain two fresh readings of the same physical file with differing hashes,
        # preserve each submitted scope rather than overwriting one snapshot with the
        # other; only the first physical source fans out to existing aliases.
        for physical, source in incoming_physical.items():
            template_key = source["template_key"]
            for target_scope, target_title in owner_paths.get(physical, []):
                target_key = self._doc_scope_key(target_scope, target_title)
                if target_key in plans:
                    continue
                plans[target_key] = {
                    "scope": target_scope,
                    "title": target_title,
                    "body": source["body"],
                    "content_hash": source["content_hash"],
                    "physical": physical,
                    "template_key": template_key,
                }

        # A template's text deliberately comes from its primary changed source.  The
        # target's SQL/Qdrant metadata still carries its own relative title/scope; the
        # semantic content and vectors are identical because it is the same file.
        for plan in plans.values():
            template_key = plan["template_key"]
            if template_key not in templates:
                templates[template_key] = self.chunker.chunk_entry(
                    entry_id=f"journal_template_{len(templates)}",
                    title=plan["title"], summary=None, body=plan["body"], tags=[],
                    source_kind="doc", client=client)

        by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for plan in plans.values():
            by_scope[plan["scope"]].append({
                "title": plan["title"],
                "body": plan["body"],
                "chunks": templates[plan["template_key"]],
                "template_key": plan["template_key"],
            })

        created: list[dict[str, Any]] = []
        for target_scope, documents in by_scope.items():
            project_id, stored_scope, rows = self.repository.create_doc_entries_bulk(
                project_slug=target_scope,
                owner_user_id=owner_user_id,
                client=client,
                documents=documents,
            )
            for row in rows:
                row["project_id"] = project_id
                row["project_slug"] = stored_scope
            created.extend(rows)

        pending = [row for row in created if not row["deduped"]]
        try:
            vectors_by_template: dict[str, tuple[list[list[float]], list[dict[str, float]] | None]] = {}
            required_templates = list(dict.fromkeys(
                row["template_key"] for row in pending))
            if required_templates:
                template_chunks = [
                    chunk for template_key in required_templates
                    for chunk in templates[template_key]
                ]
                dense, sparse = self._embed(
                    [chunk.content for chunk in template_chunks], True, priority="background")
                offset = 0
                for template_key in required_templates:
                    count = len(templates[template_key])
                    vectors_by_template[template_key] = (
                        dense[offset:offset + count],
                        sparse[offset:offset + count] if sparse is not None else None,
                    )
                    offset += count

                groups = [JournalChunkUpsert(
                    chunks=row["chunks"],
                    project_id=row["project_id"],
                    project_slug=row["project_slug"],
                    entry_id=row["entry_id"],
                    owner_user_id=owner_user_id,
                    client=client,
                    source_kind="doc",
                    tags=[],
                    dense_vectors=vectors_by_template[row["template_key"]][0],
                    sparse_vectors=vectors_by_template[row["template_key"]][1],
                ) for row in pending]
                expected_points = sum(len(group.chunks) for group in groups)
                if hasattr(self.vector_store, "upsert_chunk_groups"):
                    written = self.vector_store.upsert_chunk_groups(groups)
                else:  # compatibility for old test/local vector adapters only
                    written = sum(self.vector_store.upsert_chunks(
                        group.chunks,
                        project_id=group.project_id,
                        project_slug=group.project_slug,
                        entry_id=group.entry_id,
                        owner_user_id=group.owner_user_id,
                        client=group.client,
                        source_kind=group.source_kind,
                        tags=group.tags,
                        dense_vectors=group.dense_vectors,
                        sparse_vectors=group.sparse_vectors,
                    ) for group in groups)
                if written != expected_points:
                    raise RuntimeError(
                        f"journal Qdrant bulk write incomplete: wrote {written} of {expected_points} points")
                self.repository.mark_index_results_bulk(
                    [row["entry_id"] for row in pending], success=True)

            # Promotion is intentionally outside the individual write loop.  A target
            # failure above leaves every old revision live; no scope is partially
            # latest-wins promoted merely because it happened to be written first.
            for row in created:
                self._supersede_prior_doc_versions(JournalRecordRequest(
                    project_slug=row["project_slug"], client=client, source_kind="doc",
                    title=row["title"], body="promotion", owner_user_id=owner_user_id,
                    dedupe_policy="return_existing"), row["entry_id"])
        except Exception as exc:
            self.repository.mark_index_results_bulk(
                [row["entry_id"] for row in pending], success=False, error=str(exc))
            # A later 256-point Qdrant batch can fail after an earlier one landed.
            # These entries were never promoted, so remove only their speculative
            # points when the backend is still reachable.  Cleanup is best-effort:
            # preserving old revisions is the correctness boundary, and a down
            # Qdrant must not hide the original write error behind cleanup failure.
            pending_ids = [row["entry_id"] for row in pending]
            try:
                if hasattr(self.vector_store, "delete_by_entry_ids"):
                    self.vector_store.delete_by_entry_ids(pending_ids)
                else:
                    for entry_id in pending_ids:
                        self.vector_store.delete_by_entry_id(entry_id)
            except Exception:  # noqa: BLE001 - original Qdrant failure is authoritative
                logger.warning("journal bulk fan-out could not clean speculative vectors", exc_info=True)
            raise

        indexed_chunks = sum(len(row["chunks"]) for row in pending)
        return {"files": len(created), "deduped": len(created) - len(pending),
                "chunks": indexed_chunks, "indexed": indexed_chunks,
                "chunk_counts": {row["title"]: len(row["chunks"]) for row in created}}

    def _fan_out_doc(self, request: JournalRecordRequest) -> int:
        """方案2 (text side): after recording a doc, refresh the SAME physical file wherever
        it's indexed under ANOTHER scope of this owner (matched by absolute path =
        project_slug/title), so every copy carries this record's latest content + vectors.

        Re-records into each other scope with fan-out disabled (no recursion — the outer
        call already visits every other scope); latest-wins supersede retires that scope's
        prior version there, leaving exactly one live copy per scope so BOTH scopes stay
        complete. Only titled source_kind="doc" entries participate; ACE memory is per-scope
        and never syncs. Best-effort: an enumeration failure skips the sync (the primary
        record already succeeded). Returns how many other-scope copies were refreshed."""
        if request.source_kind != "doc" or not request.title:
            return 0
        this_abs = _abspath(request.project_slug, request.title)
        try:
            owned = self.repository.all_live_doc_files(owner_user_id=request.owner_user_id)
        except Exception:
            logger.exception("journal fan-out: enumerating owner docs failed — skipping sync")
            return 0
        targets = {(slug, title) for slug, title in owned
                   if slug != request.project_slug and _abspath(slug, title) == this_abs}
        updated = 0
        for slug, title in targets:
            try:
                self.record(JournalRecordRequest(
                    project_slug=slug, client=request.client, source_kind="doc",
                    title=title, body=request.body, summary=request.summary,
                    tags=request.tags, metadata=request.metadata,
                    workspace_hint=request.workspace_hint,
                    owner_user_id=request.owner_user_id, dedupe_policy="return_existing",
                ), _fan_out=False)
                updated += 1
                logger.info("journal fan-out: synced %s -> scope=%s title=%s", this_abs, slug, title)
            except Exception:
                logger.exception("journal fan-out: failed syncing %s -> scope=%s title=%s",
                                 this_abs, slug, title)
        return updated

    def _supersede_prior_doc_versions(self, request: JournalRecordRequest, entry_id: str) -> None:
        """Latest-wins for indexed files (W153): a doc represents its file's CURRENT
        content, so an edited re-index must retire prior versions instead of piling up.

        Scoped to source_kind="doc" + return_existing (how journal_index_dir records
        files) + a title (the file path = the file's identity). Memory stays append-only
        because it never enters this branch. Runs on both the fresh and the dedup path
        (excluding the winning entry), so re-indexing unchanged content still sweeps any
        pre-existing stale duplicate — exactly one live version per file, always.
        """
        if request.source_kind != "doc" or request.dedupe_policy != "return_existing" or not request.title:
            return
        stale = self.repository.superseded_version_ids(
            project_slug=self._project_scope_aliases(
                request.project_slug, owner_user_id=request.owner_user_id)[1],
            source_kind=request.source_kind,
            title=request.title,
            owner_user_id=request.owner_user_id,
            exclude_entry_id=entry_id,
            case_sensitive=not is_windows_scope(request.project_slug),
        )
        for old_id in stale:
            self.delete_entry(old_id, reason="superseded by newer version", owner_user_id=request.owner_user_id)

    def list_entries(
        self,
        *,
        project_slug: str,
        source_kind: str | None = None,
        client: str | None = None,
        tag: str | None = None,
        q: str | None = None,
        owner_user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> JournalEntryListResponse:
        _canonical, project_slugs = self._project_scope_aliases(
            project_slug, owner_user_id=owner_user_id)
        rows, total = self.repository.list_entries(
            project_slug=project_slugs,
            source_kind=source_kind,
            client=client,
            tag=tag,
            q=q,
            owner_user_id=owner_user_id,
            limit=limit,
            offset=offset,
        )
        return JournalEntryListResponse(
            entries=[self._entry_response(e, include_body=False, include_chunks=False) for e in rows],
            total=total,
        )

    def get_entry(self, entry_id: str, owner_user_id: str | None = None) -> JournalEntryResponse | None:
        entry = self.repository.get_entry(entry_id, owner_user_id=owner_user_id)
        return self._entry_response(entry, include_body=True, include_chunks=True) if entry else None

    def delete_entry(
        self,
        entry_id: str,
        reason: str | None = None,
        owner_user_id: str | None = None,
    ) -> JournalDeleteResponse:
        deleted = self.repository.soft_delete_entry(entry_id, reason=reason, owner_user_id=owner_user_id)
        deleted_vectors = self.vector_store.delete_by_entry_id(entry_id) if deleted else 0
        return JournalDeleteResponse(entry_id=entry_id, deleted=deleted, deleted_vectors=deleted_vectors)

    def _delete_entries(
        self, entry_ids: list[str], *, reason: str, owner_user_id: str | None,
    ) -> tuple[list[str], int]:
        """Delete a batch of entries (each: SQL soft-delete + vector purge), accumulating
        the ones actually removed and their vector count. Shared by delete_doc_scope
        (whole scope) and reconcile_doc_scope (stale snapshot subset)."""
        removed_ids = self.repository.soft_delete_entries_bulk(
            entry_ids, reason=reason, owner_user_id=owner_user_id)
        # MatchAny delete is one wait=True/retrying Qdrant mutation rather than one
        # network round trip per stale revision.
        if not removed_ids:
            deleted_vectors = 0
        elif hasattr(self.vector_store, "delete_by_entry_ids"):
            deleted_vectors = self.vector_store.delete_by_entry_ids(removed_ids)
        else:  # compatibility for pre-bulk JournalVectorStore implementations
            deleted_vectors = sum(self.vector_store.delete_by_entry_id(entry_id) for entry_id in removed_ids)
        return removed_ids, deleted_vectors

    def delete_doc_scope(
        self,
        *,
        project_slug: str,
        owner_user_id: str | None = None,
    ) -> JournalDocScopeDeleteResponse:
        """Explicit clear of every source_kind="doc" entry in a scope (+ their vectors) —
        the symmetric counterpart to delete_code. Leaves ACE memory entries untouched.
        Idempotent: an empty scope removes nothing."""
        project_slug, project_slugs = self._project_scope_aliases(
            project_slug, owner_user_id=owner_user_id)
        entry_ids = self.repository.all_doc_entry_ids(
            project_slug=project_slugs, owner_user_id=owner_user_id)
        removed_ids, deleted_vectors = self._delete_entries(
            entry_ids, reason="doc scope cleared", owner_user_id=owner_user_id)
        return JournalDocScopeDeleteResponse(
            project_slug=project_slug,
            removed_entries=len(removed_ids),
            removed_entry_ids=removed_ids,
            deleted_vectors=deleted_vectors,
            scope_aliases=[scope for scope in project_slugs if scope != project_slug],
        )

    def reconcile_doc_scope(
        self,
        *,
        project_slug: str,
        live_titles: list[str],
        owner_user_id: str | None = None,
        max_depth: int | None = None,
        patterns: list[str] | None = None,
    ) -> JournalDocReconcileResponse:
        project_slug, project_slugs = self._project_scope_aliases(
            project_slug, owner_user_id=owner_user_id)
        normalized_titles = sorted({title for title in live_titles if title})
        if not normalized_titles:
            return JournalDocReconcileResponse(
                project_slug=project_slug,
                live_titles=0,
                removed_entries=0,
                skipped=True,
                reason="empty snapshot; stale doc sweep skipped",
            )

        existing_titles = self.repository.live_doc_titles(
            project_slug=project_slugs,
            owner_user_id=owner_user_id,
        )
        case_sensitive = not is_windows_scope(project_slug)
        existing_identities = {
            identity for title in existing_titles
            if (identity := stored_relative_path_identity(
                title, case_sensitive=case_sensitive)) is not None
        }
        live_identities = {
            identity for title in normalized_titles
            if (identity := stored_relative_path_identity(
                title, case_sensitive=case_sensitive)) is not None
        }
        if existing_identities and existing_identities.isdisjoint(live_identities):
            return JournalDocReconcileResponse(
                project_slug=project_slug,
                live_titles=len(normalized_titles),
                removed_entries=0,
                skipped=True,
                reason="snapshot is disjoint from existing docs; stale doc sweep skipped",
            )

        stale_ids = self.repository.stale_doc_entry_ids_for_snapshot(
            project_slug=project_slugs,
            live_titles=normalized_titles,
            owner_user_id=owner_user_id,
            max_depth=max_depth,
            patterns=patterns,
            case_sensitive=case_sensitive,
        )
        removed_ids, deleted_vectors = self._delete_entries(
            stale_ids, reason="absent from latest doc snapshot", owner_user_id=owner_user_id)

        return JournalDocReconcileResponse(
            project_slug=project_slug,
            live_titles=len(normalized_titles),
            removed_entries=len(removed_ids),
            removed_entry_ids=removed_ids,
            deleted_vectors=deleted_vectors,
            skipped=False,
        )

    def _resolve_path_glob_ids(self, request: JournalSearchRequest) -> "_GlobResolution":
        """Resolve request.path_glob to the entry_ids it matches inside this scope, so the
        match-set can be pushed down as a search filter (top_k fills only from matching
        docs — never post-filter-squeezed, so no top_k is ever dropped). A doc's title IS
        its scope-relative path, so path_matches(title, glob) is the membership test.

        ``ids`` is None when no path_glob is set — the caller then adds ZERO filter and the
        query is byte-identical to a normal search (no perf regression). ``ids`` is a
        possibly EMPTY list when a glob matched nothing; the caller short-circuits to no
        results (an empty filter must never widen to "match everything"). The scope's doc
        count + a sample title ride along so an empty hit can be EXPLAINED (the glob is
        scope-relative and the user's depth is wrong) rather than silently returning
        nothing while the scope in fact has docs."""
        if not request.path_glob:
            return _GlobResolution(ids=None, scope_doc_count=0, sample_title=None)
        _canonical, project_slugs = self._project_scope_aliases(
            request.project_slug, owner_user_id=request.owner_user_id)
        pairs = self.repository.scoped_live_doc_id_titles(
            project_slug=project_slugs,
            owner_user_id=request.owner_user_id,
        )
        ids = [
            entry_id
            for entry_id, title in pairs
            if _path_matches(
                title,
                request.path_glob,
                case_sensitive=not is_windows_scope(request.project_slug),
            )
        ]
        titles = [title for _, title in pairs if title]
        # Shortest (then lexical) title = the most representative example of the scope's
        # actual path shape — deterministic so the notice is stable.
        sample = min(titles, key=lambda t: (len(t), t)) if titles else None
        return _GlobResolution(ids=ids, scope_doc_count=len(pairs), sample_title=sample)

    @staticmethod
    def _path_glob_empty_notice(request: JournalSearchRequest, res: "_GlobResolution") -> str | None:
        """Explain why a path_glob returned nothing — but only when the scope actually HAS
        docs (else a hint about 'docs in scope' would be misleading). None ⇒ plain no-match."""
        if res.scope_doc_count == 0:
            return None
        sample = f" A doc here looks like {res.sample_title!r}." if res.sample_title else ""
        return (f"path_glob {request.path_glob!r} matched 0 of {res.scope_doc_count} doc(s) in "
                f"scope {request.project_slug!r}. path_glob is relative to THIS scope's root; "
                f"check scope_dir and the glob's depth.{sample}")

    @staticmethod
    def _empty_search_response(request: JournalSearchRequest, *, engine: str,
                              notice: str | None = None) -> JournalSearchResponse:
        # Keep the metrics shape stable with the normal search paths (both carry
        # retrieve_ms) so consumers can read metrics["retrieve_ms"] unconditionally.
        return JournalSearchResponse(
            query=request.query,
            project_slug=request.project_slug,
            mode=request.mode,
            total=0,
            hits=[],
            notice=notice,
            metrics={"retrieve_ms": 0.0, "engine": engine, "candidates": 0,
                     "returned": 0, "path_glob": request.path_glob},
        )

    def search(self, request: JournalSearchRequest) -> JournalSearchResponse:
        if request.mode == "keyword":
            return self._keyword_search(request)
        _canonical, project_slugs = self._project_scope_aliases(
            request.project_slug, owner_user_id=request.owner_user_id)
        res = self._resolve_path_glob_ids(request)
        glob_ids = res.ids
        if glob_ids is not None and not glob_ids:
            return self._empty_search_response(
                request, engine="qdrant", notice=self._path_glob_empty_notice(request, res))
        t0 = time.perf_counter()
        dense, sparse = self._embed(
            [request.query], True, priority="interactive")
        query_dense = dense[0] if dense else []
        query_sparse = sparse[0] if sparse else None
        filters: dict[str, Any] = {
            "project_slug": project_slugs[0] if len(project_slugs) == 1 else project_slugs,
        }
        if request.owner_user_id is not None:
            filters["owner_user_id"] = request.owner_user_id
        if request.source_kind:
            filters["source_kind"] = request.source_kind
        if request.client:
            filters["client"] = request.client
        if request.title_only:
            filters["level"] = 1
        elif request.level is not None:
            filters["level"] = request.level
        if request.tags:
            filters["tags"] = request.tags
        if glob_ids is not None:
            # Push the matching id-set down as a MatchAny filter so top_k fills only from
            # matching docs — dense/sparse/hybrid alike — with no post-filter drop.
            filters["entry_id"] = glob_ids

        retrieve_k = request.top_k * 5 if (request.dedup_by_entry or request.rerank) else request.top_k
        raw = self.vector_store.search(
            dense_vector=query_dense,
            sparse_vector=query_sparse,
            mode=request.mode,
            top_k=retrieve_k,
            filters=filters,
        )
        retrieve_ms = round((time.perf_counter() - t0) * 1000, 1)

        chunk_hits = [self._raw_hit_to_chunk(hit) for hit in raw]
        candidates_before_rerank = len(chunk_hits)
        rerank_ms = 0.0
        if request.rerank and chunk_hits:
            t_rerank = time.perf_counter()
            chunk_hits = self._rerank_chunks(request.query, chunk_hits)
            rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 1)
        if request.dedup_by_entry:
            chunk_hits = self._dedup_chunks_by_entry(chunk_hits)
        chunk_hits = chunk_hits[:request.top_k]

        entries = self.repository.get_entries_by_ids(
            [chunk.entry_id for chunk in chunk_hits],
            owner_user_id=request.owner_user_id,
        )
        chunks_by_entry: dict[str, list[JournalChunkResponse]] = defaultdict(list)
        for chunk in chunk_hits:
            chunks_by_entry[chunk.entry_id].append(chunk)

        hits: list[JournalSearchHit] = []
        emitted: set[str] = set()
        for chunk in chunk_hits:
            if request.dedup_by_entry and chunk.entry_id in emitted:
                continue
            entry = entries.get(chunk.entry_id)
            if entry is None:
                continue
            grouped = chunks_by_entry[chunk.entry_id]
            best = grouped[0]
            hits.append(JournalSearchHit(
                entry=self._entry_response(entry, include_body=False, include_chunks=False),
                score=best.score or 0.0,
                best_chunk=best,
                chunks=(
                    self._limit_returned_chunks(grouped, request.chunk_limit)
                    if request.include_chunks else []
                ),
            ))
            emitted.add(chunk.entry_id)

        return JournalSearchResponse(
            query=request.query,
            project_slug=request.project_slug,
            mode=request.mode,
            total=len(hits),
            hits=hits,
            metrics={
                "retrieve_ms": retrieve_ms,
                "rerank_ms": rerank_ms,
                "candidates": len(raw),
                "candidates_before_rerank": candidates_before_rerank,
                "candidates_after_rerank": len(chunk_hits),
                "returned": len(hits),
                "dedup_by_entry": request.dedup_by_entry,
                "rerank": request.rerank,
            },
        )

    def _keyword_search(self, request: JournalSearchRequest) -> JournalSearchResponse:
        """Embedding-free FTS5 keyword search — resilient when embed endpoint is down."""
        _canonical, project_slugs = self._project_scope_aliases(
            request.project_slug, owner_user_id=request.owner_user_id)
        res = self._resolve_path_glob_ids(request)
        glob_ids = res.ids
        if glob_ids is not None and not glob_ids:
            return self._empty_search_response(
                request, engine="fts5", notice=self._path_glob_empty_notice(request, res))
        t0 = time.perf_counter()
        limit = request.top_k * 5 if request.rerank else request.top_k
        rows, _ = self.repository.keyword_search(
            query=request.query,
            project_slug=project_slugs,
            source_kind=request.source_kind,
            owner_user_id=request.owner_user_id,
            entry_ids=glob_ids,
            limit=limit,
            load_chunks=not (request.include_chunks and request.chunk_limit is not None),
        )
        retrieve_ms = round((time.perf_counter() - t0) * 1000, 1)
        hits = self._keyword_hits(
            rows,
            query=request.query,
            top_k=request.top_k,
            include_chunks=request.include_chunks,
            owner_user_id=request.owner_user_id,
            chunk_limit=request.chunk_limit,
        )
        return JournalSearchResponse(
            query=request.query,
            project_slug=request.project_slug,
            mode="keyword",
            total=len(hits),
            hits=hits,
            metrics={"retrieve_ms": retrieve_ms, "engine": "fts5",
                     "candidates": len(rows), "returned": len(hits)},
        )

    def _keyword_hits(
        self,
        rows: list[tuple[JournalEntry, float]],
        *,
        query: str | None = None,
        top_k: int,
        include_chunks: bool,
        owner_user_id: str | None = None,
        chunk_limit: int | None = None,
    ) -> list[JournalSearchHit]:
        hits: list[JournalSearchHit] = []
        limited_chunks: dict[str, list[JournalChunk]] = {}
        if include_chunks and chunk_limit is not None:
            entry_ids = [entry.entry_id for entry, _ in rows[:top_k]]
            if query:
                limited_chunks = self.repository.matching_chunks_by_entry_ids(
                    entry_ids,
                    query=query,
                    limit=chunk_limit,
                    owner_user_id=owner_user_id,
                )
            fallback_ids = [
                entry_id
                for entry_id in entry_ids
                if not any(chunk.chunk_index != 0 for chunk in limited_chunks.get(entry_id, []))
            ]
            for entry_id in fallback_ids:
                limited_chunks.pop(entry_id, None)
            if fallback_ids:
                limited_chunks.update(self.repository.limited_chunks_by_entry_ids(
                    fallback_ids,
                    limit=chunk_limit,
                    owner_user_id=owner_user_id,
                ))
        for entry, score in rows[:top_k]:
            if entry.entry_id in limited_chunks:
                source_chunks = limited_chunks[entry.entry_id]
            else:
                source_chunks = entry.chunks
            chunk_resps = [self._chunk_response(c) for c in sorted(source_chunks, key=lambda c: c.chunk_index)]
            returned_chunks = (
                self._limit_returned_chunks(chunk_resps, chunk_limit)
                if include_chunks else []
            )
            hits.append(JournalSearchHit(
                entry=self._entry_response(entry, include_body=False, include_chunks=False),
                score=float(score),
                best_chunk=(returned_chunks[0] if returned_chunks else (chunk_resps[0] if chunk_resps else None)),
                chunks=returned_chunks,
            ))
        return hits

    def admin_search(
        self,
        *,
        query: str,
        mode: str = "keyword",
        top_k: int = 20,
        owner_user_id: str | None = None,
        project_slug: str | None = None,
        source_kind: str | None = None,
        include_chunks: bool = False,
        chunk_limit: int | None = 5,
        rerank: bool = False,
    ) -> JournalSearchResponse:
        """Cross-user/cross-project search for the admin console.

        `owner_user_id=None` spans every user; pass a value to scope to one. Keyword
        mode needs no embedding (works while the embed endpoint is down).
        """
        if mode == "keyword":
            rows, _ = self.repository.keyword_search(
                query=query, project_slug=project_slug, source_kind=source_kind,
                owner_user_id=owner_user_id, limit=top_k * (5 if rerank else 1),
                load_chunks=not (include_chunks and chunk_limit is not None),
            )
            hits = self._keyword_hits(
                rows,
                query=query,
                top_k=top_k,
                include_chunks=include_chunks,
                owner_user_id=owner_user_id,
                chunk_limit=chunk_limit,
            )
            return JournalSearchResponse(
                query=query, project_slug=project_slug or "", mode="keyword",
                total=len(hits), hits=hits,
                metrics={"engine": "fts5", "candidates": len(rows), "returned": len(hits)},
            )

        dense, sparse = self._embed([query], True, priority="interactive")
        filters: dict[str, Any] = {}
        if owner_user_id is not None:
            filters["owner_user_id"] = owner_user_id
        if project_slug:
            filters["project_slug"] = project_slug
        if source_kind:
            filters["source_kind"] = source_kind
        raw = self.vector_store.search(
            dense_vector=dense[0] if dense else [],
            sparse_vector=sparse[0] if sparse else None,
            mode=mode, top_k=top_k * 5, filters=filters,
        )
        chunk_hits = [self._raw_hit_to_chunk(hit) for hit in raw]
        if rerank and chunk_hits:
            chunk_hits = self._rerank_chunks(query, chunk_hits)
        chunk_hits = self._dedup_chunks_by_entry(chunk_hits)[:top_k]
        entries = self.repository.get_entries_by_ids(
            [chunk.entry_id for chunk in chunk_hits], owner_user_id=owner_user_id,
        )
        hits = []
        for chunk in chunk_hits:
            entry = entries.get(chunk.entry_id)
            if entry is None:
                continue
            hits.append(JournalSearchHit(
                entry=self._entry_response(entry, include_body=False, include_chunks=False),
                score=chunk.score or 0.0,
                best_chunk=chunk,
                chunks=[chunk] if include_chunks else [],
            ))
        return JournalSearchResponse(
            query=query, project_slug=project_slug or "", mode=mode,  # type: ignore[arg-type]
            total=len(hits), hits=hits,
            metrics={"candidates": len(raw), "returned": len(hits)},
        )

    def admin_list_entries(
        self,
        *,
        owner_user_id: str | None = None,
        project_slug: str | None = None,
        source_kind: str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> JournalEntryListResponse:
        rows, total = self.repository.list_entries(
            project_slug=project_slug, source_kind=source_kind, q=q,
            owner_user_id=owner_user_id, limit=limit, offset=offset,
        )
        return JournalEntryListResponse(
            entries=[self._entry_response(e, include_body=False, include_chunks=False) for e in rows],
            total=total,
        )

    def stats(self) -> dict[str, int]:
        return self.repository.stats()

    def owner_ids(self) -> list[str]:
        return self.repository.list_owner_ids()

    def pending_entries(self, *, limit: int = 100) -> JournalEntryListResponse:
        """The reindex queue as entry rows (indexed_at NULL, live) for the console UI."""
        rows = self.repository.list_pending_index(limit=limit)
        return JournalEntryListResponse(
            entries=[self._entry_response(e, include_body=False, include_chunks=False) for e in rows],
            total=len(rows),
        )

    def reindex_pending(self, *, limit: int = 100) -> dict[str, int]:
        """Re-embed + upsert entries whose vectors are missing (indexed_at NULL).

        This is the reconcile/queue drain: entries recorded while the embed endpoint
        was down kept their SQL fact but never got vectors, leaving SQL and Qdrant
        inconsistent. Re-embeds their stored chunks and marks them indexed. Safe to
        run repeatedly; returns {reindexed, failed, remaining}.
        """
        pending = self.repository.list_pending_index(limit=limit)
        reindexed = failed = 0
        for entry in pending:
            chunks = sorted(entry.chunks, key=lambda c: c.chunk_index)
            if not chunks:
                # No chunks to vectorize — nothing to reconcile; treat as indexed.
                self.repository.mark_index_result(entry.entry_id, success=True)
                reindexed += 1
                continue
            try:
                dense, sparse = self._embed(
                    [c.content for c in chunks], True, priority="background")
                project = entry.project
                self.vector_store.upsert_chunks(
                    chunks,
                    project_id=entry.project_id,
                    project_slug=project.slug if project else "",
                    entry_id=entry.entry_id,
                    owner_user_id=project.owner_user_id if project else None,
                    client=entry.client,
                    source_kind=entry.source_kind,
                    tags=entry.tags_json or [],
                    dense_vectors=dense,
                    sparse_vectors=sparse,
                )
                self.repository.mark_index_result(entry.entry_id, success=True)
                reindexed += 1
            except Exception as exc:  # noqa: BLE001 - keep draining; SQL fact survives.
                self.repository.mark_index_result(entry.entry_id, success=False, error=str(exc))
                failed += 1
        remaining = len(self.repository.list_pending_index(limit=limit + 1))
        return {"reindexed": reindexed, "failed": failed, "remaining": remaining}

    def _rerank_chunks(self, query: str, chunks: list[JournalChunkResponse]) -> list[JournalChunkResponse]:
        reranker = self._get_reranker()
        search_results = [
            SearchResult(
                doc_id=chunk.entry_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=chunk.score or 0.0,
                doc_title=chunk.heading or "",
                source="journal",
                metadata={
                    **chunk.metadata,
                    "entry_id": chunk.entry_id,
                    "project_id": chunk.project_id,
                    "level": chunk.level,
                    "chunk_index": chunk.chunk_index,
                    "retrieval_score": chunk.score,
                },
            )
            for chunk in chunks
        ]
        ranked = reranker.rerank(query, search_results, top_k=len(search_results))
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        out: list[JournalChunkResponse] = []
        for result in ranked:
            chunk = chunks_by_id.get(result.chunk_id)
            if chunk is None:
                continue
            metadata = {
                **chunk.metadata,
                "retrieval_score": chunk.score,
                "reranker_score": result.metadata.get("reranker_score", result.score),
            }
            out.append(chunk.model_copy(update={"score": float(result.score), "metadata": metadata}))
        return out

    def _get_reranker(self):
        if self._reranker is None:
            from core.retrieval.reranker import get_shared_reranker
            self._reranker = get_shared_reranker()
        return self._reranker

    @staticmethod
    def _dedup_chunks_by_entry(chunks: list[JournalChunkResponse]) -> list[JournalChunkResponse]:
        seen: set[str] = set()
        out: list[JournalChunkResponse] = []
        for chunk in chunks:
            if chunk.entry_id in seen:
                continue
            seen.add(chunk.entry_id)
            out.append(chunk)
        return out

    @staticmethod
    def _chunk_to_dict(chunk: JournalTextChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "level": chunk.level,
            "heading": chunk.heading,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _raw_hit_to_chunk(hit: dict[str, Any]) -> JournalChunkResponse:
        meta = hit.get("metadata") or {}
        return JournalChunkResponse(
            chunk_id=str(meta.get("chunk_id") or hit.get("id") or ""),
            entry_id=str(meta.get("entry_id") or ""),
            project_id=int(meta.get("project_id") or 0),
            chunk_index=int(meta.get("chunk_index") or 0),
            level=int(meta.get("level") or 0),
            heading=meta.get("heading") or None,
            content=hit.get("content") or "",
            token_count=meta.get("token_count"),
            score=float(hit.get("score") or 0.0),
            metadata={k: v for k, v in meta.items() if k not in {
                "content", "chunk_id", "entry_id", "project_id", "level", "heading", "chunk_index",
            }},
        )

    @staticmethod
    def _chunk_response(row: JournalChunk, score: float | None = None) -> JournalChunkResponse:
        return JournalChunkResponse(
            chunk_id=row.chunk_id,
            entry_id=row.entry_id,
            project_id=row.project_id,
            chunk_index=row.chunk_index,
            level=row.level,
            heading=row.heading,
            content=row.content,
            token_count=row.token_count,
            score=score,
            metadata=row.metadata_json or {},
        )

    def _entry_response(
        self,
        entry: JournalEntry,
        *,
        include_body: bool,
        include_chunks: bool,
    ) -> JournalEntryResponse:
        project = entry.project
        chunks = sorted(entry.chunks, key=lambda c: c.chunk_index) if include_chunks else []
        return JournalEntryResponse(
            entry_id=entry.entry_id,
            project_id=entry.project_id,
            project_slug=project.slug if project else "",
            client=entry.client,
            source_kind=entry.source_kind,
            title=entry.title,
            summary=entry.summary,
            body=entry.body if include_body else None,
            tags=entry.tags_json or [],
            metadata=entry.metadata_json or {},
            content_hash=entry.content_hash,
            # Prefer the repository-provided count (set when chunks aren't hydrated); else
            # fall back to the loaded chunk list (0 when chunks weren't requested).
            chunk_count=int(getattr(entry, "_chunk_count",
                                    len(entry.chunks) if include_chunks else 0)),
            indexed=entry.indexed_at is not None,
            index_error=entry.index_error,
            deleted=entry.deleted_at is not None,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            chunks=[self._chunk_response(c) for c in chunks],
        )


_SERVICE: JournalService | None = None


def get_journal_service() -> JournalService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = JournalService()
    return _SERVICE
