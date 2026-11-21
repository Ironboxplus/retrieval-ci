"""SQLAlchemy repository for ACE journal facts."""
from __future__ import annotations

import fnmatch
import hashlib
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import case, func, inspect, or_, text
from sqlalchemy.orm import Session, selectinload

from app.services.journal import fts
from app.services.scope_paths import canonicalize_scope_path, stored_relative_path_identity
from core.db.database import get_db_session
from core.db.models import JournalChunk, JournalEntry, JournalProject


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def new_entry_id() -> str:
    return f"journal_entry_{uuid.uuid4().hex[:20]}"


def new_chunk_id(entry_id: str, chunk_index: int, level: int, content: str) -> str:
    digest = hashlib.sha256(
        f"{entry_id}:{chunk_index}:{level}:{content}".encode("utf-8")
    ).hexdigest()[:20]
    return f"journal_chunk_{digest}"


class JournalRepository:
    """Repository boundary for journal SQL tables."""

    def __init__(self, session_factory: Callable[[], Any] = get_db_session):
        self.session_factory = session_factory

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            yield session

    def get_or_create_project(
        self,
        session: Session,
        *,
        slug: str,
        display_name: str | None = None,
        workspace_hint: str | None = None,
        owner_user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JournalProject:
        slug = slug.strip()
        project_query = session.query(JournalProject).filter(JournalProject.slug == slug)
        if owner_user_id is None:
            project_query = project_query.filter(JournalProject.owner_user_id.is_(None))
        else:
            project_query = project_query.filter(JournalProject.owner_user_id == owner_user_id)
        project = project_query.one_or_none()
        if project is not None:
            changed = False
            if display_name and project.display_name != display_name:
                project.display_name = display_name
                changed = True
            if workspace_hint and project.workspace_hint != workspace_hint:
                project.workspace_hint = workspace_hint
                changed = True
            if metadata:
                project.metadata_json = {**(project.metadata_json or {}), **metadata}
                changed = True
            if changed:
                project.updated_at = _utcnow()
            return project

        project = JournalProject(
            slug=slug,
            display_name=display_name or slug,
            workspace_hint=workspace_hint,
            owner_user_id=owner_user_id,
            metadata_json=metadata or {},
        )
        session.add(project)
        session.flush()
        return project

    def equivalent_project_slugs(
        self, *, owner_user_id: str | None, project_slug: str,
    ) -> list[str]:
        """Canonical scope plus legacy Windows spellings owned by the same user."""
        canonical = canonicalize_scope_path(project_slug)
        with self.session() as session:
            # Legacy writers differed in component casing and sometimes slash style.
            # Narrow in SQLite first so the hot path never hydrates every project slug
            # owned by a large tenant; Python canonicalization is only the final guard.
            comparable_slug = func.lower(func.replace(JournalProject.slug, "\\", "/"))
            query = session.query(JournalProject.slug).filter(
                comparable_slug == canonical.casefold())
            if owner_user_id is None:
                query = query.filter(JournalProject.owner_user_id.is_(None))
            else:
                query = query.filter(JournalProject.owner_user_id == owner_user_id)
            stored = [row[0] for row in query.all()]
        return list(dict.fromkeys([
            canonical,
            *(slug for slug in stored if canonicalize_scope_path(slug) == canonical),
        ]))

    def create_entry(
        self,
        *,
        project_slug: str,
        client: str,
        source_kind: str,
        title: str | None,
        summary: str | None,
        body: str,
        tags: list[str],
        metadata: dict[str, Any],
        workspace_hint: str | None = None,
        owner_user_id: str | None = None,
        dedupe_policy: str = "append",
    ) -> tuple[JournalEntry, JournalProject, bool]:
        body_hash = content_hash(body)
        with self.session() as session:
            project = self.get_or_create_project(
                session,
                slug=project_slug,
                workspace_hint=workspace_hint,
                owner_user_id=owner_user_id,
            )
            if dedupe_policy == "return_existing":
                filters = [
                    JournalEntry.project_id == project.project_id,
                    JournalEntry.source_kind == source_kind,
                    JournalEntry.content_hash == body_hash,
                    JournalEntry.deleted_at.is_(None),
                ]
                if source_kind == "doc" and title:
                    filters.append(JournalEntry.title == title)
                existing = (
                    session.query(JournalEntry)
                    .options(selectinload(JournalEntry.chunks), selectinload(JournalEntry.project))
                    .filter(*filters)
                    .order_by(JournalEntry.created_at.desc())
                    .first()
                )
                if existing is not None:
                    self._detach_entry(session, existing)
                    return existing, project, True

            entry = JournalEntry(
                entry_id=new_entry_id(),
                project_id=project.project_id,
                client=client,
                source_kind=source_kind,
                title=title,
                summary=summary,
                body=body,
                tags_json=tags,
                metadata_json=metadata,
                content_hash=body_hash,
            )
            session.add(entry)
            session.flush()
            session.refresh(entry)
            fts.ensure_fts(session)
            fts.index_entry(
                session,
                entry_id=entry.entry_id,
                owner_user_id=owner_user_id,
                project_slug=project.slug,
                source_kind=source_kind,
                title=title,
                summary=summary,
                body=body,
                tags=tags,
            )
            session.expunge(entry)
            session.expunge(project)
            return entry, project, False

    def create_doc_entries_bulk(
        self, *, project_slug: str, owner_user_id: str | None, client: str,
        documents: Sequence[dict[str, Any]],
    ) -> tuple[int, str, list[dict[str, Any]]]:
        """Create changed document facts/chunks in one SQLite+FTS transaction.

        The caller supplies already-chunked document *templates*.  IDs cannot be
        allocated until this transaction creates each JournalEntry, so templates are
        rebound here to that entry's ID before either SQLite or Qdrant sees them.
        This is essential for cross-scope fan-out: copies of the same physical file
        may reuse vectors but must never share a JournalChunk/point identity.

        Equal title+body hashes are returned as deduped rows without changing SQL;
        new revisions are inserted in one transaction and become live only after
        their vectors have been written.
        """
        results: list[dict[str, Any]] = []
        with self.session() as session:
            project = self.get_or_create_project(
                session, slug=project_slug, owner_user_id=owner_user_id)
            fts.ensure_fts(session)
            for document in documents:
                title, body = document["title"], document["body"]
                body_hash = content_hash(body)
                existing = (session.query(JournalEntry)
                    .filter(JournalEntry.project_id == project.project_id,
                            JournalEntry.source_kind == "doc",
                            JournalEntry.title == title,
                            JournalEntry.content_hash == body_hash,
                            JournalEntry.deleted_at.is_(None))
                    .order_by(JournalEntry.created_at.desc()).first())
                if existing is not None:
                    # A previous Qdrant outage leaves this exact SQL/FTS revision
                    # pending (``indexed_at is NULL``) so keyword/vector search
                    # keeps serving the older indexed revision.  The same snapshot
                    # must *retry* that pending entry after Qdrant recovers; treating
                    # it as hash-deduped would make the MCP job claim success without
                    # ever writing a vector or promoting latest-wins.
                    if existing.indexed_at is None:
                        retry_chunks = [replace(
                            chunk,
                            entry_id=existing.entry_id,
                            chunk_id=new_chunk_id(
                                existing.entry_id, chunk.chunk_index,
                                chunk.level, chunk.content),
                        ) for chunk in document["chunks"]]
                        results.append({"entry_id": existing.entry_id, "title": title,
                                        "project_id": project.project_id, "deduped": False,
                                        "chunks": retry_chunks,
                                        "template_key": document.get("template_key")})
                        continue
                    results.append({"entry_id": existing.entry_id, "title": title,
                                    "project_id": project.project_id, "deduped": True,
                                    "chunks": document["chunks"],
                                    "template_key": document.get("template_key")})
                    continue
                entry = JournalEntry(
                    entry_id=new_entry_id(), project_id=project.project_id,
                    client=client, source_kind="doc", title=title, body=body,
                    tags_json=[], metadata_json={}, content_hash=body_hash,
                )
                session.add(entry)
                session.flush()
                fts.index_entry(session, entry_id=entry.entry_id,
                    owner_user_id=owner_user_id, project_slug=project.slug,
                    source_kind="doc", title=title, summary=None, body=body, tags=[])
                # The caller's templates deliberately have a semantic/shared
                # identity.  Persisting those IDs would make two owner scopes point
                # at the same chunk and Qdrant point.  Bind the final entry ID here,
                # while preserving content/order so its primary vectors are reusable.
                chunks = [replace(
                    chunk,
                    entry_id=entry.entry_id,
                    chunk_id=new_chunk_id(
                        entry.entry_id, chunk.chunk_index, chunk.level, chunk.content),
                ) for chunk in document["chunks"]]
                for chunk in chunks:
                    session.add(JournalChunk(
                        chunk_id=chunk.chunk_id, entry_id=entry.entry_id,
                        project_id=project.project_id, chunk_index=chunk.chunk_index,
                        level=chunk.level, heading=chunk.heading, content=chunk.content,
                        token_count=chunk.token_count, metadata_json=chunk.metadata or {},
                    ))
                results.append({"entry_id": entry.entry_id, "title": title,
                                "project_id": project.project_id, "deduped": False,
                                "chunks": chunks,
                                "template_key": document.get("template_key")})
            session.flush()
            return project.project_id, project.slug, results

    def mark_index_results_bulk(self, entry_ids: Sequence[str], *, success: bool,
                                error: str | None = None) -> None:
        if not entry_ids:
            return
        with self.session() as session:
            session.query(JournalEntry).filter(JournalEntry.entry_id.in_(list(entry_ids))).update(
                {JournalEntry.indexed_at: _utcnow() if success else None,
                 JournalEntry.index_error: None if success else (error or "index failed")},
                synchronize_session=False)

    def soft_delete_entries_bulk(self, entry_ids: Sequence[str], *, reason: str,
                                  owner_user_id: str | None = None) -> list[str]:
        if not entry_ids:
            return []
        with self.session() as session:
            query = session.query(JournalEntry).join(JournalProject).filter(
                JournalEntry.entry_id.in_(list(entry_ids)), JournalEntry.deleted_at.is_(None))
            if owner_user_id is None:
                query = query.filter(JournalProject.owner_user_id.is_(None))
            else:
                query = query.filter(JournalProject.owner_user_id == owner_user_id)
            ids = [row[0] for row in query.with_entities(JournalEntry.entry_id).all()]
            if ids:
                now = _utcnow()
                session.query(JournalEntry).filter(JournalEntry.entry_id.in_(ids)).update(
                    {JournalEntry.deleted_at: now, JournalEntry.deleted_reason: reason},
                             synchronize_session=False)
                for entry_id in ids:
                    fts.delete_entry(session, entry_id)
            return ids

    def superseded_version_ids(
        self,
        *,
        project_slug: str | Sequence[str],
        source_kind: str,
        title: str,
        owner_user_id: str | None,
        exclude_entry_id: str,
        case_sensitive: bool = True,
    ) -> list[str]:
        """Live entry_ids for the same logical file, excluding the current one.

        A "logical file" is (owner-scoped project, source_kind, title). Used by the
        latest-wins doc re-index (W153): an edited file gets a new content_hash so it
        can't hash-dedup; its prior versions must be superseded so search returns only
        the current content, not stale accumulated copies.
        """
        with self.session() as session:
            query = (
                session.query(JournalEntry.entry_id, JournalEntry.title)
                .join(JournalProject, JournalEntry.project_id == JournalProject.project_id)
                .filter(
                    JournalEntry.source_kind == source_kind,
                    JournalEntry.deleted_at.is_(None),
                    JournalEntry.entry_id != exclude_entry_id,
                )
            )
            if isinstance(project_slug, str):
                query = query.filter(JournalProject.slug == project_slug)
            else:
                query = query.filter(JournalProject.slug.in_(list(project_slug)))
            if owner_user_id is None:
                query = query.filter(JournalProject.owner_user_id.is_(None))
            else:
                query = query.filter(JournalProject.owner_user_id == owner_user_id)
            target_identity = stored_relative_path_identity(
                title, case_sensitive=case_sensitive)
            return [
                entry_id for entry_id, candidate_title in query.all()
                if target_identity is not None
                and stored_relative_path_identity(
                    candidate_title or "", case_sensitive=case_sensitive,
                ) == target_identity
            ]

    def _scoped_live_docs(self, session: Session, *, owner_user_id: str | None,
                          project_slug: str | Sequence[str] | None = None):
        """Base query over LIVE source_kind="doc" entries for this owner, optionally
        narrowed to one project — the shared join+filter behind live_doc_titles /
        all_doc_entry_ids / stale_doc_entry_ids_for_snapshot / all_live_doc_files
        (project_slug=None → every scope, for the cross-scope fan-out). Callers pick
        columns (.with_entities), ordering, and any extra filters."""
        query = (
            session.query(JournalEntry)
            .join(JournalProject, JournalEntry.project_id == JournalProject.project_id)
            .filter(
                JournalEntry.source_kind == "doc",
                JournalEntry.deleted_at.is_(None),
            )
        )
        if project_slug is not None:
            if isinstance(project_slug, str):
                query = query.filter(JournalProject.slug == project_slug)
            else:
                query = query.filter(JournalProject.slug.in_(list(project_slug)))
        if owner_user_id is None:
            return query.filter(JournalProject.owner_user_id.is_(None))
        return query.filter(JournalProject.owner_user_id == owner_user_id)

    def live_doc_titles(
        self,
        *,
        project_slug: str | Sequence[str],
        owner_user_id: str | None,
    ) -> list[str]:
        with self.session() as session:
            query = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .filter(JournalEntry.title.isnot(None))
                .with_entities(JournalEntry.title)
                .order_by(JournalEntry.title)
            )
            return [row[0] for row in query.all()]

    def all_doc_entry_ids(
        self, *, project_slug: str | Sequence[str], owner_user_id: str | None,
    ) -> list[str]:
        """Every live source_kind="doc" entry_id in the scope — the target set for an
        explicit doc-scope clear (delete_journal_docs), distinct from the snapshot sweep."""
        with self.session() as session:
            query = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .with_entities(JournalEntry.entry_id)
                .order_by(JournalEntry.created_at)
            )
            return [row[0] for row in query.all()]

    def scoped_live_doc_id_titles(
        self, *, project_slug: str | Sequence[str], owner_user_id: str | None,
    ) -> list[tuple[str, str]]:
        """(entry_id, title) for every LIVE titled doc in ONE scope — the candidate set
        journal_search's path_glob matches against (title == the doc's scope-relative
        path). The matching entry_ids are pushed down as a search filter so top_k is
        filled only from matching docs, never post-filter-squeezed (no dropped top_k)."""
        with self.session() as session:
            rows = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .filter(JournalEntry.title.isnot(None))
                .with_entities(JournalEntry.entry_id, JournalEntry.title)
                .all()
            )
            return [(row[0], row[1]) for row in rows]

    def all_live_doc_files(self, *, owner_user_id: str | None) -> list[tuple[str, str]]:
        """Every (project_slug, title) for this owner's LIVE, titled source_kind="doc"
        entries across ALL scopes — the cross-scope fan-out (方案2) matches these by
        absolute path (project_slug/title) to find the same physical file indexed under
        another scope and refresh it."""
        with self.session() as session:
            query = (
                self._scoped_live_docs(session, owner_user_id=owner_user_id)
                .filter(JournalEntry.title.isnot(None))
                .with_entities(JournalProject.slug, JournalEntry.title)
            )
            return [(slug, title) for slug, title in query.all()]

    def stale_doc_entry_ids_for_snapshot(
        self,
        *,
        project_slug: str | Sequence[str],
        live_titles: list[str],
        owner_user_id: str | None,
        max_depth: int | None = None,
        patterns: list[str] | None = None,
        case_sensitive: bool = True,
    ) -> list[str]:
        with self.session() as session:
            query = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .with_entities(JournalEntry.entry_id, JournalEntry.title)
            )
            rows = query.order_by(JournalEntry.created_at).all()
        live_identities = {
            identity for title in live_titles
            if (identity := stored_relative_path_identity(
                title, case_sensitive=case_sensitive)) is not None
        }

        def _stale_candidate(title: str | None) -> bool:
            if title is None:
                return True
            identity = stored_relative_path_identity(
                title, case_sensitive=case_sensitive)
            if identity is None:
                return False  # malformed historical data cannot be proven stale safely
            if max_depth is not None and identity.count("/") > max_depth:
                return False
            return identity not in live_identities

        rows = [
            (entry_id, title) for entry_id, title in rows if _stale_candidate(title)
        ]
        globs = [g for g in (patterns or []) if g]
        if globs and globs != ["*"]:
            # Only sweep titles WITHIN the scan's PATTERN envelope: a "*.txt" re-run must not
            # reconcile the ".md" docs it never globbed for (same class as the depth valve —
            # W169b data-loss bug). Match the basename case-sensitively (fnmatchcase, backend-
            # OS-independent) so any casing mismatch errs toward KEEPING the doc, never sweeping
            # a file outside the glob. "*" (or no patterns) = full reconcile, unchanged.
            def _in_pattern(title: str | None) -> bool:
                if not title:
                    return False  # untitled doc: can't classify → keep, never sweep
                identity = stored_relative_path_identity(
                    title, case_sensitive=case_sensitive)
                if identity is None:
                    return False
                base = identity.rsplit("/", 1)[-1]
                effective_globs = globs if case_sensitive else [g.casefold() for g in globs]
                return any(fnmatch.fnmatchcase(base, glob) for glob in effective_globs)

            return [entry_id for entry_id, title in rows if _in_pattern(title)]
        return [entry_id for entry_id, _ in rows]

    def replace_chunks(self, entry_id: str, chunks: list[dict[str, Any]]) -> list[JournalChunk]:
        with self.session() as session:
            entry = session.query(JournalEntry).filter(JournalEntry.entry_id == entry_id).one()
            session.query(JournalChunk).filter(JournalChunk.entry_id == entry_id).delete()
            rows: list[JournalChunk] = []
            for item in chunks:
                row = JournalChunk(
                    chunk_id=item["chunk_id"],
                    entry_id=entry.entry_id,
                    project_id=entry.project_id,
                    chunk_index=item["chunk_index"],
                    level=item["level"],
                    heading=item.get("heading"),
                    content=item["content"],
                    token_count=item.get("token_count"),
                    metadata_json=item.get("metadata") or {},
                )
                session.add(row)
                rows.append(row)
            session.flush()
            for row in rows:
                session.refresh(row)
                session.expunge(row)
            return sorted(rows, key=lambda r: r.chunk_index)

    def mark_index_result(self, entry_id: str, *, success: bool, error: str | None = None) -> None:
        with self.session() as session:
            entry = session.query(JournalEntry).filter(JournalEntry.entry_id == entry_id).one()
            entry.indexed_at = _utcnow() if success else None
            entry.index_error = None if success else (error or "index failed")

    def get_entry(
        self,
        entry_id: str,
        *,
        include_deleted: bool = False,
        owner_user_id: str | None = None,
    ) -> JournalEntry | None:
        with self.session() as session:
            query = (
                session.query(JournalEntry)
                .options(selectinload(JournalEntry.chunks), selectinload(JournalEntry.project))
                .filter(JournalEntry.entry_id == entry_id)
            )
            if owner_user_id is not None:
                query = query.join(JournalProject).filter(JournalProject.owner_user_id == owner_user_id)
            if not include_deleted:
                query = query.filter(JournalEntry.deleted_at.is_(None))
            entry = query.one_or_none()
            return self._detach_entry(session, entry)

    def get_entries_by_ids(
        self,
        entry_ids: list[str],
        *,
        owner_user_id: str | None = None,
    ) -> dict[str, JournalEntry]:
        if not entry_ids:
            return {}
        order = {entry_id: i for i, entry_id in enumerate(entry_ids)}
        with self.session() as session:
            query = (
                session.query(JournalEntry)
                .options(selectinload(JournalEntry.chunks), selectinload(JournalEntry.project))
                .filter(JournalEntry.entry_id.in_(entry_ids), JournalEntry.deleted_at.is_(None))
            )
            if owner_user_id is not None:
                query = query.join(JournalProject).filter(JournalProject.owner_user_id == owner_user_id)
            rows = query.all()
            detached = [self._detach_entry(session, row) for row in rows]
            return dict(sorted(
                ((entry.entry_id, entry) for entry in detached if entry is not None),
                key=lambda kv: order.get(kv[0], 10**9),
            ))

    def list_pending_index(self, *, limit: int = 100) -> list[JournalEntry]:
        """Live entries whose vectors are missing (indexed_at NULL) — the reindex queue.

        These were recorded while the embed endpoint was down (SQL fact persisted, index
        failed). Ordered oldest-first so a reconcile drains the backlog in arrival order.
        """
        with self.session() as session:
            rows = (
                session.query(JournalEntry)
                .options(selectinload(JournalEntry.chunks), selectinload(JournalEntry.project))
                .filter(JournalEntry.deleted_at.is_(None), JournalEntry.indexed_at.is_(None))
                .order_by(JournalEntry.created_at.asc())
                .limit(limit)
                .all()
            )
            return [e for e in (self._detach_entry(session, r) for r in rows) if e is not None]

    def list_entries(
        self,
        *,
        project_slug: str | Sequence[str] | None = None,
        source_kind: str | None = None,
        client: str | None = None,
        tag: str | None = None,
        q: str | None = None,
        owner_user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JournalEntry], int]:
        with self.session() as session:
            query = (
                session.query(JournalEntry)
                .join(JournalProject, JournalProject.project_id == JournalEntry.project_id)
                .options(selectinload(JournalEntry.chunks), selectinload(JournalEntry.project))
                .filter(JournalEntry.deleted_at.is_(None))
            )
            if project_slug is not None:
                if isinstance(project_slug, str):
                    query = query.filter(JournalProject.slug == project_slug)
                else:
                    query = query.filter(JournalProject.slug.in_(list(project_slug)))
            if owner_user_id is not None:
                query = query.filter(JournalProject.owner_user_id == owner_user_id)
            if source_kind:
                query = query.filter(JournalEntry.source_kind == source_kind)
            if client:
                query = query.filter(JournalEntry.client == client)
            if tag:
                query = query.filter(JournalEntry.tags_json.like(f"%{tag}%"))
            if q:
                like = f"%{q}%"
                query = query.filter(
                    (JournalEntry.title.like(like))
                    | (JournalEntry.summary.like(like))
                    | (JournalEntry.body.like(like))
                )
            total = query.with_entities(func.count(JournalEntry.id)).scalar() or 0
            rows = (
                query.order_by(JournalEntry.created_at.desc(), JournalEntry.id.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [e for e in (self._detach_entry(session, row) for row in rows) if e], total

    def keyword_search(
        self,
        *,
        query: str,
        project_slug: str | Sequence[str] | None = None,
        source_kind: str | None = None,
        owner_user_id: str | None = None,
        entry_ids: list[str] | None = None,
        limit: int = 20,
        load_chunks: bool = True,
    ) -> tuple[list[tuple[JournalEntry, float]], int]:
        """Embedding-free FTS5 keyword search. Returns `[(entry, score)]` best-first.
        `entry_ids` (path_glob resolution) restricts the match to that set inside the FTS
        query so top_k is never post-filter-squeezed."""
        with self.session() as session:
            fts.ensure_fts(session)
            matches = fts.search(
                session,
                query,
                owner_user_id=owner_user_id,
                project_slug=project_slug,
                source_kind=source_kind,
                entry_ids=entry_ids,
                limit=limit,
            )
            if not matches:
                return [], 0
            scores = dict(matches)
            entry_ids = [entry_id for entry_id, _ in matches]
            options = [selectinload(JournalEntry.project)]
            if load_chunks:
                options.append(selectinload(JournalEntry.chunks))
            entries = (
                session.query(JournalEntry)
                .options(*options)
                .filter(JournalEntry.entry_id.in_(entry_ids), JournalEntry.deleted_at.is_(None),
                        or_(JournalEntry.source_kind != "doc", JournalEntry.indexed_at.isnot(None),
                            JournalEntry.index_error.is_(None)))
                .all()
            )
            chunk_counts = self._chunk_counts(session, entry_ids) if not load_chunks else {}
            by_id = {}
            for entry in entries:
                detached = self._detach_entry(session, entry, include_chunks=load_chunks)
                if detached is not None and not load_chunks:
                    setattr(detached, "_chunk_count", chunk_counts.get(detached.entry_id, 0))
                by_id[entry.entry_id] = detached
            ordered = [
                (by_id[entry_id], scores[entry_id])
                for entry_id in entry_ids
                if by_id.get(entry_id) is not None
            ]
            return ordered, len(ordered)

    def _ranked_chunks_by_entry_ids(
        self,
        entry_ids: list[str],
        *,
        limit: int,
        order_by,
        owner_user_id: str | None = None,
        extra_filter=None,
    ) -> dict[str, list[JournalChunk]]:
        """Return at most `limit` chunks per live entry, ranked by `order_by`.

        Shared bounded-hydration path for MCP search output: callers render only a few
        concrete snippets, so this never loads entire large docs. `extra_filter` narrows
        the candidate chunks (e.g. keyword hits); `order_by` decides which win per entry.
        """
        if not entry_ids or limit <= 0:
            return {}
        with self.session() as session:
            chunk_rank = func.row_number().over(
                partition_by=JournalChunk.entry_id,
                order_by=order_by,
            ).label("chunk_rank")
            ranked_query = (
                session.query(JournalChunk.id.label("chunk_row_id"), chunk_rank)
                .join(JournalEntry, JournalEntry.entry_id == JournalChunk.entry_id)
                .filter(
                    JournalChunk.entry_id.in_(entry_ids),
                    JournalEntry.deleted_at.is_(None),
                )
            )
            if extra_filter is not None:
                ranked_query = ranked_query.filter(extra_filter)
            if owner_user_id is not None:
                ranked_query = ranked_query.join(
                    JournalProject,
                    JournalProject.project_id == JournalEntry.project_id,
                ).filter(JournalProject.owner_user_id == owner_user_id)
            ranked = ranked_query.subquery()
            rows = (
                session.query(JournalChunk)
                .join(ranked, JournalChunk.id == ranked.c.chunk_row_id)
                .filter(ranked.c.chunk_rank <= limit)
                .order_by(JournalChunk.entry_id, ranked.c.chunk_rank)
                .all()
            )
            by_entry: dict[str, list[JournalChunk]] = {entry_id: [] for entry_id in entry_ids}
            for row in rows:
                self._safe_expunge(session, row)
                by_entry.setdefault(row.entry_id, []).append(row)
            return by_entry

    def limited_chunks_by_entry_ids(
        self,
        entry_ids: list[str],
        *,
        limit: int,
        owner_user_id: str | None = None,
    ) -> dict[str, list[JournalChunk]]:
        """Return at most `limit` chunks per entry, preferring body chunks over summary."""
        return self._ranked_chunks_by_entry_ids(
            entry_ids,
            limit=limit,
            owner_user_id=owner_user_id,
            order_by=(
                case((JournalChunk.chunk_index == 0, 1), else_=0),
                JournalChunk.chunk_index,
            ),
        )

    def matching_chunks_by_entry_ids(
        self,
        entry_ids: list[str],
        *,
        query: str,
        limit: int,
        owner_user_id: str | None = None,
    ) -> dict[str, list[JournalChunk]]:
        """Return bounded chunks whose text matches the same tokens as FTS search.

        `journal_fts` is entry-level for resilience and simple reindexing. MCP output,
        however, needs concrete paragraphs. This keeps that payload bounded by selecting
        only chunks that contain at least one sanitized keyword token, ranked by token-hit
        count and then by document order.
        """
        tokens = fts.query_tokens(query)
        if not entry_ids or limit <= 0 or not tokens:
            return {}

        score_expr = None
        for token in tokens:
            pattern = fts.like_token_pattern(token)
            token_hit = case(
                (
                    or_(
                        func.lower(JournalChunk.content).like(pattern, escape="\\"),
                        func.lower(JournalChunk.heading).like(pattern, escape="\\"),
                    ),
                    1,
                ),
                else_=0,
            )
            score_expr = token_hit if score_expr is None else score_expr + token_hit

        return self._ranked_chunks_by_entry_ids(
            entry_ids,
            limit=limit,
            owner_user_id=owner_user_id,
            order_by=(
                score_expr.desc(),
                case((JournalChunk.chunk_index == 0, 1), else_=0),
                JournalChunk.chunk_index,
            ),
            extra_filter=score_expr > 0,
        )

    def _chunk_counts(self, session: Session, entry_ids: list[str]) -> dict[str, int]:
        if not entry_ids:
            return {}
        rows = (
            session.query(JournalChunk.entry_id, func.count(JournalChunk.id))
            .filter(JournalChunk.entry_id.in_(entry_ids))
            .group_by(JournalChunk.entry_id)
            .all()
        )
        return {entry_id: int(count or 0) for entry_id, count in rows}

    def stats(self) -> dict[str, int]:
        """Aggregate counts for the admin overview (live entries only)."""
        with self.session() as session:
            entries, indexed = (
                session.query(
                    func.count(JournalEntry.id),
                    func.count(JournalEntry.indexed_at),
                )
                .filter(JournalEntry.deleted_at.is_(None))
                .one()
            )
            projects, owners = (
                session.query(
                    func.count(JournalProject.project_id),
                    func.count(func.distinct(JournalProject.owner_user_id)),
                )
                .one()
            )
            return {
                "entries": int(entries or 0),
                "indexed": int(indexed or 0),
                "projects": int(projects or 0),
                "owners": int(owners or 0),
            }

    def list_owner_ids(self) -> list[str]:
        """Distinct non-null owner ids across journal projects."""
        with self.session() as session:
            rows = (
                session.query(JournalProject.owner_user_id)
                .filter(JournalProject.owner_user_id.isnot(None))
                .distinct()
                .all()
            )
            return [r[0] for r in rows]

    def reindex_fts(self) -> int:
        """Backfill the FTS5 table from live (non-deleted) entries. Idempotent."""
        with self.session() as session:
            fts.ensure_fts(session)
            session.execute(text(f"DELETE FROM {fts.TABLE}"))
            rows = (
                session.query(JournalEntry)
                .join(JournalProject, JournalProject.project_id == JournalEntry.project_id)
                .filter(JournalEntry.deleted_at.is_(None))
                .all()
            )
            count = 0
            for entry in rows:
                fts.index_entry(
                    session,
                    entry_id=entry.entry_id,
                    owner_user_id=entry.project.owner_user_id if entry.project else None,
                    project_slug=entry.project.slug if entry.project else "",
                    source_kind=entry.source_kind,
                    title=entry.title,
                    summary=entry.summary,
                    body=entry.body,
                    tags=entry.tags_json or [],
                )
                count += 1
            return count

    def soft_delete_entry(
        self,
        entry_id: str,
        reason: str | None = None,
        owner_user_id: str | None = None,
    ) -> bool:
        with self.session() as session:
            query = session.query(JournalEntry).filter(JournalEntry.entry_id == entry_id)
            if owner_user_id is not None:
                query = query.join(JournalProject).filter(JournalProject.owner_user_id == owner_user_id)
            entry = query.one_or_none()
            if entry is None or entry.deleted_at is not None:
                return False
            entry.deleted_at = _utcnow()
            entry.deleted_reason = reason
            fts.ensure_fts(session)
            fts.delete_entry(session, entry_id)
            return True

    @staticmethod
    def _detach_entry(
        session: Session,
        entry: JournalEntry | None,
        *,
        include_chunks: bool = True,
    ) -> JournalEntry | None:
        if entry is None:
            return None
        # Force load before expunge.
        if include_chunks:
            list(entry.chunks)
            setattr(entry, "_chunk_count", len(entry.chunks))
        _ = entry.project
        if include_chunks:
            for chunk in entry.chunks:
                JournalRepository._safe_expunge(session, chunk)
        JournalRepository._safe_expunge(session, entry.project)
        JournalRepository._safe_expunge(session, entry)
        return entry

    @staticmethod
    def _safe_expunge(session: Session, obj: Any | None) -> None:
        if obj is not None and inspect(obj).session is session:
            session.expunge(obj)
