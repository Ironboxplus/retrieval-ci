"""SQLAlchemy repository for ACE journal facts."""
from __future__ import annotations

import fnmatch
import hashlib
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import case, func, inspect, or_, select, text
from sqlalchemy.orm import Session, defer, selectinload

from app.services.journal import fts
from app.services.scope_paths import (
    PathReach,
    canonicalize_scope_path,
    directory_reach,
    glob_reach,
    has_dot_directory_segment,
    stored_relative_path_identity,
)
from core.db.database import get_db_session
from core.db.write_batches import paced_batches
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


def title_basename_key(title: str | None) -> str:
    """Last path segment of a stored relative path, casefolded.

    Two titles with the same path identity (or the same physical file under related
    scopes) always share this key, whatever the slash style, leading slash or case rules
    of their scope - ``//`` and ``.`` segments are skipped exactly as path normalisation
    skips them.  It is a lossless prefilter in front of the exact (and ~10x dearer)
    identity / abspath comparison, never the comparison itself.
    """
    for segment in reversed((title or "").replace("\\", "/").split("/")):
        if segment not in ("", "."):
            return segment.casefold()
    return ""


@lru_cache(maxsize=65536)
def path_identity(path: str, case_sensitive: bool) -> str | None:
    """Memoised ``stored_relative_path_identity``: a scope sweep needs each stored title's
    identity several times (existing set, stale test, snapshot match) and the pathlib-based
    normalisation dominates a large scope's cost."""
    return stored_relative_path_identity(path, case_sensitive=case_sensitive)


_BASENAME_IN_BATCH = 500  # well under SQLite's bound-variable cap


def basename_candidates_filter(keys: Sequence[str]):
    """SQL superset of the titles whose ``title_basename_key`` is in ``keys``.

    Evaluated inside SQLite (a trailing-segment expression + ``IN``), so a lookup for a few
    files transfers and normalises only the rows that can match instead of every live doc of
    the scope.  Never narrower than the Python key: legacy backslashes are normalised like
    ``stored_relative_path_identity``; titles ending in ``/`` or a ``.`` segment (their last
    segment is not the SQL one), and - because SQLite lower() is ASCII-only while the key is
    casefolded - titles with non-ASCII text, always stay candidates for the exact check."""
    normalised = func.replace(JournalEntry.title, "\\", "/")
    last_segment = func.substr(
        normalised, func.length(func.rtrim(normalised, func.replace(normalised, "/", ""))) + 1)
    wanted = sorted(set(keys))
    return or_(
        *[func.lower(last_segment).in_(wanted[i:i + _BASENAME_IN_BATCH])
          for i in range(0, len(wanted), _BASENAME_IN_BATCH)],
        normalised.like("%/"), normalised.like("%/."), normalised.in_((".", "")),
        JournalEntry.title.op("GLOB")("*[^" + chr(1) + "-" + chr(127) + "]*"),
    )


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def path_reach_title_prefilter(reach: PathReach | None, *, case_sensitive: bool):
    """SQL superset of the titles a path reach can accept, or None for "unbounded".

    The exact gitignore semantics stay in Python (``path_matches``); this only keeps a
    per-search ``path_glob`` or directory prefix from loading every doc title of the
    scope. It must never exclude a title Python would accept, so it errs wide: LIKE is
    ASCII-case-insensitive (wider than either case rule), legacy spellings (backslashes,
    one leading slash) are normalised the way ``stored_relative_path_identity`` does, and
    titles whose path normalisation could change their segments (``//``, ``.`` segments)
    or - for case-insensitive scopes - that hold non-ASCII text (casefold != LIKE folding)
    always stay candidates. The anchors themselves come from ``scope_paths.glob_reach`` /
    ``directory_reach``, the same ones the catalog's range queries use.
    """
    if reach is None:
        return None
    title = func.replace(JournalEntry.title, "\\", "/")

    def like(expression: str):
        return title.like(expression, escape="\\")

    def rooted(anchor: str) -> tuple[str, str]:
        return anchor, "/" + anchor  # one stray leading slash is a legacy spelling

    terms = []
    for prefix in reach.prefixes:
        terms += [like(_like_escape(variant) + "%") for variant in rooted(prefix)]
    for entry in reach.entries:
        for variant in rooted(entry):
            terms += [like(_like_escape(variant)), like(_like_escape(variant) + "/%")]
    for directory in reach.directories:
        terms += [like(_like_escape(variant) + "/%") for variant in rooted(directory)]
    for name in reach.any_depth:
        terms += [like("%/" + _like_escape(name)), like("%/" + _like_escape(name) + "/%")]
    odd = [like("%//%"), like("./%"), like("%/./%"), like("%/."), title == "."]
    if not case_sensitive:
        odd.append(JournalEntry.title.op("GLOB")("*[^\x01-\x7f]*"))
    return or_(*terms, *odd)


def path_glob_title_prefilter(pattern: str, *, case_sensitive: bool):
    """SQL superset of the titles ``path_matches(title, pattern)`` can accept, or None."""
    return path_reach_title_prefilter(
        glob_reach(pattern, case_sensitive=case_sensitive), case_sensitive=case_sensitive)


# Chunk rows one bulk-create transaction writes before it commits. SQLite has one write lock per
# file and a scope's insert costs ~0.3 ms per chunk (ORM rows + FTS), so one transaction for a
# 79,264-chunk scope held the lock ~27 s of the 30 s busy_timeout and made every other writer fail.
BULK_CREATE_BATCH_CHUNKS = 2000


class JournalRepository:
    """Repository boundary for journal SQL tables."""

    def __init__(self, session_factory: Callable[[], Any] = get_db_session):
        self.session_factory = session_factory
        self._fts_aligned = False
        self._fts_align_lock = threading.Lock()

    @contextmanager
    def session(self) -> Iterator[Session]:
        # Every FTS write goes through here, and the alignment runs in its own
        # session BEFORE the caller's opens: the rebuild needs the SQLite write lock,
        # so running it inside a caller that already wrote would block on itself.
        self.ensure_fts_aligned()
        with self.session_factory() as session:
            yield session

    def ensure_fts_aligned(self) -> None:
        """Once per repository, rebuild `journal_fts` if its rowids are not entry ids.

        Failure must propagate: writing by rowid into a misaligned table would delete
        another entry's row. The flag is only set after the check or rebuild succeeds.
        """
        if self._fts_aligned:
            return
        with self._fts_align_lock:
            if self._fts_aligned:
                return
            started = time.monotonic()
            with self.session_factory() as session:
                fts.ensure_fts(session)
                aligned = fts.is_aligned(session)
            if aligned:
                fts.logger.info("journal_fts rowids aligned with entry ids; no rebuild "
                                "(check %.2fs)", time.monotonic() - started)
            else:
                rebuilt = self.reindex_fts()
                fts.logger.info("journal_fts realigned to entry ids: rebuilt %d live "
                                "entries in %.2fs", rebuilt, time.monotonic() - started)
            self._fts_aligned = True

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
                rowid=entry.id,
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
        """Create changed document facts/chunks, a SQLite+FTS transaction per batch of documents.

        The caller supplies already-chunked document *templates*.  IDs cannot be
        allocated until a transaction creates each JournalEntry, so templates are
        rebound here to that entry's ID before either SQLite or Qdrant sees them.
        This is essential for cross-scope fan-out: copies of the same physical file
        may reuse vectors but must never share a JournalChunk/point identity.

        Equal title+body hashes are returned as deduped rows without changing SQL;
        new revisions become live only after their vectors have been written, so
        committing between batches (``BULK_CREATE_BATCH_CHUNKS``) leaves nothing a
        reader sees half-done, and an interrupted run is finished by the next one
        (the pending entries are retried, the written ones deduped).  One transaction
        for the whole scope kept SQLite's single write lock for tens of seconds.
        """
        results: list[dict[str, Any]] = []
        project_id, stored_scope = 0, project_slug
        for batch in paced_batches(
                documents, BULK_CREATE_BATCH_CHUNKS, weight=lambda document: len(document["chunks"]) or 1):
            project_id, stored_scope, rows = self._create_doc_entries_batch(
                project_slug=project_slug, owner_user_id=owner_user_id, client=client,
                documents=batch)
            results.extend(rows)
        return project_id, stored_scope, results

    def _create_doc_entries_batch(
        self, *, project_slug: str, owner_user_id: str | None, client: str,
        documents: Sequence[dict[str, Any]],
    ) -> tuple[int, str, list[dict[str, Any]]]:
        """One transaction of ``create_doc_entries_bulk``."""
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
                fts.index_entry(session, rowid=entry.id, entry_id=entry.entry_id,
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
            rows = query.with_entities(JournalEntry.entry_id, JournalEntry.id).all()
            ids = [row[0] for row in rows]
            if ids:
                now = _utcnow()
                session.query(JournalEntry).filter(JournalEntry.entry_id.in_(ids)).update(
                    {JournalEntry.deleted_at: now, JournalEntry.deleted_reason: reason},
                             synchronize_session=False)
                fts.delete_entries(session, [row[1] for row in rows])
            return ids

    def live_doc_ids_by_identity(
        self,
        *,
        project_slug: str | Sequence[str],
        source_kind: str,
        owner_user_id: str | None,
        titles: Sequence[str],
        case_sensitive: bool = True,
    ) -> dict[str, list[str]]:
        """Live entry_ids of the given files' logical identities, keyed by identity.

        A "logical file" is (owner-scoped project, source_kind, title identity).  One read
        of the scope serves any number of titles, so a bulk latest-wins sweep costs a
        single query instead of one scope read per changed file; the cheap basename bucket
        keeps the exact identity comparison off every unrelated row.
        """
        wanted = {
            identity for title in titles
            if (identity := path_identity(title, case_sensitive)) is not None
        }
        if not wanted:
            return {}
        buckets = {title_basename_key(identity) for identity in wanted}
        with self.session() as session:
            query = (
                session.query(JournalEntry.entry_id, JournalEntry.title)
                .join(JournalProject, JournalEntry.project_id == JournalProject.project_id)
                .filter(
                    JournalEntry.source_kind == source_kind,
                    JournalEntry.deleted_at.is_(None),
                    basename_candidates_filter(list(buckets)),
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
            found: dict[str, list[str]] = defaultdict(list)
            for entry_id, candidate_title in query.all():
                if title_basename_key(candidate_title) not in buckets:
                    continue
                identity = path_identity(candidate_title or "", case_sensitive)
                if identity in wanted:
                    found[identity].append(entry_id)
            return dict(found)

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

        Used by the latest-wins doc re-index (W153): an edited file gets a new
        content_hash so it can't hash-dedup; its prior versions must be superseded so
        search returns only the current content, not stale accumulated copies.
        """
        identity = path_identity(title, case_sensitive)
        found = self.live_doc_ids_by_identity(
            project_slug=project_slug, source_kind=source_kind, owner_user_id=owner_user_id,
            titles=[title], case_sensitive=case_sensitive)
        return [entry_id for entry_id in found.get(identity, []) if entry_id != exclude_entry_id]

    def _scoped_live_docs(self, session: Session, *, owner_user_id: str | None,
                          project_slug: str | Sequence[str] | None = None):
        """Base query over LIVE source_kind="doc" entries for this owner, optionally
        narrowed to one project — the shared join+filter behind doc_inventory /
        all_doc_entry_ids / scoped_live_doc_id_titles / live_doc_files_in_scopes
        (project_slug=None → every scope). Callers pick columns (.with_entities),
        ordering, and any extra filters."""
        query = (
            session.query(JournalEntry)
            .join(JournalProject, JournalEntry.project_id == JournalProject.project_id)
            .filter(
                JournalEntry.source_kind == "doc",
                JournalEntry.deleted_at.is_(None),
            )
        )
        owner = (JournalProject.owner_user_id.is_(None) if owner_user_id is None
                 else JournalProject.owner_user_id == owner_user_id)
        if project_slug is not None:
            slugs = [project_slug] if isinstance(project_slug, str) else list(project_slug)
            # The scope's project ids as an IN subquery: it pins the entry rows to
            # ix_journal_entries_project_kind. Through the join alone, three or more
            # slugs (a Windows scope's case variants) made the planner, which has no
            # statistics, walk every live doc of every project via source_kind.
            scope_ids = select(JournalProject.project_id).where(
                JournalProject.slug.in_(slugs), owner)
            query = query.filter(JournalEntry.project_id.in_(scope_ids),
                                 JournalProject.slug.in_(slugs))
        return query.filter(owner)

    def doc_inventory(
        self,
        *,
        project_slug: str | Sequence[str],
        owner_user_id: str | None,
    ) -> list[tuple[str, str | None]]:
        """(entry_id, title) of every LIVE doc in the scope, oldest first.

        One read shared by the snapshot sweep's disjointness guard and its stale
        selection (``stale_doc_entry_ids``)."""
        with self.session() as session:
            query = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .with_entities(JournalEntry.entry_id, JournalEntry.title)
                .order_by(JournalEntry.created_at)
            )
            return [(row[0], row[1]) for row in query.all()]

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
        self,
        *,
        project_slug: str | Sequence[str],
        owner_user_id: str | None,
        path_glob: str | None = None,
        case_sensitive: bool = True,
        path_prefix: str | None = None,
    ) -> list[tuple[str, str]]:
        """(entry_id, title) for LIVE titled docs in ONE scope — the candidate set
        journal_search's path_glob matches against (title == the doc's scope-relative
        path). The matching entry_ids are pushed down as a search filter so top_k is
        filled only from matching docs, never post-filter-squeezed (no dropped top_k).

        With ``path_glob`` the result is a SUPERSET of its matches (see
        ``path_glob_title_prefilter``); the caller still applies ``path_matches``. With
        ``path_prefix`` (a scope-relative directory) it is a superset of the docs below
        that directory instead, and ``path_glob`` is not consulted here."""
        with self.session() as session:
            query = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .filter(JournalEntry.title.isnot(None))
            )
            if path_prefix:
                prefilter = path_reach_title_prefilter(
                    directory_reach(path_prefix, case_sensitive=case_sensitive),
                    case_sensitive=case_sensitive)
            elif path_glob:
                prefilter = path_glob_title_prefilter(path_glob, case_sensitive=case_sensitive)
            else:
                prefilter = None
            if prefilter is not None:
                query = query.filter(prefilter)
            rows = query.with_entities(JournalEntry.entry_id, JournalEntry.title).all()
            return [(row[0], row[1]) for row in rows]

    def scope_doc_summary(
        self, *, project_slug: str | Sequence[str], owner_user_id: str | None,
    ) -> tuple[int, str | None]:
        """(count of titled live docs, shortest-then-lexical title) for one scope — what
        explains an empty path_glob match, without listing the scope's docs."""
        with self.session() as session:
            base = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .filter(JournalEntry.title.isnot(None))
            )
            count = base.with_entities(func.count(JournalEntry.id)).scalar() or 0
            sample = (
                base.with_entities(JournalEntry.title)
                .order_by(func.length(JournalEntry.title), JournalEntry.title)
                .limit(1).scalar()
            ) if count else None
            return int(count), sample

    def scope_has_live_entries(
        self, *, project_slug: Sequence[str], owner_user_id: str | None,
    ) -> bool:
        """Whether the scope holds any live entry of any kind (docs, memory, conversation
        segments): one indexed existence probe, no listing."""
        with self.session() as session:
            query = (
                session.query(JournalEntry.id)
                .join(JournalProject, JournalEntry.project_id == JournalProject.project_id)
                .filter(JournalEntry.deleted_at.is_(None),
                        JournalProject.slug.in_(list(project_slug)))
            )
            if owner_user_id is None:
                query = query.filter(JournalProject.owner_user_id.is_(None))
            else:
                query = query.filter(JournalProject.owner_user_id == owner_user_id)
            return query.limit(1).first() is not None

    def owner_project_slugs(self, *, owner_user_id: str | None) -> list[str]:
        """Every project slug of this owner (a small table, unlike its doc entries)."""
        with self.session() as session:
            query = session.query(JournalProject.slug)
            if owner_user_id is None:
                query = query.filter(JournalProject.owner_user_id.is_(None))
            else:
                query = query.filter(JournalProject.owner_user_id == owner_user_id)
            return [row[0] for row in query.all()]

    def live_doc_files_in_scopes(
        self,
        *,
        owner_user_id: str | None,
        project_slug: Sequence[str],
        titles: Sequence[str],
    ) -> list[tuple[str, str]]:
        """(project_slug, title) of the owner's LIVE titled docs in ``project_slug`` whose
        last path segment matches one of ``titles`` — the cross-scope fan-out (方案2)
        candidates for the same physical file under another scope.  The caller compares
        absolute paths exactly; the basename bucket only skips rows that cannot match."""
        if not project_slug or not titles:
            return []
        buckets = {title_basename_key(title) for title in titles}
        with self.session() as session:
            query = (
                self._scoped_live_docs(session, owner_user_id=owner_user_id,
                                       project_slug=list(project_slug))
                .filter(JournalEntry.title.isnot(None), basename_candidates_filter(list(buckets)))
                .with_entities(JournalProject.slug, JournalEntry.title)
            )
            return [(slug, title) for slug, title in query.all()
                    if title_basename_key(title) in buckets]

    @staticmethod
    def stale_doc_entry_ids(
        inventory: Sequence[tuple[str, str | None]],
        *,
        live_titles: list[str],
        max_depth: int | None = None,
        patterns: list[str] | None = None,
        case_sensitive: bool = True,
    ) -> list[str]:
        """Entry_ids of ``inventory`` (from ``doc_inventory``) that a snapshot proves stale."""
        live_identities = {
            identity for title in live_titles
            if (identity := path_identity(title, case_sensitive)) is not None
        }

        def _stale_candidate(title: str | None) -> bool:
            if title is None:
                return True
            identity = path_identity(title, case_sensitive)
            if identity is None:
                return False  # malformed historical data cannot be proven stale safely
            if has_dot_directory_segment(identity):
                return False  # the walker never enters dot-dirs: outside every envelope
            if max_depth is not None and identity.count("/") > max_depth:
                return False
            return identity not in live_identities

        rows = [
            (entry_id, title) for entry_id, title in inventory if _stale_candidate(title)
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
                identity = path_identity(title, case_sensitive)
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

    @staticmethod
    def _entry_load_options(*, load_chunks: bool, load_body: bool) -> list:
        options = [selectinload(JournalEntry.project)]
        if load_chunks:
            options.append(selectinload(JournalEntry.chunks))
        if not load_body:
            options.append(defer(JournalEntry.body))
        return options

    def _detach_entries(
        self, session: Session, rows: Sequence[JournalEntry], *, load_chunks: bool,
    ) -> list[JournalEntry]:
        """Detach rows; without hydrated chunks, attach the chunk count in ONE grouped query
        (what a response needs) instead of loading every chunk row of every entry."""
        counts = {} if load_chunks else self._chunk_counts(session, [r.entry_id for r in rows])
        detached: list[JournalEntry] = []
        for row in rows:
            entry = self._detach_entry(session, row, include_chunks=load_chunks)
            if entry is None:
                continue
            if not load_chunks:
                setattr(entry, "_chunk_count", counts.get(entry.entry_id, 0))
            detached.append(entry)
        return detached

    def get_entries_by_ids(
        self,
        entry_ids: list[str],
        *,
        owner_user_id: str | None = None,
        load_chunks: bool = True,
        load_body: bool = True,
    ) -> dict[str, JournalEntry]:
        """Live entries by id, in ``entry_ids`` order.  ``load_chunks=False`` / ``load_body=False``
        skip hydrating chunk rows / the body for callers that render neither (a deferred body
        raises if touched on the detached entry); ``_chunk_count`` is still set."""
        if not entry_ids:
            return {}
        order = {entry_id: i for i, entry_id in enumerate(entry_ids)}
        with self.session() as session:
            query = (
                session.query(JournalEntry)
                .options(*self._entry_load_options(load_chunks=load_chunks, load_body=load_body))
                .filter(JournalEntry.entry_id.in_(entry_ids), JournalEntry.deleted_at.is_(None))
            )
            if owner_user_id is not None:
                query = query.join(JournalProject).filter(JournalProject.owner_user_id == owner_user_id)
            detached = self._detach_entries(session, query.all(), load_chunks=load_chunks)
            return dict(sorted(
                ((entry.entry_id, entry) for entry in detached),
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
        load_chunks: bool = True,
        load_body: bool = True,
    ) -> tuple[list[JournalEntry], int]:
        with self.session() as session:
            query = (
                session.query(JournalEntry)
                .join(JournalProject, JournalProject.project_id == JournalEntry.project_id)
                .options(*self._entry_load_options(load_chunks=load_chunks, load_body=load_body))
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
            return self._detach_entries(session, rows, load_chunks=load_chunks), total

    def keyword_search(
        self,
        *,
        query: str,
        project_slug: str | Sequence[str] | None = None,
        source_kind: str | None = None,
        owner_user_id: str | None = None,
        entry_ids: list[str] | None = None,
        include_non_doc: bool = False,
        limit: int = 20,
        load_chunks: bool = True,
    ) -> tuple[list[tuple[JournalEntry, float]], int]:
        """Embedding-free FTS5 keyword search. Returns `[(entry, score)]` best-first.
        `entry_ids` (path_glob resolution) restricts the match to that set inside the FTS
        query so top_k is never post-filter-squeezed; `include_non_doc` also admits every
        entry that is not a doc (see ``fts.search``)."""
        with self.session() as session:
            fts.ensure_fts(session)
            matches = fts.search(
                session,
                query,
                owner_user_id=owner_user_id,
                project_slug=project_slug,
                source_kind=source_kind,
                entry_ids=entry_ids,
                include_non_doc=include_non_doc,
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
        """Rebuild the FTS5 table (rowid = entry id) from live entries, in one transaction.

        Idempotent; also the alignment step for tables written with auto rowids. It
        opens the session factory directly because `session()` runs the alignment
        guard that calls this. Entries are read in id-ordered batches so a table of
        large bodies is never held in memory at once.
        """
        batch_size = 200
        with self.session_factory() as session:
            fts.ensure_fts(session)
            session.execute(text(f"DELETE FROM {fts.TABLE}"))
            count = 0
            last_id = 0
            while True:
                rows = (
                    session.query(
                        JournalEntry.id, JournalEntry.entry_id, JournalProject.owner_user_id,
                        JournalProject.slug, JournalEntry.source_kind, JournalEntry.title,
                        JournalEntry.summary, JournalEntry.body, JournalEntry.tags_json,
                    )
                    .join(JournalProject, JournalProject.project_id == JournalEntry.project_id)
                    .filter(JournalEntry.deleted_at.is_(None), JournalEntry.id > last_id)
                    .order_by(JournalEntry.id)
                    .limit(batch_size)
                    .all()
                )
                if not rows:
                    return count
                for (row_id, entry_id, owner, slug, source_kind, title, summary, body,
                     tags) in rows:
                    fts.index_entry(
                        session, rowid=row_id, entry_id=entry_id, owner_user_id=owner,
                        project_slug=slug, source_kind=source_kind, title=title,
                        summary=summary, body=body, tags=tags or [],
                    )
                count += len(rows)
                last_id = rows[-1][0]

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
            fts.delete_entry(session, entry.id)
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
