"""The Journal enumeration paths as they were before they were narrowed.

Kept only as the equivalence oracle and the work baseline for
``test_journal_enumeration.py``: every override reproduces the earlier behaviour on top of
the current repository/service, so a differential test can run the same job through both and
compare results and the SQL work each one does.  Nothing in the product imports this module.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Sequence

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import JournalDocReconcileResponse, JournalSearchRequest
from app.services.journal.service import JournalService, _GlobResolution
from app.services.scope_paths import (
    is_windows_scope,
    path_matches as _path_matches,
    stored_relative_path_identity,
)
from core.db.database import Base
from core.db.models import JournalChunk, JournalEntry


class Work:
    """SQL work done while ``active``: statements issued, SQLite VM steps, ORM rows loaded."""

    VM_STEP_GRANULARITY = 1000

    def __init__(self) -> None:
        self.active = False
        self.statements = 0
        self.vm_steps = 0
        self.chunk_rows = 0

    def reset(self) -> None:
        self.statements = self.vm_steps = self.chunk_rows = 0

    @contextmanager
    def measure(self):
        self.reset()
        self.active = True
        try:
            yield self
        finally:
            self.active = False

    def __repr__(self) -> str:
        return (f"statements={self.statements} vm_steps={self.vm_steps} "
                f"chunk_rows_loaded={self.chunk_rows}")


def build_repository(database_path, work: Work, repository_cls=JournalRepository):
    """A file-backed SQLite journal whose statements, VM steps and chunk loads are counted."""
    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _count_vm_steps(dbapi_connection, _record):
        def tick():
            if work.active:
                work.vm_steps += Work.VM_STEP_GRANULARITY
            return 0
        dbapi_connection.set_progress_handler(tick, Work.VM_STEP_GRANULARITY)

    @event.listens_for(engine, "before_cursor_execute")
    def _count_statements(*_args):
        if work.active:
            work.statements += 1

    @event.listens_for(JournalChunk, "load")
    def _count_chunk_loads(*_args):
        if work.active:
            work.chunk_rows += 1

    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return repository_cls(session_factory=session_factory), engine


class LegacyJournalRepository(JournalRepository):
    """Whole-scope / whole-owner reads, exactly as they were."""

    def legacy_all_live_doc_files(self, *, owner_user_id):
        with self.session() as session:
            from core.db.models import JournalProject
            query = (
                self._scoped_live_docs(session, owner_user_id=owner_user_id)
                .filter(JournalEntry.title.isnot(None))
                .with_entities(JournalProject.slug, JournalEntry.title)
            )
            return [(slug, title) for slug, title in query.all()]

    def legacy_superseded_version_ids(self, *, project_slug, source_kind, title,
                                      owner_user_id, exclude_entry_id, case_sensitive=True):
        with self.session() as session:
            from core.db.models import JournalProject
            query = (
                session.query(JournalEntry.entry_id, JournalEntry.title)
                .join(JournalProject, JournalEntry.project_id == JournalProject.project_id)
                .filter(JournalEntry.source_kind == source_kind,
                        JournalEntry.deleted_at.is_(None),
                        JournalEntry.entry_id != exclude_entry_id)
            )
            if isinstance(project_slug, str):
                query = query.filter(JournalProject.slug == project_slug)
            else:
                query = query.filter(JournalProject.slug.in_(list(project_slug)))
            if owner_user_id is None:
                query = query.filter(JournalProject.owner_user_id.is_(None))
            else:
                query = query.filter(JournalProject.owner_user_id == owner_user_id)
            target_identity = stored_relative_path_identity(title, case_sensitive=case_sensitive)
            return [
                entry_id for entry_id, candidate_title in query.all()
                if target_identity is not None
                and stored_relative_path_identity(
                    candidate_title or "", case_sensitive=case_sensitive) == target_identity
            ]

    def legacy_live_doc_titles(self, *, project_slug, owner_user_id):
        with self.session() as session:
            query = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .filter(JournalEntry.title.isnot(None))
                .with_entities(JournalEntry.title)
                .order_by(JournalEntry.title)
            )
            return [row[0] for row in query.all()]

    def legacy_scoped_live_doc_id_titles(self, *, project_slug, owner_user_id):
        with self.session() as session:
            rows = (
                self._scoped_live_docs(session, project_slug=project_slug,
                                       owner_user_id=owner_user_id)
                .filter(JournalEntry.title.isnot(None))
                .with_entities(JournalEntry.entry_id, JournalEntry.title)
                .all()
            )
            return [(row[0], row[1]) for row in rows]


class LegacyJournalService(JournalService):
    """Per-file supersede, owner-wide inventory, double-read reconcile, full glob listing."""

    def _related_scope_doc_files(self, *, owner_user_id, project_slug, titles):
        return self.repository.legacy_all_live_doc_files(owner_user_id=owner_user_id)

    def _supersede_doc_versions(self, winners: Sequence[tuple[str, str, str]], *, owner_user_id):
        retired = 0
        for project_slug, title, entry_id in winners:
            stale = self.repository.legacy_superseded_version_ids(
                project_slug=self._project_scope_aliases(
                    project_slug, owner_user_id=owner_user_id)[1],
                source_kind="doc", title=title, owner_user_id=owner_user_id,
                exclude_entry_id=entry_id, case_sensitive=not is_windows_scope(project_slug))
            for old_id in stale:
                response = self.delete_entry(
                    old_id, reason="superseded by newer version", owner_user_id=owner_user_id)
                retired += int(response.deleted)
        return retired

    def reconcile_doc_scope(self, *, project_slug, live_titles, owner_user_id=None,
                            max_depth=None, patterns=None):
        project_slug, project_slugs = self._project_scope_aliases(
            project_slug, owner_user_id=owner_user_id)
        normalized_titles = sorted({title for title in live_titles if title})
        if not normalized_titles:
            return JournalDocReconcileResponse(
                project_slug=project_slug, live_titles=0, removed_entries=0, skipped=True,
                reason="empty snapshot; stale doc sweep skipped")
        existing_titles = self.repository.legacy_live_doc_titles(
            project_slug=project_slugs, owner_user_id=owner_user_id)
        case_sensitive = not is_windows_scope(project_slug)
        existing = {i for t in existing_titles
                    if (i := stored_relative_path_identity(t, case_sensitive=case_sensitive))
                    is not None}
        live = {i for t in normalized_titles
                if (i := stored_relative_path_identity(t, case_sensitive=case_sensitive))
                is not None}
        if existing and existing.isdisjoint(live):
            return JournalDocReconcileResponse(
                project_slug=project_slug, live_titles=len(normalized_titles),
                removed_entries=0, skipped=True,
                reason="snapshot is disjoint from existing docs; stale doc sweep skipped")
        # The second, independent read of the same scope.
        inventory = self.repository.doc_inventory(
            project_slug=project_slugs, owner_user_id=owner_user_id)
        stale_ids = self.repository.stale_doc_entry_ids(
            inventory, live_titles=normalized_titles, max_depth=max_depth,
            patterns=patterns, case_sensitive=case_sensitive)
        removed_ids, deleted_vectors = self._delete_entries(
            stale_ids, reason="absent from latest doc snapshot", owner_user_id=owner_user_id)
        return JournalDocReconcileResponse(
            project_slug=project_slug, live_titles=len(normalized_titles),
            removed_entries=len(removed_ids), removed_entry_ids=removed_ids,
            deleted_vectors=deleted_vectors, skipped=False)

    def _resolve_path_glob_ids(self, request: JournalSearchRequest):
        if not request.path_glob:
            return _GlobResolution(ids=None, scope_doc_count=0, sample_title=None)
        _canonical, project_slugs = self._project_scope_aliases(
            request.project_slug, owner_user_id=request.owner_user_id)
        pairs = self.repository.legacy_scoped_live_doc_id_titles(
            project_slug=project_slugs, owner_user_id=request.owner_user_id)
        ids = [entry_id for entry_id, title in pairs
               if _path_matches(title, request.path_glob,
                                case_sensitive=not is_windows_scope(request.project_slug))]
        titles = [title for _, title in pairs if title]
        sample = min(titles, key=lambda t: (len(t), t)) if titles else None
        return _GlobResolution(ids=ids, scope_doc_count=len(pairs), sample_title=sample)
