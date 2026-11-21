"""SQLAlchemy repository for the index-inventory catalog."""
from __future__ import annotations

import fnmatch
import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, NamedTuple

from sqlalchemy import func, or_, select, tuple_
from sqlalchemy.orm import Session, aliased

from core.db.database import get_db_session
from core.db.write_batches import YIELD_S, paced_batches
from core.db.models import (
    CATALOG_FOLDING_PATH_SQL, CATALOG_FOLDING_PROJECT_SQL, CATALOG_LEGACY_PATH_SQL,
    CodeScopeVisibility, IndexCatalog,
    IndexScopePolicy,
)
from app.services.scope_paths import (
    PathReach, directory_reach, glob_reach, has_dot_directory_segment, is_windows_scope,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _within_sweep_envelope(rel_path: str, patterns: list[str] | None,
                           max_depth: int | None, kind: str = "code") -> bool:
    """Is this catalog row inside the reporting run's scan envelope, hence eligible for the
    replace-sweep? Mirrors the journal stale-doc reconcile envelope (W169b): a row is swept
    only if its depth (slash count) is within ``max_depth`` AND its basename matches one of
    ``patterns``. None on an axis = no bound there (code reports pass neither → whole kind,
    unchanged). Basename matched case-sensitively (fnmatchcase) so a casing mismatch errs
    toward KEEPING the row, never sweeping a file outside the glob. Doc rows under a
    dot-directory are outside every envelope (the Journal walker never enters them, so a
    report can't have scanned them); code walks DO enter dot-dirs (``.github/``), so code
    rows keep the whole-kind sweep."""
    norm = rel_path.replace("\\", "/")
    if kind == "doc" and has_dot_directory_segment(norm):
        return False
    if max_depth is not None and norm.count("/") > max_depth:
        return False
    globs = [g for g in (patterns or []) if g]
    if globs and globs != ["*"]:
        base = norm.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatchcase(base, g) for g in globs)
    return True


# Keeps every IN (...) well under SQLite's bound-parameter limit (999 on old builds).
_IN_CHUNK = 400
# Rows one report transaction writes before it commits (~0.25 ms per new row, so ~0.5 s of lock).
REPORT_BATCH_ROWS = 2000


def _chunks(values: list, size: int = _IN_CHUNK) -> Iterator[list]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _in_inventory_order(rows: list) -> list:
    """Order one scope's rows as ``ORDER BY owner_user_id, host, project, rel_path`` would.

    With ``project`` pinned that clause makes SQLite prefer ``ix_index_catalog_scope_lower_path``
    to avoid the sort, which walks the owner's whole index instead of seeking the scope
    (583K entries to return ~1.7K rows). The scope's rows are sorted here instead. Str ordering
    is by code point, which equals SQLite's BINARY order over UTF-8 bytes."""
    return sorted(rows, key=lambda r: (r.owner_user_id, r.host, r.project, r.rel_path))


# Upper bound of "starts with"; no stored path contains the last code point.
_RANGE_END = "\U0010ffff"


def _path_ranges(reach: PathReach) -> list[tuple[str, str]] | None:
    """``[low, high)`` ranges over the comparison spelling that cover every canonical path
    ``reach`` can accept, or None when some anchor is not a range (a bare name matches at
    any depth). An ``entry`` is the file itself plus everything below it; ``dir/`` and
    ``dir0`` bound the paths below ``dir`` because ``0`` is the character after ``/``."""
    if reach.any_depth:
        return None
    ranges = [(prefix, prefix + _RANGE_END) for prefix in reach.prefixes]
    for entry in reach.entries:
        ranges += [(entry, entry + "\x00"), (entry + "/", entry + "0")]
    ranges += [(directory + "/", directory + "0") for directory in reach.directories]
    return ranges


# A stored string that is not already its own identity: a leading ``/`` or ``./``, a
# trailing ``/`` or ``/.``, ``//``, ``/./``, ``..``, a backslash, or a drive-like ``:``.
_NON_CANONICAL_PATH = re.compile(r"^(?:\./|/)|/\.?$|//|/\./|\.\.|[\\:]|^\.?$")


def _is_canonical_spelling(path: str) -> bool:
    """True when the stored string is already its own identity, so the pathlib-based
    normalisation (about 5 microseconds a path) cannot change or reject it. Anything
    unusual, including a drive-like ``a:`` segment, still takes the exact validator."""
    return _NON_CANONICAL_PATH.search(path) is None


class BaselineRow(NamedTuple):
    """A catalog row reduced to what an incremental diff compares against. Duck-types the
    ``IndexCatalog`` attributes ``IndexCatalogService.inventory`` reads; the columns a
    baseline never needs (and that cost the most to load: ``metadata_json``) are None."""
    owner_user_id: str
    host: str
    project: str
    root_dir: str | None
    rel_path: str
    kind: str
    language: str | None
    chunk_count: int
    content_hash: str | None
    size_bytes: int | None
    collection: str | None = None
    indexed_at: datetime | None = None
    metadata_json: dict | None = None


class IndexCatalogRepository:
    def __init__(self, session_factory: Callable[[], Any] = get_db_session):
        self.session_factory = session_factory

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            yield session

    @staticmethod
    def _policy_query(
        session: Session, *, owner_user_id: str, project: str,
        host: str | None = None, kind: str | None = None,
    ):
        query = session.query(IndexScopePolicy).filter(
            IndexScopePolicy.owner_user_id == owner_user_id,
            IndexScopePolicy.project == project,
        )
        if host is not None:
            query = query.filter(IndexScopePolicy.host == host)
        if kind is not None:
            query = query.filter(IndexScopePolicy.kind == kind)
        return query

    def report_files(
        self,
        *,
        owner_user_id: str,
        host: str,
        project: str,
        root_dir: str | None,
        files: list[dict[str, Any]],
        replace: bool = True,
        patterns: list[str] | None = None,
        max_depth: int | None = None,
    ) -> tuple[int, int]:
        """Upsert reported files; when ``replace`` soft-delete absent ones. Returns (upserted, removed)."""
        # Key by (rel_path, kind): code and doc inventories are reported by different
        # tools (index_code vs journal_index_dir) and coexist under one project — they
        # must never collide or clobber each other (W151).
        fold = (lambda value: value.replace("\\", "/").casefold()) if is_windows_scope(project) else (
            lambda value: value.replace("\\", "/"))
        seen: dict[tuple[str, str], str] = {}
        for f in files:
            key = (fold(f["rel_path"]), f.get("kind", "code"))
            prior = seen.setdefault(key, f["rel_path"])
            if prior != f["rel_path"]:
                raise ValueError(
                    f"case-insensitive catalog path collision: {prior!r} and {f['rel_path']!r}")
        reported_keys = set(seen)
        reported_kinds = {kind for _, kind in reported_keys}
        upserted = 0
        removed = 0
        scope = (
            IndexCatalog.owner_user_id == owner_user_id,
            IndexCatalog.host == host,
            IndexCatalog.project == project,
        )
        # Full-sync sweep: soft-delete files absent from a NON-EMPTY report, but
        # ONLY within the kinds this report actually covers — a code report must
        # not wipe the project's doc inventory (W151). An empty report never wipes
        # anything (that would let an accidental index on a source-less directory
        # silently erase a user's catalog); explicit removal goes through
        # delete_files(). The sweep needs only (id, path, kind) of live rows and
        # ends in UPDATEs by id, so it never builds an ORM object per scope row.
        # The stale ids are chosen first (a read) and soft-deleted last.
        swept_ids: list[int] = []
        if replace and reported_keys:
            with self.session() as session:
                for row_id, row_path, row_kind in session.query(
                    IndexCatalog.id, IndexCatalog.rel_path, IndexCatalog.kind,
                ).filter(*scope, IndexCatalog.kind.in_(sorted(reported_kinds)),
                         IndexCatalog.deleted_at.is_(None)):
                    key = (fold(row_path), row_kind)
                    if (key not in reported_keys
                            # ...and within THIS run's scan envelope — a "*.txt" doc report
                            # must not sweep the "*.md" rows it never scanned (W169b).
                            and _within_sweep_envelope(key[0], patterns, max_depth, row_kind)):
                        swept_ids.append(row_id)
        # One transaction per batch of rows (core/db/write_batches.py): a scope of 100k new rows held
        # SQLite's write lock ~25 s in one. What readers see in between is always consistent: upserts
        # come first and the soft deletes last, so every file of the new snapshot is live from the
        # moment its batch commits, no file of it is ever hidden, and the stale files stay visible
        # (the old inventory plus part of the new one) until the final batches remove them. A crash
        # in between leaves that superset, which the next report converges.
        for batch in paced_batches(files, REPORT_BATCH_ROWS):
            with self.session() as session:
                existing = self._rows_of_reported_files(
                    session, scope, files=batch, kinds=reported_kinds, fold=fold,
                    windows=is_windows_scope(project))
                for f in batch:
                    kind = f.get("kind", "code")
                    row = existing.get((fold(f["rel_path"]), kind))
                    # content_changed_at is the only "this file changed" signal: indexed_at moves
                    # on every report. A new row, another content hash, or a row coming back from
                    # a soft delete (its duplicates came back with it) is a change; an unchanged
                    # re-report must leave the column alone.
                    content_changed = (
                        row is None or row.deleted_at is not None
                        or row.content_hash != f.get("content_hash"))
                    if row is None:
                        row = IndexCatalog(
                            owner_user_id=owner_user_id, host=host, project=project,
                            rel_path=f["rel_path"], kind=kind,
                        )
                        session.add(row)
                    if content_changed:
                        row.content_changed_at = _utcnow()
                    row.root_dir = root_dir
                    row.rel_path = f["rel_path"].replace("\\", "/")
                    row.kind = kind
                    row.collection = f.get("collection")
                    row.language = f.get("language")
                    row.chunk_count = int(f.get("chunk_count", 0) or 0)
                    row.content_hash = f.get("content_hash")
                    row.size_bytes = f.get("size_bytes")
                    row.metadata_json = f.get("metadata") or {}
                    row.indexed_at = _utcnow()
                    row.deleted_at = None
                    upserted += 1
        if swept_ids:
            time.sleep(YIELD_S)
            now = _utcnow()
            for batch in paced_batches(swept_ids, REPORT_BATCH_ROWS):
                with self.session() as session:
                    for ids in _chunks(batch):
                        session.query(IndexCatalog).filter(IndexCatalog.id.in_(ids)).update(
                            {IndexCatalog.deleted_at: now}, synchronize_session=False)
                        removed += len(ids)
        return upserted, removed

    @staticmethod
    def _rows_of_reported_files(
        session: Session, scope: tuple, *, files: list[dict[str, Any]],
        kinds: set[str], fold: Callable[[str], str], windows: bool,
    ) -> dict[tuple[str, str], IndexCatalog]:
        """Existing catalog rows (live or soft-deleted) that a report's files resolve to,
        keyed like the report: (folded path, kind). Only those rows are loaded — a
        one-file incremental report must not cost the size of the scope.

        POSIX scopes match the path spelling exactly (plus the all-backslash legacy
        spelling). Windows scopes match case-folded identity through the
        ``lower(rel_path)`` expression index; SQLite ``lower()`` is ASCII-only, so a
        non-ASCII path cannot be located that way and the scope is read instead — the
        previous behaviour, kept for the rare case it is needed for."""
        wanted = {(fold(f["rel_path"]), f.get("kind", "code")) for f in files}
        if not wanted:
            return {}
        paths = {path for path, _ in wanted}
        if windows and not all(path.isascii() for path in paths):
            candidates = session.query(IndexCatalog).filter(*scope)
            return {(fold(row.rel_path), row.kind): row for row in candidates}
        spellings = sorted(paths | {path.replace("/", "\\") for path in paths})
        found: dict[tuple[str, str], IndexCatalog] = {}
        for chunk in _chunks(spellings):
            match = (func.lower(IndexCatalog.rel_path).in_([p.lower() for p in chunk])
                     if windows else IndexCatalog.rel_path.in_(chunk))
            for row in session.query(IndexCatalog).filter(
                    *scope, IndexCatalog.kind.in_(sorted(kinds)), match):
                key = (fold(row.rel_path), row.kind)
                if key in wanted:
                    found[key] = row
        return found

    def upsert_policy(
        self, *, owner_user_id: str, host: str, project: str, kind: str,
        root_dir: str, auto_refresh: bool, respect_gitignore: bool,
        exclude_globs: list[str], min_check_interval_s: int,
    ) -> IndexScopePolicy:
        with self.session() as session:
            row = self._policy_query(
                session, owner_user_id=owner_user_id, host=host,
                project=project, kind=kind).one_or_none()
            if row is None:
                row = IndexScopePolicy(
                    owner_user_id=owner_user_id, host=host, project=project, kind=kind)
                session.add(row)
            row.root_dir = root_dir
            row.auto_refresh = int(auto_refresh)
            row.respect_gitignore = int(respect_gitignore)
            row.exclude_globs_json = list(exclude_globs)
            row.min_check_interval_s = min_check_interval_s
            session.flush()
            session.refresh(row)
            session.expunge(row)
            return row

    def get_policy(
        self, *, owner_user_id: str, host: str, project: str, kind: str,
    ) -> IndexScopePolicy | None:
        with self.session() as session:
            row = self._policy_query(
                session, owner_user_id=owner_user_id, host=host,
                project=project, kind=kind).one_or_none()
            if row is not None:
                session.expunge(row)
            return row

    def list_policies(
        self,
        *,
        owner_user_id: str | None = None,
        host: str | None = None,
        project: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, str]]:
        """Return compact scope registrations, including scopes with zero file rows."""
        with self.session() as session:
            query = session.query(IndexScopePolicy)
            if owner_user_id is not None:
                query = query.filter(IndexScopePolicy.owner_user_id == owner_user_id)
            if host is not None:
                query = query.filter(IndexScopePolicy.host == host)
            if project is not None:
                query = query.filter(IndexScopePolicy.project == project)
            if kind is not None:
                query = query.filter(IndexScopePolicy.kind == kind)
            rows = query.order_by(
                IndexScopePolicy.owner_user_id,
                IndexScopePolicy.host,
                IndexScopePolicy.project,
            ).all()
            return [{
                "owner_user_id": row.owner_user_id,
                "host": row.host,
                "project": row.project,
                "root_dir": row.root_dir,
                "kind": row.kind,
            } for row in rows]

    def has_inventory(
        self,
        *,
        owner_user_id: str,
        project: str,
        kind: str,
    ) -> bool:
        """Whether a durable completed inventory row exists for a scope/kind."""
        with self.session() as session:
            return session.query(IndexCatalog.id).filter(
                IndexCatalog.owner_user_id == owner_user_id,
                IndexCatalog.project == project,
                IndexCatalog.kind == kind,
                IndexCatalog.deleted_at.is_(None),
            ).first() is not None

    @staticmethod
    def _visibility_out(row: CodeScopeVisibility) -> dict[str, Any]:
        return {
            "owner_user_id": row.owner_user_id,
            "host": row.host,
            "project": row.project,
            "state": row.state,
            "job_id": row.job_id,
            "previous_completed_snapshot_available": bool(
                row.previous_completed_snapshot_available),
        }

    def set_code_scope_visibility(
        self,
        *,
        owner_user_id: str,
        host: str,
        project: str,
        job_id: str | None,
        state: str,
    ) -> dict[str, Any]:
        """Persist Code KB visibility before/after an in-place vector mutation.

        ``completed`` is deliberately a durable state rather than deleting the row: the
        next writer needs to know that a prior completed snapshot existed.  A newer
        preparing job owns the marker; an older completion must never overwrite it.
        """
        if state not in {"preparing", "completed", "recovery_required"}:
            raise ValueError(f"unsupported code scope visibility state: {state}")
        with self.session() as session:
            row = session.query(CodeScopeVisibility).filter(
                CodeScopeVisibility.owner_user_id == owner_user_id,
                CodeScopeVisibility.project == project,
            ).one_or_none()
            if row is None:
                row = CodeScopeVisibility(
                    owner_user_id=owner_user_id,
                    project=project,
                    host=host,
                    state=state,
                    job_id=job_id,
                    previous_completed_snapshot_available=(state == "completed"),
                )
                session.add(row)
            else:
                if state == "completed" and row.job_id not in (None, job_id):
                    raise RuntimeError("code scope visibility belongs to a newer index job")
                prior_completed = bool(row.previous_completed_snapshot_available) or (
                    row.state == "completed")
                row.host = host
                row.state = state
                row.job_id = job_id
                row.previous_completed_snapshot_available = (
                    True if state == "completed" else prior_completed)
            session.flush()
            session.refresh(row)
            result = self._visibility_out(row)
            session.expunge(row)
            return result

    def code_scope_visibility(
        self,
        *,
        owner_user_id: str,
        projects: list[str],
    ) -> dict[str, Any] | None:
        """Return the strongest marker among equivalent scope spellings for one owner."""
        project_set = [project for project in dict.fromkeys(projects) if project]
        if not project_set:
            return None
        rank = {"recovery_required": 0, "preparing": 1, "completed": 2}
        with self.session() as session:
            rows = session.query(CodeScopeVisibility).filter(
                CodeScopeVisibility.owner_user_id == owner_user_id,
                CodeScopeVisibility.project.in_(project_set),
            ).all()
            if not rows:
                return None
            row = min(rows, key=lambda candidate: (rank.get(candidate.state, -1), candidate.project))
            result = self._visibility_out(row)
            session.expunge(row)
            return result

    def list_code_scope_visibility(
        self,
        *,
        owner_user_id: str | None = None,
        host: str | None = None,
        project: str | None = None,
        incomplete_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self.session() as session:
            query = session.query(CodeScopeVisibility)
            if owner_user_id is not None:
                query = query.filter(CodeScopeVisibility.owner_user_id == owner_user_id)
            if host is not None:
                query = query.filter(CodeScopeVisibility.host == host)
            if project is not None:
                query = query.filter(CodeScopeVisibility.project == project)
            if incomplete_only:
                query = query.filter(CodeScopeVisibility.state != "completed")
            rows = query.order_by(
                CodeScopeVisibility.owner_user_id,
                CodeScopeVisibility.project,
            ).all()
            result = [self._visibility_out(row) for row in rows]
            for row in rows:
                session.expunge(row)
            return result

    def clear_code_scope_visibility(self, *, owner_user_id: str, project: str) -> int:
        """Explicit deletion is an authorized recovery boundary for a broken scope."""
        with self.session() as session:
            return session.query(CodeScopeVisibility).filter(
                CodeScopeVisibility.owner_user_id == owner_user_id,
                CodeScopeVisibility.project == project,
            ).delete(synchronize_session=False)

    def list_inventory(
        self,
        *,
        owner_user_id: str | None = None,
        host: str | None = None,
        project: str | None = None,
        kind: str | None = None,
    ) -> list[IndexCatalog]:
        with self.session() as session:
            query = session.query(IndexCatalog).filter(IndexCatalog.deleted_at.is_(None))
            if owner_user_id is not None:
                query = query.filter(IndexCatalog.owner_user_id == owner_user_id)
            if host is not None:
                query = query.filter(IndexCatalog.host == host)
            if project is not None:
                query = query.filter(IndexCatalog.project == project)
            if kind is not None:
                query = query.filter(IndexCatalog.kind == kind)
            if project is not None:
                rows = _in_inventory_order(query.all())
            else:
                rows = query.order_by(
                    IndexCatalog.owner_user_id, IndexCatalog.host,
                    IndexCatalog.project, IndexCatalog.rel_path,
                ).all()
            for row in rows:
                session.expunge(row)
            return rows

    def list_inventory_baseline(
        self,
        *,
        owner_user_id: str | None = None,
        host: str | None = None,
        project: str | None = None,
        kind: str | None = None,
    ) -> list[BaselineRow]:
        """``list_inventory`` for incremental-diff baselines: same rows, same order, but
        bare columns (no ORM entity, no ``metadata_json`` JSON decode, no ``indexed_at``)."""
        with self.session() as session:
            query = session.query(
                IndexCatalog.owner_user_id, IndexCatalog.host, IndexCatalog.project,
                IndexCatalog.root_dir, IndexCatalog.rel_path, IndexCatalog.kind,
                IndexCatalog.language, IndexCatalog.chunk_count,
                IndexCatalog.content_hash, IndexCatalog.size_bytes,
            ).filter(IndexCatalog.deleted_at.is_(None))
            if owner_user_id is not None:
                query = query.filter(IndexCatalog.owner_user_id == owner_user_id)
            if host is not None:
                query = query.filter(IndexCatalog.host == host)
            if project is not None:
                query = query.filter(IndexCatalog.project == project)
            if kind is not None:
                query = query.filter(IndexCatalog.kind == kind)
            if project is not None:
                return _in_inventory_order([BaselineRow(*row) for row in query])
            return [BaselineRow(*row) for row in query.order_by(
                IndexCatalog.owner_user_id, IndexCatalog.host,
                IndexCatalog.project, IndexCatalog.rel_path,
            )]

    def list_code_paths(self, *, owner_user_id: str, project: str) -> set[str]:
        """Live ``code`` rel_paths the catalog holds for one scope, across hosts. A single
        bare column: the baseline an incremental job's removals are checked against
        without enumerating the project's paths in Qdrant."""
        with self.session() as session:
            return {row[0] for row in session.query(IndexCatalog.rel_path).filter(
                IndexCatalog.owner_user_id == owner_user_id,
                IndexCatalog.project == project,
                IndexCatalog.kind == "code",
                IndexCatalog.deleted_at.is_(None),
            )}

    def first_scope_with_live_rows(
        self, *, owner_user_id: str, kind: str, candidates: Sequence[str],
    ) -> str | None:
        """The first of ``candidates`` (exact scope spellings, in priority order) under
        which the owner holds a live ``kind`` row, else None.

        One statement of index probes (owner, kind, deleted_at, project), each stopping
        at its first row whatever the scope's size - a path's worth of levels costs a
        round trip, not a read of any scope's files. Plain driver SQL: the probes are
        the same shape every time and compiling them would cost more than running them.
        """
        candidates = [scope for scope in dict.fromkeys(candidates) if scope]
        if not candidates:
            return None
        probe = (
            "SELECT ? WHERE EXISTS (SELECT 1 FROM index_catalog WHERE owner_user_id = ? "
            "AND kind = ? AND deleted_at IS NULL AND project = ?)"
        )
        params: list[str] = []
        for scope in candidates:
            params += [scope, owner_user_id, kind, scope]
        with self.session() as session:
            found = {row[0] for row in session.connection().exec_driver_sql(
                " UNION ALL ".join([probe] * len(candidates)), tuple(params))}
        return next((scope for scope in candidates if scope in found), None)

    def project_spellings(
        self, *, owner_user_id: str, kind: str, folded_scopes: Sequence[str],
    ) -> list[str]:
        """Stored scope names of the owner's live ``kind`` rows that may be spellings of
        one of ``folded_scopes`` (each ``sql_foldable``): equal under SQLite's ``lower()``,
        plus the scopes whose spelling ``lower()`` cannot fold (normally none).

        The caller still decides which candidates are really the same directory. Each
        scope costs one index seek per distinct spelling it has (``min(project)`` past the
        last one found), so the rows read are those few spellings, never the project's
        files and never the owner's other scopes. Plain driver SQL: the statements must
        repeat the partial indexes' WHERE verbatim (core/db/models.py::IndexCatalog).
        """
        scopes = [scope for scope in dict.fromkeys(folded_scopes) if scope]
        if not scopes:
            return []
        # The first step also bounds ``project`` (> ''): without a range the planner reads
        # the owner-wide covering index for a bare min(project) instead of this scope's.
        next_spelling = (
            "(SELECT min(project) FROM index_catalog WHERE owner_user_id = ? AND kind = ? "
            "AND deleted_at IS NULL AND lower(project) = lower({scope}) AND project > {after})"
        )
        statement = (
            f"WITH RECURSIVE wanted(scope) AS (VALUES {','.join(['(?)'] * len(scopes))}), "
            "spelling(scope, project) AS ("
            f"SELECT scope, {next_spelling.format(scope='wanted.scope', after=chr(39) * 2)} "
            "FROM wanted UNION ALL "
            f"SELECT scope, {next_spelling.format(scope='spelling.scope', after='spelling.project')} "
            "FROM spelling WHERE project IS NOT NULL) "
            "SELECT project FROM spelling WHERE project IS NOT NULL"
        )
        owner_kind = (owner_user_id, kind)
        with self.session() as session:
            connection = session.connection()
            found = {row[0] for row in connection.exec_driver_sql(
                statement, (*scopes, *owner_kind, *owner_kind))}
            # Spellings lower() cannot fold are not reachable by the range above. Their
            # partial index holds only such rows (normally none); an unmigrated database
            # lacks it and answers by scanning instead.
            present = connection.exec_driver_sql(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                ("ix_index_catalog_live_odd_project",)).first() is not None
            clause = " INDEXED BY ix_index_catalog_live_odd_project" if present else ""
            found.update(row[0] for row in connection.exec_driver_sql(
                f"SELECT DISTINCT project FROM index_catalog{clause} "
                "WHERE owner_user_id = ? AND kind = ? AND deleted_at IS NULL "
                f"AND {CATALOG_FOLDING_PROJECT_SQL}", owner_kind))
        return sorted(found)

    def matching_code_paths(
        self, *, owner_user_id: str, project: str | list[str] | None,
        pattern: str | None, case_sensitive: bool, path_prefix: str | None = None,
    ) -> tuple[list[str], int]:
        """Resolve a ``path_glob`` and/or a directory prefix to the owner's matching files
        without enumerating Qdrant's path facet.

        ``path_prefix`` is a scope-relative directory: only files below it are candidates
        and ``pattern`` (when given) is matched against the path *relative to it*, so a
        glob written for a subdirectory keeps meaning the same thing there. Candidates
        come from index ranges over the literal part of the prefix or glob, so the rows
        read track the match count instead of the scope size; a glob with no literal
        anchor (``*.md``) still reads the scope. Gitignore semantics are decided here, by
        ``pathspec``, once per candidate.

        Returns ``(paths, count)``: the stored spellings, and how many indexed files were
        inspected - the files below ``path_prefix`` when one is given, else the scope's.
        """
        import pathspec
        from app.services import scope_paths

        matcher = candidate_regex = None
        reach: PathReach | None = None
        paths_wanted = True
        if pattern:
            normalized = scope_paths.normalize_relative_path(pattern)
            matcher = pathspec.PathSpec.from_lines(
                'gitignore', [normalized if case_sensitive else normalized.casefold()])
            candidate_regex = next(
                (item.regex for item in matcher.patterns if item.include), None)
            if candidate_regex is None:
                # A comment or negation-only pattern selects nothing; the scope size is
                # still reported so the caller can explain the empty result.
                matcher, paths_wanted = None, False
            reach = glob_reach(normalized, case_sensitive=case_sensitive)
        prefix_identity = None
        if path_prefix:
            prefix_identity = scope_paths.stored_relative_path_identity(
                path_prefix.strip('/'), case_sensitive=case_sensitive)
            if prefix_identity is None:
                raise ValueError('path prefix must be a scope-relative directory')
            reach = directory_reach(prefix_identity, case_sensitive=case_sensitive)

        projects = None if not project else (
            [project] if isinstance(project, str) else list(project))
        base = [
            IndexCatalog.owner_user_id == owner_user_id,
            IndexCatalog.kind == 'code',
            IndexCatalog.deleted_at.is_(None),
            IndexCatalog.chunk_count > 0,
        ]
        if projects:
            base.append(IndexCatalog.project.in_(projects))
        key = IndexCatalog.rel_path if case_sensitive else func.lower(IndexCatalog.rel_path)
        ranges = _path_ranges(reach) if reach is not None else None

        odd_index = (
            "ix_index_catalog_code_odd_path" if projects
            else "ix_index_catalog_code_odd_path_any_scope")
        with self.session() as session:
            def read(*conditions) -> list[str]:
                # Core, not ORM, execution: a scope's worth of bare strings should not pay
                # per-row entity machinery.
                return session.connection().execute(
                    select(IndexCatalog.rel_path).where(*base, *conditions)).scalars().all()

            def read_odd(spelling_sql: str) -> list[str]:
                # Rows whose stored spelling a range cannot bound live in a partial index
                # of just those rows. Without per-column statistics SQLite cannot tell it
                # is tiny and picks an equally-shaped full index on a tie, so it is named
                # (SQLAlchemy cannot render SQLite's INDEXED BY) - but only when this
                # database has it: an unmigrated one still answers, by scanning.
                connection = session.connection()
                present = connection.exec_driver_sql(
                    "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                    (odd_index,)).first() is not None
                clause = f" INDEXED BY {odd_index}" if present else ""
                conditions = ["owner_user_id = ?", "kind = 'code'", "deleted_at IS NULL",
                              "chunk_count > 0", spelling_sql]
                params: list[str] = [owner_user_id]
                if projects:
                    conditions.append(f"project IN ({', '.join('?' * len(projects))})")
                    params += projects
                return [row[0] for row in connection.exec_driver_sql(
                    f"SELECT rel_path FROM index_catalog{clause} "
                    f"WHERE {' AND '.join(conditions)}", tuple(params))]

            if not paths_wanted:
                candidates: set[str] = set()
            elif ranges is None and reach is not None and reach.any_depth:
                # A bare name ("docs/", "README.md") may sit at any depth, so no range
                # bounds it - but every match contains the name, and SQLite can test that
                # inside the index scan instead of shipping the whole scope to Python.
                # Legacy spellings keep the name's characters, so none escapes this.
                candidates = set(read(or_(*[
                    func.instr(key, name) > 0 for name in reach.any_depth])))
                if not case_sensitive:
                    candidates.update(read_odd(CATALOG_FOLDING_PATH_SQL))
            elif ranges is None:
                candidates = set(read())
            else:
                candidates = set()
                for low, high in ranges:
                    candidates.update(read(key >= low, key < high))
                # Stored spellings a range cannot bound (leading slash, ``//``, backslash,
                # ``.`` segments) normalise into the same identity; a probe of the partial
                # index costs nothing when there are none.
                candidates.update(read_odd(CATALOG_LEGACY_PATH_SQL))
                if not case_sensitive:
                    candidates.update(read_odd(CATALOG_FOLDING_PATH_SQL))
            scope_count = None
            if prefix_identity is None:
                scope_count = session.execute(
                    select(func.count(IndexCatalog.rel_path.distinct())).where(*base)
                ).scalar_one()

        under_prefix = 0
        accepted: list[str] = []
        for path in candidates:
            if _is_canonical_spelling(path):
                identity = path if case_sensitive else path.casefold()
            else:
                identity = scope_paths.stored_relative_path_identity(
                    path, case_sensitive=case_sensitive)
                if identity is None:
                    continue
            if prefix_identity is not None:
                if not identity.startswith(prefix_identity + '/'):
                    continue
                under_prefix += 1
                identity = identity[len(prefix_identity) + 1:]
            if matcher is not None and (
                    candidate_regex.search(identity) is None
                    or not matcher.match_file(identity)):
                continue
            accepted.append(path)
        if len(accepted) > 100_000:
            raise ValueError('path_glob matches more than 100000 files; narrow the pattern or repository scope')
        count = scope_count if prefix_identity is None else under_prefix
        return sorted(accepted), int(count)

    def aggregate_inventory(
        self,
        *,
        owner_user_id: str | None = None,
        host: str | None = None,
        project: str | None = None,
        kind: str | None = None,
        winners_project_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Project-level SUMMARY rows for ``service.inventory(detail=False)``: one row per
        (owner_user_id, host, project, kind) with SQL-side ``file_count``/``chunk_count``
        sums and freshness, instead of ``list_inventory()``'s O(all files) ORM download
        (208,718 rows / ~8s for one production call — see
        docs/index_catalog_scope_and_inventory_plan.md A.1/A.2). No ORM entities are
        instantiated (this selects bare columns, not ``IndexCatalog`` rows) and there is
        no ``ORDER BY`` over the underlying file table.

        ``root_dir``/``last_commit`` must match what the DETAIL path derives from the
        same rows, INCLUDING when several files tie on the newest ``indexed_at`` (common:
        one index run stamps a whole batch with one timestamp). DETAIL walks
        ``list_inventory()``'s ``ORDER BY ... rel_path`` rows and keeps the first row that
        beats the running max (strict ``>``), so its rule is "among rows tied at
        MAX(indexed_at), the smallest rel_path wins", and its ``root_dir`` is simply the
        project's smallest-rel_path row.

        Selecting ``root_dir``/``metadata_json`` as bare columns next to ``MAX()`` — which
        SQLite resolves from the max row (https://www.sqlite.org/lang_select.html#bareagg)
        — does NOT reproduce that: on a tie SQLite may return any of the tied rows, so
        SUMMARY and DETAIL could report different ``last_commit`` for the same project.
        The two winner rows are therefore picked explicitly, per group, by an index seek on
        ``ix_index_catalog_live_summary`` (the group's newest stamp is the group aggregate
        itself; the smallest rel_path at that stamp is the next index column), identical to
        DETAIL's tie-break. Ranking every row of the owner with ROW_NUMBER() instead cost
        4.5 s of a 5.5 s call on 583K rows. Cost stays ~(project x kind) rows returned and
        ``metadata_json`` is deserialised only for those winners.

        ``winners_project_contains`` (an already-lowercased, forward-slash needle) limits
        the winner computation to projects whose name contains it. Counts and freshness still cover
        every project, so totals are unchanged; only projects outside the needle come back
        with ``root_dir``/``last_commit`` unset, which a caller that is about to drop them
        never reads.
        """
        group_cols = (
            IndexCatalog.owner_user_id, IndexCatalog.host,
            IndexCatalog.project, IndexCatalog.kind,
        )

        def _scoped(query):
            query = query.filter(IndexCatalog.deleted_at.is_(None))
            if owner_user_id is not None:
                query = query.filter(IndexCatalog.owner_user_id == owner_user_id)
            if host is not None:
                query = query.filter(IndexCatalog.host == host)
            if project is not None:
                query = query.filter(IndexCatalog.project == project)
            if kind is not None:
                query = query.filter(IndexCatalog.kind == kind)
            return query

        with self.session() as session:
            groups = _scoped(session.query(
                *group_cols,
                func.count(IndexCatalog.id).label("file_count"),
                func.coalesce(func.sum(IndexCatalog.chunk_count), 0).label("chunk_count"),
                func.max(IndexCatalog.indexed_at).label("last_indexed_at"),
            )).group_by(*group_cols).subquery("g")

            # SQLite lower() is ASCII-only, so a non-ASCII needle cannot be pushed down
            # faithfully; it just computes winners for every project.
            winners_filter = None
            if winners_project_contains and winners_project_contains.isascii():
                winners_filter = func.instr(
                    func.lower(func.replace(groups.c.project, "\\", "/")),
                    winners_project_contains) > 0

            def winner_path(*, freshest: bool):
                """rel_path of one group's winner row, read by an index seek per group.

                Freshest: among the live rows stamped with the group's newest ``indexed_at``
                (``IS`` so an all-NULL group still resolves, to its smallest path), the
                smallest rel_path. Otherwise the group's smallest rel_path."""
                row = aliased(IndexCatalog)
                query = session.query(row.rel_path).filter(
                    row.owner_user_id == groups.c.owner_user_id,
                    row.host == groups.c.host,
                    row.project == groups.c.project,
                    row.kind == groups.c.kind,
                    row.deleted_at.is_(None),
                )
                if freshest:
                    query = query.filter(row.indexed_at.is_not_distinct_from(groups.c.last_indexed_at))
                if winners_filter is not None:
                    query = query.filter(winners_filter)
                return query.correlate(groups).order_by(row.rel_path.asc()).limit(1).scalar_subquery()

            group_rows = session.query(
                groups,
                winner_path(freshest=True).label("fresh_rel_path"),
                winner_path(freshest=False).label("first_rel_path"),
            ).order_by(
                # Cheap: sorts only the ~(project x kind) aggregated result rows, not the
                # underlying file table — kept so SUMMARY node ordering matches DETAIL's
                # (owner, host, project) ordering for a stable admin-console/tool display.
                groups.c.owner_user_id, groups.c.host, groups.c.project, groups.c.kind,
            ).all()

            # root_dir and metadata of the winner rows only (two unique-index probes per
            # project): the freshest row's metadata is deserialised for those rows alone.
            wanted = {
                (g.owner_user_id, g.host, g.project, path, g.kind)
                for g in group_rows
                for path in (g.fresh_rel_path, g.first_rel_path) if path is not None
            }
            winner_rows: dict[tuple, tuple[Any, Any]] = {}
            for batch in _chunks(sorted(wanted), 150):
                for owner, host_name, project_name, path, row_kind, root_dir, metadata in session.query(
                    IndexCatalog.owner_user_id, IndexCatalog.host, IndexCatalog.project,
                    IndexCatalog.rel_path, IndexCatalog.kind,
                    IndexCatalog.root_dir, IndexCatalog.metadata_json,
                ).filter(tuple_(
                    IndexCatalog.owner_user_id, IndexCatalog.host, IndexCatalog.project,
                    IndexCatalog.rel_path, IndexCatalog.kind,
                ).in_(batch)):
                    winner_rows[(owner, host_name, project_name, path, row_kind)] = (
                        root_dir, metadata)

        rows: list[dict[str, Any]] = []
        for g in group_rows:
            def winner(path):
                return winner_rows.get(
                    (g.owner_user_id, g.host, g.project, path, g.kind), (None, None))
            meta = winner(g.fresh_rel_path)[1] if g.fresh_rel_path is not None else None
            rows.append({
                "owner_user_id": g.owner_user_id,
                "host": g.host,
                "project": g.project,
                "kind": g.kind,
                "root_dir": winner(g.first_rel_path)[0] if g.first_rel_path is not None else None,
                "last_commit": (meta or {}).get("git_commit"),
                # The winning row's path, so the caller can merge these per-kind rows
                # into one project node using DETAIL's cross-kind tie-break (smallest
                # rel_path wins an indexed_at tie) rather than kind order.
                "last_commit_rel_path": g.fresh_rel_path,
                "file_count": int(g.file_count or 0),
                "chunk_count": int(g.chunk_count or 0),
                "last_indexed_at": g.last_indexed_at,
            })
        return rows

    def list_projects(
        self,
        *,
        owner_user_id: str,
        host: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Distinct (project, host, root_dir) for an owner — the project list for the
        nested-scope overlap check. Returns ~project-count rows of bare string columns (no
        ORM entities, no metadata_json/content_hash deserialisation, no ORDER BY), not the
        O(all files) per-file tree the guard used to download; the caller then loads a
        single project's files only for the rare ancestor candidates."""
        with self.session() as session:
            query = (
                session.query(
                    IndexCatalog.project, IndexCatalog.host, IndexCatalog.root_dir,
                )
                .filter(
                    IndexCatalog.owner_user_id == owner_user_id,
                    IndexCatalog.deleted_at.is_(None),
                )
            )
            if host is not None:
                query = query.filter(IndexCatalog.host == host)
            if kind is not None:
                query = query.filter(IndexCatalog.kind == kind)
            rows = query.distinct().all()
            return [{"project": r[0], "host": r[1], "root_dir": r[2]} for r in rows]

    def list_owners(self) -> list[dict[str, Any]]:
        with self.session() as session:
            rows = (
                session.query(
                    IndexCatalog.owner_user_id,
                    func.count(IndexCatalog.id).label("file_count"),
                    func.coalesce(func.sum(IndexCatalog.chunk_count), 0).label("chunk_count"),
                    func.count(func.distinct(IndexCatalog.host)).label("host_count"),
                    func.count(func.distinct(IndexCatalog.project)).label("project_count"),
                )
                .filter(IndexCatalog.deleted_at.is_(None))
                .group_by(IndexCatalog.owner_user_id)
                .order_by(IndexCatalog.owner_user_id)
                .all()
            )
            return [
                {
                    "owner_user_id": r[0],
                    "file_count": int(r[1] or 0),
                    "chunk_count": int(r[2] or 0),
                    "host_count": int(r[3] or 0),
                    "project_count": int(r[4] or 0),
                }
                for r in rows
            ]

    def delete_files(
        self,
        *,
        owner_user_id: str,
        project: str,
        host: str | None = None,
        kind: str | None = None,
        rel_paths: list[str] | None = None,
    ) -> int:
        """Soft-delete a project's catalog rows. host=None spans EVERY host (code vectors
        are host-agnostic — W148 — so clearing a project must sweep all its hosts); kind
        filters code vs doc so clearing a code index leaves the dir's journal docs (W151)."""
        with self.session() as session:
            query = session.query(IndexCatalog).filter(
                IndexCatalog.owner_user_id == owner_user_id,
                IndexCatalog.project == project,
                IndexCatalog.deleted_at.is_(None),
            )
            if host is not None:
                query = query.filter(IndexCatalog.host == host)
            if kind is not None:
                query = query.filter(IndexCatalog.kind == kind)
            now = _utcnow()
            if rel_paths is None:
                return query.update({IndexCatalog.deleted_at: now}, synchronize_session=False)
            # Distinct chunks never overlap, and rows a chunk already deleted no longer
            # match deleted_at IS NULL, so a repeated path is not counted twice.
            return sum(
                query.filter(IndexCatalog.rel_path.in_(chunk)).update(
                    {IndexCatalog.deleted_at: now}, synchronize_session=False)
                for chunk in _chunks(list(rel_paths)))

    def delete_policy(
        self,
        *,
        owner_user_id: str,
        project: str,
        host: str | None = None,
        kind: str | None = None,
    ) -> int:
        with self.session() as session:
            return self._policy_query(
                session, owner_user_id=owner_user_id, project=project,
                host=host, kind=kind).delete(synchronize_session=False)

    def overview(self) -> dict[str, int]:
        with self.session() as session:
            row = (
                session.query(
                    func.count(IndexCatalog.id),
                    func.count(func.distinct(IndexCatalog.owner_user_id)),
                    func.count(func.distinct(IndexCatalog.host)),
                    func.count(func.distinct(IndexCatalog.project)),
                    func.coalesce(func.sum(IndexCatalog.chunk_count), 0),
                )
                .filter(IndexCatalog.deleted_at.is_(None))
                .one()
            )
            return {
                "files": int(row[0] or 0),
                "users": int(row[1] or 0),
                "hosts": int(row[2] or 0),
                "projects": int(row[3] or 0),
                "chunks": int(row[4] or 0),
            }
