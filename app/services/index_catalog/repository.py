"""SQLAlchemy repository for the index-inventory catalog."""
from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.db.database import get_db_session
from core.db.models import CodeScopeVisibility, IndexCatalog, IndexScopePolicy
from app.services.scope_paths import is_windows_scope


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _within_sweep_envelope(rel_path: str, patterns: list[str] | None,
                           max_depth: int | None) -> bool:
    """Is this catalog row inside the reporting run's scan envelope, hence eligible for the
    replace-sweep? Mirrors the journal stale-doc reconcile envelope (W169b): a row is swept
    only if its depth (slash count) is within ``max_depth`` AND its basename matches one of
    ``patterns``. None on an axis = no bound there (code reports pass neither → whole kind,
    unchanged). Basename matched case-sensitively (fnmatchcase) so a casing mismatch errs
    toward KEEPING the row, never sweeping a file outside the glob."""
    norm = rel_path.replace("\\", "/")
    if max_depth is not None and norm.count("/") > max_depth:
        return False
    globs = [g for g in (patterns or []) if g]
    if globs and globs != ["*"]:
        base = norm.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatchcase(base, g) for g in globs)
    return True


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
        with self.session() as session:
            existing = {
                (fold(row.rel_path), row.kind): row
                for row in session.query(IndexCatalog).filter(
                    IndexCatalog.owner_user_id == owner_user_id,
                    IndexCatalog.host == host,
                    IndexCatalog.project == project,
                )
            }
            for f in files:
                kind = f.get("kind", "code")
                row = existing.get((fold(f["rel_path"]), kind))
                if row is None:
                    row = IndexCatalog(
                        owner_user_id=owner_user_id, host=host, project=project,
                        rel_path=f["rel_path"], kind=kind,
                    )
                    session.add(row)
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
            # Full-sync sweep: soft-delete files absent from a NON-EMPTY report, but
            # ONLY within the kinds this report actually covers — a code report must
            # not wipe the project's doc inventory (W151). An empty report never wipes
            # anything (that would let an accidental index on a source-less directory
            # silently erase a user's catalog); explicit removal goes through
            # delete_files().
            if replace and reported_keys:
                for (rel_path, kind), row in existing.items():
                    if (kind in reported_kinds and (rel_path, kind) not in reported_keys
                            and row.deleted_at is None
                            # ...and within THIS run's scan envelope — a "*.txt" doc report
                            # must not sweep the "*.md" rows it never scanned (W169b).
                            and _within_sweep_envelope(rel_path, patterns, max_depth)):
                        row.deleted_at = _utcnow()
                        removed += 1
        return upserted, removed

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
            rows = query.order_by(
                IndexCatalog.owner_user_id, IndexCatalog.host,
                IndexCatalog.project, IndexCatalog.rel_path,
            ).all()
            for row in rows:
                session.expunge(row)
            return rows

    def aggregate_inventory(
        self,
        *,
        owner_user_id: str | None = None,
        host: str | None = None,
        project: str | None = None,
        kind: str | None = None,
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
        We therefore pick the two winner rows explicitly with ROW_NUMBER(), which makes
        the tie-break deterministic and identical to DETAIL's. Cost stays
        ~(project x kind) rows returned and ``metadata_json`` is deserialised only for
        those winners — never for the full file population ``list_inventory()`` downloads.
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
            agg_rows = _scoped(session.query(
                *group_cols,
                func.count(IndexCatalog.id).label("file_count"),
                func.coalesce(func.sum(IndexCatalog.chunk_count), 0).label("chunk_count"),
                func.max(IndexCatalog.indexed_at).label("last_indexed_at"),
            )).group_by(*group_cols).order_by(
                # Cheap: sorts only the ~(project x kind) aggregated result rows, not the
                # underlying file table — kept so SUMMARY node ordering matches DETAIL's
                # (owner, host, project) ordering for a stable admin-console/tool display.
                IndexCatalog.owner_user_id, IndexCatalog.host,
                IndexCatalog.project, IndexCatalog.kind,
            ).all()

            # Two winners per group: the freshest row (for last_commit) and the
            # smallest-rel_path row (for root_dir). DESC puts SQLite's NULLs last, so a
            # row with a real indexed_at always outranks an unindexed one; if the whole
            # group is NULL the aggregate above reports last_indexed_at=None and the
            # service leaves last_commit unset, exactly as DETAIL does.
            partition = list(group_cols)
            inner = _scoped(session.query(
                *group_cols,
                IndexCatalog.root_dir,
                IndexCatalog.rel_path,
                IndexCatalog.metadata_json,
                func.row_number().over(
                    partition_by=partition,
                    order_by=(IndexCatalog.indexed_at.desc(), IndexCatalog.rel_path.asc()),
                ).label("rn_fresh"),
                func.row_number().over(
                    partition_by=partition,
                    order_by=IndexCatalog.rel_path.asc(),
                ).label("rn_path"),
            )).subquery()
            winners = session.query(inner).filter(
                (inner.c.rn_fresh == 1) | (inner.c.rn_path == 1)
            ).all()

        freshest: dict[tuple, tuple[Any, Any]] = {}
        first_path: dict[tuple, Any] = {}
        for w in winners:
            key = (w.owner_user_id, w.host, w.project, w.kind)
            if w.rn_fresh == 1:
                freshest[key] = (w.metadata_json, w.rel_path)
            if w.rn_path == 1:
                first_path[key] = w.root_dir
        rows: list[dict[str, Any]] = []
        for r in agg_rows:
            key = (r[0], r[1], r[2], r[3])
            meta, fresh_rel_path = freshest.get(key, (None, None))
            rows.append({
                "owner_user_id": r[0],
                "host": r[1],
                "project": r[2],
                "kind": r[3],
                "root_dir": first_path.get(key),
                "last_commit": (meta or {}).get("git_commit"),
                # The winning row's path, so the caller can merge these per-kind rows
                # into one project node using DETAIL's cross-kind tie-break (smallest
                # rel_path wins an indexed_at tie) rather than kind order.
                "last_commit_rel_path": fresh_rel_path,
                "file_count": int(r[4] or 0),
                "chunk_count": int(r[5] or 0),
                "last_indexed_at": r[6],
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
            if rel_paths is not None:
                query = query.filter(IndexCatalog.rel_path.in_(rel_paths))
            count = 0
            for row in query:
                row.deleted_at = _utcnow()
                count += 1
            return count

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
