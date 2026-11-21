"""Persistence for the background duplicate-code analysis (app/services/code_kb/analysis.py).

Three questions live here, all answered from SQLite so they survive restarts:

* what changed in a scope since the last analysis (``changed_files``): read from the
  index catalog's ``content_changed_at`` / ``deleted_at`` against the scope's watermark,
  so any number of index runs merge into one answer and nothing needs to be remembered
  in process;
* which duplicate file pairs currently stand (``code_duplicate_pairs``), maintained
  incrementally by replacing every pair that mentions a touched file;
* where the analysis stands (``code_scope_analyses``): watermark, the version of the
  latest result worth reporting, and how much of it was already delivered.

Keys are (owner, canonical scope). Callers resolve scope spellings; this module only
stores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence

from sqlalchemy import or_

from app.services.scope_paths import canonicalize_scope_path
from core.code_kb.duplicate_rollup import DuplicateFilePair
from core.db.database import get_db_session
from core.db.models import CodeDuplicatePair, CodeScopeAnalysis, IndexCatalog

logger = logging.getLogger("code_kb.analysis_store")

KIND_DUPLICATES = "duplicates"
# SQLite caps bound parameters per statement (999 on older builds).
_IN_CHUNK = 400


def _chunks(values: Sequence[str]) -> Iterable[list[str]]:
    for start in range(0, len(values), _IN_CHUNK):
        yield list(values[start:start + _IN_CHUNK])


@dataclass(frozen=True)
class AnalysisState:
    watermark: datetime | None
    version: int
    delivered_version: int
    last_run_at: datetime | None
    summary: dict | None


@dataclass(frozen=True)
class ChangedFiles:
    """Files of one scope that changed after a watermark.

    ``newest`` is the latest change timestamp seen: the next watermark. It is taken from
    the rows themselves rather than the clock, so a change committed a moment before the
    read can never be skipped by a clock that is slightly ahead."""
    changed: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    newest: datetime | None = None

    def __bool__(self) -> bool:
        return bool(self.changed or self.removed)


@dataclass(frozen=True)
class StoredPair:
    path_a: str
    path_b: str
    shared_chunks: int
    max_score: float
    first_seen_at: datetime
    examples: list[dict]


def _example_json(example: Any) -> dict:
    def chunk(ref: Any) -> dict:
        return {"path": ref.path, "start_line": ref.start_line,
                "end_line": ref.end_line, "label": ref.label}
    return {"a": chunk(example.a), "b": chunk(example.b), "score": example.score}


def pair_to_json(pair: DuplicateFilePair) -> dict:
    """A file pair as stored and as shown on the Tasks page."""
    return {"path_a": pair.path_a, "path_b": pair.path_b,
            "shared_chunks": pair.shared_chunks, "max_score": pair.max_score,
            "examples": [_example_json(example) for example in pair.examples]}


def _stored(row: CodeDuplicatePair) -> StoredPair:
    return StoredPair(
        path_a=row.path_a, path_b=row.path_b, shared_chunks=row.shared_chunks,
        max_score=row.max_score, first_seen_at=row.first_seen_at,
        examples=list(row.examples_json or []))


class CodeAnalysisStore:
    def __init__(self, session_factory: Callable[[], Any] = get_db_session):
        self._session_factory = session_factory

    # ------------------------------------------------------------------ state
    def get_state(self, owner_user_id: str, project: str,
                  kind: str = KIND_DUPLICATES) -> AnalysisState | None:
        with self._session_factory() as session:
            row = session.query(CodeScopeAnalysis).filter_by(
                owner_user_id=owner_user_id, project=project, kind=kind).one_or_none()
            if row is None:
                return None
            return AnalysisState(
                watermark=row.watermark, version=row.version,
                delivered_version=row.delivered_version, last_run_at=row.last_run_at,
                summary=dict(row.summary_json) if row.summary_json else None)

    # ---------------------------------------------------------------- changes
    def changed_files(
        self, owner_user_id: str, projects: Sequence[str], since: datetime | None,
    ) -> ChangedFiles:
        """Code files of any spelling of the scope that changed after ``since``.

        ``since=None`` is the first analysis of the scope: it has no stored pairs, so a
        removed file has nothing to clean up and only live files with a recorded change
        are returned. Rows indexed before ``content_changed_at`` existed have no record
        and are not guessed at."""
        scope = list(dict.fromkeys(projects))
        base = (
            IndexCatalog.owner_user_id == owner_user_id,
            IndexCatalog.kind == "code",
            IndexCatalog.project.in_(scope),
        )
        with self._session_factory() as session:
            changed_filter = IndexCatalog.content_changed_at.isnot(None) if since is None \
                else IndexCatalog.content_changed_at > since
            changed_rows = session.query(
                IndexCatalog.rel_path, IndexCatalog.content_changed_at,
            ).filter(*base, IndexCatalog.deleted_at.is_(None), changed_filter).all()
            removed_rows = [] if since is None else session.query(
                IndexCatalog.rel_path, IndexCatalog.deleted_at,
            ).filter(*base, IndexCatalog.deleted_at > since).all()
        changed = {path: stamp for path, stamp in changed_rows}
        removed = {path: stamp for path, stamp in removed_rows}
        # A file deleted since it was last (re)reported is a removal, whatever else is true.
        for path in removed:
            changed.pop(path, None)
        stamps = [*changed.values(), *removed.values()]
        return ChangedFiles(
            changed=tuple(sorted(changed)), removed=tuple(sorted(removed)),
            newest=max(stamps) if stamps else None)

    # ------------------------------------------------------------------ pairs
    @staticmethod
    def _touching(session: Any, owner_user_id: str, project: str,
                  paths: Sequence[str]) -> list[CodeDuplicatePair]:
        rows: dict[int, CodeDuplicatePair] = {}
        for chunk in _chunks(list(dict.fromkeys(paths))):
            query = session.query(CodeDuplicatePair).filter(
                CodeDuplicatePair.owner_user_id == owner_user_id,
                CodeDuplicatePair.project == project,
                or_(CodeDuplicatePair.path_a.in_(chunk), CodeDuplicatePair.path_b.in_(chunk)))
            for row in query:
                rows[row.id] = row
        return list(rows.values())

    def pairs_touching(self, owner_user_id: str, project: str,
                       paths: Sequence[str]) -> dict[tuple[str, str], StoredPair]:
        with self._session_factory() as session:
            return {(row.path_a, row.path_b): _stored(row)
                    for row in self._touching(session, owner_user_id, project, paths)}

    def all_pairs(self, owner_user_id: str, project: str) -> list[StoredPair]:
        with self._session_factory() as session:
            rows = session.query(CodeDuplicatePair).filter_by(
                owner_user_id=owner_user_id, project=project,
            ).order_by(CodeDuplicatePair.path_a, CodeDuplicatePair.path_b).all()
            return [_stored(row) for row in rows]

    # -------------------------------------------------------------------- run
    def apply_run(
        self,
        owner_user_id: str,
        project: str,
        *,
        touched_paths: Sequence[str],
        pairs: Sequence[DuplicateFilePair],
        watermark: datetime | None,
        summary: dict | None,
        now: datetime,
    ) -> int:
        """Record one analysis in a single transaction and return the state's version.

        Every stored pair that mentions a touched file is replaced by ``pairs`` (the
        recomputed ones); a pair that existed before keeps its ``first_seen_at``. The
        watermark only moves forward. ``summary`` is a result worth reporting: it bumps
        the version and replaces the stored summary, while ``None`` leaves both alone, so
        a run that found nothing to say does not manufacture a notification."""
        with self._session_factory() as session:
            existing = self._touching(session, owner_user_id, project, touched_paths)
            first_seen = {(row.path_a, row.path_b): row.first_seen_at for row in existing}
            for row in existing:
                session.delete(row)
            session.flush()
            for pair in pairs:
                key = (pair.path_a, pair.path_b)
                session.add(CodeDuplicatePair(
                    owner_user_id=owner_user_id, project=project,
                    path_a=pair.path_a, path_b=pair.path_b,
                    shared_chunks=pair.shared_chunks, max_score=pair.max_score,
                    examples_json=[_example_json(example) for example in pair.examples],
                    first_seen_at=first_seen.get(key, now), updated_at=now))
            state = session.query(CodeScopeAnalysis).filter_by(
                owner_user_id=owner_user_id, project=project, kind=KIND_DUPLICATES,
            ).one_or_none()
            if state is None:
                state = CodeScopeAnalysis(
                    owner_user_id=owner_user_id, project=project, kind=KIND_DUPLICATES,
                    version=0, delivered_version=0)
                session.add(state)
            if watermark is not None and (state.watermark is None or watermark > state.watermark):
                state.watermark = watermark
            state.last_run_at = now
            if summary is not None:
                state.version = (state.version or 0) + 1
                state.summary_json = summary
            session.flush()
            return int(state.version or 0)

    def claim_undelivered(self, owner_user_id: str, project: str,
                          kind: str = KIND_DUPLICATES) -> str | None:
        """The stored summary if it was not delivered yet, claiming it for this caller.

        One indexed read answers "nothing to deliver" (the common case). Otherwise a
        conditional UPDATE decides who gets it: only the statement that moves
        ``delivered_version`` up to the version it read changes a row, so of any number
        of concurrent callers exactly one receives the text and the rest get None,
        whichever client or machine they are."""
        with self._session_factory() as session:
            row = session.query(
                CodeScopeAnalysis.version, CodeScopeAnalysis.delivered_version,
                CodeScopeAnalysis.summary_json,
            ).filter_by(owner_user_id=owner_user_id, project=project, kind=kind).first()
            if row is None:
                return None
            version, delivered, summary = row
            if (version or 0) <= (delivered or 0):
                return None
            claimed = session.query(CodeScopeAnalysis).filter(
                CodeScopeAnalysis.owner_user_id == owner_user_id,
                CodeScopeAnalysis.project == project,
                CodeScopeAnalysis.kind == kind,
                or_(CodeScopeAnalysis.delivered_version.is_(None),
                    CodeScopeAnalysis.delivered_version < version),
            ).update({CodeScopeAnalysis.delivered_version: version},
                     synchronize_session=False)
            text = (summary or {}).get("text") if isinstance(summary, dict) else None
            return text if claimed == 1 and text else None

    def clear(self, owner_user_id: str, project: str) -> None:
        with self._session_factory() as session:
            session.query(CodeDuplicatePair).filter_by(
                owner_user_id=owner_user_id, project=project).delete(synchronize_session=False)
            session.query(CodeScopeAnalysis).filter_by(
                owner_user_id=owner_user_id, project=project).delete(synchronize_session=False)


def claim_scope_notice(owner_user_id: str | None, scope: str | None, *,
                       store: CodeAnalysisStore | None = None) -> str | None:
    """The deliver-once summary for one scope, for a route to attach to its response.

    ``scope`` is the scope that actually holds the data (the resolved ancestor when a
    subdirectory search was redirected); its canonical spelling is the analysis key. An
    ambiguous call (no owner, no single scope) gets nothing. This sits on the search hot
    path, so it never raises: any failure is logged and reads as "nothing to deliver"."""
    if not owner_user_id or not scope:
        return None
    try:
        try:
            project = canonicalize_scope_path(scope) or scope
        except ValueError:
            project = scope  # not a path scope: the name as given
        return (store or CodeAnalysisStore()).claim_undelivered(owner_user_id, project)
    except Exception:  # noqa: BLE001 - never fail a search over a notification
        logger.warning("scope notice lookup failed scope=%s", scope, exc_info=True)
        return None
