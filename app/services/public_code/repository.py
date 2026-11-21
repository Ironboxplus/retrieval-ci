"""SQL repository for public-code sources, checks, and revision promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Callable, Iterable

from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError

from app.services.public_code.source_catalog import PublicCodeSource, PublicCodeSourceCatalog
from app.services.public_code.update_checks import PublicCodeUpdateCheck
from core.db.database import get_db_session
from core.db.models import PublicCodeJobRecord, PublicCodeRevision, PublicCodeSourceRecord
from core.indexing.revision_manifest import RevisionManifest


_UPDATE_POLICIES = {"manual", "scheduled"}


@dataclass(frozen=True, slots=True)
class PublicCodeSourceState:
    """Source state read by API, front-end, update scheduler, and indexing jobs."""

    source_id: str
    domain: str
    relative_path: str
    upstream_url: str
    default_ref: str
    proxy_profile: str | None
    credential_profile: str | None
    language_overrides: dict[str, str]
    source_origin: str
    created_by_user_id: str | None
    active_manifest_file_count: int
    active_chunk_count: int
    active_vector_count: int
    active_promoted_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None
    update_policy: str
    update_interval_minutes: int | None
    active_revision: str | None
    last_mirrored_revision: str | None
    last_remote_revision: str | None
    last_check_error: str | None
    last_checked_at: datetime | None


@dataclass(frozen=True, slots=True)
class PublicCodeSearchScope:
    source_ids: tuple[str, ...]
    active_revisions: dict[str, str]


class PublicCodeRepository:
    """The only SQL boundary for public-code catalog and revision facts."""

    def __init__(self, session_factory: Callable[[], Any] = get_db_session):
        self._session_factory = session_factory

    def reconcile_catalog(self, catalog: PublicCodeSourceCatalog) -> tuple[PublicCodeSourceState, ...]:
        """Upsert checked-in source metadata without altering operator-owned policy."""

        with self._session_factory() as session:
            for source in catalog.sources:
                record = session.get(PublicCodeSourceRecord, source.source_id)
                if record is None:
                    record = PublicCodeSourceRecord(
                        source_id=source.source_id,
                        domain=source.domain,
                        relative_path=source.relative_path,
                        upstream_url=source.upstream_url,
                        default_ref=source.default_ref,
                        proxy_profile=source.proxy_profile,
                        credential_profile=source.credential_profile,
                        language_overrides=dict(source.language_overrides),
                        source_origin="bootstrap",
                    )
                    session.add(record)
                    continue
                record.domain = source.domain
                record.relative_path = source.relative_path
                record.upstream_url = source.upstream_url
                record.default_ref = source.default_ref
                record.proxy_profile = source.proxy_profile
                record.credential_profile = source.credential_profile
                record.language_overrides = dict(source.language_overrides)
            session.flush()
            rows = (
                _source_state_query(session)
                .filter(PublicCodeSourceRecord.source_id.in_([source.source_id for source in catalog.sources]))
                .order_by(PublicCodeSourceRecord.source_id.asc())
                .all()
            )
            return tuple(_source_state(record, revision) for record, revision in rows)

    def create_source(
        self,
        source: PublicCodeSource,
        *,
        update_policy: str,
        update_interval_minutes: int | None,
        created_by_user_id: str | None = None,
    ) -> PublicCodeSourceState:
        """Register one operator-managed source without mutating the bootstrap catalog."""

        if update_policy not in _UPDATE_POLICIES:
            raise ValueError(f"unsupported public-code update policy: {update_policy}")
        update_interval_minutes = _validated_update_interval(
            update_policy,
            update_interval_minutes,
        )
        with self._session_factory() as session:
            if session.get(PublicCodeSourceRecord, source.source_id) is not None:
                raise ValueError(f"public-code source already exists: {source.source_id}")
            path_owner = (
                session.query(PublicCodeSourceRecord.source_id)
                .filter_by(relative_path=source.relative_path)
                .one_or_none()
            )
            if path_owner is not None:
                raise ValueError(
                    "public-code relative path already exists: "
                    f"{source.relative_path} ({path_owner[0]})"
                )
            record = PublicCodeSourceRecord(
                source_id=source.source_id,
                domain=source.domain,
                relative_path=source.relative_path,
                upstream_url=source.upstream_url,
                default_ref=source.default_ref,
                proxy_profile=source.proxy_profile,
                credential_profile=source.credential_profile,
                language_overrides=dict(source.language_overrides),
                source_origin="api",
                created_by_user_id=created_by_user_id,
                update_policy=update_policy,
                update_interval_minutes=update_interval_minutes,
            )
            session.add(record)
            try:
                session.flush()
            except IntegrityError as exc:
                raise ValueError(
                    "public-code source or relative path already exists"
                ) from exc
            return _source_state(record, None)

    @property
    def sources(self) -> tuple[PublicCodeSource, ...]:
        return self.list_source_definitions()

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({source.domain for source in self.list_source_definitions()}))

    def list_source_definitions(self) -> tuple[PublicCodeSource, ...]:
        with self._session_factory() as session:
            records = (
                session.query(PublicCodeSourceRecord)
                .order_by(PublicCodeSourceRecord.source_id.asc())
                .all()
            )
            return tuple(_source_definition(record) for record in records)

    def select(
        self,
        *,
        source_ids: Iterable[str] = (),
        domains: Iterable[str] = (),
    ) -> tuple[PublicCodeSource, ...]:
        catalog = PublicCodeSourceCatalog(
            catalog_id="public-code-runtime-registry",
            sources=self.list_source_definitions(),
        )
        return catalog.select(source_ids=source_ids, domains=domains)

    def get_source(self, source_id: str) -> PublicCodeSourceState | None:
        with self._session_factory() as session:
            row = (
                _source_state_query(session)
                .filter(PublicCodeSourceRecord.source_id == source_id)
                .one_or_none()
            )
            return _source_state(*row) if row is not None else None

    def list_sources(self) -> tuple[PublicCodeSourceState, ...]:
        with self._session_factory() as session:
            rows = _source_state_query(session).order_by(PublicCodeSourceRecord.source_id.asc()).all()
            return tuple(_source_state(record, revision) for record, revision in rows)

    def active_revisions(self, source_ids: Iterable[str]) -> dict[str, str]:
        """Return only promoted revisions for the caller's selected source IDs."""

        selected_source_ids = tuple(source_ids)
        if not selected_source_ids:
            return {}
        with self._session_factory() as session:
            rows = (
                session.query(PublicCodeSourceRecord.source_id, PublicCodeSourceRecord.active_revision)
                .filter(
                    PublicCodeSourceRecord.source_id.in_(selected_source_ids),
                    PublicCodeSourceRecord.active_revision.is_not(None),
                )
                .order_by(PublicCodeSourceRecord.source_id.asc())
                .all()
            )
            return {source_id: revision for source_id, revision in rows if revision}

    def resolve_search_scope(
        self,
        *,
        source_ids: Iterable[str] = (),
        domains: Iterable[str] = (),
    ) -> PublicCodeSearchScope:
        """Resolve source selection and active revisions with one projected SQL query."""

        requested_source_ids = frozenset(source_ids)
        requested_domains = frozenset(domains)
        with self._session_factory() as session:
            rows = (
                session.query(
                    PublicCodeSourceRecord.source_id,
                    PublicCodeSourceRecord.domain,
                    PublicCodeSourceRecord.active_revision,
                )
                .order_by(PublicCodeSourceRecord.source_id.asc())
                .all()
            )
        known_source_ids = {source_id for source_id, _domain, _revision in rows}
        known_domains = {domain for _source_id, domain, _revision in rows}
        unknown_source_ids = sorted(requested_source_ids - known_source_ids)
        if unknown_source_ids:
            raise ValueError(
                f"Unknown public-code source IDs: {', '.join(unknown_source_ids)}"
            )
        unknown_domains = sorted(requested_domains - known_domains)
        if unknown_domains:
            raise ValueError(f"Unknown public-code domains: {', '.join(unknown_domains)}")

        selected_rows = rows
        if requested_source_ids or requested_domains:
            selected_rows = [
                row
                for row in rows
                if row[0] in requested_source_ids or row[1] in requested_domains
            ]
        selected_source_ids = tuple(row[0] for row in selected_rows)
        active_revisions = {
            source_id: revision
            for source_id, _domain, revision in selected_rows
            if revision
        }
        return PublicCodeSearchScope(
            source_ids=selected_source_ids,
            active_revisions=active_revisions,
        )

    def set_update_policy(
        self,
        *,
        source_id: str,
        update_policy: str,
        update_interval_minutes: int | None,
    ) -> PublicCodeSourceState:
        """Persist a supported operator update policy."""

        if update_policy not in _UPDATE_POLICIES:
            raise ValueError(f"unsupported public-code update policy: {update_policy}")
        update_interval_minutes = _validated_update_interval(
            update_policy,
            update_interval_minutes,
        )

        with self._session_factory() as session:
            record = _require_source(session, source_id)
            record.update_policy = update_policy
            record.update_interval_minutes = update_interval_minutes
            session.flush()
            return _source_state(record, _active_revision(session, record))

    def record_update_checks(
        self,
        checks: Iterable[PublicCodeUpdateCheck],
        *,
        checked_at: datetime | None = None,
    ) -> None:
        """Store observed remote/mirror state after a read-only check operation."""

        now = checked_at or _utcnow()
        with self._session_factory() as session:
            for check in checks:
                record = _require_source(session, check.source_id)
                record.last_mirrored_revision = check.local_revision
                record.last_remote_revision = check.remote_revision
                record.last_check_error = check.message if check.status == "check_failed" else None
                record.last_checked_at = now

    def list_due_scheduled_source_ids(self, *, now: datetime | None = None) -> tuple[str, ...]:
        """Return scheduled sources whose persisted check interval has elapsed."""

        current_time = now or _utcnow()
        with self._session_factory() as session:
            records = (
                session.query(PublicCodeSourceRecord)
                .filter_by(update_policy="scheduled")
                .order_by(PublicCodeSourceRecord.source_id.asc())
                .all()
            )
            due_source_ids = []
            for record in records:
                if record.update_interval_minutes is None:
                    continue
                if (
                    record.last_remote_revision
                    and record.active_revision != record.last_remote_revision
                ):
                    due_source_ids.append(record.source_id)
                    continue
                if record.last_checked_at is None:
                    due_source_ids.append(record.source_id)
                    continue
                next_check_at = record.last_checked_at + timedelta(
                    minutes=record.update_interval_minutes
                )
                if next_check_at <= current_time:
                    due_source_ids.append(record.source_id)
            return tuple(due_source_ids)

    def record_candidate_revision(
        self,
        source_id: str,
        manifest: RevisionManifest,
        *,
        chunk_count: int,
    ) -> None:
        """Persist the candidate facts before vector promotion can make them active."""

        if chunk_count < 0:
            raise ValueError("chunk_count must not be negative")
        with self._session_factory() as session:
            _require_source(session, source_id)
            revision = (
                session.query(PublicCodeRevision)
                .filter_by(source_id=source_id, revision=manifest.revision_id)
                .one_or_none()
            )
            if revision is None:
                revision = PublicCodeRevision(
                    source_id=source_id,
                    revision=manifest.revision_id,
                    state="candidate",
                    manifest_file_count=len(manifest.files),
                    chunk_count=chunk_count,
                    cleanup_pending=False,
                )
                session.add(revision)
                return
            if revision.state == "active":
                return
            revision.state = "candidate"
            revision.manifest_file_count = len(manifest.files)
            revision.chunk_count = chunk_count
            revision.vector_count = 0
            revision.cleanup_pending = False
            revision.error_message = None

    def begin_candidate_revision(
        self,
        source_id: str,
        manifest: RevisionManifest,
    ) -> None:
        """Persist the in-progress candidate before the first vector write."""

        with self._session_factory() as session:
            _require_source(session, source_id)
            revision = (
                session.query(PublicCodeRevision)
                .filter_by(source_id=source_id, revision=manifest.revision_id)
                .one_or_none()
            )
            if revision is None:
                revision = PublicCodeRevision(
                    source_id=source_id,
                    revision=manifest.revision_id,
                    state="candidate",
                    manifest_file_count=len(manifest.files),
                    chunk_count=0,
                    vector_count=0,
                    cleanup_pending=False,
                )
                session.add(revision)
                return
            if revision.state == "active":
                return
            revision.state = "candidate"
            revision.manifest_file_count = len(manifest.files)
            revision.chunk_count = 0
            revision.vector_count = 0
            revision.cleanup_pending = False
            revision.error_message = None

    def mark_abandoned_candidates_failed(self, *, reason: str) -> tuple[tuple[str, str], ...]:
        """Recover candidates left by a previous backend process before jobs restart."""

        with self._session_factory() as session:
            revisions = (
                session.query(PublicCodeRevision)
                .filter(PublicCodeRevision.state == "candidate")
                .order_by(PublicCodeRevision.source_id.asc(), PublicCodeRevision.id.asc())
                .all()
            )
            recovered: list[tuple[str, str]] = []
            for revision in revisions:
                revision.state = "failed"
                revision.cleanup_pending = True
                revision.error_message = reason
                recovered.append((revision.source_id, revision.revision))
            return tuple(recovered)

    def validate_candidate_revision(
        self,
        source_id: str,
        revision_name: str,
        *,
        vector_count: int,
    ) -> None:
        """Mark a candidate validated only when Qdrant exactly matches its chunk count."""

        if vector_count < 0:
            raise ValueError("vector_count must not be negative")
        with self._session_factory() as session:
            _require_source(session, source_id)
            revision = _require_revision(session, source_id, revision_name)
            if revision.state not in {"candidate", "validated", "active"}:
                raise ValueError(
                    f"revision is not validatable for {source_id}: {revision_name} ({revision.state})"
                )
            if revision.chunk_count != vector_count:
                raise ValueError(
                    "vector count mismatch for "
                    f"{source_id}:{revision_name}: expected {revision.chunk_count}, got {vector_count}"
                )
            revision.vector_count = vector_count
            if revision.state != "active":
                revision.state = "validated"

    def promote_revision(self, source_id: str, revision_name: str) -> str | None:
        """Activate a validated revision and return the revision it superseded."""

        with self._session_factory() as session:
            source = _require_source(session, source_id)
            revision = _require_revision(session, source_id, revision_name)
            if revision.state not in {"validated", "active"}:
                raise ValueError(
                    f"revision must be validated before promotion for {source_id}: "
                    f"{revision_name} ({revision.state})"
                )

            previous_revision = (
                source.active_revision
                if source.active_revision and source.active_revision != revision_name
                else None
            )
            if previous_revision:
                previous = (
                    session.query(PublicCodeRevision)
                    .filter_by(source_id=source_id, revision=previous_revision)
                    .one_or_none()
                )
                if previous is not None:
                    previous.state = "superseded"
                    previous.cleanup_pending = True
            revision.state = "active"
            revision.cleanup_pending = False
            revision.promoted_at = _utcnow()
            source.active_revision = revision_name
            return previous_revision

    def mark_revision_failed(
        self,
        source_id: str,
        revision_name: str,
        *,
        error_message: str,
        cleanup_pending: bool = True,
    ) -> None:
        """Persist a failed candidate without changing the source's active revision."""

        with self._session_factory() as session:
            _require_source(session, source_id)
            revision = (
                session.query(PublicCodeRevision)
                .filter_by(source_id=source_id, revision=revision_name)
                .one_or_none()
            )
            if revision is None:
                revision = PublicCodeRevision(
                    source_id=source_id,
                    revision=revision_name,
                    state="failed",
                    manifest_file_count=0,
                    chunk_count=0,
                    vector_count=0,
                    cleanup_pending=cleanup_pending,
                    error_message=error_message,
                )
                session.add(revision)
                return
            if revision.state == "active":
                raise ValueError(f"active revision cannot be marked failed: {source_id}:{revision_name}")
            revision.state = "failed"
            revision.cleanup_pending = cleanup_pending
            revision.error_message = error_message

    def list_cleanup_revisions(self) -> tuple[tuple[str, str], ...]:
        with self._session_factory() as session:
            rows = (
                session.query(PublicCodeRevision.source_id, PublicCodeRevision.revision)
                .filter(PublicCodeRevision.cleanup_pending.is_(True))
                .order_by(PublicCodeRevision.source_id.asc(), PublicCodeRevision.id.asc())
                .all()
            )
            return tuple(rows)

    def mark_revision_cleaned(self, source_id: str, revision_name: str) -> None:
        with self._session_factory() as session:
            revision = _require_revision(session, source_id, revision_name)
            revision.cleanup_pending = False
            if revision.state == "candidate":
                revision.state = "failed"
                revision.error_message = (
                    revision.error_message
                    or "incomplete candidate recovered by durable cleanup"
                )

    def save_job_snapshot(self, snapshot: dict[str, Any], *, max_retained_jobs: int) -> None:
        with self._session_factory() as session:
            record = session.get(PublicCodeJobRecord, snapshot["job_id"])
            if record is None:
                record = PublicCodeJobRecord(job_id=snapshot["job_id"])
                session.add(record)
            record.operation = snapshot["operation"]
            record.source_ids = list(snapshot["source_ids"])
            record.status = snapshot["status"]
            record.snapshot = dict(snapshot)
            session.flush()
            terminal_records = (
                session.query(PublicCodeJobRecord)
                .filter(PublicCodeJobRecord.status.in_(("completed", "failed")))
                .order_by(PublicCodeJobRecord.updated_at.desc(), PublicCodeJobRecord.created_at.desc())
                .all()
            )
            for stale_record in terminal_records[max_retained_jobs:]:
                session.delete(stale_record)

    def get_job_snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            record = session.get(PublicCodeJobRecord, job_id)
            return dict(record.snapshot) if record is not None else None

    def list_job_snapshots(self, *, limit: int) -> tuple[dict[str, Any], ...]:
        with self._session_factory() as session:
            records = (
                session.query(PublicCodeJobRecord)
                .order_by(PublicCodeJobRecord.updated_at.desc(), PublicCodeJobRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return tuple(dict(record.snapshot) for record in records)

    def fail_incomplete_jobs(self, *, reason: str) -> None:
        with self._session_factory() as session:
            records = (
                session.query(PublicCodeJobRecord)
                .filter(PublicCodeJobRecord.status.in_(("queued", "running")))
                .all()
            )
            for record in records:
                snapshot = dict(record.snapshot)
                snapshot["status"] = "failed"
                snapshot["phase"] = "failed"
                snapshot["message"] = reason
                snapshot["errors"] = {"job": reason}
                snapshot["finished_at"] = time.time()
                record.status = "failed"
                record.snapshot = snapshot

    def get_revision_state(self, source_id: str, revision_name: str) -> str | None:
        with self._session_factory() as session:
            revision = (
                session.query(PublicCodeRevision.state)
                .filter_by(source_id=source_id, revision=revision_name)
                .one_or_none()
            )
            return revision[0] if revision is not None else None


def _source_state(
    record: PublicCodeSourceRecord,
    revision: PublicCodeRevision | None,
) -> PublicCodeSourceState:
    return PublicCodeSourceState(
        source_id=record.source_id,
        domain=record.domain,
        relative_path=record.relative_path,
        upstream_url=record.upstream_url,
        default_ref=record.default_ref or "HEAD",
        proxy_profile=record.proxy_profile,
        credential_profile=record.credential_profile,
        language_overrides=dict(record.language_overrides or {}),
        source_origin=record.source_origin or "bootstrap",
        created_by_user_id=record.created_by_user_id,
        active_manifest_file_count=revision.manifest_file_count if revision is not None else 0,
        active_chunk_count=revision.chunk_count if revision is not None else 0,
        active_vector_count=revision.vector_count if revision is not None else 0,
        active_promoted_at=revision.promoted_at if revision is not None else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
        update_policy=record.update_policy,
        update_interval_minutes=record.update_interval_minutes,
        active_revision=record.active_revision,
        last_mirrored_revision=record.last_mirrored_revision,
        last_remote_revision=record.last_remote_revision,
        last_check_error=record.last_check_error,
        last_checked_at=record.last_checked_at,
    )


def _source_state_query(session: Any):
    return session.query(PublicCodeSourceRecord, PublicCodeRevision).outerjoin(
        PublicCodeRevision,
        and_(
            PublicCodeRevision.source_id == PublicCodeSourceRecord.source_id,
            PublicCodeRevision.revision == PublicCodeSourceRecord.active_revision,
        ),
    )


def _active_revision(
    session: Any,
    source: PublicCodeSourceRecord,
) -> PublicCodeRevision | None:
    if not source.active_revision:
        return None
    return (
        session.query(PublicCodeRevision)
        .filter_by(source_id=source.source_id, revision=source.active_revision)
        .one_or_none()
    )


def _source_definition(record: PublicCodeSourceRecord) -> PublicCodeSource:
    return PublicCodeSource(
        source_id=record.source_id,
        domain=record.domain,
        relative_path=record.relative_path,
        upstream_url=record.upstream_url,
        default_ref=record.default_ref or "HEAD",
        proxy_profile=record.proxy_profile,
        credential_profile=record.credential_profile,
        language_overrides=dict(record.language_overrides or {}),
    )


def _require_source(session: Any, source_id: str) -> PublicCodeSourceRecord:
    source = session.get(PublicCodeSourceRecord, source_id)
    if source is None:
        raise ValueError(f"unknown public-code source: {source_id}")
    return source


def _require_revision(session: Any, source_id: str, revision_name: str) -> PublicCodeRevision:
    revision = (
        session.query(PublicCodeRevision)
        .filter_by(source_id=source_id, revision=revision_name)
        .one_or_none()
    )
    if revision is None:
        raise ValueError(f"candidate revision does not exist for {source_id}: {revision_name}")
    return revision


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _validated_update_interval(
    update_policy: str,
    update_interval_minutes: int | None,
) -> int | None:
    if update_policy == "manual":
        return None
    if update_interval_minutes is None:
        raise ValueError("scheduled update policy requires update_interval_minutes")
    if not 1 <= update_interval_minutes <= 43_200:
        raise ValueError("update_interval_minutes must be between 1 and 43200")
    return update_interval_minutes
