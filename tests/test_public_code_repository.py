"""TDD for durable public-code source and revision state."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.public_code.repository import PublicCodeRepository
from app.services.public_code.source_catalog import PublicCodeSource, PublicCodeSourceCatalog
from app.services.public_code.update_checks import PublicCodeUpdateCheck
from core.db.database import Base
from core.indexing.revision_manifest import RevisionFile, RevisionManifest


def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        session = local_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return factory


def test_idempotent_migration_restores_public_code_cleanup_index(monkeypatch):
    from core.db import database

    migration_engine = create_engine("sqlite://")
    Base.metadata.create_all(migration_engine)
    with migration_engine.begin() as connection:
        connection.execute(text("DROP INDEX ix_public_code_revisions_cleanup_pending"))

    monkeypatch.setattr(database, "engine", migration_engine)
    database._apply_idempotent_migrations()

    with migration_engine.connect() as connection:
        indexes = {
            row[1]
            for row in connection.execute(text("PRAGMA index_list(public_code_revisions)"))
        }
    assert "ix_public_code_revisions_cleanup_pending" in indexes


def source_catalog(*, ffmpeg_url: str = "https://example.invalid/ffmpeg.git") -> PublicCodeSourceCatalog:
    return PublicCodeSourceCatalog(
        catalog_id="fixture",
        sources=(
            PublicCodeSource(
                source_id="ffmpeg",
                domain="multimedia",
                relative_path="multimedia/ffmpeg",
                upstream_url=ffmpeg_url,
            ),
            PublicCodeSource(
                source_id="cutlass",
                domain="gpu",
                relative_path="gpu/cutlass",
                upstream_url="https://example.invalid/cutlass.git",
            ),
        ),
    )


def manifest(revision: str) -> RevisionManifest:
    return RevisionManifest(
        revision_id=revision,
        files=(
            RevisionFile(
                path="libavcodec/sample.c",
                content_hash="blob-a",
                language="c",
                content_role="source",
                size_bytes=17,
            ),
        ),
    )


def test_catalog_reconciliation_updates_source_metadata_without_overwriting_policy():
    repository = PublicCodeRepository(session_factory=session_factory())

    repository.reconcile_catalog(source_catalog())
    repository.set_update_policy(
        source_id="ffmpeg",
        update_policy="manual",
        update_interval_minutes=None,
    )
    repository.reconcile_catalog(source_catalog(ffmpeg_url="https://example.invalid/new-ffmpeg.git"))

    ffmpeg = repository.get_source("ffmpeg")
    assert ffmpeg is not None
    assert ffmpeg.upstream_url == "https://example.invalid/new-ffmpeg.git"
    assert ffmpeg.update_policy == "manual"
    assert ffmpeg.update_interval_minutes is None


def test_scheduled_policy_requires_interval_and_becomes_due_from_persisted_check_time():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())

    with pytest.raises(ValueError, match="requires update_interval_minutes"):
        repository.set_update_policy(
            source_id="ffmpeg",
            update_policy="scheduled",
            update_interval_minutes=None,
        )

    state = repository.set_update_policy(
        source_id="ffmpeg",
        update_policy="scheduled",
        update_interval_minutes=60,
    )
    assert state.update_policy == "scheduled"
    assert state.update_interval_minutes == 60
    now = datetime(2026, 7, 19, 12, 0, 0)
    assert repository.list_due_scheduled_source_ids(now=now) == ("ffmpeg",)

    repository.record_candidate_revision("ffmpeg", manifest("same"), chunk_count=10)
    repository.validate_candidate_revision("ffmpeg", "same", vector_count=10)
    repository.promote_revision("ffmpeg", "same")

    repository.record_update_checks((
        PublicCodeUpdateCheck(
            source_id="ffmpeg",
            domain="multimedia",
            local_revision="same",
            remote_revision="same",
            status="up_to_date",
        ),
    ), checked_at=now)

    assert repository.list_due_scheduled_source_ids(now=now + timedelta(minutes=59)) == ()
    assert repository.list_due_scheduled_source_ids(now=now + timedelta(minutes=60)) == ("ffmpeg",)


def test_scheduled_source_with_unapplied_remote_revision_is_due_after_restart():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())
    repository.set_update_policy(
        source_id="ffmpeg",
        update_policy="scheduled",
        update_interval_minutes=43_200,
    )
    repository.record_candidate_revision("ffmpeg", manifest("old"), chunk_count=10)
    repository.validate_candidate_revision("ffmpeg", "old", vector_count=10)
    repository.promote_revision("ffmpeg", "old")
    now = datetime(2026, 7, 19, 12, 0, 0)
    repository.record_update_checks((
        PublicCodeUpdateCheck(
            source_id="ffmpeg",
            domain="multimedia",
            local_revision="old",
            remote_revision="new",
            status="update_available",
        ),
    ), checked_at=now)

    assert repository.list_due_scheduled_source_ids(now=now) == ("ffmpeg",)


def test_registered_source_is_immediately_selectable_without_restarting_the_catalog():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())
    source = PublicCodeSource(
        source_id="pion-webrtc",
        domain="multimedia",
        relative_path="multimedia/pion-webrtc",
        upstream_url="https://github.com/pion/webrtc.git",
        default_ref="main",
        proxy_profile="local-proxy",
        credential_profile="github-read",
        language_overrides={".inc": "c"},
    )

    state = repository.create_source(
        source,
        update_policy="manual",
        update_interval_minutes=None,
        created_by_user_id="gateway-admin",
    )

    assert state.source_id == "pion-webrtc"
    assert state.default_ref == "main"
    assert state.proxy_profile == "local-proxy"
    assert state.credential_profile == "github-read"
    assert state.language_overrides == {".inc": "c"}
    assert repository.domains == ("gpu", "multimedia")
    assert repository.select(source_ids=("pion-webrtc",)) == (source,)


def test_registered_source_rejects_duplicate_identity_and_mirror_path():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())
    source = PublicCodeSource(
        source_id="pion-webrtc",
        domain="multimedia",
        relative_path="multimedia/pion-webrtc",
        upstream_url="https://github.com/pion/webrtc.git",
    )
    repository.create_source(source, update_policy="manual", update_interval_minutes=None)

    with pytest.raises(ValueError, match="source already exists"):
        repository.create_source(source, update_policy="manual", update_interval_minutes=None)
    with pytest.raises(ValueError, match="relative path already exists"):
        repository.create_source(
            PublicCodeSource(
                source_id="pion-datachannel",
                domain="multimedia",
                relative_path="multimedia/pion-webrtc",
                upstream_url="https://github.com/pion/datachannel.git",
            ),
            update_policy="manual",
            update_interval_minutes=None,
        )


def test_database_uniqueness_race_is_reported_as_a_source_conflict():
    base_factory = session_factory()

    @contextmanager
    def conflict_factory():
        with base_factory() as session:
            original_flush = session.flush

            def conflicting_flush(*args, **kwargs):
                if session.new:
                    raise IntegrityError("insert", {}, RuntimeError("unique constraint"))
                return original_flush(*args, **kwargs)

            session.flush = conflicting_flush
            yield session

    repository = PublicCodeRepository(session_factory=conflict_factory)
    source = PublicCodeSource(
        source_id="pion-webrtc",
        domain="multimedia",
        relative_path="multimedia/pion-webrtc",
        upstream_url="https://github.com/pion/webrtc.git",
    )

    with pytest.raises(ValueError, match="source or relative path already exists"):
        repository.create_source(source, update_policy="manual", update_interval_minutes=None)


def test_update_check_results_are_persisted_per_source_for_ui_and_scheduler_reads():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())

    repository.record_update_checks(
        (
            PublicCodeUpdateCheck(
                source_id="ffmpeg",
                domain="multimedia",
                local_revision="local",
                remote_revision="remote",
                status="update_available",
            ),
            PublicCodeUpdateCheck(
                source_id="cutlass",
                domain="gpu",
                local_revision=None,
                remote_revision=None,
                status="check_failed",
                message="remote unavailable",
            ),
        )
    )

    ffmpeg = repository.get_source("ffmpeg")
    cutlass = repository.get_source("cutlass")
    assert ffmpeg is not None and cutlass is not None
    assert ffmpeg.last_mirrored_revision == "local"
    assert ffmpeg.last_remote_revision == "remote"
    assert ffmpeg.last_check_error is None
    assert cutlass.last_check_error == "remote unavailable"
    assert cutlass.last_checked_at is not None


def test_promoting_a_candidate_revision_supersedes_the_previous_active_revision():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())

    repository.record_candidate_revision("ffmpeg", manifest("commit-a"), chunk_count=10)
    repository.validate_candidate_revision("ffmpeg", "commit-a", vector_count=10)
    repository.promote_revision("ffmpeg", "commit-a")
    repository.record_candidate_revision("ffmpeg", manifest("commit-b"), chunk_count=12)
    repository.validate_candidate_revision("ffmpeg", "commit-b", vector_count=12)
    repository.promote_revision("ffmpeg", "commit-b")

    source = repository.get_source("ffmpeg")
    assert source is not None
    assert source.active_revision == "commit-b"
    assert repository.get_revision_state("ffmpeg", "commit-a") == "superseded"
    assert repository.get_revision_state("ffmpeg", "commit-b") == "active"


def test_source_state_includes_active_revision_counts_for_management_views():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())
    repository.record_candidate_revision("ffmpeg", manifest("commit-a"), chunk_count=10)
    repository.validate_candidate_revision("ffmpeg", "commit-a", vector_count=10)
    repository.promote_revision("ffmpeg", "commit-a")

    source = repository.get_source("ffmpeg")

    assert source.active_manifest_file_count == 1
    assert source.active_chunk_count == 10
    assert source.active_vector_count == 10
    assert source.active_promoted_at is not None
    assert source.created_at is not None
    assert source.updated_at is not None


def test_candidate_cannot_promote_before_exact_vector_count_validation():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())
    repository.record_candidate_revision("ffmpeg", manifest("commit-a"), chunk_count=10)

    with pytest.raises(ValueError, match="validated"):
        repository.promote_revision("ffmpeg", "commit-a")
    with pytest.raises(ValueError, match="vector count mismatch"):
        repository.validate_candidate_revision("ffmpeg", "commit-a", vector_count=9)

    repository.validate_candidate_revision("ffmpeg", "commit-a", vector_count=10)
    repository.promote_revision("ffmpeg", "commit-a")

    assert repository.get_revision_state("ffmpeg", "commit-a") == "active"


def test_failed_candidate_does_not_replace_active_revision():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())
    repository.record_candidate_revision("ffmpeg", manifest("commit-a"), chunk_count=10)
    repository.validate_candidate_revision("ffmpeg", "commit-a", vector_count=10)
    repository.promote_revision("ffmpeg", "commit-a")
    repository.record_candidate_revision("ffmpeg", manifest("commit-b"), chunk_count=12)

    repository.mark_revision_failed("ffmpeg", "commit-b", error_message="embedding failed")

    assert repository.get_source("ffmpeg").active_revision == "commit-a"
    assert repository.get_revision_state("ffmpeg", "commit-b") == "failed"


def test_revision_cleanup_ownership_is_durable_across_candidate_and_promotion_states():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())
    repository.begin_candidate_revision("ffmpeg", manifest("commit-a"))

    assert repository.list_cleanup_revisions() == ()

    repository.record_candidate_revision("ffmpeg", manifest("commit-a"), chunk_count=10)
    repository.validate_candidate_revision("ffmpeg", "commit-a", vector_count=10)
    repository.promote_revision("ffmpeg", "commit-a")
    assert repository.list_cleanup_revisions() == ()

    repository.begin_candidate_revision("ffmpeg", manifest("commit-b"))
    repository.record_candidate_revision("ffmpeg", manifest("commit-b"), chunk_count=12)
    repository.validate_candidate_revision("ffmpeg", "commit-b", vector_count=12)
    repository.promote_revision("ffmpeg", "commit-b")
    assert repository.list_cleanup_revisions() == (("ffmpeg", "commit-a"),)

    repository.mark_revision_cleaned("ffmpeg", "commit-a")
    assert repository.list_cleanup_revisions() == ()


def test_startup_recovery_marks_abandoned_candidates_for_cleanup():
    repository = PublicCodeRepository(session_factory=session_factory())
    repository.reconcile_catalog(source_catalog())
    repository.begin_candidate_revision("ffmpeg", manifest("commit-a"))

    recovered = repository.mark_abandoned_candidates_failed(
        reason="backend restarted during indexing"
    )

    assert recovered == (("ffmpeg", "commit-a"),)
    assert repository.list_cleanup_revisions() == (("ffmpeg", "commit-a"),)


def test_job_snapshots_survive_repository_instances_and_active_jobs_are_interrupted():
    factory = session_factory()
    repository = PublicCodeRepository(session_factory=factory)
    snapshot = {
        "job_id": "job-a",
        "operation": "index",
        "source_ids": ["ffmpeg"],
        "total_sources": 1,
        "status": "running",
        "completed_sources": 0,
        "results": [],
        "errors": {},
        "phase": "embedding",
        "current_source_id": "ffmpeg",
        "current_path": "src/a.c",
        "files_done": 1,
        "files_total": 2,
        "chunks_done": 3,
        "message": "embedding",
        "source_progress": {},
        "created_at": 1.0,
        "started_at": 2.0,
        "finished_at": None,
    }
    repository.save_job_snapshot(snapshot, max_retained_jobs=10)

    second_repository = PublicCodeRepository(session_factory=factory)
    second_repository.fail_incomplete_jobs(reason="service restarted")
    recovered = second_repository.get_job_snapshot("job-a")

    assert recovered is not None
    assert recovered["status"] == "failed"
    assert recovered["errors"] == {"job": "service restarted"}
    assert second_repository.list_job_snapshots(limit=10) == (recovered,)
