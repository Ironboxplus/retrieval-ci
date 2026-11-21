"""Application services for the shared public code knowledge base."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config import settings
from app.services.pipeline.queue_tasks import QueueTasks
from app.services.public_code.access_profiles import GitAccessProfileResolver
from app.services.public_code.bundle_import import PublicCodeBundleImportService
from app.services.public_code.control_service import PublicCodeControlService
from app.services.public_code.cleanup_service import PublicCodeRevisionCleanupService
from app.services.public_code.git_mirror import GitMirrorSourceReader
from app.services.public_code.index_service import PublicCodeIndexService
from app.services.public_code.jobs import PublicCodeJobService
from app.services.public_code.repository import PublicCodeRepository
from app.services.public_code.search_service import PublicCodeSearchService
from app.services.public_code.source_catalog import PublicCodeSourceCatalog
from app.services.public_code.update_checks import (
    GitMirrorRevisionProbe,
    PublicCodeUpdateCheckService,
)
from app.services.public_code.update_scheduler import PublicCodeUpdateScheduler
from app.services.public_code.vector_store import PublicCodeVectorStore
from core.code_kb.isolated_chunker import get_isolated_code_chunker


@lru_cache(maxsize=1)
def _get_public_code_bootstrap_catalog() -> PublicCodeSourceCatalog:
    """Load the checked-in source declaration once per backend process."""

    return PublicCodeSourceCatalog.from_json_file(Path(settings.public_code.catalog_path))


@lru_cache(maxsize=1)
def get_public_code_repository() -> PublicCodeRepository:
    repository = PublicCodeRepository()
    repository.reconcile_catalog(_get_public_code_bootstrap_catalog())
    repository.mark_abandoned_candidates_failed(
        reason="backend restarted before candidate revision promotion"
    )
    return repository


@lru_cache(maxsize=1)
def get_public_code_vector_store() -> PublicCodeVectorStore:
    vector_store = PublicCodeVectorStore()
    vector_store.start_payload_index_maintenance()
    return vector_store


@lru_cache(maxsize=1)
def get_public_code_access_profiles() -> GitAccessProfileResolver:
    return GitAccessProfileResolver(default_proxy_url=settings.proxy.https_proxy)


@lru_cache(maxsize=1)
def get_public_code_control_service() -> PublicCodeControlService:
    repository = get_public_code_repository()
    checks = PublicCodeUpdateCheckService(
        source_registry=repository,
        revision_probe=GitMirrorRevisionProbe(
            mirror_root=Path(settings.public_code.mirror_root),
            access_profiles=get_public_code_access_profiles(),
        ),
    )
    return PublicCodeControlService(
        repository=repository,
        update_checks=checks,
        source_validator=get_public_code_access_profiles().validate_source,
    )


@lru_cache(maxsize=1)
def get_public_code_search_service() -> PublicCodeSearchService:
    return PublicCodeSearchService(
        repository=get_public_code_repository(),
        vector_store=get_public_code_vector_store(),
    )


@lru_cache(maxsize=1)
def get_public_code_index_service() -> PublicCodeIndexService:
    return PublicCodeIndexService(
        source_registry=get_public_code_repository(),
        repository=get_public_code_repository(),
        source_reader=GitMirrorSourceReader(
            mirror_root=Path(settings.public_code.mirror_root),
            timeout_seconds=settings.public_code.git_command_timeout_seconds,
            network_timeout_seconds=settings.public_code.git_network_timeout_seconds,
            network_retry_attempts=settings.public_code.git_network_retry_attempts,
            network_retry_backoff_seconds=(
                settings.public_code.git_network_retry_backoff_seconds
            ),
            access_profiles=get_public_code_access_profiles(),
        ),
        vector_store=get_public_code_vector_store(),
        chunk_fn=get_isolated_code_chunker(
            timeout_seconds=settings.public_code.native_chunk_timeout_seconds,
        ),
        private_vector_store_factory=_get_private_code_vector_store,
    )


def _get_private_code_vector_store():
    from app.services.code_kb import get_code_kb_service

    return get_code_kb_service().vector_store


@lru_cache(maxsize=1)
def get_public_code_job_service() -> PublicCodeJobService:
    return PublicCodeJobService(
        source_registry=get_public_code_repository(),
        control_service=get_public_code_control_service(),
        index_service=get_public_code_index_service(),
        job_store=get_public_code_repository(),
        tasks=QueueTasks(),
    )


@lru_cache(maxsize=1)
def get_public_code_import_service() -> PublicCodeBundleImportService:
    return PublicCodeBundleImportService(
        import_root=Path(settings.public_code.import_root),
        mirror_root=Path(settings.public_code.mirror_root),
        repository=get_public_code_repository(),
        max_bundle_bytes=settings.public_code.max_import_bytes,
        git_timeout_seconds=settings.public_code.import_git_timeout_seconds,
    )


@lru_cache(maxsize=1)
def get_public_code_cleanup_service() -> PublicCodeRevisionCleanupService:
    return PublicCodeRevisionCleanupService(
        repository=get_public_code_repository(),
        vector_store=get_public_code_vector_store(),
    )


@lru_cache(maxsize=1)
def get_public_code_update_scheduler() -> PublicCodeUpdateScheduler:
    return PublicCodeUpdateScheduler(
        repository=get_public_code_repository(),
        control_service=get_public_code_control_service(),
        job_service=get_public_code_job_service(),
        maintenance_service=get_public_code_cleanup_service(),
    )


__all__ = [
    "get_public_code_cleanup_service",
    "get_public_code_access_profiles",
    "get_public_code_control_service",
    "get_public_code_import_service",
    "get_public_code_index_service",
    "get_public_code_job_service",
    "get_public_code_repository",
    "get_public_code_search_service",
    "get_public_code_update_scheduler",
    "get_public_code_vector_store",
]
