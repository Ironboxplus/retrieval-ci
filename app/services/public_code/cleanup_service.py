"""Durable cleanup retry for non-active public-code vector revisions."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol


logger = logging.getLogger(__name__)


class CleanupRevisionRepository(Protocol):
    def list_cleanup_revisions(self) -> tuple[tuple[str, str], ...]: ...

    def mark_revision_cleaned(self, source_id: str, revision_name: str) -> None: ...


class CleanupVectorStore(Protocol):
    def delete_revision(self, source_id: str, revision: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PublicCodeCleanupResult:
    cleaned: tuple[tuple[str, str], ...]
    errors: dict[str, str]


class PublicCodeRevisionCleanupService:
    """Retry SQL-owned Qdrant cleanup without touching active revision filters."""

    def __init__(self, *, repository: CleanupRevisionRepository, vector_store: CleanupVectorStore):
        self._repository = repository
        self._vector_store = vector_store

    def run_once(self) -> PublicCodeCleanupResult:
        cleaned: list[tuple[str, str]] = []
        errors: dict[str, str] = {}
        for source_id, revision in self._repository.list_cleanup_revisions():
            key = f"{source_id}:{revision}"
            try:
                self._vector_store.delete_revision(source_id, revision)
                self._repository.mark_revision_cleaned(source_id, revision)
                cleaned.append((source_id, revision))
            except Exception as exc:
                errors[key] = str(exc)
                logger.exception(
                    "public-code durable revision cleanup failed source=%s revision=%s",
                    source_id,
                    revision,
                )
        return PublicCodeCleanupResult(cleaned=tuple(cleaned), errors=errors)
