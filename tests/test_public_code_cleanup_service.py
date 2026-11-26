"""TDD for durable Qdrant cleanup retries owned by SQL revision state."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.public_code.cleanup_service import PublicCodeRevisionCleanupService


@dataclass
class Repository:
    pending: tuple[tuple[str, str], ...]
    cleaned: list[tuple[str, str]] = field(default_factory=list)

    def list_cleanup_revisions(self):
        return self.pending

    def mark_revision_cleaned(self, source_id, revision):
        self.cleaned.append((source_id, revision))


@dataclass
class VectorStore:
    fail_revision: str | None = None
    deleted: list[tuple[str, str]] = field(default_factory=list)

    def delete_revision(self, source_id, revision):
        if revision == self.fail_revision:
            raise RuntimeError("qdrant unavailable")
        self.deleted.append((source_id, revision))


def test_cleanup_retries_every_sql_owned_revision_and_preserves_failures_for_next_cycle():
    repository = Repository((("pion", "candidate"), ("ffmpeg", "old")))
    vector_store = VectorStore(fail_revision="old")
    service = PublicCodeRevisionCleanupService(
        repository=repository,
        vector_store=vector_store,
    )

    result = service.run_once()

    assert result.cleaned == (("pion", "candidate"),)
    assert result.errors == {"ffmpeg:old": "qdrant unavailable"}
    assert repository.cleaned == [("pion", "candidate")]
    assert vector_store.deleted == [("pion", "candidate")]
