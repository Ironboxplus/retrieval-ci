"""Regression coverage for owner-wide Journal bulk fan-out.

These tests deliberately use a file-backed SQLite database and real temporary
Markdown files.  The vector boundary is a recording implementation rather than
Qdrant itself, so the assertions can prove the application sends one batched
write with the exact per-scope payloads without requiring a running production
collection.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import JournalBulkDocItem, JournalRecordRequest
from app.services.journal.service import JournalService
from app.services.journal.chunker import JournalTextChunk
from app.services.journal.vector_store import JournalChunkUpsert, JournalVectorStore
from core.db.database import Base
from core.db.models import JournalProject


def _repository(database_path: Path) -> JournalRepository:
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
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

    class InventoryCountingRepository(JournalRepository):
        def __init__(self) -> None:
            super().__init__(session_factory=session_factory)
            self.inventory_reads = 0

        def all_live_doc_files(self, *, owner_user_id):
            self.inventory_reads += 1
            return super().all_live_doc_files(owner_user_id=owner_user_id)

    return InventoryCountingRepository()


class CountingEmbedder:
    """A deterministic local embedding implementation with an inspectable call log."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, include_sparse):
        copied = list(texts)
        self.calls.append(copied)
        return (
            [[float(len(text)), 1.0] for text in copied],
            [{"terms": float(len(text.split()))} for text in copied] if include_sparse else None,
        )


class RecordingBatchVectorStore:
    """Concrete vector boundary that records every production-style bulk group."""

    def __init__(self, *, fail_after_first_group: bool = False) -> None:
        self.fail_after_first_group = fail_after_first_group
        self.groups = []
        self.deleted: list[str] = []

    def upsert_chunk_groups(self, groups):
        if self.fail_after_first_group:
            # Model a real multi-scope grouped write whose first Qdrant payload
            # reached the server and whose later payload failed.  The service must
            # keep old SQL versions live and clean only these speculative new IDs.
            self.groups.extend(groups[:1])
            raise RuntimeError("isolated qdrant write failure")
        self.groups.extend(groups)
        return sum(len(group.chunks) for group in groups)

    def upsert_chunks(self, chunks, **kwargs):
        raise AssertionError("bulk fan-out must use one grouped Qdrant write boundary")

    def delete_by_entry_id(self, entry_id):
        self.deleted.append(entry_id)
        return 0

    def delete_by_entry_ids(self, entry_ids):
        self.deleted.extend(entry_ids)
        return 0


def _bulk_item(path: Path, title: str) -> JournalBulkDocItem:
    body = path.read_text(encoding="utf-8")
    return JournalBulkDocItem(
        title=title,
        body=body,
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        size_bytes=len(body.encode("utf-8")),
    )


def test_bulk_fanout_reuses_one_embedding_and_writes_unique_scope_payloads(tmp_path: Path):
    """A changed parent copy refreshes the existing child copy without re-embedding it.

    The scopes are actual nested directories and the input comes from a real
    file.  This catches the former provisional-ID bug as well as the former
    per-scope upsert/embedding loop.
    """
    parent = tmp_path / "workspace"
    child = parent / "docs"
    child.mkdir(parents=True)
    guide = child / "guide.md"
    guide.write_text("# Guide\n\nnew owner-wide physical-path content", encoding="utf-8")

    repository = _repository(tmp_path / "journal.sqlite")
    embedder = CountingEmbedder()
    vectors = RecordingBatchVectorStore()
    service = JournalService(repository=repository, vector_store=vectors, embed_fn=embedder)
    old_child = service.record(JournalRecordRequest(
        project_slug=str(child), client="mcp", source_kind="doc", title="guide.md",
        body="# Guide\n\nold child content", owner_user_id="owner-a",
        dedupe_policy="return_existing",
    ))
    embedder.calls.clear()
    vectors.groups.clear()
    repository.inventory_reads = 0

    result = service.index_docs_bulk(
        project_slug=str(parent), owner_user_id="owner-a", client="mcp",
        files=[_bulk_item(guide, "docs/guide.md")],
    )

    assert result["files"] == 2
    assert repository.inventory_reads == 1, "bulk planning must scan this owner's paths once"
    assert len(embedder.calls) == 1, "primary chunks must be embedded exactly once"
    assert len(vectors.groups) == 2, "primary and target must share one grouped Qdrant call"
    primary, target = vectors.groups
    assert primary.entry_id != target.entry_id
    assert {chunk.entry_id for chunk in primary.chunks} == {primary.entry_id}
    assert {chunk.entry_id for chunk in target.chunks} == {target.entry_id}
    assert {chunk.chunk_id for chunk in primary.chunks}.isdisjoint(
        {chunk.chunk_id for chunk in target.chunks})
    assert primary.dense_vectors == target.dense_vectors
    assert primary.sparse_vectors == target.sparse_vectors
    assert primary.project_slug != target.project_slug

    child_live = service.list_entries(
        project_slug=str(child), source_kind="doc", owner_user_id="owner-a")
    assert child_live.total == 1
    assert "new owner-wide" in service.get_entry(child_live.entries[0].entry_id, "owner-a").body
    assert old_child.entry.entry_id in vectors.deleted


def test_bulk_fanout_failure_keeps_every_scope_old_version_live(tmp_path: Path):
    """No latest-wins deletion may run until *all* owner-wide target writes succeed."""
    parent = tmp_path / "workspace"
    child = parent / "docs"
    child.mkdir(parents=True)
    guide = child / "guide.md"
    guide.write_text("new parent and child body", encoding="utf-8")

    repository = _repository(tmp_path / "journal.sqlite")
    vectors = RecordingBatchVectorStore()
    service = JournalService(repository=repository, vector_store=vectors, embed_fn=CountingEmbedder())
    old_child = service.record(JournalRecordRequest(
        project_slug=str(child), client="mcp", source_kind="doc", title="guide.md",
        body="old child body", owner_user_id="owner-a", dedupe_policy="return_existing"))
    old_parent = service.record(JournalRecordRequest(
        project_slug=str(parent), client="mcp", source_kind="doc", title="docs/guide.md",
        body="old parent body", owner_user_id="owner-a", dedupe_policy="return_existing"))
    vectors.fail_after_first_group = True

    with pytest.raises(RuntimeError, match="isolated qdrant write failure"):
        service.index_docs_bulk(
            project_slug=str(parent), owner_user_id="owner-a", client="mcp",
            files=[_bulk_item(guide, "docs/guide.md")],
        )

    assert old_child.entry.entry_id not in vectors.deleted
    assert old_parent.entry.entry_id not in vectors.deleted
    assert len(vectors.deleted) == 2, "only this request's speculative entries are cleaned"
    assert len(vectors.groups) == 1, "the failure happens after a real first group lands"
    assert service.get_entry(old_child.entry.entry_id, "owner-a").deleted is False
    assert service.get_entry(old_parent.entry.entry_id, "owner-a").deleted is False


def test_bulk_fanout_windows_unc_is_owner_isolated_and_parent_boundary_safe(tmp_path: Path):
    """Case-fold Windows/UNC identity, never cross owners or sibling-prefix scopes."""
    source = tmp_path / "guide.md"
    source.write_text("shared Windows body", encoding="utf-8")
    repository = _repository(tmp_path / "journal.sqlite")
    vectors = RecordingBatchVectorStore()
    service = JournalService(repository=repository, vector_store=vectors, embed_fn=CountingEmbedder())

    # Same physical file for owner-a (UNC case differences, parent/child scopes).
    child_a = r"\\SERVER\Share\Repo\Docs"
    parent_a = r"\\server\share\repo"
    old_a = service.record(JournalRecordRequest(
        project_slug=child_a, client="mcp", source_kind="doc", title="Guide.MD",
        body="old a", owner_user_id="owner-a", dedupe_policy="return_existing"))
    # Same spelling but another owner must never become a target.
    other_owner = service.record(JournalRecordRequest(
        project_slug=child_a, client="mcp", source_kind="doc", title="Guide.MD",
        body="old b", owner_user_id="owner-b", dedupe_policy="return_existing"))
    # Prefix sibling is not a parent/child and cannot match the physical path.
    sibling = service.record(JournalRecordRequest(
        project_slug=r"\\server\share\repo-backup", client="mcp", source_kind="doc",
        title="docs/guide.md", body="sibling", owner_user_id="owner-a",
        dedupe_policy="return_existing"))
    vectors.groups.clear()

    service.index_docs_bulk(
        project_slug=parent_a, owner_user_id="owner-a", client="mcp",
        files=[_bulk_item(source, "docs/guide.md")],
    )

    updated_a = service.list_entries(project_slug=child_a, source_kind="doc", owner_user_id="owner-a")
    assert updated_a.total == 1
    assert "shared Windows" in service.get_entry(updated_a.entries[0].entry_id, "owner-a").body
    assert service.get_entry(other_owner.entry.entry_id, "owner-b").body == "old b"
    assert service.get_entry(sibling.entry.entry_id, "owner-a").body == "sibling"
    assert old_a.entry.entry_id in vectors.deleted


def test_bulk_windows_legacy_scope_reuses_existing_project_not_parallel_alias(tmp_path: Path):
    """A case-only Windows spelling is one scope, including the write path."""
    source = tmp_path / "guide.md"
    source.write_text("canonicalized replacement", encoding="utf-8")
    repository = _repository(tmp_path / "journal.sqlite")
    vectors = RecordingBatchVectorStore()
    # Bypass current request validation to represent a project written before scope
    # canonicalization existed.  The bulk request must update this project in place.
    old, _project, _deduped = repository.create_entry(
        project_slug="E:/JETB/Research", client="mcp", source_kind="doc",
        title="Docs/Guide.md", summary=None, body="legacy old body", tags=[], metadata={},
        owner_user_id="owner-a", dedupe_policy="append")
    service = JournalService(repository=repository, vector_store=vectors, embed_fn=CountingEmbedder())

    service.index_docs_bulk(
        project_slug=r"e:\jetb\research", owner_user_id="owner-a", client="mcp",
        files=[_bulk_item(source, "docs/guide.md")],
    )

    with repository.session() as session:
        projects = session.query(JournalProject).filter(
            JournalProject.owner_user_id == "owner-a").all()
    assert [project.slug for project in projects] == ["E:/JETB/Research"]
    live = service.list_entries(
        project_slug="E:/jetb/research", source_kind="doc", owner_user_id="owner-a")
    assert live.total == 1
    assert live.entries[0].entry_id != old.entry_id


def test_bulk_fanout_three_scopes_still_embeds_the_physical_file_once(tmp_path: Path):
    """Embedding work scales with changed physical files, not fan-out scope count."""
    root = tmp_path
    parent = root / "workspace"
    child = parent / "docs"
    child.mkdir(parents=True)
    guide = child / "guide.md"
    guide.write_text("one physical file across three nested scopes", encoding="utf-8")
    repository = _repository(tmp_path / "journal.sqlite")
    vectors = RecordingBatchVectorStore()
    embedder = CountingEmbedder()
    # Seed the two existing aliases directly, without legacy single-record fan-out,
    # so the bulk request must create root + parent + child entries in one plan.
    repository.create_entry(
        project_slug=str(parent), client="mcp", source_kind="doc", title="docs/guide.md",
        summary=None, body="old parent", tags=[], metadata={}, owner_user_id="owner-a",
        dedupe_policy="append")
    repository.create_entry(
        project_slug=str(child), client="mcp", source_kind="doc", title="guide.md",
        summary=None, body="old child", tags=[], metadata={}, owner_user_id="owner-a",
        dedupe_policy="append")
    service = JournalService(repository=repository, vector_store=vectors, embed_fn=embedder)

    result = service.index_docs_bulk(
        project_slug=str(root), owner_user_id="owner-a", client="mcp",
        files=[_bulk_item(guide, "workspace/docs/guide.md")],
    )

    assert result["files"] == 3
    assert len(embedder.calls) == 1
    assert len(vectors.groups) == 3
    assert len({group.entry_id for group in vectors.groups}) == 3
    assert len({chunk.chunk_id for group in vectors.groups for chunk in group.chunks}) == sum(
        len(group.chunks) for group in vectors.groups)


class _RecordingQdrantBoundary:
    """Small concrete QdrantStore-shaped boundary used to exercise production points."""

    supports_sparse = False

    def __init__(self) -> None:
        self.batches = []

    def _upsert_batch(self, points, *, wait):
        self.batches.append((list(points), wait))


def test_production_vector_store_flattens_scope_groups_into_waiting_qdrant_batches():
    """The real JournalVectorStore, not the service test adapter, owns batching.

    260 + 20 points crosses the production 256-point boundary.  All payloads
    retain the source scope/entry IDs, and every actual Qdrant mutation asks for
    ``wait=True`` before the service is allowed to promote latest-wins versions.
    """
    boundary = _RecordingQdrantBoundary()
    vectors = JournalVectorStore(collection_name="journal_chunks_test", store=boundary)

    def group(scope: str, entry_id: str, count: int) -> JournalChunkUpsert:
        chunks = [JournalTextChunk(
            chunk_id=f"{entry_id}-chunk-{index}", entry_id=entry_id,
            chunk_index=index, level=2, content=f"content {scope} {index}")
            for index in range(count)]
        return JournalChunkUpsert(
            chunks=chunks, project_id=7, project_slug=scope, entry_id=entry_id,
            owner_user_id="owner-a", client="mcp", source_kind="doc", tags=[],
            dense_vectors=[[float(index), 1.0] for index in range(count)],
        )

    assert vectors.upsert_chunk_groups([
        group("C:/Repo", "entry-parent", 260),
        group("C:/Repo/Docs", "entry-child", 20),
    ]) == 280
    assert [len(points) for points, _wait in boundary.batches] == [256, 24]
    assert all(wait is True for _points, wait in boundary.batches)
    points = [point for batch, _wait in boundary.batches for point in batch]
    assert {point.payload["entry_id"] for point in points} == {"entry-parent", "entry-child"}
    assert {point.payload["project_slug"] for point in points} == {"C:/Repo", "C:/Repo/Docs"}
    assert len({point.id for point in points}) == 280
