"""Dot-directory docs (``.conversations/...``) sit outside every journal_index_dir scan envelope.

The Journal doc walker never descends into dot-directories, so a recursive ``*.md`` replace
run cannot have scanned them.  The delete-reconcile envelopes (journal stale-doc reconcile and
the catalog replace-sweep) must therefore never treat such rows as "missing locally".  All tests
run against real SQLite; the vector boundary is a recording store, as in test_journal_bulk_fanout.
"""
from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.schemas import IndexFileRecord, IndexReportRequest
from app.services.index_catalog.service import IndexCatalogService
from app.services.journal.jobs import JournalBulkIndexQueue
from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import (
    JournalBulkDocItem, JournalBulkIndexRequest, JournalRecordRequest,
)
from app.services.journal.service import JournalService
from core.db.database import Base

CONVERSATION = ".conversations/claude/2026-11-04_x/001.md"


def _session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return session_factory


class RecordingVectorStore:
    def __init__(self) -> None:
        self.groups = []
        self.deleted: list[str] = []

    def upsert_chunk_groups(self, groups, *, on_batch=None):
        self.groups.extend(groups)
        written = sum(len(group.chunks) for group in groups)
        if on_batch is not None:
            on_batch(written, written)
        return written

    def upsert_chunks(self, chunks, **kwargs):
        raise AssertionError("bulk path must use the grouped write")

    def delete_by_entry_id(self, entry_id):
        self.deleted.append(entry_id)
        return 0

    def delete_by_entry_ids(self, entry_ids):
        self.deleted.extend(entry_ids)
        return 0


def _embed(texts, include_sparse):
    return ([[float(len(t)), 1.0] for t in texts],
            [{"1": 1.0} for _ in texts] if include_sparse else None)


def _service() -> tuple[JournalService, RecordingVectorStore]:
    store = RecordingVectorStore()
    service = JournalService(
        repository=JournalRepository(session_factory=_session_factory()),
        vector_store=store, embed_fn=_embed)
    return service, store


def _record(service: JournalService, title: str, client: str = "mcp"):
    return service.record(JournalRecordRequest(
        project_slug="retrival", client=client, source_kind="doc", title=title,
        body=f"body of {title}", owner_user_id="user-1", dedupe_policy="return_existing",
    )).entry.entry_id


def _item(title: str) -> JournalBulkDocItem:
    body = f"# {title}\n\nbody of {title}"
    return JournalBulkDocItem(
        title=title, body=body, size_bytes=len(body.encode()),
        content_hash=hashlib.sha256(body.encode()).hexdigest())


def _live_titles(service: JournalService) -> set[str]:
    return {e.title for e in service.list_entries(
        project_slug="retrival", source_kind="doc", owner_user_id="user-1").entries}


# ---- journal stale-doc reconcile ---------------------------------------------------------

@pytest.mark.parametrize("patterns,max_depth", [
    (["*.md"], None),   # the recursive journal_index_dir envelope
    (None, None),       # full reconcile
    (["*"], None),      # "*" is the full-reconcile alias
    (["*.md"], 3),
])
def test_reconcile_never_stales_dot_directory_docs(patterns, max_depth):
    service, store = _service()
    conversation = _record(service, CONVERSATION, client="claude")
    keep = _record(service, "notes/keep.md")
    gone = _record(service, "notes/a.md")

    result = service.reconcile_doc_scope(
        project_slug="retrival", live_titles=["notes/keep.md"], owner_user_id="user-1",
        patterns=patterns, max_depth=max_depth)

    assert result.removed_entry_ids == [gone]
    assert conversation not in store.deleted and keep not in store.deleted
    assert _live_titles(service) == {CONVERSATION, "notes/keep.md"}


def test_reconcile_dot_directory_at_any_depth_and_windows_separators():
    service, _ = _service()
    _record(service, "docs/.hidden/x.md")
    _record(service, ".a/.b/y.md")
    _record(service, "docs\\.win\\z.md")
    keep = _record(service, "keep.md")

    result = service.reconcile_doc_scope(
        project_slug="retrival", live_titles=["keep.md"], owner_user_id="user-1")

    assert result.removed_entries == 0
    assert keep not in result.removed_entry_ids


def test_reconcile_dot_file_basename_in_normal_directory_is_still_stale():
    service, _ = _service()
    hidden = _record(service, "notes/.hidden.md")
    keep = _record(service, "notes/keep.md")

    result = service.reconcile_doc_scope(
        project_slug="retrival", live_titles=["notes/keep.md"], owner_user_id="user-1",
        patterns=["*.md"])

    assert result.removed_entry_ids == [hidden]
    assert keep not in result.removed_entry_ids


# ---- catalog replace-sweep ----------------------------------------------------------------

def _catalog() -> IndexCatalogService:
    return IndexCatalogService(
        repository=IndexCatalogRepository(session_factory=_session_factory()))


def _report(svc, files, *, replace=True, patterns=None, max_depth=None):
    return svc.report(IndexReportRequest(
        host="4090", project="retrival", root_dir="/repo", replace=replace,
        patterns=patterns, max_depth=max_depth,
        files=[IndexFileRecord(
            rel_path=p, kind=k, collection="code_chunks" if k == "code" else "journal_chunks",
            chunk_count=1) for p, k in files]), owner_user_id="user-1")


def _inventory(svc) -> set[tuple[str, str]]:
    return {(f.rel_path, f.kind)
            for u in svc.inventory(owner_user_id="user-1").users
            for h in u.hosts for p in h.projects for d in p.dirs for f in d.files}


@pytest.mark.parametrize("patterns", [["*.md"], None, ["*"]])
def test_catalog_doc_sweep_keeps_dot_directory_rows(patterns):
    svc = _catalog()
    _report(svc, [(CONVERSATION, "doc"), ("notes/a.md", "doc"), ("notes/b.md", "doc")])

    result = _report(svc, [("notes/b.md", "doc")], patterns=patterns)

    assert result.removed == 1
    assert _inventory(svc) == {(CONVERSATION, "doc"), ("notes/b.md", "doc")}


def test_catalog_doc_sweep_still_sweeps_dot_file_in_normal_directory():
    svc = _catalog()
    _report(svc, [("notes/.hidden.md", "doc"), ("notes/b.md", "doc")])

    result = _report(svc, [("notes/b.md", "doc")], patterns=["*.md"])

    assert result.removed == 1
    assert _inventory(svc) == {("notes/b.md", "doc")}


def test_catalog_code_sweep_still_covers_dot_directory_rows():
    # Code walks DO enter dot-dirs (.github/workflows/...); a code report passes no
    # patterns/max_depth, and its whole-kind sweep must keep reaching those rows.
    svc = _catalog()
    _report(svc, [(".github/workflows/ci.yml", "code"), ("src/a.py", "code"),
                  (CONVERSATION, "doc")])

    result = _report(svc, [("src/a.py", "code")])

    assert result.removed == 1
    assert _inventory(svc) == {("src/a.py", "code"), (CONVERSATION, "doc")}


# ---- bulk-index client passthrough, end to end through the queue ---------------------------

def _run_job(queue: JournalBulkIndexQueue, request: JournalBulkIndexRequest) -> None:
    job = queue.submit(request, owner_user_id="user-1")
    deadline = time.time() + 30
    while time.time() < deadline:
        current = queue.get(job.job_id, owner_user_id="user-1")
        if current.status.value in {"completed", "failed"}:
            assert current.status.value == "completed", current.error
            return
        time.sleep(0.02)
    raise AssertionError("journal bulk job did not finish")


def _entries(service: JournalService):
    return service.list_entries(
        project_slug="retrival", source_kind="doc", owner_user_id="user-1").entries


def test_bulk_index_client_defaults_to_mcp_and_passes_through():
    service, _ = _service()
    queue = JournalBulkIndexQueue(service=service, catalog_service=_catalog())

    assert JournalBulkIndexRequest(
        project_slug="retrival", host="h", root_dir="/repo").client == "mcp"
    _run_job(queue, JournalBulkIndexRequest(
        project_slug="retrival", host="h", root_dir="/repo", replace=False,
        files=[_item("notes/plain.md")]))
    _run_job(queue, JournalBulkIndexRequest(
        project_slug="retrival", host="h", root_dir="/repo", replace=False, client="claude",
        files=[_item(CONVERSATION)]))

    clients = {e.title: e.client for e in _entries(service)}
    assert clients == {"notes/plain.md": "mcp", CONVERSATION: "claude"}


def test_recursive_replace_job_keeps_conversation_docs_and_catalog_rows():
    service, store = _service()
    catalog = _catalog()
    queue = JournalBulkIndexQueue(service=service, catalog_service=catalog)
    base = dict(project_slug="retrival", host="h", root_dir="/repo")

    _run_job(queue, JournalBulkIndexRequest(
        **base, replace=False, client="claude",
        files=[_item(CONVERSATION)],
        report_files=[{"rel_path": CONVERSATION, "kind": "doc",
                       "collection": "journal_chunks", "chunk_count": 1}]))
    _run_job(queue, JournalBulkIndexRequest(
        **base, replace=False,
        files=[_item("notes/a.md"), _item("notes/b.md")],
        report_files=[{"rel_path": p, "kind": "doc", "collection": "journal_chunks",
                       "chunk_count": 1} for p in ("notes/a.md", "notes/b.md")]))

    # journal_index_dir recursive "*.md" replace after notes/a.md was deleted locally.
    _run_job(queue, JournalBulkIndexRequest(
        **base, replace=True, patterns=["*.md"], max_depth=None,
        live_titles=["notes/b.md"],
        report_files=[{"rel_path": "notes/b.md", "kind": "doc",
                       "collection": "journal_chunks", "chunk_count": 1}]))

    assert {e.title for e in _entries(service)} == {CONVERSATION, "notes/b.md"}
    assert {(rel, kind) for rel, kind in _inventory(catalog)} == {
        (CONVERSATION, "doc"), ("notes/b.md", "doc")}
