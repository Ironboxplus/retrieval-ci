"""TDD for ACE journal service, chunking, and vector boundary."""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from app.models.schemas import SearchResult
from app.services.journal.chunker import JournalChunker
from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import JournalRecordRequest, JournalSearchRequest
from app.services.journal.service import JournalService


def _repo():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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

    return JournalRepository(session_factory=session_factory)


def _fake_embed(texts, include_sparse):
    dense = [[float(len(text)), 1.0, 0.0] for text in texts]
    sparse = [{"1": float(len(text.split()))} for text in texts] if include_sparse else None
    return dense, sparse


class CapturingVectorStore:
    def __init__(self):
        self.upserts = []
        self.deleted = []
        self.search_results = []
        self.search_calls = []

    def upsert_chunks(self, chunks, **kwargs):
        self.upserts.append((chunks, kwargs))
        return len(chunks)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        results = list(self.search_results)
        # Honor the entry_id push-down filter (path_glob) so the semantic push-down test
        # is meaningful: the real Qdrant store applies a MatchAny on entry_id, so top_k
        # fills only from matching docs. Mirror that here.
        allowed = (kwargs.get("filters") or {}).get("entry_id")
        if allowed is not None:
            allowed_set = set(allowed)
            results = [r for r in results if (r.get("metadata") or {}).get("entry_id") in allowed_set]
        return results

    def delete_by_entry_id(self, entry_id):
        self.deleted.append(entry_id)
        return 3


class FailingVectorStore(CapturingVectorStore):
    def upsert_chunks(self, chunks, **kwargs):
        raise RuntimeError("qdrant down")


class ToggleableVectorStore(CapturingVectorStore):
    """A vector store that starts down (raises) and can be brought up — models the
    embed/index endpoint recovering, so entries recorded while down become pending."""

    def __init__(self):
        super().__init__()
        self.up = True

    def upsert_chunks(self, chunks, **kwargs):
        if not self.up:
            raise RuntimeError("qdrant down")
        return super().upsert_chunks(chunks, **kwargs)


def test_default_embedder_marks_journal_writes_background_and_search_interactive(monkeypatch):
    import numpy as np
    from core.indexing.embedders import EmbeddingResult

    class PrioritySpyEmbedder:
        def __init__(self):
            self.priorities = []

        def embed_texts(self, texts, *, priority="interactive"):
            self.priorities.append(priority)
            return EmbeddingResult(
                dense=np.asarray([[float(len(text)), 1.0, 0.0] for text in texts]),
                sparse=[{"1": 1.0} for _ in texts],
            )

    embedder = PrioritySpyEmbedder()
    monkeypatch.setattr("core.indexing.embedders.get_embedder", lambda: embedder)
    service = JournalService(
        repository=_repo(), vector_store=CapturingVectorStore())

    service.record(JournalRecordRequest(
        project_slug="retrival", client="codex", source_kind="memory",
        body="journal scheduler priority marker",
    ))
    service.search(JournalSearchRequest(
        project_slug="retrival", query="scheduler priority", mode="hybrid", top_k=3,
    ))

    assert embedder.priorities == ["background", "interactive"]


def test_default_journal_background_embedding_runs_in_bounded_batches(monkeypatch):
    import numpy as np
    from core.indexing.embedders import EmbeddingResult
    from app.services.journal.service import JOURNAL_EMBED_BATCH

    class BatchSpyEmbedder:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts, *, priority="interactive"):
            self.calls.append((priority, len(texts)))
            return EmbeddingResult(
                dense=np.asarray([[1.0, 0.0] for _ in texts]),
                sparse=[{"1": 1.0} for _ in texts],
            )

    embedder = BatchSpyEmbedder()
    monkeypatch.setattr("core.indexing.embedders.get_embedder", lambda: embedder)
    service = JournalService(
        repository=_repo(), vector_store=CapturingVectorStore())
    total = JOURNAL_EMBED_BATCH * 2 + 7

    dense, sparse = service._embed(
        [f"chunk {index}" for index in range(total)],
        True,
        priority="background",
    )

    # Journal's own ingestion batches (progress and cancel points) are the only split now: each
    # is one background call of the shared embedder, never cut into smaller model pieces.
    assert embedder.calls == [("background", JOURNAL_EMBED_BATCH), ("background", JOURNAL_EMBED_BATCH),
                              ("background", 7)]
    assert sum(size for _, size in embedder.calls) == total
    assert len(dense) == total
    assert sparse is not None and len(sparse) == total


def test_record_rejects_binary_body():
    """Server-side backstop (W161): a body carrying NUL bytes is binary garbage — a
    png/pickle/parquet that slipped past every CLIENT extension filter (a raw POST, a
    forked plugin, an unlabeled binary). Reject at the trust boundary so 'no binaries in
    the journal' holds no matter which client is talking. But a mis-encoded (U+FFFD-dense)
    body is legitimate, wrongly-decoded prose — it must still go through, never a 400
    (W162 review CR1: NUL is the only signal; GBK docs are not binaries)."""
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    with pytest.raises(ValueError):
        service.record(JournalRecordRequest(
            project_slug="retrival", client="mcp", source_kind="doc",
            title="plot.png", body="\x89PNG\r\n\x00\x00\x00\rIHDR\x00\x00binary",
        ))
    # A replacement-char-dense body (GBK/Latin-1 read as UTF-8) is mis-encoded prose,
    # NOT binary — must index, not 400.
    ok_mis = service.record(JournalRecordRequest(
        project_slug="retrival", client="mcp", source_kind="doc",
        title="legacy.md", body="����� 中文文档 ����",
    ))
    assert ok_mis.entry.entry_id
    # A normal note with one stray replacement char is text — must pass.
    ok = service.record(JournalRecordRequest(
        project_slug="retrival", client="codex", source_kind="memory",
        body="a perfectly ordinary journal note with a lone � typo in it",
    ))
    assert ok.entry.entry_id


def test_pending_entries_lists_only_unindexed_live_entries():
    vector_store = ToggleableVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    vector_store.up = False
    failed = service.record(JournalRecordRequest(
        project_slug="retrival", client="codex", source_kind="memory",
        body="recorded while the index was down",
    ))
    vector_store.up = True
    service.record(JournalRecordRequest(
        project_slug="retrival", client="codex", source_kind="memory",
        body="recorded while the index was healthy",
    ))

    pending = service.pending_entries()
    assert pending.total == 1
    assert pending.entries[0].entry_id == failed.entry.entry_id
    assert pending.entries[0].indexed is False


def test_reindex_pending_reconciles_and_is_idempotent():
    vector_store = ToggleableVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    vector_store.up = False
    service.record(JournalRecordRequest(
        project_slug="retrival", client="codex", source_kind="memory",
        title="Dropped vectors", body="SQL kept this but vectors never landed.",
        tags=["ops"], owner_user_id="user-1",
    ))
    assert service.pending_entries().total == 1

    # Index recovers → drain the queue.
    vector_store.up = True
    result = service.reindex_pending()
    assert result == {"reindexed": 1, "failed": 0, "remaining": 0}
    assert service.pending_entries().total == 0

    # The re-upsert carried the entry's real project/owner/source metadata.
    _, kwargs = vector_store.upserts[-1]
    assert kwargs["project_slug"] == "retrival"
    assert kwargs["owner_user_id"] == "user-1"
    assert kwargs["source_kind"] == "memory"

    # Idempotent: a second drain finds nothing.
    assert service.reindex_pending() == {"reindexed": 0, "failed": 0, "remaining": 0}


def test_reindex_pending_reports_failures_and_keeps_entry_pending():
    vector_store = ToggleableVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    vector_store.up = False
    service.record(JournalRecordRequest(
        project_slug="retrival", client="codex", source_kind="memory",
        body="still cannot be vectorized",
    ))

    # Index still down when we try to drain → the entry stays pending.
    result = service.reindex_pending()
    assert result == {"reindexed": 0, "failed": 1, "remaining": 1}
    assert service.pending_entries().total == 1


class PreferenceReranker:
    def rerank(self, query, results, top_k=10):
        ranked = sorted(results, key=lambda item: "preferred" in item.content, reverse=True)
        return [
            SearchResult(
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                content=item.content,
                score=0.99 if "preferred" in item.content else 0.11,
                doc_title=item.doc_title,
                source=item.source,
                metadata={**item.metadata, "reranker_score": 0.99 if "preferred" in item.content else 0.11},
            )
            for item in ranked[:top_k]
        ]


def test_keyword_mode_search_is_embedding_free():
    repo = _repo()

    def _boom_embed(texts, include_sparse):
        raise AssertionError("keyword search must not call the embed endpoint")

    vector_store = CapturingVectorStore()
    service = JournalService(repository=repo, vector_store=vector_store, embed_fn=_boom_embed)
    service.record(JournalRecordRequest(
        project_slug="retrival", client="codex", source_kind="memory",
        title="GPU incident", body="GPU0 fell off the bus, CUDA init failed W131.",
    ))

    resp = service.search(JournalSearchRequest(
        project_slug="retrival", query="cuda", mode="keyword", top_k=5,
    ))

    assert resp.mode == "keyword"
    assert resp.metrics.get("engine") == "fts5"
    assert resp.total == 1
    assert "CUDA" in resp.hits[0].entry.title or resp.hits[0].best_chunk is not None
    assert vector_store.search_calls == []


def test_keyword_search_caps_included_chunks_for_mcp_payloads():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    body = "\n\n".join(
        f"# Section {idx}\nchunk limit payload marker {idx}"
        for idx in range(1, 7)
    )
    created = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="mcp",
        source_kind="doc",
        title="large-journal.md",
        body=body,
    ))

    response = service.search(JournalSearchRequest(
        project_slug="retrival",
        query="payload marker",
        mode="keyword",
        top_k=1,
        include_chunks=True,
        chunk_limit=3,
    ))

    assert created.chunk_count > 3
    assert response.total == 1
    assert len(response.hits[0].chunks) == 3
    assert all(chunk.chunk_index > 0 for chunk in response.hits[0].chunks)


def test_keyword_search_chunk_limit_returns_matching_late_chunk():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    body = "\n\n".join(
        [f"# Section {idx}\nordinary text {idx}" for idx in range(1, 6)]
        + ["# Late\nUNIQUE_NEEDLE only appears in the final section"]
    )
    service.record(JournalRecordRequest(
        project_slug="retrival",
        client="mcp",
        source_kind="doc",
        title="late-match.md",
        body=body,
    ))

    response = service.search(JournalSearchRequest(
        project_slug="retrival",
        query="UNIQUE_NEEDLE",
        mode="keyword",
        top_k=1,
        include_chunks=True,
        chunk_limit=3,
    ))

    assert response.total == 1
    assert response.hits[0].best_chunk is not None
    assert "UNIQUE_NEEDLE" in response.hits[0].best_chunk.content
    assert any("UNIQUE_NEEDLE" in chunk.content for chunk in response.hits[0].chunks)
    assert len(response.hits[0].chunks) <= 3


def test_keyword_search_title_match_falls_back_to_body_chunks():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    service.record(JournalRecordRequest(
        project_slug="retrival",
        client="mcp",
        source_kind="doc",
        title="UNIQUE_TITLE.md",
        body="# Body\nordinary paragraph without the title token",
    ))

    response = service.search(JournalSearchRequest(
        project_slug="retrival",
        query="UNIQUE_TITLE",
        mode="keyword",
        top_k=1,
        include_chunks=True,
        chunk_limit=3,
    ))

    assert response.total == 1
    assert response.hits[0].best_chunk is not None
    assert response.hits[0].best_chunk.chunk_index > 0
    assert "ordinary paragraph" in response.hits[0].best_chunk.content
    assert all(chunk.chunk_index > 0 for chunk in response.hits[0].chunks)


def test_keyword_search_escapes_identifier_underscore_for_chunk_hydration():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    service.record(JournalRecordRequest(
        project_slug="retrival",
        client="mcp",
        source_kind="doc",
        title="delete_code.md",
        body="# Early\ndelete-code is only a fuzzy neighbor\n\n# Late\ndelete_code is the literal identifier",
    ))

    response = service.search(JournalSearchRequest(
        project_slug="retrival",
        query="delete_code",
        mode="keyword",
        top_k=1,
        include_chunks=True,
        chunk_limit=1,
    ))

    assert response.total == 1
    assert response.hits[0].best_chunk is not None
    assert "delete_code" in response.hits[0].best_chunk.content
    assert "delete-code" not in response.hits[0].best_chunk.content


def test_keyword_search_chunk_limit_does_not_hydrate_all_entry_chunks(monkeypatch):
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    body = "\n\n".join(
        f"# Section {idx}\nchunk hydration cost marker {idx}"
        for idx in range(1, 10)
    )
    created = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="mcp",
        source_kind="doc",
        title="huge-journal.md",
        body=body,
    ))
    loaded_chunks = []
    original_chunk_response = service._chunk_response

    def capture_chunk_response(row, score=None):
        loaded_chunks.append(row.chunk_id)
        return original_chunk_response(row, score=score)

    monkeypatch.setattr(service, "_chunk_response", capture_chunk_response)

    response = service.search(JournalSearchRequest(
        project_slug="retrival",
        query="hydration cost marker",
        mode="keyword",
        top_k=1,
        include_chunks=True,
        chunk_limit=3,
    ))

    assert created.chunk_count > 3
    assert len(response.hits[0].chunks) == 3
    assert len(loaded_chunks) == 3


def test_admin_keyword_search_chunks_are_bounded_by_default(monkeypatch):
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    body = "\n\n".join(
        f"# Section {idx}\nadmin chunk hydration marker {idx}"
        for idx in range(1, 10)
    )
    created = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="mcp",
        source_kind="doc",
        title="admin-large-journal.md",
        body=body,
    ))
    loaded_chunks = []
    original_chunk_response = service._chunk_response

    def capture_chunk_response(row, score=None):
        loaded_chunks.append(row.chunk_id)
        return original_chunk_response(row, score=score)

    monkeypatch.setattr(service, "_chunk_response", capture_chunk_response)

    response = service.admin_search(
        query="admin chunk hydration marker",
        mode="keyword",
        top_k=1,
        include_chunks=True,
    )

    assert created.chunk_count > 5
    assert len(response.hits[0].chunks) == 5
    assert len(loaded_chunks) == 5


def test_admin_search_spans_users_when_owner_is_none():
    repo = _repo()
    service = JournalService(repository=repo, vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    for owner in ("user-1", "user-2"):
        service.record(JournalRecordRequest(
            project_slug="retrival", client="codex", source_kind="memory",
            title=None, body=f"shared observability note for {owner}",
            owner_user_id=owner,
        ))

    spanning = service.admin_search(query="observability", mode="keyword", top_k=10)
    scoped = service.admin_search(query="observability", mode="keyword", top_k=10, owner_user_id="user-1")

    assert spanning.total == 2
    assert scoped.total == 1


def test_chunker_keeps_short_text_and_adds_summary_chunk():
    chunks = JournalChunker().chunk_entry(
        entry_id="journal_entry_test",
        title="GPU indexing fix",
        summary="Use GPU only",
        body="Short but important.",
        tags=["gpu"],
        source_kind="memory",
        client="codex",
    )

    assert chunks[0].level == 1
    assert "GPU indexing fix" in chunks[0].content
    assert any(chunk.level == 2 and chunk.content == "Short but important." for chunk in chunks)


def test_chunker_splits_markdown_headings_and_marks_long_fences_as_detail():
    body = "# Decision\nUse SQL.\n\n## Logs\n```text\n" + ("x\n" * 1400) + "```"

    chunks = JournalChunker().chunk_entry(
        entry_id="journal_entry_test",
        title=None,
        summary=None,
        body=body,
        tags=[],
        source_kind="handoff",
        client="claude",
    )

    assert any(chunk.heading == "Decision" and chunk.level == 2 for chunk in chunks)
    assert any(chunk.level == 3 and chunk.metadata.get("kind") == "fenced_block" for chunk in chunks)


def test_record_persists_sql_and_upserts_vectors():
    vector_store = CapturingVectorStore()
    service = JournalService(
        repository=_repo(),
        vector_store=vector_store,
        embed_fn=_fake_embed,
    )

    response = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="codex",
        source_kind="session_summary",
        title="Journal plan",
        body="Record work memory and retrieve it later.",
        tags=["memory"],
    ))

    assert response.indexed is True
    assert response.chunk_count >= 2
    assert vector_store.upserts
    chunks, kwargs = vector_store.upserts[0]
    assert kwargs["project_slug"] == "retrival"
    assert kwargs["owner_user_id"] is None
    assert kwargs["source_kind"] == "session_summary"
    assert len(chunks) == response.chunk_count


def test_record_keeps_sql_entry_when_vector_index_fails():
    service = JournalService(
        repository=_repo(),
        vector_store=FailingVectorStore(),
        embed_fn=_fake_embed,
    )

    response = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        body="This must not disappear when Qdrant is down.",
    ))
    loaded = service.get_entry(response.entry.entry_id)

    assert response.indexed is False
    assert "qdrant down" in (response.index_error or "")
    assert loaded is not None
    assert loaded.body == "This must not disappear when Qdrant is down."
    assert loaded.chunks


def test_search_dedups_by_entry_and_returns_best_chunk():
    vector_store = CapturingVectorStore()
    service = JournalService(
        repository=_repo(),
        vector_store=vector_store,
        embed_fn=_fake_embed,
    )
    created = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title="Dedup design",
        body="First chunk. Second chunk.",
    ))
    entry_id = created.entry.entry_id
    first_chunk = created.entry.chunks[0].chunk_id
    second_chunk = created.entry.chunks[-1].chunk_id
    vector_store.search_results = [
        {"id": first_chunk, "content": "summary", "score": 0.9, "metadata": {
            "entry_id": entry_id, "chunk_id": first_chunk, "project_id": created.entry.project_id,
            "chunk_index": 0, "level": 1, "heading": "summary",
        }},
        {"id": second_chunk, "content": "body", "score": 0.8, "metadata": {
            "entry_id": entry_id, "chunk_id": second_chunk, "project_id": created.entry.project_id,
            "chunk_index": 1, "level": 2, "heading": "body",
        }},
    ]

    response = service.search(JournalSearchRequest(
        project_slug="retrival",
        query="dedup design",
        top_k=10,
        dedup_by_entry=True,
    ))

    assert response.total == 1
    assert response.hits[0].entry.entry_id == entry_id
    assert response.hits[0].best_chunk.chunk_id == first_chunk


def test_search_filters_vectors_and_sql_by_owner_user_id():
    vector_store = CapturingVectorStore()
    service = JournalService(
        repository=_repo(),
        vector_store=vector_store,
        embed_fn=_fake_embed,
    )
    user_one = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title="User one",
        body="private user one memory",
        owner_user_id="user-1",
    ))
    user_two = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title="User two",
        body="private user two memory",
        owner_user_id="user-2",
    ))
    user_one_chunk = user_one.entry.chunks[-1].chunk_id
    user_two_chunk = user_two.entry.chunks[-1].chunk_id
    vector_store.search_results = [
        {"id": user_two_chunk, "content": "private user two memory", "score": 0.99, "metadata": {
            "entry_id": user_two.entry.entry_id, "chunk_id": user_two_chunk, "project_id": user_two.entry.project_id,
            "chunk_index": 1, "level": 2, "owner_user_id": "user-2",
        }},
        {"id": user_one_chunk, "content": "private user one memory", "score": 0.88, "metadata": {
            "entry_id": user_one.entry.entry_id, "chunk_id": user_one_chunk, "project_id": user_one.entry.project_id,
            "chunk_index": 1, "level": 2, "owner_user_id": "user-1",
        }},
    ]

    response = service.search(JournalSearchRequest(
        project_slug="retrival",
        query="private memory",
        owner_user_id="user-1",
        top_k=5,
        dedup_by_entry=True,
    ))

    assert vector_store.search_calls[-1]["filters"]["owner_user_id"] == "user-1"
    assert response.total == 1
    assert response.hits[0].entry.entry_id == user_one.entry.entry_id


def test_search_reranks_candidate_pool_before_deduping():
    vector_store = CapturingVectorStore()
    service = JournalService(
        repository=_repo(),
        vector_store=vector_store,
        embed_fn=_fake_embed,
        reranker=PreferenceReranker(),
    )
    first = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title="Original top hit",
        body="ordinary retrieval hit",
    ))
    second = service.record(JournalRecordRequest(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title="Rerank winner",
        body="preferred semantic hit",
    ))
    first_chunk = first.entry.chunks[-1].chunk_id
    second_chunk = second.entry.chunks[-1].chunk_id
    vector_store.search_results = [
        {"id": first_chunk, "content": "ordinary retrieval hit", "score": 0.91, "metadata": {
            "entry_id": first.entry.entry_id, "chunk_id": first_chunk, "project_id": first.entry.project_id,
            "chunk_index": 1, "level": 2, "heading": "body",
        }},
        {"id": second_chunk, "content": "preferred semantic hit", "score": 0.44, "metadata": {
            "entry_id": second.entry.entry_id, "chunk_id": second_chunk, "project_id": second.entry.project_id,
            "chunk_index": 1, "level": 2, "heading": "body",
        }},
    ]

    response = service.search(JournalSearchRequest(
        project_slug="retrival",
        query="semantic preference",
        top_k=2,
        dedup_by_entry=True,
        rerank=True,
    ))

    assert response.total == 2
    assert response.hits[0].entry.entry_id == second.entry.entry_id
    assert response.hits[0].score == 0.99
    assert response.hits[0].best_chunk.metadata["reranker_score"] == 0.99
    assert response.metrics["candidates_before_rerank"] == 2
    assert response.metrics["candidates_after_rerank"] == 2


def test_doc_reindex_is_latest_wins_not_append():
    # W153: journal_index_dir records indexed files as source_kind="doc" with
    # dedupe_policy="return_existing". An edited file gets a NEW content_hash, so the
    # hash-based dedup can't catch it — append-only (correct for memory) then leaves
    # BOTH the stale and the current version live, doubling every re-indexed file in
    # search. A doc represents the file's CURRENT content: re-indexing must supersede
    # the prior version (latest-wins), keeping exactly one live version per file.
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    def index_doc(body: str):
        return service.record(JournalRecordRequest(
            project_slug="retrival", client="mcp", source_kind="doc",
            title="plan.md", body=body, tags=["doc"], owner_user_id="user-1",
            dedupe_policy="return_existing",
        ))

    first = index_doc("plan.md original content")

    # Unchanged re-index → hash dedup still applies (no new version, no supersede).
    same = index_doc("plan.md original content")
    assert same.deduped is True
    live = service.list_entries(project_slug="retrival", source_kind="doc", owner_user_id="user-1")
    assert live.total == 1

    # Edited re-index → new hash → supersede the stale version, keep only the current.
    second = index_doc("plan.md edited content with substantially more text")
    assert second.deduped is False
    live = service.list_entries(project_slug="retrival", source_kind="doc", owner_user_id="user-1")
    assert live.total == 1, "edited doc must replace prior version, not accumulate"
    assert live.entries[0].entry_id == second.entry.entry_id
    # The superseded version's vectors were cleared from Qdrant.
    assert first.entry.entry_id in vector_store.deleted


def test_doc_reindex_same_content_different_titles_are_distinct_files():
    # File identity is the relative path title, not only the content hash. Two docs with
    # identical boilerplate must remain two searchable files instead of deduping into one.
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    first = service.record(JournalRecordRequest(
        project_slug="retrival", client="mcp", source_kind="doc",
        title="api/README.md", body="shared project overview", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))
    second = service.record(JournalRecordRequest(
        project_slug="retrival", client="mcp", source_kind="doc",
        title="user/README.md", body="shared project overview", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))

    assert second.deduped is False
    assert second.entry.entry_id != first.entry.entry_id
    live = service.list_entries(project_slug="retrival", source_kind="doc", owner_user_id="user-1")
    assert {entry.title for entry in live.entries} == {"api/README.md", "user/README.md"}


def test_doc_scope_reconcile_deletes_stale_titles_from_snapshot():
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    keep = service.record(JournalRecordRequest(
        project_slug="retrival", client="mcp", source_kind="doc",
        title="keep.md", body="still present", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))
    stale = service.record(JournalRecordRequest(
        project_slug="retrival", client="mcp", source_kind="doc",
        title="stale.md", body="removed from disk", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))

    result = service.reconcile_doc_scope(
        project_slug="retrival",
        live_titles=["keep.md"],
        owner_user_id="user-1",
    )

    assert result.project_slug == "retrival"
    assert result.live_titles == 1
    assert result.removed_entries == 1
    assert stale.entry.entry_id in result.removed_entry_ids
    assert stale.entry.entry_id in vector_store.deleted
    live = service.list_entries(project_slug="retrival", source_kind="doc", owner_user_id="user-1")
    assert live.total == 1
    assert live.entries[0].entry_id == keep.entry.entry_id


def test_windows_reconcile_keeps_case_variant_live_title_and_deletes_real_stale():
    vector_store = CapturingVectorStore()
    service = JournalService(
        repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)
    keep = service.record(JournalRecordRequest(
        project_slug="E:/Repo", client="mcp", source_kind="doc",
        title="Docs/Current.MD", body="current", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))
    stale = service.record(JournalRecordRequest(
        project_slug="E:/Repo", client="mcp", source_kind="doc",
        title="Docs/Stale.MD", body="stale", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))

    result = service.reconcile_doc_scope(
        project_slug=r"e:\repo",
        live_titles=["docs/current.md"],
        owner_user_id="user-1",
        patterns=["*.md"],
    )

    assert result.removed_entry_ids == [stale.entry.entry_id]
    assert keep.entry.entry_id not in result.removed_entry_ids


def test_windows_reconcile_keeps_legacy_backslash_title_in_mixed_snapshot():
    repo = _repo()
    legacy_foo, _, _ = repo.create_entry(
        project_slug="E:/Repo", client="mcp", source_kind="doc",
        title=r"Docs\Foo.md", summary=None, body="legacy foo",
        tags=[], metadata={}, owner_user_id="user-1",
    )
    current_bar, _, _ = repo.create_entry(
        project_slug="E:/Repo", client="mcp", source_kind="doc",
        title="docs/bar.md", summary=None, body="current bar",
        tags=[], metadata={}, owner_user_id="user-1",
    )
    stale, _, _ = repo.create_entry(
        project_slug="E:/Repo", client="mcp", source_kind="doc",
        title=r"Docs\Gone.md", summary=None, body="gone",
        tags=[], metadata={}, owner_user_id="user-1",
    )
    service = JournalService(
        repository=repo, vector_store=CapturingVectorStore(), embed_fn=_fake_embed)

    result = service.reconcile_doc_scope(
        project_slug="E:/repo",
        live_titles=["docs/foo.md", "docs/bar.md"],
        owner_user_id="user-1",
    )

    assert result.removed_entry_ids == [stale.entry_id]
    assert legacy_foo.entry_id not in result.removed_entry_ids
    assert current_bar.entry_id not in result.removed_entry_ids


def test_windows_doc_case_only_rename_remains_latest_wins():
    service = JournalService(
        repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    first = service.record(JournalRecordRequest(
        project_slug="E:/Repo", client="mcp", source_kind="doc",
        title="Docs/A.md", body="old body", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))
    second = service.record(JournalRecordRequest(
        project_slug=r"e:\repo", client="mcp", source_kind="doc",
        title="docs/a.md", body="new body", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))

    live = service.list_entries(
        project_slug="E:/repo", source_kind="doc", owner_user_id="user-1")
    assert [entry.entry_id for entry in live.entries] == [second.entry.entry_id]
    assert first.entry.entry_id != second.entry.entry_id


def test_posix_doc_title_case_variants_remain_distinct():
    service = JournalService(
        repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    for title in ("Docs/A.md", "docs/a.md"):
        service.record(JournalRecordRequest(
            project_slug="/repo", client="mcp", source_kind="doc",
            title=title, body=f"body for {title}", owner_user_id="user-1",
            dedupe_policy="return_existing",
        ))

    live = service.list_entries(
        project_slug="/repo", source_kind="doc", owner_user_id="user-1")
    assert {entry.title for entry in live.entries} == {"Docs/A.md", "docs/a.md"}


def test_doc_scope_reconcile_empty_snapshot_is_safe_noop():
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)
    service.record(JournalRecordRequest(
        project_slug="retrival", client="mcp", source_kind="doc",
        title="keep.md", body="must survive bad empty scan", owner_user_id="user-1",
        dedupe_policy="return_existing",
    ))

    result = service.reconcile_doc_scope(
        project_slug="retrival",
        live_titles=[],
        owner_user_id="user-1",
    )

    assert result.removed_entries == 0
    assert result.skipped is True
    assert "empty" in result.reason
    assert service.list_entries(project_slug="retrival", source_kind="doc", owner_user_id="user-1").total == 1


def test_doc_scope_reconcile_max_depth_keeps_deeper_unscanned_docs():
    # Finding #4 safety valve: a scope first indexed recursively (root + sub/b.md), then
    # re-run NON-recursively (max_depth=0), must not sweep the deeper docs the shallow scan
    # never looked at — only stale titles WITHIN the scan's depth envelope are reconciled.
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    def index_doc(title: str, body: str):
        return service.record(JournalRecordRequest(
            project_slug="retrival", client="mcp", source_kind="doc",
            title=title, body=body, owner_user_id="user-1", dedupe_policy="return_existing",
        ))

    keep_root = index_doc("a.md", "root doc still present")
    deep = index_doc("sub/b.md", "nested doc a shallow scan never sees")
    stale_root = index_doc("gone.md", "root doc removed from disk")

    # A non-recursive re-run sees only root files → live={a.md}, depth envelope 0.
    result = service.reconcile_doc_scope(
        project_slug="retrival",
        live_titles=["a.md"],
        owner_user_id="user-1",
        max_depth=0,
    )

    assert result.removed_entries == 1
    assert stale_root.entry.entry_id in result.removed_entry_ids
    # The deeper doc is outside the shallow scan's envelope → preserved, not swept.
    assert deep.entry.entry_id not in result.removed_entry_ids
    assert deep.entry.entry_id not in vector_store.deleted
    live = {e.title for e in service.list_entries(
        project_slug="retrival", source_kind="doc", owner_user_id="user-1").entries}
    assert live == {"a.md", "sub/b.md"}


def test_doc_scope_reconcile_patterns_keeps_docs_outside_the_scan_glob():
    # Pattern-envelope valve (same class as the depth valve): a scope first indexed with
    # "*.md", then re-run with "*.txt", must NOT sweep the .md docs — this scan's glob never
    # looked for them. Only stale titles matching THIS run's pattern(s) are reconciled.
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    def index_doc(title: str, body: str):
        return service.record(JournalRecordRequest(
            project_slug="retrival", client="mcp", source_kind="doc",
            title=title, body=body, owner_user_id="user-1", dedupe_policy="return_existing",
        ))

    keep_md = index_doc("notes.md", "markdown the *.txt scan never looks for")
    keep_txt = index_doc("current.txt", "still present on disk")
    stale_txt = index_doc("gone.txt", "removed .txt from disk")

    # A "*.txt" re-run: live snapshot is the surviving .txt file, envelope = ["*.txt"].
    result = service.reconcile_doc_scope(
        project_slug="retrival",
        live_titles=["current.txt"],
        owner_user_id="user-1",
        patterns=["*.txt"],
    )

    assert result.removed_entries == 1
    assert stale_txt.entry.entry_id in result.removed_entry_ids
    # The .md doc is outside the *.txt envelope → preserved (not a "disappeared file").
    assert keep_md.entry.entry_id not in result.removed_entry_ids
    assert keep_md.entry.entry_id not in vector_store.deleted
    live = {e.title for e in service.list_entries(
        project_slug="retrival", source_kind="doc", owner_user_id="user-1").entries}
    assert live == {"notes.md", "current.txt"}
    assert keep_txt  # (referenced) the surviving .txt stays live via the snapshot


def test_delete_doc_scope_clears_all_docs_but_keeps_memory():
    # Symmetric to delete_code: an explicit scope clear drops EVERY source_kind="doc" entry
    # (+ its vectors) in the scope, but leaves ACE memory entries (source_kind="memory")
    # untouched — journal_record memories and journal_index_dir docs coexist in one scope.
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)

    def rec(kind, title, body):
        return service.record(JournalRecordRequest(
            project_slug="retrival", client="mcp", source_kind=kind,
            title=title, body=body, owner_user_id="user-1", dedupe_policy="return_existing",
        ))

    d1 = rec("doc", "a.md", "doc one")
    d2 = rec("doc", "sub/b.md", "doc two")
    mem = rec("memory", None, "a lesson worth keeping")

    result = service.delete_doc_scope(project_slug="retrival", owner_user_id="user-1")

    assert result.removed_entries == 2
    assert set(result.removed_entry_ids) == {d1.entry.entry_id, d2.entry.entry_id}
    assert d1.entry.entry_id in vector_store.deleted
    assert d2.entry.entry_id in vector_store.deleted
    # memory survives
    assert mem.entry.entry_id not in vector_store.deleted
    live_docs = service.list_entries(project_slug="retrival", source_kind="doc", owner_user_id="user-1")
    assert live_docs.total == 0
    live_mem = service.list_entries(project_slug="retrival", source_kind="memory", owner_user_id="user-1")
    assert live_mem.total == 1


def test_delete_doc_scope_empty_is_idempotent_noop():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    result = service.delete_doc_scope(project_slug="nothing-here", owner_user_id="user-1")
    assert result.removed_entries == 0
    assert result.removed_entry_ids == []


def test_doc_reindex_dedup_still_supersedes_stale_duplicates():
    # W153 backfill: the pre-fix state left two live versions of one file. A later
    # re-index whose content matches the CURRENT version dedups on content_hash — but
    # latest-wins means exactly one live version per file, so the OTHER stale same-file
    # version must still be swept even on the dedup path.
    repo = _repo()
    vector_store = CapturingVectorStore()
    service = JournalService(repository=repo, vector_store=vector_store, embed_fn=_fake_embed)
    # Two accumulated versions (append never supersedes) = the pre-fix duplicate state.
    stale, _, _ = repo.create_entry(
        project_slug="retrival", client="http", source_kind="doc", title="CLAUDE.md",
        summary=None, body="stale 03:17 content", tags=["doc"], metadata={},
        owner_user_id="user-1", dedupe_policy="append")
    current, _, _ = repo.create_entry(
        project_slug="retrival", client="mcp", source_kind="doc", title="CLAUDE.md",
        summary=None, body="current 08:07 content", tags=["doc"], metadata={},
        owner_user_id="user-1", dedupe_policy="append")
    assert service.list_entries(
        project_slug="retrival", source_kind="doc", owner_user_id="user-1").total == 2

    # Re-index the unchanged current content → dedups to `current`, sweeps `stale`.
    resp = service.record(JournalRecordRequest(
        project_slug="retrival", client="mcp", source_kind="doc", title="CLAUDE.md",
        body="current 08:07 content", tags=["doc"], owner_user_id="user-1",
        dedupe_policy="return_existing"))
    assert resp.deduped is True
    live = service.list_entries(project_slug="retrival", source_kind="doc", owner_user_id="user-1")
    assert live.total == 1
    assert live.entries[0].entry_id == current.entry_id
    assert stale.entry_id in vector_store.deleted


def test_memory_stays_append_only():
    # W153 must not touch memory semantics: distinct notes still accumulate, and the
    # supersede path is scoped to source_kind="doc" only.
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)
    for body in ("insight one about retrieval", "insight two about ranking"):
        service.record(JournalRecordRequest(
            project_slug="retrival", client="codex", source_kind="memory",
            title="scratch", body=body, owner_user_id="user-1",
        ))
    live = service.list_entries(project_slug="retrival", source_kind="memory", owner_user_id="user-1")
    assert live.total == 2
    assert vector_store.deleted == []


# --------------------------------------------------------------------------- #
# 方案2: cross-scope write-time fan-out for indexed docs (text side)
# --------------------------------------------------------------------------- #
def test_doc_fan_out_refreshes_same_file_in_other_scope_both_complete():
    """Recording a doc that also lives under another scope (same owner, matched by
    absolute path) refreshes that scope's copy to the latest content and reports the
    count. Both scopes stay complete (no docs dropped/added on either side)."""
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(),
                             embed_fn=_fake_embed)
    # Child scope /proj/sub: x.txt (OLD) + its own y.txt.
    service.record(JournalRecordRequest(project_slug="/proj/sub", client="mcp", source_kind="doc",
                                        title="x.txt", body="alpha OLD content",
                                        dedupe_policy="return_existing"))
    service.record(JournalRecordRequest(project_slug="/proj/sub", client="mcp", source_kind="doc",
                                        title="y.txt", body="child only content",
                                        dedupe_policy="return_existing"))
    # Parent scope /proj: top.txt + sub/x.txt (the SAME physical file, NEW content).
    service.record(JournalRecordRequest(project_slug="/proj", client="mcp", source_kind="doc",
                                        title="top.txt", body="parent only content",
                                        dedupe_policy="return_existing"))
    resp = service.record(JournalRecordRequest(project_slug="/proj", client="mcp", source_kind="doc",
                                               title="sub/x.txt", body="alpha NEW content",
                                               dedupe_policy="return_existing"))

    assert resp.duplicates_updated == 1  # only sub/x.txt == /proj/sub/x.txt matched
    # Child scope still complete; shared file refreshed; child-only file untouched.
    child = {e.title: e for e in service.list_entries(project_slug="/proj/sub",
                                                       source_kind="doc").entries}
    assert set(child) == {"x.txt", "y.txt"}, "child must stay complete (no drop, no foreign add)"
    assert "NEW" in service.get_entry(child["x.txt"].entry_id).body, "shared file must refresh"
    assert "child only" in service.get_entry(child["y.txt"].entry_id).body
    # Parent scope complete.
    parent = {e.title for e in service.list_entries(project_slug="/proj",
                                                    source_kind="doc").entries}
    assert parent == {"top.txt", "sub/x.txt"}


def test_doc_fan_out_zero_when_no_other_scope_shares_file():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(),
                             embed_fn=_fake_embed)
    service.record(JournalRecordRequest(project_slug="/a", client="mcp", source_kind="doc",
                                        title="a.txt", body="content a",
                                        dedupe_policy="return_existing"))
    resp = service.record(JournalRecordRequest(project_slug="/b", client="mcp", source_kind="doc",
                                               title="b.txt", body="content b",
                                               dedupe_policy="return_existing"))
    assert resp.duplicates_updated == 0


def test_memory_entries_never_fan_out():
    """ACE memory is append-only, per-scope — it must never sync across scopes."""
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(),
                             embed_fn=_fake_embed)
    service.record(JournalRecordRequest(project_slug="/proj/sub", client="codex",
                                        source_kind="memory", body="a lesson learned here"))
    resp = service.record(JournalRecordRequest(project_slug="/proj", client="codex",
                                               source_kind="memory", body="a lesson learned here"))
    assert resp.duplicates_updated == 0


# --------------------------------------------------------------------------- #
# path_glob: restrict journal search to a subtree / single doc within a scope
# (symmetric with search_code's path_glob; title == the doc's rel path).
# --------------------------------------------------------------------------- #
def _doc(service, title, body="shared bandwidth limiter marker"):
    return service.record(JournalRecordRequest(
        project_slug="p", client="mcp", source_kind="doc",
        title=title, body=body, dedupe_policy="return_existing"))


def test_keyword_search_path_glob_restricts_to_subtree():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    for t in ("sub/a.md", "b.md", "sub/deep/c.md"):
        _doc(service, t)
    resp = service.search(JournalSearchRequest(
        project_slug="p", query="bandwidth limiter marker", mode="keyword",
        top_k=10, path_glob="sub/"))
    assert {h.entry.title for h in resp.hits} == {"sub/a.md", "sub/deep/c.md"}


def test_windows_scope_path_glob_is_case_insensitive_and_accepts_backslashes():
    """Directory semantics match Code KB when the journal scope is an NTFS path."""
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    scope = "E:/jetb/pythons/playground/octopus/axonhub/llm"
    service.record(JournalRecordRequest(
        project_slug=scope,
        client="mcp",
        source_kind="doc",
        title="Docs/Compact/Guide.md",
        body="RequestTypeCompact guide marker",
        dedupe_policy="return_existing",
    ))

    response = service.search(JournalSearchRequest(
        project_slug=scope,
        query="RequestTypeCompact",
        mode="keyword",
        top_k=5,
        path_glob="docs\\compact\\",
    ))

    assert {hit.entry.title for hit in response.hits} == {"Docs/Compact/Guide.md"}


def test_windows_search_reads_legacy_mixed_case_project_slug():
    repo = _repo()
    # Bypass current request validation to model data written before canonical scopes.
    repo.create_entry(
        project_slug="E:/JETB/Repo",
        client="mcp",
        source_kind="doc",
        title="Docs/Compact.md",
        summary=None,
        body="RequestTypeCompact legacy marker",
        tags=[],
        metadata={},
        owner_user_id="user-1",
    )
    service = JournalService(
        repository=repo, vector_store=CapturingVectorStore(), embed_fn=_fake_embed)

    response = service.search(JournalSearchRequest(
        project_slug=r"e:\jetb\repo",
        query="RequestTypeCompact",
        mode="keyword",
        owner_user_id="user-1",
    ))

    assert {hit.entry.title for hit in response.hits} == {"Docs/Compact.md"}


def test_windows_list_and_delete_include_legacy_mixed_case_project_slug():
    repo = _repo()
    entry, _, _ = repo.create_entry(
        project_slug="E:/JETB/Repo",
        client="mcp",
        source_kind="doc",
        title="Docs/Compact.md",
        summary=None,
        body="legacy scope doc",
        tags=[],
        metadata={},
        owner_user_id="user-1",
    )
    service = JournalService(
        repository=repo, vector_store=CapturingVectorStore(), embed_fn=_fake_embed)

    listed = service.list_entries(
        project_slug=r"e:\jetb\repo", source_kind="doc", owner_user_id="user-1")
    deleted = service.delete_doc_scope(
        project_slug=r"e:\jetb\repo", owner_user_id="user-1")

    assert [item.entry_id for item in listed.entries] == [entry.entry_id]
    assert deleted.removed_entry_ids == [entry.entry_id]


def test_journal_path_glob_reads_legacy_stored_leading_slash_without_raising():
    repo = _repo()
    # Bypass the current strict request schema to represent a historical stored title.
    repo.create_entry(
        project_slug="p", client="mcp", source_kind="doc",
        title="/docs/compact/guide.md", summary=None,
        body="RequestTypeCompact guide marker", tags=[], metadata={},
    )
    service = JournalService(
        repository=repo, vector_store=CapturingVectorStore(), embed_fn=_fake_embed)

    response = service.search(JournalSearchRequest(
        project_slug="p", query="RequestTypeCompact", mode="keyword", top_k=5,
        path_glob="docs/compact/",
    ))

    assert {hit.entry.title for hit in response.hits} == {"/docs/compact/guide.md"}


def test_keyword_search_path_glob_single_file():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    for t in ("sub/a.md", "sub/b.md"):
        _doc(service, t)
    resp = service.search(JournalSearchRequest(
        project_slug="p", query="bandwidth limiter marker", mode="keyword",
        top_k=10, path_glob="sub/a.md"))
    assert {h.entry.title for h in resp.hits} == {"sub/a.md"}


def test_keyword_search_path_glob_star_extension():
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    _doc(service, "notes/a.md")
    _doc(service, "notes/a.txt")
    resp = service.search(JournalSearchRequest(
        project_slug="p", query="bandwidth limiter marker", mode="keyword",
        top_k=10, path_glob="*.md"))
    assert {h.entry.title for h in resp.hits} == {"notes/a.md"}


def test_keyword_search_path_glob_single_file_is_exact():
    """A single-file path_glob must return ONLY that file, never a same-stem sibling
    (the startswith bug: 'sub/a.md' must not also match 'sub/a.mdx')."""
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    for t in ("sub/a.md", "sub/a.mdx", "sub/a.md.bak"):
        _doc(service, t)
    resp = service.search(JournalSearchRequest(
        project_slug="p", query="bandwidth limiter marker", mode="keyword",
        top_k=10, path_glob="sub/a.md"))
    assert {h.entry.title for h in resp.hits} == {"sub/a.md"}


def test_keyword_search_path_glob_batches_large_match_set():
    """A path_glob match-set larger than the SQL IN batch must still return correct hits
    (batched + merged), not raise 'too many SQL variables'."""
    from app.services.journal import fts
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    n = fts._IN_BATCH + 25
    for i in range(n):
        _doc(service, f"sub/f{i}.md")
    _doc(service, "other/z.md")
    resp = service.search(JournalSearchRequest(
        project_slug="p", query="bandwidth limiter marker", mode="keyword",
        top_k=5, path_glob="sub/"))
    assert resp.hits and all(h.entry.title.startswith("sub/") for h in resp.hits)


def test_path_glob_no_match_but_scope_has_docs_returns_notice():
    """A path_glob that matches nothing while the scope DOES have docs returns a notice
    (0-of-N + a sample title) — so the user sees path_glob is relative to THIS scope's root
    and their glob depth is wrong, instead of a bare 'no matching' that hides the real docs.
    This is the exact 服务计算 case: flat basenames vs a sub-dir glob."""
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    for t in ("flat_a.md", "flat_b.md"):  # flat basenames — no sub-dir
        _doc(service, t)
    resp = service.search(JournalSearchRequest(
        project_slug="p", query="bandwidth limiter marker", mode="keyword",
        top_k=10, path_glob="subdir/*.md"))
    assert resp.hits == []
    assert resp.notice is not None
    assert "0 of 2" in resp.notice
    assert "flat_a.md" in resp.notice  # a sample title reveals the actual (flat) path shape


def test_path_glob_no_match_empty_scope_has_no_notice():
    """When the scope genuinely has no docs, a path_glob miss returns NO notice — a hint
    about 'docs in scope' would be misleading when there are none."""
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    resp = service.search(JournalSearchRequest(
        project_slug="p", query="x", mode="keyword", top_k=10, path_glob="sub/*.md"))
    assert resp.hits == []
    assert resp.notice is None


def test_semantic_path_glob_no_match_but_scope_has_docs_returns_notice():
    """Same notice on the semantic (qdrant) path, not just keyword."""
    service = JournalService(repository=_repo(), vector_store=CapturingVectorStore(), embed_fn=_fake_embed)
    _doc(service, "flat_a.md")
    resp = service.search(JournalSearchRequest(
        project_slug="p", query="bandwidth limiter marker", mode="hybrid",
        top_k=10, path_glob="subdir/*.md"))
    assert resp.hits == []
    assert resp.notice is not None and "0 of 1" in resp.notice


def test_semantic_search_path_glob_restricts_to_subtree():
    vector_store = CapturingVectorStore()
    service = JournalService(repository=_repo(), vector_store=vector_store, embed_fn=_fake_embed)
    a = _doc(service, "sub/a.md", body="alpha marker")
    b = _doc(service, "b.md", body="beta marker")

    def _hit(created, score):
        ch = created.entry.chunks[0]
        return {"id": ch.chunk_id, "content": "x", "score": score, "metadata": {
            "entry_id": created.entry.entry_id, "chunk_id": ch.chunk_id,
            "project_id": created.entry.project_id, "chunk_index": 0, "level": 1, "heading": "h"}}
    vector_store.search_results = [_hit(b, 0.9), _hit(a, 0.8)]  # b ranks higher but is filtered out

    resp = service.search(JournalSearchRequest(
        project_slug="p", query="marker", top_k=10, path_glob="sub/"))
    assert {h.entry.title for h in resp.hits} == {"sub/a.md"}
