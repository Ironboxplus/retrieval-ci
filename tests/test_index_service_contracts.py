"""TDD: IndexService contracts that crawl/document/paper pipelines depend on.

Three contracts pinned here (all surfaced via task 53 real-network debug 2026-05-20):

1. **Singleton cache by chunker_type** — every `IndexService()` constructor
   path lazily reloads BGE-M3 from disk; under concurrent crawl + index, that
   race-loads the .pt file and 2/3 docs fail with "Unable to load weights from
   pytorch checkpoint file". Fix: shared cached instances keyed by chunker_type.

2. **content_type pass-through** — `index_document(content_type=...)` must
   propagate to chunker.chunk_document(content_type=...) so HTML inputs get
   tags stripped before section detection. Without this, webpage crawl produces
   garbage chunks (HTML tags get interpreted as markdown plaintext).

3. **Webpage source_type defaults to HTML chunker** — when crawl calls
   index_document(source_type='webpage') without explicit chunker_type, the
   service must pick a HTML-aware chunker (SectionAwareChunker), not the
   legacy multigranularity TextChunker that produces L1/L2/L3 chunks.
"""
from __future__ import annotations

import pytest


def test_get_index_service_returns_singleton_per_chunker_type():
    """Two calls with same chunker_type must return same instance (eliminates model reload race)."""
    from app.services.index_service import get_index_service

    a = get_index_service(chunker_type="html")
    b = get_index_service(chunker_type="html")
    assert a is b, "Same chunker_type must return cached singleton"


def test_get_index_service_distinct_for_distinct_chunker_type():
    from app.services.index_service import get_index_service

    a = get_index_service(chunker_type="html")
    b = get_index_service(chunker_type="section_l1")
    assert a is not b


def test_get_index_service_caches_default_path():
    from app.services.index_service import get_index_service

    a = get_index_service()
    b = get_index_service()
    assert a is b


def test_chunker_html_strips_tags_via_content_type():
    """The HTML chunker route must strip tags so chunk content is plain text."""
    from core.indexing.chunker import create_chunker

    html = "<html><body><h1>Title</h1><p>Para one " + ("word " * 200) + "</p></body></html>"
    chunker = create_chunker(content_type="html")
    chunks = chunker.chunk_document(doc_id="t", text=html, content_type="html")
    assert chunks
    sample = chunks[0].content
    assert "<html>" not in sample and "<body>" not in sample and "<p>" not in sample, (
        f"HTML tags not stripped from chunk content: {sample[:120]!r}"
    )
    assert "word" in sample.lower() or "para" in sample.lower(), (
        f"Visible text not preserved in chunk: {sample[:120]!r}"
    )


def test_index_document_pipes_content_type_to_chunker():
    """index_document(content_type='html') must pass to chunker.chunk_document."""
    from unittest.mock import MagicMock, patch
    from app.services.index_service import IndexService

    captured = {}

    def fake_chunk_document(*, doc_id, text, **kwargs):
        captured["doc_id"] = doc_id
        captured["text"] = text
        captured["kwargs"] = kwargs
        return [{"id": "c1", "content": "stripped text", "level": 1, "metadata": {}}]

    fake_chunker = MagicMock()
    fake_chunker.chunk_document.side_effect = fake_chunk_document

    fake_embed_result = MagicMock()
    fake_embed_result.embed_chunks = MagicMock(return_value=([
        MagicMock(id="c1", content="stripped text", metadata={}, level=1)
    ], None))

    svc = IndexService(chunker_type="html")
    with patch.object(IndexService, "chunker", new=fake_chunker), \
         patch.object(IndexService, "new_embedder", new=fake_embed_result), \
         patch.object(IndexService, "chroma_store", new=MagicMock(add_chunks=MagicMock())), \
         patch.object(IndexService, "_save_chunks_to_db"), \
         patch.object(IndexService, "_ensure_index_device_ready"):
        svc.index_document(
            doc_id=999_999,
            content="<html><body><h1>X</h1><p>body content here</p></body></html>",
            title="Test",
            source="https://test.example",
            source_type="webpage",
            content_type="html",
        )

    assert captured.get("kwargs", {}).get("content_type") == "html", (
        f"index_document did not propagate content_type='html' to chunker; got kwargs={captured.get('kwargs')}"
    )


def test_index_document_webpage_source_auto_html_when_no_chunker_type():
    """source_type='webpage' should auto-infer content_type='html' for the chunker call."""
    from unittest.mock import MagicMock, patch
    from app.services.index_service import IndexService

    captured = {}

    def fake_chunk_document(*, doc_id, text, **kwargs):
        captured["kwargs"] = kwargs
        return [{"id": "c1", "content": "x", "level": 1, "metadata": {}}]

    fake_chunker = MagicMock()
    fake_chunker.chunk_document.side_effect = fake_chunk_document

    fake_embedder = MagicMock()
    fake_embedder.embed_chunks = MagicMock(return_value=([
        MagicMock(id="c1", content="x", metadata={}, level=1)
    ], None))

    svc = IndexService()
    with patch.object(IndexService, "chunker", new=fake_chunker), \
         patch.object(IndexService, "new_embedder", new=fake_embedder), \
         patch.object(IndexService, "chroma_store", new=MagicMock(add_chunks=MagicMock())), \
         patch.object(IndexService, "_save_chunks_to_db"), \
         patch.object(IndexService, "_ensure_index_device_ready"):
        svc.index_document(
            doc_id=999_998,
            content="<html><body>hello</body></html>",
            title="Test",
            source="https://x.test",
            source_type="webpage",
        )

    assert captured.get("kwargs", {}).get("content_type") == "html", (
        f"source_type='webpage' should auto-set content_type='html'; got {captured.get('kwargs')}"
    )
