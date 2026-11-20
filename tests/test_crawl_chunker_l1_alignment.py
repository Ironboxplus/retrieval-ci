"""TDD: webpage indexing must produce L1+L2 chunks aligned with paper corpus.

Bug discovered 2026-05-20: crawl_tasks._index_crawled_result uses
chunker_type='html' which routes through SectionAwareChunker. That class
emits only L2 chunks — no L1 title_summary. Effect: searches against
webpages bypass the L1 retrieval signal that papers benefit from.

Fix: route webpage indexing through SemanticSectionChunker
(semantic_inner_split=False) so L1 (title + first N tokens) is emitted.
Same schema as Stage R ICLR ingest.

Contract:
1. crawl_tasks._index_crawled_result uses chunker_type='section_l1' (not 'html')
2. create_chunker('section_l1') returns SemanticSectionChunker
3. Chunking an HTML page with content_type='html' produces:
   - exactly 1 L1 chunk with level=1, metadata.section_name='title_summary'
   - >=1 L2 chunks with level=2
"""
from __future__ import annotations

import inspect
import pytest

WEBPAGE_HTML = """<html>
<head><title>Demo Page</title></head>
<body>
<h1>Demo Page Title</h1>
<p>This is the introduction paragraph that has enough text to seed the L1
summary chunk. We are testing whether the chunker emits both a title-level
summary chunk and section-level body chunks for webpage content.</p>
<h2>Section Alpha</h2>
<p>Section alpha discusses retrieval pipelines. Dense plus sparse fusion
via reciprocal rank fusion is the canonical hybrid retrieval recipe used in
modern dual-encoder systems.</p>
<h2>Section Beta</h2>
<p>Section beta talks about chunkers. Section-aware chunkers split along
heading hierarchy, while semantic chunkers split along embedding-space
similarity drop-offs. This project uses a hybrid of both.</p>
</body>
</html>"""


def test_index_crawled_result_uses_section_l1_chunker():
    """The crawl indexing path must call get_index_service with section_l1."""
    from app.tasks import crawl_tasks

    src = inspect.getsource(crawl_tasks._index_crawled_result)
    assert "chunker_type=\"section_l1\"" in src or "chunker_type='section_l1'" in src, (
        "_index_crawled_result must request the section_l1 chunker so webpage "
        "indexing emits L1+L2 chunks (Stage R schema). Current source:\n"
        + src[:600]
    )


def test_section_l1_factory_returns_semantic_section_chunker():
    from core.indexing.chunker import create_chunker, SemanticSectionChunker

    chunker = create_chunker(content_type="section_l1")
    assert isinstance(chunker, SemanticSectionChunker), (
        f"Expected SemanticSectionChunker, got {type(chunker).__name__}"
    )
    assert chunker.semantic_inner_split is False, (
        "section_l1 must skip bge-m3 boundary detection (fast mode)"
    )


def test_webpage_html_produces_l1_and_l2_chunks():
    from core.indexing.chunker import create_chunker

    chunker = create_chunker(content_type="section_l1")
    chunks = chunker.chunk_document(
        doc_id="test_webpage",
        text=WEBPAGE_HTML,
        content_type="html",
    )
    assert chunks, "chunker returned no chunks for valid HTML input"

    l1_chunks = [c for c in chunks if c.level == 1]
    l2_chunks = [c for c in chunks if c.level == 2]

    assert len(l1_chunks) == 1, (
        f"Expected exactly 1 L1 chunk, got {len(l1_chunks)}. "
        f"All levels: {[c.level for c in chunks]}"
    )
    assert l2_chunks, (
        f"Expected >=1 L2 chunks, got {len(l2_chunks)}. "
        f"All levels: {[c.level for c in chunks]}"
    )

    l1 = l1_chunks[0]
    assert l1.metadata.get("section_name") == "title_summary", (
        f"L1 chunk metadata.section_name must be 'title_summary', got "
        f"{l1.metadata!r}"
    )
    assert l1.content and l1.content.strip(), "L1 chunk must have non-empty content"
