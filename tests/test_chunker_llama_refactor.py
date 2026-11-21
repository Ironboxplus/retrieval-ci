"""TDD spec for chunker.py library-first refactor.

These tests assert that the chunker delegates to llama-index NodeParsers
instead of langchain-text-splitters. Old SectionAwareChunker uses
``MarkdownHeaderTextSplitter`` + ``RecursiveCharacterTextSplitter`` from
langchain — the refactor should drop those and route through:

  - ``MarkdownNodeParser``  (header-aware split, preserves section_name)
  - ``HTMLNodeParser``      (HTML strip + element-aware split)
  - ``SentenceSplitter``    (size-bounded fallback for sections)
  - ``SemanticSplitterNodeParser`` (semantic boundaries within sections)

Behavior tests of ``chunk_document`` already exist in
``test_section_chunker.py`` and ``test_section_chunker_fallback.py`` —
those test the public contract and must continue to pass through the
refactor. This file adds the *implementation* contract: which library
actually runs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.indexing.chunker import (
    Chunker,
    SectionAwareChunker,
    SemanticSectionChunker,
    create_chunker,
)


SAMPLE_MARKDOWN = """# Paper Title

Authors: A, B

# 1 INTRODUCTION

Deep learning has revolutionized many fields. This work proposes a novel
approach that improves accuracy while reducing compute. We achieve
state-of-the-art results on multiple benchmarks.

## 1.1 Motivation

Existing methods suffer from quadratic complexity. We address this with a
linear-time alternative that preserves expressivity.

# 2 METHOD

Our method consists of three stages: encoding, reasoning, and decoding.
Each stage uses a different architectural pattern.

## 2.1 Encoding

We use a transformer encoder with rotary positional embeddings.
"""

SAMPLE_HTML = """<html><body>
<h1>Title</h1>
<p>Intro paragraph with enough text to be a real chunk and not get
filtered as noise. This sentence adds length.</p>
<h2>Section A</h2>
<p>Body of section A. Another sentence here to pad. And a third sentence
for good measure so the chunker has material to work with.</p>
<script>console.log('should be stripped');</script>
<style>.x { color: red; }</style>
</body></html>"""


def test_markdown_chunker_invokes_llama_index_markdown_node_parser():
    """The markdown path must use llama-index MarkdownNodeParser, not
    langchain MarkdownHeaderTextSplitter."""
    from llama_index.core.node_parser import MarkdownNodeParser

    real_get_nodes = MarkdownNodeParser.get_nodes_from_documents
    call_count = {"n": 0}

    def spy(self, docs, *a, **kw):
        call_count["n"] += 1
        return real_get_nodes(self, docs, *a, **kw)

    with patch.object(MarkdownNodeParser, "get_nodes_from_documents", spy):
        chunker = create_chunker("markdown")
        chunks = chunker.chunk_document(doc_id="doc1", text=SAMPLE_MARKDOWN)

    assert call_count["n"] >= 1, "MarkdownNodeParser.get_nodes_from_documents was never called"
    assert len(chunks) >= 2, f"expected ≥2 chunks from headered markdown, got {len(chunks)}"
    assert all(c.metadata.get("section_name") for c in chunks), \
        "every chunk should carry section_name metadata"


def test_html_chunker_invokes_llama_index_html_node_parser():
    """The html path must use llama-index HTMLNodeParser (which uses bs4
    internally) instead of our hand-rolled BeautifulSoup strip."""
    from llama_index.core.node_parser import HTMLNodeParser

    real_get_nodes = HTMLNodeParser.get_nodes_from_documents
    call_count = {"n": 0}

    def spy(self, docs, *a, **kw):
        call_count["n"] += 1
        return real_get_nodes(self, docs, *a, **kw)

    with patch.object(HTMLNodeParser, "get_nodes_from_documents", spy):
        chunker = create_chunker("html")
        # html content_type must be passed through so the chunker hits
        # the html flatten path inside _preprocess.
        chunks = chunker.chunk_document(doc_id="doc1", text=SAMPLE_HTML, content_type="html")

    assert call_count["n"] >= 1, "HTMLNodeParser.get_nodes_from_documents was never called"
    assert len(chunks) >= 1
    body_text = " ".join(c.content for c in chunks)
    assert "console.log" not in body_text, "script/style content must be stripped by HTMLNodeParser"


def test_semantic_chunker_invokes_llama_index_semantic_splitter_for_long_sections():
    """The semantic path must use llama-index SemanticSplitterNodeParser
    when a section exceeds min_section_chars."""
    from llama_index.core.node_parser import SemanticSplitterNodeParser

    SemanticSectionChunker.reset_cache()

    long_text = (
        "# Section\n\n"
        + ("This is a representative sentence that takes up some space. " * 80)
    )

    call_count = {"n": 0}
    real = SemanticSplitterNodeParser.get_nodes_from_documents

    def spy(self, docs, *a, **kw):
        call_count["n"] += 1
        # Return docs as a single node to avoid running the real embedder
        from llama_index.core.schema import TextNode
        return [TextNode(text=d.text) for d in docs]

    with patch.object(SemanticSplitterNodeParser, "get_nodes_from_documents", spy):
        chunker = SemanticSectionChunker(min_section_chars=100)
        chunker._semantic_parser_cache = SemanticSplitterNodeParser.__new__(SemanticSplitterNodeParser)
        chunks = chunker.chunk_document(doc_id="doc1", text=long_text)

    assert call_count["n"] >= 1, "SemanticSplitterNodeParser was not invoked for long section"
    assert len(chunks) >= 1


def test_semantic_embed_adapter_yields_between_bounded_background_batches():
    import numpy as np
    from types import SimpleNamespace
    from core.indexing.chunker import _BgeM3LlamaAdapter

    class BatchSpyEmbedder:
        _EMBED_BATCH = 3

        def __init__(self):
            self.calls = []

        def embed_texts(self, texts, *, priority="interactive"):
            self.calls.append((priority, len(texts)))
            return SimpleNamespace(
                dense=np.asarray([[float(index), 1.0] for index, _ in enumerate(texts)]))

    embedder = BatchSpyEmbedder()
    adapter = _BgeM3LlamaAdapter.build(embedder)
    dense = adapter._get_text_embeddings([f"sentence {index}" for index in range(8)])

    assert embedder.calls == [
        ("background", 3), ("background", 3), ("background", 2),
    ]
    assert len(dense) == 8


def test_chunker_header_split_uses_llama_index_not_langchain():
    """The HEADER split must use llama-index MarkdownNodeParser. The
    size-bound fallback can still use langchain's RecursiveCharacterTextSplitter
    (the right tool for char-bound jobs; llama-index's SentenceSplitter
    is token-based and makes ``max_chunk_chars`` a leaky abstraction).

    Concretely: the legacy ``_header_splitter`` attribute (which used to
    point at langchain ``MarkdownHeaderTextSplitter``) must be gone, and
    ``_md_parser`` must come from ``llama_index``."""
    chunker = create_chunker("markdown")

    assert not hasattr(chunker, "_header_splitter"), (
        "_header_splitter (langchain) should be removed by the refactor; "
        "use llama-index MarkdownNodeParser instead"
    )
    assert hasattr(chunker, "_md_parser"), \
        "refactored chunker should expose _md_parser (llama_index MarkdownNodeParser)"
    md_mod = type(chunker._md_parser).__module__
    assert md_mod.startswith("llama_index"), \
        f"_md_parser must be llama_index, got {md_mod}"


def test_factory_routes_html_type():
    """Factory must accept 'html' (today it routes through SectionAware's
    auto-detect — after refactor, can return a dedicated wrapper or
    still SectionAware, but must not raise)."""
    chunker = create_chunker("html")
    assert isinstance(chunker, Chunker)
    chunks = chunker.chunk_document(doc_id="d", text=SAMPLE_HTML)
    assert len(chunks) >= 1


def test_section_name_metadata_from_markdown_header():
    """MarkdownNodeParser populates header metadata. The refactored
    chunker must map that to ``Chunk.metadata['section_name']`` so
    downstream consumers don't need to learn llama-index's key names."""
    chunker = create_chunker("markdown")
    chunks = chunker.chunk_document(doc_id="d", text=SAMPLE_MARKDOWN)
    section_names = {c.metadata.get("section_name") for c in chunks}
    # At least the H1 sections should appear
    assert any("INTRODUCTION" in (n or "").upper() for n in section_names), \
        f"expected an INTRODUCTION section_name; got {section_names}"
