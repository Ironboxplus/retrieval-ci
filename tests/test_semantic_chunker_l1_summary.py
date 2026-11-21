"""TDD spec for SemanticSectionChunker L1 (title + 500 bge-m3 token) emission.

Per 2026-05-19 user-mandated schema: every paper gets one L1 chunk = title +
first 500 bge-m3 tokens of body, plus N L2 chunks from semantic section split.
No L3 (legacy 512-char fine-grain dropped). title_only filter (level=1) hits L1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.indexing.chunker import SemanticSectionChunker


SAMPLE_MD = """# A Novel Approach to Linear Attention

Authors: Alice, Bob, Carol

# 1 INTRODUCTION

Transformers have quadratic complexity in sequence length. This work proposes
a linear-time alternative that preserves expressivity. We achieve SOTA on
multiple benchmarks while reducing FLOPs by 10x. Our method scales to 1M
context length on a single GPU.

# 2 METHOD

Our method consists of three stages: encoding, reasoning, and decoding. The
encoder uses linear attention with kernel feature maps. The reasoning module
applies iterative refinement. Decoding is standard transformer with KV cache.

## 2.1 Linear Attention

We use the Performer kernel approximation. This avoids the O(N^2) softmax
computation. Memory is O(N*d) instead of O(N^2).

## 2.2 Iterative Refinement

The reasoner runs K=8 iterations. Each iteration updates hidden state via
gated residual connections. Convergence is monitored by L2 norm change.

# 3 EXPERIMENTS

We evaluate on Long Range Arena, PG19, and arxiv-pdf benchmarks. Our method
matches full transformer accuracy at 5x throughput. Ablations show the kernel
choice matters less than the iteration count.

# 4 CONCLUSION

Linear attention with iterative refinement is a practical alternative to full
softmax attention for long contexts.
"""


def test_l1_chunk_emitted_first_and_unique():
    """SemanticSectionChunker emits exactly one L1 chunk at the start."""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("doc1", SAMPLE_MD)
    l1_chunks = [c for c in chunks if c.level == 1]
    assert len(l1_chunks) == 1, f"expected exactly 1 L1 chunk, got {len(l1_chunks)}"
    # L1 should be the first chunk in returned list
    assert chunks[0].level == 1, "L1 must come before L2 chunks"


def test_l1_content_includes_paper_title():
    """L1 chunk content must include the first H1 title text."""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("doc1", SAMPLE_MD)
    l1 = next(c for c in chunks if c.level == 1)
    assert "Novel Approach to Linear Attention" in l1.content, \
        f"L1 content missing title; got: {l1.content[:200]!r}"


def test_l1_content_capped_around_500_bge_m3_tokens():
    """L1 body is truncated to ~500 bge-m3 tokens. Using SP tokenizer the
    exact char count varies; assert ratio not exact equal."""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("doc1", SAMPLE_MD)
    l1 = next(c for c in chunks if c.level == 1)
    # 500 bge-m3 tokens ≈ 1800-2500 chars for English academic text
    # SAMPLE_MD is ~1450 chars total → whole body fits in 500 tokens
    # so L1 should contain the whole sample (no truncation triggered)
    assert len(l1.content) <= 3000, f"L1 unexpectedly long: {len(l1.content)}"
    # title + at least a meaningful portion of intro must be present
    assert "Transformers have quadratic complexity" in l1.content


def test_l1_truncation_triggers_on_long_doc():
    """When doc body exceeds 500 bge-m3 tokens, L1 must be truncated."""
    chunker = SemanticSectionChunker()
    # Build a doc with >> 500 tokens of body
    long_body = "This is a sentence that adds tokens to the body. " * 200  # ~1000 sentences
    long_md = f"# Long Paper Title\n\n# 1 INTRODUCTION\n\n{long_body}"
    chunks = chunker.chunk_document("docL", long_md)
    l1 = next(c for c in chunks if c.level == 1)
    # truncated L1 must be much shorter than full body
    assert len(l1.content) < len(long_md) * 0.5, \
        f"L1 not truncated: l1_len={len(l1.content)} body_len={len(long_md)}"
    # but still contains title
    assert "Long Paper Title" in l1.content


def test_no_l3_chunks_emitted():
    """New schema drops level=3 entirely."""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("doc1", SAMPLE_MD)
    l3 = [c for c in chunks if c.level == 3]
    assert len(l3) == 0, f"unexpected L3 chunks: {len(l3)}"
    # only levels 1 and 2 should appear
    levels = {c.level for c in chunks}
    assert levels <= {1, 2}, f"unexpected levels: {levels}"


def test_l2_chunks_still_present_with_section_names():
    """L2 chunks unchanged: one per section with section_name metadata."""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("doc1", SAMPLE_MD)
    l2_chunks = [c for c in chunks if c.level == 2]
    assert len(l2_chunks) >= 4, f"too few L2 chunks (expected ≥4 sections), got {len(l2_chunks)}"
    section_names = {c.metadata.get("section_name", "") for c in l2_chunks}
    assert len(section_names) >= 2, f"L2 chunks lost section_name diversity: {section_names}"


def test_l1_skipped_on_empty_doc():
    """Empty/whitespace doc returns empty list (no spurious L1)."""
    chunker = SemanticSectionChunker()
    assert chunker.chunk_document("d", "") == []
    assert chunker.chunk_document("d", "   \n   ") == []


def test_l1_metadata_section_name_marker():
    """L1 chunk metadata must mark it as title_summary for downstream filter."""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("doc1", SAMPLE_MD)
    l1 = next(c for c in chunks if c.level == 1)
    assert l1.metadata.get("section_name") == "title_summary", \
        f"L1 metadata.section_name != 'title_summary': {l1.metadata}"


SAMPLE_MD_WITH_ABSTRACT = """# A Real Paper Title

#### ABSTRACT

This paper introduces FooBar, a novel method for solving Baz. We
demonstrate state-of-the-art results on Qux benchmark with 10x speedup.
Our approach combines technique A with technique B in a principled way.

#### 1 INTRODUCTION

Long body text that should NOT appear in L1 when an abstract is present.
This intro continues for several paragraphs about background and motivation.

#### 2 METHOD

More body text only L2 should see.
"""


def test_l1_prefers_abstract_when_present():
    """When document has an `## Abstract` (any depth, case-insensitive) section,
    L1 = title + abstract body, NOT title + first N tokens of intro."""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("doc_abs", SAMPLE_MD_WITH_ABSTRACT)
    l1 = next(c for c in chunks if c.level == 1)
    # Title present
    assert "A Real Paper Title" in l1.content
    # Abstract body present
    assert "FooBar, a novel method for solving Baz" in l1.content
    # Intro body absent (would only appear in N-token fallback)
    assert "Long body text that should NOT appear in L1" not in l1.content


def test_l1_falls_back_to_n_tokens_when_no_abstract():
    """When no abstract heading exists, fall back to title + first N bge-m3 tokens."""
    no_abs = """# Title Only

# 1 INTRODUCTION

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. This intro must appear
in L1 as the fallback summary source.
"""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("doc_noabs", no_abs)
    l1 = next(c for c in chunks if c.level == 1)
    assert "Title Only" in l1.content
    # Fallback path took the body (intro)
    assert "Lorem ipsum" in l1.content or "intro must appear in L1" in l1.content


def test_l1_uses_title_hint_when_markdown_has_no_h1():
    no_h1 = """## Abstract

This abstract is present, but marker failed to preserve the paper H1.

## Introduction

Body text.
"""
    chunker = SemanticSectionChunker(semantic_inner_split=False)

    chunks = chunker.chunk_document(
        "doc_title_hint",
        no_h1,
        title_hint="Canonical SQL Paper Title",
    )

    l1 = next(c for c in chunks if c.level == 1)
    assert l1.content.startswith("Canonical SQL Paper Title\n\n")
    assert l1.metadata["title"] == "Canonical SQL Paper Title"


def test_l1_prefers_document_h1_over_title_hint():
    chunker = SemanticSectionChunker(semantic_inner_split=False)

    chunks = chunker.chunk_document(
        "doc_h1",
        SAMPLE_MD_WITH_ABSTRACT,
        title_hint="Stale External Title",
    )

    l1 = next(c for c in chunks if c.level == 1)
    assert l1.metadata["title"] == "A Real Paper Title"
    assert "Stale External Title" not in l1.content


def test_l1_abstract_case_insensitive_and_any_depth():
    """`# abstract`, `## Abstract`, `### ABSTRACT` all trigger the abstract path."""
    for heading in ("# abstract", "## Abstract", "### ABSTRACT", "#### Abstract"):
        md = f"# Paper Title\n\n{heading}\n\nKey abstract content here describing the contribution.\n\n## Body\n\nIrrelevant body.\n"
        chunker = SemanticSectionChunker()
        chunks = chunker.chunk_document("d", md)
        l1 = next(c for c in chunks if c.level == 1)
        assert "Key abstract content here" in l1.content, \
            f"abstract path failed for heading={heading!r}: L1={l1.content[:200]!r}"
        assert "Irrelevant body" not in l1.content


def test_factory_section_l1_returns_fast_semantic_chunker():
    """create_chunker('section_l1' / 'fast') returns SemanticSectionChunker
    with semantic_inner_split=False — keeps L1+L2 schema, skips bge-m3 inside."""
    from core.indexing.chunker import create_chunker
    for key in ("section_l1", "fast", "semantic_fast"):
        ck = create_chunker(key)
        assert isinstance(ck, SemanticSectionChunker), f"key {key!r}: got {type(ck).__name__}"
        assert ck.semantic_inner_split is False, f"key {key!r}: semantic_inner_split must be False"


def test_fast_mode_skips_bge_m3_but_keeps_l1():
    """semantic_inner_split=False: L1 still emitted, L2 uses size-splitter
    (no semantic_chunker.init.ok log, no bge-m3 model load for boundaries).
    Tokenizer for L1 truncation still loads."""
    chunker = SemanticSectionChunker(semantic_inner_split=False)
    chunks = chunker.chunk_document("doc_fast", SAMPLE_MD)
    assert chunks[0].level == 1, "L1 must still be emitted in fast mode"
    l2_chunks = [c for c in chunks if c.level == 2]
    assert len(l2_chunks) >= 4, f"L2 chunks missing in fast mode: {len(l2_chunks)}"
    # No L3 (schema invariant)
    assert not any(c.level == 3 for c in chunks)


def test_l2_section_name_strips_html_span_anchors():
    """Marker emits `<span id="page-X-Y"></span>` inline with headings; the
    section_name must NOT include those anchors — they leak into Qdrant
    payload + frontend UI tags otherwise."""
    md_with_anchors = """# <span id="page-0-2"></span><span id="page-0-0"></span>A Foundation Model

## <span id="page-2-0"></span>3.1 Datasets

Body content for section 3.1 about datasets used in the experiments.
Each dataset has several thousand samples. Authors curated splits carefully.

## 3.2 Architecture

Body content for section 3.2 about model architecture, training procedure, and
hyperparameters used throughout the experiments described in this paper.
"""
    chunker = SemanticSectionChunker()
    chunks = chunker.chunk_document("d_anchors", md_with_anchors)
    l2_names = {c.metadata.get("section_name", "") for c in chunks if c.level == 2}
    for name in l2_names:
        assert "<span" not in name, f"section_name leaked HTML: {name!r}"
        assert "page-" not in name, f"section_name has page anchor: {name!r}"
    # Verify the actual section titles survive
    assert any("3.1" in n and "Datasets" in n for n in l2_names), f"3.1 Datasets section_name lost: {l2_names}"
    assert any("3.2" in n and "Architecture" in n for n in l2_names), f"3.2 Architecture section_name lost: {l2_names}"
