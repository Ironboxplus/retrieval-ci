"""Every chunker's output is at most ``chunk_max_tokens`` real tokens per chunk.

Real BGE-M3 tokenizer, no model: the chunkers only need the tokenizer to count and cut.
"""
from __future__ import annotations

import re

import pytest

from tests.embedding_model_support import bge_m3_snapshot, real_text, text_of_tokens


@pytest.fixture(scope="module", autouse=True)
def _snapshot():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    yield
    mp.undo()


def _profile():
    from core.indexing.embedding_profile import get_profile
    return get_profile()


def _count(text: str) -> int:
    from core.indexing.embedding_profile import count_tokens
    return count_tokens([text], _profile())[0]


def _chunk(content: str, start_idx: int, *, level: int = 2, metadata: dict | None = None):
    from app.models.schemas import Chunk
    return Chunk(id="original", doc_id="d1", content=content, level=level, start_idx=start_idx,
                 end_idx=start_idx + len(content), metadata=metadata or {"section_name": "s"})


def test_enforce_budget_leaves_chunks_within_budget_untouched():
    from core.indexing.chunker import enforce_budget

    chunks = [_chunk("short ascii text", 0), _chunk("中文也很短", 20)]
    assert enforce_budget(chunks) == chunks
    assert all(a is b for a, b in zip(enforce_budget(chunks), chunks))


def test_enforce_budget_splits_an_over_budget_chunk_and_keeps_offsets_consistent():
    from core.indexing.chunker import enforce_budget
    from core.indexing.embedding_profile import get_profile, load_tokenizer

    limit = _profile().chunk_max_tokens
    document = "x" * 100 + real_text(load_tokenizer(get_profile()), 6000) + " tail"
    big = document[100:]
    chunks = [_chunk("before chunk", 0), _chunk(big, 100, level=3, metadata={"chunk_size": 2048}),
              _chunk("after chunk", len(document))]

    out = enforce_budget(chunks)

    assert out[0] is chunks[0] and out[-1] is chunks[-1]
    pieces = out[1:-1]
    assert len(pieces) >= 3
    assert all(_count(p.content) <= limit for p in pieces)
    assert all(p.level == 3 and p.metadata == {"chunk_size": 2048} and p.doc_id == "d1" for p in pieces)
    assert len({p.id for p in pieces}) == len(pieces) and all(p.id != "original" for p in pieces)
    # the chunk guaranteed document[start:end] == content, so each piece still does
    assert all(document[p.start_idx:p.end_idx] == p.content for p in pieces)
    assert re.sub(r"\s+", "", "".join(p.content for p in pieces)) == re.sub(r"\s+", "", big)
    assert pieces[0].start_idx == 100 and pieces[-1].end_idx == len(document)


def test_enforce_budget_splits_cjk_without_spaces():
    from core.indexing.chunker import enforce_budget

    cjk = "检索增强生成的向量模型需要处理很长的中文段落。" * 400        # ~9,600 chars, no whitespace
    out = enforce_budget([_chunk(cjk, 0)])
    assert len(out) > 1 and all(_count(p.content) <= _profile().chunk_max_tokens for p in out)
    assert "".join(p.content for p in out) == cjk


def test_semantic_section_chunker_caps_an_oversized_abstract_to_one_summary():
    """The L1 chunk is title + abstract; an abstract over L1_ABSTRACT_MAX_TOKENS is capped to one
    L1 of L1_SUMMARY_TOKENS tokens (it used to become several L1 pieces; test_paper_abstract_cap.py)."""
    from core.indexing.chunker import SemanticSectionChunker
    from core.indexing.embedding_profile import get_profile, load_tokenizer

    abstract = text_of_tokens(load_tokenizer(get_profile()), 5000)
    text = (f"# A Very Long Paper\n\n## Abstract\n\n{abstract}\n\n## Introduction\n\n"
            + "Body text of the introduction. " * 50)
    chunks = SemanticSectionChunker(semantic_inner_split=False).chunk_document("doc", text)
    assert chunks
    assert max(_count(c.content) for c in chunks) <= _profile().chunk_max_tokens
    l1 = [c for c in chunks if c.level == 1]
    assert len(l1) == 1 and l1[0].metadata["l1_source"] == "abstract_capped"


def test_section_aware_and_text_chunkers_stay_within_budget_on_hostile_text():
    from core.indexing.chunker import SectionAwareChunker, TextChunker

    hostile = "漢字" * 3000 + "\n\n" + "aGVsbG8gd29ybGQ=" * 600           # unbroken CJK, then a base64 run
    limit = _profile().chunk_max_tokens
    for chunker in (SectionAwareChunker(), TextChunker()):
        chunks = chunker.chunk_document("doc", hostile)
        assert chunks and max(_count(c.content) for c in chunks) <= limit
