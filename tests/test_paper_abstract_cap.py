"""The L1 abstract cap of the paper chunker, with the real BGE-M3 tokenizer.

``## Abstract`` with no heading after it used to make the whole body the "abstract": one giant L1
chunk (or, once ``enforce_budget`` cut it, several L1 pieces). An abstract longer than
``L1_ABSTRACT_MAX_TOKENS`` is not an abstract: the summary is its first ``L1_SUMMARY_TOKENS``
tokens and ``l1_source`` says ``abstract_capped``. A real abstract is untouched.
"""
from __future__ import annotations

import pytest

from core.indexing.chunker import SemanticSectionChunker, create_chunker
from core.indexing.embedding_profile import count_tokens, get_profile, load_tokenizer
from tests.embedding_model_support import bge_m3_snapshot, text_of_tokens

TITLE = "Capped Abstracts In Paper Chunking"


@pytest.fixture(autouse=True)
def tokenizer_env(monkeypatch):
    monkeypatch.setenv("BGE_M3_PATH", bge_m3_snapshot())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture()
def tokenizer():
    return load_tokenizer(get_profile())


def paper(abstract: str, *, later_heading: bool) -> str:
    text = f"# {TITLE}\n\n## Abstract\n\n{abstract}\n"
    if later_heading:
        text += f"\n## 1 Introduction\n\n{text_of_tokens(load_tokenizer(get_profile()), 300)}\n"
    return text


def chunk(markdown: str):
    return create_chunker("section_l1").chunk_document("7", markdown, title_hint=TITLE)


def l1_of(chunks):
    found = [c for c in chunks if c.level == 1]
    assert len(found) == 1, [(c.level, c.metadata.get("l1_source"), len(c.content)) for c in found]
    return found[0]


def summary_tokens(tokenizer, l1) -> int:
    summary = l1.content.split("\n\n", 1)[1]
    return len(tokenizer(summary, add_special_tokens=False)["input_ids"])


def test_the_limits_are_the_planned_ones():
    assert SemanticSectionChunker.L1_ABSTRACT_MAX_TOKENS == 1024
    assert SemanticSectionChunker.L1_SUMMARY_TOKENS == 500


def test_an_abstract_with_no_heading_after_it_gives_one_capped_l1_and_every_chunk_fits(tokenizer):
    body = text_of_tokens(tokenizer, 5000)
    chunks = chunk(paper(body, later_heading=False))

    l1 = l1_of(chunks)

    assert l1.metadata["l1_source"] == "abstract_capped"
    assert l1.content.startswith(f"{TITLE}\n\n")
    assert body.startswith(" ".join(l1.content.split("\n\n", 1)[1].split()[:40]))     # the head of the body, not all of it
    assert len(l1.content) < len(body) // 5
    assert 400 <= summary_tokens(tokenizer, l1) <= 505
    assert max(count_tokens([c.content for c in chunks], get_profile())) <= get_profile().chunk_max_tokens
    assert [c.level for c in chunks].count(2) >= 3                                       # the body is still searchable as L2


def test_an_abstract_followed_by_headings_is_capped_when_it_is_far_longer_than_an_abstract(tokenizer):
    chunks = chunk(paper(text_of_tokens(tokenizer, 5000), later_heading=True))

    l1 = l1_of(chunks)

    assert l1.metadata["l1_source"] == "abstract_capped" and summary_tokens(tokenizer, l1) <= 505


def test_the_cap_starts_just_above_1024_tokens(tokenizer):
    at_limit = text_of_tokens(tokenizer, 1024)
    above = text_of_tokens(tokenizer, 1025)

    kept = l1_of(chunk(paper(at_limit, later_heading=True)))
    capped = l1_of(chunk(paper(above, later_heading=True)))

    assert kept.metadata["l1_source"] == "abstract" and kept.content == f"{TITLE}\n\n{at_limit}"
    assert capped.metadata["l1_source"] == "abstract_capped" and summary_tokens(tokenizer, capped) <= 505


def test_a_normal_abstract_is_exactly_what_it_was_before(tokenizer):
    abstract = ("We study how long abstracts leak into the summary chunk of a paper. "
                "A capped summary keeps the first tokens only, and a real abstract is kept whole.")

    l1 = l1_of(chunk(paper(abstract, later_heading=True)))

    assert l1.level == 1 and l1.start_idx == 0 and l1.end_idx == len(abstract)
    assert l1.content == f"{TITLE}\n\n{abstract}"
    assert l1.metadata == {"section_name": "title_summary", "title": TITLE, "l1_source": "abstract"}


def test_a_paper_without_an_abstract_heading_keeps_the_token_prefix_summary(tokenizer):
    body = text_of_tokens(tokenizer, 3000)
    markdown = f"# {TITLE}\n\n## 1 Introduction\n\n{body}\n"

    l1 = l1_of(chunk(markdown))

    assert l1.metadata["l1_source"] == "n_tokens_fallback" and 400 <= summary_tokens(tokenizer, l1) <= 505


def test_an_abstract_the_size_of_the_whole_document_does_not_hand_the_tokenizer_an_unbounded_text(tokenizer):
    from core.indexing.token_budget import SEGMENT_CHARS

    body = " ".join([text_of_tokens(tokenizer, 4000)] * 18)                # over SEGMENT_CHARS characters
    assert len(body) > SEGMENT_CHARS

    l1 = l1_of(chunk(paper(body, later_heading=False)))

    assert l1.metadata["l1_source"] == "abstract_capped" and summary_tokens(tokenizer, l1) <= 505


def appendix_contents_line(entries: int) -> str:
    """The shape measured in ICML 2026 paper 105830: a table of contents with dot leaders, one H1 line."""
    leaders = " ".join(["."] * 90)
    return "# Appendix " + " ".join(f"• [{i}.](#page-{i}) Section {i} {leaders} [{i}](#page-{i})"
                                    for i in range(entries))


def test_a_later_h1_line_that_is_no_title_does_not_become_the_title(tokenizer):
    # No H1 title at the top (the title is an H2), and a whole-line H1 table of contents further down.
    abstract = "We search tabular architectures progressively with a zero-cost proxy."
    contents = appendix_contents_line(24)
    assert count_tokens([contents], get_profile())[0] > get_profile().chunk_max_tokens
    markdown = (f"## {TITLE}\n\n## Abstract\n\n{abstract}\n\n## 1 Introduction\n\n"
                f"{text_of_tokens(tokenizer, 300)}\n\n{contents}\n\n## A Notation\n\n{text_of_tokens(tokenizer, 300)}\n")

    chunks = chunk(markdown)

    l1 = l1_of(chunks)
    assert l1.content == f"{TITLE}\n\n{abstract}" and l1.metadata["title"] == TITLE
    assert max(count_tokens([c.content for c in chunks], get_profile())) <= get_profile().chunk_max_tokens


def test_a_title_over_the_title_limit_is_cut_to_it_when_no_short_title_exists(tokenizer):
    abstract = "A short abstract."
    long_title = text_of_tokens(tokenizer, 3000)
    markdown = f"## Abstract\n\n{abstract}\n\n## 1 Introduction\n\n{text_of_tokens(tokenizer, 300)}\n"

    l1 = l1_of(create_chunker("section_l1").chunk_document("7", markdown, title_hint=long_title))

    title = l1.content.split("\n\n", 1)[0]
    assert long_title.startswith(" ".join(title.split()[:20]))
    assert len(tokenizer(title, add_special_tokens=False)["input_ids"]) <= SemanticSectionChunker.L1_TITLE_MAX_TOKENS
    assert l1.content.endswith(f"\n\n{abstract}") and l1.metadata["l1_source"] == "abstract"
