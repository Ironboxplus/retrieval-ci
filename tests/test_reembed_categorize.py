"""categorize_texts with texts longer than the segment bound, real BGE-M3 tokenizer.

``tokens_over_limit`` counts a text over ``SEGMENT_CHARS`` only until it passes the limit it is
given, so for such a text it returns a lower bound just above that limit. The scope categories
compare the count with several thresholds (512, 2,048, 8,192), so a 30,000-token text must not
be filed under the smallest one: it is OVERSIZED, not a chunk to embed (the embed host refuses
it with InputTooLong).
"""
from __future__ import annotations

import random

import pytest

from tests.embedding_model_support import bge_m3_snapshot, text_of_tokens


@pytest.fixture(scope="module", autouse=True)
def _snapshot():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    yield
    mp.undo()


def _real_code_lines(chars: int) -> str:
    """Repeated lines of this repository's own code until ``chars`` characters."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app" / "services" / "code_kb" / "service.py").read_text(
        encoding="utf-8")
    lines = [line for line in source.splitlines() if line.strip()]
    out, total, rng = [], 0, random.Random(5)
    while total < chars:
        line = rng.choice(lines)
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


def _tokens(text: str) -> int:
    from core.indexing.embedding_profile import count_tokens, get_profile
    return count_tokens([text], get_profile())[0]


SCOPES = ["code_chunks:truncated", "public_code_chunks:truncated", "papers:pipeline", "arxiv:all", "papers:uploads"]


def _sparse_prefix_then_dense_code() -> str:
    """Over the bound in characters, with almost no tokens in the first segment: ~3,000 tokens of
    text and 200,000 characters of blank lines, then ~30,000 tokens of real code. The first
    segment of this text is what a count that stops at the smallest threshold sees."""
    from core.indexing.embedding_profile import get_profile, load_tokenizer

    return text_of_tokens(load_tokenizer(get_profile()), 3000) + "\n\n" * 100_000 + "\n" + _real_code_lines(120_000)


@pytest.mark.parametrize("scope_name", SCOPES)
def test_a_text_over_the_segment_bound_and_over_the_model_limit_is_oversized(scope_name):
    from app.services.pipeline.reembed_scopes import OVERSIZED, categorize_texts, get_scope
    from core.indexing.embedding_profile import get_profile
    from core.indexing.token_budget import SEGMENT_CHARS

    huge = _sparse_prefix_then_dense_code()
    assert len(huge) > SEGMENT_CHARS
    real = _tokens(huge)
    assert real > 3 * get_profile().max_seq_len, "the fixture must be far over the model limit"

    (category, tokens), = categorize_texts(get_scope(scope_name), [huge])

    assert category == OVERSIZED, f"{scope_name}: {real} real tokens were filed as {category} ({tokens})"
    assert tokens > get_profile().max_seq_len


@pytest.mark.parametrize("scope_name, expected", [
    ("code_chunks:truncated", "chunk"),
    ("public_code_chunks:truncated", "chunk"),
    ("papers:pipeline", "chunk"),
    ("arxiv:all", "chunk"),
    ("papers:uploads", "chunk"),
])
def test_a_text_over_the_segment_bound_that_fits_the_model_limit_is_still_a_chunk(scope_name, expected):
    from app.services.pipeline.reembed_scopes import categorize_texts, get_scope
    from core.indexing.embedding_profile import get_profile, load_tokenizer
    from core.indexing.token_budget import SEGMENT_CHARS

    # ~3,000 tokens of text, then 250,000 characters of blank lines: past the bound in characters,
    # not in tokens (whitespace runs collapse to almost nothing).
    text = text_of_tokens(load_tokenizer(get_profile()), 3000) + "\n\n" * 125_000 + "tail"
    assert len(text) > SEGMENT_CHARS
    real = _tokens(text)
    assert 2048 < real < get_profile().max_seq_len, real

    (category, tokens), = categorize_texts(get_scope(scope_name), [text])

    assert category == expected, f"{scope_name}: {real} real tokens were filed as {category} ({tokens})"
    scope = get_scope(scope_name)
    floor = min(t for t in (scope.skip_up_to, scope.stamp_up_to, scope.skip_above, get_profile().max_seq_len) if t)
    assert abs(tokens - real) <= 8 if real > floor else tokens == floor     # counted, not just bounded


def test_short_texts_are_categorized_as_before_next_to_a_huge_one():
    from app.services.pipeline.reembed_scopes import CHUNK, OVERSIZED, SKIP, categorize_texts, get_scope
    from core.indexing.embedding_profile import get_profile, load_tokenizer
    from core.indexing.token_budget import SEGMENT_CHARS

    tokenizer = load_tokenizer(get_profile())
    texts = ["short ascii", text_of_tokens(tokenizer, 600), _real_code_lines(SEGMENT_CHARS + 1_000),
             text_of_tokens(tokenizer, 9000)]

    out = categorize_texts(get_scope("code_chunks:truncated"), texts)

    assert [category for category, _ in out] == [SKIP, CHUNK, OVERSIZED, OVERSIZED]
    assert out[1][1] == _tokens(texts[1]) and out[3][1] == _tokens(texts[3])
