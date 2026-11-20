"""The ASCII shortcut (no tokenizing when a text provably fits) against the real tokenizer.

SentencePiece puts a dummy-prefix "▁" in front of the text, a token that covers no input
character, so N ASCII characters can cost N + 1 content tokens + 2 specials. Every consumer of
the shortcut goes through ``ascii_within_limit``; these tests brute-force the worst case.
"""
from __future__ import annotations

import random
import string

import pytest

from core.indexing.embedding_profile import (
    ASCII_TOKEN_OVERHEAD,
    ascii_within_limit,
    count_tokens,
    get_profile,
    tokens_over_limit,
)
from core.indexing.token_budget import split_to_budget

LIMIT = 64
# Everything printable plus the control characters ASCII allows.
ALPHABET = string.printable + "\x00\x01\x1f\x7f"


def _worst_case_texts(length: int) -> list[str]:
    rng = random.Random(20261114)
    texts = []
    for ch in ALPHABET:
        texts += [ch * length, (ch + " ") * (length // 2) + "a" * (length % 2), " " + ch * (length - 1),
                  ch * (length - 1) + " "]
    for _ in range(6000):
        pool = rng.sample(ALPHABET, rng.choice([2, 3, 5, 10, len(ALPHABET)]))
        texts.append("".join(rng.choice(pool) for _ in range(length)))
    return texts


def test_overhead_is_the_dummy_prefix_plus_both_specials():
    assert ASCII_TOKEN_OVERHEAD == 3


def test_no_ascii_text_the_shortcut_accepts_exceeds_the_limit():
    profile = get_profile()
    length = LIMIT - ASCII_TOKEN_OVERHEAD
    texts = _worst_case_texts(length)
    assert all(ascii_within_limit(text, LIMIT) for text in texts)
    counts = count_tokens(texts, profile)
    assert max(counts) <= LIMIT
    # The bound is tight, so this is a real test of the margin and not a loose one.
    assert max(counts) == LIMIT


def test_one_character_more_can_exceed_the_limit():
    profile = get_profile()
    text = "," * (LIMIT - 2)
    assert not ascii_within_limit(text, LIMIT)
    assert count_tokens([text], profile) == [LIMIT + 1]


@pytest.mark.parametrize("length", [LIMIT - 2, LIMIT - 1, LIMIT, LIMIT + 5])
def test_tokens_over_limit_reports_a_worst_case_the_old_margin_let_through(length):
    profile = get_profile()
    text = "," * length
    real = count_tokens([text], profile)[0]
    reported = tokens_over_limit([text], LIMIT, profile)
    assert reported == ([(0, real)] if real > LIMIT else [])


def test_split_to_budget_does_not_return_an_over_budget_ascii_text_whole():
    profile = get_profile()
    text = "," * (LIMIT - 2)
    spans = split_to_budget(text, max_tokens=LIMIT, profile=profile)
    assert "".join(text[a:b] for a, b in spans) == text
    assert max(count_tokens([text[a:b] for a, b in spans], profile)) <= LIMIT
    assert len(spans) > 1


def test_edge_reserve_is_backed_by_the_final_recount_on_adversarial_ascii():
    """EDGE_TOKENS only plans the packing; every span is recounted with the real tokenizer."""
    profile = get_profile()
    texts = _worst_case_texts(300)[:400]
    for prefer in ("line", "space"):
        for text in texts:
            spans = split_to_budget(text, max_tokens=LIMIT, profile=profile, prefer=prefer)
            assert "".join(text[a:b] for a, b in spans) == text
        flat = [text[a:b] for text in texts
                for a, b in split_to_budget(text, max_tokens=LIMIT, profile=profile, prefer=prefer)]
        assert max(count_tokens(flat, profile)) <= LIMIT
