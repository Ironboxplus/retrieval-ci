"""JournalChunker sizes chunks in model tokens, not whitespace words.

A CJK body has almost no whitespace, so 520 "words" was the whole 66,039-token document.
Real BGE-M3 tokenizer, no model.
"""
from __future__ import annotations

import random
import string

import pytest

from tests.embedding_model_support import bge_m3_snapshot


@pytest.fixture(scope="module", autouse=True)
def _snapshot():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    yield
    mp.undo()


def _counts(texts: list[str]) -> list[int]:
    from core.indexing.embedding_profile import count_tokens, get_profile
    return count_tokens(texts, get_profile())


def _chunk(body: str, **kw):
    from app.services.journal.chunker import JournalChunker
    return JournalChunker(**kw).chunk_entry(
        entry_id="entry-1", title="t", summary=None, body=body, tags=[], source_kind="doc", client="mcp")


def _cjk_lines(count: int, width: int, seed: int) -> str:
    rng = random.Random(seed)
    return "\n".join("".join(chr(rng.randrange(0x4E00, 0x9FA5)) for _ in range(width)) for _ in range(count))


def test_a_66k_token_cjk_body_becomes_chunks_within_the_chunk_budget():
    from core.indexing.embedding_profile import get_profile

    body = _cjk_lines(920, 120, seed=7)                 # ~110,000 characters, no spaces
    assert _counts([body])[0] > 60_000

    chunks = _chunk(body)

    limit = get_profile().chunk_max_tokens
    assert max(_counts([c.content for c in chunks])) <= limit        # red: one chunk of ~66K tokens
    assert len(chunks) > 60_000 // limit
    assert sum(len(c.content) for c in chunks[1:]) >= len(body.replace("\n", ""))   # overlap only adds


def test_target_tokens_now_means_model_tokens():
    rng = random.Random(11)
    words = ["retrieval", "index", "embedding", "chunk", "scope", "vector", "payload", "journal"]
    body = "# Notes\n\n" + " ".join(rng.choice(words) + str(rng.randrange(100)) for _ in range(6000))

    chunks = _chunk(body, target_tokens=300, overlap_tokens=50)
    bodies = [c for c in chunks if c.chunk_index > 0]
    counts = _counts([c.content for c in bodies])

    assert len(bodies) > 6
    assert max(counts) <= 300 + 2 + 4          # specials + at most a word-start marker or two
    assert min(counts[:-1]) >= 250             # windows are full, not word-count short
    assert [c.token_count for c in bodies] == counts
    # consecutive windows overlap
    assert all(bodies[i + 1].content[:20] in bodies[i].content for i in range(len(bodies) - 1))


def test_windows_start_and_end_on_word_boundaries_for_spaced_text():
    words = ["alpha", "beta", "gamma", "delta"]
    valid = {words[i % 4] + str(i) for i in range(4000)}      # "alpha12" is valid, "alpha1" is not
    body = " ".join(words[i % 4] + str(i) for i in range(4000))
    chunks = [c for c in _chunk(body, target_tokens=200, overlap_tokens=30) if c.chunk_index > 0]
    assert len(chunks) > 5
    for chunk in chunks:
        assert set(chunk.content.split()) <= valid                # no word was cut in half


def test_a_short_entry_is_still_one_body_chunk():
    chunks = _chunk("A short journal note about the retrieval index.")
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[1].content == "A short journal note about the retrieval index."


def test_a_fenced_block_over_the_budget_is_split_instead_of_cut():
    from core.indexing.embedding_profile import get_profile

    rng = random.Random(5)
    alphabet = string.ascii_letters + string.digits + string.punctuation
    code = "\n".join("".join(rng.choice(alphabet) for _ in range(960)) for _ in range(3))
    limit = get_profile().chunk_max_tokens
    assert _counts([code])[0] > limit, "the fixture must exceed the budget for this test to mean anything"

    chunks = _chunk(f"intro\n\n```text\n{code}\n```\n\noutro")

    assert max(_counts([c.content for c in chunks])) <= limit
    assert len([c for c in chunks if c.heading == "detail"]) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
