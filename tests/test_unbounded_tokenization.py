"""No tokenizer call sees an unbounded text.

A 91 MB journal entry with no Markdown headings is one section; tokenizing it whole built a
~25 million element offset table (about 2 GB) and swapped a production host to a halt. Every path
that cuts text into token-sized pieces now works on segments of at most ``SEGMENT_CHARS``
characters. Real BGE-M3 tokenizer behind a thin shell that records the length of every text it is
handed; no model.
"""
from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path

import pytest

from tests.embedding_model_support import bge_m3_snapshot

REPO = Path(__file__).resolve().parents[1]
LIMIT = 200_000                    # SEGMENT_CHARS, written out so the old code can run these tests too
TWO_MB = 2_000_000


@pytest.fixture(scope="module", autouse=True)
def _snapshot():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    yield
    mp.undo()


class RecordingTokenizer:
    """The real tokenizer; remembers the length of every text it is asked to tokenize."""

    def __init__(self, real, lengths: list[int]):
        self._real, self._lengths = real, lengths

    def _note(self, text) -> None:
        self._lengths.extend(len(t) for t in ([text] if isinstance(text, str) else text))

    def __call__(self, text, *args, **kwargs):
        self._note(text)
        return self._real(text, *args, **kwargs)

    def encode(self, text, *args, **kwargs):
        self._note(text)
        return self._real.encode(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture()
def tokenized(monkeypatch) -> list[int]:
    """Lengths of every text the tokenizer is handed while the test runs."""
    from core.indexing import embedding_profile

    lengths: list[int] = []
    real_loader = embedding_profile._tokenizer
    monkeypatch.setattr(embedding_profile, "_tokenizer",
                        lambda model_id, path: RecordingTokenizer(real_loader(model_id, path), lengths))
    return lengths


def jsonl_log(size: int, seed: int = 3) -> str:
    """A conversation log: one JSON object per line, no Markdown headings, ~``size`` characters."""
    rng = random.Random(seed)
    words = ("retrieval index chunk vector embedding model sparse dense token budget journal scope "
             "qdrant payload fingerprint reuse tool call result error stack trace").split()
    lines, total = [], 0
    while total < size:
        line = json.dumps({"role": rng.choice(["user", "assistant", "tool"]), "n": len(lines),
                           "text": " ".join(rng.choice(words) + str(rng.randrange(1000)) for _ in range(28))})
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def chunk(body: str, chunker=None):
    from app.services.journal.chunker import JournalChunker
    return (chunker or JournalChunker()).chunk_entry(
        entry_id="entry-1", title="t", summary=None, body=body, tags=[], source_kind="doc", client="mcp")


def counts(texts: list[str]) -> list[int]:
    from core.indexing.embedding_profile import count_tokens, get_profile
    return count_tokens(texts, get_profile())


# ----------------------------------------------------------------------------- the defect
def test_a_2mb_headerless_journal_body_is_never_tokenized_in_one_piece(tokenized):
    from core.indexing.embedding_profile import get_profile

    body = jsonl_log(TWO_MB)
    assert body.count("\n") > 5000 and not re.search(r"^#", body, re.MULTILINE)

    chunks = chunk(body)

    assert max(tokenized) <= LIMIT, f"largest tokenizer input: {max(tokenized):,} characters"
    limit = get_profile().chunk_max_tokens
    assert len(chunks) > 1000
    assert max(counts([c.content for c in chunks])) <= limit
    assert all(c.content in body for c in chunks[1:])                    # still slices of the entry


def test_a_500kb_text_without_any_newline_or_space_completes_in_bounded_pieces(tokenized):
    from core.indexing.embedding_profile import get_profile

    rng = random.Random(9)
    alphabet = string.ascii_letters + string.digits + "+/"
    body = "".join(rng.choice(alphabet) for _ in range(500_000))          # base64
    assert not any(ch.isspace() for ch in body)

    chunks = chunk(body)

    assert max(tokenized) <= LIMIT
    assert max(counts([c.content for c in chunks])) <= get_profile().chunk_max_tokens
    assert len(chunks) > 100


def test_split_to_budget_segments_a_huge_text_and_still_partitions_it(tokenized):
    from core.indexing.token_budget import split_to_budget

    text = jsonl_log(TWO_MB)
    spans = split_to_budget(text, max_tokens=512, prefer="line")

    assert max(tokenized) <= LIMIT
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))            # an exact partition
    assert max(counts([text[a:b] for a, b in spans])) <= 512


def test_tokens_over_limit_and_enforce_budget_do_not_tokenize_a_huge_chunk_whole(tokenized):
    from app.models.schemas import Chunk
    from core.indexing.chunker import enforce_budget
    from core.indexing.embedding_profile import get_profile, tokens_over_limit

    profile = get_profile()
    text = jsonl_log(TWO_MB)
    over = tokens_over_limit([text], profile.chunk_max_tokens, profile)
    assert [i for i, _ in over] == [0] and over[0][1] > profile.chunk_max_tokens
    assert max(tokenized) <= LIMIT

    tokenized.clear()
    big = Chunk(id="c", doc_id="d", content=text, level=3, start_idx=0, end_idx=len(text), metadata={})
    pieces = enforce_budget([big])
    assert max(tokenized) <= LIMIT
    assert len(pieces) > 300 and max(counts([p.content for p in pieces])) <= profile.chunk_max_tokens


def test_fit_records_to_budget_does_not_tokenize_a_huge_code_record_whole(tokenized):
    from core.code_kb.vector_records import CodeChunkVectorRecord, fit_records_to_budget
    from core.indexing.embedding_profile import get_profile

    code = jsonl_log(TWO_MB)
    record = CodeChunkVectorRecord(
        rel_path="data/log.jsonl", language="json", start_line=1, end_line=code.count("\n") + 1, label="log",
        content="# path: data/log.jsonl\n" + code, code=code, chunk_ordinal=0)

    pieces = fit_records_to_budget([record])

    assert max(tokenized) <= LIMIT
    assert len(pieces) > 300
    assert max(counts([p.content for p in pieces])) <= get_profile().chunk_max_tokens


def test_the_paper_summary_fallback_does_not_tokenize_a_whole_document(tokenized):
    from core.indexing.chunker import SemanticSectionChunker

    body = jsonl_log(TWO_MB)                               # no heading, no abstract: the token-prefix path
    chunk_ = SemanticSectionChunker()._build_l1_chunk("d1", body)

    assert chunk_ is not None and chunk_.metadata["l1_source"] == "n_tokens_fallback"
    assert max(tokenized) <= LIMIT


# ------------------------------------------------------------------------- segment_text itself
def test_the_constant_is_the_agreed_bound():
    from core.indexing.token_budget import SEGMENT_CHARS
    assert SEGMENT_CHARS == LIMIT


@pytest.mark.parametrize("text", ["", "x", "a b\nc" * 10], ids=["empty", "one-char", "short"])
def test_a_text_within_the_bound_is_one_segment(text):
    from core.indexing.token_budget import segment_text
    assert segment_text(text, 1000) == [(0, len(text))]


def test_segments_cut_at_the_last_newline_then_the_last_space_then_hard():
    from core.indexing.token_budget import segment_text

    lines = segment_text("aaaa\nbbbb\ncccc\ndddd", 12)              # window "aaaa\nbbbb\ncc": last newline ends bbbb
    assert lines == [(0, 10), (10, 19)]

    spaces = segment_text("aaaa bbbb cccc dddd", 12)               # no newline: the last space ends "bbbb "
    assert spaces == [(0, 10), (10, 19)]

    hard = segment_text("x" * 25, 10)                              # neither: cut at the bound
    assert hard == [(0, 10), (10, 20), (20, 25)]


@pytest.mark.parametrize("text, bound", [
    pytest.param(jsonl_log(60_000), 7_000, id="jsonl"),
    pytest.param("word " * 5_000, 999, id="words"),
    pytest.param("\n" + "y" * 5_000, 777, id="lone-newline-at-window-start"),
    pytest.param("字" * 4_321, 500, id="cjk"),
    pytest.param("a\n" * 3_000, 1, id="one-char-bound"),
])
def test_segments_partition_the_text_and_respect_the_bound(text, bound):
    from core.indexing.token_budget import segment_text

    spans = segment_text(text, bound)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(0 < b - a <= bound for a, b in spans)
    assert "".join(text[a:b] for a, b in spans) == text


# ------------------------------------------------------- small documents come out exactly as before
class ReferenceJournalChunker:
    """The journal chunker as it was before segmenting: ``_window_text`` over the whole section."""

    @staticmethod
    def build():
        from app.services.journal.chunker import _SNAP_TOKENS, JournalChunker
        from core.indexing.embedding_profile import ascii_within_limit, get_profile
        from core.indexing.token_budget import SPECIAL_TOKENS, token_offsets

        class Reference(JournalChunker):
            def _window_text(self, text: str) -> list[str]:
                text = text.strip()
                if not text:
                    return []
                if ascii_within_limit(text, self.target_tokens + SPECIAL_TOKENS):
                    return [text]
                tokens = token_offsets(text, get_profile())
                if len(tokens) <= self.target_tokens:
                    return [text]

                def starts_word(index: int) -> bool:
                    return index == 0 or text[tokens[index][0] - 1].isspace()

                windows: list[str] = []
                start = 0
                while True:
                    end = min(start + self.target_tokens, len(tokens))
                    if end < len(tokens):
                        end = next((i for i in range(end, max(start, end - _SNAP_TOKENS), -1) if starts_word(i)), end)
                    windows.append(text[tokens[start][0]:tokens[end - 1][1]])
                    if end >= len(tokens):
                        return windows
                    next_start = max(start + 1, end - self.overlap_tokens)
                    start = next((i for i in range(next_start, max(start, next_start - _SNAP_TOKENS), -1)
                                  if starts_word(i)), next_start)

        return Reference()


def _documents() -> dict[str, str]:
    from app.services.journal.chunker import JournalChunker
    from core.indexing.token_budget import SEGMENT_CHARS

    rng = random.Random(21)
    docs = {
        "cjk_190k": "\n".join("".join(chr(rng.randrange(0x4E00, 0x9FA5)) for _ in range(95)) for _ in range(2000)),
        "base64_150k": "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(150_000)),
        "jsonl_190k": jsonl_log(190_000),
        "short": "A short journal note about the retrieval index.",
        "fenced": "intro\n\n```text\n" + "\n".join("".join(rng.choice(string.printable[:94]) for _ in range(960))
                                                   for _ in range(3)) + "\n```\n\noutro",
    }
    for path in sorted(REPO.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if path.name == "journal.md":
            text = text[:400_000]            # real journal entries; the rest only repeats the shape
        sections = JournalChunker._split_markdown_sections(text)
        if all(len(section) <= SEGMENT_CHARS for _, section in sections):
            docs[path.name] = text
    return docs


def test_documents_within_the_bound_are_chunked_exactly_as_before():
    docs = _documents()
    private = (REPO / "journal.md").exists()  # the public CI snapshot leaves the notes out
    assert len(docs) >= (12 if private else 5) and "jsonl_190k" in docs
    assert "journal.md" in docs or not private
    reference = ReferenceJournalChunker.build()
    for name, body in docs.items():
        old, new = chunk(body, reference), chunk(body)
        assert [(c.chunk_id, c.chunk_index, c.level, c.heading, c.content, c.token_count, c.metadata) for c in new] == \
               [(c.chunk_id, c.chunk_index, c.level, c.heading, c.content, c.token_count, c.metadata) for c in old], name


def test_split_to_budget_of_a_text_within_the_bound_is_unchanged_by_segmenting():
    from core.indexing.token_budget import split_to_budget

    text = jsonl_log(150_000)
    spans = split_to_budget(text, max_tokens=2048, prefer="line")
    assert spans[0][0] == 0 and spans[-1][1] == len(text) and len(spans) > 10
    # packed in one pass: every span but the last is as full as the greedy packing allows
    full = counts([text[a:b] for a, b in spans[:-1]])
    assert min(full) > 2048 - 120
