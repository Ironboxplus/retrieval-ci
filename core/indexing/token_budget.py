"""Split text into spans that each fit a token budget, measured with the model's tokenizer.

This is the final guarantee behind every chunker: whatever size a chunker aims for, no span
that leaves here is longer than ``max_tokens`` real tokens (the model's special tokens
included). Spans are character ranges of the original text that partition it exactly, so a
caller keeps its own offsets and line numbers; text is never re-decoded from token ids
(decode round trips add tokens and rewrite the text).

No tokenizer call here, or in the chunkers built on it, sees more than ``SEGMENT_CHARS``
characters: longer text is cut into segments (``segment_text``) and each one is tokenized on
its own. Tokenizing a whole document builds one offset tuple per token, which for a 91 MB
conversation log was about 2 GB and swapped a production host to a halt.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from typing import Any, Literal

from core.indexing.embedding_profile import (
    EmbeddingProfile,
    ascii_within_limit,
    count_tokens,
    get_profile,
    load_tokenizer,
)

# <s> and </s>; the same count the tokenizer adds in ``count_tokens``.
SPECIAL_TOKENS = 2
# A span's two edges can cost tokens that the same characters inside the whole text do not:
# a span that starts inside a word gets a word-start marker token, and one that ends in
# whitespace gets a token for that space. Packing keeps both free so a full span is not pushed
# over the budget by its own edges (the verification pass below catches anything left).
EDGE_TOKENS = 2

# A line with its newline / a word with the whitespace around it. Matches are contiguous, so
# their ends are exactly the places a span may end without cutting a unit.
_LINES = re.compile(r"[^\n]*\n|[^\n]+")
_WORDS = re.compile(r"\s*\S+\s*")

Prefer = Literal["line", "space"]

# The longest text any single tokenizer call gets. A token covers at least one character, so a
# segment's offset table has at most this many (start, end) tuples: about 15 MB of Python
# objects, tokenized in well under a second. Long enough that every real chunk, section and
# source file is one segment (a chunk is at most ``chunk_max_tokens``, a few KB); only a
# document with no structure at all is ever cut.
SEGMENT_CHARS = 200_000


def segment_text(text: str, max_chars: int = SEGMENT_CHARS) -> list[tuple[int, int]]:
    """Character ranges of at most ``max_chars`` that partition ``text`` exactly.

    A text within the bound is one range. Otherwise each range ends after the last newline
    inside the bound (whole lines stay together), failing that after the last whitespace, and
    failing that at the bound itself (base64, an unbroken CJK run). Nothing overlaps between
    ranges: a caller that needs context across a cut has to carry it itself.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    size = len(text)
    if size <= max_chars:
        return [(0, size)]
    segments: list[tuple[int, int]] = []
    start = 0
    while size - start > max_chars:
        limit = start + max_chars
        cut = text.rfind("\n", start + 1, limit)
        if cut >= 0:
            cut += 1
        else:
            cut = next((i + 1 for i in range(limit - 1, start, -1) if text[i].isspace()), limit)
        segments.append((start, cut))
        start = cut
    segments.append((start, size))
    return segments


def count_tokens_capped(
    text: str, cap: int, profile: EmbeddingProfile | None = None, *, tokenizer: Any = None,
) -> int:
    """Real token count of ``text`` (specials included), segment by segment, stopping early.

    Exact for a text of at most ``SEGMENT_CHARS``. For a longer one the segments are counted in
    order and the scan stops as soon as the running total passes ``cap``: the result is then
    a lower bound that already exceeds ``cap``, which is all an over-the-limit check needs.
    Segments are cut at whitespace, so the sum is the count of the whole text to within the
    word-start marker a segment may gain.
    """
    profile = profile or get_profile()
    if len(text) <= SEGMENT_CHARS:
        return count_tokens([text], profile, tokenizer=tokenizer)[0]
    total = SPECIAL_TOKENS
    for start, end in segment_text(text):
        total += count_tokens([text[start:end]], profile, tokenizer=tokenizer)[0] - SPECIAL_TOKENS
        if total > cap:
            break
    return total


def token_offsets(text: str, profile: EmbeddingProfile | None = None) -> list[tuple[int, int]]:
    """Character range of every real token of ``text`` (no special tokens), from one tokenize.

    The tokenizer maps a word-start token to the whitespace before the word as well; the
    ranges returned here start at the word, so a cut at a token start never lands inside
    the whitespace that belongs between two words.
    """
    tokenizer = load_tokenizer(profile or get_profile())
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False,
        return_attention_mask=False, return_token_type_ids=False, verbose=False,
    )
    ranges = []
    for start, end in encoded["offset_mapping"]:
        while start < end and text[start].isspace():
            start += 1
        ranges.append((int(start), int(end)))
    return ranges


def split_to_budget(
    text: str,
    *,
    max_tokens: int,
    profile: EmbeddingProfile | None = None,
    prefer: Prefer = "line",
) -> list[tuple[int, int]]:
    """Character spans partitioning ``text``, each at most ``max_tokens`` real tokens.

    Whole lines (``prefer="line"``) or whitespace-delimited words (``"space"``) are packed
    greedily; a single line or word over the budget is cut at token offsets. A text that
    already fits comes back as ``[(0, len(text))]`` after one tokenize. A text longer than
    ``SEGMENT_CHARS`` is packed segment by segment (see ``segment_text``), so no span crosses a
    segment edge and no tokenizer call sees more than one segment.
    """
    if prefer not in ("line", "space"):
        raise ValueError(f"prefer must be 'line' or 'space', not {prefer!r}")
    if max_tokens <= SPECIAL_TOKENS + EDGE_TOKENS:
        raise ValueError(f"max_tokens must exceed {SPECIAL_TOKENS + EDGE_TOKENS}")
    if not text:
        return [(0, 0)]
    if ascii_within_limit(text, max_tokens):
        return [(0, len(text))]

    profile = profile or get_profile()
    if len(text) > SEGMENT_CHARS:
        spans: list[tuple[int, int]] = []
        for start, end in segment_text(text):
            spans.extend((start + a, start + b) for a, b in split_to_budget(
                text[start:end], max_tokens=max_tokens, profile=profile, prefer=prefer))
        return spans
    tokenizer = load_tokenizer(profile)
    offsets = token_offsets(text, profile)
    if len(offsets) + SPECIAL_TOKENS <= max_tokens:
        return [(0, len(text))]
    budget = max_tokens - SPECIAL_TOKENS - EDGE_TOKENS
    return _verified(text, _pack(text, offsets, budget, prefer), max_tokens, budget, prefer, profile, tokenizer)


def _pack(text: str, offsets: list[tuple[int, int]], budget: int, prefer: Prefer) -> list[tuple[int, int]]:
    """Greedy spans of at most ``budget`` tokens by the single tokenization ``offsets``."""
    starts = [start for start, _ in offsets]
    pattern = _LINES if prefer == "line" else _WORDS
    unit_ends = [m.end() for m in pattern.finditer(text)]
    if not unit_ends or unit_ends[-1] != len(text):
        unit_ends.append(len(text))      # all-whitespace text, or a tail the pattern left

    spans: list[tuple[int, int]] = []
    span_start, span_tokens, position = 0, 0, 0
    for end in unit_ends:
        first, last = bisect_left(starts, position), bisect_left(starts, end)
        unit_tokens = last - first
        if unit_tokens > budget:
            if position > span_start:
                spans.append((span_start, position))
            # Cut the oversized unit at the start offset of every budget-th token; the
            # remainder stays open so the units after it can still join it.
            cuts = [starts[i] for i in range(first + budget, last, budget)]
            edges = [position, *cuts]
            spans.extend((a, b) for a, b in zip(edges, edges[1:]) if b > a)
            span_start = edges[-1]
            span_tokens = last - (first + budget * len(cuts))
        elif span_tokens + unit_tokens > budget:
            spans.append((span_start, position))
            span_start, span_tokens = position, unit_tokens
        else:
            span_tokens += unit_tokens
        position = end
    if len(text) > span_start:
        spans.append((span_start, len(text)))
    return spans


def _verified(
    text: str, spans: list[tuple[int, int]], max_tokens: int, budget: int,
    prefer: Prefer, profile: EmbeddingProfile, tokenizer: Any,
) -> list[tuple[int, int]]:
    """Re-count every span with the real tokenizer; re-split any that a cut made too long.

    A span cut inside a word tokenizes slightly differently from the same characters inside
    the whole text (a leading word-start marker), so the single-tokenization packing is a
    plan, not a proof. The overshoot shrinks the budget of the failing span until it fits.
    """
    counts = count_tokens([text[a:b] for a, b in spans], profile, tokenizer=tokenizer)
    out: list[tuple[int, int]] = []
    for (a, b), tokens in zip(spans, counts):
        if tokens <= max_tokens:
            out.append((a, b))
            continue
        smaller = budget - (tokens - max_tokens)
        if smaller < 1:
            raise ValueError(f"cannot split {b - a} characters into spans of {max_tokens} tokens")
        piece = text[a:b]
        piece_offsets = token_offsets(piece, profile)
        inner = _pack(piece, piece_offsets, smaller, prefer)
        out.extend(
            (a + x, a + y)
            for x, y in _verified(piece, inner, max_tokens, smaller, prefer, profile, tokenizer)
        )
    return out
