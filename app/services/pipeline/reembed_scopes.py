"""The named scopes a re-embed task can run, and what each does with a stale point.

A point is stale when its ``embed_fp`` payload is not the current profile's fingerprint (or is
absent): its vector was computed before the profile existed. The scopes are fixed presets, not
free-form filters, so a task can only touch a collection the way a reviewed preset says.

Qdrant cannot filter on token length, so the token-length part of a scope is applied to the
scrolled text with the model's real tokenizer: ``categorize_texts`` puts every stale point in
exactly one category.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

from qdrant_client import models

from core.indexing.embedding_profile import EmbeddingProfile, get_profile, tokens_over_limit

STAMP = "stamp"            # the stored vector is already what the profile computes: only the fingerprint is missing
CHUNK = "chunk"            # re-embed the stored text
OVERSIZED = "oversized"    # longer than the model's limit: cannot be embedded as it is
SKIP = "skip"              # another scope's work (or already right)
CATEGORIES = (STAMP, CHUNK, OVERSIZED, SKIP)

# The tokens a pre-profile code/journal/upload vector was cut at (FlagEmbedding's silent
# passage_max_length): only text longer than this was ever truncated.
OLD_CUT_TOKENS = 512
# The limit every pre-profile paper/arXiv vector was computed under (the old worker's max_length).
OLD_WORKER_TOKENS = 2048


@dataclass(frozen=True)
class ReembedScope:
    name: str
    family: str                          # the request's ``collection``: what precedes the colon
    kind: str                            # "code" | "papers" | "journal" | "paper_rebuild"
    source_types: tuple[str, ...] = ()   # papers: only these ``source_type`` values
    skip_up_to: int = 0                  # tokens <= this are another scope's work
    skip_above: Optional[int] = None     # tokens > this are another scope's work
    stamp_up_to: int = 0                 # tokens <= this only need the fingerprint
    resplit: bool = False                # an oversized point is cut into pieces (code collections)
    text_field: str = "content"          # the payload field that was embedded


SCOPES: dict[str, ReembedScope] = {scope.name: scope for scope in (
    # Points whose embedded text is longer than the old cut: their vectors really were truncated.
    ReembedScope("code_chunks:truncated", "code_chunks", "code", skip_up_to=OLD_CUT_TOKENS, resplit=True),
    ReembedScope("public_code_chunks:truncated", "public_code_chunks", "code", skip_up_to=OLD_CUT_TOKENS,
                 resplit=True),
    # The rest: only the dtype changes (bf16 -> fp16), so it can wait and run in the background.
    ReembedScope("code_chunks:align", "code_chunks", "code", skip_above=OLD_CUT_TOKENS),
    ReembedScope("public_code_chunks:align", "public_code_chunks", "code", skip_above=OLD_CUT_TOKENS),
    # Journal chunks were cut by the whitespace-word chunker (one reached 66,039 tokens): rebuilt from SQL.
    ReembedScope("journal_chunks:rebuild", "journal_chunks", "journal"),
    # Written live by the backend: all re-embedded.
    ReembedScope("papers:uploads", "papers", "papers", source_types=("upload", "webpage")),
    # Written by the worker at 2,048 tokens in fp16: unchanged below that.
    ReembedScope("papers:pipeline", "papers", "papers", source_types=("paper",), stamp_up_to=OLD_WORKER_TOKENS),
    ReembedScope("arxiv:all", "arxiv", "papers", stamp_up_to=OLD_WORKER_TOKENS),
    # The few paper/arXiv documents whose chunks are over the limit (the old worker cut them at 2,048
    # tokens) or that hold two L1 chunks (a body mistaken for an abstract): re-chunked from their
    # Markdown with the abstract-capped chunker and replaced, not re-embedded in place. One scope per
    # collection: a document is rebuilt in the collection its payload routes to.
    ReembedScope("papers:rebuild", "papers", "paper_rebuild"),
    ReembedScope("arxiv:rebuild", "arxiv", "paper_rebuild"),
)}


def get_scope(name: str) -> ReembedScope:
    try:
        return SCOPES[name]
    except KeyError:
        raise ValueError(f"unknown re-embed scope {name!r}; scopes: {sorted(SCOPES)}") from None


def collection_name(scope: ReembedScope) -> str:
    """The Qdrant collection the scope works on, resolved like the collection's own writer does."""
    if scope.family == "code_chunks":
        return "code_chunks"                      # CodeVectorStore's default collection name
    if scope.family == "public_code_chunks":
        from app.services.public_code.vector_store import PUBLIC_CODE_COLLECTION
        return PUBLIC_CODE_COLLECTION
    if scope.family == "journal_chunks":
        return os.environ.get("JOURNAL_QDRANT_COLLECTION", "journal_chunks")
    from core.storage.corpus_layout import paper_collections
    return paper_collections()[scope.family]


def scope_conditions(scope: ReembedScope) -> list[models.Condition]:
    if not scope.source_types:
        return []
    return [models.FieldCondition(key="source_type", match=models.MatchAny(any=list(scope.source_types)))]


def stale_filter(scope: ReembedScope, fingerprint: str) -> models.Filter:
    """The scope's points whose ``embed_fp`` is not ``fingerprint`` (absent counts as not equal)."""
    return models.Filter(
        must=scope_conditions(scope),
        must_not=[models.FieldCondition(key="embed_fp", match=models.MatchValue(value=fingerprint))],
    )


def current_filter(scope: ReembedScope, fingerprint: str) -> models.Filter:
    """The scope's points that already carry ``fingerprint``."""
    return models.Filter(must=[
        *scope_conditions(scope),
        models.FieldCondition(key="embed_fp", match=models.MatchValue(value=fingerprint)),
    ])


def categorize(scope: ReembedScope, tokens: int, max_seq_len: int) -> str:
    if tokens <= scope.skip_up_to:
        return SKIP
    if tokens <= scope.stamp_up_to:
        return STAMP
    if scope.skip_above is not None and tokens > scope.skip_above:
        return SKIP
    return CHUNK if tokens <= max_seq_len else OVERSIZED


def categorize_texts(
    scope: ReembedScope, texts: Sequence[str], profile: Optional[EmbeddingProfile] = None,
) -> list[tuple[str, int]]:
    """``(category, tokens)`` of each text by its real token count (specials included).

    Texts that provably fit the smallest threshold of the scope (``ascii_within_limit``) are
    not tokenized: they are reported at that threshold, which puts them in the same category
    as their true count would.

    ``tokens_over_limit`` is exact for a text of at most ``SEGMENT_CHARS`` characters. For a longer
    one it counts segment by segment and stops as soon as the total passes the limit it was given
    (``floor``, the smallest threshold), so its count is only a lower bound just above ``floor``,
    and ``categorize`` compares the count with every threshold of the scope. Such a text is
    counted again against ``profile.max_seq_len``: the largest threshold, so the result is exact
    when the text fits the model and still above it, hence OVERSIZED, when it does not.
    """
    from core.indexing.token_budget import SEGMENT_CHARS, count_tokens_capped

    profile = profile or get_profile()
    floor = min(t for t in (scope.skip_up_to, scope.stamp_up_to, scope.skip_above, profile.max_seq_len) if t)
    exact = dict(tokens_over_limit(list(texts), floor, profile))
    for index in exact:
        if len(texts[index]) > SEGMENT_CHARS:
            exact[index] = count_tokens_capped(texts[index], profile.max_seq_len, profile)
    out = []
    for index in range(len(texts)):
        tokens = exact.get(index, floor)
        out.append((categorize(scope, tokens, profile.max_seq_len), tokens))
    return out
