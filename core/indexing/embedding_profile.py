"""The embedding profile: every number that decides what a stored vector is.

One frozen profile per model holds the sequence limit, dtype, pooling, batch and chunk
budget, and its ``fingerprint`` names the part that changes vectors. The backend embedder,
the remote embed worker, the chunkers and every Qdrant writer read this module instead of
keeping their own copy of a number.

Standard library only (transformers is imported when a tokenizer is first needed), so the
standalone embed worker can import it without the project's ``config`` layer.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Sequence

# Same variable the settings layer reads (``Settings`` nested delimiter + ``EmbeddingSettings.model_name``).
MODEL_ID_ENV = "EMBEDDING__MODEL_NAME"
# Deployment override: load weights (and the tokenizer) from a local directory while the
# model id stays the repo id, so profile lookup and dimension stay correct.
MODEL_PATH_ENV = "BGE_M3_PATH"

DEFAULT_MODEL_ID = "BAAI/bge-m3"
# Models without a profile (the dense-only sentence-transformers rows) batch with this.
DEFAULT_ENCODE_BATCH_SIZE = 64

# Fields that change the vector computed for a given text, so they are part of the
# fingerprint. ``encode_batch_size`` (only groups texts for the GPU) and ``chunk_max_tokens``
# (decides where text is cut, not what a cut text encodes to) are deliberately left out:
# changing either must not mark stored vectors stale.
_FINGERPRINT_FIELDS = (
    "model_id",       # which weights
    "revision",       # which snapshot of them
    "dimension",      # output width
    "max_seq_len",    # input tokens beyond it are an error, but the limit is what the vector was made under
    "dtype",          # fp16 and bf16 vectors differ in the low bits
    "pooling",        # cls vs mean gives different vectors
    "normalize",      # unit length or not
    "sparse",         # whether a lexical vector is produced alongside
)


@dataclass(frozen=True)
class EmbeddingProfile:
    model_id: str = DEFAULT_MODEL_ID
    # The snapshot arc99 serves and the 2026-11-14 provenance check re-encoded against.
    revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    dimension: int = 1024
    # The model's own sentence_bert_config.json limit; counts the two special tokens.
    max_seq_len: int = 8192
    dtype: str = "float16"
    pooling: str = "cls"
    normalize: bool = True
    sparse: bool = True
    encode_batch_size: int = DEFAULT_ENCODE_BATCH_SIZE
    # Ceiling on one stored chunk. It is the limit every existing paper/arXiv vector was
    # computed under (worker max_length 2048); measured chunk lengths sit below it for
    # >99.99% of paper chunks and >99% of code chunks, so it moves almost no existing chunk
    # while bounding the outliers (journal 66K tokens, public code 362K tokens).
    chunk_max_tokens: int = 2048

    @cached_property
    def fingerprint(self) -> str:
        """First 16 hex of sha256 over the canonical JSON of the vector-changing fields."""
        canonical = json.dumps(
            {name: getattr(self, name) for name in _FINGERPRINT_FIELDS},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


PROFILES: dict[str, EmbeddingProfile] = {
    DEFAULT_MODEL_ID: EmbeddingProfile(),
}


def get_profile(model_id: str | None = None) -> EmbeddingProfile:
    """The profile of ``model_id``, or of the configured model when omitted."""
    name = model_id or os.environ.get(MODEL_ID_ENV) or DEFAULT_MODEL_ID
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"no embedding profile for model {name!r}; profiles: {sorted(PROFILES)}"
        ) from None


def current_embed_fp() -> str:
    """Fingerprint stamped on every point written under the configured model."""
    return get_profile().fingerprint


@lru_cache(maxsize=4)
def _tokenizer(model_id: str, model_path: str | None) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path or model_id)


def load_tokenizer(profile: EmbeddingProfile) -> Any:
    """The profile's tokenizer: from ``BGE_M3_PATH`` when set, else the model id."""
    return _tokenizer(profile.model_id, os.environ.get(MODEL_PATH_ENV))


def count_tokens(
    texts: Sequence[str], profile: EmbeddingProfile, *, tokenizer: Any = None,
) -> list[int]:
    """Real token count of each text with the model's special tokens, never truncated."""
    if not texts:
        return []
    tokenizer = tokenizer or load_tokenizer(profile)
    encoded = tokenizer(
        list(texts), add_special_tokens=True, truncation=False,
        return_attention_mask=False, return_token_type_ids=False, verbose=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


# <s> and </s>, plus the dummy-prefix "▁" SentencePiece puts in front of the text, which
# covers no input character ("." is 4 tokens, "," * 62 is 65).
ASCII_TOKEN_OVERHEAD = 3


def ascii_within_limit(text: str, limit: int) -> bool:
    """True when ``text`` provably fits ``limit`` tokens without tokenizing it.

    Every token of an ASCII text covers at least one character (ASCII has no normalisation
    that adds characters), so ``len(text)`` content tokens plus the fixed overhead is an
    upper bound. The one definition of that shortcut; callers must not copy the rule.
    """
    return text.isascii() and len(text) + ASCII_TOKEN_OVERHEAD <= limit


def tokens_over_limit(
    texts: Sequence[str], limit: int, profile: EmbeddingProfile, *, tokenizer: Any = None,
) -> list[tuple[int, int]]:
    """``(index, tokens)`` of every text longer than ``limit`` tokens (specials included).

    Texts ``ascii_within_limit`` vouches for are not tokenized at all; everything else gets
    a real count. A text longer than ``token_budget.SEGMENT_CHARS`` is counted segment by
    segment and only until it passes ``limit`` (see ``count_tokens_capped``), so its ``tokens``
    is a lower bound above the limit rather than the whole text's count: no tokenizer call
    sees an unbounded text.
    """
    from core.indexing.token_budget import SEGMENT_CHARS, count_tokens_capped    # it imports this module

    undecided = [i for i, text in enumerate(texts) if not ascii_within_limit(text, limit)]
    if not undecided:
        return []
    short = [i for i in undecided if len(texts[i]) <= SEGMENT_CHARS]
    counted = dict(zip(short, count_tokens([texts[i] for i in short], profile, tokenizer=tokenizer)))
    for i in undecided:
        if i not in counted:
            counted[i] = count_tokens_capped(texts[i], limit, profile, tokenizer=tokenizer)
    return [(i, counted[i]) for i in undecided if counted[i] > limit]
