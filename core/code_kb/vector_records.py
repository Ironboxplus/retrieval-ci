"""Stable code-chunk vector facts shared by every code corpus.

The module deliberately has no Qdrant client and no identity policy.  Callers
provide an opaque scope namespace for point identity and their own small scope
payload (private owner/project or public source/revision/domain).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Mapping, Sequence

from core.indexing.embedding_profile import (
    EmbeddingProfile,
    count_tokens,
    current_embed_fp,
    get_profile,
    tokens_over_limit,
)
from core.indexing.token_budget import EDGE_TOKENS, SPECIAL_TOKENS, split_to_budget


_CANONICAL_PAYLOAD_FIELDS = frozenset(
    {
        "path",
        "language",
        "start_line",
        "end_line",
        "label",
        "content",
        "code",
        # Fingerprint of the embedding profile the stored vector was computed under; vector
        # reuse compares it with the current one like any other canonical field.
        "embed_fp",
    }
)


@dataclass(frozen=True, slots=True)
class CodeChunkVectorRecord:
    """One flattened AST or line-window code chunk ready for vector storage."""

    rel_path: str
    language: str | None
    start_line: int
    end_line: int
    label: str
    content: str
    code: str
    chunk_ordinal: int = 0


def code_chunk_point_id(scope_namespace: str, record: CodeChunkVectorRecord) -> int:
    """Return a deterministic Qdrant-safe ID isolated by an opaque scope namespace."""

    return code_chunk_identity_id(
        scope_namespace,
        rel_path=record.rel_path,
        start_line=record.start_line,
        label=record.label,
        chunk_ordinal=record.chunk_ordinal,
    )


def code_chunk_identity_id(
    scope_namespace: str,
    *,
    rel_path: str,
    start_line: int,
    label: str,
    chunk_ordinal: int = 0,
) -> int:
    """Return an ID from the stable, storage-relevant part of a code chunk."""

    identity = f"{scope_namespace}:{rel_path}:{start_line}:{label}:{chunk_ordinal}"
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:15], 16)


def build_code_chunk_payload(
    record: CodeChunkVectorRecord,
    *,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine caller-owned scope fields with the canonical code-chunk payload."""

    reserved_fields = _CANONICAL_PAYLOAD_FIELDS.intersection(scope)
    if reserved_fields:
        names = ", ".join(sorted(reserved_fields))
        raise ValueError(f"scope cannot override code chunk field(s): {names}")

    return {
        **dict(scope),
        "path": record.rel_path,
        "language": record.language or "",
        "start_line": record.start_line,
        "end_line": record.end_line,
        "label": record.label,
        "content": record.content,
        "code": record.code,
        "embed_fp": current_embed_fp(),
    }


def fit_records_to_budget(records: Sequence[CodeChunkVectorRecord]) -> list[CodeChunkVectorRecord]:
    """Split every record whose embed text exceeds the chunk token budget into line pieces.

    The budget is checked on ``content`` (what the model embeds). A piece takes whole lines
    of ``code`` (a single line over the budget is cut at token offsets), carries the same
    ``# path:`` header, the line range of its own lines and the same label. ``chunk_ordinal``
    is re-enumerated per file over the final pieces, so every piece has its own point id.
    Records within the budget come back unchanged.
    """

    records = list(records)
    profile = get_profile()
    oversized = {
        index for index, _ in tokens_over_limit(
            [record.content for record in records], profile.chunk_max_tokens, profile)
    }
    pieces: list[CodeChunkVectorRecord] = []
    for index, record in enumerate(records):
        pieces.extend(_split_record(record, profile) if index in oversized else [record])

    next_ordinal: dict[str, int] = {}
    numbered = []
    for piece in pieces:
        ordinal = next_ordinal.get(piece.rel_path, 0)
        next_ordinal[piece.rel_path] = ordinal + 1
        numbered.append(piece if piece.chunk_ordinal == ordinal else replace(piece, chunk_ordinal=ordinal))
    return numbered


def _split_record(record: CodeChunkVectorRecord, profile: EmbeddingProfile) -> list[CodeChunkVectorRecord]:
    limit = profile.chunk_max_tokens
    header = f"# path: {record.rel_path}\n"
    budget = limit - (count_tokens([header], profile)[0] - SPECIAL_TOKENS)
    while budget > SPECIAL_TOKENS + EDGE_TOKENS:
        spans = _merge_blank_spans(
            record.code, split_to_budget(record.code, max_tokens=budget, profile=profile, prefer="line"))
        pieces = [_piece(record, header, start, end) for start, end in spans]
        over = tokens_over_limit([piece.content for piece in pieces], limit, profile)
        if not over:
            return pieces
        budget -= max(tokens - limit for _, tokens in over)     # the header and joint cost a few tokens
    raise ValueError(f"cannot fit a chunk of {record.rel_path} into {limit} tokens")


def _merge_blank_spans(code: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fold a whitespace-only span into its neighbour: a chunk of nothing is not a chunk."""
    merged: list[tuple[int, int]] = []
    carry = None
    for start, end in spans:
        if carry is not None:
            start, carry = carry, None
        if not code[start:end].strip():
            if merged:
                merged[-1] = (merged[-1][0], end)
            else:
                carry = start
            continue
        merged.append((start, end))
    return merged or [(0, len(code))]


def _piece(record: CodeChunkVectorRecord, header: str, start: int, end: int) -> CodeChunkVectorRecord:
    code = record.code[start:end]
    first_line = record.start_line + record.code.count("\n", 0, start)
    return replace(
        record,
        start_line=first_line,
        end_line=max(first_line + code.rstrip("\n").count("\n"), first_line),
        content=f"{header}{code}",
        code=code,
    )
