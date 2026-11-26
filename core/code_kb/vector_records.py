"""Stable code-chunk vector facts shared by every code corpus.

The module deliberately has no Qdrant client and no identity policy.  Callers
provide an opaque scope namespace for point identity and their own small scope
payload (private owner/project or public source/revision/domain).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping


_CANONICAL_PAYLOAD_FIELDS = frozenset(
    {
        "path",
        "language",
        "start_line",
        "end_line",
        "label",
        "content",
        "code",
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
    }
