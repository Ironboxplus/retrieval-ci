"""Core contracts shared by private and public code-vector collections."""

from __future__ import annotations

import pytest

from core.code_kb.vector_records import (
    CodeChunkVectorRecord,
    build_code_chunk_payload,
    code_chunk_point_id,
)
from core.indexing.embedding_profile import get_profile


def record() -> CodeChunkVectorRecord:
    return CodeChunkVectorRecord(
        rel_path="src/encoder.c",
        language="c",
        start_line=10,
        end_line=24,
        label="encode_frame",
        content="# path: src/encoder.c\nint encode_frame(void);",
        code="int encode_frame(void);",
    )


def test_payload_builder_combines_opaque_scope_with_canonical_code_chunk_fields():
    payload = build_code_chunk_payload(
        record(),
        scope={"source_id": "ffmpeg", "revision": "commit", "domain": "multimedia"},
    )

    assert payload == {
        "source_id": "ffmpeg",
        "revision": "commit",
        "domain": "multimedia",
        "path": "src/encoder.c",
        "language": "c",
        "start_line": 10,
        "end_line": 24,
        "label": "encode_frame",
        "content": "# path: src/encoder.c\nint encode_frame(void);",
        "code": "int encode_frame(void);",
        "embed_fp": get_profile().fingerprint,
    }


def test_point_identity_is_stable_inside_a_scope_and_isolated_between_scopes():
    item = record()

    first = code_chunk_point_id("public:ffmpeg:commit", item)
    second = code_chunk_point_id("public:ffmpeg:commit", item)
    other_scope = code_chunk_point_id("private:user:project", item)

    assert first == second
    assert first != other_scope


def test_point_identity_distinguishes_chunks_on_the_same_source_line():
    first = record()
    second = CodeChunkVectorRecord(
        rel_path=first.rel_path,
        language=first.language,
        start_line=first.start_line,
        end_line=first.end_line,
        label=first.label,
        content=first.content,
        code=first.code,
        chunk_ordinal=1,
    )

    assert code_chunk_point_id("public:ffmpeg:commit", first) != code_chunk_point_id(
        "public:ffmpeg:commit", second
    )


def test_scope_cannot_override_canonical_code_chunk_payload_fields():
    with pytest.raises(ValueError, match="cannot override code chunk field"):
        build_code_chunk_payload(record(), scope={"path": "wrong.py"})
