"""Reconcile the SQL chunk rows of one paper with the chunks the canonical chunker emits now.

Pure planning plus the SQL apply, shared by ``scripts/reconcile_paper_chunk_schema.py`` (one
conference slice) and the ``papers:rebuild`` re-embed scope (documents whose chunks were too
long). Matching chunks are retained, missing canonical chunks are inserted, obsolete chunks are
removed and row order/metadata are updated in place. The plan lists the exact stale Qdrant point
ids so vector deletion can happen only after the new chunks are embedded and upserted.

A chunk's identity is ``(level, start_idx, content)``: the same triple that makes up its Qdrant
point id (``scripts/remote_embed/dump_chunks.py``), so a retained chunk keeps its point.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.models.schemas import Chunk
from scripts.remote_embed.dump_chunks import chunk_id_int, chunk_id_str


@dataclass(frozen=True)
class ExistingChunkRecord:
    row_id: int
    content: str
    level: int
    chunk_index: int
    start_idx: int | None
    end_idx: int | None
    token_count: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetainedChunk:
    row_id: int
    expected_index: int
    chunk: Chunk
    requires_update: bool


@dataclass(frozen=True)
class InsertedChunk:
    expected_index: int
    chunk: Chunk


@dataclass
class DocumentReconciliation:
    retained: list[RetainedChunk] = field(default_factory=list)
    inserts: list[InsertedChunk] = field(default_factory=list)
    delete_row_ids: list[int] = field(default_factory=list)
    stale_point_ids: list[int] = field(default_factory=list)


def _normalized_start(value: int | None) -> int:
    return int(value or 0)


def _vector_identity(content: str, level: int, start_idx: int | None) -> tuple[int, int, str]:
    return int(level), _normalized_start(start_idx), content


def _expected_identity(chunk: Chunk) -> tuple[int, int, str]:
    return _vector_identity(chunk.content, chunk.level, chunk.start_idx)


def _existing_identity(chunk: ExistingChunkRecord) -> tuple[int, int, str]:
    return _vector_identity(chunk.content, chunk.level, chunk.start_idx)


def chunk_point_id(doc_id: int, chunk: Chunk) -> int:
    """The Qdrant point id of ``chunk`` of document ``doc_id`` (what the paper ingest derives)."""
    return chunk_id_int(chunk_id_str(doc_id, chunk.content, chunk.level, _normalized_start(chunk.start_idx)))


def _validate_expected_schema(expected: Sequence[Chunk]) -> None:
    if not expected:
        raise ValueError("canonical paper chunker returned no chunks")
    l1_count = sum(chunk.level == 1 for chunk in expected)
    if l1_count != 1:
        raise ValueError(f"canonical paper schema requires one L1 chunk, got {l1_count}")
    invalid_levels = sorted({chunk.level for chunk in expected if chunk.level not in (1, 2)})
    if invalid_levels:
        raise ValueError(f"canonical paper schema only permits levels 1 and 2: {invalid_levels}")
    if expected[0].level != 1:
        raise ValueError("canonical paper schema requires L1 as chunk_index 0")


def _retained_row_requires_update(
    existing: ExistingChunkRecord,
    expected_index: int,
    expected: Chunk,
) -> bool:
    return any((
        existing.chunk_index != expected_index,
        _normalized_start(existing.start_idx) != _normalized_start(expected.start_idx),
        existing.end_idx != expected.end_idx,
        existing.token_count != len(expected.content),
        existing.metadata != expected.metadata,
    ))


def build_document_reconciliation(
    doc_id: int,
    existing: Sequence[ExistingChunkRecord],
    expected: Sequence[Chunk],
) -> DocumentReconciliation:
    """Return the exact SQL/vector mutations needed for one document."""
    _validate_expected_schema(expected)
    available: dict[tuple[int, int, str], deque[ExistingChunkRecord]] = defaultdict(deque)
    for row in existing:
        available[_existing_identity(row)].append(row)

    plan = DocumentReconciliation()
    expected_identities = {_expected_identity(chunk) for chunk in expected}
    retained_ids: set[int] = set()

    for expected_index, chunk in enumerate(expected):
        identity = _expected_identity(chunk)
        candidates = available.get(identity)
        if candidates:
            row = candidates.popleft()
            retained_ids.add(row.row_id)
            plan.retained.append(RetainedChunk(
                row_id=row.row_id,
                expected_index=expected_index,
                chunk=chunk,
                requires_update=_retained_row_requires_update(row, expected_index, chunk),
            ))
        else:
            plan.inserts.append(InsertedChunk(expected_index=expected_index, chunk=chunk))

    stale_points: set[int] = set()
    for row in existing:
        if row.row_id in retained_ids:
            continue
        plan.delete_row_ids.append(row.row_id)
        identity = _existing_identity(row)
        if identity not in expected_identities:
            stale_points.add(chunk_id_int(chunk_id_str(
                doc_id,
                row.content,
                row.level,
                _normalized_start(row.start_idx),
            )))
    plan.stale_point_ids = sorted(stale_points)
    return plan


def _decode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    return json.loads(value)


def load_existing_chunks(conn: sqlite3.Connection, doc_id: int) -> list[ExistingChunkRecord]:
    rows = conn.execute(
        """
        SELECT id, content, level, chunk_index, start_idx, end_idx,
               token_count, metadata_json
        FROM chunks
        WHERE doc_id = ?
        ORDER BY chunk_index, id
        """,
        (doc_id,),
    ).fetchall()
    return [ExistingChunkRecord(
        row_id=row[0],
        content=row[1],
        level=row[2],
        chunk_index=row[3],
        start_idx=row[4],
        end_idx=row[5],
        token_count=row[6],
        metadata=_decode_metadata(row[7]),
    ) for row in rows]


def chunk_metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))


def apply_document_plan(
    conn: sqlite3.Connection,
    doc_id: int,
    plan: DocumentReconciliation,
) -> list[int]:
    if plan.delete_row_ids:
        placeholders = ",".join("?" for _ in plan.delete_row_ids)
        conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", plan.delete_row_ids)

    for item in plan.retained:
        if not item.requires_update:
            continue
        chunk = item.chunk
        conn.execute(
            """
            UPDATE chunks
            SET chunk_index = ?, start_idx = ?, end_idx = ?, token_count = ?, metadata_json = ?
            WHERE id = ?
            """,
            (
                item.expected_index,
                chunk.start_idx,
                chunk.end_idx,
                len(chunk.content),
                chunk_metadata_json(chunk.metadata),
                item.row_id,
            ),
        )

    inserted_ids: list[int] = []
    for item in plan.inserts:
        chunk = item.chunk
        cursor = conn.execute(
            """
            INSERT INTO chunks (
                doc_id, content, level, chunk_index, start_idx, end_idx,
                token_count, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                chunk.content,
                chunk.level,
                item.expected_index,
                chunk.start_idx,
                chunk.end_idx,
                len(chunk.content),
                chunk_metadata_json(chunk.metadata),
            ),
        )
        inserted_ids.append(int(cursor.lastrowid))
    return inserted_ids


def apply_document_plan_atomically(
    conn: sqlite3.Connection,
    doc_id: int,
    plan: DocumentReconciliation,
) -> list[int]:
    """``apply_document_plan`` in a savepoint: a failure leaves the document's rows as they were."""
    conn.execute("SAVEPOINT reconcile_paper")
    try:
        inserted_ids = apply_document_plan(conn, doc_id, plan)
        conn.execute("RELEASE SAVEPOINT reconcile_paper")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT reconcile_paper")
        conn.execute("RELEASE SAVEPOINT reconcile_paper")
        raise
    return inserted_ids
