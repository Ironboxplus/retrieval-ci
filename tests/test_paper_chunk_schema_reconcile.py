"""TDD for reconciling paper chunks to the canonical L1+L2 schema."""
from __future__ import annotations

import argparse
import json
import sqlite3

from core.indexing.chunker import Chunk
from core.indexing.paper_chunk_reconcile import ExistingChunkRecord, build_document_reconciliation
from scripts.reconcile_paper_chunk_schema import reconcile_slice, validate_stale_point_scope


def _chunk(content: str, level: int, *, section: str, start: int = 0) -> Chunk:
    return Chunk(
        id=f"expected-{level}-{section}",
        doc_id="42",
        content=content,
        level=level,
        start_idx=start,
        end_idx=start + len(content),
        metadata={"section_name": section},
    )


def _existing(
    row_id: int,
    content: str,
    level: int,
    *,
    index: int,
    section: str,
    start: int = 0,
) -> ExistingChunkRecord:
    return ExistingChunkRecord(
        row_id=row_id,
        content=content,
        level=level,
        chunk_index=index,
        start_idx=start,
        end_idx=start + len(content),
        token_count=len(content),
        metadata={"section_name": section},
    )


def test_reconcile_adds_l1_and_preserves_matching_l2():
    existing = [_existing(10, "body", 2, index=0, section="Method")]
    expected = [
        _chunk("title\n\nabstract", 1, section="title_summary"),
        _chunk("body", 2, section="Method"),
    ]

    plan = build_document_reconciliation(42, existing, expected)

    assert [item.chunk.content for item in plan.inserts] == ["title\n\nabstract"]
    assert [item.row_id for item in plan.retained] == [10]
    assert plan.delete_row_ids == []
    assert plan.stale_point_ids == []


def test_reconcile_removes_only_old_short_fragments():
    existing = [
        _existing(10, "body", 2, index=0, section="Method"),
        _existing(11, "tiny", 2, index=1, section="Tail"),
    ]
    expected = [
        _chunk("title\n\nabstract", 1, section="title_summary"),
        _chunk("body", 2, section="Method"),
    ]

    plan = build_document_reconciliation(42, existing, expected)

    assert plan.delete_row_ids == [11]
    assert len(plan.stale_point_ids) == 1
    assert [item.row_id for item in plan.retained] == [10]


def test_reconcile_inserts_missing_l2_and_updates_retained_order_and_metadata():
    existing = [_existing(10, "body", 2, index=8, section="old")]
    expected = [
        _chunk("title\n\nabstract", 1, section="title_summary"),
        _chunk("body", 2, section="Method"),
        _chunk("results", 2, section="Results"),
    ]

    plan = build_document_reconciliation(42, existing, expected)

    assert [(item.expected_index, item.chunk.content) for item in plan.inserts] == [
        (0, "title\n\nabstract"),
        (2, "results"),
    ]
    assert len(plan.retained) == 1
    retained = plan.retained[0]
    assert retained.row_id == 10
    assert retained.expected_index == 1
    assert retained.requires_update is True


def test_reconcile_duplicate_row_does_not_delete_shared_valid_point():
    existing = [
        _existing(10, "body", 2, index=0, section="Method"),
        _existing(11, "body", 2, index=1, section="Method"),
    ]
    expected = [
        _chunk("title\n\nabstract", 1, section="title_summary"),
        _chunk("body", 2, section="Method"),
    ]

    plan = build_document_reconciliation(42, existing, expected)

    assert plan.delete_row_ids == [11]
    assert plan.stale_point_ids == []


def test_reconcile_is_idempotent_when_rows_match_expected_schema():
    expected = [
        _chunk("title\n\nabstract", 1, section="title_summary"),
        _chunk("body", 2, section="Method"),
    ]
    existing = [
        _existing(10, "title\n\nabstract", 1, index=0, section="title_summary"),
        _existing(11, "body", 2, index=1, section="Method"),
    ]

    plan = build_document_reconciliation(42, existing, expected)

    assert plan.inserts == []
    assert plan.delete_row_ids == []
    assert plan.stale_point_ids == []
    assert all(item.requires_update is False for item in plan.retained)


def test_reconcile_slice_matches_section_l1_and_is_idempotent(tmp_path):
    from core.indexing.chunker import create_chunker

    db_path = tmp_path / "retrieval.db"
    md_path = tmp_path / "paper.md"
    report_path = tmp_path / "reconcile.json"
    markdown = """# Canonical Paper Title

## Abstract

This abstract is sufficiently descriptive for the canonical title summary chunk.

## Method

This method section is deliberately long enough to remain a valid level two
chunk after the canonical short-fragment filter is applied to the document.

## Tail

tiny
"""
    md_path.write_text(markdown, encoding="utf-8")

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE documents (id INTEGER PRIMARY KEY, file_path TEXT);
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY, title TEXT, doc_id INTEGER,
            venue TEXT, year INTEGER
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER, content TEXT, level INTEGER, chunk_index INTEGER,
            start_idx INTEGER, end_idx INTEGER, token_count INTEGER,
            metadata_json TEXT
        );
    """)
    conn.execute("INSERT INTO documents VALUES (42, ?)", (str(md_path),))
    conn.execute(
        "INSERT INTO papers VALUES (?, ?, 42, 'ICML', 2026)",
        ("icml-2026-test", "Canonical Paper Title"),
    )
    old_chunks = create_chunker("markdown").chunk_document("42", markdown)
    for index, chunk in enumerate(old_chunks):
        conn.execute(
            """
            INSERT INTO chunks (
                doc_id, content, level, chunk_index, start_idx, end_idx,
                token_count, metadata_json
            ) VALUES (42, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.content,
                chunk.level,
                index,
                chunk.start_idx,
                chunk.end_idx,
                len(chunk.content),
                json.dumps(chunk.metadata),
            ),
        )
    conn.commit()
    conn.close()

    args = argparse.Namespace(
        db=db_path, paper_ids_file=None,
        conf="ICML",
        year=2026,
        limit=0,
        execute=True,
        commit_batch=1,
        checkpoint_interval=1,
        report_path=report_path,
    )
    first = reconcile_slice(args)

    expected = create_chunker("section_l1").chunk_document(
        "42",
        markdown,
        title_hint="Canonical Paper Title",
    )
    conn = sqlite3.connect(db_path)
    actual = conn.execute(
        "SELECT content, level, chunk_index, metadata_json FROM chunks "
        "WHERE doc_id=42 ORDER BY chunk_index, id"
    ).fetchall()
    conn.close()
    assert [(row[0], row[1], row[2]) for row in actual] == [
        (chunk.content, chunk.level, index) for index, chunk in enumerate(expected)
    ]
    assert [json.loads(row[3]) for row in actual] == [chunk.metadata for chunk in expected]
    assert first["inserted"] >= 1
    assert first["failed"] == 0

    second = reconcile_slice(args)
    assert second["inserted"] == 0
    assert second["deleted"] == 0
    assert second["updated"] == 0
    assert second["failed"] == 0


def test_stale_point_scope_accepts_only_requested_venue_and_year():
    validate_stale_point_scope(
        [
            {"venue": "ICML", "year": 2026},
            {"venue": "ICML", "year": 2026},
        ],
        conference="ICML",
        year=2026,
    )


def test_stale_point_scope_rejects_cross_slice_payload():
    import pytest

    with pytest.raises(RuntimeError, match="outside requested slice"):
        validate_stale_point_scope(
            [{"venue": "ICLR", "year": 2026}],
            conference="ICML",
            year=2026,
        )
