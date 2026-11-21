"""TDD coverage for deterministic SQL PDF-path backfill.

The production artifact resolver can locate a canonical PDF when historical
``papers.pdf_path`` values are absent.  This companion maintenance tool repairs
only that metadata gap; it must never scan the artifact tree or overwrite an
already-populated path.
"""
from __future__ import annotations

import sqlite3

from scripts.sync.backfill_paper_pdf_paths import (
    backfill_missing_pdf_paths,
    build_canonical_pdf_path,
)


WINDOWS_PDF_ROOT = (
    r"C:\Users\Arc\Prd\retrival\data\papers\ccf_a_main_conference_corpus\pdf"
)


def _connection_with_papers() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            venue TEXT,
            year INTEGER,
            pdf_path TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO papers(paper_id, venue, year, pdf_path) VALUES (?, ?, ?, ?)",
        [
            ("aaai-2026-37659", "AAAI", 2026, None),
            ("usenix-2025-42", "USENIX Security", 2025, "   "),
            ("icml-2025-99", "ICML", 2025, r"D:\archive\keep-me.pdf"),
        ],
    )
    return connection


def test_build_canonical_pdf_path_matches_existing_corpus_layout():
    assert build_canonical_pdf_path(
        WINDOWS_PDF_ROOT,
        venue="USENIX Security",
        year=2025,
        paper_id="usenix-2025-42",
    ) == (
        r"C:\Users\Arc\Prd\retrival\data\papers\ccf_a_main_conference_corpus"
        r"\pdf\usenix security\2025\usenix-2025-42.pdf"
    )


def test_dry_run_reports_missing_rows_without_changing_sqlite():
    connection = _connection_with_papers()

    result = backfill_missing_pdf_paths(
        connection,
        path_root=WINDOWS_PDF_ROOT,
        apply=False,
    )

    assert result.candidates == 2
    assert result.updated == 0
    assert connection.execute(
        "SELECT pdf_path FROM papers WHERE paper_id = 'aaai-2026-37659'"
    ).fetchone()[0] is None


def test_apply_only_fills_blank_paths_and_preserves_existing_metadata():
    connection = _connection_with_papers()

    result = backfill_missing_pdf_paths(
        connection,
        path_root=WINDOWS_PDF_ROOT,
        apply=True,
    )

    assert result.candidates == 2
    assert result.updated == 2
    assert connection.execute(
        "SELECT pdf_path FROM papers WHERE paper_id = 'aaai-2026-37659'"
    ).fetchone()[0] == (
        r"C:\Users\Arc\Prd\retrival\data\papers\ccf_a_main_conference_corpus"
        r"\pdf\aaai\2026\aaai-2026-37659.pdf"
    )
    assert connection.execute(
        "SELECT pdf_path FROM papers WHERE paper_id = 'usenix-2025-42'"
    ).fetchone()[0] == (
        r"C:\Users\Arc\Prd\retrival\data\papers\ccf_a_main_conference_corpus"
        r"\pdf\usenix security\2025\usenix-2025-42.pdf"
    )
    assert connection.execute(
        "SELECT pdf_path FROM papers WHERE paper_id = 'icml-2025-99'"
    ).fetchone()[0] == r"D:\archive\keep-me.pdf"
