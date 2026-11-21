"""Fill missing ``papers.pdf_path`` values from the canonical corpus layout.

This repairs historical metadata only.  It does not walk the PDF tree, download
files, touch Markdown, modify chunks, or write Qdrant.  The canonical path is
derived from the same ``venue/year/paper_id`` convention used by the collector,
so a dry-run and an apply complete in SQLite without a slow SMB scan.

Usage:
    .venv/Scripts/python.exe scripts/sync/backfill_paper_pdf_paths.py
    .venv/Scripts/python.exe scripts/sync/backfill_paper_pdf_paths.py --execute
    /data/arc/order/.venv/bin/python scripts/sync/backfill_paper_pdf_paths.py \
        --database /data/arc/retrival/data/db/retrieval.db --execute
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pipeline.paper_crawler.ccf_a_main_corpus import sanitize_file_component


DEFAULT_DATABASE_PATH = REPOSITORY_ROOT / "data" / "db" / "retrieval.db"
DEFAULT_WINDOWS_PDF_ROOT = (
    r"C:\Users\Arc\Prd\retrival\data\papers\ccf_a_main_conference_corpus\pdf"
)
_MISSING_PATH_SQL = "TRIM(COALESCE(pdf_path, '')) = ''"


@dataclass(frozen=True)
class PdfPathBackfillResult:
    """Counts from one dry-run or transactional metadata update."""

    candidates: int
    skipped_incomplete_metadata: int
    updated: int


def build_canonical_pdf_path(
    path_root: str,
    *,
    venue: str,
    year: int,
    paper_id: str,
) -> str:
    """Return the historical Windows path format used by existing SQL rows."""
    root = path_root.rstrip("\\/")
    return "\\".join(
        (
            root,
            venue.lower(),
            str(year),
            f"{sanitize_file_component(paper_id)}.pdf",
        )
    )


def backfill_missing_pdf_paths(
    connection: sqlite3.Connection,
    *,
    path_root: str,
    apply: bool,
) -> PdfPathBackfillResult:
    """Plan or transactionally fill only blank ``papers.pdf_path`` values.

    The guarded update repeats the blank-path predicate so a concurrent writer
    cannot have its newer path overwritten between the select and update.
    """
    rows = connection.execute(
        f"""
        SELECT paper_id, venue, year
        FROM papers
        WHERE {_MISSING_PATH_SQL}
        ORDER BY paper_id
        """
    ).fetchall()

    updates: list[tuple[str, str]] = []
    skipped_incomplete_metadata = 0
    for paper_id, venue, year in rows:
        if not paper_id or not venue or year is None:
            skipped_incomplete_metadata += 1
            continue
        updates.append(
            (
                build_canonical_pdf_path(
                    path_root,
                    venue=str(venue),
                    year=int(year),
                    paper_id=str(paper_id),
                ),
                str(paper_id),
            )
        )

    updated = 0
    if apply and updates:
        with connection:
            cursor = connection.executemany(
                f"""
                UPDATE papers
                SET pdf_path = ?
                WHERE paper_id = ? AND {_MISSING_PATH_SQL}
                """,
                updates,
            )
            updated = cursor.rowcount

    return PdfPathBackfillResult(
        candidates=len(updates),
        skipped_incomplete_metadata=skipped_incomplete_metadata,
        updated=updated,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DATABASE_PATH),
        help="SQLite database containing the papers table.",
    )
    parser.add_argument(
        "--path-root",
        default=DEFAULT_WINDOWS_PDF_ROOT,
        help="Canonical PDF root to record in existing Windows-path format.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the transaction. The default is a read-only dry-run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database_path = Path(args.database)
    if not database_path.is_file():
        print(f"SQLite database not found: {database_path}", file=sys.stderr)
        return 2

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        result = backfill_missing_pdf_paths(
            connection,
            path_root=args.path_root,
            apply=args.execute,
        )
    finally:
        connection.close()

    mode = "APPLIED" if args.execute else "DRY-RUN"
    print(f"mode: {mode}")
    print(f"candidate missing pdf_path rows: {result.candidates}")
    print(f"skipped incomplete paper metadata: {result.skipped_incomplete_metadata}")
    print(f"updated rows: {result.updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
