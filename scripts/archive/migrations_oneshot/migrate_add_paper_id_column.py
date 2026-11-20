"""Add papers.paper_id column + UNIQUE INDEX, backfill from documents.metadata_json.

Architecture: Paper.paper_id is the single source of truth for the business
paper_id (e.g. "aaai-2022-20281"). Today the value lives in documents.metadata_json
(JSON blob, unindexed, no UNIQUE guarantee). This migration:

  1. ALTER TABLE papers ADD COLUMN paper_id VARCHAR(100)
  2. UPDATE papers SET paper_id = json_extract(documents.metadata_json, '$.paper_id')
  3. CREATE UNIQUE INDEX ix_papers_paper_id  (if no duplicates)

Idempotent: safe to re-run. Dry-run prints stats without writing.

Usage:
    python scripts/migrate_add_paper_id_column.py [--db data/db/retrieval.db] [--execute]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return any(row[1] == col for row in conn.execute(f"PRAGMA table_info({table})"))


def _has_index(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (name,)
    ).fetchone() is not None


def run(db_path: str, execute: bool) -> None:
    db = Path(db_path)
    if not db.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Step 1: column
        if _has_column(conn, "papers", "paper_id"):
            print("[skip] papers.paper_id already exists")
        elif execute:
            conn.execute("ALTER TABLE papers ADD COLUMN paper_id VARCHAR(100)")
            print("[done] ALTER TABLE papers ADD COLUMN paper_id")
        else:
            print("[plan] ALTER TABLE papers ADD COLUMN paper_id VARCHAR(100)")

        # Step 2: backfill
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        with_doc = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE doc_id IS NOT NULL"
        ).fetchone()[0]
        print(f"[stats] papers total={total}, with doc_id={with_doc}")

        # Pull (paper.id, documents.metadata_json) pairs
        rows = conn.execute("""
            SELECT p.id, d.metadata_json
            FROM papers p
            LEFT JOIN documents d ON d.id = p.doc_id
            WHERE p.paper_id IS NULL OR p.paper_id = ''
        """).fetchall()

        updates: list[tuple[str, int]] = []
        missing = 0
        for paper_pk, md_json in rows:
            if not md_json:
                missing += 1
                continue
            try:
                md = json.loads(md_json)
            except (json.JSONDecodeError, TypeError):
                missing += 1
                continue
            pid = md.get("paper_id")
            if not pid:
                missing += 1
                continue
            updates.append((str(pid), paper_pk))

        print(f"[plan] {len(updates)} rows will be backfilled, {missing} have no paper_id in metadata_json")

        if execute and updates:
            conn.executemany("UPDATE papers SET paper_id = ? WHERE id = ?", updates)
            conn.commit()
            print(f"[done] backfilled {len(updates)} rows")

        # Step 3: detect duplicates BEFORE creating UNIQUE INDEX
        dup = conn.execute("""
            SELECT paper_id, COUNT(*) c FROM papers
            WHERE paper_id IS NOT NULL AND paper_id != ''
            GROUP BY paper_id HAVING c > 1
        """).fetchall()
        if dup:
            print(f"[warn] {len(dup)} duplicate paper_id values; cannot create UNIQUE INDEX:")
            for pid, c in dup[:5]:
                print(f"        {pid!r}: {c} occurrences")
            return

        # Step 3b: create UNIQUE INDEX
        if _has_index(conn, "ix_papers_paper_id"):
            print("[skip] ix_papers_paper_id already exists")
        elif execute:
            conn.execute("CREATE UNIQUE INDEX ix_papers_paper_id ON papers(paper_id)")
            conn.commit()
            print("[done] CREATE UNIQUE INDEX ix_papers_paper_id")
        else:
            print("[plan] CREATE UNIQUE INDEX ix_papers_paper_id ON papers(paper_id)")

        # Final report
        if execute:
            after = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE paper_id IS NOT NULL AND paper_id != ''"
            ).fetchone()[0]
            print(f"[final] paper_id coverage: {after}/{total} ({100*after/total:.1f}%)")
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/retrieval.db")
    ap.add_argument("--execute", action="store_true", help="actually write; default is dry-run")
    args = ap.parse_args()
    run(args.db, args.execute)
