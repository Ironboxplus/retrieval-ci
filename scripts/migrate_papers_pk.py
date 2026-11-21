"""方案 X 迁移：papers 表砸了重来。

变更：
  - 删除 papers.id (int autoincrement) 列
  - paper_id (VARCHAR 100) 升级为 PRIMARY KEY
  - doc_id (INTEGER) 改 NOT NULL + UNIQUE + FK→documents.id ON DELETE CASCADE
  - 业务列、索引、行数全保留

安全栏：
  - 前置：retrieval.db → retrieval.db.bak.{ts}（hardlink 失败回落 copy）
  - 事务包裹（BEGIN IMMEDIATE … COMMIT）
  - 失败 ROLLBACK + 保留 bak
  - --dry-run 仅验证、不写入
  - --verify 单独跑结构断言

执行前 stop uvicorn 避免 db lock (W2 坑)。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "db" / "retrieval.db"


BUSINESS_COLUMNS = [
    "doc_id", "arxiv_id", "doi", "title", "abstract", "authors_json",
    "year", "venue", "paper_source", "pdf_path", "txt_path", "citation_count",
    "track_type", "impact_level", "venue_tier", "venue_source", "paper_id",
]


def _backup(db_path: Path) -> Path:
    ts = int(time.time())
    bak = db_path.with_suffix(f".db.bak.{ts}")
    print(f"[backup] {db_path} -> {bak}", flush=True)
    shutil.copy2(db_path, bak)
    size = bak.stat().st_size / (1024 ** 3)
    print(f"[backup] OK, {size:.2f} GB", flush=True)
    return bak


def _precheck(conn: sqlite3.Connection) -> dict:
    info = {}
    info["papers_count"] = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    info["paper_id_null"] = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE paper_id IS NULL OR paper_id = ''"
    ).fetchone()[0]
    info["paper_id_dupes"] = conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT paper_id) FROM papers WHERE paper_id IS NOT NULL"
    ).fetchone()[0]
    info["doc_id_null"] = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE doc_id IS NULL"
    ).fetchone()[0]
    info["doc_id_dupes"] = conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT doc_id) FROM papers WHERE doc_id IS NOT NULL"
    ).fetchone()[0]
    info["orphans"] = conn.execute(
        "SELECT COUNT(*) FROM papers p LEFT JOIN documents d ON p.doc_id = d.id WHERE d.id IS NULL"
    ).fetchone()[0]
    info["existing_columns"] = [r[1] for r in conn.execute("PRAGMA table_info(papers)")]
    return info


def _print_precheck(info: dict) -> bool:
    print("[precheck]", flush=True)
    for k, v in info.items():
        print(f"  {k:<20} = {v}", flush=True)
    failures = []
    if info["paper_id_null"]:
        failures.append(f"paper_id NULL/empty rows: {info['paper_id_null']}")
    if info["paper_id_dupes"]:
        failures.append(f"paper_id duplicates: {info['paper_id_dupes']}")
    if info["doc_id_null"]:
        failures.append(f"doc_id NULL rows: {info['doc_id_null']}")
    if info["doc_id_dupes"]:
        failures.append(f"doc_id duplicates: {info['doc_id_dupes']}")
    if info["orphans"]:
        failures.append(f"orphan doc_id (no matching documents.id): {info['orphans']}")
    if failures:
        print("[precheck] FAIL:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return False
    print("[precheck] OK", flush=True)
    return True


def _verify_structure(conn: sqlite3.Connection, expected_count: int) -> bool:
    cols = {r[1]: r for r in conn.execute("PRAGMA table_info(papers)")}
    ok = True

    if "id" in cols:
        print("[verify] FAIL: papers.id column still present", flush=True)
        ok = False

    if "paper_id" not in cols:
        print("[verify] FAIL: paper_id missing", flush=True)
        ok = False
    else:
        pk = cols["paper_id"][5]
        notnull = cols["paper_id"][3]
        if pk != 1:
            print(f"[verify] FAIL: paper_id pk flag={pk} (want 1)", flush=True)
            ok = False
        if notnull != 1:
            print(f"[verify] FAIL: paper_id notnull={notnull} (want 1)", flush=True)
            ok = False

    if "doc_id" not in cols:
        print("[verify] FAIL: doc_id missing", flush=True)
        ok = False
    else:
        notnull = cols["doc_id"][3]
        if notnull != 1:
            print(f"[verify] FAIL: doc_id notnull={notnull} (want 1)", flush=True)
            ok = False

    fks = list(conn.execute("PRAGMA foreign_key_list(papers)"))
    doc_id_fk = [fk for fk in fks if fk[3] == "doc_id"]
    if not doc_id_fk:
        print("[verify] FAIL: doc_id has no FK", flush=True)
        ok = False
    elif doc_id_fk[0][2] != "documents" or doc_id_fk[0][4] != "id":
        print(f"[verify] FAIL: doc_id FK target wrong: {doc_id_fk[0]}", flush=True)
        ok = False

    # doc_id UNIQUE check
    has_unique = False
    for idx in conn.execute("PRAGMA index_list(papers)"):
        if idx[2] == 1:
            ic = [r[2] for r in conn.execute(f"PRAGMA index_info({idx[1]})")]
            if ic == ["doc_id"]:
                has_unique = True
                break
    if not has_unique:
        print("[verify] FAIL: doc_id has no UNIQUE index", flush=True)
        ok = False

    actual_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    if actual_count != expected_count:
        print(f"[verify] FAIL: row count {actual_count} != expected {expected_count}", flush=True)
        ok = False

    if ok:
        print(f"[verify] OK: papers={actual_count}, paper_id PK, doc_id NOT NULL UNIQUE FK", flush=True)
    return ok


def _migrate(conn: sqlite3.Connection, expected_count: int) -> None:
    print("[migrate] BEGIN IMMEDIATE", flush=True)
    conn.execute("BEGIN IMMEDIATE")

    # PRAGMA foreign_keys=OFF inside transaction is not effective for SQLite;
    # set it outside. We set it on the connection in main().
    print("[migrate] CREATE papers_new", flush=True)
    conn.execute("""
        CREATE TABLE papers_new (
            paper_id     VARCHAR(100) PRIMARY KEY NOT NULL,
            doc_id       INTEGER NOT NULL UNIQUE,
            arxiv_id     VARCHAR(50),
            doi          VARCHAR(100),
            title        VARCHAR(1000) NOT NULL,
            abstract     TEXT,
            authors_json JSON,
            year         INTEGER,
            venue        VARCHAR(200),
            paper_source VARCHAR(50),
            pdf_path     VARCHAR(1000),
            txt_path     VARCHAR(1000),
            citation_count INTEGER DEFAULT 0,
            track_type   VARCHAR(50),
            impact_level VARCHAR(50),
            venue_tier   VARCHAR(10),
            venue_source VARCHAR(100),
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    """)

    print("[migrate] INSERT papers_new SELECT ...", flush=True)
    conn.execute("""
        INSERT INTO papers_new (
            paper_id, doc_id, arxiv_id, doi, title, abstract, authors_json,
            year, venue, paper_source, pdf_path, txt_path, citation_count,
            track_type, impact_level, venue_tier, venue_source
        )
        SELECT
            paper_id, doc_id, arxiv_id, doi, title, abstract, authors_json,
            year, venue, paper_source, pdf_path, txt_path, citation_count,
            track_type, impact_level, venue_tier, venue_source
        FROM papers
    """)

    inserted = conn.execute("SELECT COUNT(*) FROM papers_new").fetchone()[0]
    print(f"[migrate] inserted {inserted} rows (expected {expected_count})", flush=True)
    if inserted != expected_count:
        raise RuntimeError(f"row count mismatch: inserted {inserted}, expected {expected_count}")

    print("[migrate] DROP papers", flush=True)
    conn.execute("DROP TABLE papers")

    print("[migrate] RENAME papers_new -> papers", flush=True)
    conn.execute("ALTER TABLE papers_new RENAME TO papers")

    # Recreate non-unique business indexes
    print("[migrate] CREATE business indexes", flush=True)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_papers_arxiv_id ON papers(arxiv_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_papers_doi ON papers(doi)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_papers_year ON papers(year)")

    print("[migrate] COMMIT", flush=True)
    conn.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="precheck + verify only, no writes")
    p.add_argument("--verify-only", action="store_true", help="check current schema; no migration")
    p.add_argument("--skip-backup", action="store_true", help="dangerous: skip .bak file")
    args = p.parse_args()

    if not DB_PATH.exists():
        print(f"[fatal] DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(2)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        if args.verify_only:
            cnt = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            ok = _verify_structure(conn, cnt)
            sys.exit(0 if ok else 1)

        info = _precheck(conn)
        if not _print_precheck(info):
            sys.exit(1)
        expected_count = info["papers_count"]

        if args.dry_run:
            print("[dry-run] precheck passed; no writes performed", flush=True)
            sys.exit(0)

        conn.close()
        if not args.skip_backup:
            _backup(DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = OFF")

        _migrate(conn, expected_count)

        conn.execute("PRAGMA foreign_keys = ON")
        ok = _verify_structure(conn, expected_count)
        if not ok:
            print("[fatal] post-migration verification failed", file=sys.stderr)
            sys.exit(3)

        # FK integrity check
        violations = list(conn.execute("PRAGMA foreign_key_check"))
        if violations:
            print(f"[fatal] FK violations after migration: {violations}", file=sys.stderr)
            sys.exit(4)
        print("[done] migration complete, FK integrity OK", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
