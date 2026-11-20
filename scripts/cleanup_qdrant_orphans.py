"""Scan Qdrant `papers` collection for orphan points (doc_id no longer in SQL).

Default: DRY-RUN. Reports orphan ratio + sample IDs. Writes nothing.
Pass --execute to actually delete orphan points (irreversible without backup).

Cause: historical reindex / INCIDENT 2026-05-15 22:50 deleted rows from
SQLite `documents` table without cascading the corresponding Qdrant points.
Sample-based survey shows ~48% of current 2.27M points are orphans.

Usage:
  .venv/Scripts/python.exe scripts/cleanup_qdrant_orphans.py            # dry run
  .venv/Scripts/python.exe scripts/cleanup_qdrant_orphans.py --execute  # really delete
  .venv/Scripts/python.exe scripts/cleanup_qdrant_orphans.py --batch 5000 --limit 100000
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Bypass HTTP_PROXY which routes localhost:6333 through 127.0.0.1:10808 → ReadTimeout
for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/retrieval.db")
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--collection", default="papers")
    ap.add_argument("--batch", type=int, default=2000, help="Qdrant scroll batch size")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N points (0=all)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually delete orphans. Default is dry-run (read-only).")
    ap.add_argument("--delete-batch", type=int, default=1000,
                    help="How many orphan point IDs to delete per request")
    return ap.parse_args()


def load_sql_doc_ids(db_path: str) -> set[int]:
    if not Path(db_path).exists():
        sys.exit(f"SQL DB not found: {db_path}")
    db = sqlite3.connect(db_path)
    try:
        rows = db.execute("SELECT id FROM documents").fetchall()
    finally:
        db.close()
    return {r[0] for r in rows}


def main() -> None:
    args = parse_args()
    mode = "EXECUTE (DESTRUCTIVE)" if args.execute else "DRY-RUN (read-only)"
    print(f"[{mode}] qdrant={args.qdrant} collection={args.collection} db={args.db}")

    sql_doc_ids = load_sql_doc_ids(args.db)
    print(f"[load] SQL documents.id rows: {len(sql_doc_ids):,}")

    qc = QdrantClient(url=args.qdrant, timeout=60)
    total = qc.count(args.collection, exact=True).count
    print(f"[load] Qdrant {args.collection} total points: {total:,}")

    scanned = 0
    orphan_count = 0
    live_count = 0
    missing_doc_id = 0
    orphan_buffer: list = []
    sample_orphan_ids: list = []
    sample_orphan_doc_ids: list[int] = []

    deleted_total = 0
    t0 = time.time()
    offset = None

    while True:
        page, offset = qc.scroll(
            args.collection,
            limit=args.batch,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        if not page:
            break

        for p in page:
            scanned += 1
            doc_id_raw = p.payload.get("doc_id") if p.payload else None
            if doc_id_raw is None:
                missing_doc_id += 1
                continue
            try:
                doc_id = int(doc_id_raw)
            except (ValueError, TypeError):
                missing_doc_id += 1
                continue
            if doc_id in sql_doc_ids:
                live_count += 1
                continue
            orphan_count += 1
            orphan_buffer.append(p.id)
            if len(sample_orphan_ids) < 10:
                sample_orphan_ids.append(p.id)
                sample_orphan_doc_ids.append(doc_id)

            if args.execute and len(orphan_buffer) >= args.delete_batch:
                qc.delete(args.collection, points_selector=PointIdsList(points=orphan_buffer), wait=True)
                deleted_total += len(orphan_buffer)
                orphan_buffer.clear()

        if scanned % 20000 == 0 or scanned >= total:
            elapsed = time.time() - t0
            rate = scanned / elapsed if elapsed > 0 else 0
            pct_orphan = 100 * orphan_count / scanned if scanned else 0
            print(f"  scanned={scanned:,}/{total:,} orphan={orphan_count:,} ({pct_orphan:.1f}%) "
                  f"deleted={deleted_total:,} rate={rate:,.0f}pt/s "
                  f"elapsed={elapsed:.0f}s")

        if offset is None:
            break
        if args.limit and scanned >= args.limit:
            print(f"[stop] hit --limit {args.limit}")
            break

    # Flush remaining orphans in execute mode
    if args.execute and orphan_buffer:
        qc.delete(args.collection, points_selector=PointIdsList(points=orphan_buffer), wait=True)
        deleted_total += len(orphan_buffer)
        orphan_buffer.clear()

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"[{mode}] complete in {elapsed:.0f}s")
    print(f"  scanned        : {scanned:,}")
    print(f"  live (in SQL)  : {live_count:,}  ({100*live_count/max(scanned,1):.1f}%)")
    print(f"  orphan         : {orphan_count:,}  ({100*orphan_count/max(scanned,1):.1f}%)")
    print(f"  missing doc_id : {missing_doc_id:,}")
    if args.execute:
        print(f"  DELETED        : {deleted_total:,}")
        post = qc.count(args.collection, exact=True).count
        print(f"  Qdrant after   : {post:,} (was {total:,}, diff={total-post:,})")
    else:
        print(f"  DRY-RUN: nothing deleted. Re-run with --execute to remove.")
        print(f"  sample orphan point IDs (10): {sample_orphan_ids}")
        print(f"  sample orphan doc_ids (10):   {sample_orphan_doc_ids}")


if __name__ == "__main__":
    main()
