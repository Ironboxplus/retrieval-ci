#!/usr/bin/env python
"""Reconcile an existing paper slice to the canonical L1+L2 chunk schema.

The operation is content-aware and idempotent. Matching chunks are retained,
missing canonical chunks are inserted, obsolete chunks are removed, and row
ordering/metadata are updated in place. A report records exact stale Qdrant
point IDs so vector deletion can happen only after new chunks are embedded and
upserted.

Examples:
    # Read-only audit.
    python scripts/reconcile_paper_chunk_schema.py --conf ICML --year 2026

    # Apply SQL reconciliation and write a durable report.
    python scripts/reconcile_paper_chunk_schema.py \
        --conf ICML --year 2026 --execute

    # After embedding/upserting newly inserted chunks, delete obsolete vectors.
    python scripts/reconcile_paper_chunk_schema.py \
        --delete-stale-vectors data/papers/ccf_a_main_conference_corpus/reports/paper_chunk_schema_reconcile.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.indexing.chunker import create_chunker
from core.indexing.paper_chunk_reconcile import (
    apply_document_plan_atomically,
    build_document_reconciliation,
    load_existing_chunks,
)
from core.storage.qdrant_operations import retry_qdrant_operation
from pipeline.paper_crawler.runtime_files import write_text_atomic


DEFAULT_DB = Path("data/db/retrieval.db")
DEFAULT_REPORT = Path(
    "data/papers/ccf_a_main_conference_corpus/reports/"
    "paper_chunk_schema_reconcile.json"
)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(report, ensure_ascii=False, indent=2))


def reconcile_slice(args: argparse.Namespace) -> dict[str, Any]:
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    rows = conn.execute(
        """
        SELECT p.paper_id, p.title, p.doc_id, d.file_path
        FROM papers p
        JOIN documents d ON d.id = p.doc_id
        WHERE LOWER(p.venue) = LOWER(?) AND p.year = ?
        ORDER BY p.doc_id
        """,
        (args.conf, args.year),
    ).fetchall()
    if args.limit > 0:
        rows = rows[:args.limit]

    chunker = create_chunker("section_l1")
    counters = {
        "papers": len(rows),
        "processed": 0,
        "failed": 0,
        "inserted": 0,
        "deleted": 0,
        "updated": 0,
        "retained": 0,
    }
    inserted_chunk_ids: list[int] = []
    stale_point_ids: set[int] = set()
    failures: list[dict[str, str]] = []
    started = time.time()

    for position, (paper_id, title, doc_id, file_path) in enumerate(rows, 1):
        try:
            if not file_path or not Path(file_path).exists():
                raise FileNotFoundError(file_path or "missing document file_path")
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
            expected = chunker.chunk_document(str(doc_id), text, title_hint=title)
            existing = load_existing_chunks(conn, doc_id)
            plan = build_document_reconciliation(doc_id, existing, expected)

            if args.execute:
                inserted_chunk_ids.extend(apply_document_plan_atomically(conn, doc_id, plan))

            counters["processed"] += 1
            counters["inserted"] += len(plan.inserts)
            counters["deleted"] += len(plan.delete_row_ids)
            counters["updated"] += sum(item.requires_update for item in plan.retained)
            counters["retained"] += len(plan.retained)
            stale_point_ids.update(plan.stale_point_ids)
        except Exception as exc:
            counters["failed"] += 1
            failures.append({"paper_id": paper_id, "error": f"{type(exc).__name__}: {exc}"})

        if args.execute and position % args.commit_batch == 0:
            conn.commit()
        if position % args.checkpoint_interval == 0 or position == len(rows):
            print(
                f"[progress] {position}/{len(rows)} inserted={counters['inserted']} "
                f"deleted={counters['deleted']} updated={counters['updated']} "
                f"failed={counters['failed']}",
                flush=True,
            )

    if args.execute:
        conn.commit()
    conn.close()

    report = {
        "conference": args.conf.upper(),
        "year": args.year,
        "executed": bool(args.execute),
        **counters,
        "inserted_chunk_ids": inserted_chunk_ids,
        "inserted_chunk_id_min": min(inserted_chunk_ids) if inserted_chunk_ids else None,
        "inserted_chunk_id_max": max(inserted_chunk_ids) if inserted_chunk_ids else None,
        "stale_point_ids": sorted(stale_point_ids),
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_report(args.report_path, report)
    return report


def _batches(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for offset in range(0, len(values), size):
        yield list(values[offset:offset + size])


def validate_stale_point_scope(
    payloads: Iterable[dict[str, Any]],
    *,
    conference: str,
    year: int,
) -> None:
    expected_venue = conference.upper()
    for payload in payloads:
        venue = str(payload.get("venue") or "").upper()
        point_year = payload.get("year")
        if venue != expected_venue or int(point_year or 0) != year:
            raise RuntimeError(
                "stale point report includes a point outside requested slice: "
                f"expected {expected_venue} {year}, got {venue or '<missing>'} {point_year}"
            )


def delete_stale_vectors(
    report_path: Path,
    *,
    qdrant_url: str,
    collection: str,
    batch_size: int,
) -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointIdsList

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("executed"):
        raise ValueError("refusing Qdrant deletion from an audit-only report")
    conference = str(report.get("conference") or "")
    year = int(report.get("year") or 0)
    if not conference or not year:
        raise ValueError("reconcile report is missing conference/year scope")
    point_ids = [int(value) for value in report.get("stale_point_ids", [])]
    client = QdrantClient(url=qdrant_url, timeout=180)

    existing = 0
    for batch in _batches(point_ids, batch_size):
        records = client.retrieve(
            collection_name=collection,
            ids=batch,
            with_payload=["venue", "year"],
            with_vectors=False,
        )
        validate_stale_point_scope(
            (record.payload or {} for record in records),
            conference=conference,
            year=year,
        )
        existing += len(records)
    print(
        f"[qdrant] preflight validated {existing}/{len(point_ids)} existing "
        f"stale points inside {conference.upper()} {year}",
        flush=True,
    )

    deleted = 0
    for batch in _batches(point_ids, batch_size):
        retry_qdrant_operation(
            f"delete(collection={collection} points={len(batch)})",
            lambda batch=batch: client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=batch),
                wait=True,
            ),
        )
        deleted += len(batch)
        print(f"[qdrant] deleted {deleted}/{len(point_ids)} stale points", flush=True)
    return deleted


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf")
    parser.add_argument("--year", type=int)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=0, help="0 means all papers in the slice")
    parser.add_argument("--commit-batch", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--execute", action="store_true", help="Apply SQL mutations; default is audit-only")
    parser.add_argument(
        "--delete-stale-vectors",
        type=Path,
        metavar="REPORT",
        help="Delete report.stale_point_ids after new vectors have been upserted",
    )
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="papers")
    parser.add_argument("--qdrant-batch", type=int, default=1000)
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if args.delete_stale_vectors:
        deleted = delete_stale_vectors(
            args.delete_stale_vectors,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            batch_size=args.qdrant_batch,
        )
        print(json.dumps({"deleted_stale_vectors": deleted}, indent=2))
        return 0
    if not args.conf or args.year is None:
        parser.error("--conf and --year are required for chunk reconciliation")

    report = reconcile_slice(args)
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in {"inserted_chunk_ids", "stale_point_ids", "failures"}
    }, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
