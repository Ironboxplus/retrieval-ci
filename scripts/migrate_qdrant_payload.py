"""Migrate Qdrant `papers` collection payload to canonical schema.

Legacy -> canonical:
  conference (string)         -> venue (string)
  author_names (pipe-sep str) -> authors (list[str])

Default: DRY-RUN. Pass --execute to apply.

Implementation: all chunks of a paper share the same venue/authors, so we
group by doc_id, build one patch per unique doc_id, then issue
set_payload + delete_payload using a Qdrant filter selector keyed on doc_id.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Bypass HTTP_PROXY which routes localhost:6333 through 127.0.0.1:10808.
for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


def canonicalize_patch(payload: dict) -> tuple[dict, list[str]]:
    """Compute (fields_to_set, keys_to_delete) so payload becomes canonical."""
    add: dict = {}
    drop: list[str] = []

    conf = payload.get("conference")
    if conf:
        if not payload.get("venue"):
            add["venue"] = conf
        drop.append("conference")

    author_names = payload.get("author_names")
    if author_names is not None:
        existing = payload.get("authors")
        if not (isinstance(existing, list) and existing):
            if isinstance(author_names, str):
                authors_list = [a.strip() for a in author_names.split("|") if a.strip()]
            elif isinstance(author_names, list):
                authors_list = [a for a in author_names if a]
            else:
                authors_list = []
            if authors_list:
                add["authors"] = authors_list
        drop.append("author_names")

    return add, drop


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--collection", default="papers")
    ap.add_argument("--scan-batch", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N points (0=all)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually call set_payload/delete_payload. Default is dry-run.")
    ap.add_argument("--apply-batch-print", type=int, default=2000,
                    help="Log progress every N applied docs in execute mode")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    mode = "EXECUTE (WRITE)" if args.execute else "DRY-RUN (read-only)"
    print(f"[{mode}] qdrant={args.qdrant} collection={args.collection}", flush=True)

    qc = QdrantClient(url=args.qdrant, timeout=120)
    total = qc.count(args.collection, exact=True).count
    print(f"[load] total points: {total:,}", flush=True)

    doc_to_plan: dict[int, tuple[dict, list[str]]] = {}
    scanned = 0
    skipped_missing_doc_id = 0
    t0 = time.time()
    offset = None

    while True:
        page, offset = qc.scroll(
            args.collection,
            limit=args.scan_batch,
            offset=offset,
            with_payload=["doc_id", "venue", "conference", "authors", "author_names"],
            with_vectors=False,
        )
        if not page:
            break

        for p in page:
            scanned += 1
            payload = p.payload or {}
            doc_id_raw = payload.get("doc_id")
            if doc_id_raw is None:
                skipped_missing_doc_id += 1
                continue
            try:
                doc_id = int(doc_id_raw)
            except (ValueError, TypeError):
                skipped_missing_doc_id += 1
                continue

            if doc_id in doc_to_plan:
                continue  # chunks of same paper share venue/authors; one sample suffices

            add, drop = canonicalize_patch(payload)
            if add or drop:
                doc_to_plan[doc_id] = (add, drop)

        if scanned % 50000 == 0 or scanned >= total:
            elapsed = time.time() - t0
            rate = scanned / elapsed if elapsed > 0 else 0
            print(f"  scanned={scanned:,}/{total:,} plans={len(doc_to_plan):,} "
                  f"rate={rate:,.0f}pt/s elapsed={elapsed:.0f}s", flush=True)

        if offset is None:
            break
        if args.limit and scanned >= args.limit:
            print(f"[stop] hit --limit {args.limit}", flush=True)
            break

    elapsed_scan = time.time() - t0
    print()
    print("=" * 60)
    print(f"[scan complete] in {elapsed_scan:.0f}s")
    print(f"  scanned points : {scanned:,}")
    print(f"  unique docs needing migration: {len(doc_to_plan):,}")
    print(f"  skipped (no doc_id): {skipped_missing_doc_id:,}")

    if not doc_to_plan:
        print("  -> No migration needed.")
        return

    venue_count = sum(1 for add, _ in doc_to_plan.values() if "venue" in add)
    authors_count = sum(1 for add, _ in doc_to_plan.values() if "authors" in add)
    drop_conf = sum(1 for _, drop in doc_to_plan.values() if "conference" in drop)
    drop_an = sum(1 for _, drop in doc_to_plan.values() if "author_names" in drop)
    print(f"  docs setting venue       : {venue_count:,}")
    print(f"  docs setting authors     : {authors_count:,}")
    print(f"  docs dropping conference : {drop_conf:,}")
    print(f"  docs dropping author_names: {drop_an:,}")

    sample = list(doc_to_plan.items())[:3]
    print(f"  sample plans (3): {sample}")

    if not args.execute:
        print()
        print("DRY-RUN: nothing modified. Re-run with --execute to apply.")
        return

    # Apply: per-doc_id selector, batch via set_payload + delete_payload
    print()
    print(f"[execute] applying {len(doc_to_plan):,} per-doc patches...", flush=True)
    t1 = time.time()
    applied = 0
    failed = 0

    for doc_id, (add, drop) in doc_to_plan.items():
        # Qdrant payload stores doc_id as STRING (e.g. "42605"). MatchValue with
        # int 42605 won't match any point — silently apply to zero. Must pass str.
        selector = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id)))])
        try:
            if add:
                qc.set_payload(args.collection, payload=add, points=selector, wait=False)
            if drop:
                qc.delete_payload(args.collection, keys=drop, points=selector, wait=False)
            applied += 1
        except Exception as e:
            failed += 1
            if failed < 10:
                print(f"  fail doc_id={doc_id}: {type(e).__name__}: {e}", flush=True)

        if applied > 0 and applied % args.apply_batch_print == 0:
            elapsed = time.time() - t1
            rate = applied / elapsed if elapsed > 0 else 0
            print(f"  applied={applied:,}/{len(doc_to_plan):,} failed={failed:,} "
                  f"rate={rate:.1f}docs/s elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t1
    print()
    print("=" * 60)
    print(f"[execute complete] in {elapsed:.0f}s")
    print(f"  applied: {applied:,}")
    print(f"  failed : {failed:,}")
    post = qc.count(args.collection, exact=True).count
    print(f"  Qdrant points after: {post:,}")


if __name__ == "__main__":
    main()
