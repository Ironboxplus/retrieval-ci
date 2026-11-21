"""Preindex harvested OpenReview reviews into SQL (documents + chunks).

This is the ONLY new stage in the review-ingest chain — everything downstream is
reused verbatim:

    harvest_openreview.py  (Arc, 10808)  ->  *.jsonl
    preindex_reviews.py    (THIS)        ->  SQL documents(source_type=review)+chunks
    dump_chunks.py --min-chunk-id N      ->  chunks_*.jsonl.gz   (incremental)
    embed_worker.py                      ->  vectors_*.parquet   (BGE-M3, GPU)
    local_ingest.py                      ->  Qdrant (papers collection)

论文关联 (paper association):
  * accepted papers in corpus  -> review chunks carry the EXISTING corpus paper_id
    (CorpusMatcher: ICLR forum-id / NeurIPS title).  Reviews group under the real
    paper and can deep-link to it.
  * rejected / withdrawn papers -> NOT created as papers.  The review Document
    keeps only title + OpenReview pdf_url (per user decision), keyed by the
    openreview paper_id.  Never enters the papers table / paper counts / normal
    paper search; reachable only via source_type=review.

Payload fields (paper_id / venue / year / venue_tier) ride on each chunk's
metadata_json so dump_chunks.py lifts them into the Qdrant payload — review docs
have no papers-table row to LEFT JOIN.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.reviews.review_ingest import CorpusMatcher, build_review_chunks  # noqa: E402
from scripts.reviews.review_venues import VENUE_TIER  # noqa: E402  (name.upper() -> CCF tier)

DB_PATH = "data/db/retrieval.db"


def _content_hash(record: dict) -> str:
    """Stable idempotency key: one review Document per submission."""
    return f"review-{record.get('venue')}-{record.get('year')}-{record['openreview_id']}"


def load_corpus_matcher(conn: sqlite3.Connection, venue_years) -> CorpusMatcher:
    """Build a CorpusMatcher from the papers table, restricted to the
    (venue, year) pairs present in the harvest so we never scan all 94k rows."""
    rows = []
    for venue, year in sorted(venue_years):
        cur = conn.execute(
            "SELECT paper_id, title, venue, year FROM papers WHERE upper(venue)=? AND year=?",
            (venue.upper(), int(year)),
        )
        rows.extend({"paper_id": r[0], "title": r[1], "venue": r[2], "year": r[3]}
                    for r in cur)
    return CorpusMatcher(rows)


def preindex_records(conn: sqlite3.Connection, records: list[dict],
                     matcher: CorpusMatcher) -> dict:
    """Write review Documents + Chunks for records that have reviews.

    Idempotent: a record whose content_hash already has a Document is skipped.
    Returns stats incl. the min/max new chunk id (for dump_chunks --min-chunk-id).
    """
    stats = {"docs": 0, "chunks": 0, "matched": 0, "rejected": 0,
             "skipped_existing": 0, "skipped_no_chunks": 0,
             "min_chunk_id": None, "max_chunk_id": None}

    for rec in records:
        if rec.get("n_reviews", 0) <= 0:
            continue
        chash = _content_hash(rec)
        if conn.execute("SELECT 1 FROM documents WHERE content_hash=?", (chash,)).fetchone():
            stats["skipped_existing"] += 1
            continue

        venue = (rec.get("venue") or "").upper()
        corpus_pid = matcher.resolve(venue, rec["year"], rec["openreview_id"], rec.get("title"))
        in_corpus = corpus_pid is not None
        paper_id = corpus_pid or rec["paper_id"]
        tier = VENUE_TIER.get(venue)

        chunks = build_review_chunks(rec, paper_id, venue_tier=tier)
        if not chunks:
            stats["skipped_no_chunks"] += 1
            continue

        doc_meta = {
            "openreview_id": rec["openreview_id"],
            "paper_id": paper_id,
            "in_corpus": in_corpus,
            "venue": rec.get("venue"),
            "year": rec.get("year"),
            "venue_tier": tier,
            "accepted": rec.get("accepted"),
            "decision": rec.get("decision"),
            "avg_rating": rec.get("avg_rating"),
            "ratings": rec.get("ratings"),
            "n_reviews": rec.get("n_reviews"),
            "pdf_url": rec.get("pdf_url"),
            "forum_url": rec.get("forum_url"),
            "abstract": rec.get("abstract"),
            "reviews": rec.get("reviews"),
            "meta_review": rec.get("meta_review"),
        }
        cur = conn.execute(
            "INSERT INTO documents (title, source, source_type, data_source, "
            "content_hash, url, status, metadata_json) VALUES (?,?,?,?,?,?,?,?)",
            (rec.get("title"), "openreview", "review", "system", chash,
             rec.get("forum_url"), "indexed", json.dumps(doc_meta, ensure_ascii=False)),
        )
        doc_id = cur.lastrowid
        stats["docs"] += 1
        stats["matched" if in_corpus else "rejected"] += 1

        for ci, ch in enumerate(chunks):
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, content, level, chunk_index, "
                "start_idx, end_idx, metadata_json) VALUES (?,?,?,?,?,?,?)",
                (doc_id, ch["content"], ch["level"], ci, ch["start_idx"],
                 ch["start_idx"], json.dumps(ch["meta"], ensure_ascii=False)),
            )
            cid = cur.lastrowid
            stats["chunks"] += 1
            if stats["min_chunk_id"] is None or cid < stats["min_chunk_id"]:
                stats["min_chunk_id"] = cid
            if stats["max_chunk_id"] is None or cid > stats["max_chunk_id"]:
                stats["max_chunk_id"] = cid

    conn.commit()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="harvested *.jsonl")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    records: list[dict] = []
    for p in args.input:
        records += [json.loads(l) for l in Path(p).open(encoding="utf-8")]
    venue_years = {((r.get("venue") or "").upper(), r["year"])
                   for r in records if r.get("n_reviews", 0) > 0}
    print(f"loaded {len(records)} records; venue_years={sorted(venue_years)}", flush=True)

    conn = sqlite3.connect(args.db)
    try:
        matcher = load_corpus_matcher(conn, venue_years)
        stats = preindex_records(conn, records, matcher)
    finally:
        conn.close()

    print("DONE preindex_reviews:", json.dumps(stats, ensure_ascii=False), flush=True)
    if stats["min_chunk_id"] is not None:
        print(f"  -> next: dump_chunks.py --min-chunk-id {stats['min_chunk_id']}", flush=True)


if __name__ == "__main__":
    main()
