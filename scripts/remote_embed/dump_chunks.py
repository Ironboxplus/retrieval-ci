"""Dump SQLite chunks → jsonl.gz shards for remote BGE-M3 embed.

Per-line schema (JSON):
    {
        "chunk_id_str": "{doc_id}_L{level}_{start_idx}_{sha16}",
        "point_id":     int (sha256 of chunk_id_str, first 15 hex chars),
        "doc_id":       int,
        "content":      str,
        "level":        int,
        "start_idx":    int,
        "end_idx":      int,
        "metadata":     dict (title, source, source_type, paper_id, venue, year, ...)
    }

Output:
    data/remote_embed/chunks/chunks_NNNN.jsonl.gz   (one shard = 100_000 chunks)
    data/remote_embed/chunks/done.txt               (one shard path per line on success)

Resume: shards listed in done.txt are skipped on re-run.

Which chunks are dumped:
    (default)              every chunk (legacy row-skip resume from done.txt)
    --min-chunk-id/--max-chunk-id   a chunk-id range
    --doc-ids-file PATH    the chunks of the listed documents only (one doc id per line):
                           a pipeline task's own documents, whatever other writers added
                           to the chunks table meanwhile
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

DB_PATH = "data/db/retrieval.db"
OUT_DIR = Path("data/remote_embed/chunks")
SHARD_SIZE = 100_000


def chunk_id_str(doc_id: int, content: str, level: int, start_idx: int) -> str:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}_L{level}_{start_idx}_{content_hash}"


def chunk_id_int(chunk_id_str_value: str) -> int:
    return int(hashlib.sha256(chunk_id_str_value.encode()).hexdigest()[:15], 16)


def iter_chunks(conn: sqlite3.Connection, min_chunk_id: int | None = None,
                max_chunk_id: int | None = None, only_want_docs: bool = False):
    """Chunks in id order. ``only_want_docs`` keeps the chunks of the documents in the
    ``want_docs`` temp table of ``conn`` (see ``load_want_docs``)."""
    bounds, params = [], []
    if only_want_docs:
        bounds.append("c.doc_id IN (SELECT doc_id FROM want_docs)")
    if min_chunk_id is not None:
        bounds.append("c.id >= ?")
        params.append(min_chunk_id)
    if max_chunk_id is not None:
        bounds.append("c.id <= ?")
        params.append(max_chunk_id)
    where = ("WHERE " + " AND ".join(bounds)) if bounds else ""
    # papers.full_text_source arrives with the backend's init_db migration; a
    # database it has not touched yet dumps as before (no preprint flag).
    paper_columns = {row[1] for row in conn.execute("PRAGMA table_info(papers)")}
    full_text_source = "p.full_text_source" if "full_text_source" in paper_columns else "NULL"
    sql = f"""
    SELECT
        c.id, c.doc_id, c.content, c.level, c.start_idx, c.end_idx,
        c.metadata_json,
        d.title, d.source, d.source_type, d.metadata_json AS doc_meta,
        p.paper_id AS paper_pk, p.venue, p.year, p.venue_tier, p.track_type,
        p.arxiv_id, p.doi, p.citation_count, p.withdrawn, {full_text_source} AS full_text_source
    FROM chunks c
    JOIN documents d ON d.id = c.doc_id
    LEFT JOIN papers p ON p.doc_id = d.id
    {where}
    ORDER BY c.id
    """
    yield from conn.execute(sql, params)


def load_want_docs(conn: sqlite3.Connection, ids_file: Path) -> int:
    """Load the doc ids of ``ids_file`` (one integer per line, blank lines ignored) into the
    ``want_docs`` temp table; returns how many distinct documents it lists."""
    ids = [int(line) for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    conn.execute("CREATE TEMP TABLE want_docs(doc_id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT OR IGNORE INTO want_docs(doc_id) VALUES (?)", [(i,) for i in ids])
    return conn.execute("SELECT COUNT(*) FROM want_docs").fetchone()[0]


def build_payload(row: sqlite3.Row | tuple) -> dict:
    (
        chunk_pk, doc_id, content, level, start_idx, end_idx, chunk_meta_json,
        title, source, source_type, doc_meta_json,
        paper_pk, venue, year, venue_tier, track_type, arxiv_id, doi, citation_count, withdrawn,
        full_text_source,
    ) = row

    chunk_meta = json.loads(chunk_meta_json) if chunk_meta_json else {}
    doc_meta = json.loads(doc_meta_json) if doc_meta_json else {}
    paper_id = doc_meta.get("paper_id") or chunk_meta.get("paper_id")

    payload = {
        "title": title or chunk_meta.get("title", ""),
        "source": source or chunk_meta.get("source", ""),
        "source_type": source_type,
    }
    if paper_id:
        payload["paper_id"] = paper_id
    if venue:
        payload["venue"] = venue
    if year is not None:
        payload["year"] = year
    if venue_tier:
        payload["venue_tier"] = venue_tier
    if track_type:
        payload["track_type"] = track_type
    if arxiv_id:
        payload["arxiv_id"] = arxiv_id
    if doi:
        payload["doi"] = doi
    if citation_count is not None:
        payload["citation_count"] = citation_count
    if withdrawn:
        # Optional payload key: present only as True (papers_payload.OPTIONAL_PAPER_KEYS).
        payload["withdrawn"] = True
    if full_text_source in ("arxiv", "oa_repo"):
        # Optional payload key, True only: the text is not the publisher's version.
        payload["preprint_full_text"] = True
    payload.update({k: v for k, v in chunk_meta.items() if k not in payload})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--min-chunk-id", type=int, default=None,
                        help="Cursor-based incremental: dump only chunks with c.id >= this value")
    parser.add_argument("--max-chunk-id", type=int, default=None,
                        help="Upper bound (inclusive) with --min-chunk-id: one pipeline task's "
                             "own chunks, so concurrent tasks never dump each other's")
    parser.add_argument("--start-shard", type=int, default=None,
                        help="Override starting shard index (default: auto-resume from done.txt)")
    parser.add_argument("--doc-ids-file", type=Path, default=None,
                        help="Dump only the chunks of the documents listed in this file (one doc id "
                             "per line). Exclusive with --min-chunk-id/--max-chunk-id")
    args = parser.parse_args()
    if args.doc_ids_file is not None and (args.min_chunk_id is not None or args.max_chunk_id is not None):
        parser.error("--doc-ids-file cannot be combined with --min-chunk-id/--max-chunk-id")
    by_docs = args.doc_ids_file is not None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()

    conn = sqlite3.connect(args.db)
    if by_docs:
        n_docs = load_want_docs(conn, args.doc_ids_file)
        total = conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id IN (SELECT doc_id FROM want_docs)"
                             ).fetchone()[0]
        print(f"task dump: chunks of {n_docs:,} documents \u2192 {total:,} rows"
              f"  shard_size = {args.shard_size:,}")
    elif args.min_chunk_id is not None:
        upper = args.max_chunk_id if args.max_chunk_id is not None else 2 ** 62
        total = conn.execute("SELECT COUNT(*) FROM chunks WHERE id >= ? AND id <= ?",
                             (args.min_chunk_id, upper)).fetchone()[0]
        print(f"incremental dump: chunks WHERE id in [{args.min_chunk_id:,}, {upper:,}] → {total:,} rows"
              f"  shard_size = {args.shard_size:,}")
    else:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"chunks total = {total:,}  shard_size = {args.shard_size:,}  shards = {total // args.shard_size + 1}")

    # Resume: find first shard index whose file is NOT in done.txt.
    # CRITICAL: never call gzip.open(..., 'wt') on a shard before knowing we will
    # actually write to it — 'wt' truncates the file, which would wipe already-done shards.
    if args.start_shard is not None:
        shard_idx = args.start_shard
        print(f"start-shard override: starting at shard {shard_idx}")
    else:
        shard_idx = 0
        while f"chunks_{shard_idx:04d}.jsonl.gz" in done:
            shard_idx += 1
        if shard_idx > 0:
            print(f"resuming at shard {shard_idx} ({shard_idx} done shards skipped)")

    # When using --min-chunk-id or --doc-ids-file, SQL already filters; otherwise apply legacy
    # row-skip for backward compat (each done shard = shard_size rows from the head of SQL stream).
    sql_filtered = args.min_chunk_id is not None or by_docs
    skip = 0 if sql_filtered else shard_idx * args.shard_size
    if not sql_filtered and skip > 0:
        print(f"  legacy row-skip: skipping first {skip:,} SQL rows to match done shards")

    in_shard, written = 0, 0
    shard_name = f"chunks_{shard_idx:04d}.jsonl.gz"
    fh = gzip.open(out_dir / shard_name, "wt", encoding="utf-8")

    def open_shard(i: int):
        name = f"chunks_{i:04d}.jsonl.gz"
        return name, gzip.open(out_dir / name, "wt", encoding="utf-8")

    for i, row in enumerate(iter_chunks(conn, min_chunk_id=args.min_chunk_id,
                                        max_chunk_id=args.max_chunk_id, only_want_docs=by_docs)):
        if i < skip:
            continue
        chunk_pk, doc_id, content, level, start_idx, *_ = row
        cid_str = chunk_id_str(doc_id, content, level, start_idx)
        record = {
            "chunk_id_str": cid_str,
            "point_id": chunk_id_int(cid_str),
            "doc_id": doc_id,
            "content": content,
            "level": level,
            "start_idx": start_idx,
            "end_idx": row[5],
            "metadata": build_payload(row),
        }
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        in_shard += 1
        written += 1

        if in_shard >= args.shard_size:
            fh.close()
            with done_path.open("a") as f:
                f.write(shard_name + "\n")
            print(f"  shard {shard_name} closed ({in_shard:,} rows)  total written = {written:,}")
            shard_idx += 1
            in_shard = 0
            shard_name, fh = open_shard(shard_idx)

    fh.close()
    if in_shard > 0:
        with done_path.open("a") as f:
            f.write(shard_name + "\n")
        print(f"  shard {shard_name} closed ({in_shard:,} rows)  total written = {written:,}")
    else:
        # Empty trailing shard
        (out_dir / shard_name).unlink(missing_ok=True)

    conn.close()
    print(f"DONE  total = {written:,}")


if __name__ == "__main__":
    main()
