#!/usr/bin/env python
"""Backfill papers.source_version / papers.withdrawn for arXiv papers indexed before
those columns existed, from the harvest database.

    # what would change (the default; writes nothing)
    python scripts/maintenance/backfill_arxiv_source_version.py --db data/db/retrieval.db
    # write it, then mark the withdrawn papers' Qdrant points too
    python scripts/maintenance/backfill_arxiv_source_version.py --db data/db/retrieval.db --apply --sync-qdrant

source_version = the version of the HTML that was downloaded and indexed (harvest
html_version; latest_version when that is empty). It is only ever filled where it
is NULL, so a re-run changes nothing and a value the pipeline already wrote is
never overwritten. withdrawn follows the harvest flag (newly withdrawn papers get
withdrawn_at = now). Only SQL is edited unless --sync-qdrant is given, which makes
every withdrawn paper's points carry the payload `withdrawn: true` in each paper
collection that exists (core/storage/qdrant_operations.set_paper_withdrawn;
idempotent, so it is also the way to repair drift).

The delta pipeline refuses to start while any arXiv paper has no source_version
(arxiv_stages.freeze_delta): run this first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SQLITE_MAX_IN = 900
NEW_STYLE_PAPER_ID = re.compile(r"^arxiv-\d{4}\.\d{4,5}$")


def plan_backfill(db: sqlite3.Connection, harvest: sqlite3.Connection) -> dict:
    """What to write: {"versions": {paper_id: version}, "withdrawn": {paper_id: flag}} plus counts."""
    papers = db.execute(
        "SELECT paper_id, arxiv_id, source_version, withdrawn FROM papers WHERE paper_id LIKE 'arxiv-%'"
    ).fetchall()
    by_arxiv_id = {}
    for pid, arxiv_id, version, withdrawn in papers:
        # New-style ids are recoverable from the paper id when the column is empty.
        arxiv_id = arxiv_id or (pid[len("arxiv-"):] if NEW_STYLE_PAPER_ID.match(pid) else None)
        if arxiv_id:
            by_arxiv_id[arxiv_id] = (pid, version, bool(withdrawn))
    versions: dict[str, str] = {}
    withdrawn: dict[str, bool] = {}
    ids = sorted(by_arxiv_id)
    seen = 0
    for start in range(0, len(ids), SQLITE_MAX_IN):
        chunk = ids[start:start + SQLITE_MAX_IN]
        for arxiv_id, html_version, latest_version, flag in harvest.execute(
                "SELECT arxiv_id, html_version, latest_version, withdrawn FROM papers "
                f"WHERE arxiv_id IN ({','.join('?' * len(chunk))})", chunk):
            seen += 1
            pid, current_version, current_flag = by_arxiv_id[arxiv_id]
            new_version = html_version or latest_version
            if current_version is None and new_version:
                versions[pid] = new_version
            if current_flag != bool(flag):
                withdrawn[pid] = bool(flag)
    return {
        "versions": versions, "withdrawn": withdrawn,
        "counts": {
            "arxiv_papers": len(papers),
            "without_arxiv_id": len(papers) - len(by_arxiv_id),
            "in_harvest": seen,
            "not_in_harvest": len(by_arxiv_id) - seen,
            "source_version_to_set": len(versions),
            "withdrawn_to_set": sum(1 for f in withdrawn.values() if f),
            "withdrawn_to_clear": sum(1 for f in withdrawn.values() if not f),
        },
    }


def apply_backfill(db: sqlite3.Connection, plan: dict) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db:  # one transaction: all of it or none
        db.executemany("UPDATE papers SET source_version=? WHERE paper_id=? AND source_version IS NULL",
                       [(v, pid) for pid, v in plan["versions"].items()])
        db.executemany("UPDATE papers SET withdrawn=?, withdrawn_at=? WHERE paper_id=?",
                       [(int(flag), now if flag else None, pid) for pid, flag in plan["withdrawn"].items()])


def sync_qdrant(db: sqlite3.Connection, client, collections: list[str]) -> int:
    """Set `withdrawn: true` on the points of every withdrawn paper, in each collection."""
    from core.storage.qdrant_operations import set_paper_withdrawn

    ids = [pid for (pid,) in db.execute("SELECT paper_id FROM papers WHERE withdrawn=1")]
    for name in collections:
        set_paper_withdrawn(client, name, ids, True)
    return len(ids)


def main() -> int:
    from pipeline.paper_crawler.arxiv_corpus import harvest_root

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True, help="the application SQLite database")
    parser.add_argument("--harvest-db", type=Path, default=harvest_root() / "arxiv_cs.sqlite3")
    parser.add_argument("--apply", action="store_true", help="write the changes (default: only report them)")
    parser.add_argument("--sync-qdrant", action="store_true",
                        help="with --apply: also set withdrawn on the points of every withdrawn paper")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"))
    args = parser.parse_args()
    if args.sync_qdrant and not args.apply:
        parser.error("--sync-qdrant needs --apply")

    db = sqlite3.connect(args.db, timeout=60)
    db.execute("PRAGMA busy_timeout=60000")
    harvest = sqlite3.connect(f"file:{args.harvest_db}?mode=ro", uri=True)
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(papers)")}
        if not {"source_version", "withdrawn", "withdrawn_at"} <= columns:
            parser.error("papers lacks source_version/withdrawn/withdrawn_at: start the backend once (init_db migrates)")
        plan = plan_backfill(db, harvest)
        report = {"mode": "apply" if args.apply else "dry-run", **plan["counts"]}
        if args.apply:
            apply_backfill(db, plan)
            if args.sync_qdrant:
                from qdrant_client import QdrantClient

                from core.storage.corpus_layout import paper_collections

                client = QdrantClient(url=args.qdrant_url, timeout=120)
                names = [n for n in paper_collections().values() if client.collection_exists(n)]
                report["qdrant_withdrawn_papers"] = sync_qdrant(db, client, names)
                report["qdrant_collections"] = names
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        harvest.close()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
