"""Drive the full OpenReview review harvest across the venue registry.

Network goes through the explicit 10808 proxy. Writes and id cursors are committed
incrementally. Legacy offset-only output is kept intact and requires a separate
refresh directory. Only genuine exhaustion writes a completion marker. A browser
challenge, exhausted rate-limit retries, or request budget stops the whole run.

Default: at most three HTTP attempts, twenty seconds between successful calls.
These are conservative local controls, not a claim about OpenReview's quota.

    python scripts/reviews/harvest_all.py --out-dir data/reviews/public-refresh --only ICML
    # Re-run with the same directory to continue after reviewing the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.reviews.harvest_openreview import (  # noqa: E402
    OpenReviewHarvester, RateLimited, AccessChallengeRequired, RequestBudgetReached,
)
from scripts.reviews.review_venues import REVIEW_VENUES  # noqa: E402
from pipeline.paper_crawler.runtime_files import write_text_atomic  # noqa: E402

logger = logging.getLogger("harvest_all")


def _save_cursor(path: Path, cursor: dict) -> None:
    write_text_atomic(path, json.dumps(cursor), durable=True)


def harvest_venue(h, venue: str, year: int, name: str, out: Path) -> int:
    """Commit data before its cursor; resume without truncation or duplicate rows.

    Old offset files used a different sort order. They cannot safely seed an id
    cursor, so keep them intact and require a new output directory for refresh.
    """
    cursor_path = out.with_suffix(".cursor.json")
    done = out.with_suffix(".done")
    if done.exists():
        return int(done.read_text())
    if out.exists() and out.stat().st_size and not cursor_path.exists():
        raise RuntimeError("legacy harvest has no id cursor; use a new output directory")
    cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else {
        "count": 0, "after": None, "bytes": 0}
    if not out.exists() and cursor["bytes"]:
        raise RuntimeError("checkpoint data file is missing")
    if not cursor_path.exists():
        _save_cursor(cursor_path, cursor)
    with out.open("r+b" if out.exists() else "w+b") as f:
        if f.seek(0, os.SEEK_END) < cursor["bytes"]:
            raise RuntimeError("checkpoint extends beyond committed data")
        f.truncate(cursor["bytes"])
        f.seek(cursor["bytes"])
        for count, rec in h.iter_harvest(venue, year, name,
                start_offset=cursor["count"], after=cursor["after"]):
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
            cursor = {"count": count, "after": rec["openreview_id"], "bytes": f.tell()}
            _save_cursor(cursor_path, cursor)
            if count % 50 == 0:
                logger.info("CHECKPOINT %s %s: %d submissions", name, year, count)
    done.write_text(str(cursor["count"]))
    return cursor["count"]


def _file_stats(path: Path) -> tuple[int, int, int]:
    """(submissions, with_reviews, accepted) of a written jsonl."""
    subs = wr = acc = 0
    if not path.exists():
        return 0, 0, 0
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        subs += 1
        if r.get("n_reviews", 0) > 0:
            wr += 1
        if r.get("accepted"):
            acc += 1
    return subs, wr, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/reviews")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or "http://127.0.0.1:10808")
    ap.add_argument("--call-pace", type=float, default=20.0,
                    help="seconds between successful API calls; conservative local policy, not an official quota")
    ap.add_argument("--max-requests", type=int, default=3,
                    help="HTTP attempt budget, including retries; 0 explicitly allows an unbounded run")
    ap.add_argument("--pace", type=float, default=3.0, help="seconds between (venue,year)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these venue names")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = OpenReviewHarvester(proxy=args.proxy, pace=args.call_pace,
                           max_requests=args.max_requests or None)

    venues = REVIEW_VENUES
    if args.only:
        keep = {n.upper() for n in args.only}
        venues = [v for v in venues if v["name"].upper() in keep]

    grand = {"venue_years_done": 0, "submissions": 0, "with_reviews": 0, "accepted": 0,
             "rate_limited": 0}
    for v in venues:
        for year in v["years"]:
            tag = f"{v['name']}_{year}"
            out = out_dir / f"{tag}.jsonl"
            done = out_dir / f"{tag}.done"
            if done.exists():
                logger.info("SKIP %s (done)", tag)
                continue
            vid = v["id"].format(year=year)
            try:
                n = harvest_venue(h, vid, year, v["name"], out)
                if n == 0:
                    out.unlink(missing_ok=True)
                    logger.info("EMPTY %s (no submissions / not public)", tag)
                    continue
                subs, wr, acc = _file_stats(out)
                logger.info("DONE %s -> %d subs (%d w/reviews, %d accepted)", tag, subs, wr, acc)
                grand["venue_years_done"] += 1
                grand["submissions"] += subs
                grand["with_reviews"] += wr
                grand["accepted"] += acc
            except RateLimited as e:
                grand["rate_limited"] += 1
                logger.warning("RATE-LIMITED %s (%s); stopping all venues", tag, e)
                raise
            except (AccessChallengeRequired, RequestBudgetReached):
                logger.warning("STOP %s; data and cursor preserved", tag)
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("ERROR %s: %s; harvest incomplete", tag, e)
                raise
            time.sleep(args.pace)

    logger.info("ALL DONE: %s", json.dumps(grand))


if __name__ == "__main__":
    main()
