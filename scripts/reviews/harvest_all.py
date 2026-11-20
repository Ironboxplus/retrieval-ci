"""Drive the full OpenReview review harvest across the venue registry.

Arc-side only (10808 proxy, W111). Robust to OpenReview's aggressive unauth rate
limit:
  * streams + writes incrementally, checkpointing the consumed offset per
    (venue, year) to ``{tag}.offset`` — a killed / rate-limited run RESUMES
    mid-venue, never re-harvesting from scratch.
  * a (venue, year) is marked complete only by a ``{tag}.done`` marker (written
    when the stream genuinely exhausts). RateLimited NEVER writes .done, so a
    truncated venue is retried, not silently accepted (the bug that capped
    ICLR_2024 at exactly 1000).
  * default call pace 2s keeps the sustained request rate under the limit so 429
    is the exception (rode out by the harvester's backoff), not the rule.

    python scripts/reviews/harvest_all.py --out-dir data/reviews
    # re-run the same line to resume after a rate-limit stop.
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
from scripts.reviews.harvest_openreview import OpenReviewHarvester, RateLimited  # noqa: E402
from scripts.reviews.review_venues import REVIEW_VENUES  # noqa: E402

logger = logging.getLogger("harvest_all")


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
    ap.add_argument("--call-pace", type=float, default=2.0,
                    help="seconds between API calls (keep under the rate limit)")
    ap.add_argument("--pace", type=float, default=3.0, help="seconds between (venue,year)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these venue names")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = OpenReviewHarvester(proxy=args.proxy, pace=args.call_pace)

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
            ckpt = out_dir / f"{tag}.offset"
            done = out_dir / f"{tag}.done"
            if done.exists():
                logger.info("SKIP %s (done)", tag)
                continue
            start = int(ckpt.read_text()) if ckpt.exists() else 0
            vid = v["id"].format(year=year)
            if start:
                logger.info("RESUME %s from offset %d", tag, start)
            f = out.open("a" if start else "w", encoding="utf-8")
            n = start
            try:
                for off, rec in h.iter_harvest(vid, year, v["name"], start_offset=start):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n = off
                    if off % 100 == 0:
                        f.flush()
                        ckpt.write_text(str(off))
                f.flush(); f.close()
                ckpt.write_text(str(n))
                done.write_text(str(n))
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
                f.flush(); f.close()
                ckpt.write_text(str(n))
                grand["rate_limited"] += 1
                logger.warning("RATE-LIMITED %s at offset %d (%s) — re-run to resume", tag, n, e)
            except Exception as e:  # noqa: BLE001
                f.flush(); f.close()
                logger.warning("ERROR %s at offset %d: %s", tag, n, e)
            time.sleep(args.pace)

    logger.info("ALL DONE: %s", json.dumps(grand))


if __name__ == "__main__":
    main()
