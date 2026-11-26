"""Harvest public OpenReview reviews (reviews + rebuttals + meta-review + decision)
for a venue into a structured JSONL, one record per submission.

Confirmed public-review venues (probed 2026-06-16): ICLR.cc/{year}/Conference,
NeurIPS.cc/{year}/Conference, MIDL.io/{year}/Conference. ICML / AAAI / thecvf
(CVPR/ICCV/ECCV) / ACL family / KDD do NOT expose reviews publicly.

Network: OpenReview (api2.openreview.net) is reached ONLY through the 10808 proxy
(W111: the 4090 box has no egress; harvest runs Arc-side). The proxy is read from
HTTPS_PROXY/HTTP_PROXY env or --proxy; we do NOT silently fall back to direct.

Scope: includes REJECTED / WITHDRAWN submissions — every submission under the
venue's Submission invitation, not just accepted (the criticism signal is richest
on rejected papers).

Output record (one per submission):
    {paper_id, openreview_id, venue, year, title, abstract, pdf_url, forum_url,
     decision, accepted, ratings:[int], avg_rating, n_reviews,
     reviews:[{reviewer, rating, confidence, soundness, presentation, contribution,
               summary, strengths, weaknesses, questions}],
     meta_review, rebuttals:[str], decision_text}

This is the harvest layer only. Chunking / embedding / ingest is a separate stage
(reviews are text-only — no PDF/marker), keyed by paper_id with source_type=review.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger("harvest_openreview")


class RateLimited(Exception):
    """Raised when a request keeps getting 429 past all retries. The caller MUST
    treat this as 'incomplete, resume later' — NEVER as 'no more data' (that bug
    silently truncated big venues at the first sustained rate-limit window)."""


API = "https://api2.openreview.net/notes"

# Each venue's review form uses different field names. Map a canonical role to an
# ordered list of candidate field names (first non-empty wins) so the harvested
# record schema is uniform across ICLR / NeurIPS / COLM / CoRL / UAI / D&B / ...
# The weakness role falls back to a venue's free-text `review` field (CoRL) when
# there's no dedicated weakness field — that's where the critique lives.
REVIEW_TEXT_MAP = {
    "summary":    ["summary", "summary_of_paper", "summary_and_contributions",
                   "Q1_Summary_and_contributions", "summary_of_contributions"],
    "strengths":  ["strengths", "Q3_Main_strengths", "reasons_to_accept"],
    "weaknesses": ["weaknesses", "Q4_Main_weakness", "reasons_to_reject",
                   "opportunities_for_improvement", "requested_changes",
                   "other_strengths_and_weaknesses", "review"],
    "questions":  ["questions", "questions_for_rebuttal", "questions_for_authors",
                   "Q5_Detailed_comments_to_the_authors", "additional_feedback"],
}
REVIEW_SCORE_MAP = {
    "rating":       ["rating", "recommendation", "summary_of_recommendation",
                     "overall_recommendation"],
    "confidence":   ["confidence"],
    "soundness":    ["soundness", "correctness", "technical_quality"],
    "presentation": ["presentation", "clarity", "clarity_of_presentation"],
    "contribution": ["contribution", "potential_impact"],
}
REVIEW_TEXT_FIELDS = tuple(REVIEW_TEXT_MAP)
REVIEW_SCORE_FIELDS = tuple(REVIEW_SCORE_MAP)


def _pick(content: dict, candidates: list):
    """First non-empty value among candidate field names (venue-flexible)."""
    for k in candidates:
        v = _val(content, k)
        if v is not None and (not isinstance(v, str) or v.strip()):
            return v
    return None


def _derive_name(venue_id: str) -> str:
    """Best-effort canonical short name from an OpenReview venue id. Correct for
    acronym-prefixed ids (ICLR.cc/.., MIDL.io/..) but NOT for org-prefixed ones
    (colmweb.org/COLM/.., auai.org/UAI/..) — pass --name explicitly there. The
    review-venue registry below carries the right name per venue."""
    head = venue_id.split("/")[0]
    return head.split(".")[0].upper()


def _val(content: dict, key: str) -> Optional[Any]:
    """v2 content fields are {'value': X}; v1 are bare. Return the value or None."""
    if key not in content:
        return None
    v = content[key]
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def _suffix(note: dict) -> str:
    invs = note.get("invitations") or ([note["invitation"]] if note.get("invitation") else ["?"])
    return invs[0].split("/-/")[-1]


def _first_int(value: Any) -> Optional[int]:
    """Ratings come as '5: marginally above' or 5 — extract the leading int."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            break
    return int(num) if num else None


class OpenReviewHarvester:
    def __init__(self, proxy: Optional[str], tries: int = 6, base_sleep: float = 1.2,
                 pace: float = 0.3):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "retrival-reviews/1.0"})
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
        else:
            # W111: never go direct. Require an explicit proxy (env or flag).
            raise SystemExit(
                "no proxy configured — set HTTPS_PROXY/HTTP_PROXY or --proxy "
                "(OpenReview must be reached via 10808; direct egress is forbidden)"
            )
        self.tries = tries
        self.base_sleep = base_sleep
        self.pace = pace

    def _get(self, params: dict) -> dict:
        last = None
        for i in range(self.tries):
            r = self.session.get(API, params=params, timeout=45)
            last = r.status_code
            if r.status_code == 200:
                time.sleep(self.pace)  # be polite to the shared proxy IP
                return r.json()
            wait = self.base_sleep * (2 ** i)
            logger.warning("HTTP %s on %s — retry %d/%d in %.1fs",
                           r.status_code, params.get("forum") or params.get("invitation"),
                           i + 1, self.tries, wait)
            time.sleep(wait)
        # Exhausted retries — DO NOT pretend "no data". Raise so the driver
        # checkpoints + resumes instead of writing a truncated file as complete.
        raise RateLimited(f"{last} after {self.tries} tries: "
                          f"{params.get('forum') or params.get('invitation')}")

    def iter_submissions(self, venue: str, limit: Optional[int]):
        """Yield submission notes for a venue, paginating. Includes rejected."""
        invitations = [f"{venue}/-/Submission", f"{venue}/-/Blind_Submission"]
        page = 500 if not limit else min(500, limit)
        for inv in invitations:
            offset, got = 0, 0
            while True:
                want = page if not limit else min(page, limit - got)
                if want <= 0:
                    return
                d = self._get({"invitation": inv, "limit": want, "offset": offset})
                notes = d.get("notes", [])
                if not notes:
                    break
                for n in notes:
                    yield n
                    got += 1
                offset += len(notes)
                if limit and got >= limit:
                    return
                if len(notes) < want:
                    break
            if got:
                return  # this invitation worked; don't try the fallback

    def forum_notes(self, forum_id: str) -> list[dict]:
        return self._get({"forum": forum_id}).get("notes", [])

    def parse_submission(self, sub: dict, venue: str, year: int,
                         venue_name: str,
                         replies: Optional[list[dict]] = None) -> dict:
        content = sub.get("content", {})
        title = _val(content, "title") or ""
        abstract = _val(content, "abstract") or ""
        venue_str = (_val(content, "venue") or "") or ""
        venueid = (_val(content, "venueid") or "") or ""
        # PDF link: OpenReview hosts the PDF even for rejected/withdrawn papers.
        # We keep ONLY the link for non-corpus (rejected) papers — they are never
        # ingested as papers, just referenced. v2 gives a relative "/pdf/..".
        pdf_val = _val(content, "pdf")
        pdf_url = None
        if pdf_val:
            pdf_url = str(pdf_val) if str(pdf_val).startswith("http") \
                else f"https://openreview.net{pdf_val}"
        forum_url = f"https://openreview.net/forum?id={sub['id']}"
        # accepted iff the venue/venueid is NOT a rejected/withdrawn/submitted marker
        low = (venue_str + " " + venueid).lower()
        accepted = not any(k in low for k in ("reject", "withdraw", "desk", "submitted to"))

        if replies is None:
            replies = self.forum_notes(sub["id"])

        reviews, rebuttals = [], []
        meta_review, decision_text = None, None
        for n in replies:
            suf = _suffix(n)
            c = n.get("content", {})
            if suf in ("Official_Review", "Review"):
                rec = {"reviewer": (n.get("signatures") or ["?"])[0].split("/")[-1]}
                for role, cands in REVIEW_SCORE_MAP.items():
                    rec[role] = _pick(c, cands)
                for role, cands in REVIEW_TEXT_MAP.items():
                    rec[role] = _pick(c, cands)
                reviews.append(rec)
            elif suf == "Meta_Review":
                meta_review = _val(c, "metareview") or _val(c, "summary") or _val(c, "recommendation")
            elif suf == "Decision":
                decision_text = _val(c, "decision") or _val(c, "recommendation")
            elif suf in ("Rebuttal", "Author_Rebuttal"):
                txt = _val(c, "rebuttal") or _val(c, "comment")
                if txt:
                    rebuttals.append(txt)

        ratings = [r for r in (_first_int(rv.get("rating")) for rv in reviews) if r is not None]
        return {
            "paper_id": f"openreview-{venue_name.lower()}-{year}-{sub['id']}",
            "openreview_id": sub["id"],
            "venue": venue_name,
            "venue_full": venue,
            "year": year,
            "title": title,
            "abstract": abstract,
            "pdf_url": pdf_url,
            "forum_url": forum_url,
            "decision": venue_str or decision_text,
            "accepted": accepted,
            "ratings": ratings,
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "n_reviews": len(reviews),
            "reviews": reviews,
            "meta_review": meta_review,
            "decision_text": decision_text,
            "rebuttals": rebuttals,
        }

    def harvest(self, venue: str, year: int, limit: Optional[int], use_batch: bool,
                venue_name: Optional[str] = None) -> list[dict]:
        name = venue_name or _derive_name(venue)
        out = []
        if use_batch:
            # Efficient path: pull submissions WITH directReplies in one call each page.
            offset = 0
            page = 50
            inv = f"{venue}/-/Submission"
            while True:
                want = page if not limit else min(page, limit - len(out))
                if want <= 0:
                    break
                d = self._get({"invitation": inv, "limit": want, "offset": offset,
                               "details": "replies"})
                notes = d.get("notes", [])
                if not notes:
                    break
                for sub in notes:
                    reps = (sub.get("details") or {}).get("replies", [])
                    out.append(self.parse_submission(sub, venue, year, name, replies=reps))
                offset += len(notes)
                logger.info("batch harvested %d submissions (offset=%d)", len(out), offset)
                if (limit and len(out) >= limit) or len(notes) < want:
                    break
            return out
        # Fallback: per-forum (1 call/paper). Slower but robust to 429 on details.
        for i, sub in enumerate(self.iter_submissions(venue, limit), 1):
            out.append(self.parse_submission(sub, venue, year, name))
            if i % 25 == 0:
                logger.info("per-forum harvested %d submissions", i)
        return out

    def iter_harvest(self, venue: str, year: int, venue_name: str,
                     start_offset: int = 0):
        """Stream (offset_consumed, record) from start_offset, paging the batch
        endpoint. Lets the driver write + checkpoint incrementally so a big venue
        resumes mid-way. Propagates RateLimited (NOT swallowed) so a truncated run
        is never mistaken for a complete one. Stops on a genuinely empty/short page.
        """
        inv = f"{venue}/-/Submission"
        offset = start_offset
        page = 50
        while True:
            d = self._get({"invitation": inv, "limit": page, "offset": offset,
                           "details": "replies"})
            notes = d.get("notes", [])
            if not notes:
                return
            for sub in notes:
                reps = (sub.get("details") or {}).get("replies", [])
                offset += 1
                yield offset, self.parse_submission(sub, venue, year, venue_name,
                                                    replies=reps)
            if len(notes) < page:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, help="e.g. ICLR.cc/2025/Conference")
    ap.add_argument("--name", default=None,
                    help="canonical venue short name (e.g. COLM, UAI). REQUIRED for "
                         "org-prefixed ids (colmweb.org/COLM/..); auto-derived otherwise.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
    ap.add_argument("--per-forum", action="store_true",
                    help="use per-forum fetch (robust) instead of batch details=replies")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    h = OpenReviewHarvester(proxy=args.proxy)
    name = args.name or _derive_name(args.venue)
    logger.info("harvest %s (name=%s) year=%s limit=%s proxy=%s batch=%s",
                args.venue, name, args.year, args.limit, args.proxy, not args.per_forum)
    t0 = time.time()
    records = h.harvest(args.venue, args.year, args.limit,
                        use_batch=not args.per_forum, venue_name=name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with_rev = [r for r in records if r["n_reviews"] > 0]
    acc = [r for r in records if r["accepted"]]
    logger.info("DONE %d submissions in %.0fs -> %s", len(records), time.time() - t0, out)
    logger.info("  with public reviews: %d/%d | accepted: %d | rejected: %d",
                len(with_rev), len(records), len(acc), len(records) - len(acc))
    if with_rev:
        import statistics
        nrev = statistics.mean(r["n_reviews"] for r in with_rev)
        wlen = statistics.mean(
            len(rv.get("weaknesses") or "") for r in with_rev for rv in r["reviews"])
        logger.info("  avg reviews/paper: %.1f | avg weakness chars: %.0f", nrev, wlen)


if __name__ == "__main__":
    main()
