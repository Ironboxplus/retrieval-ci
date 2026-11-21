"""Pure logic for ingesting OpenReview reviews into the corpus.

Two responsibilities, both side-effect free (no DB / network / GPU) so they are
unit-tested in isolation:

  CorpusMatcher       resolve a harvested submission -> an EXISTING corpus
                      paper_id (the "论文关联" join).  ICLR corpus ids are
                      ``iclr-{year}-{forum_id}`` so the openreview forum id is an
                      exact suffix match; NeurIPS ids carry an opaque hash so we
                      fall back to a normalized-title match.  Validated 100% on
                      ICLR 2025 (67/67 forum-id) and NeurIPS 2024 (145/145 title)
                      accepted samples; rejected/withdrawn -> None (not in corpus).

  build_review_chunks turn one structured review record into the chunk units the
                      representation experiment picked: per-review title+weakness
                      (S3, level 0) + one per-paper digest (S5, level 1).

The chunks carry their filter fields (paper_id / venue / year / venue_tier) in
``meta`` so dump_chunks.py can lift them into the Qdrant payload — review
Documents have no papers-table row to LEFT JOIN, so the payload fields must ride
on the chunk metadata instead.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_title(title: Optional[str]) -> str:
    """Lowercase + strip everything but [a-z0-9] for a punctuation/spacing-robust
    title join key."""
    return _NON_ALNUM.sub("", (title or "").lower())


def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


class CorpusMatcher:
    """Resolve (venue, year, openreview_id, title) -> existing corpus paper_id.

    Built from the corpus papers table dump: rows of
    ``{paper_id, title, venue, year}``.  Keyed by (VENUE_UPPER, year) so the
    harvest side (which upper-cases venue) and the corpus side (mixed case) line
    up.  Tries the forum-id suffix first (exact, ICLR), then normalized title
    (NeurIPS); returns None when the submission is not in the corpus (rejected /
    withdrawn / camera-ready title drift with a hash-suffix venue).
    """

    def __init__(self, rows: Iterable[dict]):
        self._by_suffix: dict[tuple[str, int], dict[str, str]] = {}
        self._by_title: dict[tuple[str, int], dict[str, str]] = {}
        for r in rows:
            venue = (r.get("venue") or "").upper()
            year = r.get("year")
            pid = r.get("paper_id")
            if not (venue and pid and year is not None):
                continue
            key = (venue, int(year))
            suffix = pid.rsplit("-", 1)[-1]
            self._by_suffix.setdefault(key, {})[suffix] = pid
            self._by_title.setdefault(key, {})[normalize_title(r.get("title"))] = pid

    def resolve(self, venue: str, year: int, openreview_id: str,
                title: Optional[str]) -> Optional[str]:
        key = ((venue or "").upper(), int(year))
        suffix_map = self._by_suffix.get(key)
        if suffix_map and openreview_id in suffix_map:
            return suffix_map[openreview_id]
        title_map = self._by_title.get(key)
        if title_map:
            nt = normalize_title(title)
            if nt and nt in title_map:
                return title_map[nt]
        return None


def build_review_chunks(record: dict, paper_id: str,
                        venue_tier: Optional[str] = None) -> list[dict]:
    """Record -> list of chunk dicts {content, level, start_idx, kind, meta}.

    level 0 = per-review (title-injected weakness, the S3 winner); falls back to
    the review summary when a review has no weakness text so the opinion is not
    lost.  level 1 = one per-paper digest (S5: all reviewers' weaknesses +
    meta-review), emitted only when at least one weakness exists.

    ``start_idx`` is a doc-local running counter so (level, start_idx) is unique
    per Document — dump_chunks.py hashes (doc_id, level, start_idx, content) into
    the Qdrant point id and must not collide.
    """
    title = _clean(record.get("title"))
    meta_base = {
        "paper_id": paper_id,
        "venue": record.get("venue"),
        "year": record.get("year"),
        "venue_tier": venue_tier,
    }
    chunks: list[dict] = []
    weaknesses: list[str] = []
    idx = 0

    for rv in record.get("reviews", []):
        weak = _clean(rv.get("weaknesses"))
        summary = _clean(rv.get("summary"))
        if weak:
            content = f"Paper: {title}\nWeaknesses: {weak}"
            weaknesses.append(weak)
        elif summary:
            content = f"Paper: {title}\nReview: {summary}"
        else:
            continue  # nothing usable in this review
        chunks.append({
            "content": content,
            "level": 0,
            "start_idx": idx,
            "kind": "review",
            "meta": {**meta_base,
                     "reviewer": rv.get("reviewer"),
                     "rating": rv.get("rating"),
                     "confidence": rv.get("confidence")},
        })
        idx += 1

    if weaknesses:
        digest = f"Paper: {title}\nReviewer concerns:\n" + "\n".join(
            f"- {w}" for w in weaknesses)
        meta_review = _clean(record.get("meta_review"))
        if meta_review:
            digest += f"\nMeta-review: {meta_review}"
        chunks.append({
            "content": digest,
            "level": 1,
            "start_idx": idx,
            "kind": "digest",
            "meta": dict(meta_base),
        })

    return chunks
