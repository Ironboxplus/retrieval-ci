"""TDD for the OpenReview review-ingest pure logic (scripts/reviews/review_ingest.py).

Covers the two pieces that must be exactly right before anything touches SQL/Qdrant:
  1. CorpusMatcher — resolve a harvested submission to an EXISTING corpus paper_id
     (forum-id suffix for ICLR, normalized-title for NeurIPS), or None for
     rejected/withdrawn (not in corpus).  Validated 100% on real samples.
  2. build_review_chunks — turn a structured review record into the chunk units
     the representation experiment picked: per-review (title+weakness, S3) +
     per-paper digest (S5), with stable level / start_idx for point-id hashing.

All hermetic: no DB, no network, no GPU.
"""
from __future__ import annotations

import pytest

from scripts.reviews.review_ingest import (
    CorpusMatcher,
    build_review_chunks,
    normalize_title,
)


# --------------------------------------------------------------------------- #
# normalize_title
# --------------------------------------------------------------------------- #
def test_normalize_title_strips_punct_case_space():
    assert normalize_title("Attention Is All You Need!") == "attentionisallyouneed"
    assert normalize_title("  Foo: Bar-Baz (v2) ") == "foobarbazv2"
    assert normalize_title(None) == ""
    # Two titles differing only in punctuation/case collapse to the same key.
    assert normalize_title("How does PDE order affect the convergence of PINNs?") == \
        normalize_title("how does pde order affect the convergence of pinns")


# --------------------------------------------------------------------------- #
# CorpusMatcher
# --------------------------------------------------------------------------- #
@pytest.fixture
def corpus_rows():
    # Mimic the real id schemes: ICLR suffix = openreview forum id;
    # NeurIPS suffix = an opaque hash (so forum-id match must FAIL and title win).
    return [
        {"paper_id": "iclr-2025-5sRnsubyAK", "title": "A Great ICLR Paper", "venue": "ICLR", "year": 2025},
        {"paper_id": "iclr-2025-AbCdEf1234", "title": "Another ICLR Work", "venue": "ICLR", "year": 2025},
        {"paper_id": "neurips-2024-000f947dcaff8fbffcc3f53a1314f358",
         "title": "MicroAdam: Low Space Overhead", "venue": "NeurIPS", "year": 2024},
    ]


def test_matcher_iclr_by_forum_id(corpus_rows):
    m = CorpusMatcher(corpus_rows)
    # forum-id wins even if the camera-ready title drifted from the submission.
    assert m.resolve("ICLR", 2025, "5sRnsubyAK", "A Great ICLR Paper (renamed)") == "iclr-2025-5sRnsubyAK"


def test_matcher_neurips_by_title_when_suffix_is_hash(corpus_rows):
    m = CorpusMatcher(corpus_rows)
    # forum id won't be in the hash-suffix space → fall back to normalized title.
    pid = m.resolve("NeurIPS", 2024, "zZzForumId", "MicroAdam: Low Space Overhead!")
    assert pid == "neurips-2024-000f947dcaff8fbffcc3f53a1314f358"


def test_matcher_venue_case_insensitive(corpus_rows):
    m = CorpusMatcher(corpus_rows)
    # harvest emits venue upper ("NEURIPS"); corpus stores "NeurIPS".
    assert m.resolve("NEURIPS", 2024, "x", "MicroAdam: Low Space Overhead") is not None


def test_matcher_rejected_returns_none(corpus_rows):
    m = CorpusMatcher(corpus_rows)
    # withdrawn/rejected submission: neither forum id nor title in corpus.
    assert m.resolve("ICLR", 2025, "unknownForum", "Some Withdrawn Submission") is None


def test_matcher_wrong_year_does_not_crossmatch(corpus_rows):
    m = CorpusMatcher(corpus_rows)
    assert m.resolve("ICLR", 2024, "5sRnsubyAK", "A Great ICLR Paper") is None


# --------------------------------------------------------------------------- #
# build_review_chunks
# --------------------------------------------------------------------------- #
def _rec(**over):
    base = {
        "paper_id": "openreview-iclr-2025-xYz",
        "openreview_id": "xYz",
        "venue": "ICLR",
        "year": 2025,
        "title": "Scaling Laws for Foo Models",
        "abstract": "We study foo.",
        "accepted": False,
        "avg_rating": 4.0,
        "meta_review": "Borderline; reviewers raise evaluation concerns.",
        "reviews": [
            {"reviewer": "Reviewer_a", "rating": 5, "confidence": 4,
             "summary": "Studies foo scaling.", "strengths": "Clear writing.",
             "weaknesses": "Missing baselines; only one dataset.", "questions": "Why no ImageNet?"},
            {"reviewer": "Reviewer_b", "rating": 3, "confidence": 3,
             "summary": "Foo models.", "strengths": "Novel idea.",
             "weaknesses": "Theory assumptions too strong.", "questions": ""},
        ],
    }
    base.update(over)
    return base


def test_build_chunks_per_review_plus_digest():
    chunks = build_review_chunks(_rec(), paper_id="iclr-2025-xYz")
    kinds = [c["kind"] for c in chunks]
    # two reviews with weaknesses -> 2 per-review chunks + 1 digest
    assert kinds.count("review") == 2
    assert kinds.count("digest") == 1


def test_per_review_chunk_is_title_injected_weakness():
    chunks = build_review_chunks(_rec(), paper_id="iclr-2025-xYz")
    rev = [c for c in chunks if c["kind"] == "review"][0]
    assert "Scaling Laws for Foo Models" in rev["content"]
    assert "Missing baselines" in rev["content"]
    assert rev["level"] == 0


def test_digest_aggregates_weaknesses_and_meta():
    chunks = build_review_chunks(_rec(), paper_id="iclr-2025-xYz")
    dig = [c for c in chunks if c["kind"] == "digest"][0]
    assert "Missing baselines" in dig["content"]
    assert "Theory assumptions too strong" in dig["content"]
    assert "evaluation concerns" in dig["content"]  # meta-review folded in
    assert dig["level"] == 1


def test_chunks_have_unique_start_idx_for_point_id_hash():
    chunks = build_review_chunks(_rec(), paper_id="iclr-2025-xYz")
    # (level, start_idx) must be unique within the doc so dump_chunks' point-id
    # hash (doc_id_level_startidx_contenthash) never collides.
    keys = {(c["level"], c["start_idx"]) for c in chunks}
    assert len(keys) == len(chunks)


def test_review_without_weakness_falls_back_to_summary_not_dropped():
    rec = _rec(reviews=[
        {"reviewer": "Reviewer_c", "rating": 6, "confidence": 2,
         "summary": "A solid empirical study of foo.", "strengths": "Good.",
         "weaknesses": "", "questions": ""},
    ], meta_review="")
    chunks = build_review_chunks(rec, paper_id="iclr-2025-xYz")
    rev = [c for c in chunks if c["kind"] == "review"]
    assert len(rev) == 1  # not dropped
    assert "solid empirical study" in rev[0]["content"]
    # no weaknesses anywhere -> no digest
    assert not [c for c in chunks if c["kind"] == "digest"]


def test_no_reviews_yields_no_chunks():
    assert build_review_chunks(_rec(reviews=[]), paper_id="iclr-2025-xYz") == []


def test_chunk_meta_carries_filter_fields_for_payload():
    # paper_id/venue/year/venue_tier must ride in chunk meta so dump_chunks can
    # put them in the Qdrant payload (review docs have no papers-table row).
    chunks = build_review_chunks(_rec(), paper_id="iclr-2025-xYz",
                                 venue_tier="CCF-A")
    for c in chunks:
        assert c["meta"]["paper_id"] == "iclr-2025-xYz"
        assert c["meta"]["venue"] == "ICLR"
        assert c["meta"]["year"] == 2025
        assert c["meta"]["venue_tier"] == "CCF-A"
