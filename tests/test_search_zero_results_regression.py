"""Regression: searches that should NOT zero out — real HTTP API, no mocks.

Locks two P0 bugs found 2026-05-16 AND the post-fix contract refactor:
  1. main.py enrichment did `seen_titles` dedup on doc_title → chunk-level retrieval
     for same paper collapsed to 1 chunk per paper. top_k=100 produced ~30 results.
  2. main.py forwarded `source_type="all"` literal to Qdrant filter → 0 hits because
     payload.source_type is "paper"/"webpage"/"upload", never "all".

After the user's "delete all application-layer特判" directive (2026-05-16):
  - Backend NO LONGER normalizes sentinels. Frontend (B1) drops 'all' → undefined
    BEFORE sending. Backend trusts the contract.
  - 'all' as an explicit payload filter value should therefore zero out — that is
    the *correct* boundary behavior: data integrity at the DB/payload layer, not
    application-layer guesswork.
"""
from __future__ import annotations

import os

import pytest
import requests


os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

API = os.environ.get("RETRIVAL_API", "http://127.0.0.1:8000")
TIMEOUT = 30


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


# Query known to have many chunks across many papers (verified via raw API).
COMMON_QUERY = "special token"


def _search(payload: dict) -> dict:
    r = _session().post(f"{API}/api/search", json=payload, timeout=TIMEOUT)
    assert r.status_code == 200, f"{r.status_code}: {r.text[:300]}"
    return r.json()


def test_top_k_100_returns_substantially_more_than_dedup_floor():
    """Bug 1 regression: title dedup used to collapse 100 → ~31.

    After fix: content-hash dedup still collapses arbitrary chunks that share
    deterministic content hash (legacy ghost orphans in Qdrant produce dupes
    against live points). Floor depends on Qdrant orphan ratio:
      pre orphan-cleanup:  100 → ~50  (47% orphan absorb)
      post orphan-cleanup: 100 → ~95+ (only true within-paper dedup)
    Lock floor at 40 — anything below means title dedup snuck back in.
    """
    body = _search({"query": COMMON_QUERY, "top_k": 100, "method": "hybrid"})
    assert body["total_results"] >= 40, (
        f"Got {body['total_results']} — way below dedup floor, "
        "likely title dedup re-introduced"
    )


def test_omitted_source_type_returns_unrestricted():
    """New contract: frontend omits source_type when 'all' selected.

    Backend must produce non-empty results when no source_type is sent (i.e. no
    payload filter is injected). This is the *intended* path users hit via
    Segmented = 'all'.
    """
    body = _search({"query": COMMON_QUERY, "top_k": 50, "method": "hybrid"})
    assert body["total_results"] > 0, (
        "no source_type produced 0 results — pipeline filter dispatch regressed"
    )


@pytest.mark.parametrize("invalid", ["all", "any", "null", "none", "ALL", " All "])
def test_source_type_sentinel_values_not_normalized(invalid: str):
    """Negative lock for B2: backend MUST NOT normalize 'all'/'any'/etc anymore.

    These values are NOT real Qdrant payload.source_type values (which are
    'paper'/'webpage'/'upload'), so they MUST zero out — that is the correct
    boundary behavior. Frontend is responsible for not sending them.
    """
    body = _search({
        "query": COMMON_QUERY, "top_k": 20, "method": "hybrid",
        "source_type": invalid,
    })
    assert body["total_results"] == 0, (
        f"source_type={invalid!r} returned {body['total_results']} — "
        "backend silently normalized a sentinel? Re-check main.py for特判 leakage."
    )


def test_source_type_paper_actually_filters_to_papers():
    """Valid source_type still filters correctly."""
    body = _search({"query": COMMON_QUERY, "top_k": 20, "method": "hybrid", "source_type": "paper"})
    assert body["total_results"] > 0
    for r in body["results"]:
        st = (r.get("metadata") or {}).get("source_type")
        # Qdrant payload may be missing source_type on legacy points; just assert no webpage.
        assert st != "webpage", f"source_type='paper' returned non-paper: {st}"


def test_chunk_level_results_not_deduped_by_title():
    """Same-paper multiple chunks must be allowed."""
    body = _search({"query": COMMON_QUERY, "top_k": 100, "method": "hybrid"})
    titles = [r["doc_title"] for r in body["results"]]
    # Total results must exceed unique titles for chunk-level retrieval to be useful;
    # if equal, that's classic dedup behavior we removed.
    assert len(titles) > len(set(titles)) or len(titles) >= 80, (
        f"Either dedup is back (titles {len(set(titles))} == results {len(titles)}) "
        f"or coverage is too low"
    )
