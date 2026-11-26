"""TDD: search API metadata consistency.

Every search result MUST expose the same canonical metadata schema, regardless of
whether SQLite papers table has a row matching its doc_id. Falls back to Qdrant
payload's legacy fields (conference -> venue, author_names -> authors) when SQL miss.

Real system test: requires live FastAPI server at http://localhost:8000 + Qdrant.
"""
from __future__ import annotations

import os
import pytest
import requests


API = os.environ.get("RETRIVAL_API", "http://localhost:8000")
TIMEOUT = 30


REQUIRED_KEYS = {
    "venue",        # str (non-empty if doc is a paper)
    "year",         # int | None
    "venue_tier",   # str | None ("CCF-A", "CCF-B", ...)
    "track_type",   # str | None ("main", ...)
    "authors",      # list[str]
    "paper_id",     # str | None
}


def _post_search(query: str, top_k: int = 10, mode: str = "hybrid") -> dict:
    resp = requests.post(
        f"{API}/api/search",
        json={"query": query, "top_k": top_k, "mode": mode},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def test_server_reachable():
    """Sanity: FastAPI is up.

    /api/stats counts the full Qdrant collection + SQLite chunks; at 12.6M+ points
    that's ~6s on this box, so the reachability timeout is corpus-scaled (was 5s,
    which the corpus outgrew). Still tight enough to fail fast on a hung server.
    """
    r = requests.get(f"{API}/api/stats", timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    assert body["vector_index_size"] > 0


@pytest.mark.parametrize("query", [
    "attention transformer",
    "diffusion image generation",
    "reinforcement learning policy",
])
def test_every_result_has_required_metadata_keys(query):
    """Every result must contain ALL canonical keys, even when SQL paper row missing."""
    body = _post_search(query, top_k=10)
    results = body.get("results", [])
    assert len(results) > 0, f"no results for {query!r}"

    missing_per_result = []
    for r in results:
        meta = r.get("metadata") or {}
        missing = REQUIRED_KEYS - meta.keys()
        if missing:
            missing_per_result.append({
                "doc_id": r["doc_id"],
                "missing_keys": sorted(missing),
                "actual_keys": sorted(meta.keys()),
            })
    assert not missing_per_result, (
        f"{len(missing_per_result)}/{len(results)} results missing required metadata: "
        f"{missing_per_result[:3]}"
    )


@pytest.mark.parametrize("query", [
    "attention transformer",
    "diffusion image generation",
])
def test_paper_results_have_nonempty_venue_and_authors(query):
    """For source_type=paper results, venue must be non-empty and authors must be a list."""
    body = _post_search(query, top_k=10)
    results = body.get("results", [])
    assert results

    failures = []
    for r in results:
        meta = r.get("metadata") or {}
        if meta.get("source_type") != "paper":
            continue
        if not meta.get("venue"):
            failures.append({"doc_id": r["doc_id"], "issue": "empty venue",
                             "metadata": {k: meta.get(k) for k in ("venue","conference","year")}})
            continue
        authors = meta.get("authors")
        if not isinstance(authors, list):
            failures.append({"doc_id": r["doc_id"], "issue": f"authors not list: {type(authors).__name__}",
                             "metadata": {k: meta.get(k) for k in ("authors","author_names")}})
    assert not failures, (
        f"{len(failures)}/{len(results)} paper results have inconsistent venue/authors: "
        f"{failures[:3]}"
    )


def test_orphan_qdrant_points_fall_back_to_payload():
    """Even Qdrant points whose doc_id is missing from SQL must yield enriched metadata.

    The fallback path: when SQL JOIN misses, derive venue from payload['conference']
    and authors from payload['author_names'] (pipe-separated).
    """
    body = _post_search("attention transformer", top_k=10)
    results = body.get("results", [])

    orphan_failures = []
    for r in results:
        meta = r.get("metadata") or {}
        # Heuristic: a result that has legacy field 'conference' but no SQL-derived
        # 'venue_tier' would be the orphan case. After fix, venue must still be set.
        if meta.get("conference") and not meta.get("venue"):
            orphan_failures.append({
                "doc_id": r["doc_id"],
                "conference": meta.get("conference"),
                "venue": meta.get("venue"),
            })
    assert not orphan_failures, (
        f"orphan results not normalized (conference present but venue missing): "
        f"{orphan_failures[:3]}"
    )
