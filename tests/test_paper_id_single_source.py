"""TDD: paper_id has ONE authoritative source.

Architecture decision: Paper.paper_id (SQL column, UNIQUE INDEX) is the canonical
source of truth for the business paper_id (e.g. "aaai-2022-20281"). All other
locations are derived:
- documents.metadata_json["paper_id"]: original write by scraper (kept for audit)
- Qdrant payload["paper_id"]: dumped from Paper.paper_id at ingest time
- API response metadata["paper_id"]: served from Qdrant payload (no SQL JOIN
  needed for this field; SQL only contributes fields Qdrant doesn't have).

Previous bug: main.py wrote `paper_id: p.arxiv_id` (None for all 44K rows) into
the dict, then `{**r.metadata, **paper_info}` let that None overwrite the
correct value from Qdrant payload. Result: 100/100 search results had
paper_id="None" / None.

Real system test: requires live FastAPI at http://localhost:8000 + Qdrant.
"""
from __future__ import annotations

import os
import sqlite3
import pytest
import requests


API = os.environ.get("RETRIVAL_API", "http://localhost:8000")
DB_PATH = os.environ.get("RETRIVAL_DB", "data/db/retrieval.db")
TIMEOUT = 30


def _post_search(query: str, top_k: int = 10, mode: str = "hybrid") -> dict:
    resp = requests.post(
        f"{API}/api/search",
        json={"query": query, "top_k": top_k, "method": mode},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def test_paper_table_has_paper_id_column():
    """Paper SQL table must expose paper_id column (single source of truth)."""
    conn = sqlite3.connect(DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
    conn.close()
    assert "paper_id" in cols, f"papers table missing paper_id column. Have: {sorted(cols)}"


def test_paper_id_column_backfilled():
    """After backfill, Paper.paper_id coverage must be >= 99% for paper rows."""
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with_pid = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE paper_id IS NOT NULL AND paper_id != ''"
    ).fetchone()[0]
    conn.close()
    assert total > 0, "papers table empty"
    coverage = with_pid / total
    assert coverage >= 0.99, f"paper_id coverage only {coverage:.1%} ({with_pid}/{total})"


def test_paper_id_column_unique():
    """Paper.paper_id must be unique (DB-layer guarantee, not application code)."""
    conn = sqlite3.connect(DB_PATH)
    dup = conn.execute("""
        SELECT paper_id, COUNT(*) c FROM papers
        WHERE paper_id IS NOT NULL AND paper_id != ''
        GROUP BY paper_id HAVING c > 1 LIMIT 5
    """).fetchall()
    conn.close()
    assert not dup, f"duplicate paper_id values: {dup}"


@pytest.mark.parametrize("query", [
    "reinforcement learning",
    "attention transformer",
    "diffusion image generation",
])
def test_search_response_paper_id_not_none(query):
    """Every paper result in /api/search must have a non-None, non-'None' paper_id."""
    body = _post_search(query, top_k=20)
    results = body.get("results", [])
    assert results, f"no results for {query!r}"

    bad = []
    for r in results:
        meta = r.get("metadata") or {}
        if meta.get("source_type") != "paper":
            continue
        pid = meta.get("paper_id")
        if pid is None or pid == "" or pid == "None":
            bad.append({"doc_id": r.get("doc_id"), "paper_id": pid})
    assert not bad, (
        f"{len(bad)}/{len(results)} paper results have empty/None paper_id "
        f"(architecture: Paper.paper_id must be authoritative). Samples: {bad[:3]}"
    )


def test_search_paper_id_matches_sql():
    """API-returned paper_id must equal SQL Paper.paper_id (sanity that they agree)."""
    body = _post_search("reinforcement learning", top_k=5)
    results = body.get("results", [])
    assert results

    conn = sqlite3.connect(DB_PATH)
    try:
        mismatches = []
        for r in results:
            doc_id = r.get("doc_id")
            try:
                doc_id_int = int(doc_id)
            except (ValueError, TypeError):
                continue
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE doc_id = ?", (doc_id_int,)
            ).fetchone()
            if not row:
                continue
            sql_pid = row[0]
            api_pid = (r.get("metadata") or {}).get("paper_id")
            if sql_pid and api_pid != sql_pid:
                mismatches.append({"doc_id": doc_id, "sql": sql_pid, "api": api_pid})
        assert not mismatches, f"API paper_id doesn't match SQL: {mismatches[:3]}"
    finally:
        conn.close()


def test_no_arxiv_id_used_for_paper_id():
    """Regression: main.py must NOT use Paper.arxiv_id as paper_id.

    Paper.arxiv_id is 0% populated and conceptually different from business paper_id.
    Asserts the bug we just fixed doesn't come back via grep on the source file.
    """
    main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    with open(main_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert '"paper_id": p.arxiv_id' not in src, (
        "main.py still has `\"paper_id\": p.arxiv_id` — this overwrites the "
        "correct value from Qdrant payload with None."
    )
