"""TDD for preindex_reviews.preindex_records against a temp SQLite.

Verifies the SQL write contract WITHOUT touching the real DB / Qdrant / GPU:
  * accepted-in-corpus review -> chunks carry the resolved corpus paper_id
  * rejected review           -> NOT a paper; keeps openreview paper_id + pdf_url
  * chunk metadata carries the payload filter fields (paper_id/venue/year/tier)
  * re-running is idempotent (one Document per submission)
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.reviews.preindex_reviews import load_corpus_matcher, preindex_records


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT, source TEXT, source_type TEXT, data_source TEXT,
            content_hash TEXT, url TEXT, status TEXT, metadata_json TEXT
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER, content TEXT, level INTEGER, chunk_index INTEGER,
            start_idx INTEGER, end_idx INTEGER, metadata_json TEXT
        );
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY, doc_id INTEGER, title TEXT,
            venue TEXT, year INTEGER
        );
        """
    )
    # seed one accepted ICLR 2025 paper already in the corpus
    c.execute("INSERT INTO papers (paper_id, doc_id, title, venue, year) VALUES (?,?,?,?,?)",
              ("iclr-2025-5sRnsubyAK", 1, "A Great ICLR Paper", "ICLR", 2025))
    c.commit()
    return c


def _review(weak="Missing baselines; single dataset."):
    return {"reviewer": "Reviewer_a", "rating": 4, "confidence": 4,
            "summary": "Studies foo.", "strengths": "Clear.",
            "weaknesses": weak, "questions": ""}


def _rec(oid, title, accepted, **over):
    base = {
        "paper_id": f"openreview-iclr-2025-{oid}",
        "openreview_id": oid, "venue": "ICLR", "year": 2025,
        "title": title, "abstract": "abs", "accepted": accepted,
        "pdf_url": f"https://openreview.net/pdf?id={oid}",
        "forum_url": f"https://openreview.net/forum?id={oid}",
        "decision": "Accept" if accepted else "Reject",
        "avg_rating": 4.0, "ratings": [4], "n_reviews": 1,
        "reviews": [_review()], "meta_review": "Borderline.",
    }
    base.update(over)
    return base


@pytest.fixture
def records():
    return [
        _rec("5sRnsubyAK", "A Great ICLR Paper", True),       # in corpus
        _rec("rejForum99", "A Rejected Submission", False),   # not in corpus
    ]


def test_matched_and_rejected_split(conn, records):
    m = load_corpus_matcher(conn, {("ICLR", 2025)})
    stats = preindex_records(conn, records, m)
    assert stats["docs"] == 2
    assert stats["matched"] == 1
    assert stats["rejected"] == 1
    assert stats["chunks"] >= 2
    assert stats["min_chunk_id"] is not None


def test_accepted_review_links_corpus_paper_id(conn, records):
    m = load_corpus_matcher(conn, {("ICLR", 2025)})
    preindex_records(conn, records, m)
    row = conn.execute(
        "SELECT metadata_json FROM documents WHERE content_hash=?",
        ("review-ICLR-2025-5sRnsubyAK",)).fetchone()
    meta = json.loads(row[0])
    assert meta["paper_id"] == "iclr-2025-5sRnsubyAK"  # resolved to corpus
    assert meta["in_corpus"] is True
    # its chunks carry the corpus paper_id for grouping/payload
    chunk_metas = [json.loads(r[0]) for r in conn.execute(
        "SELECT c.metadata_json FROM chunks c JOIN documents d ON d.id=c.doc_id "
        "WHERE d.content_hash=?", ("review-ICLR-2025-5sRnsubyAK",))]
    assert chunk_metas and all(cm["paper_id"] == "iclr-2025-5sRnsubyAK" for cm in chunk_metas)


def test_rejected_review_keeps_openreview_id_and_pdf(conn, records):
    m = load_corpus_matcher(conn, {("ICLR", 2025)})
    preindex_records(conn, records, m)
    row = conn.execute(
        "SELECT metadata_json FROM documents WHERE content_hash=?",
        ("review-ICLR-2025-rejForum99",)).fetchone()
    meta = json.loads(row[0])
    assert meta["paper_id"] == "openreview-iclr-2025-rejForum99"
    assert meta["in_corpus"] is False
    assert meta["pdf_url"] == "https://openreview.net/pdf?id=rejForum99"  # link kept
    # NOT added to the papers table
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


def test_chunk_meta_has_payload_filter_fields(conn, records):
    m = load_corpus_matcher(conn, {("ICLR", 2025)})
    preindex_records(conn, records, m)
    metas = [json.loads(r[0]) for r in conn.execute("SELECT metadata_json FROM chunks")]
    assert metas
    for cm in metas:
        assert cm["venue"] == "ICLR"
        assert cm["year"] == 2025
        assert cm["venue_tier"] == "CCF-A"
        assert "paper_id" in cm


def test_all_review_documents_typed_review(conn, records):
    m = load_corpus_matcher(conn, {("ICLR", 2025)})
    preindex_records(conn, records, m)
    types = {r[0] for r in conn.execute("SELECT DISTINCT source_type FROM documents")}
    assert types == {"review"}


def test_rerun_is_idempotent(conn, records):
    m = load_corpus_matcher(conn, {("ICLR", 2025)})
    preindex_records(conn, records, m)
    n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    stats2 = preindex_records(conn, records, m)
    assert stats2["skipped_existing"] == 2
    assert stats2["docs"] == 0
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == n_docs
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == n_chunks
