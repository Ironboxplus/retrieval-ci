"""Uploads have no paper_id, so the MCP must hand back a usable `doc:` id.

Before this, search_papers found an uploaded book and printed an EMPTY
`paper_id:` line, and search_within_paper/get_paper_markdown had nothing to pass
— the book was findable but not readable. `papers` gives upload/webpage payloads
no paper_id key at all (core/storage/papers_payload.py), so doc_id is the only
handle they have.

The routing functions are pure, so most of this runs without a backend. The live
acceptance at the bottom drives the real deployment end to end and is gated on
RETRIEVAL_MCP_REAL_SMOKE=1, matching test_mcp_multi_query_real.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcp_plugin"))

import retrieval_mcp as plugin  # noqa: E402

REAL_BACKEND = os.environ.get("RETRIEVAL_MCP_REAL_BACKEND", "http://10.100.100.88:18080")
REAL_SMOKE = os.environ.get("RETRIEVAL_MCP_REAL_SMOKE") == "1"


# --------------------------------------------------------------- id classification

@pytest.mark.parametrize("raw", ["doc:114854", "doc:1", "114854", "1", " doc:114854 "])
def test_document_ids_are_recognised(raw):
    assert plugin._is_document_id(raw) is True


@pytest.mark.parametrize(
    "raw",
    [
        "neurips-2024-some-paper",
        "icml-2026-example",
        "aaai-2022-20281",
        "iclr-2024-sLdVl0q68X",
        "",
    ],
)
def test_paper_ids_are_not_mistaken_for_documents(raw):
    """0 of 106,156 production paper_ids are all-digits (measured 2026-09-21),
    which is what makes the bare-number shorthand safe."""
    assert plugin._is_document_id(raw) is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("doc:114854", "114854"), ("114854", "114854"), (" doc: 114854 ", "114854")],
)
def test_document_id_strips_the_prefix(raw, expected):
    assert plugin._document_id(raw) == expected


# ------------------------------------------------------------------ scope routing

def test_document_id_scopes_by_doc_ids():
    assert plugin._scope_filter("doc:114854") == {"doc_ids": ["114854"]}


def test_paper_id_scopes_by_paper_ids():
    assert plugin._scope_filter("neurips-2024-x") == {"paper_ids": ["neurips-2024-x"]}


def test_scope_filter_never_emits_both_keys():
    """Both would become AND-ed payload conditions and match nothing; the server
    rejects the combination outright, so the client must never send it."""
    for raw in ("doc:114854", "114854", "neurips-2024-x"):
        assert len(plugin._scope_filter(raw)) == 1


# ---------------------------------------------------------------- result id line

def _one_paper(**overrides) -> dict:
    base = {"paper_id": "", "doc_id": "114854", "doc_title": "PML Advanced Topics",
            "max_score": 0.9, "chunks": [{"content": "latent variable models"}]}
    base.update(overrides)
    return {"papers": [base]}


def test_upload_result_prints_a_usable_doc_id():
    out = plugin._fmt_papers(_one_paper())
    assert "paper_id: doc:114854" in out


def test_paper_result_line_is_unchanged():
    out = plugin._fmt_papers(_one_paper(paper_id="neurips-2024-x"))
    assert "paper_id: neurips-2024-x" in out
    assert "doc:" not in out


def test_result_with_neither_id_prints_empty_rather_than_doc_colon():
    """A bare `doc:` would look like a valid handle and route to doc_ids=[''];
    an empty value at least reads as 'nothing to follow up with'."""
    out = plugin._fmt_papers(_one_paper(doc_id=""))
    assert "paper_id: \n" in out + "\n"
    assert "doc:" not in out


# ------------------------------------------------------------------- live backend

@pytest.mark.skipif(not REAL_SMOKE, reason="set RETRIEVAL_MCP_REAL_SMOKE=1")
def test_live_book_is_searchable_through_the_doc_id_it_advertises():
    """Find an upload, take the id the formatter printed, drill into it."""
    with httpx.Client(base_url=REAL_BACKEND, timeout=180.0) as client:
        found = client.post("/api/search", json={
            "query": "latent variable models", "top_k": 3,
            "source_type": "upload", "dedup_by_paper": True,
        })
        found.raise_for_status()
        papers = found.json().get("papers") or []
        assert papers, "no uploaded documents in the corpus to test against"

        rendered = plugin._fmt_papers(found.json())
        advertised = [
            line.split("paper_id:", 1)[1].strip()
            for line in rendered.splitlines() if "paper_id:" in line
        ]
        assert advertised and all(a for a in advertised), \
            f"formatter advertised an unusable id: {advertised}"

        scope = plugin._scope_filter(advertised[0])
        assert "doc_ids" in scope, "an upload must scope by doc_ids, not paper_ids"

        inner = client.post("/api/search", json={
            "query": "evidence lower bound", "top_k": 5,
            "dedup_by_paper": False, "method": "hybrid", **scope,
        })
        inner.raise_for_status()
        hits = inner.json().get("papers") or []
        assert hits, f"drilling into {advertised[0]} returned nothing"
        assert str(hits[0]["doc_id"]) == scope["doc_ids"][0]


@pytest.mark.skipif(not REAL_SMOKE, reason="set RETRIEVAL_MCP_REAL_SMOKE=1")
def test_live_paper_ids_and_doc_ids_are_mutually_exclusive_server_side():
    with httpx.Client(base_url=REAL_BACKEND, timeout=60.0) as client:
        response = client.post("/api/search", json={
            "query": "x", "paper_ids": ["neurips-2024-x"], "doc_ids": ["114854"],
        })
    assert response.status_code == 422, response.text


@pytest.mark.skipif(not REAL_SMOKE, reason="set RETRIEVAL_MCP_REAL_SMOKE=1")
def test_live_document_markdown_size_probe_costs_one_byte():
    """The inline guard must learn the size without moving the whole book."""
    with httpx.Client(base_url=REAL_BACKEND, timeout=180.0) as client:
        found = client.post("/api/search", json={
            "query": "latent variable models", "top_k": 1,
            "source_type": "upload", "dedup_by_paper": True,
        })
        found.raise_for_status()
        doc_id = str((found.json()["papers"][0])["doc_id"])
        probe = client.get(f"/api/documents/{doc_id}/markdown",
                           headers={"Range": "bytes=0-0"})
    assert probe.status_code == 206, probe.text
    assert len(probe.content) == 1
    match = plugin._CONTENT_RANGE_PATTERN.match(probe.headers["Content-Range"])
    assert match, probe.headers.get("Content-Range")
    total = int(match.group(3))
    assert total > plugin._INLINE_DOCUMENT_MARKDOWN_MAX_BYTES, (
        "expected a book large enough to exercise the inline refusal"
    )
