"""TDD: every `papers` Qdrant write path must produce EXACTLY what
project_papers_payload would produce for the same raw fields — no path may
hand-roll its own payload dict (docs/papers_payload_contract_plan.md 3.2/3.4).

Write paths (plan.md 1.1):
  1. pipeline/paper_crawler/integration/api_routes.py — single-paper reindex
  2. core/storage/qdrant_store.py::QdrantStore.add_chunks — live ingest (also
     upload/webpage's only writer)
  3. scripts/remote_embed/local_ingest.py::canonicalize_payload — batch
     parquet ingest
  4. scripts/recreate_qdrant_collection.py — index schema, covered by its own
     tests/test_recreate_collection_indexes.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "remote_embed"))

from app.models.schemas import Chunk
from core.storage.papers_payload import project_papers_payload
from core.storage.qdrant_store import QdrantStore
from pipeline.paper_crawler.integration.api_routes import _build_paper_point_payload


class _RecordingClient:
    """Fake Qdrant client that records every upserted point's payload."""

    def __init__(self):
        self.upserted_payloads = []

    def upsert(self, collection_name, points, wait=True):
        self.upserted_payloads.extend(p.payload for p in points)


def _bare_papers_store(client):
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "papers"
    store.client = client
    return store


class TestApiRoutesWritePath:
    """pipeline/paper_crawler/integration/api_routes.py::_build_paper_point_payload
    — the single-paper reindex endpoint's payload builder."""

    def test_matches_projector_output_exactly(self):
        chunk_row = {"doc_id": 42605, "content": "hello", "level": 2,
                     "start_idx": 0, "end_idx": 5}
        paper_meta = {"paper_id": "aaai-2022-42605", "venue": "AAAI",
                      "year": 2026, "venue_tier": "CCF-A"}

        payload = _build_paper_point_payload(chunk_row, paper_meta)

        expected_raw = {
            "doc_id": chunk_row["doc_id"], "paper_id": paper_meta["paper_id"],
            "source_type": "paper", "level": chunk_row["level"],
            "venue": paper_meta["venue"], "year": paper_meta["year"],
            "venue_tier": paper_meta["venue_tier"], "content": chunk_row["content"],
        }
        assert payload == project_papers_payload(expected_raw)
        assert payload["doc_id"] == "42605"  # D2 — int SQL doc_id normalized to str

    def test_missing_venue_metadata_does_not_write_a_null(self):
        chunk_row = {"doc_id": 1, "content": "c", "level": 1, "start_idx": 0, "end_idx": 1}
        paper_meta = {"paper_id": "p1", "venue": None, "year": None, "venue_tier": None}

        payload = _build_paper_point_payload(chunk_row, paper_meta)

        assert "venue" not in payload
        assert "year" not in payload
        assert "venue_tier" not in payload

    def test_no_data_source_key_reaches_the_payload(self):
        """D3: this endpoint used to write `data_source` onto paper points —
        the stale "9-field" comment it came from is gone; must never recur."""
        chunk_row = {"doc_id": 1, "content": "c", "level": 1, "start_idx": 0, "end_idx": 1}
        paper_meta = {"paper_id": "p1", "venue": "V", "year": 2026, "venue_tier": "T"}

        payload = _build_paper_point_payload(chunk_row, paper_meta)

        assert "data_source" not in payload


class TestAddChunksWritePath:
    """core/storage/qdrant_store.py::QdrantStore.add_chunks — the only writer
    for upload/webpage, and one of two paper/review writers."""

    def test_papers_collection_payload_matches_projector(self):
        client = _RecordingClient()
        store = _bare_papers_store(client)
        chunk = Chunk(id="c1", doc_id="555", content="text", level=2,
                      start_idx=0, end_idx=4, embedding=[0.0] * 4,
                      metadata={"venue": "ICML", "year": 2026, "venue_tier": "CCF-A"})

        store.add_chunks([chunk], source_type="paper", doc_title="T", doc_source="S")

        assert len(client.upserted_payloads) == 1
        payload = client.upserted_payloads[0]
        expected_raw = {
            "doc_id": "555", "content": "text", "level": 2,
            "title": "T", "source": "S", "source_type": "paper",
            "venue": "ICML", "year": 2026, "venue_tier": "CCF-A",
        }
        assert payload == project_papers_payload(expected_raw)

    def test_arbitrary_metadata_key_does_not_leak_into_papers_payload(self):
        """The historical `**chunk.metadata` backdoor: any key in chunk.metadata
        landed in the point payload unconstrained. Must be closed for `papers`."""
        client = _RecordingClient()
        store = _bare_papers_store(client)
        chunk = Chunk(id="c1", doc_id="1", content="text", level=1,
                      start_idx=0, end_idx=4, embedding=[0.0] * 4,
                      metadata={"totally_made_up_field": "should not survive",
                                "another_leak": 123})

        store.add_chunks([chunk], source_type="paper")

        payload = client.upserted_payloads[0]
        assert "totally_made_up_field" not in payload
        assert "another_leak" not in payload

    def test_data_source_kwarg_is_dropped_for_paper_source_type(self):
        client = _RecordingClient()
        store = _bare_papers_store(client)
        chunk = Chunk(id="c1", doc_id="1", content="text", level=1,
                      start_idx=0, end_idx=4, embedding=[0.0] * 4, metadata={})

        store.add_chunks([chunk], source_type="paper", data_source="system")

        assert "data_source" not in client.upserted_payloads[0]

    def test_upload_source_type_keeps_data_source_and_business_fields(self):
        """D1: upload is not a paper — its business fields must survive the
        projector instead of being clipped to the paper whitelist."""
        client = _RecordingClient()
        store = _bare_papers_store(client)
        chunk = Chunk(id="c1", doc_id="9", content="text", level=2,
                      start_idx=0, end_idx=4, embedding=[0.0] * 4,
                      metadata={"original_filename": "report.pdf", "file_size": 100,
                                "chunk_size": 100})

        store.add_chunks([chunk], source_type="upload", data_source="user",
                          doc_title="Report", doc_source="upload")

        payload = client.upserted_payloads[0]
        assert payload["data_source"] == "user"
        assert payload["original_filename"] == "report.pdf"
        assert payload["file_size"] == 100
        assert "paper_id" not in payload

    def test_non_papers_collection_is_not_constrained(self):
        """AGENTS.md 2.1: a payload projector must not be reused across
        collections with a different semantic — `code_chunks` etc. own their
        own shape and must not go through the papers projector."""
        client = _RecordingClient()
        store = QdrantStore.__new__(QdrantStore)
        store.collection_name = "code_chunks"
        store.client = client
        chunk = Chunk(id="c1", doc_id="1", content="text", level=1,
                      start_idx=0, end_idx=4, embedding=[0.0] * 4,
                      metadata={"anything_goes": True})

        store.add_chunks([chunk])  # no source_type at all — must not raise

        assert client.upserted_payloads[0]["anything_goes"] is True


class TestLocalIngestWritePath:
    """scripts/remote_embed/local_ingest.py::canonicalize_payload — the batch
    parquet ingest path (D3 parquet, the source of the historical int doc_id)."""

    def test_matches_projector_output_exactly(self):
        from local_ingest import canonicalize_payload

        raw = {"doc_id": 42605, "content": "hello", "level": 2, "source_type": "paper",
               "paper_id": "aaai-2022-42605", "venue": "AAAI", "year": 2026,
               "venue_tier": "CCF-A"}

        out = canonicalize_payload(dict(raw))

        assert out == project_papers_payload(dict(raw))
        assert out["doc_id"] == "42605"

    def test_review_source_type_shares_paper_profile(self):
        from local_ingest import canonicalize_payload

        raw = {"doc_id": 1, "content": "c", "level": 1, "source_type": "review",
               "paper_id": "p1"}
        out = canonicalize_payload(dict(raw))
        assert out == project_papers_payload(dict(raw))

    def test_legacy_shard_missing_source_type_defaults_to_paper(self):
        """local_ingest only ever ingests the papers batch pipeline; a shard
        dumped before source_type existed is a paper, not upload/webpage."""
        from local_ingest import canonicalize_payload

        out = canonicalize_payload({"doc_id": 1, "content": "c", "level": 1})
        assert out["source_type"] == "paper"
