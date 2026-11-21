"""`embed_fp` on the `papers` collection: every vector written there says which
embedding profile produced it, for all four source types, through every live
writer, and the key is KEYWORD-indexed so a `must_not embed_fp == current`
selection does not scan 23 M points.

The writers are exercised against a real in-process Qdrant (`:memory:`), so what
is asserted is what the engine stored, not what a recording client was handed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, VectorParams,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "remote_embed"))

from app.models.schemas import Chunk
from config import settings
from core.indexing.embedding_profile import current_embed_fp
from core.storage.papers_payload import project_papers_payload
from core.storage.qdrant_store import QdrantStore

DIM = 4
FP = current_embed_fp()


@pytest.mark.parametrize("source_type", ["paper", "review", "upload", "webpage"])
def test_projector_keeps_embed_fp_for_every_source_type(source_type):
    out = project_papers_payload({"source_type": source_type, "doc_id": 1, "embed_fp": FP})
    assert out["embed_fp"] == FP


@pytest.mark.parametrize("bad", [None, "", 7, True, ["x"]])
def test_projector_drops_an_embed_fp_that_is_not_a_non_empty_string(bad):
    out = project_papers_payload({"source_type": "paper", "doc_id": 1, "embed_fp": bad})
    assert "embed_fp" not in out


def test_embed_fp_is_a_keyword_index():
    from recreate_qdrant_collection import PAYLOAD_INDEXES

    assert dict(PAYLOAD_INDEXES)["embed_fp"] == PayloadSchemaType.KEYWORD


def test_local_ingest_keeps_the_workers_embed_fp():
    from local_ingest import canonicalize_payload

    out = canonicalize_payload({"doc_id": 3, "content": "c", "level": 1,
                                "source_type": "paper", "embed_fp": FP})
    assert out["embed_fp"] == FP


@pytest.fixture()
def store():
    client = QdrantClient(":memory:")
    dense_is_named = settings.embedding.supports_sparse
    params = VectorParams(size=DIM, distance=Distance.COSINE)
    client.create_collection("papers", vectors_config={"dense": params} if dense_is_named else params)
    s = QdrantStore.__new__(QdrantStore)
    s.collection_name = "papers"
    s.client = client
    s.dense_is_named = dense_is_named
    yield s
    client.close()


class _FakeEmbedder:
    """Vectors only; the datastore behaviour under test is real."""

    def embed_chunks(self, chunks):
        for chunk in chunks:
            chunk.embedding = [0.5] * DIM
        return chunks, None


def _stamped(store, **match):
    flt = Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in match.items()])
    points, _ = store.client.scroll("papers", scroll_filter=flt, limit=100, with_payload=True)
    return points


def test_index_service_stamps_every_chunk(store):
    from unittest.mock import patch

    from app.services.index_service import IndexService

    svc = IndexService(chunker_type="markdown")
    svc._qdrant_store = store
    svc._new_embedder = _FakeEmbedder()
    text = "# Title\n\n" + "A sentence about retrieval. " * 60
    with patch.object(IndexService, "_save_chunks_to_db"):
        result = svc.index_document(doc_id=4242, content=text, title="T", source="S",
                                    source_type="upload", data_source="user")
    assert result["success"], result
    points = _stamped(store, doc_id="4242")
    assert points and all(p.payload["embed_fp"] == FP for p in points)


def test_index_service_embed_fp_is_not_overridable_by_caller_metadata(store):
    from unittest.mock import patch

    from app.services.index_service import IndexService

    svc = IndexService(chunker_type="markdown")
    svc._qdrant_store = store
    svc._new_embedder = _FakeEmbedder()
    with patch.object(IndexService, "_save_chunks_to_db"):
        svc.index_document(doc_id=4243, content="# T\n\n" + "word " * 80, title="T", source="S",
                           source_type="upload", metadata={"embed_fp": "stale"})
    points = _stamped(store, doc_id="4243")
    assert points and {p.payload["embed_fp"] for p in points} == {FP}


def test_paper_indexer_stamps_every_chunk(store):
    from types import SimpleNamespace
    from unittest.mock import patch

    from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig, PaperIndexer

    class _Db:
        def add(self, doc):
            self.doc = doc

        def flush(self):
            self.doc.id = 7

    indexer = PaperIndexer(PaperIndexConfig())
    indexer.index_service._qdrant_store = store
    chunks = [Chunk(id=f"c{i}", doc_id="", content=f"chunk {i}", level=2, start_idx=0,
                    end_idx=7, embedding=[0.5] * DIM, metadata={}) for i in range(3)]
    paper = SimpleNamespace(title="P", doi=None, arxiv_id=None, pdf_url=None, url="u")
    prepared = {"paper": paper, "markdown_path": None, "content": "x", "metadata": {},
                "vector_metadata": {"venue": "ICML", "year": 2026}}
    with patch.object(indexer.index_service, "_save_chunks_to_db"), \
         patch.object(PaperIndexer, "_upsert_paper_row"), \
         patch.object(PaperIndexer, "_paper_url", return_value="u"):
        indexer._write_paper_to_stores(prepared, chunks, _Db())
    points = _stamped(store, doc_id="7")
    assert len(points) == 3 and all(p.payload["embed_fp"] == FP for p in points)
