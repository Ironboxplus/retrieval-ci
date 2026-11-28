"""Prepare or remove a private real-data Qdrant/SQLite match-paper fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.models.schemas import Chunk as SearchChunk
from app.services.match_doc.adapters import chunks_from_text
from core.db.database import get_db_session, init_db
from core.db.models import (
    Chunk,
    DataSource,
    Document,
    DocumentStatus,
    Paper,
    SourceType,
)
from core.indexing.chunker import create_chunker
from core.indexing.embedders import get_embedder
from core.storage.qdrant_store import QdrantStore


PRIVATE_COLLECTION = "test_match_paper_job_real_gate"
SOURCE_PAPER_ID = "test-real-acl-source"


def _real_chunks(path: Path, limit: int) -> list[str]:
    markdown = path.read_text(encoding="utf-8")
    chunks = chunks_from_text(
        markdown,
        chunker=create_chunker(),
        max_chunks=limit,
        content_type="markdown",
    )
    if len(chunks) < limit:
        raise RuntimeError(f"{path} produced only {len(chunks)} chunks")
    return [chunk.text for chunk in chunks]


def _reset_sql(source_chunks: list[str]) -> None:
    init_db()
    with get_db_session() as session:
        existing = session.query(Paper).filter(Paper.paper_id == SOURCE_PAPER_ID).first()
        if existing is not None:
            session.delete(existing.document)
            session.flush()
        document = Document(
            title="Real ACL source fixture",
            source="acl-2026-long-1.md",
            source_type=SourceType.paper,
            data_source=DataSource.system,
            status=DocumentStatus.indexed,
            metadata_json={"test_fixture": True},
        )
        session.add(document)
        session.flush()
        session.add(Paper(
            paper_id=SOURCE_PAPER_ID,
            doc_id=document.id,
            title="Real ACL source fixture",
            year=2026,
            venue="ACL",
            track_type="main",
            venue_tier="CCF-A",
        ))
        for index, content in enumerate(source_chunks):
            session.add(Chunk(
                doc_id=document.id,
                content=content,
                level=2,
                chunk_index=index,
                metadata_json={"paper_id": SOURCE_PAPER_ID},
            ))


def _reset_qdrant(source_chunks: list[str], other_chunks: list[str]) -> int:
    probe = QdrantStore(PRIVATE_COLLECTION)
    if probe.client.collection_exists(PRIVATE_COLLECTION):
        probe.client.delete_collection(PRIVATE_COLLECTION)
    store = QdrantStore(PRIVATE_COLLECTION)

    rows: list[tuple[str, str, str, str]] = []
    for index, text in enumerate(source_chunks):
        rows.append((SOURCE_PAPER_ID, "1", f"source-{index}", text))
        rows.append(("test-real-acl-neighbour", "2", f"acl-neighbour-{index}", text))
    for index, text in enumerate(other_chunks):
        rows.append(("test-real-kdd-neighbour", "3", f"kdd-neighbour-{index}", text))

    embedder = get_embedder()
    embedded = embedder.embed_texts([row[3] for row in rows])
    chunks = [
        SearchChunk(
            id=chunk_id,
            doc_id=doc_id,
            content=text,
            level=2,
            start_idx=0,
            end_idx=len(text),
            embedding=embedded.dense[index].tolist(),
            metadata={
                "paper_id": paper_id,
                "source_type": "paper",
                "venue": "ACL" if "acl" in paper_id else "KDD",
                "year": 2026,
                "venue_tier": "CCF-A",
            },
        )
        for index, (paper_id, doc_id, chunk_id, text) in enumerate(rows)
    ]
    store.add_chunks(
        chunks,
        source_type="paper",
        data_source="system",
        wait=True,
        sparse_vectors=embedded.sparse,
    )
    return store.count()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acl-markdown", type=Path)
    parser.add_argument("--kdd-markdown", type=Path)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    store = QdrantStore(PRIVATE_COLLECTION)
    if args.cleanup:
        if store.client.collection_exists(PRIVATE_COLLECTION):
            store.client.delete_collection(PRIVATE_COLLECTION)
        print(json.dumps({"removed": PRIVATE_COLLECTION}))
        return
    if not args.acl_markdown or not args.kdd_markdown:
        raise SystemExit("--acl-markdown and --kdd-markdown are required")
    source_chunks = _real_chunks(args.acl_markdown, args.chunks)
    other_chunks = _real_chunks(args.kdd_markdown, args.chunks)
    _reset_sql(source_chunks)
    count = _reset_qdrant(source_chunks, other_chunks)
    print(json.dumps({
        "collection": PRIVATE_COLLECTION,
        "points": count,
        "paper_id": SOURCE_PAPER_ID,
        "query_chunks": len(source_chunks),
    }))


if __name__ == "__main__":
    main()
