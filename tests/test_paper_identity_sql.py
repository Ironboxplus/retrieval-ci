"""papers.doc_id is the one paper -> document mapping; the new papers columns migrate.

Real SQLite files with the application's tables. The Qdrant leg (delete_paper
removing a paper's vectors) runs against a scratch collection on the local server.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from core.db import database
from core.db.database import Base
from core.db.models import Document as DBDocument, Paper as DBPaper, SourceType
from pipeline.paper_crawler.integration import index_integration
from pipeline.paper_crawler.integration.index_integration import (
    PaperIndexConfig,
    PaperIndexer,
    PaperSearchResult,
    existing_doc_id,
)
from tests.qdrant_scratch import drop, scratch_names
from tests.sqlite_task_db import make_session_factory


def columns(engine, table: str) -> dict[str, tuple]:
    with engine.connect() as conn:
        return {row[1]: row for row in conn.execute(text(f"PRAGMA table_info({table})"))}


def test_migration_adds_the_papers_columns_to_a_legacy_table_and_is_idempotent(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # A database from before the columns existed, holding one paper.
        conn.execute(text("INSERT INTO documents (id, title, source, source_type, status) "
                          "VALUES (1, 't', 's', 'paper', 'indexed')"))
        conn.execute(text("INSERT INTO papers (paper_id, doc_id, title) VALUES ('arxiv-2601.00001', 1, 't')"))
        for column in ("source_version", "withdrawn", "withdrawn_at"):
            conn.execute(text(f"ALTER TABLE papers DROP COLUMN {column}"))
    assert not {"source_version", "withdrawn", "withdrawn_at"} & set(columns(engine, "papers"))

    monkeypatch.setattr(database, "engine", engine)
    database._apply_idempotent_migrations()
    database._apply_idempotent_migrations()  # a second start changes nothing and does not fail

    cols = columns(engine, "papers")
    assert {"source_version", "withdrawn", "withdrawn_at"} <= set(cols)
    assert cols["withdrawn"][3] == 1  # NOT NULL
    with engine.connect() as conn:
        row = conn.execute(text("SELECT source_version, withdrawn, withdrawn_at FROM papers")).one()
    assert tuple(row) == (None, 0, None)  # existing rows read as "not withdrawn"
    engine.dispose()


def paper_stub(paper_id: str, title: str = "T"):
    return SimpleNamespace(id=paper_id, title=title, abstract="a", authors=["A"], arxiv_id=None, doi=None,
                           pdf_path=None, venue_tier="preprint", track_type="preprint")


def seed(db, paper_id: str) -> int:
    doc = DBDocument(title="t", source="s", source_type=SourceType.paper)
    db.add(doc)
    db.flush()
    db.add(DBPaper(paper_id=paper_id, doc_id=doc.id, title="t"))
    db.flush()
    return doc.id


def test_existing_doc_id_reads_papers_and_a_reindex_repoints_the_row(tmp_path):
    factory = make_session_factory(tmp_path / "ident.db")
    indexer = PaperIndexer(PaperIndexConfig(collection_name="unused_test"))
    with factory() as db:
        assert existing_doc_id(db, "arxiv-2601.00001") is None
        old = seed(db, "arxiv-2601.00001")
        assert existing_doc_id(db, "arxiv-2601.00001") == old

        new_doc = DBDocument(title="t2", source="s", source_type=SourceType.paper)
        db.add(new_doc)
        db.flush()
        row = indexer._upsert_paper_row(db, new_doc.id, paper_stub("arxiv-2601.00001", "T2"),
                                        {"year": 2026, "conference": "arXiv"}, None)
        db.flush()
        assert row.doc_id == new_doc.id and row.title == "T2"
        assert existing_doc_id(db, "arxiv-2601.00001") == new_doc.id
        assert db.query(DBPaper).count() == 1  # re-pointed, not duplicated


def test_untiered_paper_takes_its_registry_tier_not_ccf_a(tmp_path):
    """IJCAI 2026 reached SQL as CCF-A through the old blanket default."""
    factory = make_session_factory(tmp_path / "tier.db")
    indexer = PaperIndexer(PaperIndexConfig(collection_name="unused_test"))
    cases = {"ijcai-2026-0001": ("IJCAI", "CCF-B"), "eccv-2024-00004": ("ECCV", "CCF-B"),
             "tpami-2024-3356232": ("TPAMI", "CCF-A"), "madeup-2024-1": ("MadeUpVenue9999", None)}
    with factory() as db:
        for paper_id, (venue, _) in cases.items():
            doc = DBDocument(title="t", source="s", source_type=SourceType.paper)
            db.add(doc)
            db.flush()
            stub = paper_stub(paper_id)
            stub.venue_tier, stub.track_type = None, None
            indexer._upsert_paper_row(db, doc.id, stub, {"year": 2024, "conference": venue}, None)
        db.flush()
        tiers = {row.paper_id: row.venue_tier for row in db.query(DBPaper)}
    assert tiers == {paper_id: tier for paper_id, (_, tier) in cases.items()}


def test_indexer_lookups_use_sql_not_a_side_file(tmp_path, monkeypatch):
    factory = make_session_factory(tmp_path / "lookups.db")
    monkeypatch.setattr(index_integration, "get_db_session", factory)
    with factory() as db:
        doc = seed(db, "arxiv-2601.00002")
        db.query(DBDocument).filter(DBDocument.id == doc).update({"status": "indexed"})
    indexer = PaperIndexer(PaperIndexConfig(collection_name="unused_test"))
    assert not hasattr(indexer, "_paper_doc_mapping")
    assert indexer.get_indexed_doc_id("arxiv-2601.00002") == str(doc)
    assert indexer.get_indexed_doc_id("arxiv-2601.99999") is None
    assert indexer.get_indexed_paper_ids() == ["arxiv-2601.00002"]
    response = indexer.to_search_response("q", [
        PaperSearchResult(paper_id="arxiv-2601.00002", title="t", authors=[], abstract="", source="s",
                          conference="arXiv", year=2026, paper_type="x", score=1.0, hit_count=1),
        PaperSearchResult(paper_id="unknown-1", title="u", authors=[], abstract="", source="s",
                          conference="arXiv", year=2026, paper_type="x", score=1.0, hit_count=1),
    ])
    assert [a["doc_id"] for a in response["aggregated"]] == [str(doc), "unknown-1"]


@pytest.fixture()
def scratch_qdrant():
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url="http://127.0.0.1:6333", timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant: {exc}")
    names = scratch_names("identity", roles=("papers",))
    yield names["papers"]
    drop(client, names.values())


def test_delete_paper_removes_the_vectors_and_the_rows(tmp_path, monkeypatch, scratch_qdrant):
    from app.models.schemas import Chunk

    factory = make_session_factory(tmp_path / "delete.db")
    monkeypatch.setattr(index_integration, "get_db_session", factory)
    indexer = PaperIndexer(PaperIndexConfig(collection_name=scratch_qdrant))
    with factory() as db:
        doc = seed(db, "arxiv-2601.00003")
        other = seed(db, "arxiv-2601.00004")
    store = indexer.index_service.chroma_store
    store.add_chunks([
        Chunk(id=f"{d}-{i}", doc_id=str(d), content=f"text {d} {i}", level=2, start_idx=0, end_idx=5,
              metadata={"title": "t"}, embedding=[0.1 * (i + 1)] * 1024)
        for d in (doc, other) for i in range(2)
    ], source_type="paper")
    assert store.count_by_doc_id(str(doc)) == 2

    assert indexer.delete_paper("arxiv-2601.00003") is True
    assert store.count_by_doc_id(str(doc)) == 0 and store.count_by_doc_id(str(other)) == 2
    with factory() as db:
        assert existing_doc_id(db, "arxiv-2601.00003") is None
        assert db.get(DBDocument, doc) is None
        assert existing_doc_id(db, "arxiv-2601.00004") == other
    assert indexer.delete_paper("arxiv-2601.00003") is False
