"""TDD schema 断言：方案 X 砸了重来 — paper_id 做 PK，删 DBPaper.id 自增列。

执行迁移前所有用例 RED，迁移后全 GREEN。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "retrieval.db"


def _columns(conn: sqlite3.Connection, table: str):
    return {row[1]: row for row in conn.execute(f"PRAGMA table_info({table})")}


def _index_list(conn: sqlite3.Connection, table: str):
    return list(conn.execute(f"PRAGMA index_list({table})"))


@pytest.fixture
def conn():
    assert DB_PATH.exists(), f"DB not found: {DB_PATH}"
    c = sqlite3.connect(DB_PATH)
    try:
        yield c
    finally:
        c.close()


class TestPapersSchema:
    def test_papers_table_exists(self, conn):
        rows = list(conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'"))
        assert rows, "papers table missing"

    def test_papers_id_column_dropped(self, conn):
        cols = _columns(conn, "papers")
        assert "id" not in cols, (
            "papers.id (autoincrement int) must be DROPPED. Current columns: " + str(list(cols))
        )

    def test_paper_id_is_primary_key(self, conn):
        cols = _columns(conn, "papers")
        assert "paper_id" in cols, "paper_id column missing"
        # PRAGMA table_info: row = (cid, name, type, notnull, dflt, pk)
        pk_flag = cols["paper_id"][5]
        notnull_flag = cols["paper_id"][3]
        assert pk_flag == 1, f"paper_id must be PRIMARY KEY (pk flag=1), got pk={pk_flag}"
        assert notnull_flag == 1, "paper_id must be NOT NULL (PK implies it)"

    def test_doc_id_is_not_null_unique_fk(self, conn):
        cols = _columns(conn, "papers")
        assert "doc_id" in cols, "doc_id column missing"
        notnull_flag = cols["doc_id"][3]
        assert notnull_flag == 1, f"doc_id must be NOT NULL after migration, got notnull={notnull_flag}"

        # UNIQUE constraint check via index_list
        idxs = _index_list(conn, "papers")
        doc_id_unique = False
        for idx in idxs:
            # idx = (seq, name, unique, origin, partial)
            if idx[2] == 1:  # unique
                idx_cols = list(conn.execute(f"PRAGMA index_info({idx[1]})"))
                col_names = [r[2] for r in idx_cols]
                if col_names == ["doc_id"]:
                    doc_id_unique = True
                    break
        assert doc_id_unique, f"doc_id must have UNIQUE index. Found indexes: {idxs}"

    def test_doc_id_is_foreign_key_to_documents(self, conn):
        fks = list(conn.execute("PRAGMA foreign_key_list(papers)"))
        # Each row: (id, seq, table, from, to, on_update, on_delete, match)
        doc_id_fk = [fk for fk in fks if fk[3] == "doc_id"]
        assert doc_id_fk, f"doc_id must be FK. Existing FKs: {fks}"
        fk = doc_id_fk[0]
        assert fk[2] == "documents", f"doc_id FK target must be documents, got {fk[2]}"
        assert fk[4] == "id", f"doc_id FK references documents.id, got {fk[4]}"
        assert fk[6] in ("CASCADE", "RESTRICT", "NO ACTION", "SET NULL", "SET DEFAULT"), (
            f"unexpected on_delete: {fk[6]}"
        )

    def test_row_count_preserved(self, conn):
        # The paper_id-PK migration baseline was 44036 rows; the corpus only grows via
        # ingest, so assert no data loss below that floor (exact count is not pinned).
        count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        if count == 0:
            pytest.skip("requires a populated local paper corpus")
        assert count >= 44036, f"papers row count regressed below migration baseline: {count}"

    def test_paper_id_values_unique(self, conn):
        dupes = conn.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT paper_id) FROM papers"
        ).fetchone()[0]
        assert dupes == 0, f"paper_id must be unique, found {dupes} duplicates"

    def test_doc_id_values_unique(self, conn):
        dupes = conn.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT doc_id) FROM papers"
        ).fetchone()[0]
        assert dupes == 0, f"doc_id must be unique, found {dupes} duplicates"

    def test_all_doc_ids_reference_existing_documents(self, conn):
        orphans = conn.execute(
            "SELECT COUNT(*) FROM papers p LEFT JOIN documents d ON p.doc_id = d.id WHERE d.id IS NULL"
        ).fetchone()[0]
        assert orphans == 0, f"all papers.doc_id must reference an existing documents.id, found {orphans} orphans"

    def test_business_columns_preserved(self, conn):
        cols = set(_columns(conn, "papers").keys())
        required = {
            "doc_id", "paper_id", "arxiv_id", "doi", "title", "abstract",
            "authors_json", "year", "venue", "paper_source", "pdf_path",
            "txt_path", "citation_count", "track_type", "impact_level",
            "venue_tier", "venue_source",
        }
        missing = required - cols
        assert not missing, f"business columns missing after migration: {missing}"

    def test_paper_id_index_exists_for_lookup(self, conn):
        idxs = _index_list(conn, "papers")
        # PK is itself the index when paper_id is PRIMARY KEY; explicit check
        # via sqlite auto-index sqlite_autoindex_papers_1 covering paper_id
        has_pk_index = False
        for idx in idxs:
            idx_cols = list(conn.execute(f"PRAGMA index_info({idx[1]})"))
            col_names = [r[2] for r in idx_cols]
            if col_names == ["paper_id"]:
                has_pk_index = True
                break
        assert has_pk_index, f"paper_id needs covering index (PK auto-index counts). Indexes: {idxs}"


class TestORMAlignment:
    def test_orm_paper_id_is_primary_key(self):
        from core.db.models import Paper as DBPaper
        # SQLAlchemy: primary_key columns
        pk_cols = [c.name for c in DBPaper.__table__.primary_key.columns]
        assert pk_cols == ["paper_id"], f"ORM PK must be ['paper_id'], got {pk_cols}"

    def test_orm_papers_has_no_id_attribute_or_column(self):
        from core.db.models import Paper as DBPaper
        col_names = [c.name for c in DBPaper.__table__.columns]
        assert "id" not in col_names, f"ORM Paper still has 'id' column: {col_names}"

    def test_orm_doc_id_is_not_nullable(self):
        from core.db.models import Paper as DBPaper
        doc_id_col = DBPaper.__table__.c.doc_id
        assert doc_id_col.nullable is False, "ORM doc_id must be NOT NULL"

    def test_orm_doc_id_has_unique_constraint(self):
        from core.db.models import Paper as DBPaper
        doc_id_col = DBPaper.__table__.c.doc_id
        assert doc_id_col.unique is True, "ORM doc_id must be UNIQUE"
