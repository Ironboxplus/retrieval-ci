"""index_catalog.content_changed_at: the durable "this file's content changed" signal.

`indexed_at` is refreshed by every report, so it cannot say whether a file changed.
Duplicate-code analysis (app/services/code_kb/analysis.py) needs exactly that: "which
files changed since the last analysis", surviving restarts and merging many index runs.
Real SQLite through the real repository; nothing here is faked.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import text

from app.services.index_catalog.schemas import IndexFileRecord, IndexReportRequest
from core.db.models import IndexCatalog
from tests.catalog_sqlite import CatalogDb

OWNER = "owner"
PROJECT = "/repo"


@pytest.fixture()
def db():
    return CatalogDb()


def _file(path: str, content_hash: str | None = "h1") -> IndexFileRecord:
    return IndexFileRecord(
        rel_path=path, kind="code", collection="code_chunks", language="python",
        chunk_count=2, content_hash=content_hash)


def _report(db: CatalogDb, files: list[IndexFileRecord], *, replace: bool = False,
            project: str = PROJECT) -> None:
    db.service.report(
        IndexReportRequest(host="h", project=project, root_dir=project, files=files,
                           replace=replace),
        owner_user_id=OWNER)


def _row(db: CatalogDb, path: str, project: str = PROJECT) -> IndexCatalog:
    with db.session() as session:
        row = session.query(IndexCatalog).filter_by(
            owner_user_id=OWNER, project=project, rel_path=path).one()
        session.expunge(row)
        return row


def test_a_new_file_is_a_content_change(db):
    _report(db, [_file("a.py")])
    assert _row(db, "a.py").content_changed_at is not None


def test_an_unchanged_report_does_not_touch_it(db):
    _report(db, [_file("a.py", "h1")])
    first = _row(db, "a.py")
    _report(db, [_file("a.py", "h1")])
    second = _row(db, "a.py")
    assert second.content_changed_at == first.content_changed_at
    # indexed_at still moves on every report, which is exactly why it cannot be the signal.
    assert second.indexed_at >= first.indexed_at


def test_a_different_content_hash_is_a_content_change(db):
    _report(db, [_file("a.py", "h1")])
    with db.session() as session:
        session.query(IndexCatalog).filter_by(rel_path="a.py").update(
            {"content_changed_at": datetime(2026, 1, 1)})
    _report(db, [_file("a.py", "h2")])
    assert _row(db, "a.py").content_changed_at > datetime(2026, 6, 1)


def test_restoring_a_soft_deleted_row_is_a_content_change(db):
    _report(db, [_file("a.py", "h1")])
    with db.session() as session:
        session.query(IndexCatalog).filter_by(rel_path="a.py").update(
            {"content_changed_at": datetime(2026, 1, 1), "deleted_at": datetime(2026, 2, 1)})
    # Same hash as before the deletion: the file is back, so its duplicates are back too.
    _report(db, [_file("a.py", "h1")])
    row = _row(db, "a.py")
    assert row.deleted_at is None
    assert row.content_changed_at > datetime(2026, 6, 1)


def test_only_the_changed_file_of_a_batch_is_marked(db):
    _report(db, [_file("a.py", "a1"), _file("b.py", "b1")])
    with db.session() as session:
        session.query(IndexCatalog).update({"content_changed_at": datetime(2026, 1, 1)})
    _report(db, [_file("a.py", "a2"), _file("b.py", "b1")])
    assert _row(db, "a.py").content_changed_at > datetime(2026, 6, 1)
    assert _row(db, "b.py").content_changed_at == datetime(2026, 1, 1)


def test_a_windows_case_variant_report_of_an_unchanged_file_is_not_a_change(db):
    project = "E:/work/repo"
    _report(db, [_file("Src/A.py", "h1")], project=project)
    with db.session() as session:
        session.query(IndexCatalog).update({"content_changed_at": datetime(2026, 1, 1)})
    _report(db, [_file("src/a.py", "h1")], project=project)
    assert _row(db, "src/a.py", project=project).content_changed_at == datetime(2026, 1, 1)


def test_the_column_is_part_of_the_table_the_migration_patches(db):
    with db.engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(index_catalog)"))}
    assert "content_changed_at" in columns
