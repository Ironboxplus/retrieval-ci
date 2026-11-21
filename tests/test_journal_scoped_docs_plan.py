"""A scope's doc titles are read through that scope's rows, not every doc of every scope.

Real SQLite, production schema, no ``sqlite_stat1`` (production never ran ANALYZE).
Journal search of a subdirectory reads the parent's doc titles below the prefix twice
(the docs and ``.conversations/``). With only single-column indexes the planner drove the
join from ``ix_journal_entries_source_kind`` and walked every live doc of every project:
on the host (17,684 entries) each read took ~37 ms instead of ~1 ms, ~75 ms per search.
"""
from __future__ import annotations

import pytest
from sqlalchemy import event, insert, text

from app.services.journal.repository import JournalRepository
from core.db.models import JournalEntry, JournalProject
from tests.catalog_sqlite import CatalogDb

OWNER = "internal-admin"
INDEX = "ix_journal_entries_project_kind"


def _seed(db: CatalogDb, *, projects=40, docs=200) -> None:
    with db.engine.begin() as conn:
        conn.execute(insert(JournalProject), [
            dict(project_id=p + 1, slug=f"C:/work/p{p}", display_name=f"p{p}",
                 owner_user_id=OWNER) for p in range(projects)])
        conn.execute(insert(JournalEntry), [
            dict(entry_id=f"e{p}-{i}", project_id=p + 1, client="mcp", source_kind="doc",
                 title=f"docs/f{i}.md" if i % 2 else f"src/f{i}.md", body="x",
                 content_hash=f"{p}-{i}") for p in range(projects) for i in range(docs)])


def _entry_plans(db: CatalogDb, call) -> list[str]:
    statements: list[tuple[str, tuple]] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if "FROM journal_entries JOIN journal_projects" in statement:
            statements.append((statement, tuple(parameters)))

    event.listen(db.engine, "before_cursor_execute", record)
    try:
        call()
    finally:
        event.remove(db.engine, "before_cursor_execute", record)
    with db.engine.connect() as conn:
        raw = conn.connection.driver_connection
        return [row[3] for statement, parameters in statements
                for row in raw.execute("EXPLAIN QUERY PLAN " + statement, parameters)]


def _titles(repository: JournalRepository, prefix: str, spellings: int = 3):
    # A Windows scope with legacy case variants passes all its spellings; from three
    # on, the planner without statistics stopped starting from journal_projects.
    slugs = ["C:/work/p7", "C:/Work/p7", "C:/WORK/p7", "c:/work/p7"][:spellings]
    return repository.scoped_live_doc_id_titles(
        project_slug=slugs, owner_user_id=OWNER,
        case_sensitive=False, path_prefix=prefix)


@pytest.mark.parametrize("spellings", [1, 2, 3, 4])
def test_prefix_doc_titles_are_read_from_the_scope_rows_only(spellings):
    db = CatalogDb()
    _seed(db)
    repository = JournalRepository(session_factory=db.session)
    with db.engine.connect() as conn:
        assert not conn.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'").fetchall()

    _titles(repository, "docs")  # the first session builds the FTS mirror
    rows = []
    plan = _entry_plans(db, lambda: rows.extend(_titles(repository, "docs", spellings)))

    assert len(rows) == 100 and all(title.startswith("docs/") for _, title in rows)
    # Which index the planner picks is SQLite's choice and moves between versions; what
    # matters is that journal_entries is entered through a project_id index search and
    # never walked whole.
    entry_access = [line for line in plan
                    if line.startswith(("SEARCH journal_entries ", "SCAN journal_entries"))]
    assert entry_access, plan
    assert not [line for line in entry_access if line.startswith("SCAN")], plan
    assert all("USING" in line and "INDEX" in line and "project_id=?" in line
               for line in entry_access), plan


def test_existing_database_gets_the_index(tmp_path, monkeypatch):
    from core.db import database

    old = CatalogDb(path=tmp_path / "old.db")
    with old.engine.begin() as conn:
        conn.execute(text(f"DROP INDEX {INDEX}"))
    _seed(old, projects=8, docs=4)
    monkeypatch.setattr(database, "engine", old.engine)
    database._apply_idempotent_migrations()
    database._apply_idempotent_migrations()  # re-runnable
    with old.engine.connect() as conn:
        assert conn.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name = ?",
            (INDEX,)).fetchall()
    assert len(_titles(JournalRepository(session_factory=old.session), "docs")) == 2
