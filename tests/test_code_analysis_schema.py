"""Schema and migration for the duplicate-code analysis tables.

A deployed database already holds `index_catalog`; create_all never alters an existing
table, so the new column has to arrive through `_apply_idempotent_migrations`, before
the code that reads it starts. The new tables are created there too, so the migration
entry point alone is enough to bring any database up to date.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

import core.db.database as database
from core.db import models  # noqa: F401 -- registers every table on Base


@pytest.fixture()
def legacy_engine(tmp_path, monkeypatch):
    """A real SQLite file shaped like a deployed DB from before this feature."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'legacy.db'}", connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE code_scope_analyses"))
        conn.execute(text("DROP TABLE code_duplicate_pairs"))
        conn.execute(text("ALTER TABLE index_catalog DROP COLUMN content_changed_at"))
        conn.execute(text(
            "INSERT INTO index_catalog (owner_user_id, host, project, rel_path, kind, "
            "chunk_count, content_hash) VALUES ('o', 'h', '/repo', 'a.py', 'code', 1, 'x')"))
    monkeypatch.setattr(database, "engine", engine)
    yield engine
    engine.dispose()


def _columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        return {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}


def _tables(engine) -> set[str]:
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'"))}


def test_the_migration_adds_content_changed_at_without_touching_existing_rows(legacy_engine):
    assert "content_changed_at" not in _columns(legacy_engine, "index_catalog")
    database._apply_idempotent_migrations()
    assert "content_changed_at" in _columns(legacy_engine, "index_catalog")
    with legacy_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT rel_path, content_changed_at FROM index_catalog")).one()
    # No backfill: an existing file is "not known to have changed", not "changed now".
    assert tuple(row) == ("a.py", None)


def test_the_migration_creates_the_analysis_tables(legacy_engine):
    assert not {"code_scope_analyses", "code_duplicate_pairs"} & _tables(legacy_engine)
    database._apply_idempotent_migrations()
    assert {"code_scope_analyses", "code_duplicate_pairs"} <= _tables(legacy_engine)
    assert {"owner_user_id", "project", "kind", "watermark", "version",
            "delivered_version", "last_run_at", "summary_json"} <= _columns(
                legacy_engine, "code_scope_analyses")
    assert {"owner_user_id", "project", "path_a", "path_b", "shared_chunks", "max_score",
            "examples_json", "first_seen_at", "updated_at"} <= _columns(
                legacy_engine, "code_duplicate_pairs")


def test_the_migration_is_safe_to_run_twice(legacy_engine):
    database._apply_idempotent_migrations()
    database._apply_idempotent_migrations()
    assert "content_changed_at" in _columns(legacy_engine, "index_catalog")


def test_init_db_brings_a_legacy_database_up_to_date(legacy_engine):
    database.init_db()
    assert "content_changed_at" in _columns(legacy_engine, "index_catalog")
    assert {"code_scope_analyses", "code_duplicate_pairs"} <= _tables(legacy_engine)


def test_a_scope_has_one_analysis_row_and_one_row_per_file_pair(legacy_engine):
    database._apply_idempotent_migrations()
    with legacy_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO code_scope_analyses (owner_user_id, project, kind, version, "
            "delivered_version) VALUES ('o', '/repo', 'duplicates', 0, 0)"))
        conn.execute(text(
            "INSERT INTO code_duplicate_pairs (owner_user_id, project, path_a, path_b, "
            "shared_chunks, max_score, first_seen_at, updated_at) VALUES "
            "('o', '/repo', 'a.py', 'b.py', 2, 0.9, '2026-01-01', '2026-01-01')"))
    with pytest.raises(Exception, match="UNIQUE"):
        with legacy_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO code_scope_analyses (owner_user_id, project, kind, version, "
                "delivered_version) VALUES ('o', '/repo', 'duplicates', 1, 0)"))
    with pytest.raises(Exception, match="UNIQUE"):
        with legacy_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO code_duplicate_pairs (owner_user_id, project, path_a, path_b, "
                "shared_chunks, max_score, first_seen_at, updated_at) VALUES "
                "('o', '/repo', 'a.py', 'b.py', 3, 0.95, '2026-01-01', '2026-01-01')"))
