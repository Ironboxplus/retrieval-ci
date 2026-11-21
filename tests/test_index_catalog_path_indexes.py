"""The path-range indexes reach an already-deployed database through the idempotent migration.

``Base.metadata.create_all`` never adds an index to an existing table, so a production
catalog only gets ``ix_index_catalog_code_scope_path`` / ``..._lower_path`` /
``..._odd_path`` from ``_apply_idempotent_migrations``. Real SQLite file; the queries are
the repository's own, EXPLAINed after the migration ran.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.db import database
from core.db.models import IndexCatalog

NEW_INDEXES = (
    "ix_index_catalog_code_path_lookup",
    "ix_index_catalog_code_scope_path",
    "ix_index_catalog_code_scope_lower_path",
    "ix_index_catalog_code_odd_path",
    "ix_index_catalog_code_odd_path_any_scope",
)


@pytest.fixture()
def deployed_database(tmp_path, monkeypatch):
    """A catalog as it exists in production before this change: no new indexes."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'retrieval.db'}", connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for name in NEW_INDEXES:
            conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
    monkeypatch.setattr(database, "engine", engine)
    yield engine
    engine.dispose()


def _index_names(engine):
    with engine.connect() as conn:
        return {row[1] for row in conn.execute(text("PRAGMA index_list(index_catalog)"))}


def test_migration_creates_every_path_index_and_is_idempotent(deployed_database):
    assert not set(NEW_INDEXES) & _index_names(deployed_database)
    database._apply_idempotent_migrations()
    database._apply_idempotent_migrations()
    assert set(NEW_INDEXES) <= _index_names(deployed_database)


def test_fresh_databases_get_the_same_indexes_from_the_models(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    database.Base.metadata.create_all(engine)
    assert set(NEW_INDEXES) <= _index_names(engine)
    engine.dispose()


def test_migrated_database_serves_prefix_and_legacy_lookups_from_those_indexes(
        deployed_database):
    from app.services.index_catalog.repository import IndexCatalogRepository
    from contextlib import contextmanager

    database._apply_idempotent_migrations()
    session_local = sessionmaker(bind=deployed_database, expire_on_commit=False)

    @contextmanager
    def sessions():
        with session_local() as session:
            yield session
            session.commit()

    with sessions() as session:
        session.add_all([
            IndexCatalog(owner_user_id="o", host="h", project="/p", root_dir="/p",
                         rel_path=path, kind="code", chunk_count=1)
            for path in ["app/a.py", "app/b.py", "other/c.py", "app//legacy.py"]])
    statements = []
    from sqlalchemy import event

    def record(conn, cursor, statement, parameters, context, executemany):
        if "index_catalog" in statement:
            statements.append((statement, tuple(parameters)))

    event.listen(deployed_database, "before_cursor_execute", record)
    try:
        found, _count = IndexCatalogRepository(session_factory=sessions).matching_code_paths(
            owner_user_id="o", project="/p", pattern=None, case_sensitive=True,
            path_prefix="app")
    finally:
        event.remove(deployed_database, "before_cursor_execute", record)
    assert found == ["app//legacy.py", "app/a.py", "app/b.py"]
    plans = []
    with deployed_database.connect() as conn:
        for statement, parameters in statements:
            if statement.lstrip().upper().startswith("SELECT"):
                plans.extend(row[3] for row in conn.connection.driver_connection.execute(
                    "EXPLAIN QUERY PLAN " + statement, parameters))
    assert any("ix_index_catalog_code_scope_path" in line for line in plans), plans
    assert any("ix_index_catalog_code_odd_path" in line for line in plans), plans


def test_database_without_the_partial_indexes_still_answers_correctly(deployed_database):
    """Before the migration has run on a deployed database the lookups must degrade to a
    slower plan, never to an error or a missed row."""
    from app.services.index_catalog.repository import IndexCatalogRepository
    from contextlib import contextmanager

    session_local = sessionmaker(bind=deployed_database, expire_on_commit=False)

    @contextmanager
    def sessions():
        with session_local() as session:
            yield session
            session.commit()

    with sessions() as session:
        session.add_all([
            IndexCatalog(owner_user_id="o", host="h", project="/p", root_dir="/p",
                         rel_path=path, kind="code", chunk_count=1)
            for path in ["app/a.py", "other/c.py", "./app/dot.py", "app//legacy.py"]])
    repository = IndexCatalogRepository(session_factory=sessions)
    assert not {"ix_index_catalog_code_odd_path"} & _index_names(deployed_database)
    for project in ("/p", None):
        found, _count = repository.matching_code_paths(
            owner_user_id="o", project=project, pattern=None, case_sensitive=True,
            path_prefix="app")
        assert found == ["./app/dot.py", "app//legacy.py", "app/a.py"]
