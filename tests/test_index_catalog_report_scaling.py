"""Catalog writes must scale with what a report names, not with the scope size.

``report_files`` used to load every row of (owner, host, project) as ORM objects for
every report, including a one-file incremental ``replace=False`` report sent from the
Code watcher while it holds the owner's search fence. These tests run the real
repository on real SQLite and prove (1) the observable result is identical to the
previous whole-scope implementation, kept below as the oracle, and (2) the number of
catalog rows materialised no longer depends on the scope size.
"""
from __future__ import annotations

import random
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.index_catalog.repository import (
    IndexCatalogRepository, _within_sweep_envelope, _utcnow,
)
from app.services.scope_paths import is_windows_scope
from core.db.database import Base
from core.db.models import IndexCatalog


def _engine_and_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return engine, factory


def _legacy_report_files(factory, *, owner_user_id, host, project, root_dir, files,
                         replace=True, patterns=None, max_depth=None):
    """The whole-scope implementation this change replaced, verbatim, as the oracle."""
    fold = (lambda value: value.replace("\\", "/").casefold()) if is_windows_scope(project) else (
        lambda value: value.replace("\\", "/"))
    seen = {}
    for f in files:
        key = (fold(f["rel_path"]), f.get("kind", "code"))
        prior = seen.setdefault(key, f["rel_path"])
        if prior != f["rel_path"]:
            raise ValueError("case-insensitive catalog path collision")
    reported_keys = set(seen)
    reported_kinds = {kind for _, kind in reported_keys}
    upserted = removed = 0
    with factory() as session:
        existing = {
            (fold(row.rel_path), row.kind): row
            for row in session.query(IndexCatalog).filter(
                IndexCatalog.owner_user_id == owner_user_id,
                IndexCatalog.host == host,
                IndexCatalog.project == project,
            )
        }
        for f in files:
            kind = f.get("kind", "code")
            row = existing.get((fold(f["rel_path"]), kind))
            if row is None:
                row = IndexCatalog(owner_user_id=owner_user_id, host=host, project=project,
                                   rel_path=f["rel_path"], kind=kind)
                session.add(row)
            row.root_dir = root_dir
            row.rel_path = f["rel_path"].replace("\\", "/")
            row.kind = kind
            row.collection = f.get("collection")
            row.language = f.get("language")
            row.chunk_count = int(f.get("chunk_count", 0) or 0)
            row.content_hash = f.get("content_hash")
            row.size_bytes = f.get("size_bytes")
            row.metadata_json = f.get("metadata") or {}
            row.indexed_at = _utcnow()
            row.deleted_at = None
            upserted += 1
        if replace and reported_keys:
            for (rel_path, kind), row in existing.items():
                if (kind in reported_kinds and (rel_path, kind) not in reported_keys
                        and row.deleted_at is None
                        and _within_sweep_envelope(rel_path, patterns, max_depth, kind)):
                    row.deleted_at = _utcnow()
                    removed += 1
    return upserted, removed


def _legacy_delete_files(factory, *, owner_user_id, project, host=None, kind=None,
                         rel_paths=None):
    with factory() as session:
        query = session.query(IndexCatalog).filter(
            IndexCatalog.owner_user_id == owner_user_id,
            IndexCatalog.project == project,
            IndexCatalog.deleted_at.is_(None),
        )
        if host is not None:
            query = query.filter(IndexCatalog.host == host)
        if kind is not None:
            query = query.filter(IndexCatalog.kind == kind)
        if rel_paths is not None:
            query = query.filter(IndexCatalog.rel_path.in_(rel_paths))
        count = 0
        for row in query:
            row.deleted_at = _utcnow()
            count += 1
        return count


def _snapshot(factory):
    with factory() as session:
        return sorted(
            (r.owner_user_id, r.host, r.project, r.rel_path, r.kind, r.root_dir,
             r.collection, r.language, r.chunk_count, r.content_hash, r.size_bytes,
             repr(r.metadata_json), r.deleted_at is not None)
            for r in session.query(IndexCatalog))


def _file(rel_path, kind="code", **extra):
    return {"rel_path": rel_path, "kind": kind, "collection": f"{kind}_chunks",
            "chunk_count": 2, "content_hash": "h" + rel_path[-3:], **extra}


class _RowsLoaded:
    """Counts IndexCatalog ORM entities materialised, via the mapper's load event."""

    def __init__(self):
        self.count = 0

    def _on_load(self, target, context):
        self.count += 1

    def __enter__(self):
        event.listen(IndexCatalog, "load", self._on_load)
        return self

    def __exit__(self, *exc):
        event.remove(IndexCatalog, "load", self._on_load)


def _seed(factory, project, n, kind="code", host="h"):
    repo = IndexCatalogRepository(session_factory=factory)
    repo.report_files(
        owner_user_id="u", host=host, project=project, root_dir=project,
        files=[_file(f"src/pkg{i % 7}/file{i}.py", kind) for i in range(n)])
    return repo


@pytest.mark.parametrize("project", ["/home/me/repo", "C:/Users/Me/Repo"])
def test_incremental_report_materialises_only_the_reported_rows(project):
    _, factory = _engine_and_factory()
    repo = _seed(factory, project, 1500)
    with _RowsLoaded() as loaded:
        upserted, removed = repo.report_files(
            owner_user_id="u", host="h", project=project, root_dir=project,
            files=[_file("src/pkg1/file1.py"), _file("src/new_file.py")], replace=False)
    assert (upserted, removed) == (2, 0)
    # one existing row matched; the new file needs no load. Never the 1500-row scope.
    assert loaded.count == 1


def test_replace_report_loads_reported_rows_only_and_sweeps_by_id():
    _, factory = _engine_and_factory()
    project = "C:/Users/Me/Repo"
    repo = _seed(factory, project, 1200)
    keep = [_file(f"src/pkg{i % 7}/file{i}.py") for i in range(10)]
    with _RowsLoaded() as loaded:
        upserted, removed = repo.report_files(
            owner_user_id="u", host="h", project=project, root_dir=project, files=keep)
    assert (upserted, removed) == (10, 1190)
    assert loaded.count == 10
    with factory() as session:
        live = session.query(IndexCatalog).filter(IndexCatalog.deleted_at.is_(None)).count()
    assert live == 10


def test_delete_files_is_a_bulk_update_without_loading_rows():
    _, factory = _engine_and_factory()
    repo = _seed(factory, "/home/me/repo", 800)
    with _RowsLoaded() as loaded:
        n = repo.delete_files(
            owner_user_id="u", project="/home/me/repo",
            rel_paths=["src/pkg1/file1.py", "src/pkg2/file2.py", "missing.py"])
        assert n == 2
        assert repo.delete_files(owner_user_id="u", project="/home/me/repo",
                                 rel_paths=["src/pkg1/file1.py"]) == 0
        assert repo.delete_files(owner_user_id="u", project="/home/me/repo",
                                 kind="code") == 798
    assert loaded.count == 0


def test_windows_case_lookup_uses_an_index_not_a_scope_scan():
    engine, factory = _engine_and_factory()
    project = "C:/Users/Me/Repo"
    repo = _seed(factory, project, 300)
    statements: list[tuple[str, tuple]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        repo.report_files(
            owner_user_id="u", host="h", project=project, root_dir=project,
            files=[_file("SRC/pkg1/FILE1.py")], replace=False)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    lookups = [(s, p) for s, p in statements
               if s.lstrip().upper().startswith("SELECT") and "lower(" in s.lower()]
    assert lookups, "windows scope must look rows up by a case-folded path"
    with engine.connect() as conn:
        for statement, params in lookups:
            plan = " ".join(str(row[3]) for row in conn.exec_driver_sql(
                "EXPLAIN QUERY PLAN " + statement, params))
            assert "ix_index_catalog_scope_lower_path" in plan or "uq_index_catalog_file" in plan \
                or "sqlite_autoindex_index_catalog_1" in plan, plan
            assert "SCAN" not in plan.replace("SCAN CONSTANT ROW", ""), plan


def test_windows_case_variant_row_is_reused_not_duplicated():
    _, factory = _engine_and_factory()
    project = "C:/Users/Me/Repo"
    repo = _seed(factory, project, 50)
    repo.report_files(
        owner_user_id="u", host="h", project=project, root_dir=project,
        files=[_file("SRC/PKG1/FILE1.PY")], replace=False)
    with factory() as session:
        rows = session.query(IndexCatalog).filter(
            IndexCatalog.rel_path.ilike("src/pkg1/file1.py")).all()
    assert [r.rel_path for r in rows] == ["SRC/PKG1/FILE1.PY"]


def test_windows_non_ascii_case_variant_row_is_still_reused():
    _, factory = _engine_and_factory()
    project = "C:/Users/Me/Repo"
    repo = IndexCatalogRepository(session_factory=factory)
    repo.report_files(owner_user_id="u", host="h", project=project, root_dir=project,
                      files=[_file("docs/Étude.md", "doc")], replace=False)
    # SQLite lower() is ASCII-only; casefold identity must still hold for these rows.
    repo.report_files(owner_user_id="u", host="h", project=project, root_dir=project,
                      files=[_file("docs/étude.md", "doc")], replace=False)
    with factory() as session:
        assert session.query(IndexCatalog).count() == 1


def test_collision_inside_one_report_is_still_rejected():
    _, factory = _engine_and_factory()
    repo = IndexCatalogRepository(session_factory=factory)
    with pytest.raises(ValueError, match="case-insensitive catalog path collision"):
        repo.report_files(
            owner_user_id="u", host="h", project="C:/Repo", root_dir="C:/Repo",
            files=[_file("Docs/A.md"), _file("docs/a.md")])


_SEGMENTS = ["src", "Src", "docs", ".github", "a", "b"]
_NAMES = ["x.py", "X.PY", "y.md", "Y.md", "z.txt", "ß.md", "É.md", "é.md"]


def _random_paths(rng, n):
    paths = {}
    for _ in range(n):
        depth = rng.randint(0, 3)
        p = "/".join([rng.choice(_SEGMENTS) for _ in range(depth)] + [rng.choice(_NAMES)])
        paths.setdefault(p.casefold(), p)  # one spelling per identity within a report
    return list(paths.values())


@pytest.mark.parametrize("seed", range(25))
def test_random_report_and_delete_sequences_match_the_whole_scope_oracle(seed):
    rng = random.Random(seed)
    _, new_factory = _engine_and_factory()
    _, old_factory = _engine_and_factory()
    repo = IndexCatalogRepository(session_factory=new_factory)
    projects = ["C:/Users/Me/Repo", "/home/me/repo"]
    for step in range(25):
        project = rng.choice(projects)
        host = rng.choice(["h1", "h2"])
        if rng.random() < 0.8:
            kinds = rng.choice([["code"], ["doc"], ["code", "doc"]])
            files = [_file(p, rng.choice(kinds)) for p in _random_paths(rng, rng.randint(0, 6))]
            kwargs = dict(
                owner_user_id="u", host=host, project=project, root_dir=project, files=files,
                replace=rng.random() < 0.5,
                patterns=rng.choice([None, ["*.md"], ["*.py", "*.txt"], ["*"]]),
                max_depth=rng.choice([None, 0, 1, 2]))
            try:
                expected = _legacy_report_files(old_factory, **kwargs)
            except ValueError:
                with pytest.raises(ValueError):
                    repo.report_files(**kwargs)
                continue
            assert repo.report_files(**kwargs) == expected, (seed, step, kwargs)
        else:
            kwargs = dict(
                owner_user_id="u", project=project,
                host=rng.choice([None, "h1", "h2"]),
                kind=rng.choice([None, "code", "doc"]),
                rel_paths=rng.choice([None, [], _random_paths(rng, 3)]))
            assert repo.delete_files(**kwargs) == _legacy_delete_files(old_factory, **kwargs)
        assert _snapshot(new_factory) == _snapshot(old_factory), (seed, step)


def test_report_scales_past_the_sqlite_variable_limit():
    _, factory = _engine_and_factory()
    repo = _seed(factory, "/home/me/repo", 40)
    files = [_file(f"big/dir/file{i}.py") for i in range(2500)]
    assert repo.report_files(owner_user_id="u", host="h", project="/home/me/repo",
                             root_dir="/home/me/repo", files=files) == (2500, 40)
    assert repo.delete_files(owner_user_id="u", project="/home/me/repo",
                             rel_paths=[f["rel_path"] for f in files]) == 2500


@pytest.mark.parametrize("kwargs", [
    {}, {"kind": "code"}, {"host": "h"}, {"host": "h", "kind": "code"},
])
def test_list_projects_reads_the_covering_index_only(kwargs):
    engine, factory = _engine_and_factory()
    repo = _seed(factory, "C:/Users/Me/Repo", 200)
    _seed(factory, "/home/me/other", 50)
    statements: list[tuple[str, tuple]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        with _RowsLoaded() as loaded:
            rows = repo.list_projects(owner_user_id="u", **kwargs)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert {r["project"] for r in rows} == {"C:/Users/Me/Repo", "/home/me/other"}         if "host" not in kwargs or kwargs["host"] == "h" else True
    assert loaded.count == 0
    statement, params = statements[-1]
    with engine.connect() as conn:
        plan = " ".join(str(r[3]) for r in conn.exec_driver_sql(
            "EXPLAIN QUERY PLAN " + statement, params))
    assert "COVERING INDEX ix_index_catalog_owner_projects" in plan, plan
    assert "TEMP B-TREE" not in plan, plan


def test_existing_database_gets_the_catalog_indexes_from_the_idempotent_migrations(
        tmp_path, monkeypatch):
    from core.db import database

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_index_catalog_owner_projects"))
        conn.execute(text("DROP INDEX ix_index_catalog_scope_lower_path"))
    monkeypatch.setattr(database, "engine", engine)
    database._apply_idempotent_migrations()
    database._apply_idempotent_migrations()  # re-runnable
    with engine.connect() as conn:
        names = {r[0] for r in conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='index_catalog'")}
    assert {"ix_index_catalog_owner_projects", "ix_index_catalog_scope_lower_path"} <= names
