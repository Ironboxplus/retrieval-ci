"""Inventory reads that clients use as diff baselines or as a filtered summary.

``detail=True`` loaded every row of a scope as an ORM entity (``metadata_json``
included) so a client could compare content hashes; ``lean`` returns the same files
with only the compared columns. ``project_contains`` renders only the matching projects
while ``totals`` keep describing the whole inventory, so a filtered call is exactly the
unfiltered response with the other project nodes dropped.

Real SQLite and the real repository/service; ORM entity loads and SQL statements are
observed through SQLAlchemy events, nothing is stubbed.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.schemas import IndexFileRecord, IndexReportRequest
from app.services.index_catalog.service import IndexCatalogService
from core.db.database import Base
from core.db.models import IndexCatalog


def _service():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
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

    return engine, IndexCatalogService(repository=IndexCatalogRepository(session_factory=factory))


def _report(svc, host, project, kind, n, *, replace=True):
    svc.report(IndexReportRequest(
        host=host, project=project, root_dir=project, replace=replace,
        files=[IndexFileRecord(
            rel_path=f"pkg{i % 5}/f{i}.py", kind=kind, collection=f"{kind}_chunks",
            language="python", chunk_count=i % 4 + 1, content_hash=f"hash{i:04d}",
            size_bytes=100 + i, metadata={"git_commit": f"c{i % 3}", "blob": "x" * 50})
            for i in range(n)]), owner_user_id="u")


def _files(resp):
    return {
        (u.owner_user_id, h.host, p.project, f.rel_path, f.kind): f
        for u in resp.users for h in u.hosts for p in h.projects
        for d in p.dirs for f in d.files
    }


class _EntitiesLoaded:
    def __init__(self):
        self.count = 0

    def _on_load(self, target, context):
        self.count += 1

    def __enter__(self):
        event.listen(IndexCatalog, "load", self._on_load)
        return self

    def __exit__(self, *exc):
        event.remove(IndexCatalog, "load", self._on_load)


COMPARED = ("rel_path", "kind", "language", "chunk_count", "content_hash", "size_bytes")


def test_lean_detail_carries_the_compared_fields_of_every_file_without_orm_entities():
    _, svc = _service()
    _report(svc, "h1", "/work/a", "code", 60)
    _report(svc, "h2", "/work/a", "code", 15)
    _report(svc, "h1", "/work/a", "doc", 12)
    _report(svc, "h1", "/work/b", "code", 7)

    with _EntitiesLoaded() as full_loads:
        full = svc.inventory(owner_user_id="u", project="/work/a", detail=True)
    with _EntitiesLoaded() as lean_loads:
        lean = svc.inventory(owner_user_id="u", project="/work/a", detail=True, lean=True)

    assert full_loads.count == 87 and lean_loads.count == 0
    full_files, lean_files = _files(full), _files(lean)
    assert full_files.keys() == lean_files.keys() and len(lean_files) == 87
    for key, node in lean_files.items():
        assert {f: getattr(node, f) for f in COMPARED} == \
            {f: getattr(full_files[key], f) for f in COMPARED}
        assert node.indexed_at is None and node.metadata == {} and node.collection is None
    # same tree shape and counts, so a consumer cannot tell lean from full by structure
    assert lean.totals == full.totals
    assert [(d.dir, d.file_count, d.chunk_count) for u in lean.users for h in u.hosts
            for p in h.projects for d in p.dirs] == \
        [(d.dir, d.file_count, d.chunk_count) for u in full.users for h in u.hosts
         for p in h.projects for d in p.dirs]


@pytest.mark.parametrize("kind,host", [("code", None), ("doc", "h1"), ("code", "h2")])
def test_lean_respects_kind_and_host_filters(kind, host):
    _, svc = _service()
    _report(svc, "h1", "/work/a", "code", 9)
    _report(svc, "h2", "/work/a", "code", 4)
    _report(svc, "h1", "/work/a", "doc", 6)
    kwargs = dict(owner_user_id="u", project="/work/a", kind=kind, host=host, detail=True)
    assert _files(svc.inventory(**kwargs, lean=True)).keys() == _files(svc.inventory(**kwargs)).keys()


def _seed_many_projects(svc):
    for name, host in [("C:/Users/Me/retrival", "h1"), ("C:/Users/Me/Other", "h1"),
                       ("/home/me/retrival-docs", "h2"), ("/srv/data", "h2"),
                       ("E:/JETB", "h1")]:
        _report(svc, host, name, "code", 11)
    _report(svc, "h1", "C:/Users/Me/retrival", "doc", 5)


def _nodes(resp):
    return [(u.owner_user_id, h.host, p.project, p.file_count, p.chunk_count,
             p.last_indexed_at, p.last_commit, p.root_dir)
            for u in resp.users for h in u.hosts for p in h.projects]


@pytest.mark.parametrize("fragment", [
    "retrival", "RETRIVAL", "users/me", "Users\\Me\\", "jetb", "nothing-matches", "é",
])
def test_project_contains_equals_filtering_the_unfiltered_summary(fragment):
    _, svc = _service()
    _seed_many_projects(svc)
    needle = fragment.replace("\\", "/").lower()

    full = svc.inventory(owner_user_id="u", detail=False)
    filtered = svc.inventory(owner_user_id="u", detail=False, project_contains=fragment)

    expected = [n for n in _nodes(full) if needle in n[2].replace("\\", "/").lower()]
    assert _nodes(filtered) == expected
    # users and hosts survive (with possibly empty project lists) and every total is unchanged
    assert filtered.totals == full.totals
    assert [(u.owner_user_id, u.file_count, [(h.host, h.file_count, h.chunk_count)
                                              for h in u.hosts]) for u in filtered.users] == \
        [(u.owner_user_id, u.file_count, [(h.host, h.file_count, h.chunk_count)
                                           for h in u.hosts]) for u in full.users]


def test_project_contains_limits_the_winner_lookups_to_matching_projects():
    engine, svc = _service()
    _seed_many_projects(svc)
    statements: list[str] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        svc.inventory(owner_user_id="u", detail=False)
        unfiltered = [s for s in statements if "fresh_rel_path" in s.lower()]
        statements.clear()
        svc.inventory(owner_user_id="u", detail=False, project_contains="retrival")
        ascii_filtered = [s for s in statements if "fresh_rel_path" in s.lower()]
        statements.clear()
        svc.inventory(owner_user_id="u", detail=False, project_contains="é")
        non_ascii = [s for s in statements if "fresh_rel_path" in s.lower()]
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert len(unfiltered) == len(ascii_filtered) == len(non_ascii) == 1
    assert "instr(" not in unfiltered[0].lower()
    assert "instr(" in ascii_filtered[0].lower()
    # lower() is ASCII-only in SQLite: a non-ASCII needle is filtered by the service instead
    assert "instr(" not in non_ascii[0].lower()
