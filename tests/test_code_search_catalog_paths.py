"""Path globs must resolve through owner-scoped SQL, never Qdrant facets."""
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import Session

from core.db.models import IndexCatalog
from app.services.index_catalog.repository import IndexCatalogRepository


@pytest.fixture
def repository():
    engine = create_engine('sqlite://')
    IndexCatalog.__table__.create(engine)
    with engine.begin() as conn:
        conn.execute(insert(IndexCatalog), [dict(
            owner_user_id='owner', host='host', project='/large',
            rel_path=f'noise/{i}.cpp', kind='code', chunk_count=1,
        ) for i in range(100_001)])
        conn.execute(insert(IndexCatalog), [
            dict(owner_user_id=owner, host='host', project=project,
                 rel_path=path, kind=kind, chunk_count=1)
            for owner, project, path, kind in [
                ('owner', '/magpie', 'src/Magpie/EffectCompiler.cpp', 'code'),
                ('owner', '/magpie', 'src/Magpie.Core/EffectDesc.h', 'code'),
                ('other', '/magpie', 'src/Magpie/foreign.cpp', 'code'),
                ('owner', '/magpie', 'src/Magpie/notes.cpp', 'doc'),
                ('owner', 'C:/legacy', '/SRC/MAGPIE/Legacy.cpp', 'code'),
            ]])
    @contextmanager
    def sessions():
        with Session(engine) as session:
            yield session
    yield IndexCatalogRepository(session_factory=sessions)
    engine.dispose()


def test_cross_scope_glob_handles_more_than_facet_limit_without_truncation(repository):
    paths, count = repository.matching_code_paths(
        owner_user_id='owner', project=None,
        pattern='src/Magpie*/**', case_sensitive=True)
    assert set(paths) == {'src/Magpie/EffectCompiler.cpp', 'src/Magpie.Core/EffectDesc.h'}
    assert count == 100_004


def test_catalog_glob_preserves_scope_and_legacy_windows_matching(repository):
    paths, count = repository.matching_code_paths(
        owner_user_id='owner', project=['C:/legacy'],
        pattern='src/magpie/', case_sensitive=False)
    assert paths == ['/SRC/MAGPIE/Legacy.cpp']
    assert count == 1


def test_catalog_glob_distinguishes_no_match_from_missing_scope(repository):
    paths, count = repository.matching_code_paths(
        owner_user_id='owner', project='/magpie',
        pattern='missing/**', case_sensitive=True)
    assert paths == []
    assert count == 2


def test_catalog_glob_rejects_unrelated_paths_before_expensive_normalization(repository, monkeypatch):
    from app.services import scope_paths
    original = scope_paths.stored_relative_path_identity
    validated = []
    def identity(path, *, case_sensitive):
        validated.append(path)
        return original(path, case_sensitive=case_sensitive)
    monkeypatch.setattr(scope_paths, 'stored_relative_path_identity', identity)
    paths, _ = repository.matching_code_paths(owner_user_id='owner', project=None,
        pattern='src/Magpie*/**', case_sensitive=True)
    assert len(paths) == 2
    assert len(validated) < 10


def test_catalog_glob_prefilter_keeps_legacy_dot_and_separator_spellings(repository):
    with repository.session() as session:
        session.execute(insert(IndexCatalog), [dict(
            owner_user_id='owner', host='host', project='/legacy', rel_path=path,
            kind='code', chunk_count=1,
        ) for path in ['./src/Magpie/./Odd.cpp', 'src//Magpie/Separator.cpp',
                       '//src/Magpie/invalid.cpp', 'src/Magpie/../invalid.cpp']])
        session.commit()
    paths, _ = repository.matching_code_paths(owner_user_id='owner', project='/legacy',
        pattern='src/Magpie*/**', case_sensitive=True)
    assert set(paths) == {'./src/Magpie/./Odd.cpp', 'src//Magpie/Separator.cpp'}


def test_search_path_selection_never_calls_qdrant_facets(monkeypatch):
    from types import SimpleNamespace
    from app.services.code_kb.vector_store import CodeVectorStore
    calls = []
    class Catalog:
        def matching_code_paths(self, **kwargs):
            calls.append(kwargs)
            return ['src/Magpie/Core.cpp'], 123_456
    monkeypatch.setattr('app.services.index_catalog.repository.IndexCatalogRepository', Catalog)
    store = CodeVectorStore(store=SimpleNamespace())
    assert store.matching_paths(owner_user_id='owner', project=None,
        pattern='src/Magpie*/**', case_sensitive=True) == (['src/Magpie/Core.cpp'], 123_456)
    assert calls == [dict(owner_user_id='owner', project=None,
        pattern='src/Magpie*/**', case_sensitive=True)]


def test_deterministic_facet_limit_is_not_retried(monkeypatch):
    from types import SimpleNamespace
    from app.services.code_kb import vector_store
    calls = []
    def facet(**_kwargs):
        calls.append(1)
        return SimpleNamespace(hits=[SimpleNamespace(value='one'), SimpleNamespace(value='two')])
    monkeypatch.setattr(vector_store, '_PATH_FACET_LIMIT', 2)
    monkeypatch.setattr('core.storage.qdrant_operations.time.sleep', lambda _seconds: None)
    store = vector_store.CodeVectorStore(store=SimpleNamespace(client=SimpleNamespace(facet=facet)))
    with pytest.raises(RuntimeError, match='safety limit'):
        store._facet_values('path', None, operation='test facet')
    assert calls == [1]


def test_code_path_lookup_index_is_covering_and_idempotent(repository):
    from scripts.ensure_code_search_path_index import ensure_code_search_path_index
    with repository.session() as session:
        connection = session.connection().connection.driver_connection
        ensure_code_search_path_index(connection)
        ensure_code_search_path_index(connection)
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT DISTINCT rel_path FROM index_catalog "
            "WHERE owner_user_id=? AND kind='code' AND deleted_at IS NULL AND chunk_count>0",
            ('owner',),
        ).fetchall()
        assert any('COVERING INDEX ix_index_catalog_code_path_lookup' in row[3] for row in plan)
    paths, count = repository.matching_code_paths(
        owner_user_id='owner', project=None,
        pattern='src/Magpie*/**', case_sensitive=True)
    assert len(paths) == 2 and count == 100_004
