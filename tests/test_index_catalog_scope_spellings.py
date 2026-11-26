"""Case-variant spellings of a Windows scope are read from that scope's rows alone.

Real SQLite, real ``index_catalog`` schema. Production keeps ~600K catalog rows, nearly all
under one owner, with no ``sqlite_stat1``; resolving a Windows subdirectory to its indexed
ancestor used to read that owner's whole project list (75-105 ms on the host). The SQL is
EXPLAINed (index SEARCH on ``lower(project)``, never a scan) and timed on a catalog of that
shape.
"""
from __future__ import annotations

import statistics
import time

import pytest
from sqlalchemy import text

from tests.catalog_sqlite import (
    CatalogDb, capture_catalog_plans, full_scans, unbounded_reads,
)

OWNER = "internal-admin"
BUDGET_MS = 5.0


@pytest.fixture()
def db():
    catalog = CatalogDb()
    catalog.bulk_noise(owner=OWNER)
    return catalog


def equivalent(db, scope, *, kind="code", owner=OWNER):
    return db.service.equivalent_project_scopes(
        owner_user_id=owner, project=scope, kind=kind)


def resolve(db, scope, *, kind="code", owner=OWNER):
    return db.service.resolve_indexed_scope(owner_user_id=owner, kind=kind, scope=scope)


def test_spellings_of_one_scope_differing_by_case_are_all_found(db):
    db.add_files("E:/Go/OpenList", ["a.go"], owner=OWNER)
    db.add_files("e:/go/openlist", ["b.go"], owner=OWNER)
    db.add_files("E:/GO/OPENLIST", ["c.go"], owner=OWNER)
    db.add_files("E:/go/openlist-backup", ["d.go"], owner=OWNER)
    assert equivalent(db, "E:/go/OPENLIST") == [
        "E:/go/openlist", "E:/GO/OPENLIST", "E:/Go/OpenList", "e:/go/openlist"]


def test_spellings_exclude_deleted_rows_other_owners_and_other_kinds(db):
    db.add_files("E:/Go/Gone", ["a.go"], owner=OWNER, deleted=True)
    db.add_files("E:/Go/Foreign", ["a.go"], owner="someone-else")
    db.add_files("E:/Go/Notes", ["a.md"], owner=OWNER, kind="doc")
    assert equivalent(db, "E:/go/gone") == ["E:/go/gone"]
    assert equivalent(db, "E:/go/foreign") == ["E:/go/foreign"]
    assert equivalent(db, "E:/go/notes", kind="code") == ["E:/go/notes"]
    assert equivalent(db, "E:/go/notes", kind="doc") == ["E:/go/notes", "E:/Go/Notes"]


def test_posix_scope_stays_case_sensitive_and_reads_nothing(db):
    db.add_files("/Repo", ["a.py"], owner=OWNER)
    with capture_catalog_plans(db) as plans:
        assert equivalent(db, "/repo") == ["/repo"]
    assert plans == []


def test_a_spelling_that_only_casefolds_to_ascii_is_still_found(db):
    # ``lower('ß')`` stays ``ß`` while Python folds it to ``ss``: only the partial index of
    # such scope names can surface the row for an ASCII request.
    db.add_files("E:/Work/Straße", ["a.go"], owner=OWNER)
    assert equivalent(db, "e:/WORK/STRASSE") == ["E:/work/strasse", "E:/Work/Straße"]


def test_a_scope_lower_cannot_fold_is_found_by_the_project_list_fallback(db):
    db.add_files("E:/Work/RÉSUMÉ", ["a.go"], owner=OWNER)
    db.add_files("E:/work/résumé", ["b.go"], owner=OWNER)
    assert equivalent(db, "e:/WORK/résumé") == [
        "E:/work/résumé", "E:/Work/RÉSUMÉ"]


def test_legacy_spelled_ancestor_is_resolved_from_its_own_rows(db):
    db.add_files("E:/Go/OpenList", ["internal/a.go"], owner=OWNER)
    resolved = resolve(db, "e:/go/openlist/Internal/Op")
    assert (resolved.scope, resolved.path_prefix) == ("E:/go/openlist", "internal/op")
    assert resolved.aliases == ["E:/go/openlist", "E:/Go/OpenList"]


def test_spelling_lookups_are_index_searches_that_never_read_other_scopes(db):
    db.add_files("E:/Go/OpenList", ["a.go"], owner=OWNER)
    db.add_files("E:/go/openlist", ["b.go"], owner=OWNER)
    with capture_catalog_plans(db) as ancestor_plans:
        assert resolve(db, "E:/GO/openlist/app").aliases == [
            "E:/go/openlist", "E:/Go/OpenList"]
    db.add_files("E:/Legacy/Only", ["a.go"], owner=OWNER)
    with capture_catalog_plans(db) as legacy_only_plans:
        assert resolve(db, "e:/legacy/only/deeper").scope == "E:/legacy/only"
    with capture_catalog_plans(db) as miss_plans:
        assert resolve(db, "E:/nothing/here/at/all") is None
    for plans in (ancestor_plans, legacy_only_plans, miss_plans):
        assert any("ix_index_catalog_live_project_fold" in line
                   for _, details in plans for line in details)
        assert full_scans(plans) == []
        assert unbounded_reads(plans) == []


def test_existing_database_gets_the_spelling_indexes_and_answers_without_them(
        tmp_path, monkeypatch):
    from core.db import database

    old = CatalogDb(path=tmp_path / "old.db")
    with old.engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_index_catalog_live_project_fold"))
        conn.execute(text("DROP INDEX ix_index_catalog_live_odd_project"))
    assert "ix_index_catalog_live_project_fold" not in _catalog_indexes(old.engine)
    old.add_files("E:/Go/OpenList", ["a.go"], owner=OWNER)
    old.add_files("E:/Work/Straße", ["a.go"], owner=OWNER)

    def spellings():
        return (equivalent(old, "e:/go/openlist"), equivalent(old, "e:/work/strasse"))

    expected = (["E:/go/openlist", "E:/Go/OpenList"], ["E:/work/strasse", "E:/Work/Straße"])
    assert spellings() == expected  # an unmigrated database is slower, never wrong

    monkeypatch.setattr(database, "engine", old.engine)
    database._apply_idempotent_migrations()
    database._apply_idempotent_migrations()  # re-runnable
    assert {"ix_index_catalog_live_project_fold",
            "ix_index_catalog_live_odd_project"} <= _catalog_indexes(old.engine)
    assert spellings() == expected


def _catalog_indexes(engine) -> set[str]:
    with engine.connect() as conn:
        return {row[0] for row in conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='index_catalog'")}


# --------------------------------------------------------------------------------------
# Production shape: ~600K rows, one owner, no sqlite_stat1.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def production_catalog(tmp_path_factory):
    catalog = CatalogDb(path=tmp_path_factory.mktemp("catalog") / "catalog.db")
    catalog.production_shaped(owner=OWNER, rows=600_000, scopes=300)
    catalog.add_files("C:/users/arc/prd/retrival", [f"app/a{i}.py" for i in range(1_200)],
                      owner=OWNER)
    catalog.add_files("C:/Users/Arc/Prd/retrival", [f"app/b{i}.py" for i in range(50)],
                      owner=OWNER)
    catalog.add_files("D:/Legacy/Only", [f"src/c{i}.py" for i in range(50)], owner=OWNER)
    return catalog


def _median_ms(call, runs: int = 15) -> float:
    call()  # warm the page cache and the statement cache
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def test_production_shaped_catalog_has_no_statistics_and_the_expected_size(production_catalog):
    with production_catalog.engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM index_catalog").scalar() >= 600_000
        assert conn.exec_driver_sql(
            "SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_stat1'").scalar() == 0


@pytest.mark.parametrize("scope, expected_scope, expected_prefix", [
    ("C:/Users/Arc/prd/retrival/app", "C:/users/arc/prd/retrival", "app"),
    ("c:/USERS/arc/PRD/retrival/app/api/routes", "C:/users/arc/prd/retrival", "app/api/routes"),
    ("D:/legacy/only/src/deep", "D:/legacy/only", "src/deep"),
])
def test_windows_subdirectory_resolves_to_its_ancestor_within_budget(
        production_catalog, scope, expected_scope, expected_prefix):
    resolved = resolve(production_catalog, scope)
    assert (resolved.scope, resolved.path_prefix) == (expected_scope, expected_prefix)
    assert expected_scope in resolved.aliases
    assert _median_ms(lambda: resolve(production_catalog, scope)) <= BUDGET_MS


def test_case_variant_aliases_and_misses_stay_within_budget(production_catalog):
    assert equivalent(production_catalog, "C:/Users/Arc/prd/retrival") == [
        "C:/users/arc/prd/retrival", "C:/Users/Arc/Prd/retrival"]
    assert _median_ms(lambda: equivalent(
        production_catalog, "C:/Users/Arc/prd/retrival")) <= BUDGET_MS
    assert resolve(production_catalog, "E:/no/such/place") is None
    assert _median_ms(lambda: resolve(production_catalog, "E:/no/such/place")) <= BUDGET_MS


def test_production_shaped_lookups_are_index_searches(production_catalog):
    with capture_catalog_plans(production_catalog) as plans:
        resolve(production_catalog, "C:/Users/Arc/prd/retrival/app")
        resolve(production_catalog, "E:/no/such/place")
    assert full_scans(plans) == []
    assert unbounded_reads(plans) == []
