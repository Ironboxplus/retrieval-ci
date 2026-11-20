"""A scope nobody indexed resolves to the nearest indexed ancestor (code and doc alike).

Real SQLite, real index_catalog schema. The SQL the resolver issues is captured and
EXPLAINed: every lookup must be an index SEARCH, never a scan of the catalog.
"""
from __future__ import annotations

import pytest

from tests.catalog_sqlite import CatalogDb, capture_catalog_plans, unbounded_reads

OWNER = "owner"


@pytest.fixture()
def db():
    catalog = CatalogDb()
    catalog.bulk_noise(owner=OWNER)
    return catalog


def resolve(db, scope, *, kind="code", owner=OWNER, known_scopes=None):
    return db.service.resolve_indexed_scope(
        owner_user_id=owner, kind=kind, scope=scope, known_scopes=known_scopes)


def test_scope_with_its_own_index_is_not_rewritten(db):
    db.add_files("/repo", ["app/a.py"])
    db.add_files("/repo/app", ["a.py"])
    assert resolve(db, "/repo/app") is None


def test_unindexed_subdirectory_resolves_to_indexed_parent(db):
    db.add_files("/repo", ["app/a.py", "docs/b.md"])
    resolved = resolve(db, "/repo/app/api")
    assert (resolved.scope, resolved.path_prefix) == ("/repo", "app/api")
    assert resolved.aliases == ["/repo"]


def test_nearest_ancestor_wins_over_a_farther_one(db):
    db.add_files("/repo", ["a/b/c/x.py"])
    db.add_files("/repo/a/b", ["c/y.py"])
    resolved = resolve(db, "/repo/a/b/c/d")
    assert (resolved.scope, resolved.path_prefix) == ("/repo/a/b", "c/d")


def test_no_indexed_ancestor_returns_none(db):
    db.add_files("/other", ["a.py"])
    assert resolve(db, "/repo/app") is None


def test_sibling_sharing_a_string_prefix_is_not_an_ancestor(db):
    db.add_files("/repo-backup", ["app/a.py"])
    assert resolve(db, "/repo/app") is None


def test_deleted_ancestor_does_not_count(db):
    db.add_files("/repo", ["app/a.py"], deleted=True)
    assert resolve(db, "/repo/app") is None


def test_ancestor_of_another_owner_does_not_count(db):
    db.add_files("/repo", ["app/a.py"], owner="someone-else")
    assert resolve(db, "/repo/app") is None


def test_ancestor_must_have_the_requested_kind(db):
    db.add_files("/repo", ["notes/a.md"], kind="doc")
    assert resolve(db, "/repo/notes", kind="code") is None
    assert resolve(db, "/repo/notes", kind="doc").scope == "/repo"


def test_a_name_that_is_not_an_absolute_path_is_left_alone(db):
    db.add_files("/repo", ["app/a.py"])
    assert resolve(db, "retrival") is None
    assert resolve(db, "") is None


def test_windows_scope_resolves_case_insensitively_and_reports_legacy_spellings(db):
    legacy = "E:/Go/OpenList"
    db.add_files(legacy, ["internal/a.go"])
    db.add_files("e:/go/openlist", ["internal/b.go"])
    resolved = resolve(db, "e:/go/openlist/Internal/Op")
    assert resolved.path_prefix == "internal/op"
    assert resolved.aliases[0] == "E:/go/openlist"
    assert set(resolved.aliases) == {"E:/go/openlist", legacy, "e:/go/openlist"}


def test_windows_scope_with_a_legacy_spelled_own_index_is_not_rewritten(db):
    db.add_files("E:/Go/OpenList", ["a.go"])
    db.add_files("E:/Go/OpenList/Internal", ["b.go"])
    known = db.service.equivalent_project_scopes(
        owner_user_id=OWNER, project="e:/go/openlist/internal")
    assert resolve(db, "e:/go/openlist/internal", known_scopes=known) is None


def test_windows_drive_root_can_be_the_indexed_ancestor(db):
    db.add_files("E:/", ["go/a.go"])
    resolved = resolve(db, "e:/go/openlist")
    assert (resolved.scope, resolved.path_prefix) == ("E:/", "go/openlist")


def test_exact_and_ancestor_lookups_are_index_searches_not_scans(db):
    db.add_files("/repo", ["app/a.py"])
    with capture_catalog_plans(db) as exact_plans:
        assert resolve(db, "/repo") is None
    with capture_catalog_plans(db) as ancestor_plans:
        assert resolve(db, "/repo/app/deep/er/still").path_prefix == "app/deep/er/still"
    with capture_catalog_plans(db) as miss_plans:
        assert resolve(db, "/nothing/here/at/all") is None
    for plans in (exact_plans, ancestor_plans, miss_plans):
        assert plans, "the resolver must consult the catalog"
        assert unbounded_reads(plans) == []
        assert all("USING" in line and "INDEX" in line
                   for _, details in plans for line in details if "index_catalog" in line)
