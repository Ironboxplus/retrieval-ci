"""Directory-prefix and literal-prefix globs resolve through index ranges.

``matching_code_paths`` used to read every catalog row of the owner and match each one in
Python, so a subdirectory of a 61K-file scope cost the same as the whole scope. These tests
run the real repository on real SQLite: results are compared with the gitignore matcher as
oracle, the SQL is EXPLAINed (a range SEARCH, no SCAN), and the Python matcher is shown to
see only the rows under the prefix.
"""
from __future__ import annotations

import pytest

from app.services import scope_paths
from tests.catalog_sqlite import CatalogDb, capture_catalog_plans, unbounded_reads

OWNER = "owner"
SCOPE = "/repo"
WIN = "E:/go/openlist"


@pytest.fixture()
def db():
    catalog = CatalogDb()
    catalog.bulk_noise(owner=OWNER, scopes=3, files=1500)
    catalog.add_files(SCOPE, [
        "app/api/routes.py", "app/api/deep/handler.py", "app/main.py", "app/README.md",
        "application/other.py", "core/db.py", "docs/guide.md", "README.md",
    ])
    catalog.add_files("/elsewhere", ["app/api/foreign.py"])
    catalog.add_files(SCOPE, ["app/zero_chunks.py"], chunk_count=0)
    catalog.add_files(SCOPE, ["app/gone.py"], deleted=True)
    return catalog


def paths(db, *, project=SCOPE, pattern=None, prefix=None, case_sensitive=True):
    kwargs = dict(owner_user_id=OWNER, project=project, pattern=pattern,
                  case_sensitive=case_sensitive)
    if prefix is not None:
        kwargs["path_prefix"] = prefix
    return db.repository.matching_code_paths(**kwargs)


def test_prefix_selects_only_files_below_that_directory(db):
    found, count = paths(db, prefix="app")
    assert found == ["app/README.md", "app/api/deep/handler.py", "app/api/routes.py",
                     "app/main.py"]
    assert count == 4          # live files with chunks under the prefix


def test_prefix_boundary_is_a_path_segment_not_a_string_prefix(db):
    assert "application/other.py" not in paths(db, prefix="app")[0]
    assert paths(db, prefix="app/ap")[0] == []


def test_prefix_with_a_glob_relative_to_the_prefix(db):
    assert paths(db, pattern="api/**", prefix="app")[0] == [
        "app/api/deep/handler.py", "app/api/routes.py"]
    assert paths(db, pattern="*.md", prefix="app")[0] == ["app/README.md"]
    assert paths(db, pattern="deep/", prefix="app/api")[0] == ["app/api/deep/handler.py"]


def test_prefix_with_a_glob_counts_the_files_under_the_prefix(db):
    found, count = paths(db, pattern="nothing/**", prefix="app")
    assert found == [] and count == 4


def test_prefix_with_no_files_reports_zero_so_the_caller_can_explain(db):
    assert paths(db, prefix="missing/dir") == ([], 0)


def test_prefix_never_leaks_into_other_scopes(db):
    assert "app/api/foreign.py" not in paths(db, prefix="app/api")[0]


def test_prefix_special_characters_are_literal(db):
    db.add_files(SCOPE, ["we[ir]d/a.py", "we_ird/b.py", "50%/c.py", "weird/d.py"])
    assert paths(db, prefix="we[ir]d")[0] == ["we[ir]d/a.py"]
    assert paths(db, prefix="we_ird")[0] == ["we_ird/b.py"]
    assert paths(db, prefix="50%")[0] == ["50%/c.py"]


def test_prefix_keeps_legacy_spellings_that_normalise_into_it(db):
    db.add_files("/legacy", ["./app/dot.py", "/app/slash.py", "app//double.py",
                             "app/./inner.py", "app\\win.py", "app/../escape.py",
                             "//app/unc.py", "apple/no.py"])
    found, _ = paths(db, project="/legacy", prefix="app")
    assert set(found) == {"./app/dot.py", "/app/slash.py", "app//double.py",
                          "app/./inner.py", "app\\win.py"}


def test_windows_prefix_is_case_insensitive_and_keeps_legacy_projects(db):
    db.add_files(WIN, ["Internal/Core/a.go", "internal/core/b.go", "Drivers/c.go"])
    db.add_files("E:/Go/OpenList", ["INTERNAL/d.go"])
    found, count = paths(db, project=[WIN, "E:/Go/OpenList"], prefix="internal",
                         case_sensitive=False)
    assert set(found) == {"Internal/Core/a.go", "internal/core/b.go", "INTERNAL/d.go"}
    assert count == 3


def test_windows_prefix_with_uncased_non_ascii_directory_name(db):
    db.add_files(WIN, ["论文/a.md", "论文/b.md", "笔记/c.md"])
    found, _ = paths(db, project=WIN, prefix="论文", case_sensitive=False)
    assert found == ["论文/a.md", "论文/b.md"]


def test_windows_prefix_finds_a_cased_non_ascii_directory_either_way(db):
    db.add_files(WIN, ["Éclair/a.md", "éclair/b.md"])
    found, _ = paths(db, project=WIN, prefix="éclair", case_sensitive=False)
    assert set(found) == {"Éclair/a.md", "éclair/b.md"}


def test_windows_prefix_finds_a_name_that_only_case_folds_to_ascii(db):
    db.add_files(WIN, ["Straße/a.md", "strasse/b.md"])
    found, _ = paths(db, project=WIN, prefix="strasse", case_sensitive=False)
    assert set(found) == {"Straße/a.md", "strasse/b.md"}


def test_explicit_glob_with_literal_prefix_uses_the_same_range(db):
    found, _ = paths(db, pattern="app/api/**")
    assert found == ["app/api/deep/handler.py", "app/api/routes.py"]
    assert paths(db, pattern="app/", case_sensitive=True)[0] == [
        "app/README.md", "app/api/deep/handler.py", "app/api/routes.py", "app/main.py"]


def test_glob_with_no_literal_anchor_still_matches_everywhere(db):
    found, count = paths(db, pattern="*.md")
    assert found == ["README.md", "app/README.md", "docs/guide.md"]
    assert count == 8


def test_owner_wide_prefix_spans_scopes_but_not_owners(db):
    db.add_files("/other-owner-scope", ["app/api/x.py"], owner="someone-else")
    found, _ = paths(db, project=None, prefix="app/api")
    assert sorted(found) == ["app/api/deep/handler.py", "app/api/foreign.py",
                             "app/api/routes.py"]


def test_prefix_must_not_enumerate_the_scope_in_python(db, monkeypatch):
    db.add_files("/big", [f"noise/{i}.py" for i in range(3000)])
    db.add_files("/big", [f"wanted/{i}.py" for i in range(5)])
    seen = []
    original = scope_paths.stored_relative_path_identity

    def counting(path, *, case_sensitive):
        seen.append(path)
        return original(path, case_sensitive=case_sensitive)

    monkeypatch.setattr(scope_paths, "stored_relative_path_identity", counting)
    found, count = paths(db, project="/big", prefix="wanted")
    assert len(found) == 5 and count == 5
    assert len(seen) <= 5


@pytest.mark.parametrize("kwargs", [
    dict(prefix="app"),
    dict(prefix="app", pattern="api/**"),
    dict(pattern="app/api/**"),
    dict(pattern="app/api/"),
    dict(prefix="app", project=None),
])
def test_case_sensitive_lookups_search_an_index_range_and_never_scan(db, kwargs):
    with capture_catalog_plans(db) as plans:
        paths(db, **kwargs)
    assert plans
    assert unbounded_reads(plans) == [], plans
    assert any("rel_path>" in line for _, details in plans for line in details), plans


def test_case_insensitive_lookups_search_an_index_range_and_never_scan(db):
    db.add_files(WIN, ["Internal/a.go"])
    with capture_catalog_plans(db) as plans:
        paths(db, project=[WIN, "E:/Go/OpenList"], prefix="internal", case_sensitive=False)
    assert plans
    assert unbounded_reads(plans) == [], plans
    assert any("<expr>>" in line for _, details in plans for line in details), plans


def test_count_of_a_whole_scope_is_an_index_only_count(db):
    with capture_catalog_plans(db) as plans:
        found, count = paths(db, pattern="app/api/**")
    assert count == 8          # every live file with chunks in the scope
    assert unbounded_reads(plans) == [], plans


def test_bare_directory_glob_matches_that_name_at_any_depth(db):
    db.add_files(SCOPE, ["x/app/y.py", "xapp/z.py", "deep/er/app/w.py"])
    found, _ = paths(db, pattern="app/")
    assert set(found) == {
        "app/README.md", "app/api/deep/handler.py", "app/api/routes.py", "app/main.py",
        "x/app/y.py", "deep/er/app/w.py"}
    assert set(paths(db, pattern="README.md")[0]) == {"README.md", "app/README.md"}


def test_bare_name_glob_reads_only_rows_that_contain_the_name(db):
    with capture_catalog_plans(db) as plans:
        found, _ = paths(db, pattern="app/")
    assert found and unbounded_reads(plans) == []
    assert any("instr(" in statement for statement, _ in plans)


def test_bare_name_glob_on_a_windows_scope_is_case_insensitive(db):
    db.add_files(WIN, ["Src/APP/a.go", "app/b.go", "other/c.go", "Straße/APP/d.go"])
    found, _ = paths(db, project=WIN, pattern="App/", case_sensitive=False)
    assert set(found) == {"Src/APP/a.go", "app/b.go", "Straße/APP/d.go"}


def test_exact_file_glob_selects_that_file_and_not_its_siblings_or_lookalikes(db):
    db.add_files(SCOPE, ["app/main.py.bak", "app/main.pyc", "app/main.py/inner.txt"])
    assert paths(db, pattern="app/main.py")[0] == ["app/main.py", "app/main.py/inner.txt"]
