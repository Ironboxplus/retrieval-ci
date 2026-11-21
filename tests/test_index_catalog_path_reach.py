"""The literal anchors every SQL path prefilter is derived from (scope_paths.PathReach)."""
from __future__ import annotations

import pytest

from app.services.scope_paths import (
    PathReach, directory_reach, glob_reach, scope_ancestors, sql_foldable,
)
from core.db.models import CASEFOLD_TO_ASCII_CODEPOINTS


def test_casefold_to_ascii_codepoints_match_the_interpreters_unicode_tables():
    computed = tuple(
        ord(ch) for ch in map(chr, range(128, 0x30000))
        if any(ord(part) < 128 for part in ch.casefold()))
    assert CASEFOLD_TO_ASCII_CODEPOINTS == computed


@pytest.mark.parametrize("pattern, expected", [
    ("src/Magpie*/**", PathReach(prefixes=("src/Magpie",))),
    ("src/peers/", PathReach(entries=("src/peers",))),
    ("src/a.py", PathReach(entries=("src/a.py",))),
    ("README.md", PathReach(entries=("README.md",), any_depth=("README.md",))),
    ("docs/", PathReach(entries=("docs",), any_depth=("docs",))),
    ("/abs/anchored.py", PathReach(entries=("/abs/anchored.py",))),
])
def test_glob_reach_of_bounded_patterns(pattern, expected):
    assert glob_reach(pattern, case_sensitive=True) == expected


@pytest.mark.parametrize("pattern", [
    "*.md", "**/test_*.go", "?x/a.py", "!keep.py", "#comment", " padded", "/", "[ab]/c",
])
def test_glob_reach_of_unbounded_patterns_is_none(pattern):
    assert glob_reach(pattern, case_sensitive=True) is None


def test_glob_reach_folds_for_case_insensitive_scopes():
    assert glob_reach("Src/Core/**", case_sensitive=False) == PathReach(
        prefixes=("src/core/",))
    assert glob_reach("Éclair/x/**", case_sensitive=False) is None      # cased non-ASCII
    assert glob_reach("Straße/x/**", case_sensitive=False) == PathReach(
        prefixes=("strasse/x/",))        # folds to ASCII; the folding partial index covers rows
    assert glob_reach("论文/x/**", case_sensitive=False) == PathReach(prefixes=("论文/x/",))


def test_directory_reach_is_literal_and_strictly_below():
    assert directory_reach("we[ir]d/dir", case_sensitive=True) == PathReach(
        directories=("we[ir]d/dir",))
    assert directory_reach("App/Api", case_sensitive=False) == PathReach(
        directories=("app/api",))
    assert directory_reach("Éclair", case_sensitive=False) is None
    assert directory_reach("", case_sensitive=True) is None


def test_sql_foldable_accepts_ascii_and_uncased_scripts_only():
    assert sql_foldable("abc/DEF") and sql_foldable("论文/ノート")
    assert not sql_foldable("é") and not sql_foldable("Ω")


def test_scope_ancestors_nearest_first_with_the_remaining_path():
    assert scope_ancestors("/a/b/c") == [("/a/b", "c"), ("/a", "b/c"), ("/", "a/b/c")]
    assert scope_ancestors("/a") == [("/", "a")]
    assert scope_ancestors("/") == []


def test_scope_ancestors_of_windows_scopes_use_the_canonical_spelling():
    assert scope_ancestors("e:\\Go\\OpenList\\Internal") == [
        ("E:/go/openlist", "internal"), ("E:/go", "openlist/internal"),
        ("E:/", "go/openlist/internal")]


def test_scope_ancestors_of_names_that_are_not_absolute_paths_is_empty():
    assert scope_ancestors("retrival") == [] and scope_ancestors("") == []
