"""Unit tests for cross-scope path utilities (shared by code + journal path_glob)."""
from __future__ import annotations

from app.services.scope_paths import (
    canonicalize_scope_path,
    file_abspath,
    is_scope_ancestor,
    is_windows_scope,
    normalize_scope,
    normalize_relative_path,
    path_matches,
    scope_looks_like_path,
    scopes_share_files,
)


def test_file_abspath_normalises_windows_and_relative():
    assert file_abspath("C:/repo", "src/a.py") == "C:/repo/src/a.py"
    assert file_abspath("C:\\repo", "src\\a.py") == "C:/repo/src/a.py"
    # A parent scope + a nested rel path resolves to the same abspath as the child scope.
    assert file_abspath("/erp", "results/t/a.txt") == file_abspath("/erp/results/t", "a.txt")


def test_file_abspath_treats_leading_slash_rel_as_relative():
    # rel is ALWAYS scope-relative; a stray leading slash must NOT drop the scope (which
    # is what raw posixpath.join would do), else the file resolves outside its scope and
    # the nested-scope ancestor check silently misses a real overlap.
    assert file_abspath("/erp", "/notes/a.md") == "/erp/notes/a.md"
    assert file_abspath("C:/repo", "/src/a.py") == "C:/repo/src/a.py"


def test_path_matches_subtree_prefix():
    assert path_matches("sub/a.md", "sub/")
    assert path_matches("sub/deep/c.md", "sub/")
    assert not path_matches("b.md", "sub/")
    # A trailing-slash prefix must NOT match a sibling folder that merely shares the stem.
    assert not path_matches("subterfuge/x.md", "sub/")


def test_path_matches_subtree_prefix_without_trailing_slash():
    # A bare directory name is still a subtree prefix, but only at a path boundary.
    assert path_matches("sub/a.md", "sub")
    assert not path_matches("subterfuge/x.md", "sub")


def test_path_matches_single_file_is_exact_not_prefix():
    # The CRITICAL bug: a single-file pattern must match ONLY that file, never a longer
    # name that happens to start with it (documented single-file usage, e.g. "notes/a.md").
    assert path_matches("notes/a.md", "notes/a.md")
    assert not path_matches("notes/a.mdx", "notes/a.md")
    assert not path_matches("notes/a.md.bak", "notes/a.md")
    assert not path_matches("notes/a.markdown", "notes/a.md")


def test_path_matches_glob():
    assert path_matches("notes/a.md", "*.md")
    assert not path_matches("notes/a.txt", "*.md")
    assert path_matches("src/deep/x.go", "src/**/*.go")
    assert path_matches("x.md", "*.md")


def test_windows_scope_canonicalization_casefolds_all_components_and_dot_segments():
    """Windows scopes are one identity regardless of slash style or typed casing.

    The old implementation only normalised the drive letter, so ``E:/JETB/...`` and
    ``e:\\jetb\\...`` fragmented into separate Code KB projects despite naming the
    same directory on NTFS.
    """
    typed = "e:\\JETB\\Pythons\\Playground\\Octopus\\AxonHub\\LLM\\.\\"
    assert canonicalize_scope_path(typed) == "E:/jetb/pythons/playground/octopus/axonhub/llm"
    assert normalize_scope(typed) == "e:/jetb/pythons/playground/octopus/axonhub/llm"
    assert is_windows_scope(typed)


def test_windows_scope_canonicalization_supports_unc_paths():
    assert canonicalize_scope_path(
        "\\\\BuildServer\\SharedRepos\\AxonHub\\.\\LLM\\"
    ) == "//buildserver/sharedrepos/axonhub/llm"
    assert is_windows_scope(r"\\BuildServer\SharedRepos\AxonHub")


def test_drive_relative_windows_path_is_not_accepted_as_scope_relative():
    import pytest

    with pytest.raises(ValueError, match="scope-relative"):
        normalize_relative_path(r"E:conversion\planner")
    with pytest.raises(ValueError, match="Windows scope must be absolute.*E:/repo"):
        canonicalize_scope_path(r"E:repo")


def test_windows_device_scope_is_rejected_with_actionable_guidance():
    import pytest

    with pytest.raises(ValueError, match="extended-length/device.*ordinary drive or UNC"):
        canonicalize_scope_path(r"\\?\C:\Repo")
    with pytest.raises(ValueError, match="extended-length/device.*ordinary drive or UNC"):
        canonicalize_scope_path(r"\\?\UNC\server\share\Repo")


def test_windows_path_glob_accepts_backslashes_and_case_insensitive_directory_names():
    # The indexed rel_path keeps its display casing, but a Windows caller may type either
    # casing and either separator. The match must still be a boundary-safe subtree match.
    assert path_matches(
        "Conversion/Planner/Compact.go", "conversion\\planner\\", case_sensitive=False)
    assert path_matches(
        "Conversion/Planner/Compact.go", r"CONVERSION\**\*.GO", case_sensitive=False)
    assert not path_matches(
        "Conversion/Plannerish/Compact.go", "conversion\\planner\\", case_sensitive=False)


def test_path_matches_legacy_stored_leading_slash_without_treating_request_as_absolute():
    """Old manifests stored some rel paths with `/`; a new relative filter must not 500."""
    assert path_matches(
        "/Conversion/Planner/Compact.go",
        "conversion\\planner\\",
        case_sensitive=False,
    )
    # A malformed historical stored value cannot widen the request; it simply does not
    # match. The filter itself remains strict and continues to reject absolute input.
    assert not path_matches("E:/other/repo/Compact.go", "conversion/")


def test_normalize_relative_path_rejects_absolute_or_escaping_filters():
    assert normalize_relative_path(".\\conversion\\planner\\") == "conversion/planner/"
    import pytest

    with pytest.raises(ValueError, match="scope-relative"):
        normalize_relative_path(r"E:\\repo\\conversion")
    with pytest.raises(ValueError, match="must not contain '..'"):
        normalize_relative_path("conversion/../planner/")


# --- scope-comparison helpers (server-side parent/child overlap check) ---

def test_normalize_scope_lowercases_windows_drive_and_strips_trailing_slash():
    # Entire Windows paths are case-insensitive; normalize_scope is the comparison key.
    assert normalize_scope("C:\\Users\\Arc\\repo\\") == "c:/users/arc/repo"
    assert normalize_scope("/data/arc/retrival/") == "/data/arc/retrival"
    assert normalize_scope("") == "/"


def test_scope_looks_like_path():
    assert scope_looks_like_path("/data/arc/repo")
    assert scope_looks_like_path("C:\\Users\\Arc\\repo")
    # A bare LLM-invented name (not an abspath) is not a path scope.
    assert not scope_looks_like_path("my-project")
    assert not scope_looks_like_path("retrival")


def test_is_scope_ancestor_only_at_path_boundary():
    # Both args are already normalised (lowercased drive, no trailing slash).
    assert is_scope_ancestor("/erp", "/erp/results/t")
    assert is_scope_ancestor("c:/users/arc", "c:/users/arc/repo")
    # Same scope is never its own ancestor.
    assert not is_scope_ancestor("/erp", "/erp")
    # A sibling that merely shares a stem prefix is NOT an ancestor.
    assert not is_scope_ancestor("/erp", "/erp-backup/x")
    # Reversed (child, parent) is not an ancestor.
    assert not is_scope_ancestor("/erp/results", "/erp")


# --- scopes_share_files: which scopes can hold the SAME physical file ---

def test_scopes_share_files_equal_ancestor_and_descendant():
    assert scopes_share_files("/erp", "/erp")
    assert scopes_share_files("/erp", "/erp/results/t")
    assert scopes_share_files("/erp/results/t", "/erp")
    assert scopes_share_files("C:/w/erp", "C:/w/erp/sub")
    assert scopes_share_files("C:/w/erp/sub", "C:/w/erp")


def test_scopes_share_files_windows_case_and_separator_variants_are_the_same_scope():
    assert scopes_share_files("C:/w/erp", "c:\W\ERP")
    assert scopes_share_files("c:\W\ERP\\", "C:/w/erp/sub")
    assert scopes_share_files("C:/w/erp/SUB", "c:/W/ERP")


def test_scopes_share_files_posix_case_is_significant():
    assert not scopes_share_files("/repo/Foo", "/repo/foo")
    assert not scopes_share_files("/Repo", "/repo/x")


def test_scopes_share_files_rejects_siblings_and_unrelated_scopes():
    # Shared string prefix is not a shared directory.
    assert not scopes_share_files("/erp", "/erp-backup")
    assert not scopes_share_files("/erp-backup/x", "/erp")
    assert not scopes_share_files("C:/w/erp", "C:/w/erp-backup")
    assert not scopes_share_files("/a/b", "/a/c")
    assert not scopes_share_files("C:/w/erp", "D:/w/erp")


def test_scopes_share_files_relative_names_compare_lexically():
    assert scopes_share_files("proj", "proj")
    assert scopes_share_files("proj", "proj/sub")
    assert not scopes_share_files("proj", "proj-2")
    assert not scopes_share_files("proj", "other")


def test_scopes_share_files_agrees_with_file_abspath_identity():
    # The invariant the fan-out relies on: every pair that resolves to one absolute path
    # is reported as related.
    pairs = [
        (("C:/w/erp", "sub/a.py"), ("C:/w/erp/sub", "a.py")),
        (("c:/W/ERP", "a.py"), ("C:\w\erp", "a.py")),
        (("/erp", "x/y/a.py"), ("/erp/x", "y/a.py")),
    ]
    for (scope_a, rel_a), (scope_b, rel_b) in pairs:
        assert file_abspath(scope_a, rel_a) == file_abspath(scope_b, rel_b)
        assert scopes_share_files(scope_a, scope_b)


def test_scopes_share_files_malformed_scope_is_unrelated_not_an_error():
    # file_abspath/canonicalize raise ValueError on these, so such a scope can never
    # resolve a file to an absolute path; it cannot share one with anything.
    assert not scopes_share_files("E:repo", "E:repo")
    assert not scopes_share_files("E:repo", "E:/repo")
    assert not scopes_share_files("//?/C:/repo", "C:/repo")
