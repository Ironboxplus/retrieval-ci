"""Contracts for revision manifests shared by source-indexing domains."""

from __future__ import annotations

import pytest

from core.indexing.revision_manifest import (
    RevisionFile,
    RevisionManifest,
    diff_revision_manifests,
)


def source_file(path: str, content_hash: str, *, language: str = "python") -> RevisionFile:
    return RevisionFile(
        path=path,
        content_hash=content_hash,
        language=language,
        content_role="implementation",
        size_bytes=100,
    )


def test_diff_classifies_added_changed_unchanged_and_removed_files():
    active = RevisionManifest(
        revision_id="active-commit",
        files=(
            source_file("kept.py", "same"),
            source_file("changed.py", "old"),
            source_file("removed.py", "removed"),
        ),
    )
    candidate = RevisionManifest(
        revision_id="candidate-commit",
        files=(
            source_file("kept.py", "same"),
            source_file("changed.py", "new"),
            source_file("added.py", "added"),
        ),
    )

    diff = diff_revision_manifests(active, candidate)

    assert [item.path for item in diff.added] == ["added.py"]
    assert [item.path for item in diff.changed] == ["changed.py"]
    assert [item.path for item in diff.unchanged] == ["kept.py"]
    assert [item.path for item in diff.removed] == ["removed.py"]


def test_diff_is_deterministic_when_source_discovery_order_changes():
    active = RevisionManifest(
        revision_id="active",
        files=(source_file("b.py", "b"), source_file("a.py", "a")),
    )
    candidate = RevisionManifest(
        revision_id="candidate",
        files=(source_file("a.py", "a"), source_file("b.py", "new-b")),
    )

    diff = diff_revision_manifests(active, candidate)

    assert [item.path for item in diff.changed] == ["b.py"]
    assert [item.path for item in diff.unchanged] == ["a.py"]


@pytest.mark.parametrize("path", ["/absolute.py", "../outside.py", "dir/../../outside.py", ""])
def test_manifest_rejects_unsafe_or_empty_relative_paths(path: str):
    with pytest.raises(ValueError, match="relative path"):
        RevisionManifest(revision_id="candidate", files=(source_file(path, "hash"),))


def test_manifest_rejects_duplicate_paths_even_when_discovery_spelling_differs():
    with pytest.raises(ValueError, match="duplicate path"):
        RevisionManifest(
            revision_id="candidate",
            files=(source_file("src\\main.py", "one"), source_file("src/main.py", "two")),
        )
