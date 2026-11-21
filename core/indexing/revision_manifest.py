"""Immutable file manifests used by latest-wins indexing workflows.

The types in this module intentionally carry only file-level facts.  They are
shared infrastructure for any indexed source that needs to compare an active
revision with a candidate revision; source ownership, storage, and promotion
remain domain concerns outside ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.paths import safe_relative_path


@dataclass(frozen=True, slots=True)
class RevisionFile:
    """One discovered source file and the content identity used for indexing."""

    path: str
    content_hash: str
    language: str
    content_role: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RevisionManifest:
    """The normalized file inventory for one candidate or active revision."""

    revision_id: str
    files: tuple[RevisionFile, ...]

    def __post_init__(self) -> None:
        if not self.revision_id.strip():
            raise ValueError("revision_id must not be empty")

        normalized_files: list[RevisionFile] = []
        seen_paths: set[str] = set()
        for file in self.files:
            normalized_path = _normalize_relative_path(file.path)
            if normalized_path in seen_paths:
                raise ValueError(f"duplicate path in revision manifest: {normalized_path}")
            if not file.content_hash.strip():
                raise ValueError(f"content_hash must not be empty for {normalized_path}")
            if file.size_bytes < 0:
                raise ValueError(f"size_bytes must not be negative for {normalized_path}")

            seen_paths.add(normalized_path)
            normalized_files.append(replace(file, path=normalized_path))

        object.__setattr__(self, "files", tuple(normalized_files))

    def files_by_path(self) -> dict[str, RevisionFile]:
        """Return the normalized file inventory keyed by its relative path."""

        return {file.path: file for file in self.files}


@dataclass(frozen=True, slots=True)
class RevisionManifestDiff:
    """Content-based change classification from an active to candidate revision."""

    added: tuple[RevisionFile, ...]
    changed: tuple[RevisionFile, ...]
    unchanged: tuple[RevisionFile, ...]
    removed: tuple[RevisionFile, ...]


def diff_revision_manifests(
    active: RevisionManifest,
    candidate: RevisionManifest,
) -> RevisionManifestDiff:
    """Classify files deterministically using their indexing content hash.

    Metadata changes do not require re-embedding a file.  The owning domain can
    still persist those metadata changes while reusing the active vectors.
    """

    active_files = active.files_by_path()
    candidate_files = candidate.files_by_path()

    added = tuple(candidate_files[path] for path in sorted(candidate_files.keys() - active_files.keys()))
    removed = tuple(active_files[path] for path in sorted(active_files.keys() - candidate_files.keys()))

    changed_paths = sorted(
        path
        for path in candidate_files.keys() & active_files.keys()
        if candidate_files[path].content_hash != active_files[path].content_hash
    )
    unchanged_paths = sorted(
        path
        for path in candidate_files.keys() & active_files.keys()
        if candidate_files[path].content_hash == active_files[path].content_hash
    )

    return RevisionManifestDiff(
        added=added,
        changed=tuple(candidate_files[path] for path in changed_paths),
        unchanged=tuple(candidate_files[path] for path in unchanged_paths),
        removed=removed,
    )


def _normalize_relative_path(path: str) -> str:
    """Normalize a portable relative file path and reject scope escapes."""

    return safe_relative_path(path, what="manifest relative path")
