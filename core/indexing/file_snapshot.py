"""Shared, filesystem-backed incremental snapshot primitives.

The Code KB and Journal have different chunkers, collections and latest-wins
rules.  They do share one safety-critical concern: decide which *files* changed
before a client uploads bodies or a service mutates an index.  This module owns
that decision and deliberately contains no database, Qdrant or chunking code.
"""
from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import ntpath
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence


def _is_windows_root(root_dir: str) -> bool:
    """Whether a scope uses Windows path identity, even when inspected elsewhere."""
    return PureWindowsPath(root_dir).is_absolute()


def _normalise_rel_path(value: str) -> str:
    windows = PureWindowsPath(value)
    if windows.is_absolute() or windows.drive or windows.root:
        raise ValueError(f"relative path must not be absolute, drive-relative, or UNC: {value!r}")
    normalized = value.replace("\\", "/").strip("/")
    if not normalized or normalized == ".":
        raise ValueError("relative path must name a file")
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"relative path escapes the indexed root: {value!r}")
    return normalized


@dataclass(frozen=True)
class ScanSpec:
    """The non-destructive envelope of one directory snapshot.

    ``max_depth`` is the depth below ``root_dir`` (zero means root files only).
    Path matching deliberately follows the scope's filesystem semantics: Windows
    scopes are case-insensitive, while POSIX scopes remain case-sensitive.
    """

    root_dir: str
    patterns: tuple[str, ...] = ("*",)
    max_depth: int | None = None

    def __post_init__(self) -> None:
        if not self.root_dir:
            raise ValueError("root_dir must not be empty")
        if not self.patterns:
            raise ValueError("patterns must not be empty")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")

    @property
    def case_sensitive(self) -> bool:
        return not _is_windows_root(self.root_dir)

    def normalize_relative_path(self, value: str) -> str:
        return _normalise_rel_path(value)

    def matches(self, relative_path: str) -> bool:
        """True only when the file lies in this scan's pattern/depth envelope."""
        rel = self.normalize_relative_path(relative_path)
        if self.max_depth is not None and rel.count("/") > self.max_depth:
            return False
        basename = rel.rsplit("/", 1)[-1]
        if self.case_sensitive:
            return any(fnmatch.fnmatchcase(basename, pattern) for pattern in self.patterns)
        folded = basename.casefold()
        return any(fnmatch.fnmatchcase(folded, pattern.casefold()) for pattern in self.patterns)


@dataclass(frozen=True)
class FileCandidate:
    """A file observed during a snapshot, with a byte-level content identity.

    A read failure is represented explicitly instead of disappearing from the
    candidate list.  That distinction prevents a transient lock or permission
    error from being treated as a deletion during reconciliation.
    """

    abs_path: str
    rel_path: str
    content_hash: str | None
    size_bytes: int | None
    read_error: str | None = None

    @property
    def readable(self) -> bool:
        return self.content_hash is not None and self.read_error is None

    @classmethod
    def from_path(cls, path: str | Path, root_dir: str | Path) -> "FileCandidate":
        raw_path = Path(path)
        root = Path(root_dir)
        try:
            absolute = raw_path.resolve(strict=True)
            resolved_root = root.resolve(strict=True)
            rel_path = _normalise_rel_path(os.path.relpath(absolute, resolved_root))
            stat = absolute.stat()
            digest = hashlib.sha256()
            with absolute.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            return cls(
                abs_path=str(absolute), rel_path=rel_path,
                content_hash=digest.hexdigest(), size_bytes=stat.st_size,
            )
        except (OSError, ValueError) as exc:
            # Preserve the best available, normalized relative identity so callers
            # can keep an already-indexed file live rather than sweep it.
            rel_path = os.path.relpath(raw_path, root).replace("\\", "/")
            try:
                rel_path = _normalise_rel_path(rel_path)
            except ValueError as identity_error:
                # Surface an unusable path as unreadable; callers then disable
                # replacement/reconcile instead of manufacturing an unsafe key.
                rel_path = "__invalid_relative_path__"
                exc = identity_error
            return cls(
                abs_path=str(raw_path), rel_path=rel_path,
                content_hash=None, size_bytes=None, read_error=str(exc),
            )


def _known_hash(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("content_hash")
        return candidate if isinstance(candidate, str) and candidate else None
    return None


@dataclass(frozen=True)
class SnapshotDiff:
    """Diff between a new file snapshot and the server's known manifest.

    ``replace_safe`` is intentionally conservative.  Existing index state is
    never swept after an unreadable walk, an empty scan, or a wholly disjoint
    snapshot.  Explicit delete endpoints remain the only way to clear a scope.
    """

    added: tuple[str, ...]
    changed: tuple[str, ...]
    unchanged: tuple[str, ...]
    removed: tuple[str, ...]
    unreadable_keep: tuple[str, ...]
    records: tuple[FileCandidate, ...]
    replace_safe: bool
    replace_block_reason: str | None

    @property
    def live_paths(self) -> tuple[str, ...]:
        return tuple(sorted((*self.added, *self.changed, *self.unchanged, *self.unreadable_keep)))

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(sorted((*self.added, *self.changed)))

    @classmethod
    def from_candidates(
        cls,
        candidates: Sequence[FileCandidate],
        known: Mapping[str, Any],
        *,
        case_sensitive: bool = True,
    ) -> "SnapshotDiff":
        identity = (lambda path: path) if case_sensitive else (lambda path: path.casefold())
        by_identity: dict[str, list[FileCandidate]] = {}
        for candidate in candidates:
            by_identity.setdefault(identity(candidate.rel_path), []).append(candidate)
        collision = any(len(rows) > 1 for rows in by_identity.values())
        current = {identity(candidate.rel_path): candidate for candidate in candidates}
        known_by_identity = {identity(path): value for path, value in known.items()}
        known_paths = set(known_by_identity)
        readable_paths = {
            rel for rel, candidate in current.items() if candidate.readable
        }
        unreadable_keep = {
            rel for rel, candidate in current.items()
            if not candidate.readable and rel in known_paths
        }
        added = {
            rel for rel in readable_paths if rel not in known_paths
        }
        changed = {
            rel for rel in readable_paths & known_paths
            if current[rel].content_hash != _known_hash(known_by_identity[rel])
        }
        unchanged = {
            rel for rel in readable_paths & known_paths
            if current[rel].content_hash == _known_hash(known_by_identity[rel])
        }
        removed = known_paths - readable_paths - unreadable_keep

        block_reason: str | None = None
        if collision:
            block_reason = "case-insensitive path identity collision"
        elif unreadable_keep:
            block_reason = "unreadable files"
        elif known_paths and not readable_paths:
            block_reason = "empty snapshot"
        elif known_paths and readable_paths.isdisjoint(known_paths):
            block_reason = "disjoint snapshot"
        return cls(
            added=tuple(sorted(current[rel].rel_path for rel in added)),
            changed=tuple(sorted(current[rel].rel_path for rel in changed)),
            unchanged=tuple(sorted(current[rel].rel_path for rel in unchanged)),
            removed=tuple(sorted(known_path for known_path in known
                                 if identity(known_path) in removed)),
            unreadable_keep=tuple(sorted(current[rel].rel_path for rel in unreadable_keep)),
            records=tuple(current[rel] for rel in sorted(current)),
            replace_safe=block_reason is None, replace_block_reason=block_reason,
        )
