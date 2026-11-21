"""Cross-scope file identity for the write-time fan-out (方案2).

An index *scope* is a directory's absolute path (the MCP client forces this — W164);
a file inside it is addressed by its scope-relative path. The stable identity of a
physical file, shared across every scope that contains it, is its ABSOLUTE path:

    child scope  '.../transcripts'          + 'a.txt'
    parent scope '.../erp'                   + 'results/.../transcripts/a.txt'
    ── both resolve to ──▶ '.../transcripts/a.txt'

Both code KB and the journal use this to find the same file indexed under another
scope and refresh it, without a content-hash (which would falsely pair two different
files that happen to share bytes)."""
from __future__ import annotations

import ntpath
import posixpath
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath, PureWindowsPath

import pathspec

from core.paths import safe_relative_path


def _is_windows_device_path(path: str) -> bool:
    drive = PureWindowsPath(path or "").drive
    return drive.startswith("\\\\?\\") or drive.startswith("\\\\.\\")


def is_windows_scope(scope: str) -> bool:
    """Whether ``scope`` uses Windows drive or UNC path semantics.

    The backend often receives a Windows scope while itself running on Linux, so this
    cannot rely on ``os.name``.  A drive/UNC spelling is enough to select NTFS-style
    case-insensitive comparison without changing POSIX path semantics.
    """
    return PureWindowsPath(scope or "").is_absolute()


def canonicalize_scope_path(scope: str) -> str:
    """Return the durable namespace spelling for an absolute scope path.

    Windows paths are lexical-normalised with ``ntpath`` and case-folded component by
    component because NTFS is case-insensitive.  POSIX paths deliberately retain their
    case: ``/repo/Foo`` and ``/repo/foo`` can be distinct files there.  This helper is
    intentionally filesystem-free, so a compute server can canonicalise a path owned by
    another machine without pretending that it has that machine's drive mounted.
    """
    raw = (scope or "").replace("\\", "/")
    if not raw:
        return ""
    windows_path = PureWindowsPath(raw)
    if windows_path.drive and not windows_path.is_absolute():
        raise ValueError(
            "Windows scope must be absolute; use 'E:/repo', not a drive-relative "
            "spelling such as 'E:repo'."
        )
    if _is_windows_device_path(scope):
        raise ValueError(
            "Windows extended-length/device paths are not supported as index scopes; "
            "use the equivalent ordinary drive or UNC path spelling."
        )
    if is_windows_scope(raw):
        path = PureWindowsPath(ntpath.normpath(raw))
        normalized = path.as_posix().casefold()
        # PureWindowsPath owns drive/UNC parsing. Upper-casing the drive is only the
        # inventory display convention; case-folding all components supplies identity.
        if len(path.drive) == 2 and path.drive[1] == ":":
            return normalized[0].upper() + normalized[1:]
        return normalized
    return PurePosixPath(posixpath.normpath(raw)).as_posix()


def normalize_relative_path(path: str) -> str:
    """Normalise a scope-relative file path or glob without escaping its scope.

    Backslashes are accepted as Windows input.  Absolute paths and ``..`` components
    are rejected instead of being silently reinterpreted: ``path_glob`` must be relative
    to ``repo_root``/``scope_dir`` and never select a sibling scope by accident.
    """
    return safe_relative_path(
        path,
        what="path_glob/file path",
        allow_trailing_slash=True,
        reject_empty_segments=False,
    )


def _normalize_stored_relative_path(path: str) -> str | None:
    """Read one legacy storage spelling without relaxing the public filter contract."""
    raw = (path or "").replace("\\", "/")
    # Historical manifests occasionally prefixed a relative path with one slash. An
    # actual UNC path (two slashes) remains invalid and cannot widen a relative filter.
    if raw.startswith("/") and not raw.startswith("//"):
        raw = raw[1:]
    try:
        return normalize_relative_path(raw)
    except ValueError:
        return None


def has_dot_directory_segment(path: str) -> bool:
    """True if any DIRECTORY segment of a relative path starts with ".". The Journal doc
    walker never descends into dot-directories, so such a path (``.conversations/...``) is
    outside every scan envelope; a dot-FILE basename (``notes/.hidden.md``) is not."""
    segments = path.replace("\\", "/").split("/")[:-1]
    return any(s.startswith(".") and s not in {".", ".."} for s in segments)


def stored_relative_path_identity(path: str, *, case_sensitive: bool) -> str | None:
    """Comparison key for a stored relative path, including the legacy slash adapter."""
    normalized = _normalize_stored_relative_path(path)
    if normalized is None:
        return None
    return normalized if case_sensitive else normalized.casefold()


def file_abspath(scope: str, rel: str) -> str:
    """Absolute, forward-slash-normalised path of a file = its scope joined with its
    scope-relative path. Windows backslashes are normalised so a Windows client's
    scopes and a POSIX server agree on identity. ``rel`` is always scope-RELATIVE, so a
    leading slash is stripped — else posixpath.join would discard the scope entirely and
    resolve the file outside it (a false identity)."""
    # ``rel`` originated in old manifests that sometimes carried a stray leading slash;
    # retain the historical safe interpretation (still relative) here.  New API inputs
    # are validated by ``normalize_relative_path`` before they reach this helper.
    rel_normalized = _normalize_stored_relative_path(rel)
    if rel_normalized is None:
        raise ValueError("stored file path is not scope-relative")
    canonical_scope = canonicalize_scope_path(scope)
    if is_windows_scope(canonical_scope):
        return canonicalize_scope_path(
            str(PureWindowsPath(canonical_scope) / PureWindowsPath(rel_normalized)))
    return canonicalize_scope_path(
        str(PurePosixPath(canonical_scope) / PurePosixPath(rel_normalized)))


def path_matches(path: str, pattern: str, *, case_sensitive: bool = True) -> bool:
    """Match a stored relative path with pathspec's documented gitignore rules.

    ``pathspec`` supplies recursive globs, basename patterns, directory boundaries and
    exact-file semantics. The only compatibility adapter here is for old stored paths
    with one leading slash; request patterns always pass strict relative validation.
    """
    stored_path = _normalize_stored_relative_path(path)
    if stored_path is None:
        return False
    pattern = normalize_relative_path(pattern)
    if not case_sensitive:
        stored_path, pattern = stored_path.casefold(), pattern.casefold()
    matcher = pathspec.PathSpec.from_lines("gitignore", [pattern])
    return matcher.match_file(stored_path)


# --- scope comparison for the parent/child nested-scope overlap check --------------
# A scope is a directory's absolute path (W164). These normalise + compare scopes so the
# server can compute whether one indexed scope is a parent/child of another, answering the
# MCP nested-scope guard in KB instead of shipping the whole 9MB per-file inventory.

def normalize_scope(scope: str) -> str:
    """Comparison key for scope ancestry checks.

    ``canonicalize_scope_path`` preserves an uppercase Windows drive for inventory
    readability; comparison uses full case-folding so every NTFS spelling compares as the
    same directory.  Empty → ``/`` so a degenerate scope never becomes a prefix of all
    scopes.
    """
    normalized = canonicalize_scope_path(scope)
    if is_windows_scope(normalized):
        return normalized.casefold()
    return normalized.rstrip("/") or "/"


def scope_looks_like_path(scope: str) -> bool:
    """Is this scope an absolute path (POSIX ``/...`` or Windows ``c:/...``)? A bare
    LLM-invented project name is not, and must never participate in ancestor comparisons."""
    normalized = canonicalize_scope_path(scope)
    return normalized.startswith("/") or is_windows_scope(normalized)


def is_scope_ancestor(parent_norm: str, child_norm: str) -> bool:
    """True iff ``parent_norm`` is a strict ancestor of ``child_norm`` at a PATH BOUNDARY
    (so ``/erp`` is an ancestor of ``/erp/x`` but not of sibling ``/erp-backup/x``). Both
    args must already be normalised via :func:`normalize_scope`."""
    windows = is_windows_scope(parent_norm) or is_windows_scope(child_norm)
    path_type: type[PurePath] = PureWindowsPath if windows else PurePosixPath
    parent = path_type(parent_norm)
    child = path_type(child_norm)
    return parent != child and child.is_relative_to(parent)


def scope_ancestors(scope: str) -> list[tuple[str, str]]:
    """``(ancestor scope, path of ``scope`` below it)`` pairs, nearest first, up to the root.

    Only lexical: nothing here knows which ancestors are indexed. The remainder uses
    forward slashes and, for Windows scopes, the case-folded canonical spelling, so it can
    be compared with a stored relative path under :func:`stored_relative_path_identity`.
    A scope that is not an absolute path has no ancestors.
    """
    canonical = canonicalize_scope_path(scope)
    if not scope_looks_like_path(canonical):
        return []
    path_type: type[PurePath] = PureWindowsPath if is_windows_scope(canonical) else PurePosixPath
    path = path_type(canonical)
    return [
        (canonicalize_scope_path(parent.as_posix()), path.relative_to(parent).as_posix())
        for parent in path.parents
    ]


# --- literal anchors of a glob / directory, shared by every SQL path prefilter -------
# A search narrowed to a directory or ``path_glob`` must not read every stored path of the
# scope. The gitignore semantics stay in Python (``path_matches``); this only derives which
# stored paths could possibly match, in comparison spelling, so each store (journal titles,
# catalog rel_paths) can turn the same anchors into its own SQL.

_GLOB_META = frozenset("*?[")


@dataclass(frozen=True)
class PathReach:
    """Literal beginnings that bound the stored paths a glob or directory can accept.

    ``prefixes``: the path starts with one of these strings.
    ``entries``: the path equals one of these (a file) or lies under it (a directory).
    ``directories``: the path lies strictly under one of these.
    ``any_depth``: like ``entries`` but the name may sit at any directory depth.

    Strings are in canonical spelling (case-folded for case-insensitive scopes) and say
    nothing about legacy stored spellings (leading ``/``, ``./``, ``//``, backslashes):
    each store accommodates those in its own medium.
    """

    prefixes: tuple[str, ...] = ()
    entries: tuple[str, ...] = ()
    directories: tuple[str, ...] = ()
    any_depth: tuple[str, ...] = ()


def sql_foldable(text: str) -> bool:
    """Whether SQLite's ASCII-only ``lower()`` agrees with ``str.casefold`` on ``text``.

    True for ASCII and for scripts without case (CJK). False for a cased non-ASCII letter:
    ``lower('É')`` stays ``É`` in SQLite while the Python identity folds it to ``é``, so a
    range over ``lower(path)`` could miss rows the exact match accepts."""
    return all(
        ch.isascii() or (ch.casefold() == ch and ch.lower() == ch and ch.upper() == ch)
        for ch in text
    )


def glob_reach(pattern: str, *, case_sensitive: bool) -> PathReach | None:
    """Anchors of a ``path_glob``, or None when its reach is not provably bounded.

    Never narrower than ``path_matches``: a pattern whose first segment is a wildcard,
    a negation, a comment or padded with whitespace has no literal anchor.
    """
    if not pattern or pattern != pattern.strip() or pattern[0] in "!#":
        return None
    core = pattern.rstrip("/")
    if not case_sensitive:
        core = core.casefold()
        if not sql_foldable(core):
            return None
    if not core:
        return None
    anchored = "/" in core  # slash at the start or in the middle => relative to the scope root
    meta_at = next((i for i, ch in enumerate(core) if ch in _GLOB_META), None)
    if meta_at is not None:
        prefix = core[:meta_at]
        if not anchored or not prefix:
            return None
        return PathReach(prefixes=(prefix,))
    # a bare name matches that segment at any depth
    return PathReach(entries=(core,), any_depth=() if anchored else (core,))


def directory_reach(directory: str, *, case_sensitive: bool) -> PathReach | None:
    """Anchors of "every file below this scope-relative directory".

    ``directory`` is a literal path (``[`` or ``*`` in a name are ordinary characters),
    already normalised by :func:`normalize_relative_path`, without a trailing slash."""
    core = directory.strip("/")
    if not case_sensitive:
        core = core.casefold()
        if not sql_foldable(core):
            return None
    return PathReach(directories=(core,)) if core else None


def scopes_share_files(a: str, b: str) -> bool:
    """Whether a file indexed under scope ``a`` can be the SAME physical file as one
    indexed under scope ``b`` (identity is :func:`file_abspath`).

    Stored relative paths are scope-relative and never contain ``..``
    (:func:`normalize_relative_path`), so ``file_abspath(a, x) == file_abspath(b, y)``
    is only possible when the scopes are equal after normalisation (e.g. Windows case
    variants) or one is an ancestor of the other.  Siblings, including a shared string
    prefix such as ``/erp`` vs ``/erp-backup``, can never collide.

    A scope that cannot be canonicalised (drive-relative, device path) raises in
    ``file_abspath`` too, so it cannot hold a resolvable file and is unrelated to
    everything instead of failing the caller's whole enumeration.
    """
    try:
        na, nb = normalize_scope(a), normalize_scope(b)
    except ValueError:
        return False
    return na == nb or is_scope_ancestor(na, nb) or is_scope_ancestor(nb, na)
