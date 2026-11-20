#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.29,<1.30", "httpx[socks]>=0.27", "chonkie[code]>=1.6,<1.7", "tree-sitter-language-pack>=1.8,<1.13", "tree-sitter>=0.23,<0.26", "tomli>=2.0; python_version < '3.11'", "watchfiles>=1.1"]
# ///
"""Retrieval MCP — a self-contained, installable MCP server.

Package release: 0.5.15.

THIS file is standalone: it only talks to the Retrieval REST API over HTTP, so it
installs anywhere with no checkout, no GPU, no models. Paper search + journal + ACE
playbook + code search are pure REST (``mcp`` + ``httpx``). ``index_code`` AST-chunks
your repo LOCALLY before uploading the chunks, so it also pulls ``chonkie[code]`` (a
maintained structure-aware chunker that bundles tree-sitter grammars, no runtime
download) — the heavier dep; everything else is REST. Only chunks leave the machine.

Default target is the compute box directly:  http://10.100.100.111:8000
On the lab LAN that is trusted as an internal admin, so no key is needed. Off-LAN,
point RETRIEVAL_API_URL at the public gateway and set RETRIEVAL_API_KEY (sk-...).

Journal / code / playbook scope is the current directory's ABSOLUTE PATH (forced by
this client), so the same location always maps to one namespace — no hand-typed project
name to fragment. Point the client at the dir you mean via each tool's dir/repo_root arg.

Config (env):
  RETRIEVAL_API_URL     backend base URL   (default http://10.100.100.111:8000)
  RETRIEVAL_API_KEY     sk-key for the paid gateway (optional; unset on LAN)

  (The misspelled legacy names RETRIVAL_API_URL / RETRIVAL_API_KEY are still honored
  as a fallback so existing configs keep working.)

Run:
  uvx retrieval-mcp                # or: uv run retrieval_mcp.py
  python retrieval_mcp.py          # if mcp + httpx are already installed
"""
from __future__ import annotations

# Direct execution remains supported for older ``uvx``/script configurations.
# Dispatch the ordinary client commands before importing FastMCP, chonkie, or any
# of the 32 decorated tools: every such process is only a stdio-to-daemon bridge.
if __name__ == "__main__":
    import sys as _bootstrap_sys

    _bootstrap_args = _bootstrap_sys.argv[1:]
    if not _bootstrap_args or _bootstrap_args[0] in {"start", "status", "stop"}:
        from retrieval_mcp_daemon import (
            DEFAULT_PLUGIN_BACKEND_URL as _DEFAULT_PLUGIN_BACKEND_URL,
            run_thin_cli as _run_thin_cli,
        )

        raise SystemExit(_run_thin_cli(
            _bootstrap_args,
            serve_command=[__file__],
            backend_default_url=_DEFAULT_PLUGIN_BACKEND_URL,
        ))
    if _bootstrap_args[0] == "serve":
        # WMI has no stdout/stderr descriptor parameters.  Its launcher supplies
        # a private runtime log path before this module imports the full tools.
        from retrieval_mcp_daemon import redirect_daemon_output_from_environment as _redirect_daemon_output

        _redirect_daemon_output()

import asyncio
import contextvars
import fnmatch
import glob
import hashlib
import inspect
import json
import logging
import math
import ntpath
import os
import posixpath
import re
import socket
import shutil
import subprocess
import tempfile
import sys
import uuid
from contextlib import suppress
from pathlib import Path, PurePosixPath, PureWindowsPath
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
from dataclasses import dataclass
from typing import Annotated, Any, Awaitable, Callable, NamedTuple, Sequence
from urllib.parse import quote

import httpx
from chonkie import CodeChunker  # EAGER: a lazy import inside asyncio.to_thread while
# FastMCP's loop is running deadlocks on the import lock — index_code hangs forever
# with "chunking <first file>" as the last log line (W155c, minimal-repro verified).
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from watchfiles import awatch

# STDERR only — stdout is the MCP JSON-RPC channel, a stray print there corrupts the
# protocol. Every file is logged BEFORE it is parsed, so if a grammar wedges or crashes
# the process, the last stderr line names the exact culprit file (W155 forensics).
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retrieval_mcp")


# NOTE: intentionally duplicated in mcp_server/server.py — this file ships as a
# zero-dependency standalone package (pyproject bundles ONLY this file), so it must
# not import from the repo. Keep the two copies identical.
def _env(*names: str, default: str | None = None) -> str | None:
    """First set value among `names` (new spelling first, legacy fallback), else default."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


BASE_URL = (_env("RETRIEVAL_API_URL", "RETRIVAL_API_URL",
                 default="http://10.100.100.111:8000") or "").rstrip("/")
API_KEY = _env("RETRIEVAL_API_KEY", "RETRIVAL_API_KEY")
_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
# Layered timeouts (not one bare float): connect/pool fail fast so a stuck proxy or
# dead peer cannot pin a tool call for the full bulk-read window. Read/write stay
# generous for large paper artifacts and index uploads (W102 / hang forensics 2026-07-23).
_TIMEOUT_CONNECT_S = 5.0
_TIMEOUT_POOL_S = 5.0
_TIMEOUT_READ_S = 180.0
_TIMEOUT_WRITE_S = 180.0
_TIMEOUT = _TIMEOUT_READ_S  # legacy alias for callers/tests that still read the float
# Interactive search tools must not wait for the full bulk read window; host tool
# timeouts are often multi-minute and feel like a hang (search_code 2026-07-23;
# search_papers concurrent hang 2026-07-25).
_SEARCH_CODE_TIMEOUT_S = 45.0
_SEARCH_PAPERS_TIMEOUT_S = 60.0
# A duplicate scan issues one nearest-neighbour query per indexed chunk: ~29s for
# 8.6k chunks when measured on 2026-09-01. It is a deliberate, explicitly invoked
# analysis, not an interactive lookup, so it gets a much longer budget than search.
_FIND_DUPLICATES_TIMEOUT_S = 300.0
# Cap MCP stdio payload: full AST chunks are fine; methviz-style multi-KB one-liners
# and multi-hit dumps previously left the host tool call "in progress" for minutes.
_CODE_HIT_CODE_CHARS = 4000
_CODE_HIT_TOTAL_CHARS = 24000
_PAPER_ARTIFACT_ATTEMPTS = 3
_PAPER_ARTIFACT_RETRY_BASE_SECONDS = 0.5
_PAPER_ARTIFACT_RANGE_BYTES = 4 * 1024 * 1024
_PAPER_STREAM_CHUNK_BYTES = 64 * 1024
_PAPER_BATCH_CONCURRENCY = 2
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)

# journal_index_dir POSTs each file as a journal entry the backend embeds synchronously;
# fan them out with bounded concurrency so a big docs tree isn't a serial round-trip chain,
# without stampeding the shared embedder.
_JOURNAL_INDEX_CONCURRENCY = 8
# A batch reduces MCP tool round-trips, never the per-query backend work.  This
# is deliberately an in-flight cap only: it does not truncate the caller's list
# or share candidates/top_k/rerank budget between queries.
_QUERY_BATCH_CONCURRENCY = 2

JournalChunkCount = Annotated[
    int,
    Field(
        ge=1,
        le=20,
        description=(
            "Number of chunks returned per journal entry, not a character or token "
            "budget. Valid range: 1 through 20."
        ),
    ),
]

PaperBatchIds = Annotated[
    list[str],
    Field(
        min_length=1,
        max_length=10,
        description="Canonical paper IDs to download in one batch (1 through 10).",
    ),
]


def _validate_journal_chunk_count(value: int) -> int:
    if not 1 <= value <= 20:
        raise ValueError(
            "chunk_limit is the number of chunks returned per entry and must be "
            f"1 through 20; got {value}. Use chunk_limit=20 for the maximum."
        )
    return value


def _is_windows_absolute_path(path: str) -> bool:
    return PureWindowsPath(path or "").is_absolute()


def _is_windows_device_path(path: str) -> bool:
    drive = PureWindowsPath(path or "").drive
    return drive.startswith("\\\\?\\") or drive.startswith("\\\\.\\")


def _canonical_windows_scope(path: str) -> str:
    if _is_windows_device_path(path):
        raise ValueError(
            "Windows extended-length/device paths are not supported as index scopes; "
            "use the equivalent ordinary drive or UNC path spelling."
        )
    windows_path = PureWindowsPath(ntpath.normpath(path))
    if not windows_path.is_absolute():
        raise ValueError(f"Windows scope must be absolute; got {path!r}")
    normalized = windows_path.as_posix().casefold()
    if len(windows_path.drive) == 2 and windows_path.drive[1] == ":":
        return normalized[0].upper() + normalized[1:]
    return normalized


_CLIENT_CWD_LOOKUP: Callable[[], str] | None = None


def _session_cwd() -> str:
    """Directory of the MCP client whose session is making this call.

    Deliberately NOT ``os.getcwd()``.  These tools run inside an account-level singleton
    daemon shared by every project, so the process directory names whichever project
    happened to start it.  Resolving a relative scope there pointed one project at
    another: `journal_search`/`journal_recent` returned a foreign repo's notes and
    `journal_record` MISFILED entries into that repo's namespace (2026-09-06 report).
    The shim sends its own directory in the raw handshake; the daemon binds it per
    session.  Falls back to the process directory when no session is bound.
    """
    global _CLIENT_CWD_LOOKUP
    if _CLIENT_CWD_LOOKUP is None:
        try:
            from retrieval_mcp_daemon import client_working_directory
        except ImportError:
            _CLIENT_CWD_LOOKUP = os.getcwd
        else:
            _CLIENT_CWD_LOOKUP = client_working_directory
    return _CLIENT_CWD_LOOKUP()


def _session_abspath(path: str) -> str:
    """``os.path.abspath`` with the calling client's directory as the base.

    The base may use a different path flavour than the daemon's own OS, so join in the
    base's flavour; letting ``os.path`` handle a drive-letter base on POSIX would
    rewrite it into a fictitious relative path (the same hazard `_scope_path` guards).
    """
    expanded = os.path.expanduser(path)
    if _is_windows_absolute_path(expanded):
        return ntpath.normpath(expanded)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    base = _session_cwd()
    if _is_windows_absolute_path(base):
        return ntpath.normpath(ntpath.join(base, expanded))
    if posixpath.isabs(base):
        return posixpath.normpath(posixpath.join(base, expanded.replace(os.sep, "/")))
    return os.path.abspath(os.path.join(base, expanded))


def _session_save_path(save_path: str | None) -> str:
    """Resolve a caller-supplied artifact destination against the calling client.

    Downloads have the same hazard as scopes: left relative, `save_path` would resolve
    inside the account-level singleton daemon and drop one project's PDFs into whichever
    project happened to start that daemon.  A trailing separator survives the join
    because that is what marks a directory target for `_paper_artifact_destination`.
    """
    raw = save_path if save_path is not None else "."
    resolved = _session_abspath(raw)
    return resolved + "/" if raw.endswith(("/", "\\")) else resolved


def _scope_path(path: str = ".") -> str:
    """Canonical absolute scope, including cross-host Windows paths.

    A search may intentionally name a scope indexed from another machine.  Recognise
    ``E:\\...``/UNC syntax before making the path absolute so a Linux process never
    rewrites it into a fictitious local relative path. Windows component casing is
    case-folded; POSIX casing remains intact because it can identify different
    directories.  A relative scope resolves against the CALLING client's directory —
    see `_session_cwd` for why the daemon's own directory is the wrong base.
    """
    expanded = os.path.expanduser(path)
    if _is_windows_absolute_path(expanded):
        return _canonical_windows_scope(expanded)
    absolute = _session_abspath(expanded)
    if _is_windows_absolute_path(absolute):
        return _canonical_windows_scope(absolute)
    return PurePosixPath(posixpath.normpath(absolute.replace(os.sep, "/"))).as_posix()


def _normalize_path_glob(path_glob: str) -> str:
    """Accept Windows separators for a scope-relative code path filter only."""
    raw = (path_glob or "").replace("\\", "/")
    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(raw)
    if windows_path.drive or windows_path.root or posix_path.is_absolute():
        raise ValueError(
            "path_glob must be scope-relative to repo_root; use 'conversion/**', not "
            "an absolute path such as 'E:/repo/conversion'. Backslashes and Windows "
            "directory casing are accepted for relative filters."
        )
    trailing_slash = raw.endswith("/")
    if ".." in posix_path.parts:
        raise ValueError("path_glob must not contain '..'; it is relative to repo_root.")
    normalized = posix_path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("path_glob must name a scope-relative file, directory, or glob.")
    return normalized + "/" if trailing_slash else normalized


def _local_repo_root(path: str) -> str | None:
    """Return a locally mounted source directory, never inventing one for a remote scope."""
    candidate = _session_abspath(path)
    return candidate if os.path.isdir(candidate) else None


def _format_conflict_list(conflicts: list[dict[str, str]], label_fn, limit: int = 5) -> str:
    shown = []
    for conflict in conflicts[:limit]:
        host = f" on host {conflict['host']}" if conflict.get("host") else ""
        shown.append(f"{label_fn(conflict)}{host}")
    more = f"; +{len(conflicts) - limit} more" if len(conflicts) > limit else ""
    return "; ".join(shown) + more


def _format_nested_scope_block(project: str, kind: str, conflicts: list[dict[str, str]]) -> str:
    frag = _format_conflict_list(conflicts, lambda c: f"scope '{c['project']}'")
    return (f"{kind.capitalize()} scope '{project}' is already covered by ancestor {frag}. "
            f"Indexing this child scope adds no coverage over the parent and would duplicate "
            f"a namespace, so it was not indexed. To index only this subtree, delete the "
            f"parent scope first.")


def _format_gitignore_widening_warning(project: str, kind: str,
                                       conflicts: list[dict[str, str]]) -> str:
    frag = _format_conflict_list(conflicts, lambda c: f"scope '{c['project']}'")
    return (f"Warning: ancestor {frag} already indexes part of '{project}'. Indexing "
            f"proceeded because respect_gitignore=False reaches files the parent skipped, "
            f"but the non-ignored files in this subtree now exist in both {kind} scopes.")


def _format_nested_scope_warning(project: str, kind: str, descendants: list[dict[str, str]]) -> str:
    frag = _format_conflict_list(descendants, lambda c: f"child scope '{c['project']}'")
    return (f"Warning: existing {frag} under {kind} scope '{project}'. Parent indexing "
            f"proceeded because it can add files outside the child scope.")


async def _nested_scope_guard(project: str, host: str, kind: str,
                              respect_gitignore: bool = True) -> tuple[str | None, str]:
    """Return (block_message, warning) for a parent/child scope overlap.

    Indexing a child already covered by a parent scope is refused (no coverage benefit,
    just a duplicate namespace); indexing a parent over an existing child proceeds with a
    warning (the parent adds files outside the child). Fails OPEN — a transient scope-overlap
    check outage warns but never blocks indexing, since the guard is an optimization, not a
    data-safety gate.

    ``respect_gitignore`` is sent because it decides whether an overlap is real: a child
    indexing WITHOUT the filter under a parent that indexed WITH it reaches files the
    parent never had, so that pairing warns instead of blocking. Refusing it would leave
    only one way to index an ignored subtree — deleting the parent scope — which trades a
    whole repository index for one directory.

    The overlap is computed server-side (GET /api/index/scope-overlap) and returns just
    the KB-scale conflict lists — the guard no longer downloads the whole per-file
    inventory (which grew to multi-MB and began truncating over the LAN overlay)."""
    try:
        overlap = await _get(
            "/api/index/scope-overlap",
            {"project": project, "host": host, "kind": kind,
             "respect_gitignore": respect_gitignore})
    except Exception as exc:  # noqa: BLE001
        logger.warning("nested %s scope check failed for project=%s: %s", kind, project, exc)
        return None, (f"Warning: nested {kind} scope check unavailable ({exc}); "
                      f"indexed without the parent/child overlap check.")
    ancestor_blocks = overlap.get("ancestor_blocks") or []
    ancestor_widenings = overlap.get("ancestor_widenings") or []
    child_warnings = overlap.get("child_warnings") or []
    if ancestor_blocks:
        return _format_nested_scope_block(project, kind, ancestor_blocks), ""
    warnings = []
    if ancestor_widenings:
        warnings.append(_format_gitignore_widening_warning(project, kind, ancestor_widenings))
    if child_warnings:
        warnings.append(_format_nested_scope_warning(project, kind, child_warnings))
    return None, " ".join(warnings)


def _is_markdown_basename_pattern(pattern: str) -> bool:
    """Markdown basename patterns are matched case-insensitively across client OSes."""
    normalized = pattern.replace("\\", "/")
    return "/" not in normalized and normalized.lower().endswith(".md")


def _dedupe_paths(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(paths))


def _discover_markdown_files(root: str, pattern: str, recursive: bool,
                             max_depth: int | None) -> list[str]:
    """Find Markdown basename matches case-insensitively, skipping hidden paths."""
    pattern_lower = pattern.lower()

    def matches(name: str) -> bool:
        return not name.startswith(".") and fnmatch.fnmatchcase(name.lower(), pattern_lower)

    if not recursive:
        try:
            with os.scandir(root) as entries:
                return [
                    entry.path for entry in entries
                    if matches(entry.name) and entry.is_file()
                ]
        except OSError:
            return []

    depth_limit = None if max_depth is None else max(max_depth, 0)
    files: list[str] = []
    for current, dirs, names in os.walk(root):
        rel = os.path.relpath(current, root)
        depth = 0 if rel == "." else len(rel.split(os.sep))
        if depth_limit is not None and depth > depth_limit:
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        if depth_limit is not None and depth >= depth_limit:
            dirs[:] = []
        files.extend(
            os.path.join(current, name)
            for name in names
            if matches(name)
        )
    return files


def _expand_brace_patterns(pattern: str) -> list[str]:
    """Expand ONE top-level ``{a,b,c}`` alternation into concrete globs.

    "*.{md,txt}" -> ["*.md", "*.txt"]. A pattern without a single well-formed brace
    group (none, nested, or multiple groups) is returned unchanged as ``[pattern]``.
    glob has no native brace expansion, so we do it here and index/reconcile the union
    — indexing "*.md" then "*.txt" already coexists (each reconcile stays within its own
    pattern envelope), and this just lets one call cover both.
    """
    open_i = pattern.find("{")
    close_i = pattern.find("}", open_i + 1)
    if open_i == -1 or close_i == -1:
        return [pattern]
    inner = pattern[open_i + 1:close_i]
    rest = pattern[close_i + 1:]
    # Only a single, non-nested group is supported; anything else stays literal.
    if any(brace in inner or brace in rest for brace in ("{", "}")):
        return [pattern]
    prefix, suffix = pattern[:open_i], rest
    alts = [a.strip() for a in inner.split(",") if a.strip()]
    if not alts:
        return [pattern]
    return _dedupe_paths([f"{prefix}{a}{suffix}" for a in alts])


@dataclass(frozen=True)
class _JournalFileCandidate:
    """Standalone-plugin mirror of core.indexing.file_snapshot.FileCandidate."""
    abs_path: str
    rel_path: str
    content_hash: str | None
    size_bytes: int | None
    read_error: str | None = None

    @property
    def readable(self) -> bool:
        return self.content_hash is not None and self.read_error is None


def _journal_file_candidate(path: str, root: str) -> _JournalFileCandidate:
    """Hash bytes before uploading a journal body; keep read failures explicit."""
    rel_path = os.path.relpath(path, root).replace("\\", "/")
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return _JournalFileCandidate(
            abs_path=path, rel_path=rel_path, content_hash=digest.hexdigest(),
            size_bytes=os.path.getsize(path),
        )
    except OSError as exc:
        return _JournalFileCandidate(
            abs_path=path, rel_path=rel_path, content_hash=None, size_bytes=None,
            read_error=str(exc),
        )


def _flatten_inventory_doc_files(resp: dict) -> dict:
    """Detail inventory diff baseline for Journal documents only."""
    out: dict[str, dict] = {}
    for user in resp.get("users") or []:
        for host in user.get("hosts") or []:
            for project in host.get("projects") or []:
                for directory in project.get("dirs") or []:
                    for file in directory.get("files") or []:
                        if file.get("kind") == "doc" and file.get("rel_path"):
                            out[file["rel_path"].replace("\\", "/")] = file
    return out


def _journal_snapshot_diff(candidates: list[_JournalFileCandidate], known: dict, *,
                           case_sensitive: bool = True) -> tuple[set[str], bool, str | None]:
    """Return (unchanged paths, replace_safe, block reason), matching Code's guard."""
    identity = (lambda path: path) if case_sensitive else (lambda path: path.casefold())
    groups: dict[str, list[_JournalFileCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(identity(candidate.rel_path), []).append(candidate)
    if any(len(group) > 1 for group in groups.values()):
        return set(), False, "case-insensitive path identity collision"
    readable = {candidate.rel_path for candidate in candidates if candidate.readable}
    unreadable_keep = {
        candidate.rel_path for candidate in candidates
        if not candidate.readable and candidate.rel_path in known
    }
    unchanged = {
        candidate.rel_path for candidate in candidates
        if candidate.readable and candidate.rel_path in known
        and candidate.content_hash == (known[candidate.rel_path].get("content_hash") or None)
    }
    if unreadable_keep:
        return unchanged, False, "unreadable files"
    if known and not readable:
        return unchanged, False, "empty snapshot"
    if known and readable.isdisjoint(known):
        return unchanged, False, "disjoint snapshot"
    return unchanged, True, None


def _journal_known_in_envelope(known: dict, patterns: list[str], max_depth: int | None) -> dict:
    """Keep the catalog diff scoped to this pattern/depth scan only."""
    def matches(rel_path: str) -> bool:
        normalized = rel_path.replace("\\", "/").strip("/")
        # The walker skips dot-directories, so rows under one (.conversations/) were never
        # scanned and must not feed the snapshot diff / replace guard.
        if any(s.startswith(".") and s not in {".", ".."} for s in normalized.split("/")[:-1]):
            return False
        if max_depth is not None and normalized.count("/") > max_depth:
            return False
        basename = normalized.rsplit("/", 1)[-1]
        return any(
            fnmatch.fnmatchcase(
                basename.lower() if _is_markdown_basename_pattern(pattern) else basename,
                pattern.lower() if _is_markdown_basename_pattern(pattern) else pattern,
            )
            for pattern in patterns
        )
    return {rel_path: record for rel_path, record in known.items() if matches(rel_path)}


def _discover_files(root: str, pattern: str, recursive: bool, max_depth: int | None) -> list[str]:
    """Discover files under `root`.

    Markdown basename patterns (for example "*.md"/"*.MD") are matched
    case-insensitively across client OSes. Other patterns keep normal glob semantics:
    non-recursive = root only; recursive with max_depth=None = every level; recursive
    with max_depth=N = levels 0..N.
    """
    if _is_markdown_basename_pattern(pattern):
        return _discover_markdown_files(root, pattern, recursive, max_depth)

    base = glob.escape(root)
    matches: list[str] = []
    if not recursive:
        matches.extend(glob.glob(os.path.join(base, pattern)))
    elif max_depth is None:
        matches.extend(glob.glob(os.path.join(base, "**", pattern), recursive=True))
    else:
        for depth in range(max(max_depth, 0) + 1):
            matches.extend(glob.glob(os.path.join(base, *(["*"] * depth), pattern)))
    return _dedupe_paths(matches)


_GIT_LIST_TIMEOUT_S = 30
_GIT_CHECK_IGNORE_TIMEOUT_S = 10
_CHECK_IGNORE_BATCH = 2000


class GitIgnoreCheckFailed(RuntimeError):
    """Git was asked which paths are ignored and did not answer (timeout, corrupt index,
    any exit other than "not a repository"). Every ignore check fails closed: a caller
    that cannot learn what is ignored must not treat everything as allowed.
    NOTE: mirrors core.code_kb.chunker.GitIgnoreCheckFailed."""


def _gitignore_allowed(root: str, pathspecs: Sequence[str] = ()) -> set[str] | None:
    """Repo-relative paths (forward slashes) git does NOT ignore under `root`, or None
    if `root` isn't a git working tree / git isn't installed (callers then apply no
    gitignore filtering). ``pathspecs`` (relative to `root`, matched literally) narrow the
    listing to those paths; the output stays relative to `root`.

    Delegates the resolution to git itself, so nested .gitignore / negation (``!``) rules
    / ``.git/info/exclude`` / global ``core.excludesFile`` all hold — a hand-rolled parser
    gets those subtly wrong. --cached = tracked, --others = untracked, --exclude-standard
    drops ignored, -z NUL-separates so unicode/space paths match our os.walk rels.
    Decoded as UTF-8 (git's path encoding), not the ambient locale, so non-ASCII paths
    resolve the same on any host.

    Fails closed: any outcome other than an answer, "git not installed" or "not a git
    repository" raises GitIgnoreCheckFailed instead of returning None.
    NOTE: mirrors core.code_kb.chunker.gitignore_allowed."""
    command = ["git", "--literal-pathspecs", "-C", root, "ls-files",
               "--cached", "--others", "--exclude-standard", "-z"]
    if pathspecs:
        command += ["--", *pathspecs]
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # git's "not a git repository" text is matched below
    try:
        # stdin=DEVNULL: as an MCP stdio server this process's stdin is the JSON-RPC pipe;
        # a child inheriting it can wedge on that handle (W159, same as _git_commit).
        out = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=_GIT_LIST_TIMEOUT_S, encoding="utf-8", errors="surrogateescape",
            env=env)
    except FileNotFoundError:
        return None  # git is not installed → caller does no filtering
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitIgnoreCheckFailed(f"git ls-files did not finish in {root}: {exc!r}") from exc
    if out.returncode == 0:
        return {p for p in out.stdout.split("\x00") if p}
    stderr = (out.stderr or "").strip()
    if out.returncode == 128 and "not a git repository" in stderr:
        return None  # not a git repo → caller does no filtering
    raise GitIgnoreCheckFailed(
        f"git ls-files failed in {root} (exit {out.returncode}): {stderr[:200]}")


def _drop_gitignored(paths: list[str], root: str,
                     respect_gitignore: bool) -> tuple[list[str], int]:
    """Filter abs `paths` to those git does NOT ignore under `root`; returns
    (kept, dropped_count). A no-op when respect_gitignore is False or `root` isn't a repo."""
    if not respect_gitignore:
        return paths, 0
    allowed = _gitignore_allowed(root)
    if allowed is None:
        return paths, 0
    kept = [p for p in paths if os.path.relpath(p, root).replace(os.sep, "/") in allowed]
    return kept, len(paths) - len(kept)


SEARCH_PRIORITY_INSTRUCTIONS = (
    "Retrieval-first rule: in any task that needs to locate code, symbols, text in "
    "files, papers or project history, your FIRST search must be a Retrieval MCP call: "
    "search_code (mode=\"keyword\" for exact identifiers and strings, the grep "
    "replacement; mode=\"hybrid\" when names are unknown), journal_search for project "
    "history and past decisions, search_papers / search_arxiv / search_within_paper for "
    "papers. Do not start with grep, rg, find, git grep, Grep or Glob. After at least "
    "one Retrieval call in the task, use them freely to verify, list every occurrence, "
    "or read new or unindexed files. Skip the Retrieval call only when you already know "
    "the exact file path and just need to read it, the content was written in this "
    "session, or Retrieval failed or reported the scope is not indexed. Batch "
    "independent queries as one list."
)

mcp = FastMCP("retrieval", instructions=SEARCH_PRIORITY_INSTRUCTIONS)


def _multi_query_tool_description(description: str) -> str:
    """The single-query docstring, as the list-accepting replacement advertises it."""
    # Python 3.12 FastMCP preserves the decorated function docstring's common
    # indentation, while Python 3.13 normalizes it before registration.  Clean it
    # here so both runtimes advertise byte-identical descriptions and daemon
    # surface digests. The list contract itself is on the `query` parameter.
    return inspect.cleandoc(description).rstrip()


# A server-decided, deliver-once message about a scope (`scope_notice` on the search and
# job responses). A tool puts it verbatim at the very top of its output and keeps no
# state: the backend already marked it delivered. A list call collects the notices its
# queries receive here instead, so they appear once above the per-query blocks.
_SCOPE_NOTICE_SINK: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "scope_notice_sink", default=None)


def _with_scope_notice(text: str, notice: object) -> str:
    if not isinstance(notice, str) or not notice.strip():
        return text
    sink = _SCOPE_NOTICE_SINK.get()
    if sink is not None:
        sink.append(notice)
        return text
    return f"{notice}\n\n{text}"


async def _run_query_batch(
    query: str | list[str], *, tool_name: str, invoke: Callable[[str], Awaitable[str]],
) -> str:
    """Run list queries independently while preserving legacy string behavior exactly."""
    if isinstance(query, str):
        return await invoke(query)
    if not query:
        raise ValueError(f"{tool_name}.query must contain at least one string")
    for index, item in enumerate(query):
        if not isinstance(item, str):
            raise ValueError(f"{tool_name}.query[{index}] must be a string")
        if not item.strip():
            raise ValueError(f"{tool_name}.query[{index}] must not be blank")

    semaphore = asyncio.Semaphore(_QUERY_BATCH_CONCURRENCY)

    async def run_one(item: str) -> str:
        try:
            async with semaphore:
                return await invoke(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    notices: list[str] = []
    token = _SCOPE_NOTICE_SINK.set(notices)  # each task copies this context at creation
    try:
        tasks = [asyncio.create_task(run_one(item)) for item in query]
    finally:
        _SCOPE_NOTICE_SINK.reset(token)
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    body = "\n\n".join(
        f"Query {index + 1}/{len(query)}: {item!r}\n{result}"
        for index, (item, result) in enumerate(zip(query, results, strict=True))
    )
    if notices:
        body = "\n\n".join([*dict.fromkeys(notices), body])
    return body


_CLIENT: httpx.AsyncClient | None = None


def _http_timeout() -> httpx.Timeout:
    """Layered httpx timeout: fail-fast connect/pool, generous read/write."""
    return httpx.Timeout(
        connect=_TIMEOUT_CONNECT_S,
        read=_TIMEOUT_READ_S,
        write=_TIMEOUT_WRITE_S,
        pool=_TIMEOUT_POOL_S,
    )


try:  # optional: without the wheel the client simply never compresses
    import zstandard as _zstd
except ImportError:  # pragma: no cover - only on an incomplete install
    _zstd = None

# Only `index_code`/`index_dir` bodies ever get large — this repository's is 19.3 MiB
# of JSON, which zstd -3 takes to 2.2 MiB in 54 ms. Below this a round trip is
# latency-bound and compressing costs more than the bytes it saves.
_ZSTD_MIN_BYTES = 64 * 1024
_ZSTD_LEVEL = 3

# None = not yet known. Set True when a backend advertises `Accept-Encoding` on a
# response (RFC 7694: as a response header it means "I accept these in requests"),
# and False the first time one rejects a compressed body.
_SERVER_ACCEPTS_ZSTD: bool | None = None


async def _note_server_capabilities(response: httpx.Response) -> None:
    global _SERVER_ACCEPTS_ZSTD
    advertised = response.headers.get("accept-encoding")
    if advertised and "zstd" in advertised.lower():
        _SERVER_ACCEPTS_ZSTD = True


def _client() -> httpx.AsyncClient:
    """Lazily-built, reused httpx client — one TCP/TLS pool for the whole session
    instead of a fresh connection per tool call."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(
            base_url=BASE_URL, timeout=_http_timeout(), headers=_HEADERS,
            event_hooks={"response": [_note_server_capabilities]})
    return _CLIENT


def _compress_body(payload: dict) -> bytes | None:
    """The zstd-framed body, or None when plain JSON is the right thing to send."""
    if _zstd is None or _SERVER_ACCEPTS_ZSTD is False:
        return None
    raw = json.dumps(payload).encode()
    if len(raw) < _ZSTD_MIN_BYTES:
        return None
    return _zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(raw)


async def _send_json(method: str, path: str, payload: dict) -> httpx.Response:
    """POST/PUT a JSON body, compressed when it is worth it AND the backend takes it.

    A backend that predates request decompression must keep working, so the first
    compressed body doubles as the probe. An old backend hands zstd bytes straight
    to the JSON parser and answers 400 ("There was an error parsing the body") or
    422; a new one that somehow cannot decode answers 415. All three are decided
    before the route function runs, so resending cannot duplicate an index
    submission or any other side effect.

    The retry only applies while support is still unknown. Once a backend has
    advertised `Accept-Encoding` on a response, a 400 is a real error from a real
    handler and gets raised as one instead of being retried.
    """
    global _SERVER_ACCEPTS_ZSTD
    client = _client()
    body = _compress_body(payload)
    if body is None:
        return await client.request(method, path, json=payload)

    r = await client.request(method, path, content=body,
                             headers={"Content-Type": "application/json",
                                      "Content-Encoding": "zstd"})
    # Read AFTER the response hook: a backend that advertised zstd on this very
    # reply has proved it can decode, so its 400 belongs to the handler, not us.
    if _SERVER_ACCEPTS_ZSTD is not True and r.status_code in (400, 415, 422):
        _SERVER_ACCEPTS_ZSTD = False
        return await client.request(method, path, json=payload)
    return r


def _response_detail(r: httpx.Response) -> str:
    """The reason a FastAPI backend gave for an error status, or '' if none."""
    try:
        detail = r.json().get("detail")
    except (ValueError, AttributeError):
        return r.text.strip()[:500]
    if isinstance(detail, list):  # request validation: one line per bad field
        return "; ".join(
            ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            + f": {item.get('msg', '')}"
            for item in detail if isinstance(item, dict)
        )[:500]
    return str(detail)[:500] if detail else ""


def _raise_for_status(r: httpx.Response) -> None:
    """raise_for_status with the backend's reason in the message.

    httpx's own message is the status line plus an MDN link; the reason ("Unknown
    public-code source IDs: x", "top_k: ... less than or equal to 50") sits in the
    body and never reached the model. Same exception type, request and response,
    so handlers that branch on status_code are unaffected."""
    if r.is_success:
        return
    detail = _response_detail(r)
    if not detail:
        r.raise_for_status()
    raise httpx.HTTPStatusError(
        f"HTTP {r.status_code} from {r.request.method} {r.request.url.path}: {detail}",
        request=r.request, response=r)


async def _post(path: str, payload: dict) -> dict:
    r = await _send_json("POST", path, payload)
    _raise_for_status(r)
    return r.json()


async def _put(path: str, payload: dict) -> dict:
    r = await _send_json("PUT", path, payload)
    _raise_for_status(r)
    return r.json()


async def _get(
    path: str,
    params: dict | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> dict:
    r = await _client().get(path, params=params or {}, headers=headers)
    _raise_for_status(r)
    return r.json()


async def _delete(path: str, params: dict | None = None) -> dict:
    r = await _client().request("DELETE", path, params=params or {})
    _raise_for_status(r)
    return r.json()


def _paper_artifact_filename(paper_id: str, suffix: str) -> str:
    """Turn a canonical paper ID into a portable local artifact filename."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("._") or "paper"
    return f"{stem}{suffix}"


def _paper_artifact_destination(
    paper_id: str,
    suffix: str,
    save_path: str | None,
    *,
    overwrite: bool,
) -> Path:
    """Resolve a caller-local file/directory target and enforce overwrite policy."""
    raw_path = save_path or "."
    expanded = Path(raw_path).expanduser()
    path_text = str(raw_path)
    directory_target = (
        save_path is None
        or expanded.is_dir()
        or path_text.endswith(("/", "\\"))
        or (not expanded.exists() and not expanded.suffix)
    )
    destination = expanded / _paper_artifact_filename(paper_id, suffix) \
        if directory_target else expanded
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination}. Pass overwrite=True to replace it."
        )
    return destination


def _commit_temporary_file(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically publish a complete sibling file with optional replacement."""
    if overwrite:
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Destination already exists: {destination}. Pass overwrite=True to replace it."
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Filesystem cannot guarantee overwrite=False for destination {destination} "
            "because atomic hard-link publication is unavailable. Choose a local "
            "NTFS/ext4-style path or pass overwrite=True explicitly."
        ) from exc


def _retryable_paper_artifact_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and 500 <= exc.response.status_code < 600
    )


def _paper_artifact_validator(headers) -> tuple[str, str] | None:
    """Prefer a strong ETag, falling back to Last-Modified for If-Range."""
    etag = headers.get("etag")
    if etag and not etag.startswith("W/"):
        return "etag", etag
    last_modified = headers.get("last-modified")
    if last_modified:
        return "last-modified", last_modified
    return None


def _parse_content_range(value: str | None) -> tuple[int, int, int]:
    match = _CONTENT_RANGE_PATTERN.fullmatch((value or "").strip())
    if match is None:
        raise ValueError(f"Invalid PDF Content-Range header: {value!r}")
    start, end, total = map(int, match.groups())
    if end < start or total <= end:
        raise ValueError(f"Invalid PDF Content-Range bounds: {value!r}")
    return start, end, total


def _truncate_temporary_file(path: Path, size: int) -> None:
    with path.open("r+b") as output:
        output.truncate(size)


async def _append_stream(response: httpx.Response, destination: Path) -> int:
    written = 0
    with destination.open("ab") as output:
        async for chunk in response.aiter_bytes(_PAPER_STREAM_CHUNK_BYTES):
            if not chunk:
                continue
            output.write(chunk)
            written += len(chunk)
    return written


async def _stream_ranged_artifact(
    open_stream,
    temporary: Path,
    *,
    label: str,
    range_bytes: int,
    signature: bytes | None,
    signature_error: str,
) -> int:
    """Download one artifact with validated byte ranges and per-range retries."""
    temporary.write_bytes(b"")
    offset = 0
    total_size: int | None = None
    validator: tuple[str, str] | None = None
    while total_size is None or offset < total_size:
        requested_end = offset + range_bytes - 1
        if total_size is not None:
            requested_end = min(requested_end, total_size - 1)
        checkpoint = offset
        received_full_response = False

        for attempt in range(1, _PAPER_ARTIFACT_ATTEMPTS + 1):
            ranged_response = False
            declared_end: int | None = None
            try:
                async with open_stream(
                    f"bytes={offset}-{requested_end}",
                    validator[1] if validator else None,
                ) as response:
                    response.raise_for_status()
                    response_validator = _paper_artifact_validator(response.headers)
                    if validator is None:
                        validator = response_validator
                    elif response_validator != validator:
                        raise ValueError(
                            f"{label} changed during ranged download: expected "
                            f"{validator[0]} {validator[1]!r}, received "
                            f"{response_validator!r}."
                        )
                    if response.status_code == 200:
                        if offset != 0:
                            raise ValueError(
                                f"Server stopped honoring ranges for {label} after a "
                                "partial transfer."
                            )
                        content_length = response.headers.get("content-length")
                        expected = int(content_length) if content_length else None
                        written = await _append_stream(response, temporary)
                        if expected is not None and written != expected:
                            raise httpx.RemoteProtocolError(
                                f"Incomplete {label} response: expected {expected} bytes, "
                                f"received {written}."
                            )
                        offset = written
                        total_size = written
                        received_full_response = True
                    elif response.status_code == 206:
                        ranged_response = True
                        start, end, response_total = _parse_content_range(
                            response.headers.get("content-range"))
                        declared_end = end
                        if start != offset or end > requested_end:
                            raise ValueError(
                                f"{label} Content-Range does not match the requested byte range: "
                                f"requested bytes={offset}-{requested_end}, received "
                                f"bytes={start}-{end}/{response_total}."
                            )
                        if total_size is not None and response_total != total_size:
                            raise ValueError(
                                f"{label} size changed during ranged download: "
                                f"expected {total_size}, received {response_total}."
                            )
                        total_size = response_total
                        expected = end - start + 1
                        written = await _append_stream(response, temporary)
                        if written != expected:
                            raise httpx.RemoteProtocolError(
                                f"Incomplete {label} range: expected {expected} bytes, "
                                f"received {written}."
                            )
                        offset = end + 1
                    else:
                        raise ValueError(
                            f"Unexpected {label} response status {response.status_code}; "
                            "expected 200 or 206."
                        )
                break
            except httpx.HTTPError as exc:
                partial_size = temporary.stat().st_size
                valid_partial = (
                    ranged_response
                    and declared_end is not None
                    and checkpoint < partial_size <= declared_end + 1
                    and (total_size is None or partial_size <= total_size)
                )
                if valid_partial:
                    offset = partial_size
                    logger.warning(
                        "Artifact transfer interrupted for %s range=%d-%d after %d "
                        "new byte(s): %s; resuming at byte %d",
                        label, checkpoint, requested_end,
                        partial_size - checkpoint, exc, offset,
                    )
                    break
                _truncate_temporary_file(temporary, checkpoint)
                offset = checkpoint
                if not _retryable_paper_artifact_error(exc) \
                        or attempt == _PAPER_ARTIFACT_ATTEMPTS:
                    raise
                delay = _PAPER_ARTIFACT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Artifact transfer interrupted for %s range=%d-%d "
                    "(attempt %d/%d): %s; retrying in %.1fs",
                    label, checkpoint, requested_end, attempt,
                    _PAPER_ARTIFACT_ATTEMPTS, exc, delay,
                )
                await asyncio.sleep(delay)
            except BaseException:
                _truncate_temporary_file(temporary, checkpoint)
                raise
        else:
            raise RuntimeError(f"{label} range transfer exhausted without a result")

        if received_full_response:
            break

    if total_size is None or offset != total_size:
        raise ValueError(
            f"Incomplete {label} transfer: {offset}/{total_size} bytes."
        )
    with temporary.open("rb") as downloaded:
        if signature is not None and downloaded.read(len(signature)) != signature:
            raise ValueError(signature_error)
    return offset


async def _stream_pdf_ranges(open_stream, temporary: Path, paper_id: str) -> int:
    """Download a PDF with validated byte ranges, retrying only the failed range."""
    return await _stream_ranged_artifact(
        open_stream,
        temporary,
        label=f"PDF for paper '{paper_id}'",
        range_bytes=_PAPER_ARTIFACT_RANGE_BYTES,
        signature=b"%PDF-",
        signature_error=(
            f"Response for paper '{paper_id}' is not a PDF (missing %PDF- header)."
        ),
    )


async def _stream_markdown_ranges(open_stream, temporary: Path, paper_id: str) -> int:
    """Stream an extracted Markdown file without buffering it into JSON."""
    return await _stream_ranged_artifact(
        open_stream,
        temporary,
        label=f"Markdown for paper '{paper_id}'",
        range_bytes=_PAPER_ARTIFACT_RANGE_BYTES,
        signature=None,
        signature_error="",
    )


def _validate_paper_batch(
    paper_ids: list[str],
    artifacts: str,
) -> tuple[list[str], str]:
    if not 1 <= len(paper_ids) <= 10:
        raise ValueError("paper_ids must contain between 1 and at most 10 papers")
    normalized_ids = [str(paper_id).strip() for paper_id in paper_ids]
    if any(not paper_id for paper_id in normalized_ids):
        raise ValueError("paper_ids cannot contain blank values")
    normalized_artifacts = artifacts.strip().lower()
    if normalized_artifacts not in {"pdf", "markdown", "both"}:
        raise ValueError("artifacts must be one of: pdf, markdown, both")
    return list(dict.fromkeys(normalized_ids)), normalized_artifacts


def _paper_batch_destination(save_dir: str) -> Path:
    destination = Path(save_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Batch destination is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _unique_paper_artifact_filename(
    paper_id: str,
    suffix: str,
    used_names: set[str],
) -> str:
    candidate = _paper_artifact_filename(paper_id, suffix)
    if candidate in used_names:
        stem = candidate[:-len(suffix)] if suffix else candidate
        digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:12]
        candidate = f"{stem}-{digest}{suffix}"
    if candidate in used_names:
        raise ValueError(
            f"Paper IDs cannot be mapped to unique local filenames: {paper_id!r}"
        )
    used_names.add(candidate)
    return candidate


def _paper_batch_artifact_plan(
    paper_ids: list[str],
    artifacts: str,
    destination_dir: Path,
    *,
    overwrite: bool,
) -> list[tuple[str, str, Path]]:
    artifact_kinds = ("pdf", "markdown") if artifacts == "both" else (artifacts,)
    used_names: set[str] = set()
    plan: list[tuple[str, str, Path]] = []
    for paper_id in paper_ids:
        for artifact_kind in artifact_kinds:
            suffix = ".pdf" if artifact_kind == "pdf" else ".md"
            filename = _unique_paper_artifact_filename(
                paper_id, suffix, used_names)
            plan.append((paper_id, artifact_kind, destination_dir / filename))
    if not overwrite:
        existing = [destination for _, _, destination in plan if destination.exists()]
        if existing:
            raise FileExistsError(
                f"Destination already exists: {existing[0]}. "
                "Pass overwrite=True to replace existing artifacts."
            )
    return plan


def format_paper_batch_download(downloaded: dict[str, Any]) -> str:
    errors = downloaded.get("errors") or []
    summary = (
        f"Downloaded {len(downloaded.get('paper_ids') or [])} paper(s): "
        f"{len(downloaded.get('paths') or [])} artifact(s) saved to "
        f"'{downloaded.get('save_dir')}', {len(errors)} error(s)."
    )
    if not errors:
        return summary
    details = "\n".join(
        f"- {error.get('paper_id')} [{error.get('artifact')}]: {error.get('error')}"
        for error in errors
    )
    return f"{summary}\nFailures:\n{details}"


async def _download_paper_pdf(
    paper_id: str,
    save_path: str | None,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Download an original paper PDF with validated HTTP ranges."""
    destination = _paper_artifact_destination(
        paper_id, ".pdf", save_path, overwrite=overwrite)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    encoded_paper_id = quote(paper_id, safe="")

    def open_stream(range_header: str, if_range: str | None):
        headers = {"Accept-Encoding": "identity", "Range": range_header}
        if if_range:
            headers["If-Range"] = if_range
        return _client().stream(
            "GET",
            f"/api/papers/{encoded_paper_id}/pdf",
            headers=headers,
        )

    try:
        size = await _stream_pdf_ranges(open_stream, temporary, paper_id)
        _commit_temporary_file(temporary, destination, overwrite=overwrite)
        return {"path": str(destination), "size": size}
    finally:
        temporary.unlink(missing_ok=True)


async def _download_paper_markdown_file(
    paper_id: str,
    save_path: str,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Stream complete Markdown to an atomic file on the MCP host."""
    destination = _paper_artifact_destination(
        paper_id, ".md", save_path, overwrite=overwrite)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    # Uploads live under /api/documents; only papers have a crawler artifact at
    # /api/papers/{id}/markdown. Both routes are FileResponse, so the Range
    # resume logic below is identical either way.
    if _is_document_id(paper_id):
        markdown_path = f"/api/documents/{quote(_document_id(paper_id), safe='')}/markdown"
    else:
        markdown_path = f"/api/papers/{quote(paper_id, safe='')}/markdown"
    idempotency_key = f"paper-markdown-{uuid.uuid4().hex}"

    def open_stream(range_header: str, if_range: str | None):
        headers = {
            "Accept-Encoding": "identity",
            "Range": range_header,
            "X-Idempotency-Key": idempotency_key,
        }
        if if_range:
            headers["If-Range"] = if_range
        return _client().stream("GET", markdown_path, headers=headers)

    try:
        size = await _stream_markdown_ranges(open_stream, temporary, paper_id)
        _commit_temporary_file(temporary, destination, overwrite=overwrite)
        return {"path": str(destination), "size": size}
    finally:
        temporary.unlink(missing_ok=True)


async def _document_markdown_size(document_id: str) -> int | None:
    """Total Markdown bytes, via a 1-byte Range probe. None if it can't be read.

    The route does not answer HEAD (FastAPI registers GET only, so HEAD is 405),
    but a `bytes=0-0` GET returns `Content-Range: bytes 0-0/<total>` — the size
    without moving 4.7 MB across the wire just to decide whether to refuse.
    """
    try:
        response = await _client().get(
            f"/api/documents/{quote(document_id, safe='')}/markdown",
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
        )
    except httpx.HTTPError:
        return None
    match = _CONTENT_RANGE_PATTERN.match(response.headers.get("Content-Range", ""))
    return int(match.group(3)) if match else None


async def _get_paper_markdown_payload(paper_id: str) -> dict[str, Any]:
    if _is_document_id(paper_id):
        encoded = quote(_document_id(paper_id), safe="")
        path = f"/api/documents/{encoded}/fulltext"
    else:
        path = f"/api/papers/{quote(paper_id, safe='')}/fulltext"
    idempotency_key = f"paper-fulltext-{uuid.uuid4().hex}"
    for attempt in range(1, _PAPER_ARTIFACT_ATTEMPTS + 1):
        try:
            return await _get(
                path, headers={"X-Idempotency-Key": idempotency_key})
        except httpx.HTTPError as exc:
            if not _retryable_paper_artifact_error(exc) \
                    or attempt == _PAPER_ARTIFACT_ATTEMPTS:
                raise
            delay = _PAPER_ARTIFACT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Markdown transfer interrupted for paper=%s (attempt %d/%d): "
                "%s; retrying in %.1fs",
                paper_id, attempt, _PAPER_ARTIFACT_ATTEMPTS, exc, delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Markdown transfer exhausted without a result")


async def _download_paper_batch(
    paper_ids: list[str],
    save_dir: str,
    *,
    artifacts: str,
    overwrite: bool,
) -> dict[str, Any]:
    normalized_ids, normalized_artifacts = _validate_paper_batch(
        paper_ids, artifacts)
    destination_dir = _paper_batch_destination(save_dir)
    plan = _paper_batch_artifact_plan(
        normalized_ids,
        normalized_artifacts,
        destination_dir,
        overwrite=overwrite,
    )
    semaphore = asyncio.Semaphore(_PAPER_BATCH_CONCURRENCY)

    async def download_one(
        paper_id: str,
        artifact_kind: str,
        destination: Path,
    ) -> tuple[str | None, dict[str, str] | None]:
        async with semaphore:
            try:
                if artifact_kind == "pdf":
                    saved = await _download_paper_pdf(
                        paper_id,
                        str(destination),
                        overwrite=overwrite,
                    )
                else:
                    saved = await _download_paper_markdown_file(
                        paper_id,
                        str(destination),
                        overwrite=overwrite,
                    )
                return str(saved["path"]), None
            except Exception as exc:  # noqa: BLE001 - report per-artifact failures
                return None, {
                    "paper_id": paper_id,
                    "artifact": artifact_kind,
                    "error": str(exc),
                }

    results = await asyncio.gather(*(
        download_one(paper_id, artifact_kind, destination)
        for paper_id, artifact_kind, destination in plan
    ))
    return {
        "paper_ids": normalized_ids,
        "artifacts": normalized_artifacts,
        "save_dir": str(destination_dir),
        "paths": [path for path, _ in results if path is not None],
        "errors": [error for _, error in results if error is not None],
    }


def _snippet(text: str, limit: int = 280) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# Uploads and webpages have NO paper_id — `papers` dispatches payload profiles on
# source_type and only paper/review get that key (core/storage/papers_payload.py).
# Their sole handle is the numeric documents.id, so search results hand it back
# `doc:`-prefixed and the follow-up tools route on that instead of guessing.
_DOC_ID_PREFIX = "doc:"
# A whole book's Markdown is millions of characters (the Murphy PML volume is
# 4,716,048 bytes). Returning that inline would bury the caller's context, so the
# document path refuses above this and points at save_path. Papers keep their
# uncapped behaviour: this guard applies only to document ids.
_INLINE_DOCUMENT_MARKDOWN_MAX_BYTES = 400_000


def _is_document_id(raw: str) -> bool:
    """True when this id addresses a documents.id rather than a paper_id.

    Accepts the `doc:` form this MCP emits and a bare run of digits, so an id
    copied straight off the documents page also works. The two namespaces cannot
    collide: measured against production 2026-09-21, 0 of 106,156 paper_ids are
    all-digits.
    """
    value = str(raw or "").strip()
    return value.startswith(_DOC_ID_PREFIX) or value.isdigit()


def _document_id(raw: str) -> str:
    """Strip the `doc:` prefix, leaving the bare numeric documents.id."""
    value = str(raw or "").strip()
    if value.startswith(_DOC_ID_PREFIX):
        return value[len(_DOC_ID_PREFIX):].strip()
    return value


def _scope_filter(paper_id: str) -> dict[str, list[str]]:
    """Server-side scope for one id. Exactly one key — they AND if both are sent."""
    if _is_document_id(paper_id):
        return {"doc_ids": [_document_id(paper_id)]}
    return {"paper_ids": [str(paper_id).strip()]}


_JOURNAL_SNIPPET_LIMIT = 260


def _is_journal_summary_chunk(chunk: dict) -> bool:
    index = chunk.get("chunk_index")
    return index == 0 or str(index).strip() == "0"


def _journal_display_chunks(hit: dict, *, max_chunks: int = 3) -> list[dict]:
    """Select concrete journal chunks for MCP output.

    Markdown docs have a synthetic chunk-0 summary. Prefer returned body chunks
    while preserving backend order so the caller sees concrete paragraphs instead
    of only ``source_kind/client`` metadata.
    """
    candidates: list[dict] = []
    seen: set[str] = set()

    def add(chunk: dict | None) -> None:
        if not isinstance(chunk, dict):
            return
        content = str(chunk.get("content") or "")
        if not content.strip():
            return
        key = str(chunk.get("chunk_id") or chunk.get("chunk_index") or content)
        if key in seen:
            return
        seen.add(key)
        candidates.append(chunk)

    for chunk in hit.get("chunks") or []:
        add(chunk)
    if not candidates:
        add(hit.get("best_chunk") or {})

    body_chunks = [chunk for chunk in candidates if not _is_journal_summary_chunk(chunk)]
    return (body_chunks or candidates)[:max_chunks]


def _journal_chunk_label(chunk: dict) -> str:
    parts: list[str] = []
    if chunk.get("chunk_index") is not None:
        parts.append(f"chunk {chunk.get('chunk_index')}")
    heading = str(chunk.get("heading") or "").strip()
    if heading and not _is_journal_summary_chunk(chunk):
        parts.append(heading)
    return " · ".join(parts)


# ------------------------------- paper retrieval ------------------------------- #

@mcp.tool()
async def search_papers(query: str, top_k: int = 10, venue: str | None = None,
                        year: int | None = None, title_only: bool = False,
                        source_type: str | None = None) -> str:
    """Semantic hybrid (dense + sparse) search over 95k+ top-venue CS papers.

    Phrase the query as a full, specific description, not a bare keyword — e.g.
    "retrieval-augmented generation with reranking for long documents" beats "RAG".
    Returns title + venue/year + paper_id + best snippet; feed a paper_id into
    search_within_paper / match_paper.

    Args:
        query: Natural-language description of what you're looking for.
        top_k: How many papers to return.
        venue: Optional venue filter, e.g. "NeurIPS", "CVPR", "ACL" (default: all venues).
        year: Optional publication-year filter, e.g. 2024 (default: all years).
        title_only: If true, match only against titles (default false = full text).
            Only meaningful for source_type="paper" — uploads and webpages have no
            title chunk, so combining it with them returns cover pages at score 0.
        source_type: Restrict to one kind of source: "paper" (the CS corpus),
            "upload" (uploaded documents — whole books live here), "webpage", or
            "review". Default: every source. Papers outnumber the rest by orders of
            magnitude, so an uploaded book is essentially never in the top_k of a
            general topical query — pass source_type="upload" to search the books.

    Example: search_papers(query="mixture-of-experts routing stability",
                           venue="ICLR", year=2024, top_k=5)
             search_papers(query="frontdoor adjustment", source_type="upload")
    """
    t0 = time.perf_counter()
    payload: dict[str, Any] = {"query": query, "top_k": top_k, "method": "hybrid",
                               "title_only": title_only, "dedup_by_paper": True}
    if venue:
        payload["venue"] = venue
    if year:
        payload["year"] = year
    if source_type:
        payload["source_type"] = source_type
    logger.info(
        "search_papers.start query=%r top_k=%s venue=%r year=%s title_only=%s source_type=%r",
        query, top_k, venue, year, title_only, source_type,
    )
    try:
        resp = await asyncio.wait_for(
            _post("/api/search", payload), timeout=_SEARCH_PAPERS_TIMEOUT_S)
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "search_papers.timeout top_k=%s timeout_s=%.2f elapsed_ms=%.0f",
            top_k, _SEARCH_PAPERS_TIMEOUT_S, elapsed_ms,
        )
        return (
            f"search_papers timed out after {_SEARCH_PAPERS_TIMEOUT_S:.2f} seconds "
            f"(top_k={top_k}). Backend hybrid search is usually <2s when the GPU is free; "
            "retry once with a smaller top_k, or check that the backend is not saturated "
            "by concurrent indexing/search."
        )
    out = _fmt_papers(resp)
    papers = resp.get("papers") or []
    logger.info(
        "search_papers.done papers=%d out_chars=%d total_ms=%.0f",
        len(papers), len(out), (time.perf_counter() - t0) * 1000,
    )
    return out


@mcp.tool()
async def search_arxiv(query: str, top_k: int = 10, year: int | None = None,
                       include_withdrawn: bool = False) -> str:
    """search_papers for arXiv computer-science preprints, which search_papers leaves out:
    use it for recent work that may not be published at a venue yet. Phrase the query as
    for search_papers. Returns paper_ids like "arxiv-2606.32000" that search_within_paper
    and get_paper_markdown accept.

    Withdrawn papers are left out; include_withdrawn=true brings them back, marked
    [WITHDRAWN]. search_within_paper on a withdrawn paper_id works either way.

    Example: search_arxiv(query="test-time compute scaling for reasoning language models",
                          year=2026, top_k=5)
    """
    t0 = time.perf_counter()
    # venue_tier="preprint" scopes the ordinary /api/search to arXiv (the backend
    # drops its default preprint exclusion for a tier-scoped request), so this
    # works through every deployment that already forwards /api/search.
    payload: dict[str, Any] = {"query": query, "top_k": top_k, "method": "hybrid",
                               "dedup_by_paper": True, "venue_tier": "preprint"}
    if year:
        payload["year"] = year
    if include_withdrawn:
        payload["include_withdrawn"] = True
    logger.info("search_arxiv.start query=%r top_k=%s year=%s include_withdrawn=%s",
                query, top_k, year, include_withdrawn)
    try:
        resp = await asyncio.wait_for(
            _post("/api/search", payload), timeout=_SEARCH_PAPERS_TIMEOUT_S)
    except asyncio.TimeoutError:
        return (
            f"search_arxiv timed out after {_SEARCH_PAPERS_TIMEOUT_S:.2f} seconds "
            f"(top_k={top_k}); retry once with a smaller top_k."
        )
    out = _fmt_papers(resp)
    logger.info("search_arxiv.done papers=%d total_ms=%.0f",
                len(resp.get("papers") or []), (time.perf_counter() - t0) * 1000)
    return out


@mcp.tool()
async def search_within_paper(paper_id: str, query: str, top_k: int = 20) -> str:
    """Search INSIDE one paper and get every matching passage with its score.

    Use it to locate specific claims/methods/numbers within a paper you already found
    via search_papers.

    Args:
        paper_id: The id exactly as search_papers printed it. Papers use
            their canonical paper_id; uploaded documents and webpages have no
            paper_id and come back as "doc:<number>" — pass that through verbatim.
        query: What to find inside that paper.
        top_k: How many passages to return.

    Example: search_within_paper(paper_id="neurips-2023-12345",
                                 query="ablation on batch size", top_k=10)
    Example: search_within_paper(paper_id="doc:114854",
                                 query="evidence lower bound derivation")
    """
    payload = {"query": query, "top_k": top_k,
               "dedup_by_paper": False, "method": "hybrid",
               **_scope_filter(paper_id)}
    return _fmt_papers(await _post("/api/search", payload))


@mcp.tool()
async def get_paper_markdown(
    paper_id: str,
    save_path: str | None = None,
    overwrite: bool = False,
) -> str:
    """Get one paper's COMPLETE extracted Markdown (whole text, not search_within_paper's
    chunks, never truncated), or save it on this MCP host and return a short receipt.

    Args:
        paper_id: As search_within_paper takes it, "doc:<number>" ids included.
        save_path: File or directory to write (a directory receives <paper_id>.md). Omit
            to return the Markdown inline; an uploaded book ("doc:" id) can run to
            millions of characters, so inline refuses above a size limit.
        overwrite: Replace an existing destination.

    Example: get_paper_markdown(paper_id="doc:114854", save_path="./books/")
    """
    if save_path is None:
        # Check the size BEFORE fetching. Uploaded books are whole volumes: the
        # Murphy PML volume is 4,716,048 bytes, which inline would bury the
        # caller's context under roughly a million tokens. Papers keep their
        # historical uncapped behaviour, so this costs them nothing.
        if _is_document_id(paper_id):
            size = await _document_markdown_size(_document_id(paper_id))
            if size is not None and size > _INLINE_DOCUMENT_MARKDOWN_MAX_BYTES:
                return (
                    f"Document '{paper_id}' has {size:,} bytes of Markdown, over the "
                    f"{_INLINE_DOCUMENT_MARKDOWN_MAX_BYTES:,}-byte inline limit. "
                    f"Re-run with save_path to write it to a file, or use "
                    f"search_within_paper(paper_id=\"{paper_id}\", query=...) to pull "
                    f"only the passages you need."
                )
        payload = await _get_paper_markdown_payload(paper_id)
        return str(payload.get("content") or "")
    saved = await _download_paper_markdown_file(
        paper_id, _session_save_path(save_path), overwrite=overwrite)
    return (
        f"Saved Markdown for paper '{paper_id}' to '{saved['path']}' "
        f"({saved['size']} bytes)."
    )


@mcp.tool()
async def download_paper_pdf(
    paper_id: str,
    save_path: str | None = None,
    overwrite: bool = False,
) -> str:
    """Download a paper's ORIGINAL PDF to the machine running this MCP server.

    The PDF is streamed to a sibling temporary file, validated by its %PDF- header,
    and atomically moved into place. Binary data is never dumped into model context.

    Args:
        paper_id: Canonical paper ID from search_papers.
        save_path: Local file or directory. Omit to save <paper_id>.pdf in the current
            working directory; a directory receives that default filename.
        overwrite: Replace an existing destination only when true.
    """
    saved = await _download_paper_pdf(
        paper_id, _session_save_path(save_path), overwrite=overwrite)
    return f"Downloaded PDF for paper '{paper_id}' to '{saved['path']}' ({saved['size']} bytes)."


@mcp.tool()
async def download_papers(
    paper_ids: PaperBatchIds,
    save_dir: str = ".",
    artifacts: str = "pdf",
    overwrite: bool = False,
) -> str:
    """Download up to 10 papers to this MCP host with bounded concurrency.

    PDF files reuse validated HTTP Range resume; Markdown reuses complete-text retry.
    All destination conflicts are checked before network work begins, then at most two
    artifacts transfer concurrently. Failures are reported per artifact.

    Args:
        paper_ids: One through ten canonical paper IDs from search_papers.
        save_dir: Local directory that receives the downloaded artifacts.
        artifacts: pdf, markdown, or both.
        overwrite: Replace existing artifact files only when true.

    Example: download_papers(
        paper_ids=["icml-2026-paper-a", "icml-2026-paper-b"],
        save_dir="./papers/", artifacts="both")
    """
    downloaded = await _download_paper_batch(
        paper_ids,
        _session_save_path(save_dir),
        artifacts=artifacts,
        overwrite=overwrite,
    )
    return format_paper_batch_download(downloaded)


@mcp.tool()
async def match_document(text: str, top_k: int = 10) -> str:
    """Find corpus papers whose CONTENT is most similar to a passage you paste in —
    related-work discovery from an abstract/paragraph. Results aggregate per paper by
    how much of it the passage covers.

    Args:
        text: The passage/abstract to match against the corpus.
        top_k: How many papers to return.
    """
    return _fmt_match(await _post("/api/match-doc",
                                  {"input_type": "text", "text": text, "top_k": top_k}))


@mcp.tool()
async def match_paper(paper_id: str, top_k: int = 10) -> str:
    """Queue an exact, durable content-neighbour match for an indexed paper.

    The full query chunks, candidate depth and rerank pairs are retained. The call
    returns quickly with a job id; poll match_paper_status for progress/final results.

    Args:
        paper_id: Canonical id of the seed paper, from a search result.
        top_k: How many neighbours to return.
    """
    receipt = await _post(
        "/api/match-doc/jobs",
        {"input_type": "paper_id", "paper_id": paper_id, "top_k": top_k},
    )
    job_id = receipt.get("task_id")
    return (
        f"Exact match job {job_id} [{receipt.get('status', 'pending')}] queued for "
        f"paper {paper_id!r}. Poll match_paper_status(task_id={job_id})."
    )


@mcp.tool()
async def match_paper_status(task_id: int, cancel: bool = False) -> str:
    """Poll or cancel one durable exact match-paper job.

    Completed jobs return the same full paper/evidence result previously returned by
    match_paper. Cancellation never publishes a partial result.

    Args:
        task_id: Numeric job id returned by match_paper.
        cancel: Request cancellation before reading status.
    """
    # These are the match-job routes, not the generic /api/tasks ones. Task ids are
    # sequential integers and /api/tasks has no owner concept, so polling or
    # cancelling through it lets any caller reach anyone else's job.
    if cancel:
        task = await _post(f"/api/match-doc/jobs/{task_id}/cancel", {})
    else:
        task = await _get(f"/api/match-doc/jobs/{task_id}")
    status = str(task.get("status") or "unknown")
    result = task.get("result")
    if status == "completed" and isinstance(result, dict):
        return _fmt_match(result)
    line = (
        f"Exact match job {task_id} [{status}] "
        f"progress={int(task.get('progress') or 0)}/{int(task.get('total') or 0)}"
    )
    error = task.get("error")
    if error:
        line += f" error={error}"
    return line


@mcp.tool()
async def list_conferences() -> str:
    """List every venue in the corpus with its CCF tier and the years available.

    No arguments. Use it to discover valid `venue` values for search_papers.
    """
    resp = await _get("/api/conferences")
    confs = resp.get("conferences") or []
    lines = [f"{len(confs)} venues:"]
    for c in confs:
        years = c.get("years") or []
        yr = f"{min(years)}-{max(years)}" if years else "?"
        lines.append(f"- {c.get('title', c.get('sub', '?'))} [{c.get('venue_tier', '?')}] {yr}")
    return "\n".join(lines)


@mcp.tool()
async def corpus_stats() -> str:
    """Report corpus size: number of papers, text chunks, vectors, and the embedding
    model in use.
    """
    r = await _get("/api/stats")
    return (f"Corpus: {r.get('total_papers', '?')} papers, {r.get('total_chunks', '?')} chunks, "
            f"{r.get('vector_index_size', '?')} vectors ({r.get('embedding_model', '?')}).")


# ------------------------------ journal memory -------------------------------- #

@mcp.tool()
async def journal_record(body: str, title: str | None = None,
                         source_kind: str = "memory", tags: list[str] | None = None,
                         scope_dir: str = ".") -> str:
    """Record this session's detailed work memory into the authenticated journal so it
    is searchable later via journal_search.

    The journal namespace is `scope_dir`'s absolute path (default: current working
    directory), so this project's memory stays isolated from other projects — no
    hand-typed name.

    Args:
        body: The memory content — be specific (what changed, root cause, how verified).
            Required.
        title: Optional short title for the entry (default: none).
        source_kind: "memory" (default; a work note that accumulates) or "doc" (a file's
            current content, latest-wins on re-record).
        tags: Optional list of tags for filtering, e.g. ["bugfix","retrieval"].
        scope_dir: Directory whose absolute path is the journal namespace. Pass the
            same dir used with journal_index_dir to search/record in that indexed scope.

    Example: journal_record(body="Fixed the RRF fusion off-by-one in pipeline.py; root
    cause was 0-indexed rank. Verified with test_pipeline_rrf.", title="RRF rank fix",
    tags=["bugfix","retrieval"])
    """
    payload: dict[str, Any] = {"project_slug": _scope_path(scope_dir), "client": "mcp",
                               "source_kind": source_kind, "body": body, "tags": tags or []}
    if title:
        payload["title"] = title
    resp = await _post("/api/journal/entries", payload)
    e = resp.get("entry") or {}
    return (f"Recorded {e.get('entry_id', '?')} "
            f"(chunks={resp.get('chunk_count', 0)}, indexed={resp.get('indexed', False)}).")


_CONVERSATION_SYNCER: Any = None


def _conversations_module() -> Any:
    """The sibling ``retrieval_mcp_conversations`` module, imported only when enabled."""
    try:
        import retrieval_mcp_conversations as module
    except ImportError:  # loaded by path (tests, direct execution outside site-packages)
        import importlib.util
        name = "retrieval_mcp_conversations"
        module = sys.modules.get(name)
        if module is None:
            path = Path(__file__).with_name(f"{name}.py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return module


def _conversation_state_dir() -> Path:
    override = _env("RETRIEVAL_MCP_STATE_DIR")
    if override:
        return Path(override).expanduser()
    try:
        from retrieval_mcp_daemon import resolve_runtime_dir
        return resolve_runtime_dir()
    except Exception:  # noqa: BLE001 — only the state location, never worth failing a tool
        return Path.home() / ".retrieval-mcp"


def _conversation_sync_line(scope_dir: str) -> str | None:
    """Start the background Claude Code/Codex conversation sync for ``scope_dir`` and
    return the one-line status to append to the tool output (None when disabled).

    Uses the tool's already-resolved scope, never the daemon's own cwd, and only syncs a
    scope that is a local directory (a remote scope has no local transcripts)."""
    if (os.environ.get("RETRIEVAL_JOURNAL_CONVERSATIONS") or "0").strip() in {"", "0"}:
        return None
    global _CONVERSATION_SYNCER
    try:
        root = _session_abspath(scope_dir)
        if not os.path.isdir(root):
            return None
        if _CONVERSATION_SYNCER is None:
            _CONVERSATION_SYNCER = _conversations_module().ConversationSyncer(
                post=_post, get=_get, state_dir=_conversation_state_dir)
        host = _env("RETRIEVAL_HOST", "RETRIVAL_HOST", default=socket.gethostname()) or ""
        return _CONVERSATION_SYNCER.kick(host, _scope_path(root))
    except Exception as exc:  # noqa: BLE001 — conversation sync must not fail a search
        logger.warning("conversation sync could not start: %s", exc)
        return f"Conversations sync failed: {exc}"


def _with_conversation_line(text: str, line: str | None) -> str:
    return f"{text}\n\n{line}" if line else text


@mcp.tool()
async def journal_search(query: str, top_k: int = 5,
                         mode: str = "keyword", dedup_by_entry: bool = True,
                         chunk_limit: JournalChunkCount = 3, scope_dir: str = ".",
                         path_glob: str | None = None) -> str:
    """Search journal memory and indexed documents (files added with journal_index_dir).

    What it does better than rg/grep on the same files: mode="keyword" matches exact
    tokens, paths, error codes and W-numbers ranked by relevance, and keeps working when
    the embedding service is down; mode="hybrid"/"dense" find notes by meaning, so a
    decision recorded in different words is still found. rg/grep remain right for files
    that were never indexed.

    Scoped to `scope_dir` (by absolute path; default current working directory) so other
    projects' notes don't leak in.

    Args:
        query: What to look for. Free text. Example: "qdrant upsert wait flag".
        top_k: Max entries to return. Example: top_k=10.
        mode: Retrieval engine.
            "keyword" (default) = FTS5 literal terms, needs NO embedding — always works
                even when the embed endpoint is down; best for exact tokens / paths / error
                codes / W-numbers.
            "hybrid" = dense + sparse fused (best semantic recall; needs embed endpoint up).
            "dense" = pure vector similarity (paraphrase / concept match).
            "sparse" = lexical vector only.
            Example: mode="hybrid".
        dedup_by_entry: One best hit per entry. Set false to see multiple
            matching chunks from the same note. Example: dedup_by_entry=false.
        chunk_limit: NUMBER OF CHUNKS returned per entry, not characters/tokens.
            Valid 1-20; default 3. Use chunk_limit=20 for the maximum.
        scope_dir: Directory whose absolute path is the journal namespace (default "."
            = current working directory). Pass the SAME dir used with journal_index_dir.
            Example: scope_dir="C:/Users/Arc/Prd/retrival".
        path_glob: Narrow the search to a sub-directory or a single file INSIDE an
            already-indexed parent scope, matched against each doc's scope-relative path
            (its title). Works on ALL modes (dense/hybrid/sparse/keyword) and never drops
            results — the match-set is resolved and pushed down as a filter, so top_k fills
            only from matching docs (no post-filter squeeze). Accepts:
              • a subtree prefix — path_glob="notes/2026/" (everything under that folder)
              • a single file rel path — path_glob="notes/a.md"
              • a glob — path_glob="*.md" or path_glob="notes/**/*.md"
            Omit (default) for a normal whole-scope search (zero added cost).
            Example: journal_search(query="bandwidth limiter", scope_dir="C:/repo",
                                    path_glob="docs/design/")

    Example: journal_search(query="qdrant upsert wait flag", mode="keyword", top_k=5)
    """
    chunk_limit = _validate_journal_chunk_count(chunk_limit)
    normalized_path_glob = _normalize_path_glob(path_glob) if path_glob else None
    conversation_line = _conversation_sync_line(scope_dir)
    body = {"project_slug": _scope_path(scope_dir),
            "query": query, "top_k": top_k, "mode": mode,
            "dedup_by_entry": dedup_by_entry, "include_chunks": True,
            "chunk_limit": chunk_limit}
    if normalized_path_glob:
        body["path_glob"] = normalized_path_glob
    resp = await _post("/api/journal/search", body)
    notice = resp.get("scope_notice")
    hits = resp.get("hits") or []
    if not hits:
        guidance = _format_scope_miss(
            result_kind="journal entries",
            query=query,
            scope=body["project_slug"],
            scope_argument="scope_dir",
            search_tool="journal_search",
            next_step=f"journal_index_dir(dir={body['project_slug']!r})",
            path_glob=normalized_path_glob,
        )
        # Preserve the backend's path_glob-specific diagnosis, then add the scope
        # diagnosis requested by the caller instead of returning a bare miss.
        return _with_scope_notice(_with_conversation_line(
            f"{resp['notice']}\n\n{guidance}" if resp.get("notice") else guidance,
            conversation_line), notice)
    lines = [f"Found {len(hits)} journal entries:", ""]
    for i, hit in enumerate(hits, 1):
        e = hit.get("entry") or {}
        title = e.get("title") or e.get("entry_id") or "(untitled)"
        lines.append(f"{i}. {title} [{e.get('source_kind', '?')}/{e.get('client', '?')}] "
                     f"score={hit.get('score', 0.0):.3f}")
        lines.append(f"   entry_id: {e.get('entry_id', '')}")
        for chunk in _journal_display_chunks(hit, max_chunks=chunk_limit):
            label = _journal_chunk_label(chunk)
            prefix = f"   ▸ [{label}] " if label else "   ▸ "
            lines.append(prefix + _snippet(chunk.get("content", ""), _JOURNAL_SNIPPET_LIMIT))
        lines.append("")
    return _with_scope_notice(
        _with_conversation_line("\n".join(lines).rstrip(), conversation_line), notice)


@mcp.tool()
async def journal_recent(limit: int = 10, scope_dir: str = ".") -> str:
    """List the most recent journal entries for the current project (newest first).

    Scoped to `scope_dir`'s absolute path (default: current working directory).

    Args:
        limit: How many entries to list.
        scope_dir: Directory whose absolute path is the journal namespace.
    """
    resp = await _get("/api/journal/entries",
                      {"project_slug": _scope_path(scope_dir), "limit": limit})
    entries = resp.get("entries") or []
    if not entries:
        return "No recent journal entries found."
    lines = [f"Recent journal entries ({len(entries)} of {resp.get('total', len(entries))}):", ""]
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. {e.get('title') or e.get('entry_id')} [{e.get('source_kind', '?')}]  "
                     f"{e.get('entry_id', '')}")
    return "\n".join(lines)


# Rendered-image / binary formats that must NEVER enter the journal as prose, even
# under a broad pattern like "*". An SVG is XML but it's a plot; a .png/.pkl/.parquet
# is binary — their bytes embed as noise, pollute semantic search, and swamp the browse
# list (W161). Text data files (.json/.csv/.txt) are intentionally NOT here: they can
# still be indexed by an explicit pattern (the default "*.md" excludes them anyway).
_SKIP_INDEX_EXTS = frozenset({
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tif", ".tiff",
    ".heic", ".heif", ".avif", ".psd",
    ".pdf", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac", ".webm", ".ogg",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".class", ".wasm",
    ".pkl", ".npy", ".npz", ".pt", ".pth", ".onnx", ".safetensors", ".parquet",
    ".db", ".sqlite", ".sqlite3",
})


@mcp.tool()
async def journal_index_dir(
    dir: str = ".",
    recursive: bool = False,
    pattern: str = "*.md",
    max_depth: int | None = None,
    respect_gitignore: bool = True,
) -> str:
    """Deprecated: index_code now indexes documents too (.md .mdx .markdown .rst .adoc
    .txt) and keeps them fresh; use index_code instead. This tool is removed in the
    next release.

    Batch-index the text files in a LOCAL directory into the journal (searchable with
    journal_search) and list them in index_inventory — the bulk counterpart to
    journal_record. The scope, respect_gitignore and nested-scope rules are index_code's.

    Files are read on this machine and sent as whole-document entries, latest-wins: an
    unchanged file is a no-op, an edited file replaces its previous version, so search
    only ever returns each file's current content.

    Args:
        dir: Directory to index; its absolute path is the journal scope.
        recursive: Recurse into subdirectories. Dot-dirs (.git/.venv) are skipped, but
            junk such as node_modules is not — point `dir` at a clean docs tree.
        pattern: File glob; one brace group is allowed, e.g. "*.{md,txt}". Each call
            reconciles deletions only within its own pattern, so "*.md" then "*.txt"
            coexist.
        max_depth: With recursive, how many levels below `dir` to descend (0 = root
            only). Omit for unbounded.
        respect_gitignore: As in index_code.

    Example: journal_index_dir(dir="~/Prd/retrival", pattern="*.md")
    """
    # Resolve once, then derive the namespace from that same absolute path: the walk
    # and the scope it registers under must never disagree about which directory
    # `dir="."` meant.
    root = _session_abspath(dir)
    project = _scope_path(root)
    host = _env("RETRIEVAL_HOST", "RETRIVAL_HOST", default=socket.gethostname())
    block, nested_warning = await _nested_scope_guard(
        project, host or "", "doc", respect_gitignore)
    if block:
        return block
    conversation_line = _conversation_sync_line(root)
    patterns = _expand_brace_patterns(pattern)
    matched = sorted({p for pat in patterns
                      for p in _discover_files(root, pat, recursive, max_depth)
                      if os.path.isfile(p)})
    # Drop rendered-image / binary files even if the pattern matched them (W161).
    files = [p for p in matched if os.path.splitext(p)[1].lower() not in _SKIP_INDEX_EXTS]
    skipped_binary = len(matched) - len(files)
    # git subprocess off the event loop (blocking), like the index_code / _git_commit path.
    files, skipped_gitignored = await asyncio.to_thread(
        _drop_gitignored, files, root, respect_gitignore)
    notes = [n for n in (f"{skipped_binary} binary/image skipped" if skipped_binary else None,
                         f"{skipped_gitignored} gitignored" if skipped_gitignored else None) if n]
    skip_note = f", {', '.join(notes)}" if notes else ""
    if not files:
        return _with_conversation_line(
            f"No indexable text files matching {pattern!r} in {root} "
            f"(recursive={recursive}, max_depth={max_depth}{skip_note}).", conversation_line)

    effective_max_depth = None if (recursive and max_depth is None) else (max_depth if recursive else 0)
    outcome = await _index_doc_files(
        root, project, host, files, patterns, effective_max_depth,
        nested_warning=nested_warning, skip_note=skip_note)
    return _with_conversation_line(outcome.message, conversation_line)


@dataclass
class _DocIndexOutcome:
    """What one document pass did. ``message`` is the sentence journal_index_dir returns;
    ``job_id`` is the Journal bulk-index job still running for the changed files (None
    when the pass finished inline); ``records`` is the doc catalog snapshot the pass
    reported (changed rows with chunk_count 0 until their job completes)."""
    message: str
    job_id: str | None = None
    records: list[dict] = dataclasses.field(default_factory=list)
    queued: int = 0
    unchanged: int = 0
    failed: int = 0
    inventory_ok: bool = True


def _doc_catalog_row(candidate: _JournalFileCandidate, chunk_count: int) -> dict:
    return {"rel_path": candidate.rel_path, "kind": "doc", "collection": "journal_chunks",
            "chunk_count": chunk_count, "content_hash": candidate.content_hash,
            "size_bytes": candidate.size_bytes}


async def _index_doc_files(
    root: str, project: str, host: str | None, files: list[str],
    patterns: list[str], max_depth: int | None, *,
    known: dict | None = None, snapshot: bool = True,
    nested_warning: str = "", skip_note: str = "",
) -> _DocIndexOutcome:
    """The shared document pipeline behind index_code and journal_index_dir: diff ``files``
    (absolute paths, already gated by the caller) against the scope's doc catalog, send
    only the changed bodies through /api/journal/bulk-index, and let the sweep envelope
    (``patterns`` + ``max_depth``; None = unbounded) bound what a snapshot may remove.

    ``snapshot=False`` is a partial update (the watcher's changed files): nothing may be
    swept, and ``known`` (the caller's doc baseline) replaces the inventory lookup."""
    # The full-text POST is the expensive half of Journal indexing.  Diff against
    # this scope's durable catalog first, so unchanged files do not enter HTTP,
    # SQL/FTS, embedding or Qdrant.  An inventory outage deliberately falls back
    # to backend latest-wins dedupe rather than risking a missed update.
    inventory_note = ""
    if known is None:
        known = {}
        try:
            inventory = await _get("/api/index/inventory", {
                "host": host, "project": project, "kind": "doc", "detail": True, "lean": True,
            })
            known = _flatten_inventory_doc_files(inventory)
        except Exception as exc:  # noqa: BLE001
            logger.warning("document index: doc inventory unavailable; no local unchanged skip: %s", exc)
            inventory_note = "; local unchanged skip unavailable (inventory lookup failed)"
    candidates = await asyncio.to_thread(
        lambda: [_journal_file_candidate(path, root) for path in files])
    known_in_envelope = _journal_known_in_envelope(known, patterns, max_depth)
    unchanged, replace_safe, replace_block_reason = _journal_snapshot_diff(
        candidates, known_in_envelope, case_sensitive=not _is_windows_absolute_path(root))
    if not snapshot:
        # A watcher batch names only what changed; it is not a snapshot of the scope.
        replace_safe, replace_block_reason = False, "partial update"
    elif not replace_safe:
        logger.warning("document index: snapshot replace disabled for %s: %s",
                       project, replace_block_reason)

    # A re-run that finds nothing to do: every file is readable and byte-identical to its
    # catalog row, and no cataloged file of this pattern/depth is missing locally. The
    # full-snapshot report (replace=True) and the stale-doc reconcile below would then
    # rewrite the same rows and remove nothing, so skip both round trips. The text is the
    # one that unchanged path prints today.
    local_paths = {candidate.rel_path for candidate in candidates}
    if (not inventory_note and replace_safe
            and all(candidate.readable and candidate.rel_path in unchanged
                    for candidate in candidates)
            and known_in_envelope.keys() <= local_paths):
        count = len(candidates)
        prefix = f"{nested_warning} " if nested_warning else ""
        return _DocIndexOutcome(
            message=(
                f"{prefix}Indexed {count} file(s) into journal '{project}' "
                f"({count} unchanged, 0 failed{skip_note}); "
                f"reported {count} to inventory (host={host}); 0 stale doc(s) removed."),
            records=[
                _doc_catalog_row(candidate, int(
                    known_in_envelope[candidate.rel_path].get("chunk_count", 0) or 0))
                for candidate in candidates],
            unchanged=count)

    if any(candidate.rel_path not in unchanged and candidate.readable for candidate in candidates):
        report_files: list[dict] = []
        bulk_files: list[dict] = []
        failed_reads = 0
        for candidate in candidates:
            if not candidate.readable:
                failed_reads += 1
                continue
            if candidate.rel_path in unchanged:
                previous = known_in_envelope[candidate.rel_path]
                report_files.append(_doc_catalog_row(
                    candidate, int(previous.get("chunk_count", 0) or 0)))
                continue
            try:
                content = await asyncio.to_thread(
                    # Keep the uploaded body byte-faithful to the snapshot hash.
                    # ``newline=''`` observes CRLF without translating it to LF.
                    lambda path=candidate.abs_path: open(
                        path, encoding="utf-8", errors="replace", newline="").read())
            except OSError:
                logger.exception("document index: failed to read %s", candidate.abs_path)
                failed_reads += 1
                continue
            if not content.strip():
                continue
            bulk_files.append({"title": candidate.rel_path, "body": content,
                               "content_hash": candidate.content_hash, "size_bytes": candidate.size_bytes})
            report_files.append(_doc_catalog_row(candidate, 0))
        if bulk_files:
            try:
                job = await _post("/api/journal/bulk-index", {
                    "project_slug": project, "host": host or "", "root_dir": root,
                    "files": bulk_files, "report_files": report_files,
                    "replace": replace_safe and failed_reads == 0,
                    "live_titles": [row["rel_path"] for row in report_files],
                    "patterns": patterns, "max_depth": max_depth,
                })
                if job.get("job_id"):
                    return _DocIndexOutcome(
                        message=(
                            f"Queued Journal bulk index job '{job.get('job_id')}' for {len(bulk_files)} changed "
                            f"file(s) in '{project}' ({len(unchanged)} unchanged locally, {failed_reads} failed); "
                            f"poll journal_index_status(job_id='{job.get('job_id')}')."),
                        job_id=str(job.get("job_id")), records=report_files,
                        queued=len(bulk_files), unchanged=len(unchanged), failed=failed_reads,
                        inventory_ok=not inventory_note)
            except Exception:
                logger.warning("journal bulk endpoint unavailable; using compatibility path", exc_info=True)

    indexed = deduped = failed = dup_synced = 0
    records: list[dict[str, Any]] = []
    _sem = asyncio.Semaphore(_JOURNAL_INDEX_CONCURRENCY)

    def _record(candidate: _JournalFileCandidate, *, chunk_count: int) -> dict:
        return _doc_catalog_row(candidate, chunk_count)

    async def _index_one(candidate: _JournalFileCandidate) -> tuple[str, dict | None, bool, int]:
        async with _sem:
            try:
                if not candidate.readable:
                    raise OSError(candidate.read_error or "unable to read file")
                if candidate.rel_path in unchanged:
                    previous = known_in_envelope[candidate.rel_path]
                    return "unchanged", _record(
                        candidate, chunk_count=int(previous.get("chunk_count", 0) or 0)), True, 0
                with open(candidate.abs_path, encoding="utf-8", errors="replace", newline="") as fh:
                    content = fh.read()
                if not content.strip():
                    return "skip", None, False, 0
                resp = await _post("/api/journal/entries", {
                    "project_slug": project, "client": "mcp", "source_kind": "doc",
                    "title": candidate.rel_path, "body": content, "tags": [],
                    "dedupe_policy": "return_existing",
                })
                rec = _record(candidate, chunk_count=int(resp.get("chunk_count", 0) or 0))
                return ("ok", rec, bool(resp.get("deduped")),
                        int(resp.get("duplicates_updated", 0) or 0))
            except Exception:  # noqa: BLE001 — keep going; one bad file shouldn't abort the batch
                logger.exception("document index: failed to index %s", candidate.abs_path)
                return "fail", None, False, 0

    # Bounded fan-out; gather preserves submission order so `records` (and thus the
    # inventory snapshot) is identical to the serial walk.
    for status, rec, was_deduped, dup in await asyncio.gather(*(_index_one(candidate) for candidate in candidates)):
        if status in {"ok", "unchanged"}:
            indexed += 1
            if was_deduped:
                deduped += 1
            dup_synced += dup
            records.append(rec)
        elif status == "fail":
            failed += 1

    reported = 0
    dup_note = f", {dup_synced} cross-scope synced" if dup_synced else ""
    # This scan's sweep envelope — shared by the catalog replace AND the journal reconcile so
    # they stay in lockstep: depth (finding #4, non-recursive → root only) AND pattern (W169b,
    # a "*.txt" re-run must not sweep the "*.md" files it never scanned, in either store).
    prefix = f"{nested_warning} " if nested_warning else ""
    try:
        # A destructive snapshot replace is only safe when EVERY file was read this run — a
        # file that failed to read is absent from `records` and would be swept from the
        # catalog even though its journal data is still live. On any failure, upsert only
        # (replace=False), matching the stale-doc reconcile gate below (finding #3). The
        # pattern/depth envelope further scopes the sweep to what this run could have indexed.
        rep = await _post("/api/index/report", {
            "host": host, "project": project, "root_dir": root, "files": records,
            "replace": failed == 0 and replace_safe, "patterns": patterns, "max_depth": max_depth,
        })
        reported = rep.get("total_files", len(records))
    except Exception as exc:  # noqa: BLE001 — inventory reporting is best-effort
        return _DocIndexOutcome(
            message=(
                f"{prefix}Indexed {indexed} file(s) into journal '{project}' "
                f"({deduped} unchanged, {failed} failed{skip_note}{dup_note}); inventory report skipped: {exc}"),
            records=records, unchanged=deduped, failed=failed, inventory_ok=not inventory_note)
    reconcile_note = ""
    if failed:
        reconcile_note = "; stale doc reconcile skipped because some files failed"
    elif not replace_safe:
        reconcile_note = f"; stale doc reconcile skipped: {replace_block_reason}"
    else:
        try:
            reconciled = await _post("/api/journal/docs/reconcile", {
                "project_slug": project,
                "live_titles": [record["rel_path"] for record in records],
                "max_depth": max_depth,
                "patterns": patterns,
            })
            if reconciled.get("skipped"):
                reconcile_note = f"; stale doc reconcile skipped: {reconciled.get('reason')}"
            else:
                reconcile_note = f"; {reconciled.get('removed_entries', 0)} stale doc(s) removed"
        except Exception as exc:  # noqa: BLE001
            reconcile_note = f"; stale doc reconcile skipped: {exc}"
    return _DocIndexOutcome(
        message=(
            f"{prefix}Indexed {indexed} file(s) into journal '{project}' "
            f"({deduped} unchanged, {failed} failed{skip_note}{dup_note}); "
            f"reported {reported} to inventory (host={host}){inventory_note}{reconcile_note}."),
        records=records, unchanged=deduped, failed=failed, inventory_ok=not inventory_note)


@mcp.tool()
async def journal_index_status(job_id: str) -> str:
    """Poll the asynchronous Journal bulk-index receipt from journal_index_dir."""
    job = await _get(f"/api/journal/bulk-index/jobs/{quote(job_id, safe='')}")
    return (f"Journal job {job.get('job_id', job_id)}: {job.get('status', 'unknown')} "
            f"files={job.get('files_done', 0)}/{job.get('files_total', 0)} "
            f"chunks={job.get('chunks_done', 0)}/{job.get('chunks_total', 0)} "
            f"indexed={job.get('indexed', 0)} progress={job.get('progress', 0):.0%}"
            + (f" skipped_binary={', '.join(job['skipped_binary'])}" if job.get('skipped_binary') else "")
            + (f" error={job.get('error')}" if job.get('error') else ""))


@mcp.tool()
async def index_inventory(host: str | None = None, project: str | None = None,
                          limit: int = 40) -> str:
    """List YOUR indexed projects: one line per indexed directory with its file count
    and last-indexed time, grouped by machine. Scopes whose code index needs attention
    come first, then the most recently indexed. Use it to see what index_code /
    journal_index_dir have registered; narrow with project or host to find one.

    Args:
        host: Only this machine.
        project: Only projects whose path contains this fragment, e.g. "retrival" or
            "E:/JETB".
        limit: Most projects to list.
    """
    params = {k: v for k, v in {"host": host, "project_contains": project}.items() if v}
    resp = await _get("/api/index/inventory", params)
    # Filtered again here: a backend that predates project_contains returns every project.
    return _format_inventory_summary(resp, project=project, limit=limit)


def _project_freshness(project: dict) -> str:
    """' — last indexed <ts>[ @<commit>]' for a project node, or '' if unknown.

    Prefers the server-computed summary fields (last_indexed_at / last_commit) so the compact
    inventory needs no per-file tree; falls back to walking the file nodes for a detail
    response or an older server. A git commit, if the client recorded one in file metadata at
    index time, is surfaced too so you can tell WHICH revision is indexed."""
    latest = project.get("last_indexed_at") or ""
    commit = project.get("last_commit") or ""
    if not latest:  # detail response / older server → derive from the per-file nodes
        for d in project.get("dirs") or []:
            for f in d.get("files") or []:
                ts = f.get("indexed_at") or ""
                if ts > latest:
                    latest = ts
                commit = commit or ((f.get("metadata") or {}).get("git_commit") or "")
    if not latest:
        return ""
    return f" — last indexed {latest[:19].replace('T', ' ')}" + (f" @{commit}" if commit else "")


def _code_snapshot_inventory_note(project: dict) -> str | None:
    state = project.get("code_snapshot_state")
    if not state:
        return None
    line = f"code snapshot: {state}"
    job_id = project.get("code_active_index_job_id")
    if job_id:
        line += f"; job={job_id}"
    if project.get("code_recovery_required"):
        line += "; full recovery index required after interrupted write"
    previous = project.get("code_previous_completed_snapshot_available")
    if previous is False:
        line += "; completed snapshot unavailable"
    elif previous is True and state == "preparing":
        line += "; previous completed snapshot available"
    return line

def _same_scope_path(left: str, right: str) -> bool:
    return left.replace("\\", "/").rstrip("/").lower() == right.replace("\\", "/").rstrip("/").lower()


def _format_inventory_summary(resp: dict, *, project: str | None = None,
                              limit: int = 40) -> str:
    """One line per project, at most `limit`: scopes whose code snapshot is not plainly
    completed first (they need attention), then newest first. Rendering all 305 scopes
    of one user, each with its path twice and a "completed" note, was 56K characters,
    more than MCP clients accept (2026-10-09)."""
    users = resp.get("users") or []
    totals = resp.get("totals") or {}
    if not users:
        return "No indexed files reported yet."
    needle = (project or "").replace("\\", "/").lower()
    rows = []
    for user in users:
        owner = user.get("owner_user_id") or "?"
        for host in user.get("hosts") or []:
            for node in host.get("projects") or []:
                name = str(node.get("project") or "?")
                if needle and needle not in name.replace("\\", "/").lower():
                    continue
                attention = (node.get("code_snapshot_state") not in (None, "completed")
                             or bool(node.get("code_recovery_required")))
                label = host.get("host") or "?"
                if len(users) > 1:
                    label = f"{owner} @ {label}"
                rows.append((attention, node.get("last_indexed_at") or "", label, node))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    shown = rows[:max(1, limit)]
    lines = [f"Inventory: {totals.get('hosts', 0)} host(s), {totals.get('projects', 0)} "
             f"project(s), {totals.get('files', 0)} file(s), {totals.get('chunks', 0)} chunk(s)."]
    if needle:
        lines.append(f"{len(rows)} project(s) match {project!r}.")
    if len(rows) > len(shown):
        lines.append(f"Listing {len(shown)} of {len(rows)}: scopes needing attention, then "
                     "the most recently indexed. Narrow with project=<path fragment> or "
                     "host=, or raise limit.")
    per_host: dict[str, int] = {}
    for row in rows:
        per_host[row[2]] = per_host.get(row[2], 0) + 1
    groups: dict[str, list[dict]] = {}
    for _, _, label, node in shown:
        groups.setdefault(label, []).append(node)
    for label, nodes in groups.items():
        lines.append(f"▸ {label} ({len(nodes)} of {per_host[label]} project(s))")
        for node in nodes:
            name = str(node.get("project") or "?")
            line = f"  · {name} ({node.get('file_count', 0)} files){_project_freshness(node)}"
            root = node.get("root_dir")
            if root and not _same_scope_path(str(root), name):
                line += f" [root {root}]"
            note = _code_snapshot_inventory_note(node)
            if note and (node.get("code_snapshot_state") != "completed"
                         or node.get("code_recovery_required")):
                line += f"; {note}"
            lines.append(line)
    return "\n".join(lines)


# ------------------------------ shared formatters ----------------------------- #

def _fmt_papers(resp: dict) -> str:
    papers = resp.get("papers") or []
    if not papers:
        return "No matching papers found."
    lines = [f"Found {len(papers)} papers:", ""]
    for i, p in enumerate(papers, 1):
        meta = p.get("metadata") or {}
        venue = meta.get("venue") or p.get("venue") or "?"
        year = meta.get("year") or p.get("year") or "?"
        title = p.get("doc_title") or p.get("title") or "(untitled)"
        score = p.get("max_score", p.get("score", 0.0))
        withdrawn = "  [WITHDRAWN]" if meta.get("withdrawn") else ""
        lines.append(f"{i}. {title}  [{venue} {year}]  score={score:.3f}{withdrawn}")
        # An upload/webpage has no paper_id; emit its doc: handle so the id on
        # this line is always one a follow-up tool can actually use. Printing an
        # empty paper_id made books un-drillable: search found them, then
        # search_within_paper had nothing to pass.
        result_id = str(p.get("paper_id") or "").strip()
        if not result_id:
            doc_id = str(p.get("doc_id") or "").strip()
            result_id = f"{_DOC_ID_PREFIX}{doc_id}" if doc_id else ""
        lines.append(f"   paper_id: {result_id}")
        chunks = p.get("chunks") or []
        if chunks:
            lines.append(f"   ▸ {_snippet(chunks[0].get('content', ''))}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _fmt_match(resp: dict) -> str:
    papers = resp.get("papers") or []
    if not papers:
        return "No similar papers found."
    lines = [f"Top {len(papers)} content-nearest papers:", ""]
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. {p.get('title') or '(untitled)'}  [{p.get('venue', '?')} "
                     f"{p.get('year', '?')}]  score={p.get('score', 0.0):.3f}  "
                     f"coverage={p.get('coverage', 0)}")
        lines.append(f"   paper_id: {p.get('paper_id') or ''}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ------------------------------- code KB (server-side) ------------------------------ #
# Client AST-chunks the repo LOCALLY (chonkie CodeChunker); only chunks leave the machine.
# The backend embeds them with the shared BGE-M3 and stores vectors in the 4090
# `code_chunks` collection, scoped to the caller — mirrors mcp_server/ + core/code_kb/
# chunker.py so the two entry points index identically.

# file extension -> tree-sitter grammar name (chonkie passes it straight to the grammar
# pack, so 40+ languages work). .tsx grammar is JSX-capable → React .jsx maps to it.
# NOTE: kept in sync with core/code_kb/chunker.py (this file ships standalone).
LANG_BY_EXT = {
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cu": "cpp", ".cuh": "cpp",  # CUDA uses the C++ grammar.
    ".rs": "rust", ".go": "go", ".zig": "zig", ".nim": "nim", ".swift": "swift",
    ".m": "objc", ".mm": "objc",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".sc": "scala",
    ".groovy": "groovy", ".gradle": "groovy",
    ".cs": "csharp",
    ".py": "python", ".pyi": "python", ".rb": "ruby", ".php": "php", ".lua": "lua",
    ".pl": "perl", ".pm": "perl", ".r": "r", ".jl": "julia", ".dart": "dart",
    ".ex": "elixir", ".exs": "elixir", ".erl": "erlang", ".hrl": "erlang",
    ".hs": "haskell", ".ml": "ocaml", ".mli": "ocaml",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx", ".jsx": "tsx",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".vue": "vue", ".svelte": "svelte",
    ".css": "css", ".scss": "scss", ".html": "html", ".htm": "html",  # .sass → fallback (indented ≠ scss)
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".ps1": "powershell",
    ".tf": "terraform", ".hcl": "hcl", ".dockerfile": "dockerfile", ".cmake": "cmake",
    ".sol": "solidity",
    ".sql": "sql", ".graphql": "graphql", ".gql": "graphql", ".proto": "proto",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml",
    ".xml": "xml", ".ini": "ini",
}
LANG_BY_NAME = {"dockerfile": "dockerfile", "makefile": "make", "cmakelists.txt": "cmake"}
# Grammarless (or grammar-mismatched, e.g. indented .sass) code-adjacent text → line
# windows. NOT ".txt" (would sweep requirements.txt / LICENSE / data into the code KB).
FALLBACK_EXTS = {".env", ".properties", ".cfg", ".conf", ".editorconfig", ".sass"}
# Documentation: index_code hands these to the journal pipeline in the same walk. ".txt"
# is a document here (CMakeLists.txt still resolves to code first, see _walk_repo_files).
DOC_EXTS = frozenset({".md", ".mdx", ".markdown", ".rst", ".adoc", ".txt"})
# The sweep envelope of that pipeline: reconcile only ever removes docs matching these
# basenames, so a scope's other doc rows (journal_index_dir "*.json") stay untouched.
DOC_GLOBS = ["*.md", "*.mdx", "*.markdown", "*.rst", "*.adoc", "*.txt"]
_MAX_CHUNK_SIZE = 1500  # chars per chunk (chonkie tokenizer="character")
# Sources this large are cut at blank lines before chunking (see _segments): a 9,255-line
# C++ file used to come back as one 567K-char hit (2026-10-13).
_LARGE_SOURCE_THRESHOLD_CHARS = 384 * 1024
_LARGE_SOURCE_SEGMENT_CHARS = 64 * 1024
_BLANK_LINE_RE = re.compile(r"\n[ \t\r\f\v]*\n")
_CHUNK_WORKER_STACK_SIZE = 64 * 1024 * 1024
_CHUNK_WORKER_START_TIMEOUT_SECONDS = 30
# Chunks with fewer non-whitespace chars than this are brace slivers chonkie leaves at
# split boundaries — merged into a neighbor, never emitted standalone (W157).
_MIN_CHUNK_NWS = 24
# tree-sitter's traversal recurses per nesting level and C-stack-overflows around ~1200
# bracket depth — a process-killing crash Python cannot catch (W155b). Deeper sources
# (minified JS, generated JSON) go straight to line windows.
_MAX_AST_NESTING = 500
# Files bigger than this are never hand-written code (minified bundles, lockfiles,
# generated blobs) — parsing wedges tree-sitter and the chunks are search noise. Skipped.
_MAX_FILE_BYTES = 2 * 1024 * 1024
_LABEL_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:pub\s+)?(?:def|class|function|public|"
                       r"private|protected|func|fn|interface|struct|enum|impl|trait|type|val|"
                       r"var|const|let)\s+([A-Za-z_]\w*)", re.M)
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
    ".next", "coverage", ".pytest_cache", ".mypy_cache", ".code_kb", "target",
    # Project layout names are source by default; projects can opt out through
    # .gitignore or .retrievalignore instead of a global directory-name guess.
}


@dataclass
class _CodeChunk:
    path: str            # repo-relative, forward slashes
    start_line: int      # 1-based
    end_line: int
    content: str         # "# path: ...\n" + code  (this is what gets embedded)
    code: str            # raw source segment
    label: str = ""      # best-effort symbol name


def _code_label(code: str) -> str:
    m = _LABEL_RE.search(code)
    return m.group(1) if m else ""


def _ext_of(path: str) -> str:
    """Lowercased extension, treating a dotfile (`.env`) as its own extension."""
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1]
    return name if (not ext and name.startswith(".")) else ext


def _resolve_language(path: str) -> str | None:
    name = os.path.basename(path).lower()
    return LANG_BY_NAME.get(name) or LANG_BY_EXT.get(_ext_of(path))


_BUILDER_LOCAL = threading.local()
_CHUNK_EXECUTOR_START_LOCK = threading.Lock()


def _builder(language: str, max_chunk_size: int = _MAX_CHUNK_SIZE):
    """Cache CodeChunker instances per worker thread, never across threads."""
    cache = getattr(_BUILDER_LOCAL, "builders", None)
    if cache is None:
        cache = {}
        _BUILDER_LOCAL.builders = cache
    key = (language, max_chunk_size)
    builder = cache.get(key)
    if builder is None:
        builder = CodeChunker(
            language=language,
            tokenizer="character",
            chunk_size=max_chunk_size,
            include_nodes=False,
        )
        cache[key] = builder
    return builder


def _max_nesting(source: str) -> int:
    """Peak bracket-nesting depth — cheap AST-depth proxy computed WITHOUT parsing
    (the whole point is to refuse sources that would blow the parser's C stack)."""
    depth = peak = 0
    for ch in source:
        if ch in "([{":
            depth += 1
            if depth > peak:
                peak = depth
        elif ch in ")]}":
            depth = max(depth - 1, 0)
    return peak


def _char_spans(source: str, results) -> list[list[int]]:
    """[start, end) character spans of chonkie chunks, whitespace-only ones dropped.

    chonkie's CodeChunker reports UTF-8 byte offsets (tree-sitter start_byte, still in
    1.7.0), its other chunkers characters. Slicing a str with byte offsets cut every
    chunk after the first non-ASCII character short at the front and cited the wrong
    lines (2026-10-13). The convention is decided per file by checking each offset
    against the chunk's own text; an offset that locates neither is an error."""
    pieces = [(r.start_index, r.text) for r in results if r.text.strip()]
    encoded = source.encode("utf-8")
    if all(encoded[start:start + len(text.encode("utf-8"))] == text.encode("utf-8")
           for start, text in pieces):
        spans: list[list[int]] = []
        byte_pos = char_pos = 0
        for start, text in pieces:
            if start >= byte_pos:
                char_pos += len(encoded[byte_pos:start].decode("utf-8"))
            else:
                char_pos = len(encoded[:start].decode("utf-8"))
            byte_pos = start
            spans.append([char_pos, char_pos + len(text)])
        return spans
    if all(source[start:start + len(text)] == text for start, text in pieces):
        return [[start, start + len(text)] for start, text in pieces]
    raise ValueError("chunk offsets locate neither bytes nor characters of the source")


def _segments(source: str, limit: int) -> list[tuple[int, int, bool]]:
    """Cut a large source into [lo, hi) pieces of at most `limit` chars, each ending
    just before a blank line where one exists. A piece is structural when both of its
    ends sit on a blank line (or the file edge), so it holds whole declarations."""
    pieces: list[tuple[int, int, bool]] = []
    lo, clean_start, size = 0, True, len(source)
    while lo < size:
        hi, clean_end = min(size, lo + limit), True
        if hi < size:
            blank = None
            for blank in _BLANK_LINE_RE.finditer(source, lo, hi):
                pass
            if blank is not None and blank.start() > lo:
                hi = blank.start() + 1
            else:
                clean_end = False
                cut = source.rfind("\n", lo, hi)
                if cut >= lo:
                    hi = cut + 1
        pieces.append((lo, hi, clean_start and clean_end))
        lo, clean_start = hi, clean_end
    return pieces


def _chunk_source(path: str, source: str, language: str) -> list[_CodeChunk]:
    # 1-based line ranges come from counting newlines before and inside each chunk
    # (trailing blank lines stripped so a citation points at real code).
    depth = _max_nesting(source)
    if depth > _MAX_AST_NESTING:
        # Would C-stack-overflow the parser (uncatchable, kills the process — W155b).
        logger.warning("%s nesting depth %d > %d — line-window fallback, no AST",
                       path, depth, _MAX_AST_NESTING)
        return _chunk_lines(path, source)
    if len(source) < _LARGE_SOURCE_THRESHOLD_CHARS:
        return _ast_chunks(path, source, language)
    # Pieces with blank lines at both ends hold whole declarations and are AST-chunked;
    # a piece cut inside one huge node (an FFmpeg-style data table) is line-windowed.
    out: list[_CodeChunk] = []
    line_offset = 0
    for lo, hi, structural in _segments(source, _LARGE_SOURCE_SEGMENT_CHARS):
        piece = source[lo:hi]
        part = _ast_chunks(path, piece, language) if structural else _chunk_lines(path, piece)
        out.extend(dataclasses.replace(chunk, start_line=chunk.start_line + line_offset,
                                       end_line=chunk.end_line + line_offset)
                   for chunk in part)
        line_offset += piece.count("\n")
    return out


def _ast_chunks(path: str, source: str, language: str) -> list[_CodeChunk]:
    builder = _builder(language, _MAX_CHUNK_SIZE)
    spans = _char_spans(source, builder.chunk(source))  # [start, end) into source

    # Merge brace-sliver fragments (W157): chonkie splits an oversized function at child
    # boundaries and leaves the parent's braces as standalone "{"/"}" chunks — embedded,
    # those are nothing but the path header and pollute search. A tiny span joins the
    # previous real chunk; head slivers prepend to the first real chunk.
    merged: list[list[int]] = []
    pending: list[int] | None = None
    for s in spans:
        if len("".join(source[s[0]:s[1]].split())) < _MIN_CHUNK_NWS:
            if merged:
                merged[-1][1] = s[1]
            else:
                pending = [pending[0], s[1]] if pending else s
            continue
        if pending:
            s = [pending[0], s[1]]
            pending = None
        merged.append(s)
    if pending:  # whole file was slivers — emit as one chunk rather than drop it
        if merged:
            merged[-1][1] = pending[1]
        else:
            merged.append(pending)

    out: list[_CodeChunk] = []
    for lo, hi in merged:
        code = source[lo:hi]
        start = source.count("\n", 0, lo) + 1
        end = start + code.rstrip("\n").count("\n")
        out.append(_CodeChunk(path=path, start_line=start, end_line=max(end, start),
                              content=f"# path: {path}\n{code}", code=code, label=_code_label(code)))
    return out


def _chunk_lines(path: str, source: str) -> list[_CodeChunk]:
    """Line-window fallback for grammarless files — never silently drop a file. No symbol
    label: these tiers have no AST, so a code-symbol regex would only misfire.

    A single line longer than 4×_MAX_CHUNK_SIZE (minified bundles are one huge line) is
    hard-split at _MAX_CHUNK_SIZE so no chunk balloons past embedding/request limits."""
    hard_cap = 4 * _MAX_CHUNK_SIZE
    out: list[_CodeChunk] = []
    lines = source.splitlines()
    buf: list[str] = []
    buf_start = 1
    size = 0

    def _flush_hard_split(line: str, lineno: int) -> None:
        for off in range(0, len(line), _MAX_CHUNK_SIZE):
            piece = line[off:off + _MAX_CHUNK_SIZE]
            out.append(_CodeChunk(path=path, start_line=lineno, end_line=lineno,
                                  content=f"# path: {path}\n{piece}", code=piece))

    def _flush(end_line: int) -> None:
        """Emit the buffer as one chunk — only when it holds non-whitespace text (an
        all-blank window would be an empty `code`, which the server rejects with 422) —
        and always empty it; the next window starts after `end_line`."""
        nonlocal buf, buf_start, size
        if buf and "".join(buf).strip():
            code = "\n".join(buf)
            out.append(_CodeChunk(path=path, start_line=buf_start, end_line=end_line,
                                  content=f"# path: {path}\n{code}", code=code))
        buf, buf_start, size = [], end_line + 1, 0

    for i, line in enumerate(lines, 1):
        if len(line) > hard_cap:
            _flush(i - 1)
            buf_start = i + 1
            _flush_hard_split(line, i)
            continue
        line_nws = len(re.sub(r"\s", "", line))
        if buf and size + line_nws > _MAX_CHUNK_SIZE:
            _flush(i - 1)
        buf.append(line)
        size += line_nws
    _flush(len(lines))
    return out


def _chunk_file(path: str, repo_root: str) -> list[_CodeChunk]:
    language = _resolve_language(path)
    if not language and _ext_of(path) not in FALLBACK_EXTS:
        return []
    try:
        if os.path.getsize(path) > _MAX_FILE_BYTES:
            logger.warning("skipping %s (> %d bytes — generated/vendored, not code)",
                           path, _MAX_FILE_BYTES)
            return []
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []
    rel = os.path.relpath(path, repo_root).replace("\\", "/")
    # BEFORE parsing: on a wedge/hard-crash this is the last stderr line → names the file.
    logger.info("chunking %s (%s, %d bytes)", rel, language or "line-window", len(source))
    if language:
        try:
            return _chunk_source(rel, source, language)
        except Exception:  # noqa: BLE001 — degrade to line windows rather than drop the file
            logger.exception("AST chunk failed for %s (%s) → line-window fallback", rel, language)
            return _chunk_lines(rel, source)
    return _chunk_lines(rel, source)


def _load_retrievalignore(repo_root: str) -> list[str]:
    """Read simple scope-relative excludes usable with or without Git."""
    try:
        lines = Path(repo_root, ".retrievalignore").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line.strip().replace("\\", "/") for line in lines
            if line.strip() and not line.lstrip().startswith("#")]


def _excluded(rel_path: str, patterns: list[str]) -> bool:
    rel = rel_path.replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    for raw in patterns:
        pattern = raw.strip().replace("\\", "/")
        if pattern.startswith("./"):
            pattern = pattern[2:]
        if not pattern:
            continue
        if pattern.endswith("/"):
            pattern += "**"
        if fnmatch.fnmatchcase(rel, pattern):
            return True
        if pattern.endswith("/**") and rel == pattern[:-3].rstrip("/"):
            return True
    return False


def _effective_excludes(repo_root: str, exclude_globs: list[str] | None) -> list[str]:
    return list(dict.fromkeys(_load_retrievalignore(repo_root) + list(exclude_globs or [])))


def _is_hidden_rel_path(rel_path: str) -> bool:
    """True when any segment of a scope-relative path is a dot name (``.github/x.md``,
    ``docs/.draft.md``)."""
    return any(part.startswith(".") for part in rel_path.replace("\\", "/").split("/"))


def _is_doc_path(path: str, rel_path: str) -> bool:
    """A documentation file the journal pipeline owns: a DOC_EXTS extension outside every
    dot path. Dot paths are excluded because the journal's sweep envelope never covers
    them (``.conversations/`` holds synced chat docs), so a doc indexed there could
    neither be skipped as unchanged nor reconciled away."""
    return _ext_of(path) in DOC_EXTS and not _is_hidden_rel_path(rel_path)


def _walk_repo_files(repo_root: str, respect_gitignore: bool = True,
                     exclude_globs: list[str] | None = None, docs: bool = True):
    """(abs_path, rel_path, kind) for every indexable file, from ONE directory walk.

    kind is "code" — a resolvable language or a fallback ext, i.e. what would be
    chunked, decided WITHOUT parsing — or "doc" for DOC_EXTS files (``docs=False`` omits
    them). Code wins when a name qualifies as both (``CMakeLists.txt``). The skipped
    directories, exclude globs, ``.retrievalignore`` and git's ignore rules gate both
    kinds identically.

    respect_gitignore (default True) skips files git ignores — see _gitignore_allowed."""
    allowed = _gitignore_allowed(repo_root) if respect_gitignore else None
    excludes = _effective_excludes(repo_root, exclude_globs)
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not _excluded(
            os.path.relpath(os.path.join(root, d), repo_root).replace("\\", "/"), excludes)]
        for fn in files:
            path = os.path.join(root, fn)
            if _resolve_language(path) or _ext_of(path) in FALLBACK_EXTS:
                kind = "code"
            elif docs and _ext_of(path) in DOC_EXTS:
                kind = "doc"
            else:
                continue
            rel = os.path.relpath(path, repo_root).replace("\\", "/")
            if kind == "doc" and _is_hidden_rel_path(rel):
                continue
            if _excluded(rel, excludes):
                continue
            if allowed is not None and rel not in allowed:
                continue
            yield path, rel, kind


def _iter_repo_files(repo_root: str, respect_gitignore: bool = True,
                     exclude_globs: list[str] | None = None):
    """(abs_path, rel_path) for every file that WOULD be chunked — a resolvable language
    or a fallback ext — WITHOUT parsing. Lets incremental indexing hash candidates
    cheaply and AST-chunk only the changed subset. Mirrors _collect_code_files's gate.

    The code-only view of _walk_repo_files."""
    for path, rel, _kind in _walk_repo_files(
            repo_root, respect_gitignore=respect_gitignore,
            exclude_globs=exclude_globs, docs=False):
        yield path, rel


def _file_content_hash(path: str) -> str | None:
    """SHA-256 hex of a file's bytes, or None if unreadable — the content identity for
    incremental indexing. Hashing (not git-diff) is drift-proof: the server inventory is
    the single source of truth, so force-push / rename / gitignore divergence can't
    desync the index, and the cost is negligible next to the AST parse + embed it skips."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# Local AST chunking is CPU-bound but tree-sitter releases the GIL while parsing, so a
# bounded thread pool gives real parallelism (and overlaps file reads). Cap it so a huge
# tree can't spawn hundreds of threads.
_CHUNK_WORKERS = max(1, min(16, (os.cpu_count() or 4)))


def _chunk_executor(worker_count: int) -> ThreadPoolExecutor:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")

    with _CHUNK_EXECUTOR_START_LOCK:
        original_stack_size = threading.stack_size()
        stack_size_changed = original_stack_size == 0 or original_stack_size < _CHUNK_WORKER_STACK_SIZE
        ready = threading.Barrier(worker_count + 1)
        release = threading.Event()
        executor = None
        warmups = []

        def hold_worker() -> None:
            ready.wait(timeout=_CHUNK_WORKER_START_TIMEOUT_SECONDS)
            release.wait()

        try:
            if stack_size_changed:
                threading.stack_size(_CHUNK_WORKER_STACK_SIZE)
            executor = ThreadPoolExecutor(max_workers=worker_count)
            warmups = [executor.submit(hold_worker) for _ in range(worker_count)]
            ready.wait(timeout=_CHUNK_WORKER_START_TIMEOUT_SECONDS)
        except Exception:
            ready.abort()
            release.set()
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if stack_size_changed:
                threading.stack_size(original_stack_size)
            release.set()

        for warmup in warmups:
            warmup.result()
        return executor


def _chunk_file_payload(path: str, rel: str, repo_root: str,
                        content_hash: str | None = None) -> dict | None:
    """AST-chunk one file into an upload payload (with content_hash for future diffs),
    or None if it yields no chunks (empty / too big / binary)."""
    chunks = _chunk_file(path, repo_root)
    if not chunks:
        return None
    return {"rel_path": rel, "language": _resolve_language(path) or "text",
            "content_hash": content_hash if content_hash is not None else _file_content_hash(path),
            "chunks": [{"start_line": c.start_line, "end_line": c.end_line, "label": c.label,
                        "content": c.content, "code": c.code} for c in chunks]}


# Optional progress hook for long local walks. Called from the collector thread
# (not pool workers). Payload: files_total, files_done, files_changed, chunks_ready, phase.
CollectProgressCb = Callable[[dict], None]


def _collect_progress_stride(total: int) -> int:
    """How often to emit progress during a concurrent walk ( thrash-resistant )."""
    if total <= 0:
        return 1
    return max(1, min(25, total // 20 or 1))


def _emit_collect_progress(
    progress_cb: CollectProgressCb | None,
    *,
    files_total: int,
    files_done: int,
    files_changed: int = 0,
    chunks_ready: int = 0,
    phase: str = "scanning and chunking local files",
    force: bool = False,
    stride: int = 1,
) -> None:
    if progress_cb is None:
        return
    if not force and files_done not in {0, files_total} and files_done % stride != 0:
        return
    progress_cb({
        "phase": phase,
        "files_total": files_total,
        "files_done": files_done,
        "files_changed": files_changed,
        "chunks_ready": chunks_ready,
    })


def _collect_code_files(
    repo_root: str,
    respect_gitignore: bool = True,
    exclude_globs: list[str] | None = None,
    progress_cb: CollectProgressCb | None = None,
    files: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Full walk: AST-chunk every supported file into per-file upload payloads.

    Files are chunked CONCURRENTLY on a bounded thread pool; results are stored by
    submission index so output stays byte-identical to a serial walk. ``progress_cb``
    (if set) receives file-level ticks without changing the returned payloads.
    ``files`` is the already-walked (abs_path, rel_path) code list when the caller
    shares one walk with the document pass; None walks here.
    """
    if files is None:
        files = list(_iter_repo_files(
            repo_root, respect_gitignore=respect_gitignore, exclude_globs=exclude_globs))
    total = len(files)
    stride = _collect_progress_stride(total)
    _emit_collect_progress(
        progress_cb, files_total=total, files_done=0, force=True, stride=stride,
        phase="scanning and chunking local files",
    )
    if not files:
        return []
    results: list[dict | None] = [None] * total
    with _chunk_executor(min(total, _CHUNK_WORKERS)) as ex:
        futs = {
            ex.submit(_chunk_file_payload, abs_path, rel, repo_root): i
            for i, (abs_path, rel) in enumerate(files)
        }
        done = 0
        changed = 0
        chunks_ready = 0
        for fut in as_completed(futs):
            i = futs[fut]
            payload = fut.result()
            results[i] = payload
            done += 1
            if payload is not None:
                changed += 1
                chunks_ready += len(payload.get("chunks") or [])
            _emit_collect_progress(
                progress_cb,
                files_total=total,
                files_done=done,
                files_changed=changed,
                chunks_ready=chunks_ready,
                stride=stride,
                force=(done == total),
            )
    return [f for f in results if f is not None]


def _collect_incremental(
    repo_root: str,
    known_hashes: dict,
    respect_gitignore: bool = True,
    exclude_globs: list[str] | None = None,
    progress_cb: CollectProgressCb | None = None,
    files: list[tuple[str, str]] | None = None,
) -> tuple[list[dict], list[str], dict]:
    """Hash every candidate; AST-chunk ONLY files whose hash differs from known_hashes.
    ``files`` is the already-walked code list when the caller shares one walk with the
    document pass (None walks here).
    Returns (changed_payloads, manifest, current_hashes).

    manifest = paths that should hold vectors after this op = unchanged-already-indexed ∪
    changed-with-chunks; an emptied changed file (0 chunks) is in neither, so the server
    reconciles its stale chunks away — keeping incremental byte-identical to a full
    reindex of the current tree (the correctness invariant).

    Hashing + (for changed files) chunking run CONCURRENTLY on a bounded thread pool;
    results are assembled in walk order so payloads/manifest match the serial version.
    ``progress_cb`` reports files_done/files_total while the pool runs (observability only).
    """
    if files is None:
        files = list(_iter_repo_files(
            repo_root, respect_gitignore=respect_gitignore, exclude_globs=exclude_globs))
    total = len(files)
    stride = _collect_progress_stride(total)
    _emit_collect_progress(
        progress_cb, files_total=total, files_done=0, force=True, stride=stride,
        phase="scanning and chunking local files",
    )

    def _one(abs_path: str, rel: str) -> tuple[str, str | None, dict | None]:
        h = _file_content_hash(abs_path)
        if h is None:
            return rel, None, None
        if h == known_hashes.get(rel):
            return rel, h, None  # unchanged — skip the expensive AST chunk
        return rel, h, _chunk_file_payload(abs_path, rel, repo_root, content_hash=h)

    if files:
        results: list[tuple[str, str | None, dict | None] | None] = [None] * total
        with _chunk_executor(min(total, _CHUNK_WORKERS)) as ex:
            futs = {
                ex.submit(_one, abs_path, rel): i
                for i, (abs_path, rel) in enumerate(files)
            }
            done = 0
            changed = 0
            chunks_ready = 0
            for fut in as_completed(futs):
                i = futs[fut]
                item = fut.result()
                results[i] = item
                done += 1
                _rel, h, payload = item
                if payload is not None:
                    changed += 1
                    chunks_ready += len(payload.get("chunks") or [])
                _emit_collect_progress(
                    progress_cb,
                    files_total=total,
                    files_done=done,
                    files_changed=changed,
                    chunks_ready=chunks_ready,
                    stride=stride,
                    force=(done == total),
                )
        ordered = [r for r in results if r is not None]
    else:
        ordered = []

    payloads: list[dict] = []
    current: dict[str, str] = {}
    changed_with_chunks: set[str] = set()
    unreadable_keep: set[str] = set()  # existing files we couldn't read THIS run — never purge
    for rel, h, payload in ordered:
        if h is None:
            # A transient lock (AV / IDE autosave / git checkout) must NOT look like a
            # deletion: keep an already-indexed file in the manifest so it isn't reconciled
            # away. A never-indexed unreadable file is simply skipped.
            if rel in known_hashes:
                unreadable_keep.add(rel)
            continue
        current[rel] = h
        if h == known_hashes.get(rel):
            continue  # unchanged
        if payload is not None:
            payloads.append(payload)
            changed_with_chunks.add(rel)
    unchanged = {rel for rel, h in current.items() if h == known_hashes.get(rel)}
    return payloads, sorted(unchanged | changed_with_chunks | unreadable_keep), current


def _flatten_inventory_code_files(resp: dict) -> dict:
    """Flatten the inventory tree to {rel_path: file_node} for kind=code — the diff
    baseline (each node carries content_hash + chunk_count + language)."""
    out: dict[str, dict] = {}
    for u in resp.get("users") or []:
        for h in u.get("hosts") or []:
            for p in h.get("projects") or []:
                for d in p.get("dirs") or []:
                    for f in d.get("files") or []:
                        if f.get("kind", "code") == "code":
                            out[f["rel_path"]] = f
    return out


def _inventory_records(manifest: list[str], payloads: list[dict], known: dict,
                       current: dict, commit: str) -> list[dict]:
    """Full catalog snapshot for the current tree (replace=True keeps it in sync + sweeps
    removed files): changed files take fresh chunk_count/language, unchanged reuse the
    known node's; every row carries content_hash (next diff) + git commit (freshness)."""
    by_rel = {f["rel_path"]: f for f in payloads}
    meta = {"git_commit": commit} if commit else {}
    records: list[dict] = []
    for rel in manifest:
        if rel in by_rel:
            f = by_rel[rel]
            lang, chunks, ch = f["language"], len(f["chunks"]), f.get("content_hash")
        else:
            # Unchanged (or unreadable-kept) file: reuse the known node's metadata. Prefer
            # the freshly-read hash, but fall back to the stored one so an unreadable-kept
            # file's hash isn't overwritten with None (which would force a re-embed next run).
            n = known.get(rel, {})
            lang, chunks = (n.get("language") or "text"), n.get("chunk_count", 0)
            ch = current.get(rel) or n.get("content_hash")
        records.append({"rel_path": rel, "kind": "code", "collection": "code_chunks",
                        "language": lang, "chunk_count": chunks,
                        "content_hash": ch, "metadata": meta})
    return records


def _git_commit(root: str) -> str:
    """Short HEAD commit for `root`, or '' if not a git repo — best-effort provenance so
    the inventory shows which revision is indexed. Never raises."""
    try:
        # stdin=DEVNULL is REQUIRED: without it the child inherits THIS process's stdin,
        # which for an MCP stdio server is the JSON-RPC pipe from the client — git then
        # wedges on that handle for the full timeout (~5s per index_code call). (W159)
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _format_public_code_sources(sources: list[dict]) -> str:
    indexed = sum(1 for source in sources if source.get("active_revision"))
    lines = [f"Public code sources: {len(sources)} configured, {indexed} indexed."]
    for source in sources:
        revision = str(source.get("active_revision") or "none")[:10]
        policy = str(source.get("update_policy") or "manual")
        interval = source.get("update_interval_minutes")
        if policy == "scheduled" and interval:
            policy = f"scheduled/{interval}m"
        state = "error" if source.get("last_check_error") else (
            "active" if source.get("active_revision") else "configured")
        lines.append(
            f"- {source.get('source_id')} [{source.get('domain')}] {state}; "
            f"{int(source.get('active_manifest_file_count') or 0)} files, "
            f"{int(source.get('active_chunk_count') or 0)} chunks, "
            f"{int(source.get('active_vector_count') or 0)} vectors; "
            f"rev={revision}; policy={policy}; path={source.get('relative_path')}"
        )
        if source.get("last_check_error"):
            lines.append(f"  check_error: {source['last_check_error']}")
    return "\n".join(lines)


def _format_public_code_job(job: dict) -> str:
    completed = int(job.get("completed_sources") or 0)
    total = int(job.get("total_sources") or 0)
    progress = float(job.get("progress") or 0.0)
    line = (
        f"Public code job {job.get('job_id')} [{job.get('status')}] "
        f"{job.get('operation')} {completed}/{total} sources ({progress:.0%}); "
        f"phase={job.get('phase')}; files={int(job.get('files_done') or 0)}/"
        f"{int(job.get('files_total') or 0)}; chunks={int(job.get('chunks_done') or 0)}"
    )
    current = job.get("current_source_id") or job.get("current_path")
    if job.get("current_path"):
        current = f"{job.get('current_source_id') or '?'}:{job['current_path']}"
    if current:
        line += f"; current={current}"
    if job.get("message"):
        line += f"; message={job['message']}"
    errors = job.get("errors") or {}
    if errors:
        line += "; errors=" + "; ".join(
            f"{key}: {value}" for key, value in errors.items())
    return line


def _public_code_search_diagnostics(response: dict) -> list[str]:
    """Render retrieval-mode and rerank state without hiding a strict miss."""

    lines: list[str] = []
    requested_mode = response.get("requested_mode")
    effective_mode = response.get("effective_mode")
    if requested_mode is not None or effective_mode is not None:
        lines.append(
            f"Search mode: requested={requested_mode!r}, effective={effective_mode!r}."
        )

    candidate_count = response.get("candidate_count")
    rerank_candidate_count = response.get("rerank_candidate_count")
    if candidate_count is not None:
        if response.get("rerank_applied"):
            lines.append(
                "Rerank: applied to "
                f"{rerank_candidate_count or 0} of {candidate_count} retrieved candidate(s)."
            )
        elif rerank_candidate_count:
            lines.append(
                "Rerank: not applied after considering "
                f"{rerank_candidate_count} of {candidate_count} retrieved candidate(s)."
            )
        else:
            lines.append(f"Rerank: not applied; {candidate_count} retrieved candidate(s).")

    notice = response.get("notice")
    if notice:
        lines.append(f"Search diagnostic: {notice}")
    return lines


def _number_code_lines(code: str, start_line) -> str:
    """Annotate source lines without changing stored code or inventing a location."""
    if not code or type(start_line) is not int or start_line < 1:
        return code
    lines = code.replace("\r\n", "\n").rstrip("\r").split("\n")
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{start_line + offset:>{width}} | {line}"
                     for offset, line in enumerate(lines))


def _format_public_code_search(response: dict, *, show_line_numbers: bool = True) -> str:
    hits = response.get("hits") or []
    selected = response.get("selected_source_ids") or []
    header = (
        f"Public code matches for {response.get('query')!r}: {len(hits)} hit(s)"
        + (f" across {', '.join(selected)}" if selected else "")
        + "."
    )
    diagnostics = _public_code_search_diagnostics(response)
    if not hits:
        return "\n".join([header, *diagnostics])
    lines = [header]
    if diagnostics:
        lines.extend(["", *diagnostics])
    lines.append("")
    for index, hit in enumerate(hits, 1):
        revision = str(hit.get("revision") or "")[:10]
        label = f" [{hit.get('label')}]" if hit.get("label") else ""
        score_parts = [f"score={float(hit.get('score') or 0.0):.3f}"]
        if hit.get("retrieval_score") is not None:
            score_parts.append(f"retrieval_score={float(hit['retrieval_score']):.3f}")
        if hit.get("reranker_score") is not None:
            score_parts.append(f"reranker_score={float(hit['reranker_score']):.3f}")
        lines.append(
            f"{index}. {hit.get('source_id')}@{revision} "
            f"{hit.get('path')}:{hit.get('start_line')}-{hit.get('end_line')}"
            f"{label} {' '.join(score_parts)}"
        )
        lines.append(f"```{hit.get('language') or ''}")
        body = str(hit.get("code") or hit.get("content") or "").rstrip("\n")
        lines.append(_number_code_lines(body, hit.get("start_line")) if show_line_numbers else body)
        lines.append("```")
    return "\n".join(lines)


def _clip_code_hit_body(code: str, *, path: str, start_line, end_line,
                        show_line_numbers: bool = True) -> str:
    """Keep normal AST chunks intact; hard-cap pathological multi-KB one-liners."""
    raw = (code or "").rstrip("\n")
    body = _number_code_lines(raw, start_line) if show_line_numbers else raw
    if len(body) <= _CODE_HIT_CODE_CHARS:
        return body
    kept = body[:_CODE_HIT_CODE_CHARS].rstrip()
    if show_line_numbers and type(start_line) is int and start_line > 0:
        # A clipped number prefix is not a source line; retain only usable rows.
        if not re.match(r"^\s*\d+ \| ", kept.rsplit("\n", 1)[-1]):
            kept = kept.rsplit("\n", 1)[0] if "\n" in kept else ""
    cite = f"{path}:{start_line}-{end_line}"
    return (
        f"{kept}\n\n… truncated {len(body) - len(kept)} chars "
        f"(cap {_CODE_HIT_CODE_CHARS}); open {cite} for the rest."
    )


def _fmt_code_hits(hits: list[dict], header: str, *, show_line_numbers: bool = True) -> str:
    if not hits:
        return f"{header}\n(no matching code — index a repo with index_code first.)"
    lines = [header, ""]
    omitted = 0
    for i, h in enumerate(hits, 1):
        path = h.get("path") or ""
        start = h.get("start_line")
        end = h.get("end_line")
        label = f"  [{h.get('label')}]" if h.get("label") else ""
        block = [
            f"{i}. {path}:{start}-{end}{label}  score={h.get('score', 0.0):.3f}",
            # Prefer full AST chunk with real indentation; only giants are clipped.
            f"```{h.get('language') or ''}",
            _clip_code_hit_body(h.get("code") or "", path=path,
                                start_line=start, end_line=end, show_line_numbers=show_line_numbers),
            "```",
        ]
        candidate = "\n".join(lines + block)
        if len(candidate) > _CODE_HIT_TOTAL_CHARS and i > 1:
            omitted = len(hits) - i + 1
            break
        lines.extend(block)
    if omitted:
        lines.append(
            f"\n… omitted {omitted} more hit(s) "
            f"(total output cap {_CODE_HIT_TOTAL_CHARS} chars); "
            "narrow with path_glob / language / max_per_file."
        )
    return "\n".join(lines)


def _format_scope_miss(*, result_kind: str, query: str, scope: str,
                       scope_argument: str, search_tool: str, next_step: str,
                       path_glob: str | None = None,
                       requested_mode: str | None = None,
                       effective_mode: str | None = None,
                       notice: str | None = None,
                       scope_snapshot_state: str | None = None,
                       recovery_required: bool = False,
                       active_index_job_id: str | None = None,
                       path_glob_matched_files: int | None = None,
                       indexed_file_count: int | None = None,
                       scope_input: str | None = None,
                       local_source_available: bool | None = None) -> str:
    """Explain an empty scoped search without falsely asserting a missing directory."""
    if scope_snapshot_state == "recovery_required" or recovery_required:
        lines = [
            f"The {result_kind} index for scope {scope!r} requires a full recovery index "
            "after an interrupted write; this is not a no-match result.",
        ]
        if notice:
            lines.extend(["Backend diagnostic:", f"- {notice}"])
        lines.append(f"Submit a full recovery index with {next_step}, then retry {search_tool}.")
        return "\n".join(lines)
    if scope_snapshot_state == "unavailable":
        lines = [
            f"The {result_kind} index visibility for scope {scope!r} cannot be verified "
            "because Index Catalog is unavailable; this is not a no-match result.",
        ]
        if notice:
            lines.extend(["Backend diagnostic:", f"- {notice}"])
        lines.append("Check Index Catalog/backend health and retry shortly.")
        return "\n".join(lines)
    if scope_snapshot_state == "preparing":
        lines = [
            f"The {result_kind} index for scope {scope!r} is still preparing; this is "
            "not a no-match result.",
        ]
        if notice:
            lines.extend(["Backend diagnostic:", f"- {notice}"])
        if active_index_job_id:
            lines.append(
                f"Poll index_code_status(job_id={active_index_job_id!r}) and retry after "
                "the job completes."
            )
        else:
            lines.append("Poll index_code_status() and retry after the job completes.")
        return "\n".join(lines)
    lines = [
        f"No matching {result_kind} found in scope {scope!r}.",
        f"This zero-result response does not prove that {scope_argument} is wrong or absent.",
    ]
    if scope_input is not None and scope_input != scope:
        lines.append(f"Input {scope_argument}={scope_input!r} was canonicalized to {scope!r}.")
    if notice:
        lines.extend(["Backend diagnostic:", f"- {notice}"])
    elif requested_mode and effective_mode and requested_mode != effective_mode:
        lines.append(
            f"Requested mode={requested_mode!r}; backend effective_mode={effective_mode!r}."
        )
    if path_glob:
        if path_glob_matched_files is not None and indexed_file_count is not None:
            lines.append(
                f"path_glob={path_glob!r} selected {path_glob_matched_files} of "
                f"{indexed_file_count} indexed file(s)."
            )
        lines.append(
            "path_glob is scope-relative (for example 'conversion/**'), not an absolute "
            "Windows path; retry once without it to separate a path-filter miss from a "
            "content miss."
        )
    if local_source_available is False:
        lines.append(
            "The requested source directory is not available on this MCP host. This call "
            "searched stored chunks only; it cannot read or re-index another machine's "
            "filesystem."
        )
    if search_tool == "search_code":
        source_host_step = (
            "3. If an expected source file exists but this stored scope misses it, run "
            "index_code on the machine that owns that directory, then poll "
            "index_code_status(job_id)."
        )
    else:
        source_host_step = (
            "3. If an expected source document exists but this stored scope misses it, "
            "run journal_index_dir on the machine that owns that directory, then retry "
            "journal_search."
        )
    lines.extend([
        "Next steps:",
        "1. Run index_inventory() and locate the exact scope plus its host; an inventory "
        "entry proves a stored snapshot, not that this MCP host mounts the source directory.",
        f"2. Retry without filters: {search_tool}(query={query!r}, {scope_argument}={scope!r}).",
        source_host_step,
        f"4. If inventory has no equivalent scope, index it with {next_step}.",
    ])
    return "\n".join(lines)


async def _known_code_hashes(project: str) -> dict:
    """The server's inventory of this project's code files, {rel_path: node}, as the
    incremental diff baseline. Failure (offline / no auth) degrades to {} → a full pass."""
    try:
        # detail=True: the summary inventory omits the per-file content hashes we diff
        # against. lean=True asks for just the compared columns; a backend that predates
        # the flag ignores it and returns the full tree, which flattens identically.
        inv = await _get("/api/index/inventory",
                         {"project": project, "kind": "code", "detail": True, "lean": True})
        return _flatten_inventory_code_files(inv)
    except Exception as exc:  # noqa: BLE001 — offline/degraded → treat as no prior index
        logger.warning("index_code: inventory fetch failed (%s) — indexing all files", exc)
        return {}


async def _set_code_policy(
    *, project: str, host: str, root_dir: str, auto_refresh: bool,
    respect_gitignore: bool, exclude_globs: list[str],
) -> dict:
    return await _post("/api/index/policy", {
        "host": host, "project": project, "kind": "code", "root_dir": root_dir,
        "auto_refresh": auto_refresh, "respect_gitignore": respect_gitignore,
        "exclude_globs": exclude_globs,
    })


async def _get_code_policy(*, project: str, host: str) -> dict | None:
    try:
        return await _get("/api/index/policy", {
            "host": host, "project": project, "kind": "code"})
    except Exception as exc:  # missing policy means the pre-auto-refresh behavior
        logger.info("code auto-refresh policy unavailable for %s: %s", project, exc)
        return None


async def _set_doc_policy(
    *, project: str, host: str, root_dir: str, auto_refresh: bool,
    respect_gitignore: bool, exclude_globs: list[str],
) -> dict:
    """Record, under kind="doc" of the same policy endpoint, whether this scope's
    documents ride along with its code index (the watcher reads it)."""
    return await _post("/api/index/policy", {
        "host": host, "project": project, "kind": "doc", "root_dir": root_dir,
        "auto_refresh": auto_refresh, "respect_gitignore": respect_gitignore,
        "exclude_globs": exclude_globs,
    })


async def _get_doc_policy(*, project: str, host: str) -> dict | None:
    """The scope's document policy; None when none is registered (HTTP 404). Any other
    failure raises, so a caller cannot mistake an outage for "documents are off"."""
    try:
        return await _get("/api/index/policy", {
            "host": host, "project": project, "kind": "doc"})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise


async def _sync_doc_policy(
    *, project: str, host: str, root_dir: str, docs: bool, auto_refresh: bool,
    respect_gitignore: bool, exclude_globs: list[str],
) -> str:
    """Persist what this index_code call decided about the scope's documents, before any
    watcher can look. Returns a warning sentence (empty when saved).

    docs=False must also switch OFF a scope an earlier run enabled — "do not manage this
    scope's documents" has to survive a daemon restart — but only ever by an explicit
    write: a scope that never had document policy costs one read and no write."""
    try:
        if docs:
            await _set_doc_policy(
                project=project, host=host, root_dir=root_dir, auto_refresh=auto_refresh,
                respect_gitignore=respect_gitignore, exclude_globs=exclude_globs)
        else:
            existing = await _get_doc_policy(project=project, host=host)
            if existing and existing.get("auto_refresh"):
                await _set_doc_policy(
                    project=project, host=host, root_dir=root_dir, auto_refresh=False,
                    respect_gitignore=respect_gitignore, exclude_globs=exclude_globs)
    except Exception as exc:  # noqa: BLE001 — a policy outage must not stop indexing
        logger.warning("index_code: document policy persistence failed: %s", exc)
        return " Warning: document auto-refresh policy could not be saved."
    return ""


RefreshKey = tuple[str, str]
_AUTO_REFRESH_TASKS: dict[RefreshKey, asyncio.Task] = {}
_AUTO_REFRESH_RESULTS: dict[RefreshKey, dict] = {}
_AUTO_REFRESH_CHECKED_AT: dict[RefreshKey, float] = {}
_AUTO_REFRESH_INTERVALS: dict[RefreshKey, int] = {}
_AUTO_REFRESH_DISABLED_TTL_SECONDS = 60
_AUTO_REFRESH_FAILED_TTL_SECONDS = 10
_AUTO_REFRESH_ACTIVE_TTL_SECONDS = 2
_CODE_WATCH_TASKS: dict[RefreshKey, asyncio.Task] = {}
_CODE_WATCH_STATUS: dict[RefreshKey, dict] = {}
_CODE_WATCH_RETRY_AT: dict[RefreshKey, float] = {}
_CODE_WATCH_RETRY_SECONDS = 60
_CODE_WATCH_DEBOUNCE_MS = 800
# Automatic indexing (the watcher and _auto_refresh_code) runs at most once per this many
# seconds per scope: the user's rule of 2026-11-15, enforced by the backend as the floor of
# the policy's min_check_interval_s. Explicit tool calls are never throttled.
_AUTO_INDEX_MIN_INTERVAL_S = 300
_CODE_JOB_POLL_SECONDS = 0.5
_CODE_JOB_POLL_MAX_SECONDS = 5.0  # a long index run must not be queried twice a second
_CODE_JOB_RETRY_MAX_SECONDS = 5.0
_CODE_JOB_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})
_CODE_JOB_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_CODE_JOB_STATUS_RANK = {
    "preparing": -1,
    "queued": 0,
    "running": 1,
    "completed": 2,
    "failed": 2,
    "cancelled": 2,
}
_CODE_PREPARATION_HISTORY_LIMIT = 100
_CODE_PREPARATION_TASKS: dict[str, asyncio.Task] = {}
_CODE_PREPARATION_STATUS: dict[str, dict] = {}
_CODE_PREPARATION_BY_SCOPE: dict[RefreshKey, str] = {}
_CODE_JOB_WAIT_TASKS: dict[str, asyncio.Task] = {}
_CODE_SCOPE_OPERATION_LOCKS: dict[RefreshKey, asyncio.Lock] = {}
# Documents ride along with the code index through the Journal bulk-index queue; their
# jobs are polled on a different endpoint, so they get their own wait registry.
_DOC_JOB_WAIT_TASKS: dict[str, asyncio.Task] = {}
# The bulk-index job index_code last submitted for a scope: a watcher that lazily loads
# the document baseline waits for it, so it never diffs against a half-written catalog.
_DOC_PENDING_JOBS: dict[RefreshKey, str] = {}
_DOC_JOB_ACTIVE_STATUSES = frozenset({"preparing", "queued", "running"})
# A scope without document policy is re-checked at most this often, not on every batch.
_WATCH_DOCS_OFF_TTL_SECONDS = 30
_WATCH_CONTROL_FILES = frozenset({".gitignore", ".retrievalignore"})


@dataclass(frozen=True)
class _WatchSubmission:
    status: dict
    updated_nodes: dict[str, dict]
    removed_paths: list[str]
    refresh_inventory: bool = False
    # The baseline a completed refresh leaves in the catalog: exactly the records the
    # refresh job reports, so the watcher need not download the inventory again.
    refreshed_known: dict[str, dict] | None = None
    # Documents: the Journal bulk-index job still to wait for, and what the catalog
    # holds once it (or the inline deletion/reconcile) is done.
    doc_job_id: str | None = None
    updated_doc_nodes: dict[str, dict] = dataclasses.field(default_factory=dict)
    removed_doc_paths: list[str] = dataclasses.field(default_factory=list)
    refresh_docs: bool = False
    refreshed_known_docs: dict[str, dict] | None = None


@dataclass
class _WatchDocs:
    """One watcher's view of its scope's documents. Resolved lazily — by the first batch
    that touches a document or an ignore file — so a code-only session never reads the
    document catalog. ``known`` is the baseline ({rel_path: node}, restricted to the
    document sweep envelope) while the scope's documents are kept fresh, None otherwise;
    ``off_checked_at`` rate-limits the policy lookup that last said "off"."""
    known: dict[str, dict] | None = None
    off_checked_at: float = 0.0


def _cache_refresh_result(key: RefreshKey, result: dict, interval: int) -> dict:
    _AUTO_REFRESH_RESULTS[key] = result
    _AUTO_REFRESH_CHECKED_AT[key] = time.monotonic()
    _AUTO_REFRESH_INTERVALS[key] = max(0, interval)
    return result


def _invalidate_refresh(project: str, host: str | None = None) -> None:
    keys = {key for key in _AUTO_REFRESH_TASKS if key[1] == project
            and (host is None or key[0] == host)}
    keys.update(key for key in _AUTO_REFRESH_RESULTS if key[1] == project
                and (host is None or key[0] == host))
    for key in keys:
        task = _AUTO_REFRESH_TASKS.pop(key, None)
        if task is not None and not task.done():
            task.cancel()
        _AUTO_REFRESH_RESULTS.pop(key, None)
        _AUTO_REFRESH_CHECKED_AT.pop(key, None)
        _AUTO_REFRESH_INTERVALS.pop(key, None)
    watch_keys = {key for key in _CODE_WATCH_TASKS if key[1] == project
                  and (host is None or key[0] == host)}
    watch_keys.update(key for key in _CODE_WATCH_STATUS if key[1] == project
                      and (host is None or key[0] == host))
    for key in watch_keys:
        task = _CODE_WATCH_TASKS.pop(key, None)
        if task is not None and not task.done():
            task.cancel()
        _CODE_WATCH_STATUS.pop(key, None)
        _CODE_WATCH_RETRY_AT.pop(key, None)
        _DOC_PENDING_JOBS.pop(key, None)


async def _active_code_job(project: str) -> dict | None:
    jobs = await _get("/api/code/jobs", {
        "project": project, "active_only": True, "limit": 100})
    return next((job for job in jobs.get("jobs", [])
                 if job.get("project") == project
                 and job.get("status") in {"queued", "running"}), None)


def _auto_index_interval_s(policy: dict) -> int:
    """The automatic-indexing interval of a policy. The floor also applies here because a
    backend that predates it still answers min_check_interval_s=10."""
    return max(_AUTO_INDEX_MIN_INTERVAL_S,
               int(policy.get("min_check_interval_s", _AUTO_INDEX_MIN_INTERVAL_S)))


async def _auto_refresh_code(*, project: str, host: str, repo_root: str,
                             force: bool = False,
                             on_baseline: Callable[[dict[str, dict]], None] | None = None,
                             ) -> dict:
    """Diff the tree against the catalog and submit one incremental job for the delta.
    ``on_baseline`` receives the baseline the catalog holds once that job completes
    (see ``_known_from_records``) whenever a job was submitted."""
    key = (host, project)
    try:
        now = time.monotonic()
        previous = _AUTO_REFRESH_RESULTS.get(key)
        interval = _AUTO_REFRESH_INTERVALS.get(key, 0)
        if (not force and previous is not None
                and now - _AUTO_REFRESH_CHECKED_AT.get(key, 0) < interval):
            return previous

        active = await _active_code_job(project)
        if active is not None:
            result = {"state": "refresh_running", "job_id": active.get("job_id"),
                      "status": active.get("status")}
            return _cache_refresh_result(key, result, _AUTO_REFRESH_ACTIVE_TTL_SECONDS)

        policy = await _get_code_policy(project=project, host=host)
        if not policy or not policy.get("auto_refresh", False):
            return _cache_refresh_result(
                key, {"state": "disabled"}, _AUTO_REFRESH_DISABLED_TTL_SECONDS)
        root = os.path.abspath(os.path.expanduser(repo_root))
        if not os.path.isdir(root):
            return _cache_refresh_result(
                key,
                {"state": "unavailable", "message": f"repository root unavailable: {root}"},
                _AUTO_REFRESH_DISABLED_TTL_SECONDS,
            )

        interval = _auto_index_interval_s(policy)
        _AUTO_REFRESH_INTERVALS[key] = interval
        respect_gitignore = bool(policy.get("respect_gitignore", True))
        known = await _known_code_hashes(project)
        if not known:
            result = {"state": "unchecked", "message": "no completed code inventory yet"}
        else:
            known_hashes = {
                rel: node["content_hash"] for rel, node in known.items()
                if node.get("content_hash")
            }
            excludes = list(policy.get("exclude_globs") or [])
            payloads, manifest, current = await asyncio.to_thread(
                _collect_incremental, root, known_hashes,
                respect_gitignore, excludes)
            removed = sorted(set(known) - set(manifest))
            if not payloads and not removed:
                result = {"state": "current", "files": len(manifest)}
            else:
                commit = await asyncio.to_thread(_git_commit, root)
                records = _inventory_records(manifest, payloads, known, current, commit)
                request = {
                    "project": project, "host": host, "root_dir": root,
                    "files": payloads, "incremental": True, "manifest": manifest,
                    # `removed` was diffed against the catalog inventory above, so the
                    # server need not enumerate the project's paths to find it. `manifest`
                    # stays for a backend that predates removed_paths_complete.
                    "removed_paths": removed, "removed_paths_complete": True,
                    "report": {"files": records, "replace": True},
                }
                job = await _submit_code_index_with_receipt(
                    key=key, client_job_id=uuid.uuid4().hex, request=request)
                if on_baseline is not None:
                    on_baseline(_known_from_records(records))
                result = {
                    "state": "refresh_started", "job_id": job.get("job_id"),
                    "status": job.get("status", "queued"),
                    "changed": len(payloads), "removed": len(removed),
                }
        return _cache_refresh_result(key, result, interval)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # freshness must never suppress a usable search result
        logger.exception("automatic code refresh failed for %s", project)
        result = {"state": "failed", "message": str(exc)}
        return _cache_refresh_result(key, result, _AUTO_REFRESH_FAILED_TTL_SECONDS)


def _watch_rel_path(repo_root: str, path: str) -> str | None:
    root = os.path.abspath(repo_root)
    absolute = os.path.abspath(path)
    try:
        if os.path.normcase(os.path.commonpath([root, absolute])) != os.path.normcase(root):
            return None
    except ValueError:
        return None
    rel = os.path.relpath(absolute, root).replace("\\", "/")
    return "" if rel == "." else rel


def _git_ignored_watch_paths(repo_root: str, rel_paths: set[str]) -> set[str]:
    """Ask Git only about touched paths; never enumerate the repository.

    The paths go to git in sorted batches of _CHECK_IGNORE_BATCH, each under its own
    timeout. Fails closed: a batch git did not answer raises GitIgnoreCheckFailed (never
    "nothing is ignored"); only a non-repository / missing git answers set()."""
    if not rel_paths:
        return set()
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["LC_ALL"] = "C"
    ordered = sorted(rel_paths)
    ignored: set[str] = set()
    for start in range(0, len(ordered), _CHECK_IGNORE_BATCH):
        encoded = ("\0".join(ordered[start:start + _CHECK_IGNORE_BATCH]) + "\0").encode(
            "utf-8", errors="surrogateescape")
        try:
            result = subprocess.run(
                ["git", "-C", repo_root, "check-ignore", "--stdin", "-z"],
                input=encoded,
                capture_output=True,
                timeout=_GIT_CHECK_IGNORE_TIMEOUT_S,
                env=env,
            )
        except FileNotFoundError:
            return set()  # git is not installed: no filtering, as _gitignore_allowed
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitIgnoreCheckFailed(
                f"git check-ignore did not finish in {repo_root}: {exc!r}") from exc
        if result.returncode not in {0, 1}:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            if result.returncode == 128 and "not a git repository" in stderr:
                return set()
            raise GitIgnoreCheckFailed(
                f"git check-ignore failed in {repo_root} (exit {result.returncode}): "
                f"{stderr[:200]}")
        ignored.update(
            path.decode("utf-8", errors="surrogateescape")
            for path in result.stdout.split(b"\0") if path)
    return ignored


class _WatchedChanges(NamedTuple):
    """One resolved filesystem event batch. The first four fields are the code result;
    ``doc_changed`` / ``doc_removed`` are the documents (empty unless the scope's
    documents are kept fresh)."""
    payloads: list[dict]
    removed: list[str]
    current: dict[str, str]
    requires_reconcile: bool
    doc_changed: list[_JournalFileCandidate]
    doc_removed: list[str]


def _collect_watched_changes(
    repo_root: str,
    changed_paths: set[str],
    known: dict[str, dict],
    known_docs: dict[str, dict] | None,
    respect_gitignore: bool,
    exclude_globs: list[str],
) -> _WatchedChanges:
    """Resolve one filesystem event batch without walking unrelated directories.

    ``known_docs`` is the scope's document baseline ({rel_path: node}); None means the
    scope's documents are not kept fresh, and then no document is even looked at — the
    code result is exactly the code-only one."""
    root = os.path.abspath(repo_root)
    excludes = _effective_excludes(root, exclude_globs)
    candidates: dict[str, str] = {}
    doc_candidates: dict[str, str] = {}
    removed: set[str] = set()
    doc_removed: set[str] = set()
    requires_reconcile = False

    def add_candidate(path: str) -> None:
        rel = _watch_rel_path(root, path)
        if rel is None:
            return
        if rel in {".gitignore", ".retrievalignore"}:
            nonlocal requires_reconcile
            requires_reconcile = True
            return
        parts = rel.split("/")
        if any(part in _SKIP_DIRS for part in parts[:-1]):
            return
        if not (_resolve_language(path) or _ext_of(path) in FALLBACK_EXTS):
            if rel in known:
                removed.add(rel)
            if known_docs is not None:
                if _is_doc_path(path, rel) and not _excluded(rel, excludes):
                    doc_candidates[rel] = path
                elif rel in known_docs:
                    doc_removed.add(rel)
            return
        if _excluded(rel, excludes):
            if rel in known:
                removed.add(rel)
            return
        candidates[rel] = path

    try:
        for changed in changed_paths:
            absolute = os.path.abspath(changed)
            rel = _watch_rel_path(root, absolute)
            if rel is None:
                continue
            if os.path.basename(rel) == ".gitignore" or rel == ".retrievalignore":
                requires_reconcile = True
                continue
            if os.path.isdir(absolute):
                if not rel or any(part in _SKIP_DIRS for part in rel.split("/")):
                    continue
                # Git enumerates the directory (ignored files never become candidates);
                # the os.walk below is only for scopes git cannot answer for.
                listed = _gitignore_allowed(root, [rel]) if respect_gitignore else None
                if listed is not None:
                    for listed_rel in sorted(listed):
                        listed_parts = listed_rel.split("/")
                        if any(part in _SKIP_DIRS for part in listed_parts[:-1]):
                            continue
                        listed_path = os.path.join(root, *listed_parts)
                        if os.path.isfile(listed_path):  # tracked-but-deleted is not one
                            add_candidate(listed_path)
                    continue
                for walk_root, dirs, files in os.walk(absolute):
                    dirs[:] = [name for name in dirs if name not in _SKIP_DIRS]
                    for filename in files:
                        add_candidate(os.path.join(walk_root, filename))
            elif os.path.isfile(absolute):
                add_candidate(absolute)
            else:
                def under_deleted(path: str) -> bool:
                    return path == rel or bool(rel and path.startswith(rel.rstrip("/") + "/"))

                removed.update(path for path in known if under_deleted(path))
                if known_docs is not None:
                    doc_removed.update(path for path in known_docs if under_deleted(path))

        if respect_gitignore and (candidates or doc_candidates):
            ignored = _git_ignored_watch_paths(root, set(candidates) | set(doc_candidates))
            for rel in ignored:
                candidates.pop(rel, None)
                doc_candidates.pop(rel, None)
                if rel in known:
                    removed.add(rel)
                if known_docs is not None and rel in known_docs:
                    doc_removed.add(rel)
    except GitIgnoreCheckFailed as exc:
        # Fail closed: nothing may be chunked on an ignore answer git did not give. The
        # whole batch becomes a reconcile (a full, explicit ignore-aware pass that reports
        # the same failure); deletions already collected need no ignore answer and stay.
        logger.warning(
            "gitignore check failed for %s: %s; dropped %d candidate(s), reconcile required",
            root, exc, len(candidates) + len(doc_candidates))
        candidates.clear()
        doc_candidates.clear()
        requires_reconcile = True

    def inspect_file(item: tuple[str, str]) -> tuple[str, str | None, dict | None]:
        rel, path = item
        content_hash = _file_content_hash(path)
        if content_hash is None or content_hash == known.get(rel, {}).get("content_hash"):
            return rel, content_hash, None
        return rel, content_hash, _chunk_file_payload(
            path, rel, root, content_hash=content_hash)

    items = list(candidates.items())
    if items:
        with _chunk_executor(min(len(items), _CHUNK_WORKERS)) as executor:
            inspected = list(executor.map(inspect_file, items))
    else:
        inspected = []

    payloads: list[dict] = []
    current: dict[str, str] = {}
    for rel, content_hash, payload in inspected:
        if content_hash is None:
            continue
        current[rel] = content_hash
        if content_hash == known.get(rel, {}).get("content_hash"):
            continue
        if payload is None:
            if rel in known:
                removed.add(rel)
            continue
        payloads.append(payload)
        removed.discard(rel)

    doc_changed: list[_JournalFileCandidate] = []
    for rel, doc_path in sorted(doc_candidates.items()):
        candidate = _journal_file_candidate(doc_path, root)
        doc_removed.discard(rel)
        # A file that cannot be read right now (editor lock, antivirus) is not a deletion
        # and not a change: leave the catalog as it is until the next event.
        if not candidate.readable:
            continue
        if candidate.content_hash == (known_docs.get(rel, {}).get("content_hash") or None):
            continue
        doc_changed.append(candidate)
    return _WatchedChanges(
        payloads, sorted(removed), current, requires_reconcile, doc_changed, sorted(doc_removed))


def _collect_watched_code_changes(
    repo_root: str,
    changed_paths: set[str],
    known: dict[str, dict],
    respect_gitignore: bool,
    exclude_globs: list[str],
) -> tuple[list[dict], list[str], dict[str, str], bool]:
    """The code half of _collect_watched_changes (documents not looked at)."""
    changes = _collect_watched_changes(
        repo_root, changed_paths, known, None, respect_gitignore, exclude_globs)
    return changes.payloads, changes.removed, changes.current, changes.requires_reconcile


def _may_touch_documents(changed_paths: set[str]) -> bool:
    """Cheap, filesystem-light test: could this event batch concern the scope's documents?
    A document, an ignore file (it re-reconciles documents too), or a directory — which
    may hold documents — and, for a path that no longer exists, an extensionless name
    (a deleted directory)."""
    for path in changed_paths:
        name = os.path.basename(path)
        if name in _WATCH_CONTROL_FILES or _ext_of(path) in DOC_EXTS or os.path.isdir(path):
            return True
        if not os.path.exists(path) and not os.path.splitext(name)[1]:
            return True
    return False


async def _known_doc_inventory(project: str, host: str) -> dict[str, dict]:
    """The scope's document baseline from the doc inventory, restricted to the sweep
    envelope index_code manages. Raises on failure: unlike the code baseline, a wrong
    (empty) answer here would let a deletion reconcile remove live documents."""
    inventory = await _get("/api/index/inventory", {
        "host": host, "project": project, "kind": "doc", "detail": True, "lean": True,
    })
    return _journal_known_in_envelope(_flatten_inventory_doc_files(inventory), DOC_GLOBS, None)


def _known_doc_nodes(records: list[dict]) -> dict[str, dict]:
    return {
        record["rel_path"]: {
            "rel_path": record["rel_path"],
            "kind": "doc",
            "content_hash": record.get("content_hash"),
            "chunk_count": record.get("chunk_count", 0),
            "size_bytes": record.get("size_bytes"),
        }
        for record in records
    }


async def _resolve_watch_docs(docs: _WatchDocs, *, project: str, host: str) -> dict | None:
    """The document baseline when this scope's documents are kept fresh, else None.

    Kept fresh = the scope has a kind="doc" policy with auto_refresh (index_code writes it
    next to the code policy). A transient lookup failure returns None WITHOUT being
    remembered, so the next batch retries instead of treating an outage as "off"."""
    if docs.known is not None:
        return docs.known
    if docs.off_checked_at and time.monotonic() - docs.off_checked_at < _WATCH_DOCS_OFF_TTL_SECONDS:
        return None
    try:
        policy = await _get_doc_policy(project=project, host=host)
        if not policy or not policy.get("auto_refresh", False):
            docs.off_checked_at = time.monotonic()
            return None
        pending = _DOC_PENDING_JOBS.get((host, project))
        if pending:
            with suppress(Exception):
                await _wait_doc_job_shared(pending)
        docs.known = await _known_doc_inventory(project, host)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("document refresh baseline unavailable for %s: %s", project, exc)
        return None
    return docs.known


async def _refresh_scope_docs(
    *, project: str, host: str, repo_root: str, respect_gitignore: bool,
    exclude_globs: list[str],
) -> _DocIndexOutcome:
    """A full document pass over the scope (walk + diff + snapshot job) — what an ignore
    file change needs, since it can add or remove any number of documents."""
    root = os.path.abspath(os.path.expanduser(repo_root))
    walked = await asyncio.to_thread(lambda: list(_walk_repo_files(
        root, respect_gitignore, exclude_globs, docs=True)))
    doc_files = [path for path, _rel, kind in walked if kind == "doc"]
    if not doc_files:
        return _DocIndexOutcome(message="no documents found")
    return await _index_doc_files(root, project, host, doc_files, DOC_GLOBS, None)


async def _remove_watched_docs(
    *, project: str, host: str, root: str, known_docs: dict[str, dict],
    removed: list[str], keep: list[str],
) -> bool:
    """Drop deleted documents from the catalog and the journal. The sweep is bounded by the
    document envelope and its live set is the known set minus the removed ones (plus the
    documents this very batch submits). Returns False when it was refused: the backend
    skips an empty snapshot, and the catalog must not change without the journal."""
    gone = set(removed)
    remaining = {rel: node for rel, node in known_docs.items() if rel not in gone}
    if not remaining:
        logger.info("document refresh: %s would lose its last document; use index_code",
                    project)
        return False
    rows = [
        {"rel_path": rel, "kind": "doc", "collection": "journal_chunks",
         "chunk_count": int(node.get("chunk_count", 0) or 0),
         "content_hash": node.get("content_hash"), "size_bytes": node.get("size_bytes")}
        for rel, node in sorted(remaining.items())
    ]
    await _post("/api/index/report", {
        "host": host, "project": project, "root_dir": root, "files": rows,
        "replace": True, "patterns": DOC_GLOBS, "max_depth": None,
    })
    await _post("/api/journal/docs/reconcile", {
        "project_slug": project,
        "live_titles": sorted(set(remaining) | set(keep)),
        "max_depth": None,
        "patterns": DOC_GLOBS,
    })
    return True


async def _submit_watched_code_paths(
    *, project: str, host: str, repo_root: str, changed_paths: set[str],
    known: dict[str, dict], respect_gitignore: bool, exclude_globs: list[str],
    docs: _WatchDocs | None = None,
) -> _WatchSubmission:
    doc_known = docs.known if docs is not None else None
    if docs is not None and doc_known is None and _may_touch_documents(changed_paths):
        doc_known = await _resolve_watch_docs(docs, project=project, host=host)
    changes = await asyncio.to_thread(
        _collect_watched_changes,
        repo_root,
        changed_paths,
        known,
        doc_known,
        respect_gitignore,
        exclude_globs,
    )
    payloads, removed, current = changes.payloads, changes.removed, changes.current
    if changes.requires_reconcile:
        refreshed: list[dict[str, dict]] = []
        status = await _auto_refresh_code(
            project=project, host=host, repo_root=repo_root, force=True,
            on_baseline=refreshed.append)
        doc_outcome = None
        if doc_known is not None:
            doc_outcome = await _refresh_scope_docs(
                project=project, host=host, repo_root=repo_root,
                respect_gitignore=respect_gitignore, exclude_globs=exclude_globs)
        return _WatchSubmission(
            status=status,
            updated_nodes={},
            removed_paths=[],
            refresh_inventory=True,
            refreshed_known=refreshed[-1] if refreshed else None,
            doc_job_id=doc_outcome.job_id if doc_outcome else None,
            refresh_docs=doc_outcome is not None,
            refreshed_known_docs=(
                _known_doc_nodes(doc_outcome.records) if doc_outcome is not None
                and doc_outcome.inventory_ok else None),
        )
    if not payloads and not removed and not changes.doc_changed and not changes.doc_removed:
        return _WatchSubmission(
            status={"state": "current", "files": len(known)},
            updated_nodes={},
            removed_paths=[],
        )

    status: dict = {"state": "current", "files": len(known)}
    updated_nodes: dict[str, dict] = {}
    if payloads or removed:
        records = _inventory_records(
            [payload["rel_path"] for payload in payloads],
            payloads,
            known,
            current,
            "",
        )
        request = {
            "project": project,
            "host": host,
            "root_dir": repo_root,
            "files": payloads,
            "incremental": True,
            "removed_paths": removed,
            "report": {"files": records, "replace": False},
        }
        job = await _submit_code_index_with_receipt(
            key=(host, project), client_job_id=uuid.uuid4().hex, request=request)
        updated_nodes = {
            payload["rel_path"]: {
                "rel_path": payload["rel_path"],
                "content_hash": payload.get("content_hash"),
                "chunk_count": len(payload.get("chunks") or []),
                "language": payload.get("language"),
            }
            for payload in payloads
        }
        status = {
            "state": "refresh_started",
            "job_id": job.get("job_id"),
            "status": job.get("status", "queued"),
            "changed": len(payloads),
            "removed": len(removed),
        }

    doc_job_id: str | None = None
    updated_doc_nodes: dict[str, dict] = {}
    removed_doc_paths: list[str] = []
    if doc_known is not None and (changes.doc_removed or changes.doc_changed):
        root = os.path.abspath(repo_root)
        if changes.doc_removed and await _remove_watched_docs(
                project=project, host=host, root=root, known_docs=doc_known,
                removed=changes.doc_removed,
                keep=[candidate.rel_path for candidate in changes.doc_changed]):
            removed_doc_paths = list(changes.doc_removed)
        if changes.doc_changed:
            outcome = await _index_doc_files(
                root, project, host, [candidate.abs_path for candidate in changes.doc_changed],
                DOC_GLOBS, None, known=doc_known, snapshot=False)
            doc_job_id = outcome.job_id
            updated_doc_nodes = _known_doc_nodes(outcome.records)
        if status.get("state") == "current":
            status = {"state": "refresh_started", "doc_job_id": doc_job_id,
                      "status": "queued" if doc_job_id else "completed"}
        status["doc_changed"] = len(changes.doc_changed)
        status["doc_removed"] = len(removed_doc_paths)
    return _WatchSubmission(
        status=status,
        updated_nodes=updated_nodes,
        removed_paths=removed,
        doc_job_id=doc_job_id,
        updated_doc_nodes=updated_doc_nodes,
        removed_doc_paths=removed_doc_paths,
    )


def _watch_changes(repo_root: str):
    return awatch(
        repo_root,
        debounce=_CODE_WATCH_DEBOUNCE_MS,
        recursive=True,
        ignore_permission_denied=True,
    )


async def _wait_code_job(job_id: str) -> dict:
    return await _wait_backend_job(lambda: _get(f"/api/code/jobs/{job_id}"))


async def _wait_doc_job(job_id: str) -> dict:
    return await _wait_backend_job(
        lambda: _get(f"/api/journal/bulk-index/jobs/{quote(job_id, safe='')}"))


async def _wait_backend_job(fetch: Callable[[], Awaitable[dict]]) -> dict:
    """Poll one backend job receipt (``fetch`` reads it) until it is terminal."""
    retry_delay = _CODE_JOB_POLL_SECONDS
    poll_delay = _CODE_JOB_POLL_SECONDS
    while True:
        try:
            job = await fetch()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _CODE_JOB_RETRYABLE_STATUS_CODES:
                raise
            await asyncio.sleep(retry_delay)
            retry_delay = min(max(retry_delay * 2, _CODE_JOB_POLL_SECONDS),
                              _CODE_JOB_RETRY_MAX_SECONDS)
            continue
        except httpx.RequestError:
            await asyncio.sleep(retry_delay)
            retry_delay = min(max(retry_delay * 2, _CODE_JOB_POLL_SECONDS),
                              _CODE_JOB_RETRY_MAX_SECONDS)
            continue
        retry_delay = _CODE_JOB_POLL_SECONDS
        # cancelled (F2-a) is terminal too — this loop must not spin forever waiting
        # for a "completed" that a cancelled job will never reach.
        if job.get("status") in _CODE_JOB_TERMINAL_STATUSES:
            return job
        await asyncio.sleep(poll_delay)
        poll_delay = min(max(poll_delay * 2, _CODE_JOB_POLL_SECONDS), _CODE_JOB_POLL_MAX_SECONDS)


async def _wait_code_job_shared(job_id: str) -> dict:
    """Let the preparation tracker and watcher share one backend polling loop."""
    task = _CODE_JOB_WAIT_TASKS.get(job_id)
    if task is None or task.done():
        task = asyncio.create_task(_wait_code_job(job_id))
        _CODE_JOB_WAIT_TASKS[job_id] = task

        def cleanup(done: asyncio.Task) -> None:
            if _CODE_JOB_WAIT_TASKS.get(job_id) is done:
                _CODE_JOB_WAIT_TASKS.pop(job_id, None)

        task.add_done_callback(cleanup)
    return await asyncio.shield(task)


async def _wait_doc_job_shared(job_id: str) -> dict:
    """One polling loop per Journal bulk-index job, shared by the preparation tracker, the
    status tool's waiters and the watcher."""
    task = _DOC_JOB_WAIT_TASKS.get(job_id)
    if task is None or task.done():
        task = asyncio.create_task(_wait_doc_job(job_id))
        _DOC_JOB_WAIT_TASKS[job_id] = task

        def cleanup(done: asyncio.Task) -> None:
            if _DOC_JOB_WAIT_TASKS.get(job_id) is done:
                _DOC_JOB_WAIT_TASKS.pop(job_id, None)

        task.add_done_callback(cleanup)
    return await asyncio.shield(task)


def _apply_watch_submission(known: dict[str, dict], submission: _WatchSubmission) -> None:
    for rel in submission.removed_paths:
        known.pop(rel, None)
    known.update(submission.updated_nodes)


async def _settle_watch_docs(
    docs: _WatchDocs, submission: _WatchSubmission, *, project: str, host: str,
) -> str | None:
    """Wait for the submission's document job, then fold what the catalog now holds into
    the watcher's baseline. Returns an error sentence when the job did not complete (the
    baseline is then left alone, so the next event diffs against what is really there)."""
    if submission.doc_job_id:
        job = await _wait_doc_job_shared(submission.doc_job_id)
        if job.get("status") != "completed":
            return job.get("error") or (
                f"document job {submission.doc_job_id} {job.get('status', 'failed')}")
    if submission.refresh_docs:
        docs.known = (dict(submission.refreshed_known_docs)
                      if submission.refreshed_known_docs is not None
                      else await _known_doc_inventory(project, host))
    elif docs.known is not None:
        for rel in submission.removed_doc_paths:
            docs.known.pop(rel, None)
        docs.known.update(submission.updated_doc_nodes)
    return None


def _known_from_records(records: list[dict]) -> dict[str, dict]:
    return {
        record["rel_path"]: {
            "rel_path": record["rel_path"],
            "content_hash": record.get("content_hash"),
            "chunk_count": record.get("chunk_count", 0),
            "language": record.get("language"),
        }
        for record in records
    }


async def _read_watch_events(repo_root: str, pending_paths: set[str],
                             paths_ready: asyncio.Event) -> None:
    async for changes in _watch_changes(repo_root):
        pending_paths.update(path for _change, path in changes)
        if pending_paths:
            paths_ready.set()


async def _watch_code_scope(
    *, project: str, host: str, repo_root: str,
    bootstrap_job_id: str | None = None,
    initial_known: dict[str, dict] | None = None,
) -> None:
    key = (host, project)
    event_reader: asyncio.Task | None = None
    try:
        policy = await _get_code_policy(project=project, host=host)
        if not policy or not policy.get("auto_refresh", False):
            _CODE_WATCH_STATUS[key] = {"state": "disabled"}
            _CODE_WATCH_RETRY_AT[key] = time.monotonic() + _CODE_WATCH_RETRY_SECONDS
            return
        root = os.path.abspath(os.path.expanduser(repo_root))
        if not os.path.isdir(root) or _scope_path(root) != project:
            _CODE_WATCH_STATUS[key] = {
                "state": "unavailable",
                "message": f"repository root unavailable or mismatched: {root}",
            }
            _CODE_WATCH_RETRY_AT[key] = time.monotonic() + _CODE_WATCH_RETRY_SECONDS
            return

        respect_gitignore = bool(policy.get("respect_gitignore", True))
        excludes = list(policy.get("exclude_globs") or [])
        interval = _auto_index_interval_s(policy)
        last_submit: float | None = None
        # Why the last attempt submitted nothing (git could not answer, reconcile failed);
        # the countdown status keeps showing it until an attempt succeeds.
        last_error: str | None = None
        pending_paths: set[str] = set()
        paths_ready = asyncio.Event()
        event_reader = asyncio.create_task(
            _read_watch_events(root, pending_paths, paths_ready))

        if bootstrap_job_id:
            _CODE_WATCH_STATUS[key] = {
                "state": "refresh_running",
                "job_id": bootstrap_job_id,
                "status": "queued",
            }
            bootstrap = await _wait_code_job_shared(bootstrap_job_id)
            if bootstrap.get("status") == "completed" and initial_known is not None:
                known = dict(initial_known)
            else:
                known = await _known_code_hashes(project)
        elif initial_known is not None:
            known = dict(initial_known)
        else:
            known = await _known_code_hashes(project)

        _CODE_WATCH_STATUS[key] = {"state": "watching", "files": len(known)}
        docs = _WatchDocs()

        while True:
            await paths_ready.wait()
            # At most one automatic submission per interval. The first change after a
            # quiet period goes out at once (after awatch's debounce); later ones wait
            # here while the event reader keeps collecting paths, and go out as one
            # batch. The re-queue paths below (failed job, deferred control-file
            # reconcile) come back through this same wait.
            while last_submit is not None:
                remaining = interval - (time.monotonic() - last_submit)
                if remaining <= 0:
                    break
                _CODE_WATCH_STATUS[key] = {
                    "state": "watching", "files": len(known),
                    "pending_changes": len(pending_paths),
                    "next_refresh_in_s": math.ceil(remaining),
                    **({"last_error": last_error} if last_error else {}),
                }
                # Sleep in slices so the status keeps counting down and the pending count
                # keeps following the event reader.
                await asyncio.sleep(min(remaining, 1.0))
            changed_paths = set(pending_paths)
            pending_paths.clear()
            paths_ready.clear()
            last_submit = time.monotonic()
            try:
                submission = await _submit_watched_code_paths(
                    project=project,
                    host=host,
                    repo_root=root,
                    changed_paths=changed_paths,
                    known=known,
                    respect_gitignore=respect_gitignore,
                    exclude_globs=excludes,
                    docs=docs,
                )
            except GitIgnoreCheckFailed as exc:
                # git could not say what is ignored (e.g. the document reconcile walk):
                # nothing was submitted. Keep the event and retry after the interval.
                logger.warning("gitignore check failed for %s: %s", project, exc)
                _CODE_WATCH_STATUS[key] = {"state": "failed", "message": str(exc)}
                last_error = str(exc)
                pending_paths.update(changed_paths)
                paths_ready.set()
                continue
            _CODE_WATCH_STATUS[key] = submission.status
            state = submission.status.get("state")
            job_id = submission.status.get("job_id")
            if state == "failed" and submission.refresh_inventory:
                # The reconcile itself failed (_auto_refresh_code reports instead of
                # raising, e.g. GitIgnoreCheckFailed in its walk): nothing was submitted
                # and the event must not be dropped. Retry after the interval.
                last_error = str(submission.status.get("message") or "reconcile failed")
                pending_paths.update(changed_paths)
                paths_ready.set()
                continue
            last_error = None
            if state in {"refresh_started", "refresh_running"} and job_id:
                job = await _wait_code_job_shared(str(job_id))
                if job.get("status") != "completed":
                    _CODE_WATCH_STATUS[key] = {
                        "state": "failed",
                        "message": job.get("error") or f"index job {job_id} failed",
                    }
                    if state == "refresh_running" and submission.refresh_inventory:
                        # The control-file event still needs a real reconcile after the
                        # unrelated active job failed; keep it dirty instead of dropping it.
                        pending_paths.update(changed_paths)
                        paths_ready.set()
                    continue
                if state == "refresh_running" and submission.refresh_inventory:
                    # The ignore-file reconcile was deferred behind somebody else's
                    # active job. Re-run the same control event now that it completed.
                    pending_paths.update(changed_paths)
                    paths_ready.set()
                    continue
                if submission.refresh_inventory:
                    known = (dict(submission.refreshed_known)
                             if submission.refreshed_known is not None
                             else await _known_code_hashes(project))
                else:
                    _apply_watch_submission(known, submission)
            doc_error = await _settle_watch_docs(
                docs, submission, project=project, host=host)
            if doc_error:
                _CODE_WATCH_STATUS[key] = {"state": "failed", "message": doc_error}
                continue
            _CODE_WATCH_STATUS[key] = {"state": "watching", "files": len(known)}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("code filesystem watcher failed for %s", project)
        _CODE_WATCH_STATUS[key] = {"state": "failed", "message": str(exc)}
        _CODE_WATCH_RETRY_AT[key] = time.monotonic() + _CODE_WATCH_RETRY_SECONDS
    finally:
        if event_reader is not None and not event_reader.done():
            event_reader.cancel()
            with suppress(asyncio.CancelledError):
                await event_reader


def _ensure_code_watcher(
    *, project: str, host: str, repo_root: str,
    bootstrap_job_id: str | None = None,
    initial_known: dict[str, dict] | None = None,
    allow_during_preparation: bool = False,
) -> asyncio.Task | None:
    key = (host, project)
    preparation_id = _CODE_PREPARATION_BY_SCOPE.get(key)
    if preparation_id and not allow_during_preparation:
        preparation = _CODE_PREPARATION_STATUS.get(preparation_id, {})
        if preparation.get("status") not in {"completed", "failed", "cancelled"}:
            return None
    task = _CODE_WATCH_TASKS.get(key)
    if task is not None and not task.done():
        return task
    if time.monotonic() < _CODE_WATCH_RETRY_AT.get(key, 0):
        return None
    task = asyncio.create_task(
        _watch_code_scope(
            project=project,
            host=host,
            repo_root=repo_root,
            bootstrap_job_id=bootstrap_job_id,
            initial_known=initial_known,
        ))
    _CODE_WATCH_TASKS[key] = task

    def cleanup(done: asyncio.Task) -> None:
        if _CODE_WATCH_TASKS.get(key) is done:
            _CODE_WATCH_TASKS.pop(key, None)

    task.add_done_callback(cleanup)
    return task


def _format_refresh_status(status: dict | None) -> str:
    if status is None:
        return "Index freshness: checking in background."
    state = status.get("state")
    if state == "watching":
        waiting = ""
        if "next_refresh_in_s" in status:
            waiting = (f"; {status.get('pending_changes', 0)} pending change(s), "
                       f"next refresh in {status['next_refresh_in_s']}s")
        return f"Index freshness: filesystem watcher active ({status.get('files', 0)} files){waiting}."
    if state == "current":
        if status.get("message"):
            return f"Index freshness: current ({status.get('message')})."
        return f"Index freshness: current ({status.get('files', 0)} files checked)."
    if state == "refresh_started":
        documents = ""
        if status.get("doc_changed") or status.get("doc_removed"):
            documents = (f" Documents: {status.get('doc_changed', 0)} changed/new, "
                         f"{status.get('doc_removed', 0)} removed"
                         + (f" (journal job {status['doc_job_id']})." if status.get("doc_job_id")
                            else "."))
        if not status.get("job_id") and documents:
            return ("Index freshness: automatic document refresh."
                    f"{documents} Results below use the previous completed snapshot.")
        return ("Index freshness: automatic incremental refresh "
                f"job {status.get('job_id', '?')} [{status.get('status', 'queued')}]; "
                f"{status.get('changed', 0)} changed/new, {status.get('removed', 0)} removed."
                f"{documents} "
                "Results below use the previous completed snapshot.")
    if state == "refresh_running":
        return (f"Index freshness: refresh job {status.get('job_id', '?')} "
                f"[{status.get('status', 'running')}]; results below use the previous "
                "completed snapshot.")
    if state in {"failed", "unavailable", "unchecked"}:
        return f"Index freshness: {state} ({status.get('message', 'unknown reason')})."
    return ""


def _preparation_finished(status: dict) -> bool:
    """The whole index_code job is over: the code job reached a terminal state and no
    document job is still preparing, queued or running."""
    return (status.get("status") in {"completed", "failed", "cancelled"}
            and status.get("doc_status") not in _DOC_JOB_ACTIVE_STATUSES)


def _overall_job_status(job: dict) -> str | None:
    """The one status shown for an index_code job: failed as soon as either the code or
    the document job failed, still running while a document job outlives a finished code
    job, otherwise the code job's own status."""
    status = job.get("status")
    doc_status = job.get("doc_status")
    if doc_status is None:
        return status
    if doc_status == "failed":
        return "failed"
    if status in _CODE_JOB_TERMINAL_STATUSES and doc_status in _DOC_JOB_ACTIVE_STATUSES:
        return "running"
    return status


def _trim_code_preparation_history() -> None:
    if len(_CODE_PREPARATION_STATUS) <= _CODE_PREPARATION_HISTORY_LIMIT:
        return
    active_ids = set(_CODE_PREPARATION_BY_SCOPE.values())
    removable = [
        job_id for job_id, status in _CODE_PREPARATION_STATUS.items()
        if job_id not in active_ids
        and _preparation_finished(status)
    ]
    for job_id in removable[:len(_CODE_PREPARATION_STATUS) - _CODE_PREPARATION_HISTORY_LIMIT]:
        _CODE_PREPARATION_STATUS.pop(job_id, None)


def _active_code_preparation(key: RefreshKey) -> tuple[str, dict] | None:
    job_id = _CODE_PREPARATION_BY_SCOPE.get(key)
    if not job_id:
        return None
    status = _CODE_PREPARATION_STATUS.get(job_id)
    if not status or _preparation_finished(status):
        _CODE_PREPARATION_BY_SCOPE.pop(key, None)
        return None
    return job_id, status


def _code_scope_operation_lock(key: RefreshKey) -> asyncio.Lock:
    lock = _CODE_SCOPE_OPERATION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _CODE_SCOPE_OPERATION_LOCKS[key] = lock
    return lock


async def _submit_code_index_request(request: dict) -> dict:
    return await _post("/api/code/index", request)


async def _get_code_index_receipt(job_id: str) -> dict:
    return await _get(f"/api/code/jobs/{job_id}")


async def _submit_code_index_with_receipt(
    *, key: RefreshKey, client_job_id: str, request: dict,
) -> dict:
    request = dict(request)
    request["client_job_id"] = client_job_id
    async with _code_scope_operation_lock(key):
        try:
            return await _submit_code_index_request(request)
        except httpx.RequestError as submit_error:
            try:
                return await _get_code_index_receipt(client_job_id)
            except Exception:
                raise submit_error


def _backend_job_snapshot_is_stale(status: dict, backend: dict) -> bool:
    local_status = str(status.get("status", "preparing"))
    backend_status = str(backend.get("status", local_status))
    local_rank = _CODE_JOB_STATUS_RANK.get(local_status, -1)
    backend_rank = _CODE_JOB_STATUS_RANK.get(backend_status, -1)
    if local_status in _CODE_JOB_TERMINAL_STATUSES or backend_rank < local_rank:
        return True
    if backend_rank != local_rank or backend_status in _CODE_JOB_TERMINAL_STATUSES:
        return False
    local_done = int(status.get("chunks_done", 0) or 0)
    backend_done = int(backend.get("chunks_done", local_done) or 0)
    local_progress = float(status.get("progress", 0.0) or 0.0)
    backend_progress = float(backend.get("progress", local_progress) or 0.0)
    return backend_done < local_done or backend_progress < local_progress


# Local-only observability keys filled during preparing. Backend job receipts do not
# carry them; merging must not wipe the scan/chunk counters users already saw.
_LOCAL_PREPARATION_OBS_KEYS = (
    "files_done",
    "files_total",
    "files_changed",
    "chunks_ready",
    "summary",
    "phase",
)


def _update_preparation_from_backend(status: dict, backend: dict) -> None:
    if _backend_job_snapshot_is_stale(status, backend):
        return
    local_job_id = status["job_id"]
    backend_job_id = status.get("backend_job_id")
    project = status["project"]
    retained = {key: status[key] for key in _LOCAL_PREPARATION_OBS_KEYS if key in status}
    status.update(backend)
    status["job_id"] = local_job_id
    status["backend_job_id"] = backend_job_id
    status["project"] = project
    # Local scan counters / phase win: backend schema has files_total=changed-file
    # count which would otherwise clobber the full-tree files_total from preparing.
    for key, value in retained.items():
        if key == "phase" and backend.get("phase"):
            continue  # rare: backend may grow a phase field later
        status[key] = value


async def _refresh_preparation_from_backend(status: dict) -> str | None:
    """Refresh a local preparation view while its backend job is still running."""
    backend_job_id = status.get("backend_job_id")
    if not backend_job_id or status.get("status") in _CODE_JOB_TERMINAL_STATUSES:
        return None
    try:
        backend = await _get_code_index_receipt(str(backend_job_id))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return (
                f"backend job {backend_job_id} not found; backend restarted or no longer "
                "retains the job; showing cached local status. Re-run index_code if the "
                "job does not recover."
            )
        if exc.response.status_code not in _CODE_JOB_RETRYABLE_STATUS_CODES:
            raise
        return f"backend progress refresh unavailable ({exc}); showing cached local status."
    except httpx.RequestError as exc:
        return (
            f"backend progress refresh unavailable ({exc}); showing cached local status."
        )
    _update_preparation_from_backend(status, backend)
    return None


def _apply_doc_outcome(status: dict, outcome: _DocIndexOutcome, key: RefreshKey,
                       notes: tuple[str, ...] = ()) -> None:
    """Record the document pass on the preparation view. ``notes`` are warnings to show
    with it (those the pass's own message already carries are not repeated)."""
    extra = "".join(f" {note.strip()}" for note in notes
                    if note.strip() and note.strip() not in outcome.message)
    if outcome.job_id:
        unreadable = f", {outcome.failed} unreadable" if outcome.failed else ""
        status.update({
            "doc_job_id": outcome.job_id,
            "doc_status": "queued",
            "doc_files_total": outcome.queued,
            "doc_files_done": 0,
            "doc_message": f"{outcome.queued} changed, {outcome.unchanged} unchanged{unreadable}{extra}",
        })
        _DOC_PENDING_JOBS[key] = outcome.job_id
        return
    status["doc_message"] = f"{outcome.message}{extra}"
    if outcome.failed:
        status.update({"doc_status": "failed", "doc_error": outcome.message})
    else:
        status["doc_status"] = "completed"


def _close_doc_phase(status: dict, doc_status: str, reason: str) -> None:
    """The preparation ended abnormally: a document phase that would otherwise stay open
    forever (and keep the scope 'busy') is closed with the reason."""
    if status.get("doc_status") in _DOC_JOB_ACTIVE_STATUSES:
        status.update({"doc_status": doc_status, "doc_error": reason})


def _update_doc_from_backend(status: dict, job: dict) -> None:
    """Fold a Journal bulk-index job receipt into the preparation view (never backwards)."""
    new_status = str(job.get("status", status.get("doc_status", "queued")))
    current = str(status.get("doc_status", "queued"))
    if _CODE_JOB_STATUS_RANK.get(new_status, -1) < _CODE_JOB_STATUS_RANK.get(current, -1):
        return
    if current in _CODE_JOB_TERMINAL_STATUSES:
        return
    status.update({
        "doc_status": new_status,
        "doc_files_done": int(job.get("files_done", 0) or 0),
        "doc_files_total": int(job.get("files_total", status.get("doc_files_total", 0)) or 0),
        "doc_chunks_done": int(job.get("chunks_done", 0) or 0),
        "doc_chunks_total": int(job.get("chunks_total", 0) or 0),
    })
    if job.get("error"):
        status["doc_error"] = str(job["error"])


async def _refresh_doc_job_from_backend(status: dict) -> str | None:
    """Refresh the document half of a local preparation view while its job is live."""
    doc_job_id = status.get("doc_job_id")
    if not doc_job_id or status.get("doc_status") in _CODE_JOB_TERMINAL_STATUSES:
        return None
    try:
        job = await _get(f"/api/journal/bulk-index/jobs/{quote(str(doc_job_id), safe='')}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return (f"document job {doc_job_id} not found; backend restarted or no longer "
                    "retains the job; showing cached local status.")
        if exc.response.status_code not in _CODE_JOB_RETRYABLE_STATUS_CODES:
            raise
        return f"document job refresh unavailable ({exc}); showing cached local status."
    except httpx.RequestError as exc:
        return f"document job refresh unavailable ({exc}); showing cached local status."
    _update_doc_from_backend(status, job)
    return None


async def _run_code_index_preparation(
    *, local_job_id: str, project: str, host: str, repo_root: str,
    incremental: bool, respect_gitignore: bool, auto_refresh: bool,
    exclude_globs: list[str], docs: bool = False,
) -> None:
    key = (host, project)
    status = _CODE_PREPARATION_STATUS[local_job_id]
    try:
        status["phase"] = "checking scope overlap"
        block, nested_warning = await _nested_scope_guard(
            project, host, "code", respect_gitignore)
        doc_block: str | None = None
        doc_warning = ""
        doc_policy_warning = ""
        if docs:
            # The document scope is checked on its own: a block on one side never stops
            # the other, it is only reported next to it.
            doc_block, doc_warning = await _nested_scope_guard(
                project, host, "doc", respect_gitignore)
        if not docs or doc_block is None:
            doc_policy_warning = await _sync_doc_policy(
                project=project, host=host, root_dir=repo_root, docs=docs,
                auto_refresh=auto_refresh, respect_gitignore=respect_gitignore,
                exclude_globs=exclude_globs)

        walked: list[tuple[str, str, str]] | None = None

        async def ensure_walk() -> list[tuple[str, str, str]]:
            """The single directory walk code and documents share."""
            nonlocal walked
            if walked is None:
                walked = await asyncio.to_thread(lambda: list(_walk_repo_files(
                    repo_root, respect_gitignore, exclude_globs, docs=True)))
            return walked

        docs_started = False

        async def run_docs() -> None:
            """The document pass: submit changed documents through the Journal pipeline.
            It never raises: a document failure must not hide a good code index; it is
            recorded and surfaces in the job's overall status."""
            nonlocal docs_started
            if not docs or docs_started:
                return
            docs_started = True
            if doc_block is not None:
                status.update({"doc_status": "blocked", "doc_message": doc_block})
                return
            try:
                doc_files = [path for path, _rel, kind in await ensure_walk() if kind == "doc"]
                if not doc_files:
                    status.update({"doc_status": "completed",
                                   "doc_message": f"no documents found{doc_policy_warning}"})
                    return
                outcome = await _index_doc_files(
                    repo_root, project, host, doc_files, DOC_GLOBS, None,
                    nested_warning=doc_warning)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("document indexing failed for %s", project)
                status.update({"doc_status": "failed", "doc_error": str(exc)})
                return
            _apply_doc_outcome(status, outcome, key, (doc_warning, doc_policy_warning))

        async def run_code() -> tuple[str | None, str]:
            """The code pass. Returns (backend job to wait for, message to show once it
            has finished)."""
            policy_warning = ""
            status["phase"] = "saving refresh policy"
            try:
                await _set_code_policy(
                    project=project,
                    host=host,
                    root_dir=repo_root,
                    auto_refresh=auto_refresh,
                    respect_gitignore=respect_gitignore,
                    exclude_globs=exclude_globs,
                )
            except Exception as exc:
                logger.warning(
                    "index_code: policy persistence failed; indexing continues: %s", exc)
                policy_warning = " Warning: auto-refresh policy could not be saved."

            known: dict[str, dict] = {}
            manifest: list[str] | None = None
            current: dict[str, str] = {}
            status["phase"] = "scanning and chunking local files"
            walk_kwargs: dict[str, Any] = {}
            if docs and doc_block is None:
                walk_kwargs["files"] = [
                    (path, rel) for path, rel, kind in await ensure_walk() if kind == "code"]
            status.setdefault("files_done", 0)
            status.setdefault("files_total", 0)
            status.setdefault("files_changed", 0)
            status.setdefault("chunks_ready", 0)

            def _on_local_progress(update: dict) -> None:
                """Tick local file/chunk counters while AST walk runs in a worker thread."""
                files_total = int(update.get("files_total") or 0)
                files_done = int(update.get("files_done") or 0)
                files_changed = int(update.get("files_changed") or 0)
                chunks_ready = int(update.get("chunks_ready") or 0)
                local_progress = (files_done / files_total) if files_total else 0.0
                status.update({
                    "phase": str(update.get("phase") or status.get("phase")
                                 or "scanning and chunking local files"),
                    "files_total": files_total,
                    "files_done": files_done,
                    "files_changed": files_changed,
                    "chunks_ready": chunks_ready,
                    # Local phase owns progress until backend embed starts.
                    "progress": local_progress,
                })

            if incremental:
                known = await _known_code_hashes(project)
                known_hashes = {
                    rel: node["content_hash"] for rel, node in known.items()
                    if node.get("content_hash")
                }
                payloads, manifest, current = await asyncio.to_thread(
                    _collect_incremental, repo_root, known_hashes,
                    respect_gitignore, exclude_globs, _on_local_progress,
                    **walk_kwargs)
                if not payloads and set(manifest) == set(known):
                    status.update({
                        "status": "completed", "progress": 1.0,
                        "result_message": (
                            f"already up to date ({len(manifest)} files, nothing changed)"
                            f"{policy_warning}"
                        ),
                    })
                    if auto_refresh:
                        _ensure_code_watcher(
                            project=project, host=host, repo_root=repo_root,
                            initial_known=known, allow_during_preparation=True)
                    return None, ""
                commit = await asyncio.to_thread(_git_commit, repo_root)
                records = _inventory_records(manifest, payloads, known, current, commit)
                request = {
                    "project": project, "host": host, "root_dir": repo_root,
                    "files": payloads, "incremental": True, "manifest": manifest,
                    # Diffed against the catalog inventory fetched above; see _auto_refresh_code.
                    "removed_paths": sorted(set(known) - set(manifest)),
                    "removed_paths_complete": True,
                }
                summary = (
                    f"incremental: {len(payloads)} changed file(s), "
                    f"{sum(len(item['chunks']) for item in payloads)} chunk(s), "
                    f"{len(manifest)} in tree"
                )
            else:
                payloads = await asyncio.to_thread(
                    _collect_code_files, repo_root, respect_gitignore, exclude_globs,
                    _on_local_progress, **walk_kwargs)
                if not payloads:
                    status.update({
                        "status": "completed", "progress": 1.0,
                        "result_message": (
                            f"no indexable source files found under {repo_root}; the existing "
                            "index was kept unchanged. Use delete_code to clear it explicitly."
                            f"{policy_warning}"
                        ),
                    })
                    if auto_refresh:
                        _ensure_code_watcher(
                            project=project, host=host, repo_root=repo_root,
                            allow_during_preparation=True)
                    return None, ""
                commit = await asyncio.to_thread(_git_commit, repo_root)
                records = _inventory_records(
                    [item["rel_path"] for item in payloads], payloads, {}, {}, commit)
                request = {
                    "project": project, "host": host, "root_dir": repo_root,
                    "files": payloads, "incremental": False,
                }
                summary = (
                    f"full: {len(payloads)} file(s), "
                    f"{sum(len(item['chunks']) for item in payloads)} chunk(s)"
                )

            if nested_warning:
                summary = f"{nested_warning} {summary}"
            request["report"] = {"files": records, "replace": True}
            chunks_total = sum(len(item["chunks"]) for item in payloads)
            status.update({
                "phase": "submitting prepared chunks",
                "summary": summary,
                "chunks_total": chunks_total,
                "chunks_ready": chunks_total,
                "files_changed": len(payloads),
                # Local walk finished; keep progress at 1.0 until backend embed ticks.
                "progress": 1.0 if status.get("files_total") else status.get("progress", 0.0),
            })
            logger.info("%s — submitting to %s", summary, BASE_URL)
            backend_job = await _submit_code_index_with_receipt(
                key=key, client_job_id=local_job_id, request=request)
            backend_job_id = str(backend_job.get("job_id", "?"))
            status["backend_job_id"] = backend_job_id
            _update_preparation_from_backend(status, backend_job)
            status["phase"] = "backend embedding and upsert"
            await run_docs()
            if auto_refresh:
                _ensure_code_watcher(
                    project=project, host=host, repo_root=repo_root,
                    bootstrap_job_id=backend_job_id if backend_job_id != "?" else None,
                    initial_known=_known_from_records(records),
                    allow_during_preparation=True,
                )
            wait_for = (backend_job_id
                        if backend_job_id != "?" and status.get("status") not in _CODE_JOB_TERMINAL_STATUSES
                        else None)
            return wait_for, policy_warning.strip()

        if block:
            status.update({
                "status": "completed", "progress": 1.0,
                "result_message": block,
            })
            wait_code_job, final_message = None, ""
        else:
            wait_code_job, final_message = await run_code()
        await run_docs()  # a no-op unless the code pass ended before submitting (or was blocked)

        async def wait_code() -> None:
            if wait_code_job is not None and status.get("status") not in _CODE_JOB_TERMINAL_STATUSES:
                completed = await _wait_code_job_shared(wait_code_job)
                _update_preparation_from_backend(status, completed)
            if final_message:
                status["result_message"] = final_message

        async def wait_docs() -> None:
            doc_job_id = status.get("doc_job_id")
            if not doc_job_id or status.get("doc_status") not in _DOC_JOB_ACTIVE_STATUSES:
                return
            try:
                job = await _wait_doc_job_shared(str(doc_job_id))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("document job %s could not be tracked: %s", doc_job_id, exc)
                status.update({"doc_status": "failed",
                               "doc_error": f"document job {doc_job_id}: {exc}"})
                return
            _update_doc_from_backend(status, job)

        results = await asyncio.gather(wait_code(), wait_docs(), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        if docs and _DOC_PENDING_JOBS.get(key) == status.get("doc_job_id"):
            _DOC_PENDING_JOBS.pop(key, None)
        if status.get("doc_status") == "failed" and status.get("status") != "failed":
            status.update({
                "status": "failed",
                "error": f"document indexing failed: {status.get('doc_error') or 'unknown error'}",
            })
    except asyncio.CancelledError:
        status.update({"status": "cancelled", "error": "local preparation cancelled"})
        _close_doc_phase(status, "cancelled", "local preparation cancelled")
        raise
    except Exception as exc:  # keep failures pollable instead of losing a background task
        logger.exception("code index preparation failed for %s", project)
        status.update({"status": "failed", "error": str(exc)})
        _close_doc_phase(status, "failed", "code preparation failed; document job not tracked")
    finally:
        if _CODE_PREPARATION_BY_SCOPE.get(key) == local_job_id:
            _CODE_PREPARATION_BY_SCOPE.pop(key, None)


def _start_code_index_preparation(
    *, project: str, host: str, repo_root: str, incremental: bool,
    respect_gitignore: bool, auto_refresh: bool, exclude_globs: list[str],
    docs: bool = False,
) -> tuple[str, bool]:
    key = (host, project)
    existing = _active_code_preparation(key)
    if existing:
        return existing[0], False
    local_job_id = uuid.uuid4().hex
    _CODE_PREPARATION_STATUS[local_job_id] = {
        "job_id": local_job_id,
        "status": "preparing",
        "project": project,
        "progress": 0.0,
        "chunks_done": 0,
        "chunks_total": 0,
        "chunks_ready": 0,
        "files_done": 0,
        "files_total": 0,
        "files_changed": 0,
        "phase": "starting local repository scan",
    }
    if docs:
        _CODE_PREPARATION_STATUS[local_job_id].update(
            {"doc_status": "preparing", "doc_message": ""})
    _CODE_PREPARATION_BY_SCOPE[key] = local_job_id
    task = asyncio.create_task(_run_code_index_preparation(
        local_job_id=local_job_id,
        project=project,
        host=host,
        repo_root=repo_root,
        incremental=incremental,
        respect_gitignore=respect_gitignore,
        auto_refresh=auto_refresh,
        exclude_globs=exclude_globs,
        docs=docs,
    ))
    _CODE_PREPARATION_TASKS[local_job_id] = task

    def cleanup(done: asyncio.Task) -> None:
        if _CODE_PREPARATION_TASKS.get(local_job_id) is done:
            _CODE_PREPARATION_TASKS.pop(local_job_id, None)
        _trim_code_preparation_history()

    task.add_done_callback(cleanup)
    return local_job_id, True


@mcp.tool()
async def list_public_code_sources() -> str:
    """List curated public repositories, domains, active revisions, coverage counts,
    update policies, and check failures. Admin/internal access is required."""
    return _format_public_code_sources(await _get("/api/public-code/sources"))


@mcp.tool()
async def add_public_code_source(
    source_id: str,
    domain: str,
    upstream_url: str,
    relative_path: str,
    default_ref: str = "HEAD",
    proxy_profile: str | None = None,
    credential_profile: str | None = None,
    language_overrides: dict[str, str] | None = None,
    update_policy: str = "manual",
    update_interval_minutes: int | None = None,
    index_now: bool = True,
) -> str:
    """Register one curated Git repository and optionally queue its first index.

    Proxy and credential arguments are SERVER-SIDE profile names only. Never pass raw
    cookies, tokens, passwords, private keys, or Authorization header values.
    """
    response = await _post("/api/public-code/sources", {
        "source_id": source_id,
        "domain": domain,
        "upstream_url": upstream_url,
        "relative_path": relative_path,
        "default_ref": default_ref,
        "proxy_profile": proxy_profile,
        "credential_profile": credential_profile,
        "language_overrides": language_overrides or {},
        "update_policy": update_policy,
        "update_interval_minutes": update_interval_minutes,
        "index_now": index_now,
    })
    lines = [_format_public_code_sources([response.get("source") or {}])]
    if response.get("index_job"):
        lines.append(_format_public_code_job(response["index_job"]))
    return "\n".join(lines)


@mcp.tool()
async def check_public_code_updates(
    source_ids: list[str] | None = None,
    domains: list[str] | None = None,
) -> str:
    """Queue non-blocking upstream revision checks. Empty selections mean all sources."""
    job = await _post("/api/public-code/sources/check-updates", {
        "source_ids": source_ids or [], "domains": domains or [],
    })
    return _format_public_code_job(job)


@mcp.tool()
async def index_public_code_sources(
    source_ids: list[str] | None = None,
    domains: list[str] | None = None,
) -> str:
    """Queue mirror refresh, AST chunking, embedding, validation, and promotion for
    selected repositories. Empty selections mean all configured sources."""
    job = await _post("/api/public-code/sources/index", {
        "source_ids": source_ids or [], "domains": domains or [],
    })
    return _format_public_code_job(job)


@mcp.tool()
async def set_public_code_update_policy(
    source_id: str,
    update_policy: str = "manual",
    update_interval_minutes: int | None = None,
) -> str:
    """Set a repository to manual or scheduled update checks."""
    source = await _put(
        f"/api/public-code/sources/{quote(source_id, safe='')}/update-policy",
        {
            "update_policy": update_policy,
            "update_interval_minutes": update_interval_minutes,
        },
    )
    return _format_public_code_sources([source])


@mcp.tool()
async def public_code_job_status(job_id: str | None = None, limit: int = 20) -> str:
    """Poll one public-code job or import handle, or list recent durable jobs when
    job_id is omitted."""
    if job_id:
        local = _PUBLIC_IMPORTS.get(job_id)
        if local is not None:
            lines = [_format_public_import(job_id, local)]
            if local.get("job_id"):
                lines.append(_format_public_code_job(await _get(
                    f"/api/public-code/jobs/{quote(local['job_id'], safe='')}")))
            return "\n".join(lines)
        return _format_public_code_job(
            await _get(f"/api/public-code/jobs/{quote(job_id, safe='')}"))
    jobs = await _get("/api/public-code/jobs", {"limit": limit})
    if not jobs:
        return "No public-code jobs found."
    return "\n".join(_format_public_code_job(job) for job in jobs)


# ---- public-code bundle import: a clean local clone replaces the server's network clone ----
# Byte-identical in mcp_plugin/retrieval_mcp.py and mcp_server/backend.py (asserted by tests).

_PUBLIC_IMPORT_PART_BYTES = 32 * 1024 * 1024
_PUBLIC_IMPORT_PART_ATTEMPTS = 5


def _https_upstream(url: str) -> str:
    """git@github.com:owner/repo.git -> https://github.com/owner/repo.git; others unchanged."""
    match = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+)$", url.strip())
    return f"https://{match.group(1)}/{match.group(2)}" if match else url.strip()


def _upstream_key(url: str) -> str:
    """host/owner/repo, so https/ssh and with/without .git name one upstream."""
    url = _https_upstream(url).rstrip("/")
    url = url[:-4] if url.endswith(".git") else url
    rest = url.partition("://")[2] or url
    host, _, path = rest.partition("/")
    return f"{host.lower()}/{path}"


def _run_git_text(repo: str, *arguments: str, timeout: float = 120) -> tuple[int, str]:
    # stdin=DEVNULL: an MCP stdio server's stdin is the JSON-RPC pipe (W159)
    out = subprocess.run(["git", "-C", repo, *arguments], stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=timeout)
    return out.returncode, (out.stdout if out.returncode == 0 else out.stderr).strip()


def _inspect_public_clone(path: str) -> dict:
    """The clone's HEAD, branch and origin; ValueError unless it is published upstream code.

    Only a clean HEAD that is on an origin branch is accepted, so the public corpus never
    receives local edits or unpublished commits. A --depth 1 clone or a bare mirror is fine."""
    code, bare = _run_git_text(path, "rev-parse", "--is-bare-repository")
    if code != 0:
        raise ValueError(f"{path} is not a git repository: {bare}")
    repo = path
    if bare != "true":
        repo = _run_git_text(path, "rev-parse", "--show-toplevel")[1]
        code, dirty = _run_git_text(repo, "status", "--porcelain", "--untracked-files=no")
        if code != 0 or dirty:
            raise ValueError(
                f"{repo} has uncommitted changes to tracked files; only published upstream "
                "code is accepted, so commit nothing local and stash or discard them")
    code, head = _run_git_text(repo, "rev-parse", "--verify", "HEAD^{commit}")
    if code != 0:
        raise ValueError(f"{repo} has no HEAD commit: {head}")
    code, origin = _run_git_text(repo, "remote", "get-url", "origin")
    if code != 0 or not origin:
        raise ValueError(f"{repo} has no origin remote, so its upstream is unknown")
    if bare != "true":
        code, containing = _run_git_text(repo, "branch", "-r", "--contains", "HEAD")
        if code != 0 or not any(line.strip().startswith("origin/")
                                for line in containing.splitlines()):
            raise ValueError(
                f"HEAD {head[:12]} of {repo} is not on any origin branch; only published "
                "upstream code is accepted (fetch, then check out an origin branch)")
    code, branch = _run_git_text(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    return {"repo": repo, "head": head, "origin": _https_upstream(origin),
            "branch": branch if code == 0 else None}


def _resolve_public_import_source(clone: dict, sources: list[dict], *, domain: str | None,
                                  source_id: str | None) -> dict:
    """Reuse the registered source with the same upstream; otherwise describe a new one."""
    key = _upstream_key(clone["origin"])
    for source in sources:
        if _upstream_key(source.get("upstream_url") or "") != key:
            continue
        for name, given in (("source_id", source_id), ("domain", domain)):
            if given and given != source.get(name):
                raise ValueError(
                    f"{clone['origin']} is already registered as {source['source_id']} in "
                    f"domain {source['domain']}; omit {name} or pass {source.get(name)!r}")
        ref = source.get("default_ref") or "HEAD"
        if ref != "HEAD" and ref.removeprefix("refs/heads/") != clone["branch"]:
            raise ValueError(
                f"{source['source_id']} indexes branch {ref}; check out that branch "
                f"(the clone is on {clone['branch'] or 'a detached HEAD'})")
        return {"source_id": source["source_id"], "domain": source["domain"],
                "relative_path": source["relative_path"], "default_ref": ref,
                "upstream_url": source["upstream_url"], "new": False}
    if not domain:
        known = sorted({str(source.get("domain")) for source in sources})
        raise ValueError(
            f"{clone['origin']} is not a registered public source; pass domain to add it "
            f"(existing domains: {', '.join(known)})")
    name = source_id or key.rsplit("/", 1)[-1]
    return {"source_id": name, "domain": domain, "relative_path": f"{domain}/{name}",
            "default_ref": "HEAD", "upstream_url": clone["origin"], "new": True}


def _create_public_bundle(clone: dict, target: str) -> tuple[int, str]:
    """Bundle HEAD (and its branch) of the clone; returns (size, sha256)."""
    refs = ["HEAD"] + ([clone["branch"]] if clone["branch"] else [])
    code, message = _run_git_text(clone["repo"], "bundle", "create", target, *refs,
                                  timeout=3600)
    if code != 0:
        raise ValueError(f"git bundle create failed: {message}")
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return os.path.getsize(target), digest.hexdigest()


async def _run_public_import(status: dict, clone: dict, source: dict, *,
                             create: Callable[[dict], Awaitable[dict]],
                             put: Callable[[str, int, bytes], Awaitable[dict]],
                             get: Callable[[str], Awaitable[dict]],
                             complete: Callable[[str], Awaitable[dict]]) -> None:
    """Bundle, upload in resumable parts, complete. Progress lands in `status`."""
    directory = tempfile.mkdtemp(prefix="retrieval-import-")
    target = os.path.join(directory, f"{source['source_id']}.bundle")
    try:
        status["phase"] = "bundling"
        size, sha256 = await asyncio.to_thread(_create_public_bundle, clone, target)
        status.update(phase="uploading", bytes_total=size, bytes_done=0)
        created = await create({
            "source_id": source["source_id"], "domain": source["domain"],
            "relative_path": source["relative_path"], "upstream_url": source["upstream_url"],
            "default_ref": source["default_ref"], "head_commit": clone["head"],
            "size_bytes": size, "sha256": sha256,
            "private_project": clone["repo"],
        })
        import_id = created["import_id"]
        status["import_id"] = import_id
        offset, failures = 0, 0
        with open(target, "rb") as handle:
            while offset < size:
                handle.seek(offset)
                data = await asyncio.to_thread(handle.read, _PUBLIC_IMPORT_PART_BYTES)
                try:
                    offset = int((await put(import_id, offset, data))["received_bytes"])
                    failures = 0
                except httpx.HTTPError:
                    failures += 1
                    if failures >= _PUBLIC_IMPORT_PART_ATTEMPTS:
                        raise
                    await asyncio.sleep(2 ** failures)
                    # resume where the server says it is, not where this side thinks
                    offset = int((await get(import_id))["received_bytes"])
                status["bytes_done"] = offset
        receipt = await complete(import_id)
        status.update(phase="submitted", job_id=receipt["job"]["job_id"])
    except Exception as exc:
        status.update(phase="failed", error=str(exc))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _format_public_import(handle: str, status: dict) -> str:
    total = int(status.get("bytes_total") or 0)
    done = int(status.get("bytes_done") or 0)
    line = (f"Public code import {handle} [{status.get('phase')}] {status.get('source_id')} "
            f"@ {str(status.get('head') or '')[:12]}")
    if total:
        line += f"; uploaded {done / 1e6:.1f}/{total / 1e6:.1f} MB"
    if status.get("import_id"):
        line += f"; import_id={status['import_id']}"
    if status.get("error"):
        line += f"; error={status['error']}"
    return line


_PUBLIC_IMPORTS: dict[str, dict] = {}  # handle -> local bundling/upload progress
_PUBLIC_IMPORT_TASKS: set[asyncio.Task] = set()


async def _put_public_import_part(import_id: str, offset: int, data: bytes) -> dict:
    r = await _client().put(
        f"/api/public-code/imports/{quote(import_id, safe='')}/data",
        params={"offset": offset}, content=data,
        headers={"Content-Type": "application/octet-stream"},
        timeout=httpx.Timeout(600.0, connect=30.0))
    _raise_for_status(r)
    return r.json()


@mcp.tool()
async def import_public_code_repo(path: str, domain: str | None = None,
                                  source_id: str | None = None) -> str:
    """Add a repository you have cloned to the curated Public Code corpus by uploading
    it, for repositories the server cannot clone itself (pytorch, linux, llvm-project
    time out over its proxy). Admin tier.

    The clone must be clean with HEAD on an origin branch (published upstream code
    only); a --depth 1 clone or a bare mirror is enough. HEAD is packed locally as a git
    bundle and uploaded in resumable parts; the server builds the mirror (origin =
    upstream, so later updates fetch upstream), then chunks, embeds and indexes it.
    Nothing is embedded on this machine. A registered source with the same upstream is
    updated in place; otherwise pass domain to add a new one. Returns a handle at once;
    poll public_code_job_status(job_id=<handle>).

    Args:
        path: Local clone (working tree or bare mirror).
        domain: Domain for a new source, e.g. "machine-learning"; not needed when the
            upstream is already registered.
        source_id: Name for a new source; default is the repository name from origin.
    """
    clone = await asyncio.to_thread(_inspect_public_clone, _session_abspath(path))
    source = _resolve_public_import_source(
        clone, await _get("/api/public-code/sources"), domain=domain, source_id=source_id)
    handle = f"import-{uuid.uuid4().hex[:12]}"
    status = {"phase": "queued", "source_id": source["source_id"], "head": clone["head"]}
    _PUBLIC_IMPORTS[handle] = status
    task = asyncio.create_task(_run_public_import(
        status, clone, source,
        create=lambda body: _post("/api/public-code/imports", body),
        put=_put_public_import_part,
        get=lambda import_id: _get(f"/api/public-code/imports/{quote(import_id, safe='')}"),
        complete=lambda import_id: _post(
            f"/api/public-code/imports/{quote(import_id, safe='')}/complete", {}),
    ))
    _PUBLIC_IMPORT_TASKS.add(task)
    task.add_done_callback(_PUBLIC_IMPORT_TASKS.discard)
    return _public_import_receipt(handle, clone, source)


def _public_import_receipt(handle: str, clone: dict, source: dict) -> str:
    what = (f"new source {source['source_id']} in domain {source['domain']}"
            if source["new"] else f"registered source {source['source_id']}")
    return (f"Public code import {handle} started: {what} ({source['upstream_url']}) at "
            f"{clone['head'][:12]}. Bundling and uploading run in the background; the "
            "server then builds the mirror and chunks, embeds and indexes it. Poll "
            f'public_code_job_status(job_id="{handle}").')


async def _search_public_code(
    query: str,
    source_ids: list[str] | None = None,
    domains: list[str] | None = None,
    top_k: int = 10,
    mode: str = "hybrid",
    language: str | None = None,
    rerank: bool = False,
    show_line_numbers: bool = True,
) -> str:
    """Public Code corpus search (search_code with sources="public")."""
    response = await _post("/api/public-code/search", {
        "query": query,
        "source_ids": source_ids or [],
        "domains": domains or [],
        "top_k": top_k,
        "mode": mode,
        "language": language,
        "rerank": rerank,
    })
    return _format_public_code_search(response, show_line_numbers=show_line_numbers)


@mcp.tool()
async def index_code(repo_root: str = ".",
                     incremental: bool = False, respect_gitignore: bool = True,
                     auto_refresh: bool = True,
                     exclude_globs: list[str] | None = None,
                     docs: bool = True) -> str:
    """Index a code repository so you can later search it semantically with search_code.

    Walks repo_root and AST-chunks every supported source file with a `# path:` scope
    header LOCALLY, then sends only the chunks to the backend, which embeds them with
    the shared BGE-M3 and stores the vectors in the shared code index, scoped to YOU.
    Source files never leave the machine — only chunk text does.

    The index scope is repo_root's ABSOLUTE PATH — never a hand-typed name. The same
    directory always maps to one scope (so re-indexing it updates in place instead of
    fragmenting into name-variant duplicates), and search_code(repo_root=...) with the
    same path finds it.

    Coverage: 40+ languages via tree-sitter (Python, TS/TSX/JS/JSX/React, Java, Kotlin,
    Swift, Go, Rust, C/C++, C#, Ruby, PHP, Lua, Scala, Dart, SQL, HTML/CSS/SCSS,
    YAML/JSON/TOML, shell, Dockerfile, and more); grammarless config/text files fall
    back to line-window chunks. Images and other binary files are skipped.

    Documents: the same walk also finds documentation files (.md .mdx .markdown .rst
    .adoc .txt) and indexes them into the JOURNAL, not the code index — find them with
    journal_search(scope_dir=repo_root), never search_code. They are whole-document,
    latest-wins entries: an unchanged file is a no-op, an edited one replaces its
    previous version, and a deleted one is removed. Dot-directories and ignored paths
    are skipped like code. Set docs=False to index code only.

    Indexing is ASYNC and non-blocking end to end: it returns a local job_id before the
    repository walk, hashing, AST chunking, upload, GPU embedding, or Qdrant upsert
    finishes. Poll the same job_id with index_code_status through every phase; it shows
    the code job and the document job side by side.

    Args:
        repo_root: Path to the repo/dir to index (default: current directory "."). Its
            absolute path is the search scope.
        incremental: Default False — a clean wipe-and-rebuild (the original, always-correct
            path; never drifts). Set True for fast repeat indexing: hash every file,
            re-embed only those whose content changed since the last index, and reconcile
            deletions. Incremental is byte-identical to a full reindex of the current tree
            (the first index of a repo is a full pass automatically) — it just skips
            re-embedding unchanged files.
        respect_gitignore: Default True — skip files git ignores (build output, secrets,
            vendored deps) so they never pollute the index. Resolution is delegated to
            git, so nested .gitignore / negation / global excludes all hold; set False to
            index every supported file regardless. A no-op when repo_root isn't a git repo.
        docs: Default True — also index documentation files into the journal in this
            same run and keep them fresh with the code (see Documents). False indexes
            code only and switches document refresh off for this scope.

    Nested scopes: indexing a directory already fully covered by an indexed PARENT scope
    is refused (no coverage benefit); indexing a PARENT over an already-indexed child
    proceeds with a warning (it adds files outside the child). Exception: passing
    respect_gitignore=False under a parent indexed with it ON proceeds with a warning —
    that is how you index an ignored subtree without deleting the parent scope. Code and
    documents are checked separately; a block on one never stops the other.

    Example: index_code(repo_root="~/Prd/retrival", incremental=True)
    """
    host = _env("RETRIEVAL_HOST", "RETRIVAL_HOST", default=socket.gethostname()) or ""
    # Honors ~ and resolves a relative root against the CALLING client's directory, not
    # the singleton daemon's.  The scope is derived from that same absolute path so the
    # walk and the project it registers under cannot disagree.
    repo_root = _session_abspath(repo_root)
    root_abs = repo_root
    proj = _scope_path(repo_root)
    existing = _active_code_preparation((host, proj))
    if existing:
        return (f"Code indexing is already running for project '{proj}' — job "
                f"{existing[0]} [{_overall_job_status(existing[1]) or 'preparing'}]. Poll "
                f"index_code_status(job_id={existing[0]!r}); no duplicate scan was started.")
    excludes = list(exclude_globs or [])
    _invalidate_refresh(proj, host)
    local_job_id, started = _start_code_index_preparation(
        project=proj,
        host=host,
        repo_root=root_abs,
        incremental=incremental,
        respect_gitignore=respect_gitignore,
        auto_refresh=auto_refresh,
        exclude_globs=excludes,
        docs=docs,
    )
    if not started:
        return (f"Code indexing is already running for project '{proj}' — job "
                f"{local_job_id}. No duplicate scan was started.")
    documents = (" Documentation files are indexed into the journal in the same run "
                 "(search them with journal_search)." if docs else "")
    return (f"Indexing started in the background — job {local_job_id} [preparing]. "
            "Local scanning, hashing, AST chunking, upload, GPU embedding, and Qdrant "
            f"upsert all continue asynchronously. Poll index_code_status(job_id="
            f"{local_job_id!r}) for progress. Source files never leave this machine."
            f"{documents}")


def _fmt_preparing_work(job: dict) -> str:
    """Human-readable local-preparation progress (files hashed/chunked, not embed %)."""
    phase = str(job.get("phase") or "local preparation")
    files_done = int(job.get("files_done") or 0)
    files_total = int(job.get("files_total") or 0)
    files_changed = int(job.get("files_changed") or 0)
    chunks_ready = int(job.get("chunks_ready") or job.get("chunks_total") or 0)
    parts = [phase]
    if files_total > 0:
        parts.append(f"files {files_done}/{files_total}")
    if files_changed > 0:
        parts.append(f"{files_changed} changed")
    if chunks_ready > 0:
        parts.append(f"{chunks_ready} chunks ready")
    return "; ".join(parts)


def _fmt_doc_part(job: dict) -> str:
    """The document half of an index_code job line: the Journal bulk-index job that
    carried this run's documents (``docs: job <id> [status] ...``), or why there is none."""
    status = job.get("doc_status")
    if job.get("doc_job_id"):
        part = f"docs: job {job['doc_job_id']} [{status}]"
        files_total = int(job.get("doc_files_total") or 0)
        if files_total:
            part += f" {int(job.get('doc_files_done') or 0)}/{files_total} files"
        chunks_total = int(job.get("doc_chunks_total") or 0)
        if chunks_total:
            part += f", {int(job.get('doc_chunks_done') or 0)}/{chunks_total} chunks"
    else:
        part = f"docs: [{status}]"
    if job.get("doc_message"):
        part += f" {job['doc_message']}"
    if job.get("doc_error"):
        part += f" — docs error: {job['doc_error']}"
    return part


def _fmt_job_line(job: dict) -> str:
    pct = round(100 * job.get("progress", 0.0))
    removed = int(job.get("removed_total", 0) or 0)
    if job.get("status") == "preparing":
        work = _fmt_preparing_work(job)
    elif job.get("chunks_total", 0):
        work = f"{job.get('chunks_done', 0)}/{job.get('chunks_total', 0)} chunks"
        # Keep local prepare context visible after backend embed starts.
        files_changed = int(job.get("files_changed") or 0)
        files_total = int(job.get("files_total") or 0)
        if files_changed and files_total:
            work += f"; prepared {files_changed}/{files_total} files"
        elif files_changed:
            work += f"; prepared {files_changed} files"
    elif removed:
        work = f"removed {removed} path(s)"
    else:
        work = "0 chunks"
    line = (f"{job.get('job_id')} [{_overall_job_status(job)}] project={job.get('project')} "
            f"{work} ({pct}%)")
    if pos := job.get("queue_position", 0):
        line += f", queue position {pos}"
    if job.get("status") == "completed":
        line += f" — indexed {job.get('indexed', 0)} chunks."
        if removed:
            line += f" removed {removed} path(s)."
        if dup := job.get("duplicates_updated", 0):
            line += f" {dup} cross-scope copy(ies) synced."
    if backend_job_id := job.get("backend_job_id"):
        line += f" backend job {backend_job_id}."
    if summary := job.get("summary"):
        line += f" — {summary}"
    if message := job.get("result_message"):
        line += f" — {message}"
    if job.get("error"):
        line += f" — error: {job['error']}"
    if job.get("doc_status"):
        line += f" — {_fmt_doc_part(job)}"
    return line


@mcp.tool()
async def index_code_status(job_id: str | None = None) -> str:
    """Check the progress of code index jobs submitted via index_code.

    Args:
        job_id: A specific job to check (from index_code's reply). Default: None —
            lists your recent jobs (newest first) with status and progress.
    """
    if job_id:
        if local := _CODE_PREPARATION_STATUS.get(job_id):
            notice = await _refresh_preparation_from_backend(local)
            doc_notice = await _refresh_doc_job_from_backend(local)
            line = "Job " + _fmt_job_line(local)
            for note in (notice, doc_notice):
                if note:
                    line = f"{line} — {note}"
            return line
        try:
            # Opt in to the scope's deliver-once message: only a person asking about this
            # job should consume it, not the client's own progress polling.
            job = await _get(f"/api/code/jobs/{job_id}", {"scope_notice": "true"})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return (f"Job {job_id} not found — it may have finished long ago or the "
                        f"backend restarted (queue is in-memory). Re-run index_code if needed.")
            raise
        return _with_scope_notice("Job " + _fmt_job_line(job), job.get("scope_notice"))
    local_jobs = list(reversed(list(_CODE_PREPARATION_STATUS.values())))
    list_notice = None
    try:
        resp = await _get(
            "/api/code/jobs", {"limit": _CODE_PREPARATION_HISTORY_LIMIT})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in _CODE_JOB_RETRYABLE_STATUS_CODES:
            raise
        resp = {"jobs": []}
        list_notice = (
            f"backend progress refresh unavailable ({exc}); showing cached local status."
        )
    except httpx.RequestError as exc:
        resp = {"jobs": []}
        list_notice = (
            f"backend progress refresh unavailable ({exc}); showing cached local status."
        )
    backend_jobs_by_id = {
        job.get("job_id"): job for job in (resp.get("jobs") or [])
        if job.get("job_id")
    }
    for status in local_jobs:
        backend = backend_jobs_by_id.get(status.get("backend_job_id"))
        if backend:
            _update_preparation_from_backend(status, backend)
        doc_notice = await _refresh_doc_job_from_backend(status)
        if doc_notice and not list_notice:
            list_notice = doc_notice
    missing_active_backend_jobs = [
        status for status in local_jobs
        if status.get("backend_job_id")
        and status.get("status") not in _CODE_JOB_TERMINAL_STATUSES
        and status.get("backend_job_id") not in backend_jobs_by_id
    ]
    if missing_active_backend_jobs and not list_notice:
        list_notice = (
            f"backend may have restarted or omitted {len(missing_active_backend_jobs)} "
            "active job(s); showing cached local status."
        )
    mapped_backend_ids = {
        status.get("backend_job_id") for status in local_jobs
        if status.get("backend_job_id")
    }
    backend_jobs = [
        job for job in (resp.get("jobs") or [])
        if job.get("job_id") not in mapped_backend_ids
    ]
    jobs = local_jobs + backend_jobs
    if not jobs:
        if list_notice:
            return f"No cached index jobs — {list_notice}"
        return "No index jobs yet — run index_code first."
    heading = f"{len(jobs)} recent index job(s):"
    if list_notice:
        heading += f" — {list_notice}"
    return "\n".join([heading]
                     + [f"- {_fmt_job_line(j)}" for j in jobs])


async def _search_private_code(query: str, top_k: int = 8, repo_root: str = ".",
                      mode: str = "hybrid", rerank: bool = False,
                      language: str | None = None, path_glob: str | None = None,
                      max_per_file: int | None = None, show_line_numbers: bool = True) -> str:
    """Private single-scope search (search_code with sources="private" and one repo_root)."""
    t0 = time.perf_counter()
    proj = _scope_path(repo_root)
    normalized_path_glob = _normalize_path_glob(path_glob) if path_glob else None
    local_root = _local_repo_root(repo_root)
    host = _env("RETRIEVAL_HOST", "RETRIVAL_HOST", default=socket.gethostname()) or ""
    logger.info(
        "search_code.start query=%r project=%s top_k=%s mode=%s rerank=%s path_glob=%r",
        query, proj, top_k, mode, rerank, normalized_path_glob,
    )
    if local_root is not None:
        _ensure_code_watcher(project=proj, host=host, repo_root=local_root)
    payload: dict[str, Any] = {"query": query, "top_k": top_k, "mode": mode, "rerank": rerank,
                               "project": proj}
    if language:
        payload["language"] = language
    if normalized_path_glob:
        payload["path_glob"] = normalized_path_glob
    if max_per_file:
        payload["max_per_file"] = max_per_file
    try:
        resp = await asyncio.wait_for(
            _post("/api/code/search", payload), timeout=_SEARCH_CODE_TIMEOUT_S)
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "search_code.timeout project=%s timeout_s=%.2f elapsed_ms=%.0f",
            proj, _SEARCH_CODE_TIMEOUT_S, elapsed_ms,
        )
        return (
            f"search_code timed out after {_SEARCH_CODE_TIMEOUT_S:.2f} seconds "
            f"(project={proj!r}). Backend search is usually <1s; retry once, narrow with "
            "path_glob/language, or check that the backend is reachable without a stuck "
            "proxy path."
        )
    t_http = time.perf_counter()
    freshness = _CODE_WATCH_STATUS.get((host, proj)) if local_root is not None else None
    hits = resp.get("hits") or []
    header = f"Code matches for: {query!r}"
    if hits:
        rendered = _fmt_code_hits(hits, header, show_line_numbers=show_line_numbers)
        if resp.get("notice"):
            rendered += f"\n\nSearch diagnostic: {resp['notice']}"
        requested_mode = resp.get("requested_mode")
        effective_mode = resp.get("effective_mode")
        if requested_mode and effective_mode and requested_mode != effective_mode:
            rendered += (
                f"\nEffective mode: {effective_mode!r} (requested {requested_mode!r})."
            )
        aliases = resp.get("scope_aliases") or []
        if aliases:
            rendered += "\nIncluded legacy Windows scope spelling(s): " + ", ".join(repr(a) for a in aliases)
    else:
        rendered = f"{header}\n\n" + _format_scope_miss(
            result_kind="code",
            query=query,
            scope=proj,
            scope_argument="repo_root",
            search_tool="search_code",
            next_step=f"index_code(repo_root={proj!r})",
            path_glob=normalized_path_glob,
            requested_mode=resp.get("requested_mode", mode),
            effective_mode=resp.get("effective_mode"),
            notice=resp.get("notice"),
            scope_snapshot_state=resp.get("scope_snapshot_state"),
            recovery_required=bool(resp.get("recovery_required")),
            active_index_job_id=resp.get("active_index_job_id"),
            path_glob_matched_files=resp.get("path_glob_matched_files"),
            indexed_file_count=resp.get("indexed_file_count"),
            scope_input=repo_root,
            local_source_available=local_root is not None,
        )
    note = _format_refresh_status(freshness)
    out = f"{rendered}\n\n{note}" if note else rendered
    logger.info(
        "search_code.done project=%s hits=%d out_chars=%d http_ms=%.0f total_ms=%.0f",
        proj, len(hits), len(out),
        (t_http - t0) * 1000, (time.perf_counter() - t0) * 1000,
    )
    return _with_scope_notice(out, resp.get("scope_notice"))


def _format_federated_code_search(response: dict, *, show_line_numbers: bool = True) -> str:
    hits = response.get("hits") or []
    header = f"Federated code matches for {response.get('query')!r}: {len(hits)} hit(s)."
    private = response.get("private") or {}
    public = response.get("public") or {}
    diagnostics: list[str] = []
    # The private leg always runs (R1: no include_private flag) so this line is
    # unconditional, unlike the public leg below.
    line = f"- private: {private.get('candidate_count', 0)} candidate(s)"
    state = private.get("scope_snapshot_state")
    if state and state != "completed":
        line += f" (scope_snapshot_state={state})"
    if private.get("notice"):
        line += f" — {private['notice']}"
    excluded = private.get("excluded_scopes") or []
    if excluded:
        names = ", ".join(str(scope.get("project")) for scope in excluded)
        line += f" — excluded scope(s) not ready: {names}"
    diagnostics.append(line)
    if public.get("included"):
        selected = public.get("selected_source_ids") or []
        line = f"- public: {public.get('candidate_count', 0)} candidate(s)"
        if selected:
            line += " across " + ", ".join(selected)
        if public.get("notice"):
            line += f" — {public['notice']}"
        diagnostics.append(line)
    else:
        diagnostics.append("- public: not included (include_public=false)")

    if not hits:
        return "\n".join([header, *diagnostics])

    lines = [header, "", *diagnostics, ""]
    for i, hit in enumerate(hits, 1):
        if hit.get("origin") == "public":
            revision = str(hit.get("revision") or "")[:10]
            tag = f"[public:{hit.get('source_id')}@{revision}]"
        else:
            tag = "[private]"
        label = f"  [{hit.get('label')}]" if hit.get("label") else ""
        lines.append(
            f"{i}. {tag} {hit.get('path')}:{hit.get('start_line')}-{hit.get('end_line')}"
            f"{label}  score={hit.get('score', 0.0):.3f}"
        )
        lines.append(f"```{hit.get('language') or ''}")
        body = str(hit.get("code") or "").rstrip("\n")
        lines.append(_number_code_lines(body, hit.get("start_line")) if show_line_numbers else body)
        lines.append("```")
    return "\n".join(lines)


async def _search_federated_code(query: str, repo_roots: list[str] | None = None,
                          include_public: bool = False,
                          source_ids: list[str] | None = None,
                          domains: list[str] | None = None, top_k: int = 10,
                          mode: str = "hybrid", language: str | None = None,
                          path_glob: str | None = None, max_per_file: int | None = None,
                          rerank: bool = False, show_line_numbers: bool = True) -> str:
    """Federated search: several private scopes, optionally fused with Public Code."""
    projects = [_scope_path(root) for root in (repo_roots or [])]
    normalized_path_glob = _normalize_path_glob(path_glob) if path_glob else None
    logger.info(
        "search_code.federated.start query=%r projects=%s top_k=%s include_public=%s "
        "mode=%s rerank=%s",
        query, projects, top_k, include_public, mode, rerank,
    )
    payload: dict[str, Any] = {
        "query": query, "top_k": top_k, "mode": mode, "rerank": rerank,
        "include_public": include_public,
        "projects": projects, "source_ids": source_ids or [], "domains": domains or [],
    }
    if language:
        payload["language"] = language
    if normalized_path_glob:
        payload["path_glob"] = normalized_path_glob
    if max_per_file is not None:
        payload["max_per_file"] = max_per_file
    resp = await _post("/api/code/federated-search", payload)
    return _format_federated_code_search(resp, show_line_numbers=show_line_numbers)


async def _explain_unknown_public_selection(exc: httpx.HTTPStatusError) -> None:
    """Re-raise the backend's "Unknown public-code source IDs/domains" 422 with the
    names that do exist, so the next call can be corrected without a second tool."""
    reason = str(exc).split(": ", 1)[-1]
    if exc.response.status_code != 422 or not reason.startswith("Unknown public-code"):
        return
    try:
        sources = await _get("/api/public-code/sources")
    except httpx.HTTPError:
        return
    if "domains" in reason:
        valid = sorted({str(source.get("domain")) for source in sources})
        field = "domains"
    else:
        valid = sorted(str(source.get("source_id")) for source in sources
                       if source.get("active_revision"))
        field = "source_ids"
    raise ValueError(
        f"search_code: {reason}. Valid {field}: {', '.join(valid)}. "
        "Omit source_ids/domains to search the whole public corpus.") from None


_CODE_SOURCES = ("private", "public", "all")


@mcp.tool()
async def search_code(query: str, top_k: int = 8, repo_root: str | list[str] = ".",
                      mode: str = "hybrid", rerank: bool = False,
                      language: str | None = None, path_glob: str | None = None,
                      max_per_file: int | None = None, sources: str = "private",
                      source_ids: list[str] | None = None,
                      domains: list[str] | None = None,
                      show_line_numbers: bool = True) -> str:
    """Search indexed code and get each hit back as a syntax-aware chunk (usually a
    whole function or class) with `path:line` citations: your own indexed repositories
    (default), the curated Public Code corpus, or both in one ranking.

    Code blocks show source line numbers by default (e.g. `455 | code`).
    Set show_line_numbers=False to return plain code.

    What it does better than rg/grep:
      - mode="keyword" (sparse, BM25-like) is the grep replacement for exact lookups:
        symbols, error strings, config keys ("select_conn", "TokenBucket"). The same
        literal matching, but ranked by relevance and returned with the surrounding
        function, so one call usually replaces grep plus the follow-up file reads. Use it first whenever
        you know the token. It never substitutes semantic neighbours; if sparse vectors
        are unavailable it says so.
      - mode="hybrid" (default: semantic + keyword, RRF-fused) and mode="dense"
        (semantic only) find code by what it does, not what it is called: "where do we
        retry a failed upsert" finds the loop even when no word in it matches. Use them
        for concepts, unfamiliar code, or when keyword finds nothing; they recover
        matches grep cannot see. (Hybrid reports a dense-only fallback if an index
        predates sparse vectors; re-index to restore fusion.)
      - Results are ranked and capped (top_k, max_per_file), so a common token cannot
        flood the context the way grep over a large repository does.
    rg/grep remain right for files outside any indexed scope and for listing every
    occurrence exhaustively (search_code returns the best top_k, not all). Run
    index_code first to add a repository.

    Where to search:
      - repo_root: one indexed directory (default: the current directory "."), a list of
        them, or "*" for every scope you own.
      - sources: "private" (default, your indexed code) | "public" (the curated Public
        Code corpus) | "all" (both, fused by Reciprocal Rank Fusion or by rerank).
        "public" and "all" require admin tier; other callers get an explicit error,
        never a silent private-only result. Hits from several scopes or from "all" are
        tagged `[private]` or `[public:<source_id>@<rev>]`.
      - source_ids / domains: narrow the public side to curated repositories / domains.
      path_glob and max_per_file apply to your private code only.

    Args:
        query: What the code does, or an exact symbol for keyword mode.
        top_k: How many chunks to return, 1-50.
        repo_root: See "Where to search". A directory's absolute path is its scope — pass
            the same path you gave index_code. Windows drive/UNC paths accept either
            slash style and case; they canonicalize to one scope.
        mode: "keyword" | "hybrid" (default) | "dense" — see above.
        rerank: Cross-encoder re-scoring of candidates for higher precision, slower.
            Worth it when the top hits look almost-right but mis-ordered.
        language: Restrict to one tree-sitter language, e.g. "rust", "python", "go".
        path_glob: Narrow to a sub-directory or a single file inside the indexed repo,
            matched against each chunk's repo-relative path. Never drops results — the
            matching path-set is pushed down as a filter, so top_k fills only from matching
            files (no post-filter squeeze), on every mode. Accepts:
              • a subtree prefix — "easytier/src/peers/"
              • a single file rel path — "src/main.rs"
              • a glob — "*.rs", "src/**/*.go"
            Windows backslashes and casing are accepted. It is always scope-relative:
            use "conversion/**", never "E:\\repo\\conversion". Omit for a normal
            whole-repo search. Example: path_glob="easytier/src/peers/".
        max_per_file: Cap hits from any single file so one file's tests can't monopolise
            the top-k (e.g. 2). Default: no cap. Example: max_per_file=2.
        sources: See "Where to search".
        source_ids: Public side only: restrict to these curated repository ids.
        domains: Public side only: restrict to these domains.
        show_line_numbers: Prefix code with source line numbers; default True.
            Set False for plain code. This only affects display.

    Example: search_code(query="select_conn", mode="keyword", path_glob="easytier/src/peers/")
    Example: search_code(query="retry a failed NACK", sources="all", source_ids=["pion-webrtc"])
    """
    if sources not in _CODE_SOURCES:
        raise ValueError(
            f"search_code.sources must be one of {', '.join(_CODE_SOURCES)}; got {sources!r}")
    if sources == "private" and (source_ids or domains):
        raise ValueError(
            "search_code.source_ids/domains filter the public corpus; "
            "pass sources='public' or sources='all'")
    if sources == "public" and (path_glob or max_per_file):
        raise ValueError(
            "search_code.path_glob/max_per_file filter your private code; "
            "they cannot be used with sources='public'")
    several = isinstance(repo_root, list) or repo_root == "*"
    if sources == "private" and not several:
        return await _search_private_code(query, top_k, repo_root, mode, rerank,
                                          language, path_glob, max_per_file, show_line_numbers)
    try:
        if sources == "public":
            return await _search_public_code(query, source_ids, domains, top_k, mode,
                                             language, rerank, show_line_numbers)
        # An empty list or "*" = every scope you own (the federated route's default).
        roots = [] if repo_root == "*" else (
            list(repo_root) if isinstance(repo_root, list) else [repo_root])
        return await _search_federated_code(query, roots, sources == "all", source_ids,
                                            domains, top_k, mode, language, path_glob,
                                            max_per_file, rerank, show_line_numbers)
    except httpx.HTTPStatusError as exc:
        await _explain_unknown_public_selection(exc)
        raise


def _fmt_duplicate_pairs(resp: dict, *, scope: str, min_shared_chunks: int,
                         threshold: float) -> str:
    """Render a duplicate scan. File pairs, ranked by shared chunks — see
    core/code_kb/duplicate_rollup for why that ranking and not score."""
    pairs = resp.get("pairs") or []
    scanned_files = resp.get("scanned_files", 0)
    scanned_chunks = resp.get("scanned_chunks", 0)
    truncated = bool(resp.get("truncated"))

    lines: list[str] = []
    if not pairs:
        if truncated:
            lines.append(
                f"INCOMPLETE SCAN of {scope}: {resp.get('notice')} No duplicate file pairs "
                "were found in the part that was scanned, which is NOT the same as none "
                "existing.")
        else:
            lines.append(
                f"No duplicate file pairs in {scope} at threshold {threshold} with "
                f"min_shared_chunks={min_shared_chunks} "
                f"({scanned_files} files / {scanned_chunks} chunks scanned).")
            lines.append(
                "Lower min_shared_chunks to 1 to see single-chunk collisions, but expect "
                "mostly boilerplate: in the reference measurement 176 of 190 such pairs "
                "were not real duplication.")
        return "\n".join(lines)

    lines.append(
        f"{len(pairs)} duplicate file pair(s) in {scope} "
        f"(threshold {threshold}, min_shared_chunks {min_shared_chunks}; "
        f"{scanned_files} files / {scanned_chunks} chunks scanned):")
    for i, pair in enumerate(pairs, 1):
        lines.append("")
        lines.append(
            f"{i}. {pair.get('path_a')}  <->  {pair.get('path_b')}\n"
            f"   {pair.get('shared_chunks')} shared chunks, "
            f"max {pair.get('max_score', 0.0):.3f} / mean {pair.get('mean_score', 0.0):.3f}")
        for example in pair.get("examples") or []:
            a, b = example.get("a") or {}, example.get("b") or {}
            label = a.get("label") or b.get("label") or ""
            suffix = f"  [{label}]" if label else ""
            lines.append(
                f"     {a.get('path')}:{a.get('start_line')}-{a.get('end_line')}"
                f"  ~  {b.get('path')}:{b.get('start_line')}-{b.get('end_line')}"
                f"  ({example.get('score', 0.0):.3f}){suffix}")
    if truncated:
        lines.append("")
        lines.append(f"WARNING: {resp.get('notice')}")
    elif resp.get("notice"):
        lines.append("")
        lines.append(f"Note: {resp['notice']}")
    aliases = resp.get("scope_aliases") or []
    if aliases:
        lines.append("Included legacy Windows scope spelling(s): "
                     + ", ".join(repr(a) for a in aliases))
    return "\n".join(lines)


@mcp.tool()
async def find_duplicate_code(repo_root: str = ".", threshold: float = 0.92,
                              min_shared_chunks: int = 2, max_pairs: int = 50,
                              max_examples: int = 3,
                              exclude_globs: list[str] | None = None) -> str:
    """Find duplicated code and duplicated design in an indexed repository: pairs of
    FILES that contain the same logic, reported with `path:line` ranges you can diff.

    Use it before writing a new module ("does something like this already exist here?"),
    during review of a large change, or when planning a refactor. It compares what code
    MEANS, so it finds a re-implementation that shares no identifier with the original —
    which is exactly what a text search for duplicates cannot do.

    Run index_code on the repository first; this reads that index and never the disk, so
    the answer reflects the last index, not uncommitted edits.

    Why file pairs and not chunk pairs: a measurement over this tool's own codebase found
    that raising the threshold barely changes the result, because the noise is not
    weak matches — it is one-off boilerplate collisions. 176 of 190 candidate file pairs
    shared exactly ONE chunk (imports, dataclass headers, `if __name__ == "__main__"`).
    Requiring two shared chunks cut 47 findings to 14 with no false positives among them.
    So `min_shared_chunks` is the knob that matters; leave `threshold` alone unless you
    have a reason.

    Args:
        repo_root: The indexed directory to scan (default: current directory "."). Its
            absolute path is the scope — pass the same path you gave index_code.
        threshold: Similarity at which two chunks count as near-duplicates, 0-1.
            Below ~0.85 unrelated code starts matching.
        min_shared_chunks: How many shared chunks make a file pair a finding.
            Use 1 only when hunting a single copied function, and expect
            to read past a lot of boilerplate.
        max_pairs: Cap on reported file pairs, 1-500.
        max_examples: Shared chunk locations quoted per pair, 1-20.
        exclude_globs: Gitignore-syntax paths to skip. Omit for the defaults
            (`node_modules/`, `__pycache__/`, `archive/`, `/storage/`, ... ) which keep
            vendored and generated trees out. Pass [] to scan everything.

    Example: find_duplicate_code(repo_root="~/src/erp", min_shared_chunks=3)
    """
    t0 = time.perf_counter()
    proj = _scope_path(repo_root)
    logger.info(
        "find_duplicate_code.start project=%s threshold=%s min_shared=%s",
        proj, threshold, min_shared_chunks,
    )
    payload: dict[str, Any] = {
        "project": proj, "threshold": threshold,
        "min_shared_chunks": min_shared_chunks, "max_pairs": max_pairs,
        "max_examples": max_examples,
    }
    if exclude_globs is not None:
        payload["exclude_globs"] = exclude_globs
    try:
        resp = await asyncio.wait_for(
            _post("/api/code/duplicates", payload), timeout=_FIND_DUPLICATES_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("find_duplicate_code.timeout project=%s", proj)
        return (
            f"find_duplicate_code timed out after {_FIND_DUPLICATES_TIMEOUT_S:.0f}s "
            f"(project={proj!r}). The scan cost grows with indexed chunk count; narrow "
            "the scope to a subdirectory that is indexed on its own, or raise "
            "min_shared_chunks and retry."
        )
    out = _fmt_duplicate_pairs(
        resp, scope=proj, min_shared_chunks=min_shared_chunks, threshold=threshold)
    logger.info(
        "find_duplicate_code.done project=%s pairs=%d chunks=%d total_ms=%.0f",
        proj, len(resp.get("pairs") or []), resp.get("scanned_chunks", 0),
        (time.perf_counter() - t0) * 1000,
    )
    return out


@mcp.tool()
async def delete_code(repo_root: str = ".") -> str:
    """Delete an indexed code project ENTIRELY: drop all of its vectors from the shared
    code index AND remove its rows from your index inventory.

    This is the explicit cleanup path. Re-indexing an emptied directory does NOT clear a
    project — a full reindex of a source-less tree is a deliberate no-op (so a broken walk
    can't silently wipe a real index), so removal is always explicit. After this,
    search_code / index_inventory for the project return nothing.

    The scope is repo_root's ABSOLUTE PATH — pass the same path you gave index_code, so
    the same directory is targeted (never a hand-typed name).

    Args:
        repo_root: The indexed directory to clear (default: current directory "."). Its
            absolute path is the scope.
    """
    proj = _scope_path(repo_root)
    host = _env("RETRIEVAL_HOST", "RETRIVAL_HOST", default=socket.gethostname()) or ""
    key = (host, proj)
    async with _code_scope_operation_lock(key):
        preparation = _active_code_preparation(key)
        if preparation:
            task = _CODE_PREPARATION_TASKS.get(preparation[0])
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        _invalidate_refresh(proj)
        resp = await _delete("/api/code", {"project": proj})
    deleted = resp.get("deleted", 0)
    catalog = resp.get("catalog_removed", 0)
    aliases = resp.get("scope_aliases") or []
    if not deleted and not catalog:
        return (f"Nothing to delete for project '{proj}' — no indexed code or inventory "
                f"rows found (already clear, or never indexed).")
    out = (f"Deleted code project '{proj}': removed {deleted} indexed chunk(s) and "
           f"{catalog} inventory row(s). search_code for it now returns nothing.")
    if aliases:
        out += "\nIncluded legacy Windows scope spelling(s): " + ", ".join(repr(a) for a in aliases)
    return out


@mcp.tool()
async def delete_journal_docs(dir: str = ".") -> str:
    """delete_code for documents: remove every doc journal_index_dir indexed under a
    directory scope (entries, vectors and index_inventory rows). Memories written with
    journal_record in that scope are kept. Re-indexing never clears a scope, so this is
    the only way to remove one.

    Args:
        dir: The directory you gave journal_index_dir; its absolute path is the scope.
    """
    scope = _scope_path(dir)
    # The scope's watcher holds a document baseline that is about to stop being true (and
    # the backend drops the scope's document policy with its rows): let it re-resolve.
    _invalidate_refresh(scope)
    resp = await _delete("/api/journal/docs", {"project_slug": scope})
    removed = resp.get("removed_entries", 0)
    vectors = resp.get("deleted_vectors", 0)
    catalog = resp.get("catalog_removed", 0)
    if not removed and not catalog:
        return (f"Nothing to delete for doc scope '{scope}' — no indexed docs or inventory "
                f"rows found (already clear, or never indexed). ACE memories are untouched.")
    return (f"Deleted doc scope '{scope}': removed {removed} doc entry(ies) "
            f"({vectors} vector(s)) and {catalog} inventory row(s). ACE memories kept.")


# ------------------------------ ACE playbook (drop-in ace-mcp) ----------------------- #
# These 5 tools mirror angry-robot-deals/ace-mcp's signatures, backed by the server-side
# per-user playbook (/api/playbook/*, W144). ACE evolves *context* (curated lessons); it
# has NO code-generation model — these tools curate/retrieve guidance, the CALLING agent
# does the generating. Scoped to the cwd project so lessons don't leak across projects.

def _fmt_bullets(bullets: list[dict], header: str) -> str:
    if not bullets:
        return f"{header}\n(none yet — record lessons with ace_smart_reflect.)"
    lines = [header, ""]
    for i, b in enumerate(bullets, 1):
        tag = f"[{b.get('section', 'insights')}]" + (f"[{b.get('domain')}]" if b.get("domain") else "")
        sc = b.get("score")
        sim = f" sim={sc:.3f}" if isinstance(sc, (int, float)) else ""
        lines.append(f"{i}. {tag} {b.get('content', '')}  "
                     f"(+{b.get('helpful', 0)}/-{b.get('harmful', 0)}{sim})")
    return "\n".join(lines)


async def _playbook_search(query: str, top_k: int, domain: str | None) -> list[dict]:
    resp = await _post("/api/playbook/search", {"project_slug": _scope_path("."),
                                                "query": query, "top_k": top_k, "domain": domain})
    return resp.get("bullets") or []


@mcp.tool()
async def ace_context_aware(query: str, domain: str | None = None, top_k: int = 5) -> str:
    """ACE: retrieve the playbook bullets most relevant to a query — the Generator's
    context-retrieval step. Returns reusable lessons/strategies/pitfalls previously
    curated for THIS project, ranked by semantic relevance. Empty until you record
    lessons with ace_smart_reflect.

    Args:
        query: What you're about to do (a task or question).
        domain: Optional tag to scope to one area, e.g. "qdrant", "frontend" (default: all).
        top_k: How many bullets to return.
    """
    bullets = await _playbook_search(query, top_k, domain)
    return _fmt_bullets(bullets, f"Playbook context for: {query!r}")


@mcp.tool()
async def ace_enhance_prompt(prompt: str, focus_area: str | None = None) -> str:
    """ACE: augment a prompt with the project's most relevant playbook bullets, so
    the calling model has accumulated lessons in-context before it works. Returns the
    original prompt followed by a guidelines block — it does NOT call any LLM itself.

    Args:
        prompt: The prompt you intend to run.
        focus_area: Optional text to bias retrieval (default: the prompt itself).
    """
    bullets = await _playbook_search(focus_area or prompt, 5, None)
    if not bullets:
        return prompt
    guide = "\n".join(f"- {b.get('content', '')}" for b in bullets)
    return f"{prompt}\n\n# Playbook guidelines (learned for this project)\n{guide}"


@mcp.tool()
async def ace_smart_generate(prompt: str, context: str | None = None,
                             auto_enhance: bool = True) -> str:
    """ACE: return playbook-grounded GUIDANCE for a task — the lessons/strategies the
    calling agent should apply when it generates.

    Note: ACE evolves *context*, it has no code-generation model of its own. This
    returns grounded guidance plus an enhanced prompt for YOUR model to generate
    from; it does not itself write code.

    Args:
        prompt: The task you're about to do.
        context: Optional extra context to bias which lessons are retrieved (default: the prompt).
        auto_enhance: If true (default), retrieve and attach relevant playbook bullets.
    """
    bullets = await _playbook_search(context or prompt, 6, None) if auto_enhance else []
    guidance = _fmt_bullets(bullets, "Apply these learned playbook bullets:")
    enhanced = prompt
    if bullets:
        guide = "\n".join(f"- {b.get('content', '')}" for b in bullets)
        enhanced = f"{prompt}\n\n# Playbook guidelines\n{guide}"
    return f"{guidance}\n\n--- enhanced prompt for your model ---\n{enhanced}"


@mcp.tool()
async def ace_smart_reflect(code: str, auto_suggest: bool = True,
                            section: str = "insights", domain: str | None = None) -> str:
    """ACE: reflect on an outcome and curate a transferable lesson into the playbook
    — the Reflector→Curator step. Pass a short, reusable lesson as `code` (e.g.
    "Always wait=True on Qdrant upsert or batches drop silently under load").

    Near-duplicate lessons reinforce the existing bullet (grow-and-refine) instead of
    duplicating, preventing context collapse. No LLM is called here — the lesson you
    pass is what gets stored.

    Args:
        code: The lesson / observation to record (required; name kept for drop-in compat).
        auto_suggest: Accepted for drop-in compatibility; no LLM suggestion is run.
        section: patterns | best_practices | insights | failure_mode.
        domain: Optional tag to group the lesson, e.g. "qdrant", "frontend" (default: none).

    Example: ace_smart_reflect(code="Always wait=True on Qdrant upsert or batches drop
    silently under load", section="failure_mode", domain="qdrant")
    """
    lesson = (code or "").strip()
    if not lesson:
        return "Nothing to reflect: empty input."
    b = await _post("/api/playbook/bullets", {"project_slug": _scope_path("."),
                                              "content": lesson, "section": section, "domain": domain})
    verb = "reinforced existing" if b.get("deduped") else "added new"
    return (f"Playbook {verb} bullet [{b.get('section', section)}] id={b.get('id', '?')} "
            f"(+{b.get('helpful', 0)}/-{b.get('harmful', 0)}): {b.get('content', lesson)}")


@mcp.tool()
async def ace_playbook(domain: str | None = None, limit: int = 50) -> str:
    """ACE: view the current project playbook — every curated bullet with its
    helpful/harmful counters. Use it to inspect what the project has learned.

    Args:
        domain: Optional tag filter, e.g. "qdrant" (default: all domains).
        limit: Max bullets to list.
    """
    params = {k: v for k, v in {"project_slug": _scope_path("."),
                                "domain": domain, "limit": limit}.items() if v is not None}
    resp = await _get("/api/playbook/bullets", params)
    bullets = resp.get("bullets") or []
    count = resp.get("total") or len(bullets)
    return _fmt_bullets(bullets, f"Playbook ({count} bullets):")


def _install_multi_query_tools() -> None:
    """Replace only the five query schemas; their single-query functions stay intact."""
    single_search_papers = search_papers
    single_search_arxiv = search_arxiv
    single_search_within_paper = search_within_paper
    single_journal_search = journal_search
    single_search_code = search_code

    async def multi_search_papers(
        query: str | list[str], top_k: int = 10, venue: str | None = None,
        year: int | None = None, title_only: bool = False,
        source_type: str | None = None,
    ) -> str:
        return await _run_query_batch(
            query, tool_name="search_papers",
            invoke=lambda item: single_search_papers(
                item, top_k, venue, year, title_only, source_type),
        )

    async def multi_search_arxiv(
        query: str | list[str], top_k: int = 10, year: int | None = None,
        include_withdrawn: bool = False,
    ) -> str:
        return await _run_query_batch(
            query, tool_name="search_arxiv",
            invoke=lambda item: single_search_arxiv(item, top_k, year, include_withdrawn),
        )

    async def multi_search_within_paper(
        paper_id: str, query: str | list[str], top_k: int = 20,
    ) -> str:
        return await _run_query_batch(
            query, tool_name="search_within_paper",
            invoke=lambda item: single_search_within_paper(paper_id, item, top_k),
        )

    async def multi_journal_search(
        query: str | list[str], top_k: int = 5, mode: str = "keyword",
        dedup_by_entry: bool = True, chunk_limit: JournalChunkCount = 3,
        scope_dir: str = ".", path_glob: str | None = None,
    ) -> str:
        return await _run_query_batch(
            query, tool_name="journal_search",
            invoke=lambda item: single_journal_search(
                item, top_k, mode, dedup_by_entry, chunk_limit, scope_dir, path_glob),
        )

    async def multi_search_code(
        query: str | list[str], top_k: int = 8, repo_root: str | list[str] = ".",
        mode: str = "hybrid", rerank: bool = False, language: str | None = None,
        path_glob: str | None = None, max_per_file: int | None = None,
        sources: str = "private", source_ids: list[str] | None = None,
        domains: list[str] | None = None, show_line_numbers: bool = True,
    ) -> str:
        return await _run_query_batch(
            query, tool_name="search_code",
            invoke=lambda item: single_search_code(
                item, top_k, repo_root, mode, rerank, language, path_glob, max_per_file,
                sources, source_ids, domains, show_line_numbers),
        )

    replacements = (
        ("search_papers", single_search_papers, multi_search_papers),
        ("search_arxiv", single_search_arxiv, multi_search_arxiv),
        ("search_within_paper", single_search_within_paper, multi_search_within_paper),
        ("journal_search", single_journal_search, multi_journal_search),
        ("search_code", single_search_code, multi_search_code),
    )
    for name, single, replacement in replacements:
        mcp.remove_tool(name)
        mcp.add_tool(
            replacement, name=name,
            description=_multi_query_tool_description(single.__doc__ or ""),
        )
        globals()[name] = replacement


_install_multi_query_tools()

# Published-schema compaction (the same code lives in mcp_server/server.py; the
# plugin/repository parity test compares the two surfaces byte for byte).
# Pydantic titles every schema node ("top_k" -> "Top K", the arguments model ->
# "multi_search_codeArguments"): ~4K characters across the surface that repeat
# the parameter names, loaded into every client session. Validation runs on the
# pydantic argument model, not on this advertised dict, so dropping them changes
# nothing a call can do.
# Every tool returns a formatted `str`, so FastMCP's automatic structured output
# was an outputSchema of {"result": string} on all 36 tools (~5K characters) and
# every call result sent twice: once as text, once as {"result": <same text>}.
# Tools are re-registered with structured_output=False to publish text only.
_MULTI_QUERY_PARAMETER_DESCRIPTION = (
    "One query string, or a list of independent queries run with the same other "
    "arguments; results keep input order and are labeled by query."
)
_SCHEMA_CHILD_MAPS = ("properties", "$defs", "definitions", "patternProperties")
_SCHEMA_CHILD_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_CHILD_NODES = ("items", "additionalProperties", "not")


def _drop_schema_titles(node: Any) -> None:
    """Remove the `title` keyword from a JSON schema and its sub-schemas.

    Only schema positions are walked, so a PARAMETER named "title" (a key under
    `properties`) is kept."""
    if not isinstance(node, dict):
        return
    if isinstance(node.get("title"), str):
        del node["title"]
    for key in _SCHEMA_CHILD_MAPS:
        for child in (node.get(key) or {}).values():
            _drop_schema_titles(child)
    for key in _SCHEMA_CHILD_LISTS:
        for child in node.get(key) or []:
            _drop_schema_titles(child)
    for key in _SCHEMA_CHILD_NODES:
        _drop_schema_titles(node.get(key))


def _compact_published_tool_schemas(multi_query_names: frozenset[str]) -> None:
    """Final pass over every registered tool: text-only output, compact input schema.

    The list-query contract used to be appended to six descriptions; it now sits
    once in the server instructions and, per tool, on the `query` parameter."""
    for old in list(mcp._tool_manager.list_tools()):
        mcp.remove_tool(old.name)
        mcp.add_tool(
            old.fn, name=old.name, title=old.title, description=old.description,
            annotations=old.annotations, icons=old.icons, meta=old.meta,
            structured_output=False,
        )
    for tool in mcp._tool_manager.list_tools():
        _drop_schema_titles(tool.parameters)
        if tool.name in multi_query_names:
            tool.parameters["properties"]["query"]["description"] = (
                _MULTI_QUERY_PARAMETER_DESCRIPTION)


_compact_published_tool_schemas(frozenset({
    "search_papers", "search_arxiv", "search_within_paper", "journal_search", "search_code",
}))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full server only for daemon ``serve`` or explicit debugging stdio."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        # Compatibility for stale console entry points that still import this
        # module.  New package entry points use retrieval_mcp_cli.py and avoid
        # importing this file at all for the default stdio proxy path.
        from retrieval_mcp_daemon import DEFAULT_PLUGIN_BACKEND_URL, run_thin_cli

        return run_thin_cli(
            [],
            serve_command=[__file__],
            backend_default_url=DEFAULT_PLUGIN_BACKEND_URL,
        )
    if args[0] == "stdio-server":
        if len(args) != 1:
            raise SystemExit("stdio-server accepts no arguments")
        mcp.run()
        return 0
    if args[0] == "serve":
        import argparse
        from retrieval_mcp_daemon import (
            DEFAULT_PLUGIN_BACKEND_URL,
            resolve_runtime_dir,
            serve_daemon,
        )

        parser = argparse.ArgumentParser(prog="retrieval-mcp serve")
        parser.add_argument("serve")
        parser.add_argument("--state-dir")
        parser.add_argument("--instance-token", required=True)
        parsed = parser.parse_args(args)
        return serve_daemon(
            mcp,
            runtime_dir=resolve_runtime_dir(parsed.state_dir),
            instance_token=parsed.instance_token,
            backend_default_url=DEFAULT_PLUGIN_BACKEND_URL,
        )
    from retrieval_mcp_daemon import DEFAULT_PLUGIN_BACKEND_URL, run_thin_cli

    return run_thin_cli(
        args,
        serve_command=[__file__],
        backend_default_url=DEFAULT_PLUGIN_BACKEND_URL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
