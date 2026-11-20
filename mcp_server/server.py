"""MCP server exposing retrival's academic-paper retrieval engine.

Run it as a subprocess of any MCP client (Claude Code, Cursor, Codex, ...). It
forwards each tool call to the running FastAPI backend over HTTP, so it reuses the
warm BGE-M3 model + cross-encoder reranker and the 12.7M-chunk Qdrant index
without loading anything onto the GPU itself.

Config (env):
  RETRIEVAL_API_URL  backend base URL (default http://127.0.0.1:8000; legacy RETRIVAL_API_URL still honored)

Run:
  python -m mcp_server.server          # stdio transport (for MCP clients)
"""
from __future__ import annotations

# Keep the repository entry point on the same lightweight default path as the
# published package.  Only ``serve`` and explicit ``stdio-server`` import this
# module's FastMCP tool registrations in the process that receives the command.
if __name__ == "__main__":
    import sys as _bootstrap_sys

    _bootstrap_args = _bootstrap_sys.argv[1:]
    if not _bootstrap_args or _bootstrap_args[0] in {"start", "status", "stop"}:
        from mcp_plugin.retrieval_mcp_daemon import (
            DEFAULT_REPOSITORY_BACKEND_URL as _DEFAULT_REPOSITORY_BACKEND_URL,
            run_thin_cli as _run_thin_cli,
        )

        raise SystemExit(_run_thin_cli(
            _bootstrap_args,
            serve_command=["-m", "mcp_server.server"],
            backend_default_url=_DEFAULT_REPOSITORY_BACKEND_URL,
        ))
    if _bootstrap_args[0] == "serve":
        from mcp_plugin.retrieval_mcp_daemon import redirect_daemon_output_from_environment as _redirect_daemon_output

        _redirect_daemon_output()

import asyncio
import contextvars
import dataclasses
import fnmatch
import glob
import inspect
import logging
import math
import ntpath
import os
import posixpath
import re
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Awaitable, Callable, NamedTuple, Sequence

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from watchfiles import awatch

from core.code_kb.source_policy import DOC_EXTS, DOC_GLOBS
from core.code_kb.source_policy import source_extension as _ext_of
from core.indexing.file_snapshot import FileCandidate, ScanSpec, SnapshotDiff

from mcp_server.backend import (
    RetrievalClient,
    _number_code_lines,
    _INLINE_DOCUMENT_MARKDOWN_MAX_BYTES,
    _document_id,
    _is_document_id,
    format_paper_batch_download,
    format_inventory,
    format_journal_recent,
    format_journal_record,
    format_journal_search,
    format_match_results,
    format_public_code_job,
    format_public_code_search,
    format_public_code_sources,
    format_search_results,
    format_stats,
    _format_public_import,
    _inspect_public_clone,
    _resolve_public_import_source,
    _run_public_import,
    _format_inventory_summary,
)
from mcp_server.playbook import Bullet, PlaybookStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

def _env(*names: str, default: str | None = None) -> str | None:
    """First set value among `names` (new spelling first, legacy fallback), else default."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


BASE_URL = _env("RETRIEVAL_API_URL", "RETRIVAL_API_URL", default="http://127.0.0.1:8000")
# sk-key for the paid gateway. Set when RETRIEVAL_API_URL points at the gateway
# (not the 4090 directly); sent as Bearer on every call so searches are billed to
# this user. Unset => talking straight to the compute box (no auth).
# (Legacy misspelled RETRIVAL_* names are still honored as a fallback.)
API_KEY = _env("RETRIEVAL_API_KEY", "RETRIVAL_API_KEY")
# Project-local playbook with per-project isolation. Override
# per client via env so each project's MCP launch keeps its own evolving playbook.
ACE_DB = os.environ.get("ACE_PLAYBOOK_DB", os.path.join(".ace", "playbook.db"))
# Interactive search tools must not wait for the full bulk-read window; host tool
# timeouts are often multi-minute and feel like a hang (search_code 2026-07-23;
# search_papers concurrent hang 2026-07-25).
_SEARCH_CODE_TIMEOUT_S = 45.0
# A duplicate scan issues one nearest-neighbour query per indexed chunk: ~29s for
# 8.6k chunks when measured on 2026-09-01. It is a deliberate, explicitly invoked
# analysis, not an interactive lookup, so it gets a much longer budget than search.
_FIND_DUPLICATES_TIMEOUT_S = 300.0
_SEARCH_PAPERS_TIMEOUT_S = 60.0
# Cap MCP stdio payload: normal AST chunks stay complete; pathological multi-KB
# one-liners and multi-hit dumps are hard-capped so the host tool call returns.
_CODE_HIT_CODE_CHARS = 4000
_CODE_HIT_TOTAL_CHARS = 24000

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
_client = RetrievalClient(base_url=BASE_URL, api_key=API_KEY)


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


def _format_nested_scope_warning(project: str, kind: str, descendants: list[dict[str, str]]) -> str:
    frag = _format_conflict_list(descendants, lambda c: f"child scope '{c['project']}'")
    return (f"Warning: existing {frag} under {kind} scope '{project}'. Parent indexing "
            f"proceeded because it can add files outside the child scope.")


def _format_gitignore_widening_warning(project: str, kind: str,
                                       conflicts: list[dict[str, str]]) -> str:
    frag = _format_conflict_list(conflicts, lambda c: f"scope '{c['project']}'")
    return (f"Warning: ancestor {frag} already indexes part of '{project}'. Indexing "
            f"proceeded because respect_gitignore=False reaches files the parent skipped, "
            f"but the non-ignored files in this subtree now exist in both {kind} scopes.")


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
        overlap = await _client.scope_overlap(
            project=project, host=host, kind=kind, respect_gitignore=respect_gitignore)
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


def _flatten_inventory_doc_files(resp: dict) -> dict[str, dict]:
    """Return this detail inventory's Journal-doc rows keyed by normalized path."""
    out: dict[str, dict] = {}
    for user in resp.get("users") or []:
        for host in user.get("hosts") or []:
            for project in host.get("projects") or []:
                for directory in project.get("dirs") or []:
                    for file in directory.get("files") or []:
                        if file.get("kind") == "doc" and file.get("rel_path"):
                            out[file["rel_path"].replace("\\", "/")] = file
    return out


@mcp.tool()
async def search_papers(
    query: str,
    top_k: int = 10,
    venue: str | None = None,
    year: int | None = None,
    title_only: bool = False,
    source_type: str | None = None,
) -> str:
    """Search 95k+ top-venue CS papers (NeurIPS/ICML/ICLR/CVPR/ACL/KDD/CCS/NDSS/…)
    by meaning, returning the most relevant papers with a snippet and paper_id.

    This is semantic hybrid retrieval (BGE-M3 dense+sparse + cross-encoder rerank),
    NOT keyword search — phrase the query as a full, specific description (a complete
    question, a method sentence, or a concept combination), not a single keyword.

    Args:
        query: A detailed natural-language description of what you're looking for (required).
        top_k: How many papers to return, 1-50 (default 10).
        venue: Optional venue filter, e.g. "NeurIPS", "CVPR", "ACL" (default: all venues).
        year: Optional year filter, e.g. 2024 (default: all years).
        title_only: If true, match only titles + abstracts — good for a first broad
            survey to scope candidate papers without body/appendix noise (default false).
            Only meaningful for source_type="paper"; uploads and webpages have no title
            chunk, so pairing it with them returns cover pages at score 0.
        source_type: Restrict to one source kind: "paper" (the CS corpus), "upload"
            (uploaded documents — whole books live here), "webpage", or "review".
            Default: every source. Papers outnumber the rest by orders of magnitude, so
            an uploaded book essentially never reaches top_k on a general topical query
            unless you pass source_type="upload".

    Returns the paper_id of each hit; pass it to search_within_paper or match_paper.
    Example: search_papers(query="mixture-of-experts routing stability", venue="ICLR",
                           year=2024, top_k=5)
    """
    t0 = time.perf_counter()
    logger.info(
        "search_papers.start query=%r top_k=%s venue=%r year=%s title_only=%s source_type=%r",
        query, top_k, venue, year, title_only, source_type,
    )
    try:
        resp = await asyncio.wait_for(
            _client.search_papers(
                query=query, top_k=top_k, venue=venue, year=year, title_only=title_only,
                source_type=source_type),
            timeout=_SEARCH_PAPERS_TIMEOUT_S,
        )
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
    out = format_search_results(resp)
    papers = resp.get("papers") or []
    logger.info(
        "search_papers.done papers=%d out_chars=%d total_ms=%.0f",
        len(papers), len(out), (time.perf_counter() - t0) * 1000,
    )
    return out


@mcp.tool()
async def search_arxiv(query: str, top_k: int = 10, year: int | None = None,
                       include_withdrawn: bool = False) -> str:
    """Semantic hybrid search over arXiv computer-science preprints.

    search_papers covers peer-reviewed venues and leaves preprints out; use this tool
    for recent arXiv work that may not be published anywhere yet. Phrase the query as
    a full, specific description, not a bare keyword. Returns title + year + paper_id
    ("arxiv-<arXiv id>", e.g. "arxiv-2606.32000") + best snippet; feed the paper_id
    into search_within_paper / get_paper_markdown.

    Args:
        query: Natural-language description of what you're looking for (required).
        top_k: How many papers to return (default 10).
        year: Optional year filter, e.g. 2026 (default: all years).
        include_withdrawn: Withdrawn papers are left out by default; true brings them
            back, marked [WITHDRAWN]. search_within_paper on a withdrawn paper_id works
            either way.

    Example: search_arxiv(query="test-time compute scaling for reasoning language models",
                          year=2026, top_k=5)
    """
    t0 = time.perf_counter()
    logger.info("search_arxiv.start query=%r top_k=%s year=%s include_withdrawn=%s",
                query, top_k, year, include_withdrawn)
    try:
        # venue_tier="preprint" scopes /api/search to arXiv (see search_papers'
        # default preprint exclusion in core/retrieval/pipeline.retrieve_filters).
        resp = await asyncio.wait_for(
            _client.search_papers(query=query, top_k=top_k, year=year, venue_tier="preprint",
                                  include_withdrawn=include_withdrawn),
            timeout=_SEARCH_PAPERS_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return (
            f"search_arxiv timed out after {_SEARCH_PAPERS_TIMEOUT_S:.2f} seconds "
            f"(top_k={top_k}); retry once with a smaller top_k."
        )
    out = format_search_results(resp)
    logger.info("search_arxiv.done papers=%d total_ms=%.0f",
                len(resp.get("papers") or []), (time.perf_counter() - t0) * 1000)
    return out


@mcp.tool()
async def search_within_paper(paper_id: str, query: str, top_k: int = 20) -> str:
    """Search inside ONE paper and return every matching passage with its score,
    so you can see which parts of that specific paper address the query.

    Args:
        paper_id: The id exactly as search_papers printed it (required). Uploads
            and webpages have no paper_id and come back as "doc:<number>".
        query: What to look for inside this paper (required).
        top_k: Max passages to return (default 20).

    Example: search_within_paper(paper_id="neurips-2023-12345",
                                 query="ablation on batch size", top_k=10)
    Example: search_within_paper(paper_id="doc:114854",
                                 query="evidence lower bound derivation")
    """
    resp = await _client.search_within_paper(paper_id, query, top_k=top_k)
    return format_search_results(resp)


@mcp.tool()
async def get_paper_markdown(
    paper_id: str,
    save_path: str | None = None,
    overwrite: bool = False,
) -> str:
    """Get one paper's COMPLETE extracted Markdown, or save it on this MCP host.

    Unlike search_within_paper, this does not retrieve selected chunks and does not
    truncate the text. With no save_path it returns the complete Markdown verbatim.
    With save_path it writes the UTF-8 file locally and returns a compact receipt.

    Args:
        paper_id: The id exactly as search_papers printed it (required). Papers use
            their canonical paper_id; uploaded documents and webpages come back as
            "doc:<number>" — pass that through verbatim.
        save_path: Optional local file or directory. A directory receives
            <paper_id>.md. Omit to return the full Markdown instead of writing a file.
        overwrite: Replace an existing destination only when true (default false).

    An uploaded book runs to millions of characters, so for a "doc:" id the
    inline form refuses above a size limit and tells you to pass save_path.

    Example: get_paper_markdown(paper_id="icml-2026-example")
    Example: get_paper_markdown(paper_id="icml-2026-example", save_path="./papers/")
    Example: get_paper_markdown(paper_id="doc:114854", save_path="./books/")
    """
    if save_path is None:
        # Size first: an uploaded book inline would bury the caller's context
        # (the Murphy PML volume is 4,716,048 bytes). Papers stay uncapped.
        if _is_document_id(paper_id):
            size = await _client.document_markdown_size(_document_id(paper_id))
            if size is not None and size > _INLINE_DOCUMENT_MARKDOWN_MAX_BYTES:
                return (
                    f"Document '{paper_id}' has {size:,} bytes of Markdown, over the "
                    f"{_INLINE_DOCUMENT_MARKDOWN_MAX_BYTES:,}-byte inline limit. "
                    f"Re-run with save_path to write it to a file, or use "
                    f"search_within_paper(paper_id=\"{paper_id}\", query=...) to pull "
                    f"only the passages you need."
                )
        payload = await _client.get_paper_markdown(paper_id)
        return str(payload.get("content") or "")
    saved = await _client.save_paper_markdown(
        paper_id, save_path=save_path, overwrite=overwrite)
    return f"Saved Markdown for paper '{paper_id}' to '{saved['path']}' ({saved['size']} bytes)."


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
        paper_id: Canonical paper ID from search_papers (required).
        save_path: Local file or directory. Omit to save <paper_id>.pdf in the current
            working directory; a directory receives that default filename.
        overwrite: Replace an existing destination only when true (default false).
    """
    saved = await _client.download_paper_pdf(
        paper_id, save_path=save_path, overwrite=overwrite)
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
        artifacts: pdf, markdown, or both (default pdf).
        overwrite: Replace existing artifact files only when true (default false).

    Example: download_papers(
        paper_ids=["icml-2026-paper-a", "icml-2026-paper-b"],
        save_dir="./papers/", artifacts="both")
    """
    downloaded = await _client.download_papers(
        paper_ids,
        save_dir=save_dir,
        artifacts=artifacts,
        overwrite=overwrite,
    )
    return format_paper_batch_download(downloaded)


@mcp.tool()
async def match_document(text: str, top_k: int = 10) -> str:
    """Given a passage of text (e.g. an abstract, a related-work paragraph, or your
    own draft), find the corpus papers whose CONTENT is most similar — useful for
    "what prior work is closest to this?" and related-work discovery.

    The text is split into chunks, each retrieved independently, then results are
    aggregated per paper by coverage (how much of your text each paper overlaps).

    Args:
        text: The document/passage to match against the corpus (required).
        top_k: How many similar papers to return (default 10).

    Example: match_document(text="We propose a diffusion model that ...", top_k=15)
    """
    resp = await _client.match_document(text, top_k=top_k)
    return format_match_results(resp)


@mcp.tool()
async def match_paper(paper_id: str, top_k: int = 10) -> str:
    """Queue an exact, durable content-neighbour match for an indexed paper.

    The full query chunks, candidate depth and rerank pairs are retained. The call
    returns quickly with a job id; poll match_paper_status for progress/final results.

    Args:
        paper_id: The seed paper, from a search_papers result (required).
        top_k: How many neighbours to return (default 10).

    Example: match_paper(paper_id="cvpr-2024-00042", top_k=10)
    """
    receipt = await _client.match_paper(paper_id, top_k=top_k)
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
        task_id: Numeric job id returned by match_paper (required).
        cancel: Request cancellation before reading status (default false).

    Example: match_paper_status(task_id=123)
    """
    task = await _client.match_paper_status(task_id, cancel=cancel)
    status = str(task.get("status") or "unknown")
    result = task.get("result")
    if status == "completed" and isinstance(result, dict):
        return format_match_results(result)
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
    """List the conference/journal venues in the corpus, with their CCF tier and
    the years available. Use this to know which `venue` values are valid.

    No arguments. Example: list_conferences()
    """
    resp = await _client.list_conferences()
    confs = resp.get("conferences") or []
    lines = [f"{len(confs)} venues:"]
    for c in confs:
        years = c.get("years") or []
        yr = f"{min(years)}-{max(years)}" if years else "?"
        lines.append(f"- {c.get('title', c.get('sub', '?'))} "
                     f"[{c.get('venue_tier', '?')}] {yr}")
    return "\n".join(lines)


@mcp.tool()
async def corpus_stats() -> str:
    """Report corpus size: number of papers, text chunks, indexed vectors, and the
    embedding model in use.

    No arguments. Example: corpus_stats()
    """
    resp = await _client.corpus_stats()
    return format_stats(resp)


@mcp.tool()
async def journal_record(
    body: str,
    title: str | None = None,
    source_kind: str = "memory",
    tags: list[str] | None = None,
    scope_dir: str = ".",
) -> str:
    """Record this session's detailed work memory into the authenticated journal so it
    is searchable later via journal_search.

    The journal namespace is `scope_dir`'s absolute path (default: current working
    directory), so this project's memory stays isolated from other projects' — no
    hand-typed name to get wrong or fragment.

    Args:
        body: The memory content — be specific (what changed, root cause, how verified).
            Required.
        title: Optional short title (default: none).
        source_kind: "memory" (default; a work note that accumulates) or "doc" (a file's
            current content, latest-wins on re-record).
        tags: Optional list of tags for filtering, e.g. ["ops","deploy"].
        scope_dir: Directory whose absolute path is the journal namespace. Pass the
            same dir used with journal_index_dir to search/record in that indexed scope.

    Requires RETRIEVAL_API_URL to point at the public gateway or an authenticated
    compute path, plus RETRIEVAL_API_KEY when using sk-key auth.
    Example: journal_record(body="Switched 4090 backend to systemd user service + GPU2;
    verified 78 routes live.", title="systemd deploy", tags=["ops","deploy"])
    """
    project = _scope_path(scope_dir)
    resp = await _client.record_journal(
        project_slug=project,
        client="mcp",
        source_kind=source_kind,
        title=title,
        body=body,
        tags=tags or [],
    )
    return format_journal_record(resp)


@mcp.tool()
async def journal_search(
    query: str,
    top_k: int = 5,
    mode: str = "keyword",
    dedup_by_entry: bool = True,
    chunk_limit: JournalChunkCount = 3,
    scope_dir: str = ".",
    path_glob: str | None = None,
) -> str:
    """Search authenticated journal memory and indexed document content. Use this tool
    before shell text search such as rg or grep for indexed documents. Fall back when the
    content is not indexed, this tool cannot express the query, or the MCP request fails.

    Scoped to `scope_dir` (by absolute path; default current working directory) so
    other projects' notes don't leak in. Pass the same dir used with journal_index_dir
    to search that directory's indexed docs.

    Args:
        query: What to look for. Free text.
            Example: "qdrant upsert wait flag".
        top_k: Max entries to return (default 5). Example: top_k=10.
        mode: Retrieval engine.
            "keyword" (default) = FTS5 literal terms, needs NO embedding — always works
                even when the embed endpoint is down; best for exact tokens / paths / error
                codes / W-numbers.
            "hybrid" = dense + sparse fused (best semantic recall; needs embed endpoint up).
            "dense" = pure vector similarity (paraphrase / concept match).
            "sparse" = lexical vector only.
            Example: mode="hybrid".
        dedup_by_entry: One best hit per entry (default True). Set False to see multiple
            matching chunks from the same note. Example: dedup_by_entry=False.
        chunk_limit: NUMBER OF CHUNKS returned per entry, not characters/tokens.
            Valid 1-20; default 3. Use chunk_limit=20 for the maximum.
        scope_dir: Directory whose absolute path is the journal namespace (default "."
            = current working directory). Pass the SAME dir you indexed with
            journal_index_dir. Example: scope_dir="C:/Users/Arc/Prd/retrival".
        path_glob: Narrow the search to a sub-directory or a single file INSIDE an
            already-indexed parent scope, matched against each doc's scope-relative path
            (its title). Works on ALL modes (dense/hybrid/sparse/keyword) and never drops
            results — the match-set is resolved and pushed down as a filter, so top_k fills
            only from matching docs (no post-filter squeeze). Accepts:
              • a subtree prefix — path_glob="notes/2026/" (everything under that folder)
              • a single file rel path — path_glob="notes/a.md"
              • a glob — path_glob="*.md" or path_glob="notes/**/*.md"
            Omit it (default) for a normal whole-scope search (zero added cost).
            Example: journal_search(query="bandwidth limiter", scope_dir="C:/repo",
                                    path_glob="docs/design/")
    """
    chunk_limit = _validate_journal_chunk_count(chunk_limit)
    project = _scope_path(scope_dir)
    normalized_path_glob = _normalize_path_glob(path_glob) if path_glob else None
    resp = await _client.search_journal(
        project_slug=project,
        query=query,
        top_k=top_k,
        mode=mode,
        dedup_by_entry=dedup_by_entry,
        include_chunks=True,
        chunk_limit=chunk_limit,
        path_glob=normalized_path_glob,
    )
    notice = resp.get("scope_notice")
    if not (resp.get("hits") or []):
        guidance = _format_scope_miss(
            result_kind="journal entries",
            query=query,
            scope=project,
            scope_argument="scope_dir",
            search_tool="journal_search",
            next_step=f"journal_index_dir(dir={project!r})",
            path_glob=normalized_path_glob,
        )
        return _with_scope_notice(
            f"{resp['notice']}\n\n{guidance}" if resp.get("notice") else guidance, notice)
    return _with_scope_notice(format_journal_search(resp, max_chunks=chunk_limit), notice)


@mcp.tool()
async def journal_recent(limit: int = 10, scope_dir: str = ".") -> str:
    """List recent authenticated journal entries for the current project (newest first).

    Scoped to `scope_dir`'s absolute path (default: current working directory).

    Args:
        limit: How many entries to list (default 10).
        scope_dir: Directory whose absolute path is the journal namespace.

    Example: journal_recent(limit=5)
    """
    project = _scope_path(scope_dir)
    resp = await _client.recent_journal_entries(project_slug=project, limit=limit)
    return format_journal_recent(resp)


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

    Batch-index every matching text file in a LOCAL directory into the journal,
    then register them in your index inventory — the bulk counterpart to journal_record.

    Runs where THIS server runs (your machine): reads each file locally, sends its
    CONTENT to the backend as a journal entry (so the full text is searchable via
    journal_search), and reports the file list to your inventory (so index_inventory
    shows them). Non-recursive by default.

    Re-run safety: files are indexed latest-wins (source_kind="doc"). Re-indexing an
    UNCHANGED file is a no-op; an EDITED file supersedes its prior version so search
    returns only the file's CURRENT content, never stale accumulated copies (exactly
    one live version per file). The inventory row is upserted by relative path.

    Example: journal_index_dir(dir="~/Prd/retrival", pattern="*.md") — index the repo's
    root Markdown files. The journal namespace is the indexed dir's absolute path, so
    re-indexing the same directory always lands in one scope (no name-variant duplicates).

    Args:
        dir: Directory to index (default: current directory).
        recursive: Recurse into subdirectories (default: false — root files only).
            Note: dot-dirs (.git/.venv) are skipped by glob, but non-dot junk dirs
            (node_modules, etc.) are not — point `dir` at a clean docs tree.
        pattern: Glob for files to index (default: "*.md"). Supports one brace group,
            e.g. "*.{md,txt}" to cover several extensions in a single call. Indexing the
            same directory with different patterns is safe — each call only reconciles
            (sweeps deleted docs) within its OWN pattern, so "*.md" then "*.txt" coexist.
        max_depth: When recursive, cap how many directory levels below `dir` to
            descend (0 = root only, 1 = root + immediate subdirs, ...). Default
            None = unbounded. Use it to index only the shallow layers of a deep tree.
        respect_gitignore: Default True — when `dir` is inside a git repo, skip files
            git ignores (drafts, generated docs, secrets). Set False to index every
            matching file. A no-op outside a git repo.

    Nested scopes: indexing a directory already fully covered by an indexed PARENT scope
    is refused (no coverage benefit); indexing a PARENT over an already-indexed child
    proceeds with a warning (it adds files outside the child). Exception: passing
    respect_gitignore=False under a parent indexed with it ON proceeds with a warning —
    that is how you index an ignored subtree without deleting the parent scope.
    """
    from core.code_kb.chunker import drop_gitignored
    root = os.path.abspath(os.path.expanduser(dir))
    project = _scope_path(dir)
    host = _env("RETRIEVAL_HOST", "RETRIVAL_HOST", default=socket.gethostname())
    block, nested_warning = await _nested_scope_guard(
        project, host or "", "doc", respect_gitignore)
    if block:
        return block
    patterns = _expand_brace_patterns(pattern)
    matched = sorted({p for pat in patterns
                      for p in _discover_files(root, pat, recursive, max_depth)
                      if os.path.isfile(p)})
    # Drop rendered-image / binary files even if the pattern matched them (W161).
    files = [p for p in matched if os.path.splitext(p)[1].lower() not in _SKIP_INDEX_EXTS]
    skipped_binary = len(matched) - len(files)
    # git subprocess off the event loop (blocking), like the index_code / _git_commit path.
    files, skipped_gitignored = await asyncio.to_thread(
        drop_gitignored, files, root, respect_gitignore)
    notes = [n for n in (f"{skipped_binary} binary/image skipped" if skipped_binary else None,
                         f"{skipped_gitignored} gitignored" if skipped_gitignored else None) if n]
    skip_note = f", {', '.join(notes)}" if notes else ""
    if not files:
        return (f"No indexable text files matching {pattern!r} in {root} "
                f"(recursive={recursive}, max_depth={max_depth}{skip_note}).")

    effective_max_depth = None if (recursive and max_depth is None) else (max_depth if recursive else 0)
    outcome = await _index_doc_files(
        root, project, host, files, patterns, effective_max_depth,
        nested_warning=nested_warning, skip_note=skip_note)
    return outcome.message


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


def _doc_catalog_row(candidate: FileCandidate, chunk_count: int) -> dict:
    return {"rel_path": candidate.rel_path, "kind": "doc", "collection": "journal_chunks",
            "chunk_count": chunk_count, "content_hash": candidate.content_hash,
            "size_bytes": candidate.size_bytes}


def _doc_known_in_envelope(known: dict[str, dict], spec: ScanSpec) -> dict[str, dict]:
    """Keep the catalog diff scoped to this pattern/depth scan only. Rows under a
    dot-directory (``.conversations/``) were never scanned (the walk skips them) and must
    not feed the snapshot diff or the replace guard."""
    scoped: dict[str, dict] = {}
    for rel_path, record in known.items():
        normalized = rel_path.replace("\\", "/").strip("/")
        if any(part.startswith(".") and part not in {".", ".."}
               for part in normalized.split("/")[:-1]):
            continue
        try:
            if spec.matches(rel_path):
                scoped[rel_path] = record
        except ValueError:
            continue
    return scoped


async def _index_doc_files(
    root: str, project: str, host: str | None, files: list[str],
    patterns: list[str], max_depth: int | None, *,
    known: dict | None = None, snapshot: bool = True,
    nested_warning: str = "", skip_note: str = "",
) -> _DocIndexOutcome:
    """The shared document pipeline behind index_code and journal_index_dir: diff ``files``
    (absolute paths, already gated by the caller) against the scope's doc catalog, send
    only the changed bodies through the Journal bulk-index job, and let the sweep envelope
    (``patterns`` + ``max_depth``; None = unbounded) bound what a snapshot may remove.

    ``snapshot=False`` is a partial update (the watcher's changed files): nothing may be
    swept, and ``known`` (the caller's doc baseline) replaces the inventory lookup."""
    # Fetch only this scope's detailed doc inventory.  This lets an unchanged file avoid
    # reading its text body, a journal HTTP POST, SQL/FTS work, embedding and Qdrant
    # altogether.  If the inventory is temporarily unavailable we fail open to the old
    # latest-wins POST path; correctness is more important than a performance optimization.
    inventory_note = ""
    if known is None:
        known = {}
        try:
            inventory = await _client.index_inventory(
                host=host, project=project, kind="doc", detail=True, lean=True)
            known = _flatten_inventory_doc_files(inventory)
        except Exception as exc:  # noqa: BLE001 -- existing server dedupe remains correct.
            logger.warning("document index: doc inventory unavailable; no local unchanged skip: %s", exc)
            inventory_note = "; local unchanged skip unavailable (inventory lookup failed)"

    spec = ScanSpec(root_dir=root, patterns=tuple(patterns), max_depth=max_depth)
    candidates = await asyncio.to_thread(
        lambda: [FileCandidate.from_path(path, root) for path in files])
    # A previous ``*.txt`` scan is outside this ``*.md`` envelope.  It must not
    # make a valid Markdown snapshot appear disjoint or interfere with its
    # independently scoped reconcile.
    known_in_envelope = _doc_known_in_envelope(known, spec)
    diff = SnapshotDiff.from_candidates(
        candidates, known_in_envelope, case_sensitive=spec.case_sensitive)
    replace_safe, replace_block_reason = diff.replace_safe, diff.replace_block_reason
    if not snapshot:
        # A watcher batch names only what changed; it is not a snapshot of the scope.
        replace_safe, replace_block_reason = False, "partial update"
    elif not replace_safe:
        # Reuse the Code KB's deletion safety valve.  A failed/empty/disjoint walk
        # must never turn into a stale-doc or catalog sweep; changed files can still
        # be upserted, and an explicit delete remains the only scope-clear path.
        logger.warning("document index: snapshot replace disabled for %s: %s",
                       project, replace_block_reason)
    unchanged = set(diff.unchanged)

    def previous_chunks(rel_path: str) -> int:
        row = known_in_envelope.get(rel_path)
        if row is None and not spec.case_sensitive:
            folded = rel_path.casefold()
            row = next((r for k, r in known_in_envelope.items() if k.casefold() == folded), None)
        return int((row or {}).get("chunk_count", 0) or 0)

    # A re-run that finds nothing to do: every file is readable and byte-identical to its
    # catalog row, and no cataloged file of this pattern/depth is missing locally. The
    # full-snapshot report (replace=True) and the stale-doc reconcile below would then
    # rewrite the same rows and remove nothing, so skip both round trips.
    if (not inventory_note and replace_safe and diff.records
            and not diff.changed_paths and not diff.removed
            and all(candidate.readable for candidate in diff.records)):
        count = len(diff.records)
        prefix = f"{nested_warning} " if nested_warning else ""
        return _DocIndexOutcome(
            message=(
                f"{prefix}Indexed {count} file(s) into journal '{project}' "
                f"({count} unchanged, 0 failed{skip_note}); "
                f"reported {count} to inventory (host={host}); 0 stale doc(s) removed."),
            records=[_doc_catalog_row(candidate, previous_chunks(candidate.rel_path))
                     for candidate in diff.records],
            unchanged=count)

    # Changed files travel as ONE backend job.  This is deliberately before the
    # legacy per-file path below: the call returns a receipt immediately while
    # the backend performs one SQL/FTS preparation, cross-file 128-chunk embedding
    # and durable Journal-Qdrant write.  Unchanged rows are carried in the job's
    # catalog snapshot without their bodies ever crossing HTTP.
    if diff.changed_paths:
        report_files: list[dict] = []
        bulk_files: list[dict] = []
        failed_reads = 0
        for candidate in diff.records:
            if not candidate.readable:
                failed_reads += 1
                continue
            if candidate.rel_path in unchanged:
                report_files.append(_doc_catalog_row(
                    candidate, previous_chunks(candidate.rel_path)))
                continue
            try:
                content = await asyncio.to_thread(
                    # Preserve CRLF bytes when serialising valid UTF-8 text: the
                    # snapshot's hash/size are raw filesystem facts and the bulk
                    # API validates them against the submitted UTF-8 body.
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
                job = await _client.submit_journal_bulk_index(
                    project_slug=project, host=host or "", root_dir=root, files=bulk_files,
                    report_files=report_files, replace=(replace_safe and failed_reads == 0),
                    live_titles=[row["rel_path"] for row in report_files],
                    patterns=patterns, max_depth=spec.max_depth,
                )
                if job.get("job_id"):
                    return _DocIndexOutcome(
                        message=(
                            f"Queued Journal bulk index job '{job.get('job_id')}' for {len(bulk_files)} changed "
                            f"file(s) in '{project}' ({len(unchanged)} unchanged locally, "
                            f"{failed_reads} failed); poll journal_index_status(job_id='{job.get('job_id')}')."),
                        job_id=str(job.get("job_id")), records=report_files,
                        queued=len(bulk_files), unchanged=len(unchanged), failed=failed_reads,
                        inventory_ok=not inventory_note)
            except Exception:
                logger.warning("journal bulk endpoint unavailable; using compatibility path", exc_info=True)

    indexed = deduped = failed = dup_synced = 0
    records: list[dict] = []
    _sem = asyncio.Semaphore(_JOURNAL_INDEX_CONCURRENCY)

    async def _index_one(candidate: FileCandidate) -> tuple[str, dict | None, bool, int]:
        async with _sem:
            try:
                if not candidate.readable:
                    raise OSError(candidate.read_error or "unable to read file")
                if candidate.rel_path in unchanged:
                    # Reuse the durable catalog facts, but refresh the report's hash
                    # and size from this actual scan.  Crucially, do not open the
                    # body or call record_journal for this path.
                    return "unchanged", _doc_catalog_row(
                        candidate, previous_chunks(candidate.rel_path)), True, 0
                with open(candidate.abs_path, encoding="utf-8", errors="replace", newline="") as fh:
                    content = fh.read()
                if not content.strip():
                    return "skip", None, False, 0
                resp = await _client.record_journal(
                    project_slug=project, client="mcp", source_kind="doc",
                    title=candidate.rel_path, body=content, tags=[],
                    dedupe_policy="return_existing",
                )
                rec = _doc_catalog_row(candidate, int(resp.get("chunk_count", 0) or 0))
                return ("ok", rec, bool(resp.get("deduped")),
                        int(resp.get("duplicates_updated", 0) or 0))
            except Exception:  # noqa: BLE001 — keep going; one bad file shouldn't abort the batch
                logger.exception("document index: failed to index %s", candidate.abs_path)
                return "fail", None, False, 0

    # Bounded fan-out; gather preserves submission order so `records` (and thus the
    # inventory snapshot) is identical to the serial walk.
    for status, rec, was_deduped, dup in await asyncio.gather(*(_index_one(candidate) for candidate in diff.records)):
        if status in {"ok", "unchanged"}:
            indexed += 1
            if was_deduped:
                deduped += 1
            dup_synced += dup
            records.append(rec)
        elif status == "fail":
            failed += 1

    dup_note = f", {dup_synced} cross-scope synced" if dup_synced else ""
    # This scan's sweep envelope — shared by the catalog replace AND the journal reconcile so
    # they stay in lockstep: depth (finding #4, non-recursive → root only) AND pattern (W169b,
    # a "*.txt" re-run must not sweep the "*.md" files it never scanned, in either store).
    effective_max_depth = spec.max_depth
    prefix = f"{nested_warning} " if nested_warning else ""
    try:
        # A destructive snapshot replace is only safe when EVERY file was read this run —
        # a file that failed to read is absent from `records` and would be swept from the
        # catalog even though its journal data is still live. On any failure, upsert only
        # (replace=False), matching the stale-doc reconcile gate below (finding #3). The
        # pattern/depth envelope further scopes the sweep to what this run could have indexed.
        rep = await _client.report_index(
            host=host, project=project, root_dir=root, files=records,
            replace=(failed == 0 and replace_safe),
            patterns=patterns, max_depth=effective_max_depth,
        )
        reported = rep.get("total_files", len(records))
    except Exception as exc:  # noqa: BLE001 — inventory reporting is best-effort
        logger.warning("document index: inventory report failed: %s", exc)
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
            reconciled = await _client.reconcile_journal_docs(
                project_slug=project,
                live_titles=[record["rel_path"] for record in records],
                max_depth=effective_max_depth,
                patterns=patterns,
            )
            if reconciled.get("skipped"):
                reconcile_note = f"; stale doc reconcile skipped: {reconciled.get('reason')}"
            else:
                reconcile_note = f"; {reconciled.get('removed_entries', 0)} stale doc(s) removed"
        except Exception as exc:  # noqa: BLE001
            logger.warning("document index: doc reconcile failed: %s", exc)
            reconcile_note = f"; stale doc reconcile skipped: {exc}"
    return _DocIndexOutcome(
        message=(
            f"{prefix}Indexed {indexed} file(s) into journal '{project}' "
            f"({deduped} unchanged, {failed} failed{skip_note}{dup_note}); "
            f"reported {reported} to inventory (host={host}){inventory_note}{reconcile_note}."),
        records=records, unchanged=deduped, failed=failed, inventory_ok=not inventory_note)


@mcp.tool()
async def journal_index_status(job_id: str) -> str:
    """Poll a directory-level Journal bulk-index job returned by journal_index_dir."""
    job = await _client.journal_bulk_index_job(job_id)
    return (f"Journal job {job.get('job_id', job_id)}: {job.get('status', 'unknown')} "
            f"files={job.get('files_done', 0)}/{job.get('files_total', 0)} "
            f"chunks={job.get('chunks_done', 0)}/{job.get('chunks_total', 0)} "
            f"indexed={job.get('indexed', 0)} progress={job.get('progress', 0):.0%}"
            + (f" skipped_binary={', '.join(job['skipped_binary'])}" if job.get('skipped_binary') else "")
            + (f" error={job.get('error')}" if job.get('error') else ""))


# --------------------------------------------------------------------------- #
# ACE — drop-in compatible tools (angry-robot-deals/ace-mcp signatures).
# The playbook is the evolving "context" ACE curates; embeddings come from the
# backend's shared BGE-M3 via /api/embed (no model loaded here). ACE itself has
# NO code-generation model — these tools curate/retrieve guidance; the CALLING
# agent does the generating.
# --------------------------------------------------------------------------- #

# Injected local playbook (tests / offline). When None, ACE tools use the SERVER
# playbook so the admin console reflects real ACE usage — see _get_playbook().
_playbook: "PlaybookStore | ServerPlaybook | None" = None


async def _embed_fn(texts: list[str]) -> list[list[float]]:
    resp = await _client.embed(texts)
    return resp.get("dense") or []


def _bullet_from_dict(d: dict) -> Bullet:
    return Bullet(
        id=str(d.get("id", "")), content=d.get("content", ""),
        section=d.get("section", "insights"), helpful=d.get("helpful", 0),
        harmful=d.get("harmful", 0), domain=d.get("domain"),
        created_at=d.get("created_at"), score=d.get("score"),
        deduped=bool(d.get("deduped", False)),
    )


class ServerPlaybook:
    """PlaybookStore-shaped adapter over the backend's per-user playbook REST API.

    Same method surface as the local PlaybookStore so the ACE tools are agnostic to
    where the playbook lives. Centralizing on the server makes the admin console
    reflect what the ACE tools curate (per (owner, project); project = cwd scope)."""

    def __init__(self, client, project: str):
        self._client = client
        self.project = project

    async def add_bullet(self, content: str, section: str = "insights",
                         domain: str | None = None) -> Bullet:
        resp = await self._client.playbook_record(
            project_slug=self.project, content=content, section=section, domain=domain)
        return _bullet_from_dict(resp)

    async def search(self, query: str, top_k: int = 5,
                     domain: str | None = None) -> list[Bullet]:
        resp = await self._client.playbook_search(
            project_slug=self.project, query=query, top_k=top_k, domain=domain)
        return [_bullet_from_dict(b) for b in (resp.get("bullets") or [])]

    async def list_bullets(self, domain: str | None = None, limit: int = 200) -> list[Bullet]:
        resp = await self._client.playbook_list(
            project_slug=self.project, domain=domain, limit=limit)
        return [_bullet_from_dict(b) for b in (resp.get("bullets") or [])]

    async def count(self, domain: str | None = None) -> int:
        # Parity with the local PlaybookStore.count() so ace_playbook reports a real
        # total, not the page size. Bullet sets are per-(owner,project) small (hundreds).
        resp = await self._client.playbook_list(
            project_slug=self.project, domain=domain, limit=500)
        return int(resp.get("total") or len(resp.get("bullets") or []))


async def _maybe_await(value):
    """Await value if it is awaitable, else return as-is. Bridges the local
    PlaybookStore (sync list_bullets/count) and ServerPlaybook (async) surfaces."""
    return await value if inspect.isawaitable(value) else value


def _get_playbook():
    """Injected local store if set (tests/offline), else the server-backed playbook
    scoped to the cwd's absolute path."""
    if _playbook is not None:
        return _playbook
    return ServerPlaybook(_client, project=_scope_path("."))


def _format_bullets(bullets: list[Bullet], header: str) -> str:
    if not bullets:
        return f"{header}\n(none yet — record lessons with ace_smart_reflect.)"
    lines = [header, ""]
    for i, b in enumerate(bullets, 1):
        tag = f"[{b.section}]" + (f"[{b.domain}]" if b.domain else "")
        sim = f" sim={b.score:.3f}" if b.score is not None else ""
        lines.append(f"{i}. {tag} {b.content}  (+{b.helpful}/-{b.harmful}{sim})")
    return "\n".join(lines)


@mcp.tool()
async def ace_context_aware(query: str, domain: str | None = None, top_k: int = 5) -> str:
    """ACE: retrieve the playbook bullets most relevant to a query — the Generator's
    context-retrieval step. Returns reusable lessons/strategies/pitfalls previously
    curated for THIS project, ranked by semantic relevance. Empty until you record
    lessons with ace_smart_reflect.

    Args:
        query: What you're about to do (a task or question) (required).
        domain: Optional tag to scope to one area, e.g. "qdrant", "frontend" (default: all).
        top_k: How many bullets to return (default 5).
    """
    bullets = await _get_playbook().search(query, top_k=top_k, domain=domain)
    return _format_bullets(bullets, f"Playbook context for: {query!r}")


@mcp.tool()
async def ace_enhance_prompt(prompt: str, focus_area: str | None = None) -> str:
    """ACE: augment a prompt with the project's most relevant playbook bullets, so
    the calling model has accumulated lessons in-context before it works. Returns the
    original prompt followed by a guidelines block — it does NOT call any LLM itself.

    Args:
        prompt: The prompt you intend to run (required).
        focus_area: Optional text to bias retrieval (default: the prompt itself).
    """
    bullets = await _get_playbook().search(focus_area or prompt, top_k=5)
    if not bullets:
        return prompt
    guide = "\n".join(f"- {b.content}" for b in bullets)
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
        prompt: The task you're about to do (required).
        context: Optional extra context to bias which lessons are retrieved (default: the prompt).
        auto_enhance: If true (default), retrieve and attach relevant playbook bullets.
    """
    pb = _get_playbook()
    bullets = await pb.search(context or prompt, top_k=6) if auto_enhance else []
    guidance = _format_bullets(bullets, "Apply these learned playbook bullets:")
    enhanced = prompt
    if bullets:
        guide = "\n".join(f"- {b.content}" for b in bullets)
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
        auto_suggest: Accepted for drop-in compatibility; no LLM suggestion is run (default true).
        section: patterns | best_practices | insights | failure_mode (default "insights").
        domain: Optional tag to group the lesson, e.g. "qdrant", "frontend" (default: none).

    Example: ace_smart_reflect(code="Always wait=True on Qdrant upsert or batches drop
    silently under load", section="failure_mode", domain="qdrant")
    """
    lesson = (code or "").strip()
    if not lesson:
        return "Nothing to reflect: empty input."
    b = await _get_playbook().add_bullet(lesson, section=section, domain=domain)
    verb = "reinforced existing" if b.deduped else "added new"
    return (f"Playbook {verb} bullet [{b.section}] id={b.id} "
            f"(+{b.helpful}/-{b.harmful}): {b.content}")


@mcp.tool()
async def ace_playbook(domain: str | None = None, limit: int = 50) -> str:
    """ACE: view the current project playbook — every curated bullet with its
    helpful/harmful counters. Use it to inspect what the project has learned.

    Args:
        domain: Optional tag filter, e.g. "qdrant" (default: all domains).
        limit: Max bullets to list (default 50).
    """
    pb = _get_playbook()
    # list_bullets/count are sync on the local store, async on the server-backed one.
    bullets = await _maybe_await(pb.list_bullets(domain=domain, limit=limit))
    count = await _maybe_await(pb.count()) if hasattr(pb, "count") else len(bullets)
    return _format_bullets(bullets, f"Playbook ({count} bullets):")


# --------------------------------------------------------------------------- #
# Code KB — server-side (aligned with ACE playbook + journal). The CLIENT (this
# process, where the code lives) walks + AST-chunks the repo locally; the chunks
# go to the backend, which embeds them with the warm BGE-M3 and stores them in the
# 4090 Qdrant `code_chunks` collection, scoped to the caller. No local vector store
# — so no single-writer lock contention, and everything lives on 4090 like the rest.
# --------------------------------------------------------------------------- #

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


def _http_timeout() -> httpx.Timeout:
    """Layered httpx timeout used by the process-level RetrievalClient."""
    from mcp_server.backend import make_http_timeout
    return make_http_timeout()


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


def _format_code_hits(hits: list[dict], header: str, *, show_line_numbers: bool = True) -> str:
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


# Local AST chunking is CPU-bound but tree-sitter releases the GIL while parsing, so a
# bounded thread pool gives real parallelism (and overlaps file reads). Cap it so a huge
# tree can't spawn hundreds of threads.
_CHUNK_WORKERS = max(1, min(16, (os.cpu_count() or 4)))


def _chunk_executor(worker_count: int) -> ThreadPoolExecutor:
    from core.code_kb.chunker import create_chunk_executor

    return create_chunk_executor(worker_count)


def _chunk_file_payload(abs_path: str, rel: str, repo_root: str,
                        content_hash: str | None = None) -> dict | None:
    """AST-chunk one file into an upload payload (carrying its content_hash so future
    incremental indexes can skip it when unchanged), or None if it yields no chunks
    (empty / too big / binary)."""
    from core.code_kb.chunker import chunk_file, file_content_hash, resolve_language
    chunks = chunk_file(abs_path, repo_root)
    if not chunks:
        return None
    return {
        "rel_path": rel, "language": resolve_language(rel) or "text",
        "content_hash": content_hash if content_hash is not None else file_content_hash(abs_path),
        "chunks": [{"start_line": c.start_line, "end_line": c.end_line,
                    "label": c.label, "content": c.content, "code": c.code} for c in chunks],
    }


# Optional progress hook for long local walks. Called from the collector thread
# (not pool workers) so callers can safely update shared job status without locks.
# Payload keys: files_total, files_done, files_changed, chunks_ready, phase.
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


def _walk_repo_files(repo_root: str, respect_gitignore: bool = True,
                     exclude_globs: list[str] | None = None, docs: bool = True):
    """(abs_path, rel_path, kind) for every indexable file, from ONE directory walk.

    kind is "code" or "doc" (``docs=False`` omits documents); see
    core.code_kb.chunker.walk_repo_files for the gate shared by both kinds."""
    from core.code_kb.chunker import walk_repo_files
    yield from walk_repo_files(
        repo_root, respect_gitignore=respect_gitignore,
        exclude_globs=exclude_globs, docs=docs)


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
        from core.code_kb.chunker import iter_repo_files
        files = list(iter_repo_files(
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
    known_hashes: dict[str, str],
    respect_gitignore: bool = True,
    exclude_globs: list[str] | None = None,
    progress_cb: CollectProgressCb | None = None,
    files: list[tuple[str, str]] | None = None,
) -> tuple[list[dict], list[str], dict[str, str]]:
    """Hash every candidate file; AST-chunk ONLY those whose hash differs from
    known_hashes (the server's last-indexed hashes). ``files`` is the already-walked
    code list when the caller shares one walk with the document pass (None walks here).
    Returns (changed_payloads, manifest, current_hashes).

    manifest = the paths that should hold vectors after this op = unchanged-already-
    indexed ∪ changed-with-chunks. A changed file that now yields 0 chunks (emptied)
    is in neither, so the server reconciles its stale chunks away — keeping incremental
    byte-identical to a full reindex of the current tree (the correctness invariant).

    Hashing + (for changed files) chunking run CONCURRENTLY on a bounded thread pool;
    results are assembled in walk order so payloads/manifest match the serial version.
    ``progress_cb`` reports files_done/files_total while the pool runs (observability only).
    """
    from core.code_kb.chunker import file_content_hash
    if files is None:
        from core.code_kb.chunker import iter_repo_files
        files = list(iter_repo_files(
            repo_root, respect_gitignore=respect_gitignore, exclude_globs=exclude_globs))
    total = len(files)
    stride = _collect_progress_stride(total)
    _emit_collect_progress(
        progress_cb, files_total=total, files_done=0, force=True, stride=stride,
        phase="scanning and chunking local files",
    )

    def _one(abs_path: str, rel: str) -> tuple[str, str | None, dict | None]:
        h = file_content_hash(abs_path)
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


def _flatten_inventory_code_files(resp: dict) -> dict[str, dict]:
    """Flatten the inventory tree to {rel_path: file_node} for kind=code — the client's
    record of what it last indexed (hash + chunk_count + language), the diff baseline."""
    out: dict[str, dict] = {}
    for u in resp.get("users") or []:
        for h in u.get("hosts") or []:
            for p in h.get("projects") or []:
                for d in p.get("dirs") or []:
                    for f in d.get("files") or []:
                        if f.get("kind", "code") == "code":
                            out[f["rel_path"]] = f
    return out


def _inventory_records(manifest: list[str], payloads: list[dict],
                       known: dict[str, dict], current: dict[str, str],
                       commit: str) -> list[dict]:
    """Full catalog snapshot for the current tree: changed files take their fresh
    chunk_count/language, unchanged files reuse the known node's — so replace=True
    keeps the catalog exactly in sync and sweeps removed files."""
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
    """Short HEAD commit for `root`, or '' if not a git repo — best-effort provenance
    so the inventory can show which revision is indexed. Never raises."""
    try:
        # stdin=DEVNULL is REQUIRED: without it the child inherits THIS process's stdin,
        # which for an MCP stdio server is the JSON-RPC pipe from the client — git then
        # wedges on that handle for the full timeout (~5s per index_code call). (W159)
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


async def _known_code_hashes(project: str) -> dict[str, dict]:
    """The server's inventory of this project's code files, {rel_path: node}, as the
    incremental diff baseline. Failure (offline / no auth) degrades to {} → a full pass."""
    try:
        # detail=True: the summary inventory omits the per-file content hashes we diff against.
        inv = await _client.index_inventory(
            project=project, kind="code", detail=True, lean=True)
        return _flatten_inventory_code_files(inv)
    except Exception as exc:  # noqa: BLE001 — offline/degraded → treat as no prior index
        logger.warning("index_code: inventory fetch failed (%s) — indexing all files", exc)
        return {}


async def _set_code_policy(
    *, project: str, host: str, root_dir: str, auto_refresh: bool,
    respect_gitignore: bool, exclude_globs: list[str],
) -> dict:
    return await _client.upsert_index_policy(
        host=host, project=project, kind="code", root_dir=root_dir,
        auto_refresh=auto_refresh, respect_gitignore=respect_gitignore,
        exclude_globs=exclude_globs)


async def _get_code_policy(*, project: str, host: str) -> dict | None:
    try:
        return await _client.get_index_policy(host=host, project=project, kind="code")
    except Exception as exc:  # missing policy means the pre-auto-refresh behavior
        logger.info("code auto-refresh policy unavailable for %s: %s", project, exc)
        return None


async def _set_doc_policy(
    *, project: str, host: str, root_dir: str, auto_refresh: bool,
    respect_gitignore: bool, exclude_globs: list[str],
) -> dict:
    """Record, under kind="doc" of the same policy endpoint, whether this scope's
    documents ride along with its code index (the watcher reads it)."""
    return await _client.upsert_index_policy(
        host=host, project=project, kind="doc", root_dir=root_dir,
        auto_refresh=auto_refresh, respect_gitignore=respect_gitignore,
        exclude_globs=exclude_globs)


async def _get_doc_policy(*, project: str, host: str) -> dict | None:
    """The scope's document policy; None when none is registered (HTTP 404). Any other
    failure raises, so a caller cannot mistake an outage for "documents are off"."""
    try:
        return await _client.get_index_policy(host=host, project=project, kind="doc")
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
    jobs = await _client.code_index_jobs(
        project=project, active_only=True, limit=100)
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
            _AUTO_REFRESH_RESULTS[key] = result
            _AUTO_REFRESH_CHECKED_AT[key] = time.monotonic()
            return result

        policy = await _get_code_policy(project=project, host=host)
        if not policy or not policy.get("auto_refresh", False):
            return {"state": "disabled"}
        root = os.path.abspath(os.path.expanduser(repo_root))
        if not os.path.isdir(root):
            return {"state": "unavailable", "message": f"repository root unavailable: {root}"}
        interval = _auto_index_interval_s(policy)
        _AUTO_REFRESH_INTERVALS[key] = interval
        known = await _known_code_hashes(project)
        if not known:
            result = {"state": "unchecked", "message": "no completed code inventory yet"}
        else:
            known_hashes = {rel: node["content_hash"] for rel, node in known.items()
                            if node.get("content_hash")}
            payloads, manifest, current = await asyncio.to_thread(
                _collect_incremental, root, known_hashes,
                bool(policy.get("respect_gitignore", True)),
                list(policy.get("exclude_globs") or []))
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
        _AUTO_REFRESH_RESULTS[key] = result
        _AUTO_REFRESH_CHECKED_AT[key] = time.monotonic()
        return result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("automatic code refresh failed for %s", project)
        result = {"state": "failed", "message": str(exc)}
        _AUTO_REFRESH_RESULTS[key] = result
        _AUTO_REFRESH_CHECKED_AT[key] = time.monotonic()
        return result


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


_GIT_CHECK_IGNORE_TIMEOUT_S = 10
_CHECK_IGNORE_BATCH = 2000


def _git_ignored_watch_paths(repo_root: str, rel_paths: set[str]) -> set[str]:
    """Ask Git only about touched paths; never enumerate the repository.

    The paths go to git in sorted batches of _CHECK_IGNORE_BATCH, each under its own
    timeout. Fails closed: a batch git did not answer raises GitIgnoreCheckFailed (never
    "nothing is ignored"); only a non-repository / missing git answers set()."""
    from core.code_kb.chunker import GitIgnoreCheckFailed

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
            return set()  # git is not installed: no filtering, as gitignore_allowed
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
    doc_changed: list[FileCandidate]
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
    from core.code_kb.chunker import (
        DEFAULT_SKIP_DIRS,
        FALLBACK_EXTS,
        GitIgnoreCheckFailed,
        file_content_hash,
        gitignore_allowed,
        is_doc_path,
        load_retrievalignore,
        path_excluded,
        resolve_language,
    )

    root = os.path.abspath(repo_root)
    excludes = list(dict.fromkeys(load_retrievalignore(root) + list(exclude_globs)))
    candidates: dict[str, str] = {}
    doc_candidates: dict[str, str] = {}
    removed: set[str] = set()
    doc_removed: set[str] = set()
    requires_reconcile = False

    def is_supported(path: str) -> bool:
        name = os.path.basename(path).lower()
        ext = os.path.splitext(name)[1]
        if not ext and name.startswith("."):
            ext = name
        return bool(resolve_language(path) or ext in FALLBACK_EXTS)

    def add_candidate(path: str) -> None:
        rel = _watch_rel_path(root, path)
        if rel is None:
            return
        if rel in {".gitignore", ".retrievalignore"}:
            nonlocal requires_reconcile
            requires_reconcile = True
            return
        parts = rel.split("/")
        if any(part in DEFAULT_SKIP_DIRS for part in parts[:-1]):
            return
        if not is_supported(path):
            if rel in known:
                removed.add(rel)
            if known_docs is not None:
                if is_doc_path(path, rel) and not path_excluded(rel, excludes):
                    doc_candidates[rel] = path
                elif rel in known_docs:
                    doc_removed.add(rel)
            return
        if path_excluded(rel, excludes):
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
                if not rel or any(part in DEFAULT_SKIP_DIRS for part in rel.split("/")):
                    continue
                # Git enumerates the directory (ignored files never become candidates);
                # the os.walk below is only for scopes git cannot answer for.
                listed = gitignore_allowed(root, [rel]) if respect_gitignore else None
                if listed is not None:
                    for listed_rel in sorted(listed):
                        listed_parts = listed_rel.split("/")
                        if any(part in DEFAULT_SKIP_DIRS for part in listed_parts[:-1]):
                            continue
                        listed_path = os.path.join(root, *listed_parts)
                        if os.path.isfile(listed_path):  # tracked-but-deleted is not one
                            add_candidate(listed_path)
                    continue
                for walk_root, dirs, files in os.walk(absolute):
                    dirs[:] = [name for name in dirs if name not in DEFAULT_SKIP_DIRS]
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
        content_hash = file_content_hash(path)
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

    doc_changed: list[FileCandidate] = []
    for rel, doc_path in sorted(doc_candidates.items()):
        candidate = FileCandidate.from_path(doc_path, root)
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
    inventory = await _client.index_inventory(
        host=host, project=project, kind="doc", detail=True, lean=True)
    spec = ScanSpec(root_dir=project, patterns=tuple(DOC_GLOBS), max_depth=None)
    return _doc_known_in_envelope(_flatten_inventory_doc_files(inventory), spec)


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
    await _client.report_index(
        host=host, project=project, root_dir=root, files=rows,
        replace=True, patterns=DOC_GLOBS, max_depth=None)
    await _client.reconcile_journal_docs(
        project_slug=project, live_titles=sorted(set(remaining) | set(keep)),
        max_depth=None, patterns=DOC_GLOBS)
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
    return await _wait_backend_job(lambda: _client.code_index_job(job_id))


async def _wait_doc_job(job_id: str) -> dict:
    return await _wait_backend_job(lambda: _client.journal_bulk_index_job(job_id))


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
    from core.code_kb.chunker import GitIgnoreCheckFailed

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
    return await _client.code_index(**request)


async def _get_code_index_receipt(job_id: str) -> dict:
    return await _client.code_index_job(job_id)


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
        job = await _client.journal_bulk_index_job(str(doc_job_id))
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
    sources = await _client.public_code_sources()
    return format_public_code_sources(sources)


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

    `proxy_profile` and `credential_profile` are SERVER-SIDE profile names. Never pass
    raw cookies, tokens, passwords, private keys, or Authorization header values.
    `relative_path` is relative to the configured public-code mirror root.
    """
    response = await _client.add_public_code_source(
        source_id=source_id,
        domain=domain,
        upstream_url=upstream_url,
        relative_path=relative_path,
        default_ref=default_ref,
        proxy_profile=proxy_profile,
        credential_profile=credential_profile,
        language_overrides=language_overrides,
        update_policy=update_policy,
        update_interval_minutes=update_interval_minutes,
        index_now=index_now,
    )
    lines = [format_public_code_sources([response.get("source") or {}])]
    if response.get("index_job"):
        lines.append(format_public_code_job(response["index_job"]))
    return "\n".join(lines)


@mcp.tool()
async def check_public_code_updates(
    source_ids: list[str] | None = None,
    domains: list[str] | None = None,
) -> str:
    """Queue non-blocking upstream revision checks for selected sources/domains.
    Empty selections mean every configured source. Returns a pollable job id."""
    job = await _client.check_public_code_updates(
        source_ids=source_ids or [], domains=domains or [])
    return format_public_code_job(job)


@mcp.tool()
async def index_public_code_sources(
    source_ids: list[str] | None = None,
    domains: list[str] | None = None,
) -> str:
    """Queue mirror refresh, AST chunking, embedding, validation, and revision promotion
    for selected public repositories. Empty selections mean every configured source."""
    job = await _client.index_public_code_sources(
        source_ids=source_ids or [], domains=domains or [])
    return format_public_code_job(job)


@mcp.tool()
async def set_public_code_update_policy(
    source_id: str,
    update_policy: str = "manual",
    update_interval_minutes: int | None = None,
) -> str:
    """Set one repository to manual or scheduled update checks. Scheduled policies
    require `update_interval_minutes`; manual policies ignore the interval."""
    source = await _client.set_public_code_update_policy(
        source_id=source_id,
        update_policy=update_policy,
        update_interval_minutes=update_interval_minutes,
    )
    return format_public_code_sources([source])


@mcp.tool()
async def public_code_job_status(job_id: str | None = None, limit: int = 20) -> str:
    """Poll one public-code job or import handle, or list recent durable job snapshots
    when `job_id` is omitted. Interrupted jobs remain queryable and are marked failed
    after restart."""
    if job_id:
        local = _PUBLIC_IMPORTS.get(job_id)
        if local is not None:
            lines = [_format_public_import(job_id, local)]
            if local.get("job_id"):
                lines.append(format_public_code_job(
                    await _client.public_code_job(local["job_id"])))
            return "\n".join(lines)
        return format_public_code_job(await _client.public_code_job(job_id))
    jobs = await _client.public_code_jobs(limit=limit)
    if not jobs:
        return "No public-code jobs found."
    return "\n".join(format_public_code_job(job) for job in jobs)


_PUBLIC_IMPORTS: dict[str, dict] = {}  # handle -> local bundling/upload progress
_PUBLIC_IMPORT_TASKS: set[asyncio.Task] = set()


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
        clone, await _client.public_code_sources(), domain=domain, source_id=source_id)
    handle = f"import-{uuid.uuid4().hex[:12]}"
    status = {"phase": "queued", "source_id": source["source_id"], "head": clone["head"]}
    _PUBLIC_IMPORTS[handle] = status
    task = asyncio.create_task(_run_public_import(
        status, clone, source,
        create=_client.create_public_code_import,
        put=_client.upload_public_code_import_part,
        get=_client.public_code_import,
        complete=_client.complete_public_code_import,
    ))
    _PUBLIC_IMPORT_TASKS.add(task)
    task.add_done_callback(_PUBLIC_IMPORT_TASKS.discard)
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
    response = await _client.search_public_code(
        query=query,
        source_ids=source_ids or [],
        domains=domains or [],
        top_k=top_k,
        mode=mode,
        language=language,
        rerank=rerank,
    )
    return format_public_code_search(response, show_line_numbers=show_line_numbers)


@mcp.tool()
async def index_code(repo_root: str = ".",
                     incremental: bool = False, respect_gitignore: bool = True,
                     auto_refresh: bool = True,
                     exclude_globs: list[str] | None = None,
                     docs: bool = True) -> str:
    """Index a code repository so you can later search it semantically with search_code.

    Walks repo_root and AST-chunks every supported source file with a `# path:` scope
    header LOCALLY, then sends only the chunks to the backend, which embeds them with
    the shared BGE-M3 and stores the vectors in the shared (4090) code index, scoped to
    YOU. Source files never leave the machine — only chunk text does.

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
    repo_root = os.path.expanduser(repo_root)  # so the walk honors ~; also drives the scope
    root_abs = os.path.abspath(repo_root)
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
            job = await _client.code_index_job(job_id, scope_notice=True)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return (f"Job {job_id} not found — it may have finished long ago or the "
                        f"backend restarted (queue is in-memory). Re-run index_code if needed.")
            raise
        return _with_scope_notice("Job " + _fmt_job_line(job), job.get("scope_notice"))
    local_jobs = list(reversed(list(_CODE_PREPARATION_STATUS.values())))
    list_notice = None
    try:
        resp = await _client.code_index_jobs(limit=_CODE_PREPARATION_HISTORY_LIMIT)
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
    resp = await _client.index_inventory(host=host, project_contains=project or None)
    # Filtered again here: a backend that predates project_contains returns every project.
    return _format_inventory_summary(resp, project=project, limit=limit)


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
    try:
        resp = await asyncio.wait_for(
            _client.code_search(
                query=query, project=proj, top_k=top_k, mode=mode,
                rerank=rerank, language=language, path_glob=normalized_path_glob,
                max_per_file=max_per_file,
            ),
            timeout=_SEARCH_CODE_TIMEOUT_S,
        )
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
        rendered = _format_code_hits(hits, header, show_line_numbers=show_line_numbers)
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
    resp = await _client.federated_code_search(
        query=query, projects=projects, include_public=include_public,
        source_ids=source_ids, domains=domains, top_k=top_k, mode=mode,
        language=language, path_glob=normalized_path_glob, max_per_file=max_per_file,
        rerank=rerank,
    )
    return _format_federated_code_search(resp, show_line_numbers=show_line_numbers)


async def _explain_unknown_public_selection(exc: httpx.HTTPStatusError) -> None:
    """Re-raise the backend's "Unknown public-code source IDs/domains" 422 with the
    names that do exist, so the next call can be corrected without a second tool."""
    reason = str(exc).split(": ", 1)[-1]
    if exc.response.status_code != 422 or not reason.startswith("Unknown public-code"):
        return
    try:
        sources = await _client.public_code_sources()
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
    """Search indexed code and get the FULL matching chunk back with `path:line`
    citations: your own indexed repositories (default), the curated Public Code corpus,
    or both fused into one ranking. Prefer this tool over shell text search such as rg
    or grep for indexed code. Fall back when the code is not indexed, this tool cannot
    express the query, or the MCP request fails. Run index_code first when you want to
    add this repository to the index.

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

    Retrieval modes:
      - "hybrid" (default): dense semantic + sparse lexical, RRF-fused — best all-round;
        use it when you're describing behavior ("where do we retry a failed upsert").
      - "keyword": sparse lexical only (BM25-like) — use it when you KNOW the exact
        symbol/token (e.g. "select_conn", "TokenBucket") and want literal matches. It
        never substitutes dense neighbours: sparse unavailability returns an explicit diagnosis.
      - "dense": semantic only — use it for pure concept search with no shared tokens.
    (Hybrid may explicitly report a dense-only fallback if an index predates sparse vectors;
    re-index to restore fusion.)

    Args:
        query: What the code does, or an exact symbol for keyword mode.
        top_k: How many chunks to return, 1-50.
        repo_root: See "Where to search". A directory's absolute path is its scope — pass
            the same path you gave index_code. Windows drive/UNC paths accept either
            slash style and case; they canonicalize to one scope.
        mode: "hybrid" (default) | "keyword" | "dense" — see above.
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
        threshold: Similarity at which two chunks count as near-duplicates, 0-1
            (default 0.92). Below ~0.85 unrelated code starts matching.
        min_shared_chunks: How many shared chunks make a file pair a finding
            (default 2). Use 1 only when hunting a single copied function, and expect
            to read past a lot of boilerplate.
        max_pairs: Cap on reported file pairs, 1-500 (default 50).
        max_examples: Shared chunk locations quoted per pair, 1-20 (default 3).
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
    try:
        resp = await asyncio.wait_for(
            _client.code_duplicates(
                project=proj, threshold=threshold,
                min_shared_chunks=min_shared_chunks, max_pairs=max_pairs,
                max_examples=max_examples, exclude_globs=exclude_globs,
            ),
            timeout=_FIND_DUPLICATES_TIMEOUT_S,
        )
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
        resp = await _client.code_delete(project=proj)
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
    """Delete every batch-indexed DOC in a directory scope ENTIRELY: drop all of its
    journal entries + vectors (the docs `journal_index_dir` wrote) AND remove its rows
    from your index inventory. The symmetric counterpart to `delete_code`.

    This is the explicit cleanup path for docs. Re-indexing with a different pattern does
    NOT clear the scope (each re-index only reconciles within its own pattern envelope),
    and an empty snapshot is a deliberate no-op — so wiping a doc scope is always explicit.
    ACE memories (`journal_record`) in the scope are NOT touched — only source_kind="doc".
    After this, journal_search / index_inventory for the scope's docs return nothing.

    The scope is dir's ABSOLUTE PATH — pass the same path you gave journal_index_dir.

    Args:
        dir: The indexed directory whose docs to clear (default: current directory ".").
            Its absolute path is the scope.
    """
    scope = _scope_path(dir)
    # The scope's watcher holds a document baseline that is about to stop being true (and
    # the backend drops the scope's document policy with its rows): let it re-resolve.
    _invalidate_refresh(scope)
    resp = await _client.delete_journal_docs(project_slug=scope)
    removed = resp.get("removed_entries", 0)
    vectors = resp.get("deleted_vectors", 0)
    catalog = resp.get("catalog_removed", 0)
    if not removed and not catalog:
        return (f"Nothing to delete for doc scope '{scope}' — no indexed docs or inventory "
                f"rows found (already clear, or never indexed). ACE memories are untouched.")
    return (f"Deleted doc scope '{scope}': removed {removed} doc entry(ies) "
            f"({vectors} vector(s)) and {catalog} inventory row(s). ACE memories kept.")


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
        mcp.add_tool(replacement, name=name, description=single.__doc__)
        globals()[name] = replacement


_install_multi_query_tools()


def _apply_repository_tool_description_contract() -> None:
    """Keep both supported MCP entry points' published tool schemas identical.

    FastMCP v1 captures a tool description while applying ``@mcp.tool`` and does
    not offer a public in-place description update.  Re-registering the same
    repository functions through its documented remove/add APIs preserves their
    input schemas while replacing the stale repository-only descriptions with
    the package's published contract.
    """
    from mcp_plugin.retrieval_mcp_tool_contract import REPOSITORY_DESCRIPTION_OVERRIDES

    for name, description in REPOSITORY_DESCRIPTION_OVERRIDES.items():
        function = globals().get(name)
        if function is None:
            raise RuntimeError(f"missing repository MCP tool for contract entry {name!r}")
        mcp.remove_tool(name)
        mcp.add_tool(function, name=name, description=description)


_apply_repository_tool_description_contract()

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


def _apply_repository_schema_compaction() -> None:
    from mcp_plugin.retrieval_mcp_tool_contract import MULTI_QUERY_TOOL_NAMES

    _compact_published_tool_schemas(MULTI_QUERY_TOOL_NAMES)


_apply_repository_schema_compaction()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full repository tool server only for daemon/debug commands."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from mcp_plugin.retrieval_mcp_daemon import (
            DEFAULT_REPOSITORY_BACKEND_URL,
            run_thin_cli,
        )

        return run_thin_cli(
            [],
            serve_command=["-m", "mcp_server.server"],
            backend_default_url=DEFAULT_REPOSITORY_BACKEND_URL,
        )
    if args[0] == "stdio-server":
        if len(args) != 1:
            raise SystemExit("stdio-server accepts no arguments")
        logger.info("Starting retrival MCP stdio debug server (backend=%s)", BASE_URL)
        mcp.run()
        return 0
    if args[0] == "serve":
        import argparse
        from mcp_plugin.retrieval_mcp_daemon import (
            DEFAULT_REPOSITORY_BACKEND_URL,
            resolve_runtime_dir,
            serve_daemon,
        )

        parser = argparse.ArgumentParser(prog="python -m mcp_server.server serve")
        parser.add_argument("serve")
        parser.add_argument("--state-dir")
        parser.add_argument("--instance-token", required=True)
        parsed = parser.parse_args(args)
        return serve_daemon(
            mcp,
            runtime_dir=resolve_runtime_dir(parsed.state_dir),
            instance_token=parsed.instance_token,
            backend_default_url=DEFAULT_REPOSITORY_BACKEND_URL,
        )
    from mcp_plugin.retrieval_mcp_daemon import (
        DEFAULT_REPOSITORY_BACKEND_URL,
        run_thin_cli,
    )

    return run_thin_cli(
        args,
        serve_command=["-m", "mcp_server.server"],
        backend_default_url=DEFAULT_REPOSITORY_BACKEND_URL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
