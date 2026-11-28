"""Code chunker — structure-aware AST chunking via chonkie's CodeChunker.

chonkie's ``CodeChunker`` does structure-aware, lossless chunking over tree-sitter
grammars (it bundles the grammar pack, so 40+ languages work with no runtime download).
We migrated off ``astchunk`` (0.1.0, unmaintained) after it C-level SEGFAULTED on some
grammars: it held long-lived ``tree_sitter.Node`` references for window reassembly, and
under a tree-sitter runtime/grammar ABI mismatch ``node.text`` dereferenced freed memory
— an uncatchable crash that killed the MCP process and hung ``index_code`` forever (W155).
chonkie with ``include_nodes=False`` retains no nodes, so it can't hit that class of bug.

Grammarless-but-textual files (config/env/plain text we still want in the KB) fall
back to a line-window chunker so nothing is silently dropped. Our additions over the
library are glue: a `# path:` scope header on each chunk (the finding that a scope
header drives most of the retrieval gain), 1-based line ranges for `path:line`
citations (derived from chonkie's char offsets), a best-effort symbol label, and a
repo walker.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import fnmatch
import logging
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Set, Tuple

from chonkie import CodeChunker
from core.code_kb.models import CodeChunk
from core.code_kb.source_policy import (
    DEFAULT_MAX_CHUNK_SIZE,
    FALLBACK_EXTS,
    LANG_BY_EXT,
    LANG_BY_NAME,
    MAX_FILE_BYTES,
    DOC_EXTS,
    resolve_language,
    source_extension,
)

logger = logging.getLogger(__name__)

# FFmpeg's 431 KiB dcadata.c proved that tree-sitter-language-pack's recursive splitter
# can exhaust a default Windows worker stack. Such sources used to get a window as large
# as the file, so a 9,255-line C++ file came back as one 567K-char hit (2026-10-13).
# Sources this large are now cut at blank lines into pieces of at most
# LARGE_SOURCE_SEGMENT_CHARS: a piece with blank lines at both ends holds whole
# declarations and is AST-chunked with the normal window; a piece cut inside one huge
# node (a data table) is line-windowed, which never parses and takes milliseconds.
LARGE_SOURCE_THRESHOLD_CHARS = 384 * 1024
LARGE_SOURCE_SEGMENT_CHARS = 64 * 1024
_BLANK_LINE_RE = re.compile(r"\n[ \t\r\f\v]*\n")
CHUNK_WORKER_STACK_SIZE = 64 * 1024 * 1024
CHUNK_WORKER_START_TIMEOUT_SECONDS = 30

# chonkie splits an oversized function at child-node boundaries and leaves the parent's
# closing/opening braces as standalone slivers ("}\n\n", "{\n    "). Embedded, those are
# nothing but the path header — they score identically and bury real hits (W157). Any
# chunk with fewer non-whitespace chars than this is merged into its neighbor instead.
MIN_CHUNK_NWS = 24

# tree-sitter's tree traversal recurses per nesting level and C-stack-overflows around
# ~1200 bracket depth on Windows — a process-killing crash Python cannot catch (W155b).
# Sources deeper than this (minified JS, generated JSON) go straight to line windows.
MAX_AST_NESTING = 500

_LABEL_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:pub\s+)?(?:def|class|function|public|"
                       r"private|protected|func|fn|interface|struct|enum|impl|trait|type|val|"
                       r"var|const|let)\s+([A-Za-z_]\w*)", re.M)


_BUILDER_LOCAL = threading.local()
_CHUNK_EXECUTOR_START_LOCK = threading.Lock()


def _require_offline_grammar(language: str) -> None:
    """Refuse Chonkie's implicit network bootstrap on the indexing hot path.

    Current ``CodeChunker`` construction calls the language pack's ``download_all``.
    Compute nodes may intentionally have no public egress, so only construct it when
    its own offline threshold is already satisfied and the requested grammar is
    loadable. The caller catches this error and uses lossless line windows without
    waiting on the network.
    """

    try:
        from tree_sitter_language_pack import downloaded_languages, has_language

        downloaded = set(downloaded_languages())
        language_ready = has_language(language)
    except Exception as exc:
        raise RuntimeError(
            "tree-sitter grammar cache cannot be inspected; provision the matching "
            "tree-sitter-language-pack cache before starting indexing"
        ) from exc

    # Mirror Chonkie's current constructor gates without invoking either download path:
    # <=19 languages triggers download_all(), while a missing requested language triggers
    # download([language]). Both are forbidden from the offline indexing hot path.
    if len(downloaded) <= 19 or not language_ready:
        raise RuntimeError(
            "tree-sitter grammar cache is incomplete "
            f"({len(downloaded)} languages; requested grammar: {language}); "
            "provision the matching cache before starting indexing"
        )


def _builder(language: str, max_chunk_size: int) -> CodeChunker:
    """Return a builder cached inside the current worker thread."""
    cache = getattr(_BUILDER_LOCAL, "builders", None)
    if cache is None:
        cache = {}
        _BUILDER_LOCAL.builders = cache
    key = (language, max_chunk_size)
    builder = cache.get(key)
    if builder is None:
        _require_offline_grammar(language)
        builder = CodeChunker(
            language=language,
            tokenizer="character",
            chunk_size=max_chunk_size,
            include_nodes=False,
        )
        cache[key] = builder
    return builder


def _label(code: str) -> str:
    m = _LABEL_RE.search(code)
    return m.group(1) if m else ""


def _max_nesting(source: str) -> int:
    """Peak bracket-nesting depth — cheap proxy for AST depth, computed WITHOUT parsing
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


def create_chunk_executor(worker_count: int) -> ThreadPoolExecutor:
    """Create prestarted AST workers with large stacks, then restore the global default."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")

    with _CHUNK_EXECUTOR_START_LOCK:
        original_stack_size = threading.stack_size()
        stack_size_changed = original_stack_size == 0 or original_stack_size < CHUNK_WORKER_STACK_SIZE
        ready = threading.Barrier(worker_count + 1)
        release = threading.Event()
        executor: ThreadPoolExecutor | None = None
        warmups = []

        def hold_worker() -> None:
            ready.wait(timeout=CHUNK_WORKER_START_TIMEOUT_SECONDS)
            release.wait()

        try:
            if stack_size_changed:
                threading.stack_size(CHUNK_WORKER_STACK_SIZE)
            executor = ThreadPoolExecutor(max_workers=worker_count)
            warmups = [executor.submit(hold_worker) for _ in range(worker_count)]
            ready.wait(timeout=CHUNK_WORKER_START_TIMEOUT_SECONDS)
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


def chunk_code(path: str, source: str, language: str = "python",
               max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> List[CodeChunk]:
    """Chunk one source string (AST-aware, via chonkie); attach path header + line ranges.

    Line ranges are 1-based, derived by counting newlines before and inside each chunk
    (trailing blank lines stripped so a citation points at the last line of real code)."""
    depth = _max_nesting(source)
    if depth > MAX_AST_NESTING:
        # Would C-stack-overflow the parser (uncatchable, kills the process — W155b).
        logger.warning("code_kb: %s nesting depth %d > %d — line-window fallback, no AST",
                       path, depth, MAX_AST_NESTING)
        return chunk_lines(path, source, max_chunk_size)
    if len(source) < LARGE_SOURCE_THRESHOLD_CHARS:
        return _ast_chunks(path, source, language, max_chunk_size)
    chunks: List[CodeChunk] = []
    line_offset = 0
    for lo, hi, structural in _segments(source, LARGE_SOURCE_SEGMENT_CHARS):
        piece = source[lo:hi]
        part = (_ast_chunks(path, piece, language, max_chunk_size) if structural
                else chunk_lines(path, piece, max_chunk_size))
        chunks.extend(replace(chunk, start_line=chunk.start_line + line_offset,
                              end_line=chunk.end_line + line_offset) for chunk in part)
        line_offset += piece.count("\n")
    return chunks


def _ast_chunks(path: str, source: str, language: str,
                max_chunk_size: int) -> List[CodeChunk]:
    builder = _builder(language, max_chunk_size)
    spans = _char_spans(source, builder.chunk(source))  # [start, end) into source

    # Merge brace-sliver fragments (see MIN_CHUNK_NWS) into a neighbor: a tiny span
    # joins the previous real chunk; tiny spans at the head wait and prepend to the
    # first real chunk. Spans are contiguous (chonkie is lossless), so extending an
    # end index only ever absorbs the sliver itself plus dropped whitespace.
    merged: List[List[int]] = []
    pending: Optional[List[int]] = None
    for s in spans:
        if len("".join(source[s[0]:s[1]].split())) < MIN_CHUNK_NWS:
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

    chunks: List[CodeChunk] = []
    for lo, hi in merged:
        code = source[lo:hi]
        start = source.count("\n", 0, lo) + 1
        end = start + code.rstrip("\n").count("\n")
        chunks.append(CodeChunk(
            path=path, start_line=start, end_line=max(end, start),
            content=f"# path: {path}\n{code}", code=code, label=_label(code),
        ))
    return chunks


def chunk_lines(path: str, source: str,
                max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> List[CodeChunk]:
    """Line-window fallback for grammarless files: pack whole lines into windows of
    ~max_chunk_size non-whitespace chars so a file is never silently dropped. No symbol
    label — these tiers have no AST, so a code-symbol regex would only misfire.

    A single line longer than 4×max_chunk_size (minified bundles are one huge line) is
    hard-split at max_chunk_size so no chunk balloons past embedding/request limits."""
    hard_cap = 4 * max_chunk_size
    chunks: List[CodeChunk] = []
    lines = source.splitlines()
    buf: List[str] = []
    buf_start = 1
    size = 0

    def _flush_hard_split(line: str, lineno: int) -> None:
        for off in range(0, len(line), max_chunk_size):
            piece = line[off:off + max_chunk_size]
            chunks.append(CodeChunk(path=path, start_line=lineno, end_line=lineno,
                                    content=f"# path: {path}\n{piece}", code=piece))

    def _flush(end_line: int) -> None:
        """Emit the buffer as one chunk — only when it holds non-whitespace text (an
        all-blank window would be an empty `code`, which the server rejects with 422) —
        and always empty it; the next window starts after `end_line`."""
        nonlocal buf, buf_start, size
        if buf and "".join(buf).strip():
            code = "\n".join(buf)
            chunks.append(CodeChunk(path=path, start_line=buf_start, end_line=end_line,
                                    content=f"# path: {path}\n{code}", code=code))
        buf, buf_start, size = [], end_line + 1, 0

    for i, line in enumerate(lines, 1):
        if len(line) > hard_cap:
            _flush(i - 1)
            buf_start = i + 1
            _flush_hard_split(line, i)
            continue
        line_nws = len(re.sub(r"\s", "", line))
        if buf and size + line_nws > max_chunk_size:
            _flush(i - 1)
        buf.append(line)
        size += line_nws
    _flush(len(lines))
    return chunks


def chunk_source(
    path: str,
    source: str,
    *,
    language: str | None,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> List[CodeChunk]:
    """Chunk an already-read source string through the shared safe fallback path."""

    if language:
        try:
            return chunk_code(
                path,
                source,
                language=language,
                max_chunk_size=max_chunk_size,
            )
        except Exception:
            logger.exception(
                "code_kb: AST chunk failed for %s (%s) - line-window fallback",
                path,
                language,
            )
    return chunk_lines(path, source, max_chunk_size)


_ext_of = source_extension


def chunk_file(path: str, repo_root: str = "") -> List[CodeChunk]:
    ext = _ext_of(path)
    language = resolve_language(path)
    if not language and ext not in FALLBACK_EXTS:
        return []
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            logger.warning("code_kb: skipping %s (> %d bytes — generated/vendored, not code)",
                           path, MAX_FILE_BYTES)
            return []
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []
    rel = (os.path.relpath(path, repo_root) if repo_root else path).replace("\\", "/")
    # Log BEFORE parsing: if a grammar hard-crashes the process (the astchunk class of
    # bug, W155), this line is the last thing on stderr and names the exact culprit file.
    logger.info("code_kb: chunking %s (%s, %d bytes)", rel, language or "line-window", len(source))
    return chunk_source(rel, source, language=language)


DEFAULT_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
    ".next", "coverage", ".pytest_cache", ".mypy_cache", ".code_kb", "target",
    # Project layout names are source by default; projects can opt out through
    # .gitignore or .retrievalignore instead of a global directory-name guess.
}


def load_retrievalignore(repo_root: str) -> List[str]:
    try:
        lines = Path(repo_root, ".retrievalignore").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line.strip().replace("\\", "/") for line in lines
            if line.strip() and not line.lstrip().startswith("#")]


def path_excluded(rel_path: str, patterns: List[str]) -> bool:
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


_GIT_LIST_TIMEOUT_S = 30


class GitIgnoreCheckFailed(RuntimeError):
    """Git was asked which paths are ignored and did not answer (timeout, corrupt index,
    any exit other than "not a repository"). Every ignore check fails closed: a caller
    that cannot learn what is ignored must not treat everything as allowed.
    NOTE: mirrors mcp_plugin/retrieval_mcp.py GitIgnoreCheckFailed."""


def gitignore_allowed(root: str, pathspecs: Sequence[str] = ()) -> Optional[Set[str]]:
    """Repo-relative paths (forward slashes) that git does NOT ignore under `root`, or
    None if `root` isn't a git working tree / git isn't installed (callers then apply no
    gitignore filtering). ``pathspecs`` (relative to `root`, matched literally) narrow the
    listing to those paths; the output stays relative to `root`. Any other failure raises
    GitIgnoreCheckFailed (fail closed).

    Delegates the ignore resolution to git itself, so nested .gitignore files, negation
    (``!``) rules, ``.git/info/exclude`` and the user's global ``core.excludesFile`` are
    all honored — a hand-rolled parser gets those subtly wrong. ``--cached`` lists tracked
    files, ``--others`` untracked ones, ``--exclude-standard`` drops the ignored set;
    ``-z`` NUL-separates so unicode/space paths match our os.walk rels without quoting.
    Decoded as UTF-8 (git's path encoding) rather than the ambient locale, so non-ASCII
    paths resolve identically on the Windows (cp936) and Linux (UTF-8) deploy hosts."""
    command = ["git", "--literal-pathspecs", "-C", root, "ls-files",
               "--cached", "--others", "--exclude-standard", "-z"]
    if pathspecs:
        command += ["--", *pathspecs]
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # git's "not a git repository" text is matched below
    try:
        # stdin=DEVNULL: as an MCP stdio server this process's stdin is the JSON-RPC pipe,
        # and a child inheriting it can wedge on that handle (W159, same as _git_commit).
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


def drop_gitignored(paths: List[str], root: str,
                    respect_gitignore: bool = True) -> Tuple[List[str], int]:
    """Filter abs `paths` to those git does NOT ignore under `root`; returns
    (kept, dropped_count). A no-op (paths unchanged) when respect_gitignore is False or
    `root` isn't a git repo. The list-filter counterpart of gitignore_allowed, for
    callers that already have a candidate file list (e.g. a glob) rather than a walk."""
    if not respect_gitignore:
        return paths, 0
    allowed = gitignore_allowed(root)
    if allowed is None:
        return paths, 0
    kept = [p for p in paths if os.path.relpath(p, root).replace(os.sep, "/") in allowed]
    return kept, len(paths) - len(kept)


def iter_repo_chunks(repo_root: str, skip_dirs: Optional[set] = None,
                     respect_gitignore: bool = True,
                     exclude_globs: Optional[List[str]] = None) -> Iterator[CodeChunk]:
    """Yield CodeChunk for every supported source file under repo_root.

    respect_gitignore (default True) skips files git ignores — see gitignore_allowed."""
    t0 = time.time()
    n_files = n_chunks = 0
    logger.info("code_kb: walking repo %s", repo_root)
    for path, _rel in iter_repo_files(
        repo_root, skip_dirs=skip_dirs, respect_gitignore=respect_gitignore,
        exclude_globs=exclude_globs,
    ):
        chunks = chunk_file(path, repo_root)
        if chunks:
            n_files += 1
            n_chunks += len(chunks)
        yield from chunks
    logger.info("code_kb: repo %s chunked — %d files, %d chunks in %.1fs",
                repo_root, n_files, n_chunks, time.time() - t0)


def iter_repo_files(repo_root: str, skip_dirs: Optional[set] = None,
                    respect_gitignore: bool = True,
                    exclude_globs: Optional[List[str]] = None) -> Iterator[Tuple[str, str]]:
    """Yield (abs_path, rel_path) for every file that WOULD be chunked — a resolvable
    language or a fallback ext — WITHOUT parsing anything. Lets a client list + hash
    candidates cheaply and AST-chunk only the changed subset (incremental indexing).

    The selection mirrors iter_repo_chunks's file gate exactly, so a manifest built
    from this walk matches the set of files a full reindex would touch (files that a
    full index skips post-gate — too big, unreadable, empty — simply produce no chunks
    and drop out of the manifest naturally; see the incremental collectors).

    respect_gitignore (default True) skips files git ignores — see gitignore_allowed.

    The code-only view of walk_repo_files."""
    for path, rel, _kind in walk_repo_files(
            repo_root, skip_dirs=skip_dirs, respect_gitignore=respect_gitignore,
            exclude_globs=exclude_globs, docs=False):
        yield path, rel


def is_hidden_rel_path(rel_path: str) -> bool:
    """True when any segment of a scope-relative path is a dot name (``.github/x.md``)."""
    return any(part.startswith(".") for part in rel_path.replace("\\", "/").split("/"))


def is_doc_path(path: str, rel_path: str) -> bool:
    """A documentation file the journal pipeline owns: a DOC_EXTS extension outside every
    dot path (the journal's sweep envelope never covers those)."""
    return _ext_of(path) in DOC_EXTS and not is_hidden_rel_path(rel_path)


def walk_repo_files(repo_root: str, skip_dirs: Optional[set] = None,
                    respect_gitignore: bool = True,
                    exclude_globs: Optional[List[str]] = None,
                    docs: bool = True) -> Iterator[Tuple[str, str, str]]:
    """Yield (abs_path, rel_path, kind) for every indexable file from ONE directory walk.

    kind is "code" (a resolvable language or a fallback ext) or "doc" (DOC_EXTS; ``docs=False``
    omits them). Code wins when a name qualifies as both (``CMakeLists.txt``). The skipped
    directories, exclude globs, ``.retrievalignore`` and git's ignore rules gate both kinds
    identically."""
    skip = skip_dirs or DEFAULT_SKIP_DIRS
    allowed = gitignore_allowed(repo_root) if respect_gitignore else None
    excludes = list(dict.fromkeys(load_retrievalignore(repo_root) + list(exclude_globs or [])))
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in skip and not path_excluded(
            os.path.relpath(os.path.join(root, d), repo_root).replace("\\", "/"), excludes)]
        for fn in files:
            path = os.path.join(root, fn)
            if resolve_language(path) or _ext_of(path) in FALLBACK_EXTS:
                kind = "code"
            elif docs and _ext_of(path) in DOC_EXTS:
                kind = "doc"
            else:
                continue
            rel = os.path.relpath(path, repo_root).replace("\\", "/")
            if kind == "doc" and is_hidden_rel_path(rel):
                continue
            if path_excluded(rel, excludes):
                continue
            if allowed is not None and rel not in allowed:
                continue
            yield path, rel, kind


def file_content_hash(path: str) -> Optional[str]:
    """SHA-256 hex of a file's bytes, or None if unreadable. This is the content
    identity used for incremental indexing: a file whose hash equals the last indexed
    hash (from the server inventory) is unchanged and is skipped without re-chunking.

    Hashing is chosen over git-diff deliberately — it is drift-proof (the server
    inventory is the single source of truth, so force-push / rename / untracked /
    gitignore divergence can't desync the index) and its cost is negligible next to
    the AST parse + embed it lets us skip for unchanged files."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None
