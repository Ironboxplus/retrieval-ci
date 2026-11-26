"""Roll chunk-level near-duplicate matches up into file-level findings.

Why the file pair, and not the chunk pair, is the reporting unit: a 2026-09-01
probe ran every one of this repository's own indexed code chunks against its own
`code_chunks` collection (8,624 nearest-neighbour queries, 28.7s, 0 errors) and
measured where the noise actually lives. Score saturation is not the problem —
only 4 chunks reached 0.90 similarity and none reached 0.92 — so tightening the
threshold barely changes the output. What floods a chunk-level report is
one-off boilerplate collisions: 176 of 190 candidate file pairs shared exactly
ONE chunk.

Requiring two shared chunks cut the 0.92 set from 47 pairs to 14, and manual
review found no false positive among the 14. The top finding was
mcp_plugin/retrieval_mcp.py against mcp_server/server.py at 65 shared chunks,
which are character-identical over most of their length — a real duplicated
design that no chunk-level listing would have surfaced above the noise.

So: `threshold` decides what counts as a matching chunk, and `min_shared_chunks`
is the filter that decides what counts as a finding. This module is pure — it
takes matches and returns findings — so the ranking rules stay testable without
a vector database. app/services/code_kb/duplicates.py supplies the matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pathspec

# Vendored, generated and archived trees dominate any similarity report and none
# of them are duplication a human would act on. `/storage/` alone is 163 tracked
# files of Qdrant segment data in this repository. Patterns use gitignore syntax,
# so a bare `name/` matches that directory at any depth while a leading slash
# anchors it to the scope root.
#
# `/storage/` MUST stay anchored: unanchored it also swallows `core/storage/`,
# which is this repository's real Qdrant client code. Data directories sit at the
# scope root; a nested `storage/` is far more likely to be source.
DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "/storage/",
    ".git/",
    "node_modules/",
    "archive/",
    "__pycache__/",
    ".venv/",
    "site-packages/",
    "dist/",
    "*.min.js",
    "*.lock",
)


@dataclass(frozen=True)
class ChunkRef:
    """One indexed code chunk, identified by its Qdrant point id."""
    point_id: int
    path: str
    start_line: int
    end_line: int
    label: str = ""


@dataclass(frozen=True)
class ChunkMatch:
    """`source` and `target` scored as near-duplicates of each other."""
    source: ChunkRef
    target: ChunkRef
    score: float


@dataclass(frozen=True)
class DuplicateChunkPair:
    """One shared chunk, oriented to match its parent pair's (path_a, path_b)."""
    a: ChunkRef
    b: ChunkRef
    score: float


@dataclass(frozen=True)
class DuplicateFilePair:
    path_a: str
    path_b: str
    shared_chunks: int
    max_score: float
    mean_score: float
    examples: tuple[DuplicateChunkPair, ...] = field(default=())


def _spec(globs: Sequence[str], *, case_sensitive: bool) -> pathspec.PathSpec | None:
    if not globs:
        return None
    lines = globs if case_sensitive else [g.casefold() for g in globs]
    return pathspec.PathSpec.from_lines("gitignore", lines)


def is_excluded(path: str, globs: Sequence[str], *, case_sensitive: bool = True) -> bool:
    """Does this scope-relative path fall in an excluded tree?"""
    spec = _spec(globs, case_sensitive=case_sensitive)
    if spec is None:
        return False
    candidate = path.replace("\\", "/").lstrip("/")
    if not case_sensitive:
        candidate = candidate.casefold()
    return spec.match_file(candidate)


def roll_up_file_pairs(
    matches: Iterable[ChunkMatch],
    *,
    min_shared_chunks: int = 2,
    max_pairs: int | None = 50,
    max_examples: int = 3,
    exclude_globs: Sequence[str] = (),
    case_sensitive: bool = True,
) -> list[DuplicateFilePair]:
    """Group chunk matches by file pair and keep the ones worth reporting.

    ``min_shared_chunks`` is the primary filter (see the module docstring) and must
    be at least 1: zero would report file pairs that share nothing.
    """
    if min_shared_chunks < 1:
        raise ValueError(
            f"min_shared_chunks must be >= 1, got {min_shared_chunks}; "
            "0 would report file pairs that share no chunk at all")

    spec = _spec(exclude_globs, case_sensitive=case_sensitive)

    def excluded(path: str) -> bool:
        if spec is None:
            return False
        candidate = path.replace("\\", "/").lstrip("/")
        return spec.match_file(candidate if case_sensitive else candidate.casefold())

    # Keyed by the unordered chunk pair, because every chunk queries its own
    # neighbours: A->B and B->A describe ONE shared chunk, and counting both
    # would double every n and let boilerplate clear a min_shared_chunks gate.
    per_pair: dict[tuple[str, str], dict[tuple[int, int], DuplicateChunkPair]] = {}
    for m in matches:
        if m.source.path == m.target.path:
            continue  # adjacent chunks of one file are not duplicated design
        if excluded(m.source.path) or excluded(m.target.path):
            continue
        # Orient on path so a pair has exactly one identity regardless of which
        # side happened to be the querying chunk.
        if m.source.path <= m.target.path:
            key, a, b = (m.source.path, m.target.path), m.source, m.target
        else:
            key, a, b = (m.target.path, m.source.path), m.target, m.source
        chunk_key = (min(m.source.point_id, m.target.point_id),
                     max(m.source.point_id, m.target.point_id))
        bucket = per_pair.setdefault(key, {})
        prior = bucket.get(chunk_key)
        if prior is None or m.score > prior.score:
            bucket[chunk_key] = DuplicateChunkPair(a=a, b=b, score=m.score)

    findings: list[DuplicateFilePair] = []
    for (path_a, path_b), bucket in per_pair.items():
        if len(bucket) < min_shared_chunks:
            continue
        pairs = sorted(bucket.items(), key=lambda kv: (-kv[1].score, kv[0]))
        scores = [cp.score for _, cp in pairs]
        findings.append(DuplicateFilePair(
            path_a=path_a,
            path_b=path_b,
            shared_chunks=len(bucket),
            max_score=max(scores),
            mean_score=sum(scores) / len(scores),
            examples=tuple(cp for _, cp in pairs[:max_examples]),
        ))

    # n first: the probe showed n separates real findings far better than score.
    findings.sort(key=lambda f: (-f.shared_chunks, -f.max_score, f.path_a, f.path_b))
    if max_pairs is not None:
        findings = findings[:max_pairs]
    return findings
