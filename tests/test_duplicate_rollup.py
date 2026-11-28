"""File-level roll-up for duplicate-code detection.

The 2026-09-01 design probe over this repository's own `code_chunks` measured
what actually separates real duplicates from noise, and it is NOT the score
threshold: raising it from 0.90 to 0.92 barely moved the result set, because
176 of 190 candidate file pairs shared exactly ONE chunk each. Those singletons
are boilerplate collisions (imports, `if __name__ == "__main__"`, identical
dataclass headers), not duplicated designs.

The discriminating quantity is `n` — how many distinct chunk pairs two files
share. Adding `n >= 2` cut the 0.92 result set from 47 pairs to 14, and all 14
were genuine. So the file pair is the reporting unit and `min_shared_chunks` is
the real filter; the threshold only decides what counts as a chunk match.

These tests drive the roll-up directly with real records — there is no client
here to fake, because the roll-up is pure. tests/test_code_duplicates_real_qdrant.py
covers the Qdrant half against a real instance.
"""
from __future__ import annotations

import pytest

from core.code_kb.duplicate_rollup import (
    ChunkMatch,
    ChunkRef,
    DEFAULT_EXCLUDE_GLOBS,
    is_excluded,
    roll_up_file_pairs,
)


def _ref(point_id: int, path: str, start: int = 1, label: str = "fn") -> ChunkRef:
    return ChunkRef(point_id=point_id, path=path, start_line=start,
                    end_line=start + 9, label=label)


def _match(a_id: int, a_path: str, b_id: int, b_path: str, score: float) -> ChunkMatch:
    return ChunkMatch(source=_ref(a_id, a_path, a_id * 10),
                      target=_ref(b_id, b_path, b_id * 10), score=score)


def test_single_shared_chunk_is_dropped_by_the_default_min():
    """The measured noise shape: one shared chunk between two files is boilerplate."""
    pairs = roll_up_file_pairs([_match(1, "a.py", 2, "b.py", 0.95)], min_shared_chunks=2)
    assert pairs == []


def test_two_shared_chunks_survive_and_report_both():
    matches = [
        _match(1, "a.py", 2, "b.py", 0.95),
        _match(3, "a.py", 4, "b.py", 0.93),
    ]
    pairs = roll_up_file_pairs(matches, min_shared_chunks=2)
    assert len(pairs) == 1
    assert (pairs[0].path_a, pairs[0].path_b) == ("a.py", "b.py")
    assert pairs[0].shared_chunks == 2
    assert pairs[0].max_score == pytest.approx(0.95)
    assert pairs[0].mean_score == pytest.approx(0.94)


def test_reciprocal_matches_count_once_not_twice():
    """Every chunk queries its own neighbours, so A->B and B->A both come back.

    Counting both would double every `n` and let a single shared chunk clear a
    `min_shared_chunks=2` gate — silently reinstating the exact noise the gate
    exists to remove.
    """
    matches = [
        _match(1, "a.py", 2, "b.py", 0.95),
        _match(2, "b.py", 1, "a.py", 0.95),
    ]
    assert roll_up_file_pairs(matches, min_shared_chunks=2) == []
    pairs = roll_up_file_pairs(matches, min_shared_chunks=1)
    assert len(pairs) == 1 and pairs[0].shared_chunks == 1


def test_pair_orientation_is_path_sorted_so_a_pair_has_one_identity():
    matches = [
        _match(1, "z.py", 2, "a.py", 0.99),
        _match(3, "a.py", 4, "z.py", 0.98),
    ]
    pairs = roll_up_file_pairs(matches, min_shared_chunks=2)
    assert len(pairs) == 1
    assert (pairs[0].path_a, pairs[0].path_b) == ("a.py", "z.py")


def test_self_matches_within_one_file_are_not_a_duplicate_pair():
    """A chunk's nearest neighbour is very often the adjacent chunk of the same
    file; reporting that as duplication would bury every real cross-file hit."""
    assert roll_up_file_pairs(
        [_match(1, "a.py", 2, "a.py", 0.99)], min_shared_chunks=1) == []


def test_ranking_is_by_shared_chunks_then_score_then_path():
    matches = [
        _match(1, "a.py", 2, "b.py", 0.99),
        _match(3, "a.py", 4, "b.py", 0.99),
        _match(5, "c.py", 6, "d.py", 0.93),
        _match(7, "c.py", 8, "d.py", 0.93),
        _match(9, "c.py", 10, "d.py", 0.93),
    ]
    pairs = roll_up_file_pairs(matches, min_shared_chunks=2)
    # c/d wins on n=3 even though a/b scores higher — n is the primary signal.
    assert [(p.path_a, p.shared_chunks) for p in pairs] == [("c.py", 3), ("a.py", 2)]


def test_max_pairs_truncates_and_says_so():
    matches = []
    for i in range(6):
        matches.append(_match(i * 4 + 1, f"f{i}.py", i * 4 + 2, f"g{i}.py", 0.95))
        matches.append(_match(i * 4 + 3, f"f{i}.py", i * 4 + 4, f"g{i}.py", 0.95))
    pairs = roll_up_file_pairs(matches, min_shared_chunks=2, max_pairs=2)
    assert len(pairs) == 2


def test_examples_are_capped_but_never_empty():
    matches = [_match(i, "a.py", 100 + i, "b.py", 0.9) for i in range(1, 9)]
    pairs = roll_up_file_pairs(matches, min_shared_chunks=2, max_examples=3)
    assert pairs[0].shared_chunks == 8
    assert len(pairs[0].examples) == 3


def test_examples_are_the_highest_scoring_chunk_pairs():
    matches = [
        _match(1, "a.py", 2, "b.py", 0.90),
        _match(3, "a.py", 4, "b.py", 0.99),
        _match(5, "a.py", 6, "b.py", 0.95),
    ]
    pairs = roll_up_file_pairs(matches, min_shared_chunks=2, max_examples=2)
    assert [e.score for e in pairs[0].examples] == pytest.approx([0.99, 0.95])


@pytest.mark.parametrize("path", [
    "storage/collections/papers/segments/0.dat",
    ".git/objects/ab/cdef",
    "node_modules/react/index.js",
    "frontend/node_modules/lodash/lodash.js",
    "docs/archive/old_plan.md",
    "app/__pycache__/service.cpython-313.pyc",
])
def test_default_exclusions_cover_the_paths_that_flooded_the_probe(path):
    """`storage/` alone is 163 tracked files of Qdrant segment data in this repo;
    without these, the report is nothing but vendored and generated files."""
    assert is_excluded(path, DEFAULT_EXCLUDE_GLOBS)


@pytest.mark.parametrize("path", [
    "app/services/code_kb/service.py",
    "mcp_plugin/retrieval_mcp.py",
    "core/storage/qdrant_store.py",
    "storage_manager.py",
])
def test_default_exclusions_do_not_eat_real_source(path):
    assert not is_excluded(path, DEFAULT_EXCLUDE_GLOBS)


def test_excluded_files_are_dropped_during_roll_up():
    matches = [
        _match(1, "storage/a.dat", 2, "storage/b.dat", 0.99),
        _match(3, "storage/a.dat", 4, "storage/b.dat", 0.99),
        _match(5, "app/x.py", 6, "app/y.py", 0.95),
        _match(7, "app/x.py", 8, "app/y.py", 0.95),
    ]
    pairs = roll_up_file_pairs(matches, min_shared_chunks=2,
                               exclude_globs=DEFAULT_EXCLUDE_GLOBS)
    assert [(p.path_a, p.path_b) for p in pairs] == [("app/x.py", "app/y.py")]


def test_a_pair_is_excluded_when_either_side_is_excluded():
    matches = [
        _match(1, "app/x.py", 2, "node_modules/y.js", 0.99),
        _match(3, "app/x.py", 4, "node_modules/y.js", 0.99),
    ]
    assert roll_up_file_pairs(matches, min_shared_chunks=2,
                             exclude_globs=DEFAULT_EXCLUDE_GLOBS) == []


def test_case_insensitive_exclusion_for_windows_scopes():
    """Windows scopes reach Qdrant with whatever casing the client sent, and this
    project already merges `C:/x` and `c:/x` scope variants for that reason."""
    assert is_excluded("Storage/Collections/x.dat", DEFAULT_EXCLUDE_GLOBS,
                       case_sensitive=False)
    assert not is_excluded("Storage/Collections/x.dat", DEFAULT_EXCLUDE_GLOBS,
                           case_sensitive=True)


def test_min_shared_chunks_below_one_is_rejected_rather_than_silently_clamped():
    """n>=1 is already the noisy setting; n<=0 would report every file pair that
    shares nothing at all, so it is a caller bug, not a preference."""
    with pytest.raises(ValueError):
        roll_up_file_pairs([], min_shared_chunks=0)
