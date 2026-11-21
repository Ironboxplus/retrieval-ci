"""Incremental duplicate scan (`scan_changed_files`) against a real Qdrant.

The full scan asks "which chunks in this scope duplicate which" and costs a query per
chunk of the scope: measured on `.88`, 17,839 chunks take ~137 s and a 562,933-chunk
scope is infeasible. The incremental scan queries only the chunks of the files that
changed, so its cost follows the change, not the repository.

What only a real Qdrant can show: neighbours come back for chunks that were never
scrolled (so their payload must be fetched), the owner/project filter is applied by
Qdrant, and a MatchAny over many paths behaves like the rest of the filter.
"""
from __future__ import annotations

import pytest

from app.services.code_kb import duplicates
from app.services.code_kb.duplicates import scan_changed_files
from tests.code_analysis_support import OWNER, PROJECT, env, vector_store  # noqa: F401


def _scan(env, changed, *, project=PROJECT, owner=OWNER, **kwargs):
    return scan_changed_files(
        env.vector_store, owner_user_id=owner, project=project,
        changed_paths=changed, **kwargs)


def _pairs(result):
    return [(p.path_a, p.path_b) for p in result.pairs]


def test_a_changed_file_is_matched_against_files_that_were_not_scanned(env):
    env.index(PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    result = _scan(env, ["app/alpha.py"])

    assert _pairs(result) == [("app/alpha.py", "app/beta.py")]
    pair = result.pairs[0]
    assert pair.shared_chunks == 2
    assert pair.max_score == pytest.approx(1.0, abs=1e-4)
    # beta was never scrolled: its citation can only come from the neighbour's payload.
    example = pair.examples[0]
    assert example.b.path == "app/beta.py"
    assert example.b.start_line >= 1 and example.b.end_line >= example.b.start_line
    assert example.b.label.startswith("chunk")
    # Only the changed file's chunks were queried, which is the whole point.
    assert result.scanned_chunks == 2 and result.scanned_files == 1
    assert not result.too_large


def test_pairs_that_do_not_involve_a_changed_file_are_not_reported(env):
    env.index(PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1],
                        "app/gamma.py": [2, 3], "app/delta.py": [2, 3]})
    assert _pairs(_scan(env, ["app/alpha.py"])) == [("app/alpha.py", "app/beta.py")]
    assert _pairs(_scan(env, ["app/gamma.py"])) == [("app/delta.py", "app/gamma.py")]


def test_one_shared_chunk_is_noise_until_the_minimum_is_lowered(env):
    env.index(PROJECT, {"app/alpha.py": [0, 1], "app/gamma.py": [0, 2]})
    assert _pairs(_scan(env, ["app/alpha.py"])) == []
    assert _pairs(_scan(env, ["app/alpha.py"], min_shared_chunks=1)) == [
        ("app/alpha.py", "app/gamma.py")]


def test_excluded_paths_are_neither_queried_nor_reported_as_targets(env):
    env.index(PROJECT, {"app/alpha.py": [0, 1], "storage/dump.py": [0, 1]})
    assert _pairs(_scan(env, ["app/alpha.py"])) == []
    # A changed file that is itself excluded costs nothing at all.
    excluded = _scan(env, ["storage/dump.py"])
    assert excluded.pairs == [] and excluded.scanned_chunks == 0
    # Without the exclusion the duplicate is real, so the exclusion is what hid it.
    assert _pairs(_scan(env, ["app/alpha.py"], exclude_globs=())) == [
        ("app/alpha.py", "storage/dump.py")]


def test_chunks_of_one_file_are_not_duplicates_of_each_other(env):
    env.index(PROJECT, {"app/repeat.py": [0, 0, 0]})
    assert _pairs(_scan(env, ["app/repeat.py"], min_shared_chunks=1)) == []


def test_another_owner_and_another_project_stay_out_of_the_report(env):
    env.index(PROJECT, {"app/alpha.py": [0, 1]})
    env.index(PROJECT, {"app/theirs.py": [0, 1]}, owner="other-owner")
    env.index("/other-repo", {"app/elsewhere.py": [0, 1]})
    assert _scan(env, ["app/alpha.py"], min_shared_chunks=1).pairs == []
    assert _scan(env, ["app/theirs.py"], owner="other-owner", min_shared_chunks=1).pairs == []


def test_scope_spellings_do_not_multiply_the_evidence(env):
    """The same repo indexed under three spellings must read as one repo (see
    TestAliasScopesDoNotMultiplyTheEvidence in test_code_duplicates_real_qdrant.py)."""
    spellings = ["/repo", "/Repo", "/REPO"]
    for spelling in spellings:
        env.index(spelling, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1],
                             "app/gamma.py": [0, 2]})
    one = _scan(env, ["app/alpha.py"], project=spellings[0])
    allof = _scan(env, ["app/alpha.py"], project=spellings)

    assert _pairs(allof) == _pairs(one) == [("app/alpha.py", "app/beta.py")]
    assert allof.pairs[0].shared_chunks == one.pairs[0].shared_chunks == 2, (
        "alpha and beta share two chunks, however many spellings hold them")
    assert allof.scanned_chunks == one.scanned_chunks == 2
    citations = [(e.a.start_line, e.b.start_line) for e in allof.pairs[0].examples]
    assert len(citations) == len(set(citations))
    # The single-chunk noise gate must not be cleared by counting a chunk once per spelling.
    assert ("app/alpha.py", "app/gamma.py") not in _pairs(allof)


def test_a_change_over_the_chunk_cap_is_reported_not_scanned(env):
    env.index(PROJECT, {"app/big.py": [0, 1, 2, 3], "app/twin.py": [0, 1, 2, 3]})
    result = _scan(env, ["app/big.py"], max_chunks=3)
    assert result.too_large
    assert result.pairs == []
    # The scan stopped at the cap instead of reading the whole change first.
    assert result.scanned_chunks <= 3


def test_excluded_chunks_do_not_spend_the_cap(env):
    env.index(PROJECT, {"storage/a.py": [0, 1, 2, 3], "app/real.py": [4, 5],
                        "app/twin.py": [4, 5]})
    result = _scan(env, ["storage/a.py", "app/real.py"], max_chunks=2)
    assert not result.too_large
    assert _pairs(result) == [("app/real.py", "app/twin.py")]


def test_a_path_with_no_indexed_chunks_is_an_empty_scan(env):
    env.index(PROJECT, {"app/alpha.py": [0, 1]})
    result = _scan(env, ["app/never-indexed.py"])
    assert result.pairs == [] and result.scanned_chunks == 0 and not result.too_large


def test_hundreds_of_changed_paths_are_scanned_in_filter_sized_groups(env):
    env.index(PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    changed = [f"gone/f{i}.py" for i in range(450)] + ["app/alpha.py"]
    assert _pairs(_scan(env, changed)) == [("app/alpha.py", "app/beta.py")]


def test_two_changed_files_that_duplicate_each_other_are_one_pair(env):
    env.index(PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    result = _scan(env, ["app/alpha.py", "app/beta.py"])
    assert _pairs(result) == [("app/alpha.py", "app/beta.py")]
    # alpha->beta and beta->alpha describe the same two shared chunks, not four.
    assert result.pairs[0].shared_chunks == 2
    assert result.scanned_files == 2


def test_the_scan_spans_every_query_batch(env, monkeypatch):
    monkeypatch.setattr(duplicates, "_QUERY_BATCH", 2)
    env.index(PROJECT, {"app/alpha.py": [0, 1, 2, 3, 4], "app/beta.py": [0, 1, 2, 3, 4]})
    result = _scan(env, ["app/alpha.py"])
    assert result.pairs[0].shared_chunks == 5
