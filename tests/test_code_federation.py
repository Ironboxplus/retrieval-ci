"""TDD for the federated code-search fusion pure function (docs/federated_code_search_plan.md A2).

No Qdrant/service I/O here by design (AGENTS.md no-mock-tests rule does not apply: the
module under test is a pure function over deterministic in-memory candidates, which the
plan explicitly carves out as the one place a fixed-score reranker stand-in is allowed).
"""
from __future__ import annotations

import math

import pytest

from app.services.code_federation import (
    CodeFederationCandidate,
    FusedResult,
    fuse_code_hits,
)


def _cand(origin, path, *, start_line=1, end_line=10, score=0.5, label="", language=None,
          code=None, project=None, host=None, source_id=None, revision=None, domain=None):
    return CodeFederationCandidate(
        origin=origin, path=path, start_line=start_line, end_line=end_line,
        score=score, label=label, language=language, code=code or f"code for {path}",
        project=project, host=host, source_id=source_id, revision=revision, domain=domain,
    )


class _FixedScoreReranker:
    """Returns one caller-supplied score per candidate, matched by (origin, path)."""

    def __init__(self, scores_by_path: dict[tuple[str, str], float]):
        self._scores_by_path = scores_by_path
        self.calls: list[tuple[str, list[str]]] = []

    def score_pairs(self, query, texts):
        self.calls.append((query, list(texts)))
        return list(texts and [self._scores_by_path[t] for t in texts])


class _CodeKeyedReranker:
    """Scores by the candidate's `code` field (unique per candidate in these tests)."""

    def __init__(self, scores_by_code: dict[str, float]):
        self._scores_by_code = scores_by_code

    def score_pairs(self, query, texts):
        return [self._scores_by_code[t] for t in texts]


class _RaisingReranker:
    def score_pairs(self, query, texts):
        raise RuntimeError("reranker exploded")


class _WrongCountReranker:
    def score_pairs(self, query, texts):
        return [0.9] * (len(texts) - 1) if texts else []


class _NonFiniteReranker:
    def __init__(self, bad_value):
        self._bad_value = bad_value

    def score_pairs(self, query, texts):
        scores = [0.5] * len(texts)
        if scores:
            scores[0] = self._bad_value
        return scores


def test_rrf_interleaves_private_and_public_by_rank_with_private_first_on_ties():
    """item 1: private [a,b], public [x,y] -> a,x,b,y with scores 1/61,1/61,1/62,1/62."""
    private = [_cand("private", "a.py"), _cand("private", "b.py")]
    public = [_cand("public", "x.py"), _cand("public", "y.py")]

    result = fuse_code_hits(private, public, query="q", top_k=10)

    assert isinstance(result, FusedResult)
    assert result.fusion == "rrf"
    assert result.rerank_applied is False
    assert [h.path for h in result.hits] == ["a.py", "x.py", "b.py", "y.py"]
    assert [h.origin for h in result.hits] == ["private", "public", "private", "public"]
    scores = [h.score for h in result.hits]
    assert scores == pytest.approx([1 / 61, 1 / 61, 1 / 62, 1 / 62])
    assert [h.fusion_rank_score for h in result.hits] == pytest.approx(
        [1 / 61, 1 / 61, 1 / 62, 1 / 62])
    assert all(h.reranker_score is None for h in result.hits)


def test_rrf_with_one_leg_empty_preserves_the_other_legs_rank_order():
    """item 2: one leg empty -> the other leg returned in order, score = 1/(60+rank)."""
    public = [_cand("public", "x.py"), _cand("public", "y.py"), _cand("public", "z.py")]

    result = fuse_code_hits([], public, query="q", top_k=10)

    assert [h.path for h in result.hits] == ["x.py", "y.py", "z.py"]
    assert [h.score for h in result.hits] == pytest.approx([1 / 61, 1 / 62, 1 / 63])

    private = [_cand("private", "a.py"), _cand("private", "b.py")]
    result_reverse = fuse_code_hits(private, [], query="q", top_k=10)
    assert [h.path for h in result_reverse.hits] == ["a.py", "b.py"]
    assert [h.score for h in result_reverse.hits] == pytest.approx([1 / 61, 1 / 62])


def test_top_k_truncation_applies_after_global_fusion_not_per_leg():
    """item 3: truncation to top_k happens on the globally-fused ranking. A naive
    implementation that concatenates each leg's own top_k slice (private-first, in list
    order) instead of re-sorting the fused pool would return all 4 private hits and
    drop every public hit; the correct behaviour interleaves public's stronger-ranked
    entries ahead of private's weaker tail."""
    private = [_cand("private", f"p{i}.py") for i in range(1, 5)]  # ranks 1-4
    public = [_cand("public", f"q{i}.py") for i in range(1, 4)]    # ranks 1-3

    result = fuse_code_hits(private, public, query="q", top_k=4)

    assert len(result.hits) == 4
    assert [h.path for h in result.hits] == ["p1.py", "q1.py", "p2.py", "q2.py"]
    # p3.py/p4.py would appear in a naive "top_k of each leg, concatenated" truncation;
    # global fusion correctly pushes them out in favour of q1/q2's higher RRF rank.
    assert "p3.py" not in [h.path for h in result.hits]
    assert "p4.py" not in [h.path for h in result.hits]


def test_rerank_true_promotes_the_rerankers_top_public_hit_to_first():
    """item 4: reranker gives a public hit the top score -> it ranks first,
    fusion="rerank", rerank_applied=True, reranker_score is populated."""
    private = [_cand("private", "a.py", code="code-a"), _cand("private", "b.py", code="code-b")]
    public = [_cand("public", "x.py", code="code-x")]
    reranker = _CodeKeyedReranker({"code-a": 0.1, "code-b": 0.2, "code-x": 0.9})

    result = fuse_code_hits(private, public, query="q", top_k=10, reranker=reranker)

    assert result.fusion == "rerank"
    assert result.rerank_applied is True
    assert result.hits[0].path == "x.py"
    assert result.hits[0].origin == "public"
    assert result.hits[0].reranker_score == pytest.approx(0.9)
    assert result.hits[0].score == pytest.approx(0.9)
    assert [h.reranker_score for h in result.hits] == pytest.approx([0.9, 0.2, 0.1])


def test_rerank_ties_break_by_rrf_score():
    """item 5: reranker score ties -> RRF score is the tie-break."""
    # b.py (private rank2, rrf=1/62) and x.py (public rank1, rrf=1/61) tie at 0.5;
    # x.py's higher RRF score must place it first.
    private = [_cand("private", "a.py", code="code-a"), _cand("private", "b.py", code="code-b")]
    public = [_cand("public", "x.py", code="code-x")]
    reranker = _CodeKeyedReranker({"code-a": 0.9, "code-b": 0.5, "code-x": 0.5})

    result = fuse_code_hits(private, public, query="q", top_k=10, reranker=reranker)

    assert [h.path for h in result.hits] == ["a.py", "x.py", "b.py"]


@pytest.mark.parametrize("reranker", [
    _RaisingReranker(),
    _WrongCountReranker(),
    _NonFiniteReranker(float("nan")),
    _NonFiniteReranker(float("inf")),
])
def test_rerank_failure_modes_fall_back_to_rrf_order(reranker):
    """item 6: reranker exception / wrong score count / NaN / inf -> RRF fallback,
    fusion="rrf", rerank_applied=False, reranker_score all None."""
    private = [_cand("private", "a.py"), _cand("private", "b.py")]
    public = [_cand("public", "x.py")]

    result = fuse_code_hits(private, public, query="q", top_k=10, reranker=reranker)

    assert result.fusion == "rrf"
    assert result.rerank_applied is False
    assert [h.path for h in result.hits] == ["a.py", "x.py", "b.py"]
    assert all(h.reranker_score is None for h in result.hits)


def test_exact_duplicate_chunks_are_deduped_within_a_leg_but_not_across_origins():
    """item 7: an identical chunk repeated within one leg collapses to one entry;
    the same code appearing under a DIFFERENT origin is not merged (distinct sources)."""
    dup = _cand("private", "a.py", code="same code")
    dup_again = _cand("private", "a.py", code="same code")
    same_code_other_origin = _cand("public", "a.py", code="same code")

    result = fuse_code_hits([dup, dup_again], [same_code_other_origin], query="q", top_k=10)

    assert len(result.hits) == 2
    assert sorted((h.origin, h.path) for h in result.hits) == [
        ("private", "a.py"), ("public", "a.py"),
    ]


def test_identical_chunk_in_two_private_scopes_or_two_public_sources_is_not_merged():
    """The same file indexed in two repositories is two results, exactly as
    /api/code/search returns it (cross-scope fan-out copies, vendored files).
    Merging them would make owner-wide federated search drop hits the old route keeps."""
    in_repo_a = _cand("private", "a.py", code="same code", project="C:/repo-a")
    in_repo_b = _cand("private", "a.py", code="same code", project="C:/repo-b")
    in_source_x = _cand("public", "a.py", code="same code", source_id="x", revision="r1")
    in_source_y = _cand("public", "a.py", code="same code", source_id="y", revision="r1")

    result = fuse_code_hits([in_repo_a, in_repo_b], [in_source_x, in_source_y],
                            query="q", top_k=10)

    assert sorted((h.origin, h.project or h.source_id) for h in result.hits) == [
        ("private", "C:/repo-a"), ("private", "C:/repo-b"),
        ("public", "x"), ("public", "y"),
    ]


def test_reranker_receives_the_full_deduplicated_union_not_the_truncated_top_k():
    """item 8: candidates handed to the reranker are the deduped union of both legs,
    not the top_k-sliced result."""
    private = [_cand("private", f"p{i}.py", code=f"code-p{i}") for i in range(1, 4)]
    public = [_cand("public", f"q{i}.py", code=f"code-q{i}") for i in range(1, 4)]
    seen_batch_sizes: list[int] = []

    class _RecordingReranker:
        def score_pairs(self, query, texts):
            seen_batch_sizes.append(len(texts))
            return [0.5] * len(texts)

    fuse_code_hits(private, public, query="q", top_k=2, reranker=_RecordingReranker())

    assert seen_batch_sizes == [6]  # 3 private + 3 public, not top_k=2


def test_fuse_code_hits_carries_origin_specific_fields_through():
    private = [_cand("private", "a.py", project="proj", host="h1")]
    public = [_cand("public", "x.py", source_id="src", revision="deadbeef00", domain="github.com")]

    result = fuse_code_hits(private, public, query="q", top_k=10)

    private_hit = next(h for h in result.hits if h.origin == "private")
    public_hit = next(h for h in result.hits if h.origin == "public")
    assert private_hit.project == "proj" and private_hit.host == "h1"
    assert public_hit.source_id == "src"
    assert public_hit.revision == "deadbeef00"
    assert public_hit.domain == "github.com"


def test_retrieval_score_is_the_legs_own_score_not_the_fused_score():
    private = [_cand("private", "a.py", score=0.87)]
    result = fuse_code_hits(private, [], query="q", top_k=10)
    assert result.hits[0].retrieval_score == pytest.approx(0.87)
    assert result.hits[0].fusion_rank_score == pytest.approx(1 / 61)
    assert result.hits[0].score == pytest.approx(1 / 61)  # rrf mode: score == fusion score
