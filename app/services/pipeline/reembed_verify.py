"""The acceptance checks: a sample re-encoded in-process, and a second select pass.

``sample`` runs before points are only stamped (their vectors are kept, so they must be shown to
equal what the profile computes now); ``verify`` runs last and is the task's gate: nothing is
left to do, and 200 random points of the scope still equal a fresh encode.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qdrant_client import models

from app.services.pipeline.reembed_context import ReembedContext
from app.services.pipeline.reembed_scopes import CHUNK, OVERSIZED, STAMP, current_filter
from app.services.pipeline.reembed_select import iter_stamp_ids, todo_counts
from app.services.pipeline.spec import GateResult, Progress
from core.indexing.embedding_profile import get_profile, tokens_over_limit
from core.storage.qdrant_operations import retry_qdrant_operation

SAMPLE_SIZE = 200
MIN_COSINE = 0.999
MIN_SPARSE_OVERLAP = 0.9
SPARSE_TOP = 20
ENCODE_PIECE = 16
OVERSIZED_LISTED = 50


@dataclass(frozen=True)
class SampleResult:
    checked: int
    min_cosine: float
    min_overlap: float

    @property
    def summary(self) -> str:
        return (f"sample {self.checked}: min cosine {self.min_cosine:.5f}, "
                f"min sparse top-{SPARSE_TOP} overlap {self.min_overlap:.2f}")


def _top_tokens(indices: Sequence[int], values: Sequence[float]) -> set[int]:
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")[:SPARSE_TOP]
    return {int(indices[i]) for i in order}


def check_points(ctx: ReembedContext, points: list) -> SampleResult:
    """Compare the stored dense and sparse vectors of ``points`` with a re-encode of their text.

    Raises RuntimeError naming the first violation of the cosine or sparse-overlap threshold.
    Texts over the model's limit cannot be encoded and are left out (they are listed elsewhere)."""
    field, profile = ctx.scope.text_field, get_profile()
    usable = [p for p in points if isinstance((p.payload or {}).get(field), str) and p.payload[field].strip()]
    too_long = {i for i, _ in tokens_over_limit([p.payload[field] for p in usable], profile.max_seq_len, profile)}
    usable = [p for i, p in enumerate(usable) if i not in too_long]
    if not usable:
        return SampleResult(0, 1.0, 1.0)
    min_cosine, min_overlap = 1.0, 1.0
    for start in range(0, len(usable), ENCODE_PIECE):
        piece = usable[start:start + ENCODE_PIECE]
        dense, sparse = ctx.encode([p.payload[field] for p in piece])
        for point, new_dense, new_sparse in zip(piece, dense, sparse):
            stored = np.asarray(point.vector["dense"], dtype=np.float64)
            fresh = np.asarray(new_dense, dtype=np.float64)
            cosine = float(stored @ fresh / (np.linalg.norm(stored) * np.linalg.norm(fresh)))
            stored_sparse = point.vector["sparse"]
            a = _top_tokens(stored_sparse.indices, stored_sparse.values)
            b = _top_tokens([int(k) for k in new_sparse], list(new_sparse.values())) if new_sparse else set()
            overlap = len(a & b) / max(1, min(SPARSE_TOP, len(b) or len(a)))
            min_cosine, min_overlap = min(min_cosine, cosine), min(min_overlap, overlap)
            if cosine < MIN_COSINE or overlap < MIN_SPARSE_OVERLAP:
                raise RuntimeError(
                    f"sample check failed on point {point.id} of {ctx.collection}: cosine {cosine:.5f} "
                    f"(min {MIN_COSINE}), sparse top-{SPARSE_TOP} overlap {overlap:.2f} (min {MIN_SPARSE_OVERLAP})")
    return SampleResult(len(usable), min_cosine, min_overlap)


def _retrieve_with_vectors(ctx: ReembedContext, ids: list[int]) -> list:
    return retry_qdrant_operation(f"retrieve(collection={ctx.collection} sample)", lambda: ctx.client.retrieve(
        ctx.collection, ids=ids, with_payload=[ctx.scope.text_field], with_vectors=["dense", "sparse"]))


# ---------------------------------------------------------------------- sample stage
def run_sample(ctx: ReembedContext, sink) -> None:
    """Before stamping: stored vectors of the stamp candidates equal a fresh encode."""
    if ctx.state_get("sample_passed"):
        sink(f"sample: already passed ({ctx.state_get('sample_summary')})")
        return
    rng, reservoir = random.Random(ctx.task_id), []
    for seen, point_id in enumerate(iter_stamp_ids(ctx)):
        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(point_id)
        elif (slot := rng.randrange(seen + 1)) < SAMPLE_SIZE:
            reservoir[slot] = point_id
    if not reservoir:
        ctx.state_set(sample_passed=True, sample_summary="no stamp candidates")
        sink("sample: no stamp candidates")
        return
    result = check_points(ctx, _retrieve_with_vectors(ctx, reservoir))
    ctx.state_set(sample_passed=True, sample_summary=result.summary)
    sink(f"sample passed: {result.summary}")


def sample_probe(ctx: ReembedContext) -> Progress:
    return Progress(done=1 if ctx.state_get("sample_passed") else 0, total=1)


def sample_gate(ctx: ReembedContext) -> GateResult:
    return GateResult(bool(ctx.state_get("sample_passed")), str(ctx.state_get("sample_summary") or "sample check did not pass"))


# --------------------------------------------------------------------- verify stage
def verify_gate(ctx: ReembedContext) -> GateResult:
    scope = ctx.scope
    if scope.kind == "paper_rebuild":
        from app.services.pipeline.reembed_papers import verify_paper_rebuild
        return verify_paper_rebuild(ctx)
    if scope.kind == "journal":
        left, detail = _journal_left(ctx)
    else:
        counts, oversized_ids = todo_counts(ctx)
        # Oversized papers chunks are listed for a decision (their text lives in SQL); an oversized
        # code point was re-split, so one that is still there is unfinished work.
        left = counts[STAMP] + counts[CHUNK] + (counts[OVERSIZED] if scope.resplit else 0)
        listed = oversized_ids[:OVERSIZED_LISTED]
        detail = (f"stamp={counts[STAMP]} chunk={counts[CHUNK]} oversized={counts[OVERSIZED]}"
                  f"{' ids ' + str(listed) + (' ...' if len(oversized_ids) > len(listed) else '') if oversized_ids else ''}")
    if left:
        return GateResult(False, f"{left} point(s) of {ctx.scope_name} still to re-embed ({detail})")
    try:
        sample = _final_sample(ctx)
    except RuntimeError as exc:
        return GateResult(False, str(exc))
    return GateResult(True, f"0 to do ({detail}); {sample.summary}")


def _final_sample(ctx: ReembedContext) -> SampleResult:
    flt = current_filter(ctx.scope, ctx.fingerprint)
    result = retry_qdrant_operation(f"sample(collection={ctx.collection})", lambda: ctx.client.query_points(
        ctx.collection, query=models.SampleQuery(sample=models.Sample.RANDOM), query_filter=flt, limit=SAMPLE_SIZE,
        with_payload=[ctx.scope.text_field], with_vectors=["dense", "sparse"]))
    return check_points(ctx, list(result.points))


def _journal_left(ctx: ReembedContext) -> tuple[int, str]:
    from sqlalchemy import func

    from core.db.models import JournalChunk, JournalEntry

    stale = retry_qdrant_operation(f"count(collection={ctx.collection} stale)", lambda: ctx.client.count(
        ctx.collection, count_filter=models.Filter(must_not=[ctx.must_not_current()]), exact=True).count)
    limit = get_profile().chunk_max_tokens
    with ctx.repository().session() as session:
        longest = session.query(func.max(JournalChunk.token_count)).join(
            JournalEntry, JournalEntry.entry_id == JournalChunk.entry_id).filter(
            JournalEntry.deleted_at.is_(None), JournalEntry.indexed_at.is_not(None)).scalar() or 0
    counts = ctx.state_get("select_counts") or {}
    detail = (f"stale points={stale} longest SQL chunk={longest}/{limit} tokens "
              f"pending_skipped={counts.get('pending_skipped', 0)}")
    return int(stale) + (1 if longest > limit else 0), detail
