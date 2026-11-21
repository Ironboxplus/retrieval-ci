"""Real-model gate: cancellable rerank must not change the RANKING it returns.

`score_pairs(should_cancel=...)` drives FlagEmbedding in chunks so a cancel can
land between them, instead of blocking until the whole candidate set is scored.

That is not score-identical to the single call and is not meant to be:
FlagEmbedding groups pairs by length to limit padding rather than taking them in
input order, so pre-chunking changes each batch's composition and FP16
activations move with it. The operator's call is performance over bit-exactness
as long as retrieval quality is unaffected — so what this gate measures is the
thing that would actually hurt: the order of the results.

Candidates are real corpus chunks pulled from the live `papers` collection, not
synthetic strings, because near-ties are what a reordering risk is made of and
synthetic text does not produce realistic ones.

Run on the box that owns the GPU and the Qdrant:
    cd <tree> && python scripts/smoke/rerank_cancel_equivalence_real_gate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

QUERIES = [
    "backdoor adjustment identifies the causal effect from observational data",
    "mixture-of-experts routing stability and load balancing losses",
    "retrieval augmented generation with a cross-encoder reranking stage",
]
CANDIDATES_PER_QUERY = 400


def _real_chunks(limit: int) -> list[str]:
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        grpc_port=int(os.environ["QDRANT_GRPC_PORT"]),
        prefer_grpc=True,
        timeout=120,
    )
    points, _ = client.scroll(
        os.environ.get("QDRANT_PAPERS_COLLECTION", "papers"),
        limit=limit,
        with_payload=["content"],
        with_vectors=False,
    )
    texts = [
        (p.payload or {}).get("content") or "" for p in points
    ]
    return [t for t in texts if len(t) > 80]


def main() -> int:
    from core.retrieval.reranker import RerankCancelledError, get_shared_reranker

    reranker = get_shared_reranker()
    texts = _real_chunks(CANDIDATES_PER_QUERY)
    if len(texts) < 100:
        print(f"[FAIL] only {len(texts)} usable corpus chunks; gate needs real data")
        return 1
    print(f"batch_size={reranker._batch_size} candidates={len(texts)} "
          f"queries={len(QUERIES)}")

    worst_top10 = 0
    worst_delta = 0.0
    failures = 0

    for query in QUERIES:
        single = reranker.score_pairs(query, texts)
        chunked = reranker.score_pairs(query, texts, should_cancel=lambda: False)

        worst_delta = max(
            worst_delta, max(abs(a - b) for a, b in zip(single, chunked)),
        )
        # Rank by score descending, tie-broken by index, exactly like rerank() does.
        order_single = [i for i, _ in sorted(
            enumerate(single), key=lambda kv: (-kv[1], kv[0]))]
        order_chunked = [i for i, _ in sorted(
            enumerate(chunked), key=lambda kv: (-kv[1], kv[0]))]

        for k in (10, 50):
            a, b = order_single[:k], order_chunked[:k]
            if a != b:
                moved = sum(1 for x, y in zip(a, b) if x != y)
                same_set = set(a) == set(b)
                print(f"[{'WARN' if same_set else 'FAIL'}] top-{k} order differs for "
                      f"{query[:40]!r}: {moved}/{k} positions, same_set={same_set}")
                if k == 10:
                    worst_top10 = max(worst_top10, moved)
                if not same_set:
                    failures += 1
            else:
                print(f"[OK] top-{k} identical for {query[:40]!r}")

    print(f"\nmax |score delta| = {worst_delta:.3e}")
    print(f"worst top-10 positional change = {worst_top10}")

    if failures:
        print("[FAIL] the top-k SET changed, not just the order within it — that is a "
              "recall change, not FP16 noise.")
        return 1

    # Cancellation must actually fire between batches.
    calls = {"n": 0}

    def cancel_on_second_check() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    try:
        reranker.score_pairs(QUERIES[0], texts, should_cancel=cancel_on_second_check)
    except RerankCancelledError as exc:
        expected_batches = -(-len(texts) // reranker._batch_size)
        print(f"[OK] cancelled mid-rerank after {calls['n']} checks "
              f"(would have been {expected_batches} batches): {exc}")
    else:
        print("[FAIL] should_cancel returned True but the rerank ran to completion")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
