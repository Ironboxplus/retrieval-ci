"""Read-only real BGE-M3/Qdrant/reranker serial-vs-batch equality gate.

Requires cached real models and a reachable Qdrant endpoint.  It never creates,
upserts, deletes, or mutates a collection.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from app.services.match_doc.adapters import chunks_from_text
from app.services.match_doc.aggregator import aggregate_matches
from config import settings
from core.indexing.chunker import create_chunker
from core.indexing.embedders import get_embedder
from core.retrieval.pipeline import SearchPipeline, SearchRequest
from core.retrieval.reranker import get_shared_reranker
from core.storage.qdrant_store import QdrantStore


def fingerprint(response):
    return [(r.doc_id, r.chunk_id, r.score, r.content) for r in response.results]


def compare(serial, batched):
    serial_rows = fingerprint(serial)
    batch_rows = fingerprint(batched)
    serial_identity = [(doc_id, chunk_id, content) for doc_id, chunk_id, _score, content in serial_rows]
    batch_identity = [(doc_id, chunk_id, content) for doc_id, chunk_id, _score, content in batch_rows]
    serial_by_id = {(row[0], row[1]): row for row in serial_rows}
    batch_by_id = {(row[0], row[1]): row for row in batch_rows}
    shared_ids = serial_by_id.keys() & batch_by_id.keys()
    score_deltas = [
        abs(serial_by_id[key][2] - batch_by_id[key][2])
        for key in shared_ids
    ]
    first_identity_mismatch = next((
        index for index, (left, right) in enumerate(zip(serial_identity, batch_identity))
        if left != right
    ), None)
    first_mismatch = None
    if first_identity_mismatch is not None:
        index = first_identity_mismatch
        first_mismatch = {
            "index": index,
            "serial": serial_rows[index][:3],
            "batch": batch_rows[index][:3],
        }
    serial_ids = [(row[0], row[1]) for row in serial_rows]
    batch_ids = [(row[0], row[1]) for row in batch_rows]
    return {
        "serial_count": len(serial_rows),
        "batch_count": len(batch_rows),
        "candidate_multiset_equal": Counter(serial_ids) == Counter(batch_ids),
        "identity_order_content_equal": serial_identity == batch_identity,
        "scores_exact": all(delta == 0.0 for delta in score_deltas),
        "max_score_abs_delta": max(score_deltas, default=0.0),
        "first_identity_mismatch": first_mismatch,
        "serial_first_ids": serial_ids[:5],
        "batch_first_ids": batch_ids[:5],
    }


def compare_aggregates(serial_groups, batch_groups, *, max_papers: int):
    serial = aggregate_matches(
        [response.results for response in serial_groups],
        agg="coverage",
        max_papers=max_papers,
    )
    batched = aggregate_matches(
        [response.results for response in batch_groups],
        agg="coverage",
        max_papers=max_papers,
    )

    def identity(rows):
        return [
            (
                row.paper_id,
                row.doc_id,
                row.coverage,
                tuple((span.query_chunk_idx, span.paper_chunk) for span in row.matched),
            )
            for row in rows
        ]

    serial_identity = identity(serial)
    batch_identity = identity(batched)
    serial_keys = [(row.paper_id, row.doc_id) for row in serial]
    batch_keys = [(row.paper_id, row.doc_id) for row in batched]
    serial_by_key = {(row.paper_id, row.doc_id): row for row in serial}
    batch_by_key = {(row.paper_id, row.doc_id): row for row in batched}
    shared_keys = serial_by_key.keys() & batch_by_key.keys()
    score_deltas = [
        abs(serial_by_key[key].score - batch_by_key[key].score)
        for key in shared_keys
    ]
    matched_score_deltas = [
        abs(serial_span.sim - batch_spans[(serial_span.query_chunk_idx, serial_span.paper_chunk)])
        for key in shared_keys
        for serial_span in serial_by_key[key].matched
        for batch_spans in [{
            (span.query_chunk_idx, span.paper_chunk): span.sim
            for span in batch_by_key[key].matched
        }]
        if (serial_span.query_chunk_idx, serial_span.paper_chunk) in batch_spans
    ]
    serial_evidence_by_key = {
        key: identity([row])[0][2:]
        for key, row in serial_by_key.items()
    }
    batch_evidence_by_key = {
        key: identity([row])[0][2:]
        for key, row in batch_by_key.items()
    }
    first_identity_mismatch = next((
        index for index, (left, right) in enumerate(zip(serial_identity, batch_identity))
        if left != right
    ), None)
    first_mismatch = None
    if first_identity_mismatch is not None:
        index = first_identity_mismatch
        first_mismatch = {
            "index": index,
            "serial": {
                "paper_id": serial[index].paper_id,
                "score": serial[index].score,
                "coverage": serial[index].coverage,
            },
            "batch": {
                "paper_id": batched[index].paper_id,
                "score": batched[index].score,
                "coverage": batched[index].coverage,
            },
        }
    return {
        "serial_count": len(serial),
        "batch_count": len(batched),
        "paper_multiset_equal": Counter(serial_keys) == Counter(batch_keys),
        "paper_order_equal": serial_keys == batch_keys,
        "evidence_by_paper_equal": serial_evidence_by_key == batch_evidence_by_key,
        "paper_order_evidence_equal": serial_identity == batch_identity,
        "scores_exact": all(delta == 0.0 for delta in score_deltas),
        "max_paper_score_abs_delta": max(score_deltas, default=0.0),
        "max_matched_score_abs_delta": max(matched_score_deltas, default=0.0),
        "first_identity_mismatch": first_mismatch,
        "serial_paper_ids": [row.paper_id for row in serial],
        "batch_paper_ids": [row.paper_id for row in batched],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query-file", help="UTF-8 text; one query per line")
    source.add_argument("--markdown-file", help="UTF-8 paper Markdown, cut by the production chunker")
    parser.add_argument("--chunks", type=int, choices=(1, 3, 10), required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--microbatch-size", type=int, choices=range(1, 129), default=32)
    parser.add_argument("--score-tolerance", type=float, default=0.0)
    parser.add_argument("--output", help="Optional path for the JSON report")
    args = parser.parse_args()
    if args.score_tolerance < 0:
        raise SystemExit("--score-tolerance must be non-negative")
    settings.reranker.match_doc_microbatch_size = args.microbatch_size
    if args.markdown_file:
        markdown = Path(args.markdown_file).read_text(encoding="utf-8")
        queries = [
            chunk.text for chunk in chunks_from_text(
                markdown,
                chunker=create_chunker(),
                max_chunks=args.chunks,
                content_type="markdown",
            )
        ]
    else:
        queries = [
            line.strip()
            for line in Path(args.query_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:args.chunks]
    if len(queries) != args.chunks:
        raise SystemExit("query file has fewer non-empty lines than --chunks")
    pipe = SearchPipeline(embedder=get_embedder(), store=QdrantStore(),
                          reranker=get_shared_reranker())
    requests = [SearchRequest(query=q, top_k=args.top_k, mode="hybrid", dedup_by_paper=False) for q in queries]
    started = time.perf_counter(); serial = [pipe.search(r) for r in requests]
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter(); batched = pipe.search_many(requests)
    batch_seconds = time.perf_counter() - started
    comparisons = [compare(left, right) for left, right in zip(serial, batched)]
    aggregate_comparison = compare_aggregates(
        serial,
        batched,
        max_papers=args.max_papers,
    )
    max_chunk_score_delta = max(
        (item["max_score_abs_delta"] for item in comparisons),
        default=0.0,
    )
    max_score_delta = max(
        max_chunk_score_delta,
        aggregate_comparison["max_paper_score_abs_delta"],
        aggregate_comparison["max_matched_score_abs_delta"],
    )
    report = {
        "queries": len(queries),
        "serial_seconds": serial_seconds,
        "batch_seconds": batch_seconds,
        "microbatch_size": args.microbatch_size,
        "score_tolerance": args.score_tolerance,
        "max_score_abs_delta": max_score_delta,
        "comparisons": comparisons,
        "aggregate_comparison": aggregate_comparison,
    }
    rendered = json.dumps(report, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if not (
        all(item["candidate_multiset_equal"] for item in comparisons)
        and aggregate_comparison["paper_order_evidence_equal"]
        and max_score_delta <= args.score_tolerance
    ):
        raise SystemExit("exact equality failure; see comparison report")


if __name__ == "__main__":
    main()
