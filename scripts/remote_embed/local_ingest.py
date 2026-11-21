"""Ingest vectors_*.parquet into local Qdrant.

Reuses settings for collection name + URL; uses raw QdrantClient.upsert
since vectors are pre-embedded (no need to go through QdrantStore.add_chunks).

Resilience (W14, 2026-05-17 00:50): Qdrant 1.18 occasionally hangs an in-flight
upsert past httpx's default 60s when HNSW indexing is mid-merge (W8). The
client-side ResponseHandlingException kills the worker if uncaught. Each
upsert now retries with exponential backoff (1s → 2s → 4s → 8s → 16s) so
transient timeouts don't drop a shard.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from config import settings  # noqa: E402
from core.storage.corpus_layout import (  # noqa: E402
    CorpusLayout,
    check_paper_point,
    collection_for_payload,
    current_layout,
    paper_collections,
)
from core.storage.papers_payload import (  # noqa: E402
    PAPER_PROFILE_KEYS,
    project_papers_payload,
)
from core.storage.qdrant_operations import retry_qdrant_operation  # noqa: E402

# Alias kept for backward-compat callers/tests. The executable whitelist is
# now owned by core.storage.papers_payload (papers_payload_contract_plan.md
# 3.2, "papers" collection projector) — do NOT redefine the key set here.
# A local copy is exactly how api_routes.py's payload comment drifted to a
# stale "9-field" claim (including `data_source`) for months before the
# 2026-09-01 audit caught it (plan.md 1.4).
PAYLOAD_KEEP_KEYS = PAPER_PROFILE_KEYS


_VEC_SHARD_RE = re.compile(r"vectors_(\d+)")


def _shard_num(path) -> int:
    """Parse the shard NUMBER out of a vectors_NNNN.parquet path. -1 if none."""
    m = _VEC_SHARD_RE.search(Path(path).name)
    return int(m.group(1)) if m else -1


def select_shards(paths, shard_start: int, shard_end: int):
    """Filter parquet paths to those whose shard NUMBER is in [shard_start,
    shard_end). ``shard_end == 0`` → no upper bound.

    NUMBER-based (parses ``vectors_NNNN``), NOT list-position based. W81: the
    old ``sorted(glob)[shard_start:shard_end]`` slices by POSITION, so when
    concurrent pipelines share one vectors/ dir and a shard < shard_end isn't
    on disk yet (its producer hasn't ship-down'd it), every later position
    shifts and ingest grabs the WRONG files (another venue's vectors_0130)
    while skipping its own (vectors_0127). Filtering by number is immune to
    both gaps and to other venues' shards in the shared dir. Backward
    compatible: a contiguous-from-0 dir has number == position.
    """
    out = []
    for p in paths:
        n = _shard_num(p)
        if n < shard_start:
            continue
        if shard_end and n >= shard_end:
            continue
        out.append(p)
    return out


def canonicalize_payload(payload: dict) -> dict:
    """Drop legacy fields + project to the papers collection's one schema.

    Legacy promotion: old shards may still carry `conference`/`author_names`;
    promote `conference -> venue` so filter pushdown stays correct. `author_names`
    is dropped entirely (authors comes from SQL JOIN on doc_id, not payload).

    Whitelist filtering (including doc_id -> str normalization, D2) is
    delegated to project_papers_payload — this script only ever ingests the
    papers batch/remote-embed pipeline (paper + text-only review), never
    upload/webpage (those write live via QdrantStore.add_chunks, see
    papers_payload_contract_plan.md 1.1), so a payload missing `source_type`
    is a legacy shard dumped before that field existed, not an upload/webpage
    row — default it to "paper" rather than fail closed on old data.
    """
    conf = payload.pop("conference", None)
    if conf and not payload.get("venue"):
        payload["venue"] = conf
    payload.pop("author_names", None)
    payload.setdefault("source_type", "paper")
    return project_papers_payload(payload)


def _upsert_with_retry(
    client: QdrantClient,
    collection: str,
    points: list,
    max_retries: int = 5,
) -> None:
    """Wrap client.upsert with exponential-backoff retry (W14).

    Qdrant 1.18 occasionally hangs upserts >60s during HNSW mid-merge; the
    raw httpx ReadTimeout propagates as `ResponseHandlingException` and
    kills the worker. We retry the same batch with `wait=True` so retried
    writes are guaranteed idempotent (Qdrant upsert is point_id-keyed).
    """
    retry_qdrant_operation(
        f"upsert(collection={collection} points={len(points)})",
        lambda: client.upsert(collection, points=points, wait=True),
        max_attempts=max_retries,
    )


AUTO = "auto"


def upsert_shard(client: QdrantClient, collection: str, path: Path,
                 batch_size: int, dry_run: bool = False,
                 start_offset: int = 0,
                 max_points: int = 0,
                 progress_callback=None,
                 pause_s: float = 0.0,
                 layout: CorpusLayout | None = None) -> int:
    """Stream parquet via iter_batches → bounded RAM (~25 MB / batch).

    `start_offset`: skip first N rows (resume after kill).
    `max_points`: stop after upserting N points from this shard (0 = no cap).
    `progress_callback(rows_done)`: invoked after each upsert flush so caller
    can persist resume checkpoint.
    `pause_s`: sleep after each flush. Qdrant, swap and SQLite share one SATA
    SSD on .88; back-to-back wait=True upserts starve live search reads until
    gRPC hits DEADLINE_EXCEEDED (503). The gap lets search reads through.

    Reliability: every batch is upserted with wait=True so the worker only
    progresses after Qdrant ACKs the write. This is what D3 got wrong
    (wait=False → 56% silently dropped under write pressure).

    `collection="auto"` sends each point to the paper collection its payload
    belongs to under `layout` (default: the runtime corpus layout) — all of them to
    `papers` under the mixed layout. Every point passes the pre-write guard, so an
    explicit target refuses a batch holding a point that belongs elsewhere before
    that batch is written.
    """
    layout = layout or current_layout()
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    pending: dict[str, list[PointStruct]] = {}
    buffered = 0
    sent = 0
    seen = 0
    tag = "[dry]" if dry_run else "    "

    def flush() -> None:
        nonlocal buffered, sent
        for target, batch in pending.items():
            if batch and not dry_run:
                _upsert_with_retry(client, target, batch)
        sent += buffered
        pending.clear()
        buffered = 0

    cols_needed = ["point_id", "dense", "sparse_indices", "sparse_values", "payload_json"]
    for rb in pf.iter_batches(batch_size=batch_size, columns=cols_needed):
        cols = rb.to_pydict()
        n_rb = len(cols["point_id"])
        for i in range(n_rb):
            seen += 1
            if seen <= start_offset:
                continue
            payload = canonicalize_payload(json.loads(cols["payload_json"][i]))
            target = (collection_for_payload(payload, layout) if collection == AUTO
                      else collection)
            check_paper_point(target, payload, layout)
            pending.setdefault(target, []).append(PointStruct(
                id=int(cols["point_id"][i]),
                vector={
                    "dense": [float(x) for x in cols["dense"][i]],
                    "sparse": SparseVector(
                        indices=list(cols["sparse_indices"][i]),
                        values=list(cols["sparse_values"][i]),
                    ),
                },
                payload=payload,
            ))
            buffered += 1
            if buffered >= batch_size:
                flush()
                print(f"{tag} upserted {start_offset + sent:,}/{n:,}", flush=True)
                if progress_callback:
                    progress_callback(start_offset + sent)
                if pause_s and not dry_run:
                    time.sleep(pause_s)
                if max_points and sent >= max_points:
                    del cols
                    return sent
        del cols
    if buffered:
        flush()
        print(f"{tag} upserted {start_offset + sent:,}/{n:,}", flush=True)
        if progress_callback:
            progress_callback(start_offset + sent)
    return sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default="data/remote_embed/vectors")
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=AUTO,
                        help="auto (default) routes each point by the corpus layout; "
                             "an explicit name must match every point")
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--pause-ms", type=int, default=250,
                        help="Sleep between upsert batches so live search keeps disk time (0 = none)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate parquet → PointStruct construction, skip writes")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Stop after N shards (0=all). Useful with --dry-run.")
    parser.add_argument("--shard-start", type=int, default=0,
                        help="Lower index of shard slice (inclusive)")
    parser.add_argument("--shard-end", type=int, default=0,
                        help="Upper index of shard slice (exclusive). 0 = all.")
    parser.add_argument("--done-file", default=None,
                        help="Override ingest_done.txt path (per-worker file for parallel runs)")
    parser.add_argument("--max-points", type=int, default=0,
                        help="Smoke-test cap: stop after N points total (0 = no cap)")
    args = parser.parse_args()

    in_dir = Path(args.vectors)
    done_path = Path(args.done_file) if args.done_file else in_dir / "ingest_done.txt"
    progress_path = done_path.with_suffix(".progress.json")
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
    progress: dict[str, int] = (
        json.loads(progress_path.read_text()) if progress_path.exists() else {}
    )

    client = QdrantClient(url=args.url, timeout=180)
    layout = current_layout()  # one layout for the whole run
    if args.collection == AUTO:
        targets = [name for name in paper_collections().values()
                   if client.collection_exists(name)]
    else:
        targets = [args.collection]

    def _points() -> int:
        return sum(client.get_collection(name).points_count for name in targets)

    before = _points()
    layout_name = "split" if layout.split else "mixed"
    print(f"corpus layout: {layout_name}; targets={targets}", flush=True)
    mode_tag = "DRY-RUN (read-only)" if args.dry_run else "EXECUTE"
    print(f"[{mode_tag}] qdrant {args.collection} before = {before:,}", flush=True)

    shards = sorted(in_dir.glob("vectors_*.parquet"))
    if args.shard_start or args.shard_end:
        # NUMBER-based selection (W81) — never slice the sorted list by POSITION.
        shards = select_shards(shards, args.shard_start, args.shard_end)
    print(f"shards = {len(shards)} (NUMBER-slice [{args.shard_start}:{args.shard_end or 'end'}])", flush=True)
    total = 0
    processed = 0
    for path in shards:
        if not args.dry_run and path.name in done:
            print(f"SKIP {path.name}", flush=True)
            continue
        start_offset = progress.get(path.name, 0) if not args.dry_run else 0
        if start_offset:
            print(f"RESUME {path.name} from offset {start_offset:,}", flush=True)
        else:
            print(f"INGEST {path.name}", flush=True)

        def _save_progress(rows_done: int, _name=path.name):
            progress[_name] = rows_done
            progress_path.write_text(json.dumps(progress))

        shard_cap = max(0, args.max_points - total) if args.max_points else 0
        n = upsert_shard(
            client, args.collection, path, args.batch,
            dry_run=args.dry_run,
            start_offset=start_offset,
            max_points=shard_cap,
            progress_callback=_save_progress if not args.dry_run else None,
            pause_s=args.pause_ms / 1000,
            layout=layout,
        )
        total += n
        # Only mark shard "done" if we ingested it fully (no smoke-test cap hit)
        full_shard = (shard_cap == 0) or (n < shard_cap)
        if not args.dry_run and full_shard:
            with done_path.open("a") as f:
                f.write(path.name + "\n")
            progress.pop(path.name, None)
            progress_path.write_text(json.dumps(progress))
        processed += 1
        if args.max_points and total >= args.max_points:
            print(f"[stop] hit --max-points {args.max_points}", flush=True)
            break
        if args.max_shards and processed >= args.max_shards:
            print(f"[stop] hit --max-shards {args.max_shards}", flush=True)
            break

    after = _points()
    print(f"[{mode_tag}] DONE processed_chunks={total:,}  after={after:,}  delta={after - before:,}", flush=True)


if __name__ == "__main__":
    main()
