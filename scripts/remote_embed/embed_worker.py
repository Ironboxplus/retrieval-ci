"""BGE-M3 embed worker: a client of the host's BGE-M3 vLLM service.

Standalone — does NOT import the project's `config` layer (which pulls in
pydantic_settings etc.). The model runs in its own process, the vLLM service the host keeps
resident (`scripts/remote_embed/bge_m3_service.sh`, settings in
`deploy/compute88/vllm_bge_m3.yaml`); this worker loads no model and needs no GPU. It
tokenizes with the profile's tokenizer and sends token ids through
`core/indexing/embedders/vllm_client.py`, which keeps the service busy while the next
request is tokenized.

Sparse output format is `vllm_client.encode`'s (token_id_int -> float weight), the same as
`bge_m3_core.encode`'s; local_ingest.py uses it when building Qdrant SparseVector. Every
payload carries `embed_fp`, the profile fingerprint the vectors were computed under.

The shard loop (`embed_shards`) takes the model as an `encode(texts)` callable, so
this CLI and the backend's local embed stage (which encodes with its already
resident shared embedder instead of a second copy) run the same code.
Importing this module loads neither a model library nor the tokenizer and leaves the
environment alone; only `main()` does.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# encode(texts) -> (dense vectors [n, 1024], sparse weights [n] of token id -> weight).
# Token ids may be int (the service client) or str (the backend's shared embedder).
Encode = Callable[[list], tuple]
DIMENSION = 1024


def _log(message: str) -> None:
    print(message, flush=True)


def iter_lines(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _record_batches(path: Path, size: int):
    batch = []
    for record in iter_lines(path):
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def parquet_schema():
    return pa.schema([
        ("point_id", pa.int64()),
        ("chunk_id_str", pa.string()),
        ("doc_id", pa.int64()),
        ("payload_json", pa.string()),
        ("dense", pa.list_(pa.float16(), DIMENSION)),
        ("sparse_indices", pa.list_(pa.int32())),
        ("sparse_values", pa.list_(pa.float32())),
    ])


def _profile():
    """The embedding profile. The repo root goes on ``sys.path`` first: only the CLI runs
    outside it, and the profile modules need nothing from the project's ``config``."""
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from core.indexing.embedding_profile import get_profile
    return get_profile()


def build_payload(rec: dict) -> dict:
    """Payload of a paper-pipeline record (the SQL dump's fields)."""
    payload = {
        "chunk_id": rec["chunk_id_str"],
        "doc_id": rec["doc_id"],
        "content": rec["content"],
        "level": rec["level"],
        "start_idx": rec["start_idx"],
        "end_idx": rec["end_idx"],
    }
    payload.update(rec.get("metadata") or {})
    payload["embed_fp"] = _profile().fingerprint     # after the metadata: it cannot override it
    return payload


def record_row(rec: dict) -> tuple[Optional[str], Optional[int], str]:
    """(chunk_id_str, doc_id, payload_json) of one record.

    A paper-pipeline record (it has ``chunk_id_str``) gets the paper payload. A generic record
    ({point_id, content} as the re-embed pipeline ships them) has no paper columns: its payload
    is its own ``payload_json`` when it carries one, else ``"{}"`` (the points already exist and
    only their vectors are replaced).
    """
    if "chunk_id_str" in rec:
        return rec["chunk_id_str"], int(rec["doc_id"]), json.dumps(build_payload(rec), ensure_ascii=False)
    return None, None, rec.get("payload_json") or "{}"


def encode_group(encode: Encode, texts: list, encode_batch_size: Optional[int],
                 should_stop: Optional[Callable[[], bool]]):
    """(dense float16 [n, 1024], sparse rows) of `texts` in input order, or None when
    `should_stop` fired.

    The texts reach `encode` ordered by character length, ascending, in calls of at
    most `encode_batch_size` (the whole group when None), and the results are put
    back in input order. The order puts texts of similar length into the same
    request: the vLLM service packs whole sequences into engine steps by token
    budget, and a step that ends on one long sequence leaves the rest of its budget
    unused. A caller that cuts a group into small pieces (the backend's shared
    embedder, so a waiting query can run between them) gets pieces of alike length.
    Characters stand in for tokens: close enough to sort by, and free.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    step = encode_batch_size or max(len(texts), 1)
    dense_rows = np.empty((len(texts), DIMENSION), dtype=np.float16)
    sparse_rows: list = [None] * len(texts)
    for start in range(0, len(order), step):
        if should_stop is not None and should_stop():
            return None
        picked = order[start:start + step]
        dense, sparse = encode([texts[i] for i in picked])
        dense_rows[picked] = np.asarray(dense, dtype=np.float16)
        for position, i in enumerate(picked):
            sparse_rows[i] = sparse[position] or {}
    return dense_rows, sparse_rows


def _table(point_ids, chunk_ids, doc_ids, payloads, dense_parts, sparse_rows) -> pa.Table:
    """The shard's parquet table, built from arrays (a 25,000-row shard as Python lists of
    floats took 4.7 s of the worker's time with the GPU idle)."""
    dense = np.concatenate(dense_parts) if dense_parts else np.empty((0, DIMENSION), np.float16)
    lengths = np.fromiter((len(s) for s in sparse_rows), dtype=np.int32, count=len(sparse_rows))
    offsets = pa.array(np.concatenate(([0], np.cumsum(lengths, dtype=np.int64))).astype(np.int32))
    indices = np.fromiter((int(k) for s in sparse_rows for k in s.keys()), dtype=np.int32, count=int(lengths.sum()))
    values = np.fromiter((float(v) for s in sparse_rows for v in s.values()), dtype=np.float32,
                         count=int(lengths.sum()))
    schema = parquet_schema()
    return pa.Table.from_arrays([
        pa.array(point_ids, pa.int64()),
        pa.array(chunk_ids, pa.string()),
        pa.array(doc_ids, pa.int64()),
        pa.array(payloads, pa.string()),
        pa.FixedSizeListArray.from_arrays(pa.array(dense.reshape(-1), pa.float16()), DIMENSION),
        pa.ListArray.from_arrays(offsets, pa.array(indices, pa.int32())),
        pa.ListArray.from_arrays(offsets, pa.array(values, pa.float32())),
    ], schema=schema)


def process_shard(encode: Encode, in_path: Path, out_path: Path, group_size: int = 2048, *,
                  encode_batch_size: Optional[int] = None,
                  should_stop: Optional[Callable[[], bool]] = None,
                  log: Callable[[str], None] = _log) -> bool:
    """Embed one chunks shard into one parquet. False (nothing written) when stopped:
    a shard is either fully written or not at all. ``group_size`` <= 0 hands the whole
    shard to one encode call."""
    # Validate the complete shard against the single embedding profile before
    # sending any request. A late InputTooLong otherwise leaves a partial in-memory
    # result and makes a retry look like a valid completed prefix. Oversized paper
    # chunks must go through the canonical SQL reconcile/rebuild pipeline first.
    from core.indexing.embedding_profile import get_profile, tokens_over_limit
    profile = get_profile()
    for rec_batch in _record_batches(in_path, 2048):
        over = tokens_over_limit([rec["content"] for rec in rec_batch], profile.max_seq_len, profile)
        if over:
            index, tokens = over[0]
            raise ValueError(
                f"shard {in_path.name} contains input {index} with {tokens} tokens over "
                f"the model limit {profile.max_seq_len}; run canonical paper chunk reconciliation first"
            )

    point_ids, chunk_ids, doc_ids, payloads = [], [], [], []
    dense_parts: list = []
    sparse_rows: list = []

    group: list[dict] = []
    total = 0
    started = time.monotonic()

    def flush() -> bool:
        nonlocal group, total
        if not group:
            return True
        encoded = encode_group(encode, [rec["content"] for rec in group], encode_batch_size, should_stop)
        if encoded is None:
            return False
        dense, sparses = encoded
        for j, rec in enumerate(group):
            chunk_id, doc_id, payload_json = record_row(rec)
            point_ids.append(int(rec["point_id"]))
            chunk_ids.append(chunk_id)
            doc_ids.append(doc_id)
            payloads.append(payload_json)
        dense_parts.append(dense)
        sparse_rows.extend(sparses)
        total += len(group)
        log(f"    [{in_path.name}] {total:,} done ({total / max(time.monotonic() - started, 1e-9):.1f}/s)")
        group = []
        return True

    for rec in iter_lines(in_path):
        group.append(rec)
        if 0 < group_size <= len(group) and not flush():
            return False
    if not flush():
        return False

    # Written under a temporary name and renamed: a process killed while writing leaves only the
    # .part file, which no glob matches and the next write replaces. The final name therefore
    # always holds a whole parquet (2026-11-21: oomd killed the backend in the middle of this
    # write, the resume took the half-written vectors_0000.parquet for finished and the ingest
    # stage of ECCV 2026 failed on "magic bytes not found in footer").
    part = out_path.with_name(out_path.name + ".part")
    pq.write_table(_table(point_ids, chunk_ids, doc_ids, payloads, dense_parts, sparse_rows),
                   part, compression="zstd")
    os.replace(part, out_path)
    log(f"  wrote {out_path.name}  rows={total:,}")
    return True


def complete_parquet(path: Path) -> bool:
    """Whether ``path`` is a parquet file written to the end: its footer, which a writer puts
    down last, reads back. A file cut off by a kill (written before the writes were atomic)
    has no footer."""
    try:
        pq.read_metadata(path)
    except (OSError, pa.ArrowException):
        return False
    return True


def embed_shards(in_dir: Path, out_dir: Path, encode: Encode, *, rank: int = 0, world_size: int = 1,
                 group_size: int = 2048, encode_batch_size: Optional[int] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 log: Callable[[str], None] = _log) -> bool:
    """Embed this rank's chunks_*.jsonl.gz shards of `in_dir` into vectors_*.parquet in `out_dir`.

    True when every shard of the rank is done, False when `should_stop` ended the run early
    (checked before each encode call; the interrupted shard leaves no parquet and the next
    run does it again).
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / f"done.rank{rank}.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()

    shards = sorted(in_dir.glob("chunks_*.jsonl.gz"))
    my_shards = [s for i, s in enumerate(shards) if i % world_size == rank]
    log(f"rank={rank}/{world_size}  shards = {len(my_shards)}")

    for shard in my_shards:
        idx = shard.name.replace("chunks_", "").replace(".jsonl.gz", "")
        out_path = out_dir / f"vectors_{idx}.parquet"
        # W34 fix: skip if EITHER the per-rank done file lists it OR the parquet
        # already exists. The old `and` form broke across a changed world_size —
        # done.rank{N}.txt sharding no longer matches the new rank, so every shard
        # whose old-rank != new-rank got re-embedded. `or exists()` is world-size-
        # agnostic: any rank that sees a finished parquet skips it. (Lets the
        # CVPR run land high shard indices on top of 100+ old S.9 staging shards
        # without re-embedding them.)
        # A parquet that is not listed must also read back whole: one cut off by a kill is
        # embedded again, and the rename replaces it.
        if out_path.name in done or (out_path.exists() and complete_parquet(out_path)):
            log(f"SKIP {shard.name} (done)")
            continue
        if out_path.exists():
            log(f"REDO {shard.name}: {out_path.name} is not a complete parquet (cut off while being written)")
        log(f"PROCESS {shard.name} -> {out_path.name}")
        if not process_shard(encode, shard, out_path, group_size, encode_batch_size=encode_batch_size,
                             should_stop=should_stop, log=log):
            log(f"STOPPED {shard.name}")
            return False
        with done_path.open("a") as f:
            f.write(out_path.name + "\n")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dir of chunks_*.jsonl.gz")
    parser.add_argument("--output", required=True, help="dir for vectors_*.parquet")
    parser.add_argument("--model-dir", required=True,
                        help="local BGE-M3 snapshot: its tokenizer turns text into the ids the service gets")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="texts per service request (default: the client's tuned value)")
    parser.add_argument("--group-size", type=int, default=0,
                        help="chunks per encode() call; 0 = the whole shard, so the service never "
                             "waits for the next group to be read")
    parser.add_argument("--service-url", default=None,
                        help="BGE-M3 vLLM service (default: $EMBEDDING__SERVICE_URL or http://127.0.0.1:18200)")
    args = parser.parse_args()

    profile = _profile()
    from core.indexing.embedders import vllm_client
    from core.indexing.embedding_profile import MODEL_PATH_ENV, load_tokenizer

    os.environ[MODEL_PATH_ENV] = args.model_dir       # the tokenizer's files; the weights are the service's
    tokenizer = load_tokenizer(profile)
    served = vllm_client.check_service(profile, args.service_url)
    _log(f"service {vllm_client.service_url(args.service_url)} serves {served['root']}")
    per_request = {"rows_per_request": args.batch_size} if args.batch_size else {}

    embed_shards(Path(args.input), Path(args.output),
                 lambda texts: vllm_client.encode(texts, profile, base_url=args.service_url,
                                                  tokenizer=tokenizer, **per_request),
                 rank=args.rank, world_size=args.world_size, group_size=args.group_size)


if __name__ == "__main__":
    main()
