"""Compact-quantization papers index on the separate compact Qdrant instance.

`trial` copies a sample of `papers` dense vectors (read-only on production) into
test-private collections on the compact instance, one per quantization variant, and
measures what a production switch would cost and lose:

* HNSW build throughput (a quantization change discards the old graph, so a switch
  means rebuilding every segment);
* Recall@10 / Recall@100 of the quantized search against exact float search, with the
  rescore settings the backend could use;
* hot search latency and the size of the quantized copy that has to stay in RAM.

Point ids are hash-derived, so scrolling in id order samples the corpus uniformly.
Held-out sample vectors serve as queries (document-to-document), which the report
states rather than hides.

`copy` builds the full compact `papers` collection on the compact instance: same
points, payload, dense and sparse vectors as production, but a pinned compact
quantization and payload indexes on disk, so the instance can run without a
low-memory mode. Production is only scrolled. The copy is resumable from a
checkpoint, can run as several disjoint id ranges, and indexes as it goes so
memory stays bounded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models  # noqa: E402

from core.storage.papers_payload import PAPER_PROFILE_KEYS  # noqa: E402
from core.storage.qdrant_operations import retry_qdrant_operation  # noqa: E402
from pipeline.paper_crawler.runtime_files import write_text_atomic  # noqa: E402

SOURCE_COLLECTION = "papers"
PRODUCTION_PORTS = (16333, 16334)
# Production's own indexing threshold, restored once the copy is complete.
PRODUCTION_INDEXING_THRESHOLD_KB = 20000
_INTEGER_FIELDS = {"year", "level"}
TRIAL_PREFIX = "quantization_trial_"
# Kilobytes of vectors, not points: 1000 KB (~250 1024-d vectors) makes every trial
# segment get an HNSW graph, including smoke-sized samples.
INDEXING_THRESHOLD_KB = 1000


def _variants() -> dict[str, models.QuantizationConfig]:
    return {
        "int8": models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True)),
        "turbo_bits2": models.TurboQuantization(turbo=models.TurboQuantQuantizationConfig(
            bits=models.TurboQuantBitSize.BITS2, always_ram=True)),
        "turbo_bits1_5": models.TurboQuantization(turbo=models.TurboQuantQuantizationConfig(
            bits=models.TurboQuantBitSize.BITS1_5, always_ram=True)),
        "binary_two_bits": models.BinaryQuantization(binary=models.BinaryQuantizationConfig(
            always_ram=True, encoding=models.BinaryQuantizationEncoding.TWO_BITS,
            query_encoding=models.BinaryQuantizationQueryEncoding.SCALAR8BITS)),
    }


# Query-time settings worth comparing: production default (rescore follows the
# quantization's default), no rescore (no original-vector reads at all), and 2x
# oversampling for the low-bit variants.
SEARCH_SETTINGS = {
    "default": None,
    "no_rescore": models.QuantizationSearchParams(rescore=False),
    "rescore_oversample2": models.QuantizationSearchParams(rescore=True, oversampling=2.0),
}


# This script's own budget is six attempts (the shared default is five); only the
# attempt count differs from the shared policy.
RETRY_ATTEMPTS = 6


def scroll_dense(source: QdrantClient, count: int, batch: int):
    """Yield pages of (id, dense vector) from production, in id order, read-only."""
    seen = 0
    offset = None
    while seen < count:
        page, offset = retry_qdrant_operation("scroll papers", lambda: source.scroll(
            SOURCE_COLLECTION, limit=min(batch, count - seen), offset=offset,
            with_payload=False, with_vectors=["dense"]), max_attempts=RETRY_ATTEMPTS)
        points = [(p.id, p.vector["dense"]) for p in page if p.vector and "dense" in p.vector]
        seen += len(points)
        yield points
        if offset is None:
            return


def create_trial_collection(target: QdrantClient, name: str,
                            quantization: models.QuantizationConfig, indexing_threads: int):
    # Test-private name: drop-then-create unconditionally.
    target.delete_collection(name)
    target.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(
            size=1024, distance=models.Distance.COSINE, on_disk=True,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False,
                                              max_indexing_threads=indexing_threads))},
        quantization_config=quantization,
        # One segment of sample size is close to a production papers segment
        # (~360K points), so the measured build rate extrapolates per segment.
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=1, indexing_threshold=0, max_optimization_threads=1),
    )


def upload(target: QdrantClient, name: str, points):
    structs = [models.PointStruct(id=pid, vector={"dense": vec}) for pid, vec in points]
    retry_qdrant_operation(f"upsert {name}",
                           lambda: target.upsert(name, points=structs, wait=True),
                           max_attempts=RETRY_ATTEMPTS)


def build_index(target: QdrantClient, name: str, expected: int, poll: float = 5.0) -> float:
    started = time.monotonic()
    target.update_collection(name, optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=INDEXING_THRESHOLD_KB))
    while True:
        time.sleep(poll)
        info = retry_qdrant_operation(f"status {name}", lambda: target.get_collection(name),
                                      max_attempts=RETRY_ATTEMPTS)
        indexed = info.indexed_vectors_count or 0
        if info.status == models.CollectionStatus.GREEN and indexed >= expected:
            return time.monotonic() - started


def exact_neighbours(target: QdrantClient, name: str, queries, limit: int) -> list[list[int]]:
    exact = models.SearchParams(exact=True, quantization=models.QuantizationSearchParams(ignore=True))
    return [[p.id for p in target.query_points(name, query=vec, using="dense", limit=limit,
                                               search_params=exact).points]
            for _, vec in queries]


def evaluate(target: QdrantClient, name: str, queries, truth, limit: int) -> dict:
    results = {}
    for label, quantization in SEARCH_SETTINGS.items():
        params = models.SearchParams(quantization=quantization) if quantization else None
        r10, r100, latency = [], [], []
        for (_, vec), expected in zip(queries, truth):
            started = time.perf_counter()
            got = [p.id for p in target.query_points(name, query=vec, using="dense", limit=limit,
                                                     search_params=params).points]
            latency.append(time.perf_counter() - started)
            r10.append(len(set(got[:10]) & set(expected[:10])) / 10)
            r100.append(len(set(got) & set(expected)) / len(expected))
        latency.sort()
        results[label] = {
            "recall@10": round(statistics.fmean(r10), 4),
            "recall@100": round(statistics.fmean(r100), 4),
            "latency_p50_ms": round(latency[len(latency) // 2] * 1000, 1),
            "latency_p95_ms": round(latency[int(len(latency) * 0.95)] * 1000, 1),
        }
    return results


def quantized_bytes(storage_path: Path, name: str) -> int:
    root = storage_path / "collections" / name
    return sum(f.stat().st_size for f in root.rglob("quantized*") if f.is_file())


def embed_queries(embed_url: str, texts: list[str], batch: int = 16) -> list[list[float]]:
    """Dense query vectors from the backend's shared BGE-M3 (POST /api/embed), so the
    trial measures recall on real query text without loading a second model."""
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch):
        body = json.dumps({"texts": texts[start:start + batch]}).encode("utf-8")
        request = urllib.request.Request(embed_url, data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=300) as response:
            vectors.extend(json.load(response)["dense"])
    return vectors


def trial(args) -> dict:
    source = _client(args.source_url, args.source_grpc_port)
    target = _client(args.target_url, args.target_grpc_port)
    variants ={v: q for v, q in _variants().items() if not args.variants or v in args.variants}
    for variant, quantization in variants.items():
        create_trial_collection(target, TRIAL_PREFIX + variant, quantization,
                                args.indexing_threads)

    # Real query text, when given, replaces held-out chunk vectors as the queries.
    queries: list[tuple[int, list[float]]] = []
    query_kind = "held-out papers chunk vectors (document-to-document)"
    if args.query_file:
        texts = json.loads(Path(args.query_file).read_text(encoding="utf-8"))
        queries = list(enumerate(embed_queries(args.embed_url, texts)))
        query_kind = f"real search_papers query text ({len(texts)}) embedded by the backend"
    held_out = 0 if args.query_file else args.queries

    # The first ids are held out as queries; every later page is streamed into all
    # variant collections at once, so the sample never sits in host memory.
    started = time.monotonic()
    corpus = 0
    for page in scroll_dense(source, held_out + args.sample, args.batch):
        if len(queries) < held_out:
            take = held_out - len(queries)
            queries.extend(page[:take])
            page = page[take:]
        if page:
            for variant in variants:
                upload(target, TRIAL_PREFIX + variant, page)
            corpus += len(page)
            if corpus % (args.batch * 100) < len(page):
                print(f"[sample] {corpus}/{args.sample}", flush=True)
    report = {"sample_points": corpus, "queries": len(queries), "query_kind": query_kind,
              "copy_seconds": round(time.monotonic() - started, 1), "variants": {}}
    print(f"[sample] {corpus} corpus + {len(queries)} queries in "
          f"{report['copy_seconds']}s", flush=True)

    truth = None
    # Built one at a time so each build time is measured without the others.
    for variant in variants:
        name = TRIAL_PREFIX + variant
        build_s = build_index(target, name, corpus)
        if truth is None:
            truth = exact_neighbours(target, name, queries, args.limit)
        entry = {
            "hnsw_build_seconds": round(build_s, 1),
            "build_points_per_second": round(corpus / build_s, 1),
            "quantized_bytes": quantized_bytes(Path(args.target_storage), name),
            "search": evaluate(target, name, queries, truth, args.limit),
        }
        entry["quantized_bytes_per_point"] = round(entry["quantized_bytes"] / corpus, 1)
        report["variants"][variant] = entry
        print(json.dumps({variant: entry}, ensure_ascii=False), flush=True)
    return report


def _payload_index_schema(field: str):
    # Persisted on disk: the compact instance runs without a low-memory mode, so an
    # in-RAM payload index (~7.5 GiB for papers) would be loaded onto the heap.
    if field in _INTEGER_FIELDS:
        return models.IntegerIndexParams(type=models.IntegerIndexType.INTEGER,
                                         lookup=True, range=True, on_disk=True)
    return models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, on_disk=True)


PAYLOAD_INDEX_FIELDS = sorted(PAPER_PROFILE_KEYS - {"content"})


def create_compact_papers(target: QdrantClient, name: str,
                          quantization: models.QuantizationConfig, indexing_threads: int):
    """Create the compact collection, failing if it already exists (resume instead)."""
    if target.collection_exists(name):
        raise SystemExit(f"collection {name!r} already exists on the target; "
                         f"resume with the same --state file instead of recreating")
    target.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(
            size=1024, distance=models.Distance.COSINE, on_disk=True,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False,
                                              max_indexing_threads=indexing_threads))},
        sparse_vectors_config={"sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=True))},
        on_disk_payload=True,
        quantization_config=quantization,
        # Indexing stays ON during the copy. Unoptimized (appendable) segments keep
        # their sparse inverted index and payload indexes on the heap: with indexing
        # off, 3.78M copied points already held 7 GiB of anon memory on .88 and
        # systemd-oomd killed the instance (2026-10-14). Optimized segments are on
        # disk + pinned quantization, so memory stays bounded while the copy runs.
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=PRODUCTION_INDEXING_THRESHOLD_KB, max_optimization_threads=1),
    )
    for field in PAYLOAD_INDEX_FIELDS:
        retry_qdrant_operation(f"payload index {field}", lambda: target.create_payload_index(
            name, field_name=field, field_schema=_payload_index_schema(field), wait=True),
            max_attempts=RETRY_ATTEMPTS)


def _load_state(path: Path, start_id: int | None = None) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"offset": start_id, "copied": 0, "done": False}


def _save_state(path: Path, state: dict):
    write_text_atomic(path, json.dumps(state))


def copy_points(source: QdrantClient, source_name: str, target: QdrantClient, target_name: str,
                *, state_path: Path, batch: int, limit: int | None = None,
                start_id: int | None = None, end_id: int | None = None,
                disk_path: Path | None = None, min_free_bytes: int = 0) -> dict:
    """Copy every point (payload + all named vectors) with start_id <= id < end_id in
    id order, checkpointing the scroll offset after each confirmed write. Upserts
    are idempotent, so a crash between write and checkpoint only repeats one page.
    Disjoint id ranges let several copies run in parallel; the scroll is
    latency-bound (read, then a confirmed write), not CPU-bound."""
    state = _load_state(state_path, start_id)
    started = time.monotonic()
    copied_this_run = 0
    while not state["done"]:
        if limit is not None and copied_this_run >= limit:
            break
        # The copy grows to ~150 GB on a disk shared with other users' data; stop
        # (resumably) before it eats the last of the free space.
        if disk_path is not None and shutil.disk_usage(disk_path).free < min_free_bytes:
            state["stopped"] = "low disk"
            print(f"[copy] stopped: free space on {disk_path} below "
                  f"{min_free_bytes / 2**30:.0f} GiB; resume with the same --state", flush=True)
            break
        state.pop("stopped", None)
        page, next_offset = retry_qdrant_operation(f"scroll {source_name}", lambda: source.scroll(
            source_name, limit=batch, offset=state["offset"], with_payload=True,
            with_vectors=True), max_attempts=RETRY_ATTEMPTS)
        if end_id is not None and any(p.id >= end_id for p in page):
            page = [p for p in page if p.id < end_id]
            next_offset = None
        if page:
            structs = [models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                       for p in page]
            retry_qdrant_operation(f"upsert {target_name}",
                                   lambda: target.upsert(target_name, points=structs, wait=True),
                                   max_attempts=RETRY_ATTEMPTS)
        state["copied"] += len(page)
        copied_this_run += len(page)
        state["offset"] = next_offset
        state["done"] = next_offset is None
        _save_state(state_path, state)
        if copied_this_run and copied_this_run % (batch * 200) < len(page):
            rate = copied_this_run / (time.monotonic() - started)
            print(f"[copy] {state['copied']} points, {rate:.0f} pts/s this run", flush=True)
    return state


def _client(url: str, grpc_port: int | None) -> QdrantClient:
    if grpc_port:
        return QdrantClient(url=url, grpc_port=grpc_port, prefer_grpc=True, timeout=300)
    return QdrantClient(url=url, timeout=300)


def copy(args) -> dict:
    source = _client(args.source_url, args.source_grpc_port)
    target = _client(args.target_url, args.target_grpc_port)
    state_path = Path(args.state)
    if not target.collection_exists(args.collection):
        create_compact_papers(target, args.collection, _variants()[args.quantization],
                              args.indexing_threads)
    state = copy_points(source, SOURCE_COLLECTION, target, args.collection,
                        state_path=state_path, batch=args.batch, limit=args.limit,
                        start_id=args.start_id, end_id=args.end_id,
                        disk_path=Path(args.target_storage),
                        min_free_bytes=int(args.min_free_gib * 2**30))
    if state["done"]:
        source_count = source.count(SOURCE_COLLECTION, exact=True).count
        target_count = target.count(args.collection, exact=True).count
        state.update(source_count=source_count, target_count=target_count)
        print(f"[copy] complete: source={source_count} target={target_count}", flush=True)
    return state


def _refuse_production(parser: argparse.ArgumentParser, url: str, grpc_port: int | None):
    if any(url.rstrip("/").endswith(f":{p}") for p in PRODUCTION_PORTS) or \
            grpc_port in PRODUCTION_PORTS:
        parser.error("this tool never writes to the production Qdrant instance")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("trial", help="measure quantization variants on a papers sample")
    t.add_argument("--source-url", default="http://127.0.0.1:16333")
    t.add_argument("--source-grpc-port", type=int, default=16334)
    t.add_argument("--target-url", default="http://127.0.0.1:16335")
    t.add_argument("--target-grpc-port", type=int, default=16336)
    t.add_argument("--target-storage",
                   default="/home/arc/retrival-4090-migration/retrival/data/indices/qdrant_compact")
    t.add_argument("--sample", type=int, default=300_000)
    t.add_argument("--queries", type=int, default=300)
    t.add_argument("--limit", type=int, default=100, help="matches the backend prefetch limit")
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--indexing-threads", type=int, default=4)
    t.add_argument("--variants", nargs="*", choices=sorted(_variants()))
    t.add_argument("--query-file", help="JSON list of real query strings to embed")
    t.add_argument("--embed-url", default="http://127.0.0.1:18080/api/embed")
    t.add_argument("--report", required=True, help="JSON report path")

    c = sub.add_parser("copy", help="copy production papers into the compact collection")
    c.add_argument("--source-url", default="http://127.0.0.1:16333")
    c.add_argument("--source-grpc-port", type=int, default=16334)
    c.add_argument("--target-url", default="http://127.0.0.1:16335")
    c.add_argument("--target-grpc-port", type=int, default=16336)
    c.add_argument("--collection", default=SOURCE_COLLECTION,
                   help="target collection name on the compact instance")
    c.add_argument("--quantization", required=True, choices=sorted(_variants()))
    c.add_argument("--indexing-threads", type=int, default=4)
    c.add_argument("--batch", type=int, default=256)
    c.add_argument("--limit", type=int, help="stop after this many points (resume later)")
    c.add_argument("--target-storage",
                   default="/home/arc/retrival-4090-migration/retrival/data/indices/qdrant_compact",
                   help="filesystem whose free space is guarded")
    c.add_argument("--min-free-gib", type=float, default=80.0,
                   help="stop (resumably) when free space falls below this")
    c.add_argument("--start-id", type=int, help="first point id of this copy's range")
    c.add_argument("--end-id", type=int, help="exclusive upper id bound of this copy's range")
    c.add_argument("--state", required=True, help="checkpoint JSON; reuse it to resume")

    args = parser.parse_args()
    _refuse_production(parser, args.target_url, getattr(args, "target_grpc_port", None))
    if args.command == "trial":
        report = trial(args)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
        print(f"[done] report -> {args.report}")
    else:
        print(json.dumps(copy(args)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
