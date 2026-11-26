"""Split the `papers` collection into papers / arxiv / reviews without downtime.

Steps, each a subcommand, in order (journal 2026-10-27; plan.md P0):

    create   create `arxiv` and `reviews` with the `papers` config (vectors, sparse,
             HNSW, quantization, on-disk placement) and its payload indexes. HNSW
             building can be deferred until the copy is done (--defer-indexing).
             --quantized-on-disk keeps the copies' quantized vectors off the RAM
             budget while `papers` still holds its own RAM copy of them (.88: the
             Qdrant unit already sits near MemoryHigh).
    copy     copy preprints -> arxiv and reviews -> reviews, vectors reused (no
             re-embedding). Resumable from a checkpoint file; every batch is
             upserted with wait=True and bounded exponential backoff, after the
             pre-write guard; pauses while MemAvailable is low; rate-limited.
             Search keeps reading `papers` (mixed layout) the whole time.
    index    re-enable HNSW building on the copies (after --defer-indexing).
    verify   each copy holds at least the exact count of its source filter, and
             source points sampled from random id ranges have an identical copy
             (payload and vectors). Exit status 1 on any mismatch.
    layout   switch the runtime layout file the backend reads per request
             (no restart): `split` (keeps excluding the legacy copies in papers),
             or `split --clean` once they are deleted, or back to `mixed`.
    delete-legacy
             IRREVERSIBLE: delete the preprint/review copies from `papers`, in
             rate-limited id batches. Refuses unless the layout is split, verify
             passes, and --confirm-irreversible is given; run it only after the
             operator re-confirms.

Pause arXiv and review ingest from `copy` until `layout split`: while the layout
is mixed new preprints/reviews land only in `papers` and a point written behind the
copy's scroll position would be missed (verify would then report it).

Run low-priority on the host, e.g. `nice -n 19 ionice -c3 python scripts/...`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from app.models.schemas import PREPRINT_TIER  # noqa: E402
from core.storage.qdrant_operations import retry_qdrant_operation  # noqa: E402
from core.storage.corpus_layout import (  # noqa: E402
    REVIEW,
    CorpusLayout,
    check_paper_point,
    current_layout,
    layout_path,
    paper_collections,
    write_layout,
)
from core.storage.qdrant_operations import retry_qdrant_operation  # noqa: E402
from pipeline.paper_crawler.runtime_files import write_text_atomic  # noqa: E402

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _match(key: str, value: Any) -> FieldCondition:
    return FieldCondition(key=key, match=MatchValue(value=value))


def source_filter(role: str) -> Filter:
    """The `papers` points that belong to `role` (same rule as role_for_payload)."""
    if role == "reviews":
        return Filter(must=[_match("source_type", REVIEW)])
    if role == "arxiv":
        return Filter(must=[_match("venue_tier", PREPRINT_TIER)],
                      must_not=[_match("source_type", REVIEW)])
    raise ValueError(role)


ROLES = ("arxiv", "reviews")


def log(*args) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, flush=True)


class Rest:
    """Raw REST for collection config: the client models may lag the server's
    quantization types (TurboQuant), and a config copy must not drop fields."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def call(self, method: str, path: str, body: Any = None, timeout: float = 120) -> Any:
        req = urllib.request.Request(
            self.url + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                return json.load(resp)["result"]
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{method} {path}: HTTP {exc.code} "
                               f"{exc.read().decode(errors='replace')[:500]}") from exc


def collection_body(source: dict, *, max_segment_size_kb: int | None,
                    defer_indexing: bool, quantized_on_disk: bool = False) -> dict:
    """PUT /collections body reproducing `source` (a GET /collections result)."""
    config = source["config"]
    params = config["params"]
    optimizers = {k: v for k, v in config["optimizer_config"].items() if v is not None}
    if max_segment_size_kb:
        optimizers["max_segment_size"] = max_segment_size_kb
    if defer_indexing:
        optimizers["indexing_threshold"] = 0
    body = {
        "vectors": params["vectors"],
        "shard_number": params.get("shard_number", 1),
        "replication_factor": params.get("replication_factor", 1),
        "write_consistency_factor": params.get("write_consistency_factor", 1),
        "on_disk_payload": params.get("on_disk_payload", False),
        "hnsw_config": config["hnsw_config"],
        "optimizers_config": optimizers,
        "wal_config": {k: v for k, v in config.get("wal_config", {}).items()
                       if k in {"wal_capacity_mb", "wal_segments_ahead"}},
    }
    if params.get("sparse_vectors"):
        body["sparse_vectors"] = params["sparse_vectors"]
    quantization = config.get("quantization_config")
    if quantization:
        quantization = json.loads(json.dumps(quantization))
        if quantized_on_disk:
            # {"turbo": {"always_ram": true, ...}} and the scalar/product/binary forms
            for method in quantization.values():
                if isinstance(method, dict) and "always_ram" in method:
                    method["always_ram"] = False
        body["quantization_config"] = quantization
    return body


def payload_index_bodies(source: dict) -> list[dict]:
    out = []
    for field, schema in sorted((source.get("payload_schema") or {}).items()):
        params = {k: v for k, v in (schema.get("params") or {}).items() if v is not None}
        out.append({"field_name": field,
                    "field_schema": params or schema["data_type"]})
    return out


def create(rest: Rest, names: dict, *, max_segment_size_kb: int | None,
           defer_indexing: bool, quantized_on_disk: bool = False) -> None:
    source = rest.call("GET", f"/collections/{names['papers']}")
    body = collection_body(source, max_segment_size_kb=max_segment_size_kb,
                           defer_indexing=defer_indexing,
                           quantized_on_disk=quantized_on_disk)
    for role in ROLES:
        name = names[role]
        if rest.call("GET", f"/collections/{name}/exists")["exists"]:
            log(f"{name}: exists, not recreated")
        else:
            rest.call("PUT", f"/collections/{name}", body)
            log(f"{name}: created", json.dumps(body["optimizers_config"]),
                json.dumps(body.get("quantization_config")))
        # Index before the copy so points are indexed as they arrive instead of
        # rebuilding every field over millions of points afterwards.
        for index in payload_index_bodies(source):
            rest.call("PUT", f"/collections/{name}/index?wait=true", index, timeout=3600)
        log(f"{name}: payload indexes", [i["field_name"] for i in payload_index_bodies(source)])


def enable_indexing(rest: Rest, names: dict, threshold: int) -> None:
    for role in ROLES:
        rest.call("PATCH", f"/collections/{names[role]}",
                  {"optimizers_config": {"indexing_threshold": threshold}})
        log(f"{names[role]}: indexing_threshold={threshold}")


def mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def wait_for_memory(min_gb: float, *, sleep_s: float = 30,
                    probe: Callable[[], float | None] = mem_available_gb) -> None:
    while min_gb > 0:
        available = probe()
        if available is None or available >= min_gb:
            return
        log(f"MemAvailable {available:.1f}G < {min_gb}G; pausing")
        time.sleep(sleep_s)


# Server-side read timeout for exact counts and scans; the default 60 s is too short
# for an exact filtered count over ~24M points while the optimizer rewrites segments.
READ_TIMEOUT_S = 600


# Every call below is idempotent. This script's own budget is six attempts (the
# shared default is five); only the attempt count differs from the shared policy.
RETRY_ATTEMPTS = 6


def with_retry(what: str, fn: Callable[[], Any], max_retries: int = RETRY_ATTEMPTS) -> Any:
    return retry_qdrant_operation(what, fn, max_attempts=max_retries)


def upsert_with_retry(client: QdrantClient, collection: str, points: list,
                      max_retries: int = RETRY_ATTEMPTS) -> None:
    with_retry(f"upsert {collection}",
               lambda: client.upsert(collection, points=points, wait=True), max_retries)


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(path: Path, state: dict) -> None:
    write_text_atomic(path, json.dumps(state, indent=1))


def copy(client: QdrantClient, names: dict, state_path: Path, *, batch: int,
         pause_s: float, min_mem_gb: float, roles=ROLES,
         max_batches: int = 0) -> dict:
    """Copy each role's points from papers; resumes from `state_path`. Returns the
    state. `max_batches` (tests, smoke runs) stops after that many batches."""
    state = _load_state(state_path)
    batches = 0
    for role in roles:
        entry = state.setdefault(role, {"offset": None, "copied": 0, "done": False})
        target = names[role]
        while not entry["done"]:
            if max_batches and batches >= max_batches:
                return state
            wait_for_memory(min_mem_gb)
            records, next_offset = with_retry("scroll papers", lambda: client.scroll(
                names["papers"], scroll_filter=source_filter(role), limit=batch,
                offset=entry["offset"], with_payload=True, with_vectors=True,
                timeout=READ_TIMEOUT_S))
            points = [PointStruct(id=r.id, vector=r.vector, payload=r.payload)
                      for r in records]
            for point in points:
                check_paper_point(target, point.payload or {})
            if points:
                upsert_with_retry(client, target, points)
            entry["copied"] += len(points)
            entry["offset"] = next_offset
            entry["done"] = next_offset is None
            _save_state(state_path, state)
            batches += 1
            if batches % 50 == 0 or entry["done"]:
                log(f"{target}: copied {entry['copied']:,}"
                    + (" (done)" if entry["done"] else ""))
            if pause_s and not entry["done"]:
                time.sleep(pause_s)
    return state


def _same_vector(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_vector(a[k], b[k]) for k in a)
    if hasattr(a, "indices") and hasattr(b, "indices"):
        return list(a.indices) == list(b.indices) and _same_vector(list(a.values),
                                                                    list(b.values))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(abs(x - y) <= 1e-6 for x, y in zip(a, b))
    return a == b


def verify(client: QdrantClient, names: dict, *, sample: int = 200,
           windows: int = 10) -> list[str]:
    """Problems found; empty means every source point checked has an identical copy.

    Checked from the source side: the copy must hold at least as many points (after
    the layout switch new preprints/reviews land only in the copies), and `sample`
    source points from `windows` random id ranges must exist in the copy with equal
    payload and vectors."""
    problems = []
    rng = random.Random(20261027)
    for role in ROLES:
        target = names[role]
        expected = with_retry("count papers", lambda: client.count(
            names["papers"], count_filter=source_filter(role), exact=True,
            timeout=READ_TIMEOUT_S)).count
        actual = with_retry(f"count {target}", lambda: client.count(
            target, exact=True, timeout=READ_TIMEOUT_S)).count
        log(f"{target}: papers holds {expected:,}, copy {actual:,}")
        if actual < expected:
            problems.append(f"{target}: {actual} points, papers holds {expected}")
        if not expected or not sample:
            continue
        offsets = [None] + [rng.randrange(2 ** 63) for _ in range(windows - 1)]
        originals = {}
        for offset in offsets:
            records, _ = with_retry("scroll papers", lambda: client.scroll(
                names["papers"], scroll_filter=source_filter(role),
                limit=max(1, sample // windows), offset=offset,
                with_payload=True, with_vectors=True, timeout=READ_TIMEOUT_S))
            originals.update({r.id: r for r in records})
        copies = {c.id: c for c in with_retry(f"retrieve {target}", lambda: client.retrieve(
            target, ids=list(originals), with_payload=True, with_vectors=True,
            timeout=READ_TIMEOUT_S))}
        log(f"{target}: sampled {len(originals)} source points")
        for pid, original in originals.items():
            c = copies.get(pid)
            if c is None:
                problems.append(f"{target}: point {pid} has no copy")
            elif original.payload != c.payload:
                problems.append(f"{target}: point {pid} payload differs")
            elif not _same_vector(original.vector, c.vector):
                problems.append(f"{target}: point {pid} vector differs")
    return problems


def delete_legacy(client: QdrantClient, names: dict, *, batch: int, pause_s: float,
                  min_mem_gb: float, roles=ROLES, max_batches: int = 0) -> int:
    """Delete the copies from papers in id batches (resumable: re-scans from the
    start, the deleted ones are gone). Returns the number deleted."""
    deleted = batches = 0
    for role in roles:
        while True:
            if max_batches and batches >= max_batches:
                return deleted
            wait_for_memory(min_mem_gb)
            records, _ = with_retry("scroll papers", lambda: client.scroll(
                names["papers"], scroll_filter=source_filter(role), limit=batch,
                with_payload=False, with_vectors=False, timeout=READ_TIMEOUT_S))
            if not records:
                break
            ids = [r.id for r in records]
            # Last check before an irreversible write: every id is in its copy.
            present = with_retry(f"retrieve {names[role]}", lambda: client.retrieve(
                names[role], ids=ids, with_payload=False, timeout=READ_TIMEOUT_S))
            if len(present) != len(ids):
                missing = sorted(set(ids) - {p.id for p in present})[:5]
                raise RuntimeError(f"{names[role]} lacks copies of {missing}; not deleting")
            with_retry("delete papers", lambda: client.delete(
                names["papers"], points_selector=PointIdsList(points=ids), wait=True,
                timeout=READ_TIMEOUT_S))
            deleted += len(ids)
            batches += 1
            if batches % 50 == 0:
                log(f"papers: deleted {deleted:,} legacy copies")
            if pause_s:
                time.sleep(pause_s)
    log(f"papers: deleted {deleted:,} legacy copies (done)")
    return deleted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"))
    ap.add_argument("--grpc-port", type=int,
                    default=int(os.environ.get("QDRANT_GRPC_PORT", "6334")))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create")
    p.add_argument("--max-segment-size-kb", type=int, default=0)
    p.add_argument("--defer-indexing", action="store_true")
    p.add_argument("--quantized-on-disk", action="store_true",
                   help="keep the copies' quantized vectors on disk (always_ram=false) "
                        "while papers still holds its legacy copies in RAM")
    for name in ("copy", "delete-legacy"):
        p = sub.add_parser(name)
        p.add_argument("--batch", type=int, default=256)
        p.add_argument("--pause-ms", type=int, default=200)
        p.add_argument("--min-mem-available-gb", type=float, default=2.0)
        p.add_argument("--max-batches", type=int, default=0)
        p.add_argument("--role", choices=ROLES, action="append")
    sub.choices["copy"].add_argument("--state", type=Path,
                                     default=ROOT / "data" / "split_papers_corpus.state.json")
    sub.choices["delete-legacy"].add_argument("--confirm-irreversible", action="store_true")
    p = sub.add_parser("index")
    p.add_argument("--indexing-threshold", type=int, default=20000)
    p = sub.add_parser("verify")
    p.add_argument("--sample", type=int, default=200)
    p = sub.add_parser("layout")
    p.add_argument("mode", choices=("mixed", "split"))
    p.add_argument("--clean", action="store_true",
                   help="papers no longer holds legacy preprint/review copies")
    args = ap.parse_args(argv)

    names = paper_collections()
    rest = Rest(args.url)
    client = QdrantClient(url=args.url, grpc_port=args.grpc_port, prefer_grpc=True,
                          timeout=600)

    if args.cmd == "create":
        create(rest, names, max_segment_size_kb=args.max_segment_size_kb or None,
               defer_indexing=args.defer_indexing,
               quantized_on_disk=args.quantized_on_disk)
    elif args.cmd == "copy":
        copy(client, names, args.state, batch=args.batch, pause_s=args.pause_ms / 1000,
             min_mem_gb=args.min_mem_available_gb, roles=tuple(args.role or ROLES),
             max_batches=args.max_batches)
    elif args.cmd == "index":
        enable_indexing(rest, names, args.indexing_threshold)
    elif args.cmd == "verify":
        problems = verify(client, names, sample=args.sample)
        for problem in problems:
            log("MISMATCH", problem)
        log("verify:", "OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    elif args.cmd == "layout":
        layout = CorpusLayout(split=args.mode == "split", papers_legacy_mixed=not args.clean)
        if args.clean and not layout.split:
            ap.error("--clean only applies to the split layout")
        path = write_layout(layout)
        log(f"layout {path}: {layout}")
    elif args.cmd == "delete-legacy":
        layout = current_layout()
        if not (layout.split and layout.papers_legacy_mixed):
            log(f"refusing: layout at {layout_path()} is {layout}; need split with legacy copies")
            return 2
        if not args.confirm_irreversible:
            log("refusing: deleting from papers is irreversible; pass --confirm-irreversible")
            return 2
        problems = verify(client, names)
        if problems:
            log("refusing: verify failed:", problems)
            return 1
        delete_legacy(client, names, batch=args.batch, pause_s=args.pause_ms / 1000,
                      min_mem_gb=args.min_mem_available_gb,
                      roles=tuple(args.role or ROLES), max_batches=args.max_batches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
