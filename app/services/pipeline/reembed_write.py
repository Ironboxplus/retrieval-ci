"""Stages ``write``, ``stamp`` and ``resplit``: the new vectors into Qdrant.

Every Qdrant operation is idempotent and waits for the write (``wait=True``) inside the shared
retry helper, so a stage that stops anywhere is simply run again. A point a client re-indexed
after the select pass already carries the current fingerprint: every vector update and every
stamp is filtered on ``embed_fp != current``, so such a point is left exactly as the client wrote it.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from qdrant_client import models

from app.services.pipeline.reembed_context import WRITE_BATCH, ReembedContext
from app.services.pipeline.reembed_select import iter_oversized
from app.services.pipeline.spec import GateResult, Progress
from core.code_kb.vector_records import (
    CodeChunkVectorRecord, build_code_chunk_payload, code_chunk_point_id, fit_records_to_budget,
)
from core.storage.qdrant_operations import retry_qdrant_operation
from core.storage.qdrant_store import sparse_dict_to_qdrant

STAMP_BATCH = 1000
DELETE_BATCH = 1000
# Payload keys build_code_chunk_payload owns; the rest of a code point's payload is its scope.
_CODE_OWNED = ("path", "language", "start_line", "end_line", "label", "content", "code", "embed_fp")


def _not_current(ctx: ReembedContext) -> list[models.FieldCondition]:
    return [ctx.must_not_current()]


def _shard_range(ctx: ReembedContext) -> range:
    return range(int(ctx.state_get("select_shard_start", 0) or 0), int(ctx.state_get("select_shard_end", 0) or 0))


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split() if path.exists() else []


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _point_vector(dense, indices, values) -> dict[str, Any]:
    return {"dense": [float(x) for x in dense],
            "sparse": models.SparseVector(indices=[int(i) for i in indices], values=[float(v) for v in values])}


# ----------------------------------------------------------------------------- write
def run_write(ctx: ReembedContext, sink) -> None:
    shards = _shard_range(ctx)
    done_file = ctx.select_dir / "write_done.txt"
    done = set(_read_lines(done_file))
    totals = {"updated": 0, "skipped": 0, "missing": 0, "upserted": 0}
    journal = ctx.scope.kind == "journal"
    rebuild = ctx.scope.kind in ("journal", "paper_rebuild")      # new points are upserted whole, not patched in
    for index in shards:
        name = f"vectors_{index:04d}.parquet"
        if name in done:
            sink(f"write: {name} already written")
            continue
        path = ctx.vectors_dir / name
        if not path.exists():
            raise RuntimeError(f"{name} is missing: the embed stage did not finish shard {index}")
        stats = _write_parquet(ctx, path, rebuild=rebuild)
        for key, value in stats.items():
            totals[key] += value
        _append_line(done_file, name)
        sink(f"write {name}: " + " ".join(f"{k}={v}" for k, v in stats.items()))
    if journal and not ctx.state_get("journal_finalized"):
        _finalize_journal(ctx, sink)
    if ctx.scope.kind == "paper_rebuild" and not ctx.state_get("paper_finalized"):
        from app.services.pipeline.reembed_papers import finalize_paper_rebuild
        finalize_paper_rebuild(ctx, sink)
    ctx.state_set(write_counts=totals)
    sink(f"write done: {totals}")


def _payload_reader(ctx: ReembedContext):
    """``payload_json`` -> payload of a rebuilt point: the journal's is stored as built, a paper's goes
    through the paper ingest's own projection and its pre-write guard."""
    if ctx.scope.kind == "paper_rebuild":
        from app.services.pipeline.reembed_papers import paper_payload_reader
        return paper_payload_reader(ctx)
    return json.loads


def _write_parquet(ctx: ReembedContext, path: Path, *, rebuild: bool) -> dict[str, int]:
    columns = ["point_id", "dense", "sparse_indices", "sparse_values"] + (["payload_json"] if rebuild else [])
    stats = {"updated": 0, "skipped": 0, "missing": 0, "upserted": 0}
    read_payload = _payload_reader(ctx) if rebuild else None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=WRITE_BATCH, columns=columns):
        ctx.abort_if_cancelled()
        cols = batch.to_pydict()
        ids = [int(i) for i in cols["point_id"]]
        if rebuild:
            points = [models.PointStruct(
                id=ids[i], vector=_point_vector(cols["dense"][i], cols["sparse_indices"][i], cols["sparse_values"][i]),
                payload=read_payload(cols["payload_json"][i])) for i in range(len(ids))]
            retry_qdrant_operation(f"upsert(collection={ctx.collection} n={len(ids)})", lambda: ctx.client.upsert(
                ctx.collection, points=points, wait=True))
            stats["upserted"] += len(ids)
        else:
            update = _update_existing(ctx, cols, ids)
            for key, value in update.items():
                stats[key] += value
        time.sleep(ctx.pause_s)
    return stats


def _update_existing(ctx: ReembedContext, cols: dict, ids: list[int]) -> dict[str, int]:
    """Replace dense+sparse of the points of one batch that exist and are still stale, then stamp them.

    The existence read is part of the retried step: Qdrant refuses a whole update_vectors call
    that names a missing point (after applying part of it), so a point deleted since the
    select pass is dropped from the batch here instead."""
    client, fingerprint = ctx.client, ctx.fingerprint

    def step() -> dict[str, int]:
        present = {p.id: (p.payload or {}).get("embed_fp") for p in client.retrieve(
            ctx.collection, ids=ids, with_payload=["embed_fp"], with_vectors=False)}
        live = [i for i, point_id in enumerate(ids) if point_id in present]
        stale = [i for i in live if present[ids[i]] != fingerprint]
        if stale:
            client.update_vectors(
                ctx.collection,
                points=[models.PointVectors(id=ids[i], vector=_point_vector(
                    cols["dense"][i], cols["sparse_indices"][i], cols["sparse_values"][i])) for i in stale],
                update_filter=models.Filter(must_not=_not_current(ctx)), wait=True)
            client.set_payload(
                ctx.collection, {"embed_fp": fingerprint},
                points=models.FilterSelector(filter=models.Filter(
                    must=[models.HasIdCondition(has_id=[ids[i] for i in stale])], must_not=_not_current(ctx))),
                wait=True)
        return {"updated": len(stale), "skipped": len(live) - len(stale), "missing": len(ids) - len(live)}

    return retry_qdrant_operation(f"update_vectors(collection={ctx.collection} n={len(ids)})", step)


def _finalize_journal(ctx: ReembedContext, sink) -> None:
    """After every new point is written: the SQL chunk rows follow, then the old points go.

    The order keeps the entry searchable throughout (old and new points coexist until the last
    step) and every step is repeatable: replace_chunks and deleting an absent point are no-ops."""
    repo = ctx.repository()
    done_file = ctx.select_dir / "journal_sql_done.txt"
    done = set(_read_lines(done_file))
    for path in sorted(ctx.select_dir.glob("journal_sql_*.jsonl.gz")):
        if path.name in done:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                ctx.abort_if_cancelled()
                row = json.loads(line)
                repo.replace_chunks(row["entry_id"], row["chunks"])
        _append_line(done_file, path.name)
        sink(f"journal: SQL chunks of {path.name} replaced")
    old_ids = sorted({int(x) for x in _read_lines(ctx.select_dir / "journal_old_ids.txt")})
    for start in range(0, len(old_ids), DELETE_BATCH):
        ctx.abort_if_cancelled()
        batch = old_ids[start:start + DELETE_BATCH]
        retry_qdrant_operation(f"delete(collection={ctx.collection} old n={len(batch)})", lambda: ctx.client.delete(
            ctx.collection, points_selector=models.PointIdsList(points=batch), wait=True))
        time.sleep(ctx.pause_s)
    ctx.state_set(journal_finalized=True)
    sink(f"journal: {len(old_ids)} old points deleted")


def write_probe(ctx: ReembedContext) -> Progress:
    shards = _shard_range(ctx)
    names = {f"vectors_{i:04d}.parquet" for i in shards}
    done = len(names & set(_read_lines(ctx.select_dir / "write_done.txt")))
    return Progress(done=done, total=len(names))


def write_gate(ctx: ReembedContext) -> GateResult:
    probe = write_probe(ctx)
    if probe.done != probe.total:
        return GateResult(False, f"{probe.done}/{probe.total} vector shards written")
    if ctx.scope.kind == "journal" and not ctx.state_get("journal_finalized"):
        return GateResult(False, "journal rebuild not finalized")
    if ctx.scope.kind == "paper_rebuild" and not ctx.state_get("paper_finalized"):
        return GateResult(False, "paper rebuild not finalized")
    return GateResult(True, f"{ctx.state_get('write_counts')}")


# ----------------------------------------------------------------------------- stamp
def run_stamp(ctx: ReembedContext, sink) -> None:
    files = sorted(ctx.select_dir.glob("stamp_*.txt"))
    if not files:
        sink("stamp: nothing to stamp")
        return
    if not ctx.state_get("sample_passed"):
        raise RuntimeError("stamp refused: the sample check has not passed, so the stored vectors "
                           "are not yet shown to be what the profile computes")
    done_file = ctx.select_dir / "stamp_done.txt"
    done = set(_read_lines(done_file))
    for path in files:
        if path.name in done:
            sink(f"stamp: {path.name} already stamped")
            continue
        ids = [int(x) for x in _read_lines(path)]
        for start in range(0, len(ids), STAMP_BATCH):
            ctx.abort_if_cancelled()
            batch = ids[start:start + STAMP_BATCH]
            retry_qdrant_operation(f"set_payload(collection={ctx.collection} stamp n={len(batch)})",
                                   lambda: ctx.client.set_payload(
                                       ctx.collection, {"embed_fp": ctx.fingerprint},
                                       points=models.FilterSelector(filter=models.Filter(
                                           must=[models.HasIdCondition(has_id=batch)], must_not=_not_current(ctx))),
                                       wait=True))
            time.sleep(ctx.pause_s)
        _append_line(done_file, path.name)
        sink(f"stamp: {path.name} stamped ({len(ids)} ids)")


def stamp_probe(ctx: ReembedContext) -> Progress:
    names = {p.name for p in ctx.select_dir.glob("stamp_*.txt")}
    return Progress(done=len(names & set(_read_lines(ctx.select_dir / "stamp_done.txt"))), total=len(names))


# --------------------------------------------------------------------------- resplit
def _namespace(payload: dict) -> str:
    """The identity namespace the collection's own writer used for this point's scope."""
    if "source_id" in payload:                                    # public code: PublicCodeVectorStore.upsert
        return f"public:{payload['source_id']}:{payload['revision']}"
    return f"{payload['owner_user_id']}:{payload['project']}"     # private code: code_kb.vector_store.point_id


def run_resplit(ctx: ReembedContext, sink) -> None:
    if not ctx.scope.resplit:
        sink("resplit: not a code scope")
        return
    done = total = 0
    for old_id, row in iter_oversized(ctx):
        total += 1
        ctx.abort_if_cancelled()
        payload = row.get("payload")
        if payload is None:
            raise RuntimeError(f"oversized point {old_id} has no payload to re-split ({row.get('reason')})")
        current = retry_qdrant_operation(f"retrieve(collection={ctx.collection} {old_id})", lambda: ctx.client.retrieve(
            ctx.collection, ids=[old_id], with_payload=["content"], with_vectors=False))
        if not current or (current[0].payload or {}).get("content") != payload.get("content"):
            # Not the point select listed any more: re-split by an earlier run (piece 0 can take the
            # old id) or re-indexed by its client since; either way its pieces are already there.
            continue
        _resplit_point(ctx, old_id, payload)
        done += 1
        sink(f"resplit: point {old_id} ({row.get('tokens')} tokens) replaced by its pieces")
    ctx.state_set(resplit_done=done, resplit_total=total)
    sink(f"resplit done: {done} of {total} oversized points")


def _resplit_point(ctx: ReembedContext, old_id: int, payload: dict) -> None:
    record = CodeChunkVectorRecord(
        rel_path=payload["path"], language=payload.get("language") or None, start_line=int(payload["start_line"]),
        end_line=int(payload["end_line"]), label=payload["label"], content=payload["content"], code=payload["code"])
    pieces = fit_records_to_budget([record])
    scope_fields = {k: v for k, v in payload.items() if k not in _CODE_OWNED}
    namespace = _namespace(payload)
    new_ids = [code_chunk_point_id(namespace, piece) for piece in pieces]
    if len(set(new_ids)) != len(new_ids):
        raise RuntimeError(f"re-split of point {old_id} produced duplicate point ids")
    clash = [p.id for p in ctx.client.retrieve(ctx.collection, ids=new_ids, with_payload=False, with_vectors=False)
             if p.id != old_id]
    if clash:
        raise RuntimeError(f"re-split of point {old_id}: pieces would overwrite other points {clash}")
    dense, sparse = ctx.encode([piece.content for piece in pieces])
    points = [models.PointStruct(
        id=new_id,
        vector={"dense": [float(x) for x in dense[i]], "sparse": sparse_dict_to_qdrant(
            {str(k): v for k, v in sparse[i].items()})},
        payload=build_code_chunk_payload(piece, scope=scope_fields))
        for i, (new_id, piece) in enumerate(zip(new_ids, pieces))]
    retry_qdrant_operation(f"upsert(collection={ctx.collection} resplit {old_id})", lambda: ctx.client.upsert(
        ctx.collection, points=points, wait=True))
    if old_id not in new_ids:
        retry_qdrant_operation(f"delete(collection={ctx.collection} {old_id})", lambda: ctx.client.delete(
            ctx.collection, points_selector=models.PointIdsList(points=[old_id]), wait=True))


def resplit_probe(ctx: ReembedContext) -> Progress:
    return Progress(done=int(ctx.state_get("resplit_done", 0) or 0), total=int(ctx.state_get("resplit_total", 0) or 0))
