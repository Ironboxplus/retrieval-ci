"""Stage ``select``: which points of the scope need what, written as shards and side files.

Qdrant scrolls the scope's stale points (payload text only, no vectors); the model's real
tokenizer decides per point whether it is re-embedded (``chunks_NNNN.jsonl.gz``), only needs the
fingerprint (``select/stamp_NNNN.txt``) or cannot be embedded as it is (``select/oversized.jsonl``).
The journal scope has no scroll of text: its entries are rebuilt from SQL with the live write
path's own chunker and payload code. The paper rebuild scopes read their candidates from SQLite
and re-chunk the Markdown (``reembed_papers``).

The scan checkpoints (scroll offset, shard counters, category counts) in the task state each
time its buffers are written out, so a restart resumes where it stopped and a completed select is
never run twice.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Callable, Iterator

from qdrant_client import models

from app.services.pipeline.reembed_context import ReembedContext
from app.services.pipeline.reembed_scopes import (
    CATEGORIES, CHUNK, OVERSIZED, SKIP, STAMP, categorize_texts, stale_filter,
)
from app.services.pipeline.spec import GateResult, Progress
from app.services.pipeline.stages import _next_free_shard
from core.storage.qdrant_operations import retry_qdrant_operation

SCROLL_PAGE = 1000
# Points scanned between two checkpoints even when no shard is full. A papers scope scans millions
# of points and writes almost nothing: without this a restart would redo hours of scrolling.
CHECKPOINT_POINTS = 500_000
STAMP_FILE_IDS = 200_000
JOURNAL_ENTRY_BATCH = 100


def _atomic_write(path: Path, write: Callable[[Any], None], *, gz: bool = False) -> None:
    """Write ``path`` through ``write(file)`` so it either exists completely or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with (gzip.open(part, "wt", encoding="utf-8") if gz else part.open("w", encoding="utf-8")) as fh:
        write(fh)
    part.replace(path)


def _zero_counts() -> dict[str, int]:
    return {category: 0 for category in CATEGORIES}


class SelectOutput:
    """Buffers of one select run and their write-out; ``flush`` is also the checkpoint."""

    def __init__(self, ctx: ReembedContext, shard_size: int):
        self.ctx = ctx
        self.shard_size = shard_size
        self.next_shard = int(ctx.state_get("select_next_shard"))
        self.next_stamp = int(ctx.state_get("select_next_stamp", 0) or 0)
        self.next_sidecar = int(ctx.state_get("select_next_sidecar", 0) or 0)
        self.counts: dict[str, int] = {**_zero_counts(), **(ctx.state_get("select_counts") or {})}
        self.records: list[dict] = []
        self.stamps: list[int] = []
        self.oversized: list[dict] = []
        self.old_ids: list[int] = []
        self.sql_rows: list[dict] = []
        self.next_plan = int(ctx.state_get("select_next_plan", 0) or 0)
        self.plans: list[dict] = []          # paper_rebuild: one per rebuilt document
        self.failures: list[dict] = []       # paper_rebuild: documents that could not be rebuilt

    @property
    def full(self) -> bool:
        return len(self.records) >= self.shard_size or len(self.stamps) >= STAMP_FILE_IDS

    def flush(self, **checkpoint: Any) -> None:
        ctx = self.ctx
        for start in range(0, len(self.records), self.shard_size):
            rows = self.records[start:start + self.shard_size]
            path = ctx.chunks_dir / f"chunks_{self.next_shard:04d}.jsonl.gz"
            _atomic_write(path, lambda fh, rows=rows: fh.writelines(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows), gz=True)
            self.next_shard += 1
        if self.stamps:
            _atomic_write(ctx.select_dir / f"stamp_{self.next_stamp:04d}.txt",
                          lambda fh: fh.write("\n".join(map(str, self.stamps)) + "\n"))
            self.next_stamp += 1
        if self.sql_rows:
            _atomic_write(ctx.select_dir / f"journal_sql_{self.next_sidecar:04d}.jsonl.gz", lambda fh: fh.writelines(
                json.dumps(row, ensure_ascii=False) + "\n" for row in self.sql_rows), gz=True)
            self.next_sidecar += 1
        if self.plans:
            _atomic_write(ctx.select_dir / f"paper_plans_{self.next_plan:04d}.jsonl.gz", lambda fh: fh.writelines(
                json.dumps(row, ensure_ascii=False) + "\n" for row in self.plans), gz=True)
            self.next_plan += 1
        for name, rows in (("oversized.jsonl", self.oversized), ("journal_old_ids.txt", self.old_ids),
                           ("paper_failures.jsonl", self.failures)):
            if rows:
                ctx.select_dir.mkdir(parents=True, exist_ok=True)
                with (ctx.select_dir / name).open("a", encoding="utf-8") as fh:
                    fh.writelines((json.dumps(row, ensure_ascii=False) if isinstance(row, dict) else str(row)) + "\n"
                                  for row in rows)
        self.records, self.stamps, self.oversized, self.old_ids, self.sql_rows = [], [], [], [], []
        self.plans, self.failures = [], []
        ctx.state_set(select_next_shard=self.next_shard, select_shard_end=self.next_shard,
                      select_next_stamp=self.next_stamp, select_next_sidecar=self.next_sidecar,
                      select_next_plan=self.next_plan, select_counts=self.counts, **checkpoint)


def stale_pages(ctx: ReembedContext, *, offset: Any = None,
                page_limit: Callable[[], int] = lambda: SCROLL_PAGE) -> Iterator[tuple[list, Any]]:
    """``(points, next_offset)`` pages of the scope's stale points: payload text only, no vectors.

    ``page_limit`` is asked before every page, so a caller can size a page to the room it has left."""
    client, flt, field = ctx.client, stale_filter(ctx.scope, ctx.fingerprint), ctx.scope.text_field
    while True:
        limit = max(1, page_limit())
        points, next_offset = retry_qdrant_operation(
            f"scroll(collection={ctx.collection} stale)",
            lambda: client.scroll(ctx.collection, scroll_filter=flt, limit=limit, offset=offset,
                                  with_payload=[field], with_vectors=False))
        yield points, next_offset
        if next_offset is None:
            return
        offset = next_offset


def classify_page(ctx: ReembedContext, points: list) -> tuple[list[tuple[str, dict]], list[tuple[int, int, str]]]:
    """Category of every point of a page: ``([(category, row)], [(id, tokens, reason)] oversized)``.

    ``row`` is the chunk record or the stamp id's holder; a point without text is listed as
    oversized (it cannot be embedded), never silently skipped."""
    scope = ctx.scope
    usable, rows, big = [], [], []
    for point in points:
        if not isinstance(point.id, int):
            raise RuntimeError(f"point id {point.id!r} of {ctx.collection} is not an integer; the vector "
                               "shards carry int64 ids")
        text = (point.payload or {}).get(scope.text_field)
        if isinstance(text, str) and text.strip():
            usable.append((point.id, text))
        else:
            big.append((point.id, 0, "no text"))
    for (point_id, text), (category, tokens) in zip(usable, categorize_texts(scope, [t for _, t in usable])):
        if category == CHUNK:
            rows.append((CHUNK, {"point_id": point_id, "content": text}))
        elif category == STAMP:
            rows.append((STAMP, {"point_id": point_id}))
        elif category == OVERSIZED:
            big.append((point_id, tokens, f"{tokens} tokens"))
        else:
            rows.append((SKIP, {"point_id": point_id}))
    return rows, big


def _oversized_rows(ctx: ReembedContext, big: list[tuple[int, int, str]]) -> list[dict]:
    """Oversized points as ``{point_id, tokens, reason, payload}``; a code point carries its full
    payload (the re-split rebuilds it), a papers point is only listed (its text lives in SQL)."""
    payloads: dict[int, dict] = {}
    ids = [point_id for point_id, _, _ in big]
    if ids and ctx.scope.resplit:
        got = retry_qdrant_operation(f"retrieve(collection={ctx.collection} oversized)", lambda: ctx.client.retrieve(
            ctx.collection, ids=ids, with_payload=True, with_vectors=False))
        payloads = {point.id: dict(point.payload or {}) for point in got}
    return [{"point_id": point_id, "tokens": tokens, "reason": reason, "payload": payloads.get(point_id)}
            for point_id, tokens, reason in big]


def todo_counts(ctx: ReembedContext) -> tuple[dict[str, int], list[int]]:
    """A dry select pass: ``(points per category, oversized ids)`` of what is still stale now."""
    counts, oversized_ids = _zero_counts(), []
    for points, _ in stale_pages(ctx):
        rows, big = classify_page(ctx, points)
        for category, _row in rows:
            counts[category] += 1
        counts[OVERSIZED] += len(big)
        oversized_ids.extend(point_id for point_id, _, _ in big)
        ctx.abort_if_cancelled()
    return counts, oversized_ids


# ----------------------------------------------------------------------------- stage
def run_select(ctx: ReembedContext, sink) -> None:
    if ctx.state_get("select_complete"):
        sink(f"select: already complete {ctx.state_get('select_counts')}")
        return
    shard_size = ctx.shard_size()
    if ctx.state_get("select_next_shard") is None:
        start = _next_free_shard(ctx.chunks_dir, ctx.vectors_dir)
        ctx.state_set(select_shard_start=start, select_next_shard=start, select_shard_end=start,
                      select_scanned=0)
    out = SelectOutput(ctx, shard_size)
    sink(f"select {ctx.scope_name} on {ctx.collection}: shard size {shard_size}, "
         f"resuming at shard {out.next_shard}, counts {out.counts}")
    if ctx.scope.kind == "journal":
        _select_journal(ctx, sink, out)
    elif ctx.scope.kind == "paper_rebuild":
        from app.services.pipeline.reembed_papers import select_papers
        select_papers(ctx, sink, out)
    else:
        _select_points(ctx, sink, out)
    ctx.state_set(select_complete=True)
    sink(f"select done: {out.counts}; shards [{ctx.state_get('select_shard_start')}, "
         f"{ctx.state_get('select_shard_end')})")


def _select_points(ctx: ReembedContext, sink, out: SelectOutput) -> None:
    scanned = int(ctx.state_get("select_scanned", 0) or 0)
    offset = ctx.state_get("select_offset")
    if ctx.state_get("select_total") is None:
        total = retry_qdrant_operation(f"count(collection={ctx.collection} stale)", lambda: ctx.client.count(
            ctx.collection, count_filter=stale_filter(ctx.scope, ctx.fingerprint), exact=False).count)
        ctx.state_set(select_total=int(total))
        sink(f"select: about {total:,} stale points in scope")
    since_checkpoint = 0
    pages = stale_pages(ctx, offset=offset, page_limit=lambda: out.shard_size - len(out.records))
    for points, next_offset in pages:
        rows, big = classify_page(ctx, points)
        for category, row in rows:
            out.counts[category] += 1
            if category == CHUNK:
                out.records.append(row)
            elif category == STAMP:
                out.stamps.append(row["point_id"])
        out.counts[OVERSIZED] += len(big)
        out.oversized.extend(_oversized_rows(ctx, big))
        scanned += len(points)
        since_checkpoint += len(points)
        ctx.abort_if_cancelled()
        if out.full or next_offset is None or since_checkpoint >= CHECKPOINT_POINTS:
            out.flush(select_offset=next_offset, select_scanned=scanned)
            since_checkpoint = 0
            sink(f"select: scanned {scanned:,}, {out.counts}")


def _select_journal(ctx: ReembedContext, sink, out: SelectOutput) -> None:
    from sqlalchemy import func
    from sqlalchemy.orm import selectinload

    from app.services.journal.service import journal_entry_points
    from core.db.models import JournalEntry

    repo = ctx.repository()
    live = (JournalEntry.deleted_at.is_(None),)
    if ctx.state_get("select_total") is None:
        with repo.session() as session:
            total = session.query(func.count(JournalEntry.id)).filter(*live, JournalEntry.indexed_at.is_not(None)).scalar()
            pending = session.query(func.count(JournalEntry.id)).filter(*live, JournalEntry.indexed_at.is_(None)).scalar()
        # An entry whose vectors were never written has no points to replace; reindex_pending owns it.
        out.counts["pending_skipped"] = int(pending or 0)
        ctx.state_set(select_total=int(total or 0), select_counts=out.counts)
        sink(f"select: {total} indexed live entries to rebuild, {pending} pending entries left alone")
    last_id = int(ctx.state_get("select_offset", 0) or 0)
    scanned = int(ctx.state_get("select_scanned", 0) or 0)
    while True:
        built: list[tuple[str, list, list]] = []
        with repo.session() as session:
            entries = (session.query(JournalEntry).options(selectinload(JournalEntry.project))
                       .filter(*live, JournalEntry.indexed_at.is_not(None), JournalEntry.id > last_id)
                       .order_by(JournalEntry.id).limit(JOURNAL_ENTRY_BATCH).all())
            for entry in entries:
                chunks, points = journal_entry_points(entry)
                built.append((entry.entry_id, chunks, points))
            last_id = entries[-1].id if entries else last_id
        if not built:
            out.flush(select_offset=last_id, select_scanned=scanned)
            return
        current = _current_point_ids(ctx, [entry_id for entry_id, _, _ in built])
        for entry_id, chunks, points in built:
            new_ids = {point_id for point_id, _, _ in points}
            for point_id, payload, text in points:
                out.records.append({"point_id": point_id, "content": text,
                                    "payload_json": json.dumps(payload, ensure_ascii=False)})
            out.old_ids.extend(sorted(current.get(entry_id, set()) - new_ids))
            out.sql_rows.append({"entry_id": entry_id, "chunks": [
                {"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "level": c.level, "heading": c.heading,
                 "content": c.content, "token_count": c.token_count, "metadata": c.metadata} for c in chunks]})
            out.counts[CHUNK] += len(points)
            out.counts["entries"] = out.counts.get("entries", 0) + 1
            out.counts["old_ids"] = out.counts.get("old_ids", 0) + len(current.get(entry_id, set()) - new_ids)
        scanned += len(built)
        ctx.abort_if_cancelled()
        if out.full or len(built) < JOURNAL_ENTRY_BATCH:
            out.flush(select_offset=last_id, select_scanned=scanned)
            sink(f"select: {scanned} entries, {out.counts}")


def _current_point_ids(ctx: ReembedContext, entry_ids: list[str]) -> dict[str, set[int]]:
    """The point ids Qdrant holds now for each of ``entry_ids`` (entry_id is keyword-indexed)."""
    flt = models.Filter(must=[models.FieldCondition(key="entry_id", match=models.MatchAny(any=entry_ids))])
    found: dict[str, set[int]] = {}
    offset = None
    while True:
        points, offset = retry_qdrant_operation(f"scroll(collection={ctx.collection} entries)", lambda: ctx.client.scroll(
            ctx.collection, scroll_filter=flt, limit=SCROLL_PAGE, offset=offset, with_payload=["entry_id"],
            with_vectors=False))
        for point in points:
            found.setdefault(point.payload["entry_id"], set()).add(point.id)
        if offset is None:
            return found


def select_probe(ctx: ReembedContext) -> Progress:
    done, total = int(ctx.state_get("select_scanned", 0) or 0), int(ctx.state_get("select_total", 0) or 0)
    return Progress(done=done, total=max(total, done), message=f"{ctx.state_get('select_counts') or ''}")


def select_gate(ctx: ReembedContext) -> GateResult:
    if not ctx.state_get("select_complete"):
        return GateResult(False, "select did not finish")
    return GateResult(True, f"{ctx.state_get('select_counts')}")


def iter_stamp_ids(ctx: ReembedContext) -> Iterator[int]:
    for path in sorted(ctx.select_dir.glob("stamp_*.txt")):
        for token in path.read_text(encoding="utf-8").split():
            yield int(token)


def iter_oversized(ctx: ReembedContext) -> Iterator[tuple[int, dict]]:
    """The oversized points the select pass listed, one ``(point_id, row)`` at a time; a point a
    rescanned page listed twice comes once, as its last row.

    Streamed, never loaded whole: the rows carry full payloads (898 MB for code_chunks on
    2026-11-14) and reading the file at once swapped .88 until the backend stopped answering. The
    first pass keeps only each point's last line offset."""
    path = ctx.select_dir / "oversized.jsonl"
    if not path.exists():
        return
    last: dict[int, int] = {}
    with path.open("rb") as fh:
        offset = 0
        for line in fh:
            if line.strip():
                last[json.loads(line)["point_id"]] = offset
            offset += len(line)
        for point_id, at in sorted(last.items(), key=lambda item: item[1]):
            fh.seek(at)
            yield point_id, json.loads(fh.readline())
