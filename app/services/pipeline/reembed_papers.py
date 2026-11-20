"""The ``papers:rebuild`` and ``arxiv:rebuild`` scopes: paper documents with chunks that are too long.

The old paper worker cut every chunk at 2,048 tokens, so a chunk over that was embedded
truncated; the worst were bodies mistaken for an abstract (``## Abstract`` with no heading after
it). Such a chunk is wrong itself, not just badly embedded, so its document is re-chunked from its
Markdown with the current chunker and replaced, the same shape as the journal rebuild: the new
points go in first, then the SQL chunk rows, and the old points are deleted last. Every step can
run again without changing anything.

  select   candidates from SQLite (``chunks`` scanned by id range, with checkpoints and a memory
           guard), exact token counts of the candidates, then per document the chunker's output,
           ``build_document_reconciliation`` (shared with scripts/reconcile_paper_chunk_schema.py)
           and the expected points Qdrant does not have yet, as embed records shaped like
           ``scripts/remote_embed/dump_chunks.py`` writes them
  embed    the existing stage: only those new chunks
  write    upsert with the live ingest's own payload code, then the SQL plans, then delete stale points
  verify   per rebuilt document: the points in Qdrant are exactly the SQL chunks, one L1, none over the limit
"""
from __future__ import annotations

import gzip
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import psutil
from qdrant_client import models

from app.models.schemas import Chunk
from app.services.pipeline.reembed_context import ReembedContext
from app.services.pipeline.reembed_verify import SAMPLE_SIZE, check_points
from app.services.pipeline.reembed_write import _append_line, _read_lines
from app.services.pipeline.spec import GateResult
from core.indexing.chunker import create_chunker
from core.indexing.embedding_profile import get_profile, tokens_over_limit
from core.indexing.paper_chunk_reconcile import (
    DocumentReconciliation,
    apply_document_plan_atomically,
    build_document_reconciliation,
    chunk_metadata_json,
    chunk_point_id,
    load_existing_chunks,
)
from core.storage.qdrant_operations import retry_qdrant_operation

PAPER_SOURCE_TYPES = ("paper", "review")
SQL_COMMIT_DOCS = 50
LOOKUP_BATCH = 500
DELETE_BATCH = 1000
SCROLL_PAGE = 1000
COUNT_BATCH = 256
LISTED = 20


# ------------------------------------------------------------------------------ memory guard
def available_memory_bytes() -> int:
    """The host's MemAvailable (psutil reads /proc/meminfo on Linux)."""
    return int(psutil.virtual_memory().available)


def _gib(size: int) -> str:
    return f"{size / (1 << 30):.2f} GiB"


def wait_for_memory(ctx: ReembedContext, sink) -> None:
    """Hold while MemAvailable is under ``ctx.mem_pause_below``; go on once it is back above
    ``ctx.mem_resume_above``. Between the two bounds a running scan is not stopped and a paused
    one is not resumed, so a reading that wobbles around one threshold cannot make it flap."""
    available = available_memory_bytes()
    if available >= ctx.mem_pause_below:
        return
    sink(f"select paused: MemAvailable {_gib(available)} is under {_gib(ctx.mem_pause_below)}; "
         f"resumes above {_gib(ctx.mem_resume_above)}")
    started = time.monotonic()
    while True:
        ctx.abort_if_cancelled()
        time.sleep(ctx.mem_poll_s)
        available = available_memory_bytes()
        if available >= ctx.mem_resume_above:
            break
    sink(f"select resumed after {time.monotonic() - started:.0f} s: MemAvailable {_gib(available)}")


# ----------------------------------------------------------------------------------- SQL / IO
def open_papers_db(ctx: ReembedContext, *, read_only: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(str(ctx.papers_db_path()), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA query_only=ON" if read_only else "PRAGMA foreign_keys=ON")
    return conn


def chunk_to_json(chunk: Chunk) -> dict:
    return {"content": chunk.content, "level": chunk.level, "start_idx": chunk.start_idx,
            "end_idx": chunk.end_idx, "metadata": chunk.metadata}


def chunk_from_json(doc_id: int, row: dict) -> Chunk:
    return Chunk(id=f"rebuild-{doc_id}-{row['level']}", doc_id=str(doc_id), content=row["content"],
                 level=row["level"], start_idx=row["start_idx"], end_idx=row["end_idx"], metadata=row["metadata"])


def read_plans(ctx: ReembedContext) -> dict[int, dict]:
    """The per-document plans the select pass wrote: ``{doc_id: {expected, stale_point_ids, ...}}``."""
    plans: dict[int, dict] = {}
    for path in sorted(ctx.select_dir.glob("paper_plans_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                plans[int(row["doc_id"])] = row
    return plans


def read_failures(ctx: ReembedContext) -> list[dict]:
    path = ctx.select_dir / "paper_failures.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _has_changes(plan: DocumentReconciliation) -> bool:
    return bool(plan.inserts or plan.delete_row_ids or any(item.requires_update for item in plan.retained))


# --------------------------------------------------------------------------------- candidates
def double_l1_documents(conn: sqlite3.Connection) -> list[int]:
    """Documents with more than one L1 chunk: a giant abstract that ``enforce_budget`` cut in pieces.

    A document's rows need not sit in one id range (a later reconcile run adds rows at the end), so
    this is one query over the (doc_id, level) index rather than a per-batch count."""
    return [row[0] for row in conn.execute(
        "SELECT doc_id FROM chunks WHERE level = 1 GROUP BY doc_id HAVING COUNT(*) > 1 ORDER BY doc_id")]


def documents_with_long_chunks(conn: sqlite3.Connection, low: int, high: int, limit: int) -> list[int]:
    """Documents with a chunk over ``limit`` real tokens among the rows with ``low <= id < high``.

    SQL only narrows by length: ``token_count`` holds characters, and a chunk of at most ``limit``
    characters cannot be over ``limit`` tokens. The tokenizer then decides exactly."""
    profile, flagged = get_profile(), set()
    rows = conn.execute("SELECT doc_id, content FROM chunks WHERE id >= ? AND id < ? "
                        "AND (token_count > ? OR token_count IS NULL)", (low, high, limit))
    while True:
        batch = rows.fetchmany(COUNT_BATCH)
        if not batch:
            return sorted(flagged)
        flagged.update(batch[i][0] for i, _ in tokens_over_limit([content for _, content in batch], limit, profile))


@dataclass(frozen=True)
class _Document:
    title: Optional[str]
    file_path: Optional[str]
    paper_title: Optional[str]
    venue: Optional[str]
    year: Optional[int]
    source_type: Optional[str]
    venue_tier: Optional[str]
    dump_columns: tuple      # the document/paper columns ``dump_chunks.build_payload`` takes after the chunk's own


class _Planner:
    """One select run's per-document work: the chunker's output reconciled with the SQL rows."""

    def __init__(self, ctx: ReembedContext, conn: sqlite3.Connection, out, sink):
        from config import settings
        from core.storage.corpus_layout import current_layout, paper_collections
        from scripts.remote_embed.dump_chunks import build_payload

        self.ctx, self.conn, self.out, self.sink = ctx, conn, out, sink
        self.chunker = create_chunker("section_l1")
        self.layout = current_layout()
        self.home = paper_collections()[ctx.scope.family]
        self.roots = list(ctx.markdown_roots if ctx.markdown_roots is not None else settings.paper_markdown_roots)
        self.dump_payload = build_payload
        paper_columns = {row[1] for row in conn.execute("PRAGMA table_info(papers)")}
        self.full_text_source = "p.full_text_source" if "full_text_source" in paper_columns else "NULL"
        self.seen = set(read_plans(ctx)) | {int(row["doc_id"]) for row in read_failures(ctx)}

    # ---------------------------------------------------------------------------- rows
    def _document(self, doc_id: int) -> Optional[_Document]:
        row = self.conn.execute(
            "SELECT d.title, d.source, d.source_type, d.metadata_json, d.file_path, "
            "p.paper_id, p.title, p.venue, p.year, p.venue_tier, p.track_type, p.arxiv_id, p.doi, "
            f"p.citation_count, p.withdrawn, {self.full_text_source} "
            "FROM documents d LEFT JOIN papers p ON p.doc_id = d.id WHERE d.id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        title, source, source_type, meta, file_path, paper_id, paper_title, venue, year, tier, *rest = row
        return _Document(
            title=title, file_path=file_path, paper_title=paper_title, venue=venue, year=year,
            source_type=source_type, venue_tier=tier,
            dump_columns=(title, source, source_type, meta, paper_id, venue, year, tier, *rest))

    def _record(self, doc_id: int, doc: _Document, chunk: Chunk, point_id: int) -> dict:
        """The record the paper dump writes for a chunk row, payload metadata included."""
        from scripts.remote_embed.dump_chunks import chunk_id_str

        start = int(chunk.start_idx or 0)
        row = (None, doc_id, chunk.content, chunk.level, start, chunk.end_idx, chunk_metadata_json(chunk.metadata),
               *doc.dump_columns)
        return {"chunk_id_str": chunk_id_str(doc_id, chunk.content, chunk.level, start), "point_id": point_id,
                "doc_id": doc_id, "content": chunk.content, "level": chunk.level, "start_idx": start,
                "end_idx": chunk.end_idx, "metadata": self.dump_payload(row)}

    # --------------------------------------------------------------------------- planning
    def _count(self, key: str, amount: int = 1) -> None:
        self.out.counts[key] = self.out.counts.get(key, 0) + amount

    def _fail(self, doc_id: int, error: str) -> None:
        self.out.failures.append({"doc_id": doc_id, "error": error})
        self._count("failed")

    def _plan(self, doc_id: int, doc: _Document) -> tuple[list[Chunk], DocumentReconciliation]:
        from core.storage.artifact_resolver import resolve_paper_artifact

        path = resolve_paper_artifact(doc.file_path, doc.venue, doc.year, self.roots)
        if path is None:
            raise FileNotFoundError(f"markdown file not found for {doc.file_path!r}")
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        expected = self.chunker.chunk_document(str(doc_id), text, title_hint=doc.paper_title or doc.title)
        return expected, build_document_reconciliation(doc_id, load_existing_chunks(self.conn, doc_id), expected)

    def process(self, doc_id: int) -> None:
        from core.storage.corpus_layout import collection_for_payload

        if doc_id in self.seen:
            return
        self.seen.add(doc_id)
        wait_for_memory(self.ctx, self.sink)
        self.ctx.abort_if_cancelled()
        doc = self._document(doc_id)
        if doc is None:
            return self._fail(doc_id, "documents row is missing")
        routed = collection_for_payload({"source_type": doc.source_type, "venue_tier": doc.venue_tier}, self.layout)
        if doc.source_type not in PAPER_SOURCE_TYPES or routed != self.home:
            return self._count("foreign")        # another collection's (or an upload's) document
        try:
            expected, plan = self._plan(doc_id, doc)
        except Exception as exc:                 # a document that cannot be rebuilt is listed, never skipped silently
            return self._fail(doc_id, f"{type(exc).__name__}: {exc}")
        ids = [chunk_point_id(doc_id, chunk) for chunk in expected]
        missing = _missing_points(self.ctx, ids)
        records, emitted = [], set()
        for chunk, point_id in zip(expected, ids):
            if point_id in missing and point_id not in emitted:
                emitted.add(point_id)
                records.append(self._record(doc_id, doc, chunk, point_id))
        if not records and not plan.stale_point_ids and not _has_changes(plan):
            return self._count("noop")           # already what the chunker emits now
        self.out.records.extend(records)
        self.out.plans.append({"doc_id": doc_id, "paper_id": doc.dump_columns[4],
                               "expected": [chunk_to_json(c) for c in expected],
                               "stale_point_ids": plan.stale_point_ids})
        self._count("docs")
        self._count("chunk", len(records))
        self._count("stale_points", len(plan.stale_point_ids))


def _missing_points(ctx: ReembedContext, ids: list[int]) -> set[int]:
    """The ids of ``ids`` that Qdrant does not hold."""
    present: set[int] = set()
    for start in range(0, len(ids), LOOKUP_BATCH):
        batch = ids[start:start + LOOKUP_BATCH]
        found = retry_qdrant_operation(f"retrieve(collection={ctx.collection} expected)", lambda batch=batch: ctx.client.retrieve(
            ctx.collection, ids=batch, with_payload=False, with_vectors=False))
        present.update(point.id for point in found)
    return set(ids) - present


# ---------------------------------------------------------------------------------- select
def select_papers(ctx: ReembedContext, sink, out) -> None:
    """The select stage of ``paper_rebuild``: see the module doc. ``out`` is the stage's SelectOutput."""
    limit = get_profile().chunk_max_tokens
    for key in ("docs", "stale_points", "noop", "foreign", "failed"):
        out.counts.setdefault(key, 0)
    conn = open_papers_db(ctx, read_only=True)
    try:
        planner = _Planner(ctx, conn, out, sink)
        if ctx.state_get("select_total") is None:
            ctx.state_set(select_total=int(conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]))
        total = int(ctx.state_get("select_total"))
        if not ctx.state_get("select_double_l1_done"):
            double = double_l1_documents(conn)
            sink(f"select: {len(double)} documents hold more than one L1 chunk; chunks over {limit} tokens "
                 f"are looked for in id order up to {total:,}")
            for doc_id in double:
                planner.process(doc_id)
            out.flush(select_double_l1_done=True, select_offset=0, select_scanned=0)
        low, since_checkpoint = int(ctx.state_get("select_offset", 0) or 0), 0
        while low <= total:
            wait_for_memory(ctx, sink)
            ctx.abort_if_cancelled()
            high = low + ctx.scan_batch
            for doc_id in documents_with_long_chunks(conn, low, high, limit):
                planner.process(doc_id)
            low, since_checkpoint = high, since_checkpoint + ctx.scan_batch
            time.sleep(ctx.pause_s)               # the scan shares the disk with live search
            if out.full or low > total or since_checkpoint >= ctx.checkpoint_rows:
                out.flush(select_offset=low, select_scanned=min(low, total))
                since_checkpoint = 0
                sink(f"select: chunk ids below {min(low, total):,} of {total:,} scanned, {out.counts}")
    finally:
        conn.close()


# ----------------------------------------------------------------------------------- write
def paper_payload_reader(ctx: ReembedContext) -> Callable[[str], dict]:
    """``payload_json`` of an embedded record -> the point payload, by the live ingest's own code.

    ``local_ingest.canonicalize_payload`` is what ``scripts/remote_embed/local_ingest.py`` projects
    every paper point through, and the pre-write guard refuses a point that belongs to another
    collection before anything is written."""
    from core.storage.corpus_layout import check_paper_point, collection_for_payload, current_layout, paper_collections
    from scripts.remote_embed.local_ingest import canonicalize_payload

    layout = current_layout()
    home = paper_collections()[ctx.scope.family]

    def read(payload_json: str) -> dict:
        payload = canonicalize_payload(json.loads(payload_json))
        if collection_for_payload(payload, layout) != home:
            raise RuntimeError(f"point of paper {payload.get('paper_id')!r} does not belong to {home!r}")
        check_paper_point(home, payload, layout)
        return payload

    return read


def apply_sql_plans(ctx: ReembedContext, sink) -> int:
    """Replace the chunk rows of every planned document. Each document's plan is made again against
    the rows as they are now, so a document already replaced has an empty plan: running this twice
    changes nothing."""
    done_file = ctx.select_dir / "paper_sql_done.txt"
    done = set(_read_lines(done_file))
    replaced = 0
    conn = open_papers_db(ctx, read_only=False)
    try:
        for path in sorted(ctx.select_dir.glob("paper_plans_*.jsonl.gz")):
            if path.name in done:
                continue
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for number, line in enumerate(fh, 1):
                    ctx.abort_if_cancelled()
                    row = json.loads(line)
                    doc_id = int(row["doc_id"])
                    if conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone() is None:
                        sink(f"paper rebuild: document {doc_id} no longer exists, left alone")
                        continue
                    expected = [chunk_from_json(doc_id, chunk) for chunk in row["expected"]]
                    plan = build_document_reconciliation(doc_id, load_existing_chunks(conn, doc_id), expected)
                    if _has_changes(plan):
                        apply_document_plan_atomically(conn, doc_id, plan)
                        replaced += 1
                    if number % SQL_COMMIT_DOCS == 0:
                        conn.commit()
            conn.commit()
            _append_line(done_file, path.name)
    finally:
        conn.close()
    sink(f"paper rebuild: SQL chunks replaced for {replaced} documents")
    return replaced


def delete_stale_points(ctx: ReembedContext, sink) -> int:
    """Delete the points of chunks that no longer exist (deleting an absent point is a no-op)."""
    stale = sorted({int(point_id) for row in read_plans(ctx).values() for point_id in row["stale_point_ids"]})
    for start in range(0, len(stale), DELETE_BATCH):
        ctx.abort_if_cancelled()
        batch = stale[start:start + DELETE_BATCH]
        retry_qdrant_operation(f"delete(collection={ctx.collection} stale n={len(batch)})", lambda batch=batch: ctx.client.delete(
            ctx.collection, points_selector=models.PointIdsList(points=batch), wait=True))
        time.sleep(ctx.pause_s)
    sink(f"paper rebuild: {len(stale)} stale points deleted")
    return len(stale)


def finalize_paper_rebuild(ctx: ReembedContext, sink) -> None:
    """After every new point is written: the SQL rows follow, then the old points go. The document
    stays searchable throughout (old and new points coexist until the last step)."""
    apply_sql_plans(ctx, sink)
    delete_stale_points(ctx, sink)
    ctx.state_set(paper_finalized=True)


# ---------------------------------------------------------------------------------- verify
def _points_of_document(ctx: ReembedContext, conn: sqlite3.Connection, doc_id: int) -> dict[int, Any]:
    """``{point id: level}`` of the document's points: by ``doc_id`` and by ``paper_id``, because the
    ``doc_id`` index only holds the points whose ``doc_id`` is a string."""
    should = [models.FieldCondition(key="doc_id", match=models.MatchValue(value=str(doc_id)))]
    paper = conn.execute("SELECT paper_id FROM papers WHERE doc_id = ?", (doc_id,)).fetchone()
    if paper is not None:
        should.append(models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper[0])))
    found: dict[int, Any] = {}
    offset = None
    while True:
        points, offset = retry_qdrant_operation(f"scroll(collection={ctx.collection} doc {doc_id})", lambda offset=offset: ctx.client.scroll(
            ctx.collection, scroll_filter=models.Filter(should=should), limit=SCROLL_PAGE, offset=offset,
            with_payload=["level"], with_vectors=False))
        found.update({point.id: (point.payload or {}).get("level") for point in points})
        if offset is None:
            return found


def _verify_document(ctx: ReembedContext, conn: sqlite3.Connection, doc_id: int, limit: int) -> list[str]:
    from scripts.remote_embed.dump_chunks import chunk_id_int, chunk_id_str

    rows = conn.execute("SELECT content, level, start_idx FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    problems = []
    l1 = sum(1 for _, level, _ in rows if level == 1)
    if l1 != 1:
        problems.append(f"document {doc_id} has {l1} L1 chunks in SQL (exactly one is required)")
    over = tokens_over_limit([content for content, _, _ in rows], limit, get_profile())
    if over:
        problems.append(f"document {doc_id} has {len(over)} chunk(s) over {limit} tokens (longest {max(t for _, t in over)})")
    wanted = {chunk_id_int(chunk_id_str(doc_id, content, level, start or 0)) for content, level, start in rows}
    stored = _points_of_document(ctx, conn, doc_id)
    missing, extra = wanted - stored.keys(), stored.keys() - wanted
    if missing:
        problems.append(f"document {doc_id}: {len(missing)} point(s) missing in Qdrant, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"document {doc_id}: {len(extra)} extra point(s) in Qdrant, e.g. {sorted(extra)[:3]}")
    stored_l1 = sum(1 for level in stored.values() if level == 1)
    if stored_l1 != 1 and not missing and not extra:
        problems.append(f"document {doc_id} has {stored_l1} L1 points in Qdrant (exactly one is required)")
    return problems


def _new_point_sample(ctx: ReembedContext) -> list[int]:
    ids: list[int] = []
    for path in sorted(ctx.chunks_dir.glob("chunks_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            ids.extend(json.loads(line)["point_id"] for line in fh)
    return random.Random(ctx.task_id).sample(ids, min(SAMPLE_SIZE, len(ids)))


def verify_paper_rebuild(ctx: ReembedContext) -> GateResult:
    """Every rebuilt document's points are exactly its SQL chunks, and a sample of the new points
    equals a fresh encode. Documents that could not be rebuilt fail the gate: they are not repaired."""
    if not ctx.state_get("select_complete"):
        return GateResult(False, "select did not finish")
    plans, failures = read_plans(ctx), read_failures(ctx)
    if plans and not ctx.state_get("paper_finalized"):
        return GateResult(False, "paper rebuild not finalized")
    limit = get_profile().chunk_max_tokens
    problems: list[str] = []
    if failures:
        listed = "; ".join(f"{row['doc_id']} ({row['error']})" for row in failures[:LISTED])
        problems.append(f"{len(failures)} document(s) could not be rebuilt: {listed}")
    conn = open_papers_db(ctx, read_only=True)
    try:
        for doc_id in sorted(plans):
            ctx.abort_if_cancelled()
            problems.extend(_verify_document(ctx, conn, doc_id, limit))
    finally:
        conn.close()
    if problems:
        return GateResult(False, f"{len(problems)} problem(s) in {ctx.scope_name}: " + " | ".join(problems[:LISTED]))
    sampled = _new_point_sample(ctx)
    if not sampled:
        return GateResult(True, f"{len(plans)} documents rebuilt, no new points to sample")
    try:
        points = retry_qdrant_operation(f"retrieve(collection={ctx.collection} sample)", lambda: ctx.client.retrieve(
            ctx.collection, ids=sampled, with_payload=[ctx.scope.text_field], with_vectors=["dense", "sparse"]))
        sample = check_points(ctx, list(points))
    except RuntimeError as exc:
        return GateResult(False, str(exc))
    return GateResult(True, f"{len(plans)} documents rebuilt, 0 problems; {sample.summary}")
