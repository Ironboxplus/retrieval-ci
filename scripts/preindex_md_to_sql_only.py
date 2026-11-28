#!/usr/bin/env python
"""Markdown -> SQL only (no embed, no Qdrant).

Stage F Step 1 实现：把已抽出 markdown 的论文 chunk 进 SQL。
chunks 表填好后由 dump_chunks.py 导出给 kss 远程 BGE-M3 embed，最后再
local_ingest.py upsert Qdrant。

设计原则：
  - 全程 CPU，不动 embedder / GPU / QdrantStore
  - 复用 PaperIndexer._prepare_paper + chunker.chunk_document 同款 Stage C chunk
  - 直写 Document/Paper/Chunk ORM，**bypass** chroma_store.add_chunks
  - --limit N 做 smoke（写够 N 篇新/重建即停），--limit 0 全量
  - 幂等 upsert，按 paper_id 分三类（报告 new / replaced / unchanged）：
      new        papers 里没有 -> 写 Document/Paper/Chunk
      replaced   markdown 的 sha256 与已存 Document.content_hash 不同 -> 写新 Document，
                 papers.doc_id 指向它；旧 doc_id 记入报告 superseded，由 cleanup stage
                 先删 Qdrant 点再删旧 Document（本脚本不删）
      unchanged  相同 -> 不动正文
    存量文档 content_hash 为空时无从比较文本，退而比较 papers.source_version（清单的
    版本与已存的相同即 unchanged，并补写哈希），首轮不会把整个语料重建。
    三类都按清单更新 papers.source_version / withdrawn；撤稿标志变化的 paper_id 记入
    报告 withdrawn_changed，由 cleanup stage 同步到 Qdrant。
  - 报告在同一 --report-path 重跑时累积 new/replaced/superseded/withdrawn_changed：
    崩溃重跑时第二遍里这些论文已是 unchanged，不累积就会丢掉待删的旧文档。
    删除报告文件即从头开始。
  - 每 100 篇 checkpoint 写 report；含 new/replaced/撤稿变化的批次在 commit 之前先写
    （write-ahead）：报告里多出的 superseded 无害（cleanup 只删已无 papers 引用的
    文档），少了才会漏删；多出的 new 同样无害（commit 失败则该 paper 没有行，embed
    导出不到任何 chunk），少了则 embed 按报告导出时永远漏掉这些论文。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.db import get_db_session
from core.db.sqlite_retry import is_lock_contention, retry_transaction
from core.db.models import (
    DataSource,
    Document as DBDocument,
    DocumentStatus,
    Paper as DBPaper,
    SourceType,
)
from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper, build_default_corpus_root
from pipeline.paper_crawler.integration.index_integration import (
    PaperIndexConfig,
    PaperIndexer,
)
from pipeline.paper_crawler.runtime_files import load_json_with_recovery, write_json_atomic


DEFAULT_MANIFEST = build_default_corpus_root() / "manifests" / "paper_manifest.json"
DEFAULT_REPORT = build_default_corpus_root() / "reports" / "preindex_md_to_sql_only.json"
DEFAULT_PAPER_CHUNKER = "section_l1"
PAPER_CHUNKER_CHOICES = ("section_l1", "semantic", "multigranularity")
SQLITE_MAX_IN = 900  # stay under SQLite's historical 999-variable limit per IN list

NEW, REPLACED, UNCHANGED = "new", "replaced", "unchanged"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MD->SQL only (skip embed/Qdrant).")
    p.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many new/replaced papers were written; 0 = no cap")
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--commit-batch", type=int, default=20, help="SQL commit every N papers")
    p.add_argument(
        "--chunker-type",
        choices=PAPER_CHUNKER_CHOICES,
        default=DEFAULT_PAPER_CHUNKER,
        help=(
            "Paper chunk schema. Default 'section_l1' emits one title/summary "
            "L1 plus section-aware L2 chunks. L2-only 'markdown' is deliberately "
            "not accepted because title_only retrieval requires L1."
        ),
    )
    p.add_argument(
        "--conf", type=str, default=None,
        help="Optional: only process this conference (case-insensitive match on "
             "manifest 'conference'). Default None = all conferences.",
    )
    p.add_argument(
        "--year", type=int, default=None,
        help="Optional: only process this year. Default None = all years.",
    )
    return p


def _filter_by_conf_year(
    records: list[CorpusPaper],
    conf: str | None,
    year: int | None,
) -> list[CorpusPaper]:
    """Additively narrow records by conference (case-insensitive) and/or year.

    Generic for ANY conference. conf=None and year=None → records unchanged
    (preserves the existing no-filter behaviour).
    """
    filtered = records
    if conf is not None:
        conf_lc = conf.lower()
        filtered = [r for r in filtered if (r.conference or "").lower() == conf_lc]
    if year is not None:
        filtered = [r for r in filtered if r.year == year]
    return filtered


def _eligible(records: list[CorpusPaper]) -> list[CorpusPaper]:
    return [r for r in records if r.markdown_path and not r.extract_error]


# ------------------------------------------------------------------ SQL state
class Existing(NamedTuple):
    """What SQL already holds for one paper_id."""

    doc_id: int
    source_version: Optional[str]
    withdrawn: bool
    content_hash: Optional[str]


def content_hash(text: str) -> str:
    """sha256 of the Markdown as read (Document.content_hash)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _existing_papers(paper_ids: Iterable[str]) -> dict[str, Existing]:
    """paper_id -> Existing for the ids already in SQL (PK lookups, chunked)."""
    wanted = sorted(set(paper_ids))
    found: dict[str, Existing] = {}
    with get_db_session() as db:
        for start in range(0, len(wanted), SQLITE_MAX_IN):
            rows = (
                db.query(DBPaper.paper_id, DBPaper.doc_id, DBPaper.source_version,
                         DBPaper.withdrawn, DBDocument.content_hash)
                .join(DBDocument, DBDocument.id == DBPaper.doc_id)
                .filter(DBPaper.paper_id.in_(wanted[start:start + SQLITE_MAX_IN]))
                .all()
            )
            for pid, doc_id, version, withdrawn, digest in rows:
                found[pid] = Existing(doc_id, version, bool(withdrawn), digest)
    return found


def classify(existing: Optional[Existing], digest: str, record: CorpusPaper) -> str:
    """new / replaced / unchanged for one manifest record (see the module doc)."""
    if existing is None:
        return NEW
    if existing.content_hash:
        return UNCHANGED if existing.content_hash == digest else REPLACED
    # Stored before hashes were written: the text cannot be compared, so the
    # version the row carries stands in for it (None == None for conference rows).
    return UNCHANGED if existing.source_version == record.source_version else REPLACED


# --------------------------------------------------------------------- report
class Outcome(NamedTuple):
    paper_id: str
    kind: str
    n_chunks: int = 0
    superseded_doc_id: Optional[int] = None
    withdrawn_change: Optional[bool] = None  # the new flag, when it changed


class Report:
    """The run's lists. Insertion-ordered dicts, so membership is O(1) at 130K ids."""

    def __init__(self, new=None, replaced=None, superseded=None, withdrawn_changed=None):
        self.new: dict[str, None] = new or {}
        self.replaced: dict[str, None] = replaced or {}
        self.unchanged: dict[str, None] = {}
        self.superseded: dict[int, str] = superseded or {}  # old doc_id -> paper_id
        self.withdrawn_changed: dict[str, bool] = withdrawn_changed or {}  # paper_id -> new flag

    @classmethod
    def resume(cls, path: Path) -> "Report":
        """Start from the lists an earlier run of this report path already wrote."""
        previous = load_json_with_recovery(path, default_value={})
        return cls(
            new=dict.fromkeys(previous.get("new", [])),
            replaced=dict.fromkeys(previous.get("replaced", [])),
            superseded={int(e["doc_id"]): e["paper_id"] for e in previous.get("superseded", [])},
            withdrawn_changed={e["paper_id"]: bool(e["withdrawn"])
                               for e in previous.get("withdrawn_changed", [])},
        )

    def add(self, outcome: Outcome) -> None:
        {NEW: self.new, REPLACED: self.replaced, UNCHANGED: self.unchanged}[outcome.kind][outcome.paper_id] = None
        if outcome.superseded_doc_id is not None:
            self.superseded[outcome.superseded_doc_id] = outcome.paper_id
        if outcome.withdrawn_change is not None:
            self.withdrawn_changed[outcome.paper_id] = outcome.withdrawn_change

    def as_json(self) -> dict[str, Any]:
        return {
            "counts": {"new": len(self.new), "replaced": len(self.replaced),
                       "unchanged": len(self.unchanged), "superseded": len(self.superseded),
                       "withdrawn_changed": len(self.withdrawn_changed)},
            "new": list(self.new), "replaced": list(self.replaced),
            "unchanged": list(self.unchanged),
            "superseded": [{"doc_id": d, "paper_id": p} for d, p in self.superseded.items()],
            "withdrawn_changed": [{"paper_id": p, "withdrawn": w}
                                  for p, w in self.withdrawn_changed.items()],
        }


# ------------------------------------------------------------------ one paper
def _write_one_paper_to_sql(
    *,
    indexer: PaperIndexer,
    record: CorpusPaper,
    markdown_path: Path,
    text: str,
    digest: str,
    db,
) -> tuple[int, int, DBPaper]:
    """Build chunks (CPU) and write Document + Paper + Chunk rows.

    Returns (doc_id, chunk_count, paper row). For a paper SQL already has, the
    existing row is re-pointed at the new Document (PaperIndexer._upsert_paper_row).
    """
    paper = record.to_runtime_paper() if hasattr(record, "to_runtime_paper") else record
    prepared = indexer._prepare_paper(paper, markdown_path, content=text)
    chunks = indexer.index_service.chunker.chunk_document(
        doc_id="_tmp", text=prepared["content"],
        include_summary=indexer.config.include_summary,
        title_hint=paper.title,
    )
    _validate_paper_chunk_schema(chunks)

    doc = DBDocument(
        title=paper.title,
        source=indexer._paper_url(paper),
        source_type=SourceType.paper,
        data_source=DataSource(indexer.config.data_source),
        file_path=str(prepared["markdown_path"]) if prepared["markdown_path"] else None,
        url=indexer._paper_url(paper),
        status=DocumentStatus.processing,
        content_hash=digest,
        metadata_json=prepared["metadata"],
    )
    db.add(doc)
    db.flush()
    doc_id = doc.id

    for ch in chunks:
        ch.doc_id = str(doc_id)

    indexer.index_service._save_chunks_to_db(doc_id, chunks, db)
    doc.status = DocumentStatus.indexed
    row = indexer._upsert_paper_row(
        db=db, doc_id=doc_id, paper=paper,
        metadata=prepared["metadata"], markdown_path=prepared["markdown_path"],
    )
    return doc_id, len(chunks), row


def _sync_flags(row: DBPaper, record: CorpusPaper, was_withdrawn: bool) -> Optional[bool]:
    """Copy the manifest's source_version / withdrawn onto the paper row.

    A field the manifest does not carry (conference rows, older manifests) leaves
    the column alone. Returns the new withdrawn flag when it changed.
    """
    if record.source_version is not None:
        row.source_version = record.source_version
    if record.full_text_source is not None:
        row.full_text_source = record.full_text_source
    if record.withdrawn is None:
        return None
    now = bool(record.withdrawn)
    row.withdrawn = now
    if now == was_withdrawn:
        return None
    row.withdrawn_at = datetime.now(timezone.utc).isoformat(timespec="seconds") if now else None
    return now


def _process_record(*, indexer: PaperIndexer, record: CorpusPaper, markdown_path: Path,
                    existing: Optional[Existing], db) -> Outcome:
    text = markdown_path.read_text(encoding="utf-8")
    digest = content_hash(text)
    kind = classify(existing, digest, record)
    n_chunks = 0
    superseded = None
    if kind == UNCHANGED:
        if not existing.content_hash:
            db.query(DBDocument).filter(DBDocument.id == existing.doc_id).update(
                {"content_hash": digest}, synchronize_session=False)
        row = db.get(DBPaper, record.paper_id)
    else:
        _, n_chunks, row = _write_one_paper_to_sql(
            indexer=indexer, record=record, markdown_path=markdown_path,
            text=text, digest=digest, db=db,
        )
        if kind == REPLACED:
            superseded = existing.doc_id
    change = _sync_flags(row, record, existing.withdrawn if existing else False)
    return Outcome(record.paper_id, kind, n_chunks, superseded, change)


def _validate_paper_chunk_schema(chunks: list) -> None:
    if not chunks:
        raise RuntimeError("no chunks generated")
    l1_count = sum(1 for chunk in chunks if chunk.level == 1)
    if l1_count != 1:
        raise RuntimeError(
            "paper chunk schema requires exactly one level-1 title/summary "
            f"chunk; generated {l1_count}"
        )


def _set_sqlite_pragma_fast():
    """Enable WAL + relaxed sync for bulk inserts (1.5-2x speedup vs default journal=DELETE)."""
    from core.db.database import engine
    with engine.connect() as conn:
        cur = conn.connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA cache_size=-65536")  # 64MB page cache
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.close()
    print("[pragma] journal_mode=WAL, synchronous=NORMAL, cache=64MB", flush=True)


# ------------------------------------------------------------------------ run
def run(args) -> dict[str, Any]:
    # A lock timeout is infrastructure contention, not a rejected paper. The
    # write-ahead report makes the whole run replay-safe after rollback.
    return retry_transaction(lambda: _run_once(args))


def _run_once(args) -> dict[str, Any]:
    _set_sqlite_pragma_fast()
    print(f"[load] manifest = {args.manifest_path}", flush=True)
    payload = load_json_with_recovery(args.manifest_path, default_value=[])
    records = [CorpusPaper.from_manifest_record(r) for r in payload]

    conf = getattr(args, "conf", None)
    year = getattr(args, "year", None)
    if conf is not None or year is not None:
        before = len(records)
        records = _filter_by_conf_year(records, conf, year)
        print(f"[scan] slice filter conf={conf} year={year}: {before} -> {len(records)}", flush=True)

    eligible = _eligible(records)
    print(f"[scan] manifest total = {len(records)}; eligible (md_path+no_extract_err) = {len(eligible)}", flush=True)

    print("[scan] reading existing papers from SQL ...", flush=True)
    existing = _existing_papers(r.paper_id for r in eligible)
    print(f"[scan] eligible already in SQL: {len(existing)}", flush=True)

    print(f"[chunker] chunker_type = {args.chunker_type}", flush=True)
    indexer = PaperIndexer(
        PaperIndexConfig(
            collection_name="papers",
            data_source="system",
            embedding_device="cpu",  # never used; skip_qdrant path
            require_gpu=False,
            max_concurrent_indexing=1,
            chunker_type=args.chunker_type,
        )
    )

    report = Report.resume(args.report_path)
    processed = 0
    written = 0  # new + replaced papers committed this run (what --limit counts)
    failed = 0
    skipped_missing_md = 0
    total_chunks = 0
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    t0 = time.time()

    def write_report(rep: Report = report, **extra) -> None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.report_path, {
            "processed": processed, "indexed": written, "failed": failed,
            "skipped_missing_md": skipped_missing_md, "total_chunks": total_chunks,
            "elapsed_s": time.time() - t0, **rep.as_json(), **extra,
        })

    with get_db_session() as db:
        batch: list[tuple[CorpusPaper, Path, Outcome]] = []

        def pending_writes() -> int:
            return sum(1 for *_, o in batch if o.kind != UNCHANGED)

        def batch_outcomes(orphaning: list[Outcome]) -> list[Outcome]:
            """The batch's written papers plus any unchanged one whose flag flips."""
            return [o for *_, o in batch if o.kind != UNCHANGED or o in orphaning]

        def commit_batch() -> None:
            nonlocal written, total_chunks
            orphaning = [o for *_, o in batch
                         if o.superseded_doc_id is not None or o.withdrawn_change is not None]
            if orphaning or pending_writes() > 0:
                # Write-ahead: the report names every paper this commit writes, every
                # document it orphans and every flag it flips before the commit does it
                # (module doc). Embed dumps the documents of the report's new + replaced
                # papers, so a crash between this commit and the next checkpoint must
                # not lose those names.
                ahead = copy.deepcopy(report)
                for outcome in batch_outcomes(orphaning):
                    ahead.add(outcome)
                write_report(ahead)
            db.commit()
            for *_, outcome in batch:
                report.add(outcome)
                total_chunks += outcome.n_chunks
            written += pending_writes()
            batch.clear()

        def process(record: CorpusPaper, md_path: Path) -> Outcome:
            return _process_record(indexer=indexer, record=record, markdown_path=md_path,
                                   existing=existing.get(record.paper_id), db=db)

        def fail(record: CorpusPaper, exc: Exception) -> None:
            nonlocal failed
            failed += 1
            failures.append({"paper_id": record.paper_id, "err": f"{type(exc).__name__}: {exc}"})

        for record in eligible:
            if args.limit > 0 and written + pending_writes() >= args.limit:
                print(f"[scan] applied --limit {args.limit}", flush=True)
                break
            if record.paper_id in seen:  # a repeated manifest row must not write a second document
                continue
            seen.add(record.paper_id)
            processed += 1
            md_path = Path(record.markdown_path)
            if not md_path.exists():
                skipped_missing_md += 1
                continue
            try:
                batch.append((record, md_path, process(record, md_path)))
            except Exception as e:
                if is_lock_contention(e):
                    raise
                fail(record, e)
                # The failed paper may have left half-written rows in the open
                # transaction. Roll it back, then redo the batch's earlier papers
                # one commit each so none of them is lost with it.
                db.rollback()
                redo, batch[:] = list(batch), []
                for prior, prior_path, _ in redo:
                    try:
                        batch.append((prior, prior_path, process(prior, prior_path)))
                        commit_batch()
                    except Exception as e2:
                        if is_lock_contention(e2):
                            raise
                        fail(prior, e2)
                        db.rollback()
                        batch.clear()
                continue
            if len(batch) >= args.commit_batch:
                commit_batch()

            if processed % max(args.checkpoint_interval, 1) == 0 or processed == len(eligible):
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-3)
                eta_min = (len(eligible) - processed) / max(rate, 1e-3) / 60
                print(
                    f"[ckpt] {processed}/{len(eligible)} "
                    f"written={written} failed={failed} miss_md={skipped_missing_md} "
                    f"chunks={total_chunks} rate={rate:.2f} pap/s eta={eta_min:.1f}min",
                    flush=True,
                )
                write_report(rate_pap_per_sec=rate, failures_tail=failures[-20:])
        if batch:
            commit_batch()

    write_report(failures_tail=failures[-50:])
    return load_json_with_recovery(args.report_path, default_value={})


def main() -> int:
    args = build_argparser().parse_args()
    report = run(args)
    lists = {"failures_tail", "new", "replaced", "unchanged", "superseded", "withdrawn_changed"}
    print(json.dumps({k: v for k, v in report.items() if k not in lists}, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
