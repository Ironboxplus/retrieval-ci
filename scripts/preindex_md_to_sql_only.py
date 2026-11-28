#!/usr/bin/env python
"""Markdown -> SQL only (no embed, no Qdrant).

Stage F Step 1 实现：把已抽出 markdown、但 papers 表还没有的论文 chunk 进 SQL。
chunks 表填好后由 dump_chunks.py 导出给 kss 远程 BGE-M3 embed，最后再
local_ingest.py upsert Qdrant。

设计原则：
  - 全程 CPU，不动 embedder / GPU / QdrantStore
  - 复用 PaperIndexer._prepare_paper + chunker.chunk_document 同款 Stage C chunk
  - 直写 Document/Paper/Chunk ORM，**bypass** chroma_store.add_chunks
  - --limit N 做 smoke，--limit 0 全量
  - 已索引 paper（paper_id 已在 papers 表）自动跳过
  - 每 100 篇 checkpoint 写 report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.db import get_db_session
from core.db.models import (
    Chunk as DBChunk,
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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MD->SQL only (skip embed/Qdrant).")
    p.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--limit", type=int, default=0, help="0 = no cap")
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


def _existing_paper_ids() -> set[str]:
    """Return set of paper_id already in SQL papers (PK lookup, fast)."""
    with get_db_session() as db:
        rows = db.query(DBPaper.paper_id).all()
        return {r[0] for r in rows if r[0]}


def _write_one_paper_to_sql(
    *,
    indexer: PaperIndexer,
    record: CorpusPaper,
    markdown_path: Path,
    db,
) -> tuple[int, int]:
    """Build chunks (CPU) and write Document + Paper + Chunk rows. Returns (doc_id, chunk_count)."""
    paper = record.to_runtime_paper() if hasattr(record, "to_runtime_paper") else record
    prepared = indexer._prepare_paper(paper, markdown_path, content=None)
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
        metadata_json=prepared["metadata"],
    )
    db.add(doc)
    db.flush()
    doc_id = doc.id

    for ch in chunks:
        ch.doc_id = str(doc_id)

    indexer.index_service._save_chunks_to_db(doc_id, chunks, db)
    doc.status = DocumentStatus.indexed
    indexer._upsert_paper_row(
        db=db, doc_id=doc_id, paper=paper,
        metadata=prepared["metadata"], markdown_path=prepared["markdown_path"],
    )
    indexer._paper_doc_mapping[paper.id] = str(doc_id)
    return doc_id, len(chunks)


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


def run(args) -> dict[str, Any]:
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

    print("[scan] reading existing papers.paper_id from SQL ...", flush=True)
    existing_pids = _existing_paper_ids()
    print(f"[scan] existing in SQL: {len(existing_pids)}", flush=True)

    todo = [r for r in eligible if r.paper_id not in existing_pids]
    print(f"[scan] todo (eligible - existing) = {len(todo)}", flush=True)

    if args.limit > 0:
        todo = todo[:args.limit]
        print(f"[scan] applied --limit {args.limit} → {len(todo)} to process", flush=True)

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

    processed = 0
    indexed = 0
    failed = 0
    skipped_missing_md = 0
    total_chunks = 0
    failures: list[dict[str, Any]] = []
    t0 = time.time()

    with get_db_session() as db:
        batch_count = 0
        for record in todo:
            processed += 1
            md_path = Path(record.markdown_path)
            if not md_path.exists():
                skipped_missing_md += 1
                continue
            try:
                doc_id, n_chunks = _write_one_paper_to_sql(
                    indexer=indexer, record=record, markdown_path=md_path, db=db,
                )
                indexed += 1
                total_chunks += n_chunks
                batch_count += 1
                if batch_count >= args.commit_batch:
                    db.commit()
                    batch_count = 0
            except Exception as e:
                failed += 1
                failures.append({"paper_id": record.paper_id, "err": f"{type(e).__name__}: {e}"})
                db.rollback()
                continue

            if processed % max(args.checkpoint_interval, 1) == 0 or processed == len(todo):
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-3)
                eta_min = (len(todo) - processed) / max(rate, 1e-3) / 60
                print(
                    f"[ckpt] {processed}/{len(todo)} "
                    f"indexed={indexed} failed={failed} miss_md={skipped_missing_md} "
                    f"chunks={total_chunks} rate={rate:.2f} pap/s eta={eta_min:.1f}min",
                    flush=True,
                )
                report = {
                    "processed": processed, "indexed": indexed, "failed": failed,
                    "skipped_missing_md": skipped_missing_md, "total_chunks": total_chunks,
                    "elapsed_s": elapsed, "rate_pap_per_sec": rate, "failures_tail": failures[-20:],
                }
                args.report_path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(args.report_path, report)
        if batch_count:
            db.commit()

    elapsed = time.time() - t0
    final = {
        "processed": processed, "indexed": indexed, "failed": failed,
        "skipped_missing_md": skipped_missing_md, "total_chunks": total_chunks,
        "elapsed_s": elapsed, "failures_tail": failures[-50:],
    }
    write_json_atomic(args.report_path, final)
    return final


def main() -> int:
    args = build_argparser().parse_args()
    report = run(args)
    print(json.dumps({k: v for k, v in report.items() if k != "failures_tail"}, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
