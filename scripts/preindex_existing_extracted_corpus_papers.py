#!/usr/bin/env python
"""
Index already-extracted formal corpus papers without waiting for the full extract stage.

This script is intentionally manifest-read-only so it can run safely while the
formal extract stage is still checkpointing the same manifest file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper, build_default_corpus_root
from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig, PaperIndexer
from pipeline.paper_crawler.runtime_files import load_json_with_recovery, write_json_atomic


DEFAULT_MANIFEST_PATH = build_default_corpus_root() / "manifests" / "paper_manifest.json"
DEFAULT_REPORT_PATH = build_default_corpus_root() / "reports" / "preindex_existing_extracted_corpus_papers.json"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index already-extracted formal corpus papers without mutating the live manifest.",
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50, help="Papers per GPU batch.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for one run. 0 means no cap.")
    parser.add_argument(
        "--embedding-device",
        default="cuda",
        help="Explicit embedding device for indexing. Defaults to cuda.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the indexer to run on CPU when CUDA is unavailable.",
    )
    return parser


def load_manifest_records(manifest_path: Path) -> list[CorpusPaper]:
    payload = load_json_with_recovery(manifest_path, default_value=[])
    return [CorpusPaper.from_manifest_record(record) for record in payload]


def build_report(
    *,
    manifest_path: Path,
    requested_limit: int,
    processed: int,
    eligible: int,
    indexed: int,
    skipped_existing: int,
    skipped_missing_markdown: int,
    failed: int,
    recent_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "requested_limit": requested_limit,
        "processed": processed,
        "eligible": eligible,
        "indexed": indexed,
        "skipped_existing": skipped_existing,
        "skipped_missing_markdown": skipped_missing_markdown,
        "failed": failed,
        "recent_results": recent_results[-100:],
    }


async def run_preindex(
    *,
    manifest_path: Path,
    report_path: Path,
    checkpoint_interval: int,
    batch_size: int,
    limit: int,
    embedding_device: str,
    require_gpu: bool,
    heartbeat_callback: Any | None = None,
) -> dict[str, Any]:
    records = load_manifest_records(manifest_path)
    eligible_records = [
        record
        for record in records
        if record.markdown_path and not record.extract_error
    ]

    if limit > 0:
        eligible_records = eligible_records[:limit]

    indexer = PaperIndexer(
        PaperIndexConfig(
            collection_name="papers",
            data_source="system",
            embedding_device=embedding_device,
            require_gpu=require_gpu,
            max_concurrent_indexing=1,
        )
    )

    processed = 0
    indexed = 0
    skipped_existing = 0
    skipped_missing_markdown = 0
    failed = 0
    recent_results: list[dict[str, Any]] = []

    # Collect papers into batches, skipping already-indexed ones
    batch: list[CorpusPaper] = []

    for record in eligible_records:
        processed += 1
        markdown_path = Path(record.markdown_path)

        if not markdown_path.exists():
            skipped_missing_markdown += 1
            recent_results.append({"paper_id": record.paper_id, "status": "missing_markdown"})
        elif indexer.get_indexed_doc_id(record.paper_id) is not None:
            skipped_existing += 1
        else:
            batch.append(record)

        # Process batch when full or at end
        if len(batch) >= batch_size or (processed == len(eligible_records) and batch):
            try:
                result = await indexer.batch_index_papers(
                    papers=batch, skip_whoosh=True, per_batch_timeout_sec=300,
                )
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] Batch timed out at processed={processed}, skipping {len(batch)} papers")
                failed += len(batch)
                recent_results.extend(
                    [{"paper_id": getattr(p, "paper_id", "?"), "status": "timeout"} for p in batch]
                )
                result = None
            if result is not None:
                indexed += result["indexed"]
                failed += result["failed"]
                recent_results.extend(result["results"])
            batch = []

        # Heartbeat
        if heartbeat_callback and processed % max(batch_size, 1) == 0:
            heartbeat_callback(processed, len(eligible_records), indexed, failed)

        # Checkpoint
        if processed % max(checkpoint_interval, 1) == 0 or processed == len(eligible_records):
            write_json_atomic(
                report_path,
                build_report(
                    manifest_path=manifest_path,
                    requested_limit=limit,
                    processed=processed,
                    eligible=len(eligible_records),
                    indexed=indexed,
                    skipped_existing=skipped_existing,
                    skipped_missing_markdown=skipped_missing_markdown,
                    failed=failed,
                    recent_results=recent_results,
                ),
            )

    return build_report(
        manifest_path=manifest_path,
        requested_limit=limit,
        processed=processed,
        eligible=len(eligible_records),
        indexed=indexed,
        skipped_existing=skipped_existing,
        skipped_missing_markdown=skipped_missing_markdown,
        failed=failed,
        recent_results=recent_results,
    )


def _default_heartbeat(processed: int, total: int, indexed: int, failed: int) -> None:
    pct = processed / total * 100 if total else 0
    print(f"[HEARTBEAT] {processed}/{total} ({pct:.1f}%) indexed={indexed} failed={failed}", flush=True)


def main() -> int:
    args = build_argument_parser().parse_args()
    report = asyncio.run(
        run_preindex(
            manifest_path=args.manifest_path,
            report_path=args.report_path,
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size,
            limit=args.limit,
            embedding_device=args.embedding_device,
            require_gpu=not args.allow_cpu,
            heartbeat_callback=_default_heartbeat,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
