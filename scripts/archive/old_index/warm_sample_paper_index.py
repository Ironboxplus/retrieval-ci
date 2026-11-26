#!/usr/bin/env python
"""
Warm a small real paper index from the formal CCF-A corpus manifest.

This script intentionally avoids mutating the formal corpus manifest. It reads
already downloaded and extracted records, picks a bounded representative sample,
indexes those papers into the live database and search stores, and writes a
machine-readable report for downstream API verification.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    CORPUS_NAME,
    CorpusPaper,
    DEFAULT_PROXY_URL,
    build_default_corpus_root,
)
from pipeline.paper_crawler.integration.index_integration import create_runtime_paper_indexer
from pipeline.paper_crawler.models.paper import Conference, Paper, PaperMetadata, PaperSource, PaperType


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warm a bounded real paper sample into the live search indexes.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=build_default_corpus_root() / "manifests" / "paper_manifest.json",
        help="Formal corpus manifest path.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=build_default_corpus_root() / "reports" / "sample_index_warmup.json",
        help="JSON report path describing the warmed sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=24,
        help="Maximum number of extracted papers to index for warmup.",
    )
    parser.add_argument(
        "--proxy",
        default=DEFAULT_PROXY_URL,
        help="Recorded for traceability only. Warmup indexing is local and does not fetch remote content.",
    )
    return parser


def load_manifest_records(manifest_path: Path) -> list[CorpusPaper]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [CorpusPaper.from_manifest_record(record) for record in payload]


def select_warmup_sample(papers: Iterable[CorpusPaper], limit: int) -> list[CorpusPaper]:
    eligible_by_bucket: dict[str, deque[CorpusPaper]] = {}
    for paper in papers:
        if not paper.pdf_path or not paper.markdown_path:
            continue
        if paper.download_error or paper.extract_error:
            continue
        bucket_key = f"{paper.conference}-{paper.year}"
        eligible_by_bucket.setdefault(bucket_key, deque()).append(paper)

    ordered_bucket_keys = sorted(eligible_by_bucket)
    selected: list[CorpusPaper] = []

    while len(selected) < limit:
        progressed = False
        for bucket_key in ordered_bucket_keys:
            bucket = eligible_by_bucket[bucket_key]
            if not bucket:
                continue
            selected.append(bucket.popleft())
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    return selected


def build_runtime_paper(record: CorpusPaper) -> Paper:
    conference_enum = getattr(Conference, record.conference.upper(), Conference.OTHER)
    paper_source = getattr(PaperSource, "OTHER")
    paper_type = getattr(PaperType, "UNKNOWN")

    return Paper(
        id=record.paper_id,
        title=record.title,
        abstract=record.abstract or "",
        authors=list(record.authors),
        year=record.year,
        conference=conference_enum,
        source=paper_source,
        paper_page_url=record.paper_page_url,
        pdf_url=record.downloaded_from_url or record.pdf_url,
        paper_type=paper_type,
        metadata=PaperMetadata(
            venue=record.conference,
            year=record.year,
            track="main_conference",
        ),
        pdf_path=record.pdf_path,
        markdown_path=record.markdown_path,
    )


async def warm_index_sample(sample: list[CorpusPaper]) -> dict[str, object]:
    indexer = create_runtime_paper_indexer()
    results: list[dict[str, object]] = []
    indexed = 0
    skipped = 0
    failed = 0

    for record in sample:
        existing = indexer.get_paper_by_id(record.paper_id)
        if existing:
            skipped += 1
            results.append(
                {
                    "paper_id": record.paper_id,
                    "doc_id": existing.get("doc_id"),
                    "title": record.title,
                    "conference": record.conference,
                    "year": record.year,
                    "status": "already_indexed",
                    "pdf_path": record.pdf_path,
                    "markdown_path": record.markdown_path,
                }
            )
            continue

        try:
            doc_id = await indexer.index_paper(
                paper=build_runtime_paper(record),
                markdown_path=Path(record.markdown_path),
            )
            indexed += 1
            results.append(
                {
                    "paper_id": record.paper_id,
                    "doc_id": doc_id,
                    "title": record.title,
                    "conference": record.conference,
                    "year": record.year,
                    "status": "indexed",
                    "pdf_path": record.pdf_path,
                    "markdown_path": record.markdown_path,
                }
            )
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "paper_id": record.paper_id,
                    "doc_id": None,
                    "title": record.title,
                    "conference": record.conference,
                    "year": record.year,
                    "status": "failed",
                    "error": str(exc),
                    "pdf_path": record.pdf_path,
                    "markdown_path": record.markdown_path,
                }
            )

    return {
        "corpus_name": CORPUS_NAME,
        "requested": len(sample),
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }


def main() -> int:
    args = build_argument_parser().parse_args()
    manifest_path: Path = args.manifest_path
    report_path: Path = args.report_path

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    papers = load_manifest_records(manifest_path)
    sample = select_warmup_sample(papers, limit=args.limit)
    if not sample:
        raise SystemExit("No extracted papers are available for warmup indexing.")

    report = asyncio.run(warm_index_sample(sample))
    report["manifest_path"] = str(manifest_path)
    report["report_path"] = str(report_path)
    report["proxy"] = args.proxy

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
