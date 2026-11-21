"""arXiv CS corpus slices: convert downloaded HTML and write slice manifests.

    python scripts/collect/arxiv_cs_corpus.py slices                  # per-slice download state
    python scripts/collect/arxiv_cs_corpus.py convert --slice 202409  # HTML -> JSON + Markdown, then manifest
    python scripts/collect/arxiv_cs_corpus.py manifest --slice 202409 # manifest only
    python scripts/collect/arxiv_cs_corpus.py convert --ids-file ids.json --run-id 123  # one delta run

A slice is one arXiv submission month keyed YYYYMM; 202311 holds pre-2023-12
ids revised after the HTML cutoff. A delta run converts exactly the ids in
--ids-file (a JSON list, or an object whose "ids" is the list) and writes its
manifest/report under --run-id. Roots come from ARXIV_HARVEST_ROOT /
ARXIV_CORPUS_ROOT or default under settings.data_dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.paper_crawler.arxiv_corpus import (  # noqa: E402
    DELTA_KEY,
    SlicePaths,
    all_slices,
    convert_rows,
    convert_failures,
    convert_slice,
    corpus_root,
    harvest_root,
    iter_slice_counts,
    month_of,
    rows_by_ids,
    slice_label,
    write_manifest,
    write_slice_manifest,
)
from pipeline.paper_crawler.arxiv_html import HarvestStore  # noqa: E402


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["slices", "convert", "manifest"])
    parser.add_argument("--slice", type=int, default=None, help="month YYYYMM or batch YYYYMMPP")
    parser.add_argument("--ids-file", type=Path, help="delta: JSON list of arXiv ids (with --run-id)")
    parser.add_argument("--run-id", type=int, help="delta: names the run's manifest and report")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="convert: at most N papers (smoke)")
    parser.add_argument("--min-markdown-ratio", type=float, default=0.95,
                        help="convert: exit non-zero when markdown/downloaded falls below this")
    args = parser.parse_args()
    hroot, croot = harvest_root(), corpus_root()
    db = hroot / "arxiv_cs.sqlite3"
    if not db.exists():
        parser.error(f"harvest database not found: {db}")
    HarvestStore(db).close()  # adds any harvest column this code reads (convert_status ...)

    if args.command == "slices":
        for key, counts in iter_slice_counts(db):
            total = sum(counts.values())
            print(f"{key} {slice_label(key):>9}  total {total:6,}  " +
                  "  ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))
        return 0

    delta = args.ids_file is not None or args.run_id is not None
    if delta == (args.slice is not None):
        parser.error("give --slice, or --ids-file together with --run-id")
    if delta:
        if args.ids_file is None or args.run_id is None:
            parser.error("--ids-file and --run-id go together")
        ids = json.loads(args.ids_file.read_text(encoding="utf-8"))
        rows = rows_by_ids(db, ids["ids"] if isinstance(ids, dict) else ids, statuses=["ok"])
        paths = SlicePaths(croot, DELTA_KEY, args.run_id)
    else:
        if month_of(args.slice) not in all_slices(db):
            parser.error(f"--slice {args.slice} is not a known slice or batch")
    if args.command == "convert":
        if delta:
            report = convert_rows(rows, paths, hroot=hroot, workers=args.workers, limit=args.limit, log=_log)
            write_manifest(rows, paths, report["failures"], log=_log)
        else:
            report = convert_slice(args.slice, hroot=hroot, croot=croot, workers=args.workers,
                                   limit=args.limit, log=_log)
            write_slice_manifest(args.slice, hroot=hroot, croot=croot, log=_log)
        # Pages with no usable article (rejected) can never convert and are not held against the ratio.
        convertible, rejected = int(report["convertible"]), len(report["rejected"])
        ratio = int(report["markdown"]) / convertible if convertible else (1.0 if rejected else 0.0)
        if not args.limit and ratio < args.min_markdown_ratio:
            _log(f"markdown ratio {ratio:.3f} < {args.min_markdown_ratio} "
                 f"({rejected} rejected, {len(report['errors'])} errors)")
            return 2
        return 0
    if delta:
        write_manifest(rows, paths, convert_failures(paths), log=_log)
    else:
        write_slice_manifest(args.slice, hroot=hroot, croot=croot, log=_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
