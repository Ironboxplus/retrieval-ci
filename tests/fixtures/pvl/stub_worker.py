"""A real, tiny PDF-to-Markdown worker that honours marker_worker.py's contract.

The local extract stage tests run it as an actual subprocess instead of replacing
``run_local``: it converts every PDF except ``--fail-stem`` (default
``cvpr-2026-c``), writes a quality report for the one it refuses, and exits 1, which
is exactly what pvl_worker.py does when a paper fails verification.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--fail-stem", default="cvpr-2026-c")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    if args.trace is not None:
        with args.trace.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "argv": sys.argv[1:], "rank": args.rank, "world_size": args.world_size,
                "gpu": args.gpu, "stub_env": os.environ.get("STUB_ENV"),
            }) + "\n")
    time.sleep(args.sleep)

    args.output.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.input.rglob("*.pdf") if args.recursive else args.input.glob("*.pdf"))
    failed = 0
    for pdf in pdfs[args.rank::args.world_size]:
        if pdf.stem == args.fail_stem:
            failed += 1
            report = args.output / "_quality" / f"{pdf.stem}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({
                "paper": pdf.stem, "status": "failed",
                "failure_reasons": ["page 3: fabrication (partial_ratio 41 < 85)"],
            }), encoding="utf-8")
            continue
        (args.output / f"{pdf.stem}.md").write_text(
            f"# {pdf.stem}\n\n" + "body text of the paper " * 20, encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
