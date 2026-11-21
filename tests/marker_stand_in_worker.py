"""A small PDF-to-Markdown worker with marker_worker.py's command-line contract.

``--input DIR --output DIR --gpu N [--mode M] [--disable-ocr] [--fail-stem STEM]``:
for each ``*.pdf`` in DIR it reads the PDF with pypdf and writes ``<stem>.md``
into the output directory (a heading, the page count, the PDF's sha256, and the
settings the worker was started with, so a test can see which arguments and
environment reached it). A PDF whose stem is given with ``--fail-stem`` is
skipped and the exit status is 1, like marker_worker.py after a failed PDF.
It stands in for the GPU marker so the host extract stage can run for real on a
machine without the marker models; the stage cannot tell the difference.

``--sidecars`` makes it behave like pvl_worker.py's output layout: a report
``_quality/<stem>.json`` for every paper it handled, and for a paper named with
``--refuse-stem`` a failed report, the kept pages ``_pages/<stem>.json.gz`` and no Markdown.
Kept pages found in the output directory (uploaded by the stage) are the model's earlier
reading: the stem is appended to the ``--seen-log`` file, and with ``--accept-kept`` the
checks pass this time, so the Markdown is written and the kept pages are removed, as
pvl_worker.py does. ``--run-log FILE`` gets one line per start with the stems it was handed. ``--stdin-log FILE``
gets "open" or "eof": whether its stdin was still open half a second after the start.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from pypdf import PdfReader


def _report(output: Path, stem: str, refused: bool) -> None:
    quality = output / "_quality"
    quality.mkdir(parents=True, exist_ok=True)
    body = {"paper": stem, "status": "failed" if refused else "ok",
            "failure_reasons": ["stand-in refusal"] if refused else []}
    (quality / f"{stem}.json").write_text(json.dumps(body), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--mode", default="balanced")
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--fail-stem", action="append", default=[])
    parser.add_argument("--sidecars", action="store_true")
    parser.add_argument("--refuse-stem", action="append", default=[])
    parser.add_argument("--accept-kept", action="store_true")
    parser.add_argument("--seen-log", type=Path, default=None)
    parser.add_argument("--run-log", type=Path, default=None)
    parser.add_argument("--stdin-log", type=Path, default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.stdin_log is not None:  # is the stdin this worker was started with still open a moment later?
        eof = threading.Event()
        threading.Thread(target=lambda: (sys.stdin.read(), eof.set()), daemon=True).start()
        time.sleep(0.5)
        args.stdin_log.write_text("eof" if eof.is_set() else "open", encoding="utf-8")
    if args.run_log is not None:  # one line per start: the stems it was handed
        with args.run_log.open("a", encoding="utf-8") as fh:
            fh.write(" ".join(sorted(p.stem for p in args.input.glob("*.pdf"))) + "\n")
    if args.sidecars:
        (args.output / "_quality").mkdir(exist_ok=True)
        (args.output / "_quality" / "_run.json").write_text("{}", encoding="utf-8")
    failed = 0
    for pdf in sorted(args.input.glob("*.pdf")):
        if pdf.stem in args.fail_stem:
            print(f"extract.fail pdf={pdf.name}", file=sys.stderr)
            failed += 1
            continue
        kept = args.output / "_pages" / f"{pdf.stem}.json.gz"
        refused = False
        if args.sidecars:
            had_kept = kept.is_file()
            if had_kept and args.seen_log is not None:
                with args.seen_log.open("a", encoding="utf-8") as fh:
                    fh.write(pdf.stem + "\n")
            refused = pdf.stem in args.refuse_stem and not (had_kept and args.accept_kept)
            _report(args.output, pdf.stem, refused)
            if refused:
                kept.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(kept, "wt", encoding="utf-8") as fh:
                    json.dump({"0": {"stand_in": pdf.stem}}, fh)
                print(f"refused {pdf.name}")
                continue
            kept.unlink(missing_ok=True)
        pages = len(PdfReader(str(pdf)).pages)
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        (args.output / f"{pdf.stem}.md").write_text(
            f"# {pdf.stem}\n\n"
            f"This document has {pages} page(s) and was converted by the stand-in worker. It is not "
            "marker output; it exists to exercise the host extract stage with a real conversion step.\n\n"
            f"- pdf_sha256: {digest}\n- gpu: {args.gpu}\n- mode: {args.mode}\n"
            f"- disable_ocr: {args.disable_ocr}\n- tag: {os.environ.get('MARKER_TEST_TAG', '')}\n",
            encoding="utf-8")
        print(f"ok {pdf.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
