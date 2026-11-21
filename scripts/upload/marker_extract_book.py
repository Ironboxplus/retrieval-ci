"""Full-book marker extraction with force_ocr, one subprocess per page batch.

Three findings are baked in here; each cost a real incident.

1. `disable_ocr=True` destroys line-numbered books.
   `PML_Advanced_Topics.pdf` indexed into 3,783 chunks of which only 5 were
   readable prose. Its text layer is fine (PyMuPDF reads 2,357-4,382 chars/page)
   — it is a line-numbered edition, and marker's text-layer path feeds the margin
   column straight into the equation recogniser:

       $$\\frac{8}{9} \\qquad A \\sim P^{obs}(A) \\tag{38.1}$$
       $$\\underline{31} \\quad H \\leftarrow f_H(\\xi_1) \\tag{38.6}$$

   The numbers become \\frac / \\underline, equations are emitted twice, and the
   prose between them is dropped. Nothing raises. Measured on pages 1200-1215:
   8,991 bytes with disable_ocr vs 94,771 with force_ocr.

2. force_ocr over a whole book exhausts the host.
   marker renders pages and holds the batch in memory — ~41 MB peak per page. Run
   over all 1,308 pages on this 15 GB box it reached RSS 11.3 GB + 43.5 GB swap
   (of 52 GB), drove MemAvailable to 0, stalled in `folio_wait_bit_common` with
   the GPU at 0%, and slowed the production backend's API from ~1s to 14.9s.

3. `MarkerExtractor._converter` is cached at CLASS scope, so per-batch config is
   silently ignored after the first batch.
   The first fix for (2) looped over page ranges inside ONE process. Every batch
   after the first reused the converter built with `page_range` of batch 1, so an
   overnight run produced 27 byte-identical copies of pages 0-49 — `do-calculus`
   appeared 0 times in a book with a 40-page causality chapter. The per-batch byte
   counts in the log were all identical, which is the tell.

Hence: one SUBPROCESS per batch. A fresh process gets a fresh converter (correct
page_range) and the OS reclaims every page on exit (bounded memory), at the cost
of reloading models per batch. Each batch lands in its own `.part` file so an
interrupted run resumes. After merging, `--verify` asserts the parts are not all
identical — the exact check that would have caught (3) overnight instead of at
review time.

Usage:
    python scripts/upload/marker_extract_book.py --pdf FILE --out FILE.md
    python scripts/upload/marker_extract_book.py --pdf FILE --out FILE.md --batch 50
    # internal, one batch (spawned by the parent):
    python scripts/upload/marker_extract_book.py --pdf FILE --part-out P.md --page-range 0-49
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TQDM_DISABLE", "1")

# Below this much available RAM, do not start another batch. The failed run drove
# MemAvailable to 0 and took the whole box with it; stopping with N batches done
# and resumable is strictly better than thrashing into swap.
MIN_AVAILABLE_MB = 3500


def available_mb() -> int:
    """MemAvailable from /proc/meminfo — the kernel's own estimate, which already
    accounts for reclaimable cache (unlike `free`'s "free" column)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30  # non-Linux: don't block


def prose_score(text: str) -> dict:
    n = len(text) or 1
    alpha = len(re.sub(r"[^A-Za-z ]", "", text))
    return {
        "chars": len(text),
        "alpha_ratio": round(alpha / n, 3),
        "headers": len(re.findall(r"(?m)^#{1,6} ", text)),
        "sentence_ends": len(re.findall(r"[a-z]{3,}[.!?](?:\s|$)", text)),
    }


def extract_one_batch(pdf: Path, part_out: Path, page_range: str) -> int:
    """Child mode: extract ONE page range in this (fresh) process and exit.

    Must be a fresh process per batch — MarkerExtractor caches its PdfConverter at
    class scope, so a second call in the same process silently reuses the first
    call's page_range.
    """
    from pipeline.paper_crawler.converters.extractors import MarkerExtractor

    result = MarkerExtractor(
        extra_config={"force_ocr": True, "page_range": page_range}
    ).extract(pdf)
    if not result.succeeded or not (result.markdown or "").strip():
        print(f"[child FAIL] {page_range}: {result.error}", flush=True)
        return 1
    part_out.parent.mkdir(parents=True, exist_ok=True)
    part_out.write_text(result.markdown, encoding="utf-8")
    print(f"[child ok] {page_range} -> {len(result.markdown):,}B", flush=True)
    return 0


def _retired() -> None:
    """RETIRED: Marker loads torch / surya models into this process; PDF->Markdown is the registered PaddleOCR-VL worker now."""
    from core.inference_retired import EXTRACTOR_INSTEAD, RetiredInferencePath

    raise RetiredInferencePath("marker_extract_book.py (in-process torch / surya Marker)", EXTRACTOR_INSTEAD)


def main() -> int:
    _retired()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", help="merged markdown (parent mode)")
    ap.add_argument("--batch", type=int, default=50, help="pages per subprocess (default 50)")
    ap.add_argument("--parts-dir", default=None, help="default: <out>.parts/")
    ap.add_argument("--part-out", help="child mode: write this one batch here")
    ap.add_argument("--page-range", help="child mode: marker page_range, e.g. 0-49")
    args = ap.parse_args()

    pdf = Path(args.pdf)

    if args.part_out and args.page_range:
        return extract_one_batch(pdf, Path(args.part_out), args.page_range)

    if not args.out:
        ap.error("--out is required in parent mode")
    out = Path(args.out)
    if out.is_file() and out.stat().st_size > 0:
        print(f"[skip] {out} already present ({out.stat().st_size:,} bytes)", flush=True)
        return 0

    parts = Path(args.parts_dir) if args.parts_dir else out.with_suffix(out.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)

    import fitz
    with fitz.open(pdf) as d:
        pages = d.page_count

    batches = [(s, min(s + args.batch, pages)) for s in range(0, pages, args.batch)]
    print(
        f"[start] {pdf.name} pages={pages} batch={args.batch} "
        f"batches={len(batches)} avail={available_mb()}MB",
        flush=True,
    )

    t_all = time.perf_counter()
    for idx, (s, e) in enumerate(batches, 1):
        part = parts / f"{s:05d}-{e:05d}.md"
        if part.is_file() and part.stat().st_size > 0:
            print(f"[{idx}/{len(batches)}] pages {s}-{e} cached", flush=True)
            continue

        avail = available_mb()
        if avail < MIN_AVAILABLE_MB:
            print(
                f"[ABORT] only {avail}MB available (< {MIN_AVAILABLE_MB}MB) before "
                f"pages {s}-{e}; {idx - 1}/{len(batches)} batches cached — re-run "
                f"to resume once the host has memory.",
                flush=True,
            )
            return 2

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--pdf", str(pdf), "--part-out", str(part), "--page-range", f"{s}-{e - 1}"],
            cwd=str(ROOT),
        )
        dt = time.perf_counter() - t0
        if proc.returncode != 0 or not part.is_file() or part.stat().st_size == 0:
            print(f"[FAIL] pages {s}-{e} rc={proc.returncode} after {dt:.0f}s", flush=True)
            return 1
        print(
            f"[{idx}/{len(batches)}] pages {s}-{e} {dt:.0f}s "
            f"({dt / (e - s):.1f}s/pg) {part.stat().st_size:,}B avail={avail}MB",
            flush=True,
        )

    part_files = sorted(parts.glob("*.md"))
    digests = {hashlib.md5(p.read_bytes()).hexdigest() for p in part_files}
    if len(part_files) > 1 and len(digests) == 1:
        print(
            f"[FAIL] all {len(part_files)} parts are byte-identical — page_range was "
            f"not applied per batch (see finding 3 in this file's docstring). "
            f"Not writing {out}.",
            flush=True,
        )
        return 3

    merged = "\n\n".join(p.read_text(encoding="utf-8") for p in part_files)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(merged, encoding="utf-8")
    print(
        f"[done] {time.perf_counter() - t_all:.0f}s -> {out}\n"
        f"       parts={len(part_files)} distinct={len(digests)}\n"
        f"       {prose_score(merged)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
