"""Smoke test: run MarkerExtractor on N local PDFs and report timing.

Usage:
    python scripts/smoke_marker.py [N=1] [pdf_glob]

The first call warms the marker model cache (~30s); subsequent calls
reuse it, so per-PDF timing reflects steady-state cost.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

# Allow `python scripts/smoke_marker.py` to import from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    glob = sys.argv[2] if len(sys.argv) > 2 else "data/papers/pdf/iclr/2024/*.pdf"

    pdfs = sorted(Path().glob(glob))[:n]
    if not pdfs:
        print(f"no PDFs matched: {glob}")
        return 2

    from pipeline.paper_crawler.converters.extractors import MarkerExtractor

    extractor = MarkerExtractor()

    out_dir = Path("data/papers/markdown/_smoke_marker")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, pdf in enumerate(pdfs, start=1):
        t0 = perf_counter()
        result = extractor.extract(pdf)
        wall = perf_counter() - t0

        status = result.status.value
        md_len = len(result.markdown or "")
        section_hits = (result.markdown or "").count("\n## ")
        out_path = out_dir / f"{pdf.stem}.md"
        if result.markdown:
            out_path.write_text(result.markdown, encoding="utf-8")

        print(
            f"[{i}/{len(pdfs)}] {pdf.name}: status={status} wall={wall:.1f}s "
            f"md_len={md_len} sections={section_hits} → {out_path}"
        )
        if result.error:
            print(f"    err: {result.error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
