"""Compare PDF extraction strategies on a PAGE RANGE before committing to a full run.

Why this exists: `PML_Advanced_Topics.pdf` (1,308 pages) was extracted by marker
into 3,783 chunks of which only **5** were readable prose — the rest were table
soup (`| 10 9 8 7 6 5 4 3 2 1 |`) and LaTeX garbage (`\\mathbf{j} \\mathbf{j} ...`).
The PDF's text layer is intact (PyMuPDF reads 2,357-4,382 chars/page), so the loss
happened in marker's layout/table/equation models, most likely triggered by this
edition's margin line-number column.

A full re-run costs hours of GPU. Probe ~15 pages first, read the actual output,
and only then spend the night on 2,168 pages.

Usage:
    python scripts/upload/probe_extract.py --pdf FILE --start 1200 --end 1215
    python scripts/upload/probe_extract.py --pdf FILE --start 1200 --end 1215 \
        --marker-config '{"disable_ocr": false}'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TQDM_DISABLE", "1")


def prose_score(text: str) -> dict:
    """Crude but decisive: how much of this looks like readable sentences?

    A page of real textbook prose has a high alpha ratio, few pipes, and few
    dollar signs. The failed extraction scored near zero on all three.
    """
    n = len(text) or 1
    alpha = len(re.sub(r"[^A-Za-z ]", "", text))
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    sentences = re.findall(r"[a-z]{3,}[.!?](?:\s|$)", text)
    return {
        "chars": len(text),
        "alpha_ratio": round(alpha / n, 3),
        "pipes": text.count("|"),
        "dollars": text.count("$"),
        "words_4plus": len(words),
        "sentence_ends": len(sentences),
    }


def run_pymupdf(pdf: Path, start: int, end: int) -> str:
    import fitz
    doc = fitz.open(pdf)
    out = []
    for p in range(start, min(end, doc.page_count)):
        out.append(doc[p].get_text())
    doc.close()
    return "\n\n".join(out)


def run_marker(pdf: Path, start: int, end: int, extra: dict) -> str:
    from pipeline.paper_crawler.converters.extractors import MarkerExtractor
    # marker takes an inclusive "start-end" page_range string, 0-indexed.
    cfg = {"page_range": f"{start}-{end - 1}"}
    cfg.update(extra)
    ext = MarkerExtractor(extra_config=cfg)
    result = ext.extract(pdf)
    if not result.succeeded:
        raise RuntimeError(f"marker failed: {result.error}")
    return result.markdown or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--start", type=int, required=True, help="0-indexed first page")
    ap.add_argument("--end", type=int, required=True, help="exclusive last page")
    ap.add_argument("--marker-config", default="{}", help="JSON merged into marker config")
    ap.add_argument("--out-dir", default="data/extract_probe")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pdf.stem}_{args.start}-{args.end}"

    print(f"=== probe {pdf.name} pages [{args.start}, {args.end}) ===")

    t0 = time.perf_counter()
    py = run_pymupdf(pdf, args.start, args.end)
    (out_dir / f"{stem}.pymupdf.md").write_text(py, encoding="utf-8")
    print(f"[pymupdf] {time.perf_counter() - t0:.1f}s  {prose_score(py)}")

    t0 = time.perf_counter()
    try:
        mk = run_marker(pdf, args.start, args.end, json.loads(args.marker_config))
        (out_dir / f"{stem}.marker.md").write_text(mk, encoding="utf-8")
        print(f"[marker ] {time.perf_counter() - t0:.1f}s  {prose_score(mk)}")
    except Exception as exc:
        print(f"[marker ] FAILED after {time.perf_counter() - t0:.1f}s: {exc}")
        return 1

    print(f"\nwrote both to {out_dir}/{stem}.*.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
