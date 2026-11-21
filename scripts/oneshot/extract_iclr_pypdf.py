"""Extract ICLR PDFs → Markdown via pypdf, matching existing corpus layout.

Bypasses the ``--stage extract`` guard in ``ccf_a_main_corpus.py`` (which
refuses to run because build_summary.paper_count != current manifest
length). Reads the ICLR subset of the manifest directly, runs the same
pypdf-based extraction the existing 55K corpus was built with, writes
MDs to ``data/papers/ccf_a_main_conference_corpus/markdown/iclr/{year}/``,
and updates the manifest entries' ``markdown_path`` field in-place so the
downstream ``--stage index`` (or preindex) can pick them up.

Multi-processed via ``concurrent.futures.ProcessPoolExecutor`` because
``pypdf.PdfReader.extract_text`` releases the GIL for I/O but the parse
itself is CPU-bound.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader

from pipeline.paper_crawler.runtime_files import write_json_atomic, write_text_atomic

CORPUS_ROOT = Path("data/papers/ccf_a_main_conference_corpus")
MANIFEST_PATH = CORPUS_ROOT / "manifests" / "paper_manifest.json"
MARKDOWN_ROOT = CORPUS_ROOT / "markdown"

log = logging.getLogger("extract_iclr_pypdf")


def _sanitize(name: str) -> str:
    """Match ccf_a_main_corpus.sanitize_file_component (filename-safe)."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", ".", "+"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "untitled"


def _build_markdown(paper: dict, page_sections: list[str]) -> str:
    """Match the frontmatter + body shape produced by
    ``MainConferenceCorpusBuilder.build_markdown_document`` so the
    downstream chunker treats it identically to the existing 55K MDs."""
    fm_keys = (
        "paper_id", "title", "conference", "year",
        "source_name", "pdf_url", "downloaded_from_url",
        "paper_page_url", "selection_rule",
        "pdf_resolution_status", "pdf_resolution_note",
    )
    fm_lines = ["---"]
    for k in fm_keys:
        v = paper.get(k)
        if v is None:
            continue
        fm_lines.append(f"{k}: {v}")
    fm_lines.append("---")

    title = paper.get("title") or paper.get("paper_id") or ""
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        authors_str = ", ".join(authors)
    else:
        authors_str = str(authors)

    parts = [
        "\n".join(fm_lines),
        "",
        f"# {title}",
        "",
        f"Authors: {authors_str}" if authors_str else "",
        "",
        "\n\n".join(page_sections),
    ]
    return "\n".join(p for p in parts if p is not None) + "\n"


@dataclass
class _Job:
    paper_id: str
    year: int
    pdf_path: str
    markdown_path: str
    frontmatter: dict


def _extract_one(job: _Job) -> tuple[str, Optional[str], float]:
    """Return (paper_id, error_or_None, elapsed_s)."""
    t0 = perf_counter()
    try:
        pdf_path = Path(job.pdf_path)
        if not pdf_path.exists():
            return job.paper_id, f"PDF missing: {pdf_path}", perf_counter() - t0

        md_path = Path(job.markdown_path)
        if md_path.exists() and md_path.stat().st_size > 200:
            return job.paper_id, None, perf_counter() - t0  # skip (already there)

        reader = PdfReader(str(pdf_path))
        page_sections: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                page_sections.append(f"## Page {page_number}\n\n{page_text}")
        if not page_sections:
            return job.paper_id, "no extractable text", perf_counter() - t0

        md_text = _build_markdown(job.frontmatter, page_sections)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(md_path, md_text)
        return job.paper_id, None, perf_counter() - t0
    except Exception as exc:
        return job.paper_id, f"{type(exc).__name__}: {exc}", perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="*", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--update-manifest", action="store_true",
                        help="Write back markdown_path/extract_error into manifest after pass")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    iclr = [p for p in manifest
            if (p.get("conference") or "").upper() == "ICLR"
            and int(p.get("year", 0)) in set(args.years)
            and p.get("pdf_path")]
    log.info("manifest loaded: %d total, %d ICLR matching years %s",
             len(manifest), len(iclr), args.years)

    jobs: list[_Job] = []
    for p in iclr:
        md_path = MARKDOWN_ROOT / "iclr" / str(p["year"]) / f"{_sanitize(p['paper_id'])}.md"
        jobs.append(_Job(
            paper_id=p["paper_id"],
            year=int(p["year"]),
            pdf_path=p["pdf_path"],
            markdown_path=str(md_path),
            frontmatter=p,
        ))
    if args.limit:
        jobs = jobs[: args.limit]

    log.info("jobs prepared: %d (workers=%d, dry_run=%s)",
             len(jobs), args.workers, args.dry_run)
    if args.dry_run:
        for j in jobs[:5]:
            log.info("  sample: id=%s pdf=%s → md=%s", j.paper_id, j.pdf_path, j.markdown_path)
        return 0

    ok = fail = skipped = 0
    fail_log: list[tuple[str, str]] = []
    t_global = perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract_one, j): j for j in jobs}
        done_n = 0
        for fut in as_completed(futures):
            done_n += 1
            paper_id, err, dt = fut.result()
            if err is None:
                ok += 1
            elif err.startswith("PDF missing") or err == "no extractable text":
                fail += 1
                fail_log.append((paper_id, err))
            else:
                fail += 1
                fail_log.append((paper_id, err))
            if done_n % 100 == 0 or done_n == len(jobs):
                rate = done_n / max(perf_counter() - t_global, 1e-6)
                log.info("progress %d/%d ok=%d fail=%d rate=%.1f pdf/s last=%s wall=%.2fs",
                         done_n, len(jobs), ok, fail, rate, paper_id, dt)

    elapsed = perf_counter() - t_global
    log.info("done: ok=%d fail=%d total=%d elapsed=%.1fs rate=%.2f pdf/s",
             ok, fail, len(jobs), elapsed, len(jobs)/max(elapsed,1e-6))
    if fail_log[:20]:
        log.warning("first failures:")
        for pid, err in fail_log[:20]:
            log.warning("  %s — %s", pid, err)

    if args.update_manifest:
        log.info("updating manifest in place ...")
        results = {paper_id: err for paper_id, err in
                   ((p["paper_id"], None) for p in iclr)}
        for pid, err in fail_log:
            results[pid] = err
        for p in manifest:
            if (p.get("conference") or "").upper() != "ICLR":
                continue
            md_path = MARKDOWN_ROOT / "iclr" / str(p["year"]) / f"{_sanitize(p['paper_id'])}.md"
            if md_path.exists() and md_path.stat().st_size > 200:
                p["markdown_path"] = str(md_path.resolve())
                p["extract_error"] = None
            else:
                p["markdown_path"] = None
                p["extract_error"] = results.get(p["paper_id"], "not processed")
        write_json_atomic(MANIFEST_PATH, manifest)
        log.info("manifest updated: %s", MANIFEST_PATH)

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
