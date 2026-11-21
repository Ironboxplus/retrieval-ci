"""Sync paper_manifest.json pdf_path / markdown_path with on-disk reality.

Bug context (2026-05-16 23:50): manifest has 64,421 entries but only 13,608
have pdf_path filled, while the filesystem has ~57K PDFs actually present.
Cause: download phase wrote PDFs to disk but did not consistently persist
the `pdf_path` back to manifest (likely worker death mid-batch or older
download_papers code path). Same issue for `markdown_path`.

This tool is purely additive — it never deletes or overwrites a path that
already points to a live file. It only fills `pdf_path` / `markdown_path`
entries that are missing or stale (point to a path that no longer exists).

Naming convention (verified from ccf_a_main_corpus.download_single + extract):
    pdf_path      = corpus_root/pdf/{conf.lower()}/{year}/{sanitize(paper_id)}.pdf
    markdown_path = corpus_root/markdown/{conf}/{year}/{sanitize(paper_id)}.md

Usage:
    python scripts/sync_manifest_pdf_md_paths.py            # dry-run, report only
    python scripts/sync_manifest_pdf_md_paths.py --execute  # write back
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.paper_crawler.ccf_a_main_corpus import sanitize_file_component  # type: ignore

CORPUS_ROOT = ROOT / "data" / "papers" / "ccf_a_main_conference_corpus"
MANIFEST_PATH = CORPUS_ROOT / "manifests" / "paper_manifest.json"
PDF_ROOT = CORPUS_ROOT / "pdf"
MD_ROOT = CORPUS_ROOT / "markdown"


def _candidate_pdf(conf: str, year: int, paper_id: str) -> Path:
    return PDF_ROOT / conf.lower() / str(year) / f"{sanitize_file_component(paper_id)}.pdf"


def _candidate_md_paths(conf: str, year: int, paper_id: str) -> list[Path]:
    """Markdown layout has historically used both lowercase + original-case dirs."""
    fname = f"{sanitize_file_component(paper_id)}.md"
    return [
        MD_ROOT / conf / str(year) / fname,
        MD_ROOT / conf.lower() / str(year) / fname,
        MD_ROOT / conf.upper() / str(year) / fname,
    ]


def _path_alive(s: str | None) -> bool:
    return bool(s) and Path(s).exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--execute", action="store_true",
                        help="Write changes back. Default is dry-run.")
    args = parser.parse_args()

    mp = Path(args.manifest)
    print(f"[{time.strftime('%H:%M:%S')}] loading {mp}")
    data = json.loads(mp.read_text(encoding="utf-8"))
    print(f"  {len(data):,} manifest entries")

    pdf_filled = Counter()  # (conf, year) -> #newly filled
    md_filled = Counter()
    pdf_stale_cleared = Counter()
    md_stale_cleared = Counter()
    pdf_already = 0
    md_already = 0
    pdf_missing_on_fs = 0
    md_missing_on_fs = 0

    for r in data:
        conf = r.get("conference")
        year = r.get("year")
        pid = r.get("paper_id")
        if not (conf and year and pid):
            continue
        key = (conf, year)

        # ---------- pdf_path ----------
        cur_pdf = r.get("pdf_path")
        if _path_alive(cur_pdf):
            pdf_already += 1
        else:
            if cur_pdf:
                pdf_stale_cleared[key] += 1
            cand = _candidate_pdf(conf, year, pid)
            if cand.exists():
                r["pdf_path"] = str(cand)
                pdf_filled[key] += 1
            else:
                r["pdf_path"] = None  # normalize stale → None
                pdf_missing_on_fs += 1

        # ---------- markdown_path ----------
        cur_md = r.get("markdown_path")
        if _path_alive(cur_md):
            md_already += 1
        else:
            if cur_md:
                md_stale_cleared[key] += 1
            md_hit = next((c for c in _candidate_md_paths(conf, year, pid) if c.exists()), None)
            if md_hit:
                r["markdown_path"] = str(md_hit)
                md_filled[key] += 1
            else:
                r["markdown_path"] = None
                md_missing_on_fs += 1

    pdf_total_new = sum(pdf_filled.values())
    md_total_new = sum(md_filled.values())
    print()
    print(f"  pdf  already_good={pdf_already:,}  filled={pdf_total_new:,}  "
          f"missing_on_fs={pdf_missing_on_fs:,}  stale_cleared={sum(pdf_stale_cleared.values()):,}")
    print(f"  md   already_good={md_already:,}  filled={md_total_new:,}  "
          f"missing_on_fs={md_missing_on_fs:,}  stale_cleared={sum(md_stale_cleared.values()):,}")

    if pdf_filled or md_filled:
        print()
        print("  per (conf, year) newly filled:")
        keys = sorted(set(pdf_filled) | set(md_filled))
        print(f"    {'conf':<10}{'year':>6}{'pdf_new':>10}{'md_new':>10}")
        for k in keys:
            print(f"    {k[0]:<10}{k[1]:>6}{pdf_filled.get(k,0):>10}{md_filled.get(k,0):>10}")

    if args.execute and (pdf_total_new or md_total_new):
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        bak = mp.with_suffix(f".json.bak.{int(time.time())}")
        mp.rename(bak)
        tmp.rename(mp)
        print(f"\n[execute] wrote {mp}  (backup: {bak.name})")
    elif args.execute:
        print("\n[execute] no changes — manifest already in sync")
    else:
        print("\n[dry-run] use --execute to write back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
