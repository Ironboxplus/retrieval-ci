"""Scoped manifest pdf_path/markdown_path backfill for ONE (conf, year) from the
LOCAL filesystem (self-hosted / 4090 box — W106).

Why a new, slice-scoped tool instead of sync_manifest_pdf_md_paths.py:
    The full sync walks EVERY record and NULLs any pdf_path/markdown_path whose
    file is missing on THIS host — on the 4090 box that wipes ~89K records whose
    paths still point at Arc/Windows locations. This tool only ever touches
    records for the requested (conf, year) slice, so other venues are left
    exactly as they are.

Naming convention (matches ccf_a_main_corpus.download_single + extract, same as
sync_manifest_pdf_md_paths.py):
    pdf_path      = corpus_root/pdf/{conf.lower()}/{year}/{sanitize(paper_id)}.pdf
    markdown_path = corpus_root/markdown/{conf}/{year}/{sanitize(paper_id)}.md
                    (lowercase / original-case / uppercase dir all tried)

Purely additive within the slice: fills a missing/stale path when the local file
exists; leaves a record untouched when no local file is found (never NULLs other
venues, never NULLs this venue's already-good paths).

--corpus-root exists because the artifact root is not always the canonical one:
`corpus_root/pdf` is a symlink to an SMB share that can be unreachable, so the 2026
KDD/SIGIR download and marker runs deliberately wrote to data/papers/{pdf,markdown}
instead. With the root hardcoded this tool reports md_missing for every paper and
writes nothing, and preindex then selects 0 papers (it filters on markdown_path).

Usage:
    python scripts/sync/sync_local_conf_md.py --conf cvpr --year 2026
    python scripts/sync/sync_local_conf_md.py --conf cvpr --year 2026 \
        --manifest-path /tmp/paper_manifest.json
    python scripts/sync/sync_local_conf_md.py --conf kdd --year 2026 \
        --corpus-root data/papers --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.paper_crawler.ccf_a_main_corpus import sanitize_file_component  # type: ignore  # noqa: E402

CORPUS_ROOT = ROOT / "data" / "papers" / "ccf_a_main_conference_corpus"
DEFAULT_MANIFEST = CORPUS_ROOT / "manifests" / "paper_manifest.json"


def _candidate_pdf(corpus_root: Path, conf: str, year: int, paper_id: str) -> Path:
    return corpus_root / "pdf" / conf.lower() / str(year) / f"{sanitize_file_component(paper_id)}.pdf"


def _candidate_md_paths(corpus_root: Path, conf: str, year: int, paper_id: str) -> list[Path]:
    """Markdown layout has historically used both lowercase + original-case dirs."""
    fname = f"{sanitize_file_component(paper_id)}.md"
    md_root = corpus_root / "markdown"
    return [
        md_root / conf / str(year) / fname,
        md_root / conf.lower() / str(year) / fname,
        md_root / conf.upper() / str(year) / fname,
    ]


def _path_alive(s: str | None) -> bool:
    return bool(s) and Path(s).exists()


def _true_case(candidate: Path, corpus_root: Path) -> Path:
    """Re-spell an existing path with the casing the filesystem actually uses.

    Candidates are probed original-case first, and on Windows ``Path.exists()`` is
    case-insensitive — so a record whose ``conference`` is "KDD" happily matches a
    directory actually named ``kdd`` and we would record ".../KDD/...". That path
    resolves on Windows and MISSES on case-sensitive Linux, where preindex then
    selects 0 papers without raising anywhere. ``resolve()`` reports the real
    on-disk spelling; keep the caller's relative/absolute style either way.
    """
    try:
        resolved = candidate.resolve()
    except OSError:
        return candidate
    if candidate.is_absolute():
        return resolved
    try:
        return corpus_root / resolved.relative_to(corpus_root.resolve())
    except ValueError:
        # Outside corpus_root (symlink elsewhere) — an absolute path is still
        # correct, just not relative to the root the caller named.
        return resolved


def sync_slice(
    data: list[dict], conf: str, year: int, corpus_root: Path = CORPUS_ROOT
) -> dict:
    """Mutate `data` in place: fill pdf_path/markdown_path for the (conf, year)
    slice only, from the local filesystem. Returns a counts dict. Records OUTSIDE
    the slice are never read or written."""
    conf_lower = conf.lower()
    pdf_filled = md_filled = pdf_already = md_already = pdf_missing = md_missing = 0
    touched = 0

    for r in data:
        rconf = (r.get("conference") or "")
        if rconf.lower() != conf_lower or r.get("year") != year:
            continue
        pid = r.get("paper_id")
        if not pid:
            continue
        touched += 1

        cur_pdf = r.get("pdf_path")
        if _path_alive(cur_pdf):
            pdf_already += 1
        else:
            cand = _candidate_pdf(corpus_root, rconf, year, pid)
            if cand.exists():
                r["pdf_path"] = _true_case(cand, corpus_root).as_posix()
                pdf_filled += 1
            else:
                pdf_missing += 1  # leave as-is; do NOT NULL

        cur_md = r.get("markdown_path")
        if _path_alive(cur_md):
            md_already += 1
        else:
            md_hit = next(
                (c for c in _candidate_md_paths(corpus_root, rconf, year, pid) if c.exists()),
                None,
            )
            if md_hit:
                r["markdown_path"] = _true_case(md_hit, corpus_root).as_posix()
                md_filled += 1
            else:
                md_missing += 1  # leave as-is; do NOT NULL

    return {
        "slice_records": touched,
        "pdf_filled": pdf_filled, "pdf_already": pdf_already, "pdf_missing": pdf_missing,
        "md_filled": md_filled, "md_already": md_already, "md_missing": md_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", required=True, help="conference (e.g. cvpr)")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--corpus-root", default=str(CORPUS_ROOT),
        help="root holding pdf/ and markdown/ (default: the canonical corpus root)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be filled, write nothing",
    )
    args = parser.parse_args()

    mp = Path(args.manifest_path)
    corpus_root = Path(args.corpus_root)
    print(f"[{time.strftime('%H:%M:%S')}] loading {mp}")
    data = json.loads(mp.read_text(encoding="utf-8"))
    print(f"  {len(data):,} manifest entries; slice = {args.conf} {args.year}")
    print(f"  corpus_root = {corpus_root}")

    counts = sync_slice(data, args.conf, args.year, corpus_root=corpus_root)
    print(
        f"  slice_records={counts['slice_records']}  "
        f"pdf(filled={counts['pdf_filled']} already={counts['pdf_already']} missing={counts['pdf_missing']})  "
        f"md(filled={counts['md_filled']} already={counts['md_already']} missing={counts['md_missing']})"
    )

    if args.dry_run:
        print("[dry-run] manifest not written")
        return 0

    if not (counts["pdf_filled"] or counts["md_filled"]):
        print("[done] nothing to fill — slice already in sync")
        return 0

    bak = mp.with_suffix(f".json.bak.local.{int(time.time())}")
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.rename(bak)
    tmp.rename(mp)
    print(f"[execute] wrote {mp}  (backup: {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
