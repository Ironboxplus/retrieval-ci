"""One-shot: fill manifest.markdown_path for ICLR using SMB UNC paths.

Stage R.7 prep: V2 marker produced 8599 MDs on SMB
//10.100.100.9/lsb/retrival/papers/markdown/iclr/{year}/{sanitize(pid)}.md
but manifest only has 3756 (V1 leftovers) with markdown_path set.

This script:
  1. Backs up manifest
  2. For each ICLR record with missing/stale markdown_path, looks up the SMB
     candidate and fills it (preserves existing backslash-UNC convention)
  3. Atomic write
  4. Reports before/after counts

Usage:
    python scripts/sync_iclr_md_smb.py            # dry-run
    python scripts/sync_iclr_md_smb.py --execute  # write back + backup
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.paper_crawler.ccf_a_main_corpus import sanitize_file_component  # type: ignore

MANIFEST = REPO / "data/papers/ccf_a_main_conference_corpus/manifests/paper_manifest.json"
SMB_ROOT = Path("//10.100.100.9/lsb/retrival/papers/markdown/iclr")
YEARS = [2022, 2023, 2024, 2025]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    print(f"[load] {MANIFEST}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"  {len(data):,} manifest entries")
    iclr = [r for r in data if (r.get("conference") or "").lower() == "iclr"]
    print(f"  {len(iclr):,} ICLR records")

    # Index SMB MDs
    smb_files: dict[tuple[int, str], Path] = {}
    for y in YEARS:
        d = SMB_ROOT / str(y)
        if not d.exists():
            continue
        for f in d.glob("*.md"):
            smb_files[(y, f.stem)] = f
    print(f"[smb] indexed {len(smb_files):,} alive MDs across {YEARS}")

    already = 0
    filled = 0
    still_missing = 0
    year_filled = {y: 0 for y in YEARS}
    for r in iclr:
        y = r.get("year")
        pid = r.get("paper_id")
        if not (y and pid):
            continue
        stem = sanitize_file_component(pid)
        cur = r.get("markdown_path")
        if cur and Path(cur).exists():
            already += 1
            continue
        hit = smb_files.get((y, stem))
        if hit:
            # Preserve existing backslash UNC convention
            r["markdown_path"] = str(hit).replace("/", "\\")
            filled += 1
            year_filled[y] = year_filled.get(y, 0) + 1
        else:
            still_missing += 1

    total_after = already + filled
    print()
    print(f"  already_alive = {already:,}")
    print(f"  newly_filled  = {filled:,}")
    print(f"  still_missing = {still_missing:,}")
    print(f"  per-year filled: {year_filled}")
    print(f"  TOTAL ICLR with md after = {total_after:,} / {len(iclr):,}")

    if not filled:
        print("\n[exec] nothing to do; manifest already in sync")
        return 0

    if not args.execute:
        print("\n[dry-run] use --execute to write back")
        return 0

    bak = MANIFEST.with_suffix(f".json.bak.iclr-r7.{int(time.time())}")
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST.rename(bak)
    tmp.rename(MANIFEST)
    print(f"\n[execute] wrote {MANIFEST}  (backup: {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
