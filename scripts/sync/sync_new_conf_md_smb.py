"""Fill manifest.markdown_path for new conferences using SMB UNC paths.

Stage S.9 prep: marker produced MDs on SMB for EMNLP/KDD/SIGIR/SIGMOD.

Usage:
    python scripts/sync_new_conf_md_smb.py                        # dry-run all 4
    python scripts/sync_new_conf_md_smb.py --execute              # write back
    python scripts/sync_new_conf_md_smb.py --conf emnlp --execute # single conf
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.paper_crawler.ccf_a_main_corpus import sanitize_file_component  # type: ignore
from pipeline.paper_crawler.runtime_files import write_text_atomic

MANIFEST = REPO / "data/papers/ccf_a_main_conference_corpus/manifests/paper_manifest.json"
SMB_BASE = "//10.100.100.9/lsb/retrival/papers/markdown"
CONFERENCES = ["emnlp", "kdd", "sigir", "sigmod"]
YEARS = [2022, 2023, 2024, 2025]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--conf", nargs="*", default=CONFERENCES,
                        help="conferences to sync (default: all 4)")
    parser.add_argument("--year", type=int, nargs="*", default=YEARS,
                        help="years to sync (default: 2022-2025; pass 2026 for new venues)")
    args = parser.parse_args()
    confs = [c.lower() for c in args.conf]
    years = args.year

    print(f"[load] {MANIFEST}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"  {len(data):,} manifest entries")
    print(f"  conferences: {confs}")

    total_filled = 0
    total_already = 0
    total_missing = 0

    for conf in confs:
        smb_root = Path(f"{SMB_BASE}/{conf}")
        records = [r for r in data
                   if (r.get("conference") or "").lower() == conf]
        print(f"\n--- {conf.upper()} ({len(records):,} records) ---")

        smb_files: dict[tuple[int, str], Path] = {}
        for y in years:
            d = smb_root / str(y)
            if not d.exists():
                continue
            for f in d.glob("*.md"):
                smb_files[(y, f.stem)] = f
        print(f"  SMB MDs indexed: {len(smb_files):,}")

        already = filled = missing = 0
        for r in records:
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
                # Forward-slash UNC: Path("//host/share/..").exists() works on
                # Windows, but the backslash form Path(r"\\host\share\..") raises
                # WinError 3 (2026-05-31 audit) → preindex's md_path.exists()
                # would silently miss every file. Keep forward slashes.
                r["markdown_path"] = hit.as_posix()
                filled += 1
            else:
                missing += 1

        print(f"  already={already}  filled={filled}  missing={missing}")
        total_filled += filled
        total_already += already
        total_missing += missing

    print(f"\n[total] already={total_already}  filled={total_filled}  missing={total_missing}")

    if not total_filled:
        print("[done] manifest already in sync")
        return 0

    if not args.execute:
        print("[dry-run] use --execute to write back")
        return 0

    bak = MANIFEST.with_suffix(f".json.bak.s9.{int(time.time())}")
    shutil.copy2(MANIFEST, bak)
    write_text_atomic(MANIFEST, json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[execute] wrote {MANIFEST}  (backup: {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
