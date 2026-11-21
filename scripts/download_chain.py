"""Run a queue of manifest slices through download_subset.py sequentially.

Each slice is downloaded one-at-a-time (low per-host concurrency, no 429)
so the queue can run unattended after the currently-running downloads. The
underlying download_subset.py is resumable (it skips PDFs already on disk),
so re-running this chain is safe.

Usage:
    python scripts/download_chain.py
    python scripts/download_chain.py --conf "USENIX Security" --year 2023 --concurrency 3

With no slice args it runs the built-in QUEUE below.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "scripts" / "download_subset.py"

# (conference, year, concurrency) — conference must match manifest's
# `conference` field exactly. Kept low to avoid 429 / NIC saturation.
QUEUE: list[tuple[str, int, int]] = [
    ("USENIX Security", 2021, 3),
    ("USENIX Security", 2022, 3),
    ("USENIX Security", 2023, 3),
    ("USENIX Security", 2024, 3),
    ("USENIX Security", 2025, 3),
    ("AAAI", 2026, 2),
]


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_slice(conf: str, year: int, concurrency: int) -> int:
    cmd = [
        sys.executable, "-u", str(SUBSET),
        "--conf", conf, "--year", str(year),
        "--concurrency", str(concurrency),
    ]
    print(f"\n{'=' * 70}\n[{_ts()}] CHAIN START {conf} {year} (concurrency={concurrency})\n{'=' * 70}",
          flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    print(f"[{_ts()}] CHAIN END   {conf} {year} rc={rc} elapsed={elapsed:.0f}s", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", help="Single conference (overrides built-in QUEUE)")
    parser.add_argument("--year", type=int, help="Single year (with --conf)")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    if args.conf and args.year:
        queue = [(args.conf, args.year, args.concurrency)]
    else:
        queue = QUEUE

    print(f"[{_ts()}] download_chain starting: {len(queue)} slice(s)", flush=True)
    for conf, year, conc in queue:
        print(f"           queued: {conf} {year} (concurrency={conc})", flush=True)

    results: list[tuple[str, int]] = []
    for conf, year, conc in queue:
        rc = run_slice(conf, year, conc)
        results.append((f"{conf} {year}", rc))

    print(f"\n[{_ts()}] download_chain DONE", flush=True)
    for label, rc in results:
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"           {label}: {status}", flush=True)
    return 0 if all(rc == 0 for _, rc in results) else 1


if __name__ == "__main__":
    sys.exit(main())
