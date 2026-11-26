"""D4: prune redundant payload fields from Qdrant collection.

Removes payload keys that duplicate SQL Paper / Document tables (source of
truth). Qdrant payload should be the minimum set needed for retrieve-time
filter pushdown + rerank input + JOIN identity. Display fields belong in SQL.

Why: double-bookkeeping caused the D1/D2/D3 saga — old payload schema
('conference'/'author_names') diverged from canonical SQL schema
('venue'/'authors'). Minimal payload prevents future drift.

Keys deleted (redundant — fetch from SQL via doc_id/paper_id JOIN):
  title, abstract, authors, pdf_url, paper_page_url, source,
  markdown_path, keyword_terms, citation_count, citations,
  paper_type, track_type

Keys retained (minimal set):
  doc_id, paper_id, chunk_id       (identity / JOIN keys)
  source_type, level, data_source  (filter dispatch — required)
  venue, year, venue_tier          (C2 filter pushdown — required)
  content                           (rerank input)
  start_idx, end_idx, chunk_size   (chunk position, optional debug)

Usage:
  python scripts/d4_prune_payload.py --dry-run   # sample + show plan
  python scripts/d4_prune_payload.py             # execute against all points
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REDUNDANT_KEYS = [
    "title",
    "abstract",
    "authors",
    "pdf_url",
    "paper_page_url",
    "source",
    "markdown_path",
    "keyword_terms",
    "citation_count",
    "citations",
    "paper_type",
    "track_type",
]

KEEP_KEYS = {
    "doc_id",
    "paper_id",
    "chunk_id",
    "source_type",
    "level",
    "data_source",
    "venue",
    "year",
    "venue_tier",
    "content",
    "start_idx",
    "end_idx",
    "chunk_size",
}

NO_PROXY = {"http": "", "https": ""}


def sample_payload(base_url: str, collection: str, n: int = 3) -> list[dict]:
    r = requests.post(
        f"{base_url}/collections/{collection}/points/scroll",
        json={"limit": n, "with_payload": True, "with_vector": False},
        proxies=NO_PROXY,
        timeout=30,
    )
    r.raise_for_status()
    return [p["payload"] for p in r.json()["result"]["points"]]


def print_sample(label: str, samples: list[dict]) -> None:
    print(f"\n[{label}] sample (n={len(samples)}) payload classification:")
    for i, p in enumerate(samples):
        redundant_present = sorted(k for k in REDUNDANT_KEYS if k in p)
        kept_present = sorted(k for k in KEEP_KEYS if k in p)
        unknown = sorted(set(p.keys()) - set(REDUNDANT_KEYS) - KEEP_KEYS)
        print(f"  point {i} id ~ {p.get('chunk_id', '?')[:40]}")
        print(f"    redundant ({len(redundant_present)}): {redundant_present}")
        print(f"    kept      ({len(kept_present)}): {kept_present}")
        if unknown:
            print(f"    UNKNOWN   ({len(unknown)}): {unknown}  ← review whether to keep")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--collection", default="papers")
    p.add_argument("--dry-run", action="store_true",
                   help="Sample + print plan; no delete API call")
    p.add_argument("--timeout", type=int, default=1800,
                   help="HTTP timeout for delete call (seconds). Default 30 min.")
    args = p.parse_args()

    base = args.url.rstrip("/")

    # 1. inspect current state
    before = sample_payload(base, args.collection, n=3)
    print_sample("BEFORE", before)

    redundant_present_in_data: set[str] = set()
    for payload in before:
        redundant_present_in_data |= set(payload.keys()) & set(REDUNDANT_KEYS)

    if not redundant_present_in_data:
        print("\nNo redundant keys found in sample — nothing to prune. Exit.")
        return 0

    r_info = requests.get(f"{base}/collections/{args.collection}", proxies=NO_PROXY, timeout=10)
    r_info.raise_for_status()
    n_pts = r_info.json()["result"]["points_count"]
    print(f"\nWill delete {len(REDUNDANT_KEYS)} keys from {n_pts:,} points.")
    print(f"Keys: {REDUNDANT_KEYS}")

    if args.dry_run:
        print("\n[DRY-RUN] No API call made. Re-run without --dry-run to execute.")
        return 0

    # 2. submit async delete.
    # `must_not: []` does NOT match-all in Qdrant 1.18; use a trivially-true
    # condition on `level` (every point has level ∈ {1,2,3}) as the match-all
    # selector. Submit with wait=false — server-side wait timeout is 60s so a
    # collection-wide delete on 2.7M+ points would always return wait_timeout
    # even though the op is queued and will run.
    delete_body = {
        "keys": REDUNDANT_KEYS,
        "filter": {"must": [{"key": "level", "range": {"gte": 0}}]},
    }
    print(f"\nPOST /collections/{args.collection}/points/payload/delete?wait=false (async submit) ...")
    r = requests.post(
        f"{base}/collections/{args.collection}/points/payload/delete?wait=false",
        json=delete_body,
        proxies=NO_PROXY,
        timeout=30,
    )
    r.raise_for_status()
    result = r.json()
    op_id = result.get("result", {}).get("operation_id")
    print(f"  submitted: operation_id={op_id} status={result.get('result',{}).get('status')}")

    # 3. poll for completion — scroll the same 3 points every 20s until none
    # carry any redundant key. Max ~30 min wait (90 attempts × 20s).
    import time as _t
    print(f"\nPolling for completion (max ~30 min, sample-based check) ...")
    success = False
    for attempt in range(1, 91):
        _t.sleep(20)
        after = sample_payload(base, args.collection, n=3)
        leftover_per_point = [set(p.keys()) & set(REDUNDANT_KEYS) for p in after]
        n_with_redundant = sum(1 for l in leftover_per_point if l)
        n_sample = len(after)
        # also poll queue
        try:
            r_info = requests.get(f"{base}/collections/{args.collection}", proxies=NO_PROXY, timeout=5)
            queue = r_info.json()["result"]["update_queue"]["length"]
        except Exception:
            queue = "?"
        print(f"  t+{attempt*20}s: {n_with_redundant}/{n_sample} sampled points still carry redundant keys; qdrant_queue={queue}")
        if n_with_redundant == 0:
            success = True
            break

    after_final = sample_payload(base, args.collection, n=5)
    print_sample("AFTER (final, n=5)", after_final)

    if not success:
        print("\n[TIMEOUT] still has redundant keys after 30 min poll. Op may still be running; re-run script later to re-verify.")
        return 1
    print("\n[OK] all redundant keys removed from sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
