"""Probe OpenReview venues for PUBLIC review availability (unauthenticated).

We over-claimed "only ICLR + NeurIPS" earlier. This actually tests a curated set
of ML / model-centric venues: for each venue id it pulls a few submissions WITH
replies and checks whether public ``Official_Review`` notes come back without
login. Reports submissions sampled, how many carry public reviews, avg reviews,
avg weakness length — so we can decide which venues are worth harvesting.

Proxy-required (W111): reached only via 10808. Paced + backed-off to avoid 429.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.reviews.harvest_openreview import OpenReviewHarvester  # noqa: E402

# Curated candidates — id formats per OpenReview convention. One recent year each
# (a venue that exposes reviews one year almost always does so every year).
CANDIDATES = [
    "ICLR.cc/2025/Conference",
    "NeurIPS.cc/2024/Conference",
    "NeurIPS.cc/2024/Datasets_and_Benchmarks_Track",
    "ICML.cc/2024/Conference",
    "TMLR",
    "colmweb.org/COLM/2024/Conference",
    "robot-learning.org/CoRL/2024/Conference",
    "rl-conference.cc/RLC/2024/Conference",
    "logconference.io/LOG/2024/Conference",
    "aistats.org/AISTATS/2025/Conference",
    "MIDL.io/2024/Conference",
    "lifelong-ml.cc/CoLLAs/2024/Conference",
    "auai.org/UAI/2024/Conference",
    "ML_Reproducibility_Challenge/2023",
    "AAAI.org/2024/Conference",
    "KDD.org/2024/Conference",
    "thecvf.com/CVPR/2024/Conference",
    "aclweb.org/ACL/2024/Conference",
]


def probe(h: OpenReviewHarvester, venue: str, sample: int) -> dict:
    try:
        recs = h.harvest(venue, 0, limit=sample, use_batch=True)
    except Exception as e:  # noqa: BLE001
        return {"venue": venue, "error": f"{type(e).__name__}: {e}"}
    with_rev = [r for r in recs if r["n_reviews"] > 0]
    avg_rev = (sum(r["n_reviews"] for r in with_rev) / len(with_rev)) if with_rev else 0
    avg_weak = 0
    if with_rev:
        wl = [len(rv.get("weaknesses") or "") for r in with_rev for rv in r["reviews"]]
        avg_weak = sum(wl) / len(wl) if wl else 0
    return {"venue": venue, "sampled": len(recs), "with_public_reviews": len(with_rev),
            "avg_reviews": round(avg_rev, 1), "avg_weak_chars": round(avg_weak)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4, help="submissions to pull per venue")
    ap.add_argument("--pace", type=float, default=2.0, help="seconds between venues")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or "http://127.0.0.1:10808")
    ap.add_argument("--venues", nargs="*", default=CANDIDATES)
    args = ap.parse_args()

    h = OpenReviewHarvester(proxy=args.proxy, pace=0.5)
    print(f"{'venue':52} {'samp':>4} {'pubRev':>7} {'avgRev':>7} {'avgWeak':>8}")
    print("-" * 84)
    for v in args.venues:
        r = probe(h, v, args.sample)
        if "error" in r:
            print(f"{v:52} ERROR {r['error'][:40]}")
        else:
            flag = "  <-- PUBLIC" if r["with_public_reviews"] > 0 else ""
            print(f"{v:52} {r['sampled']:>4} {r['with_public_reviews']:>7} "
                  f"{r['avg_reviews']:>7} {r['avg_weak_chars']:>8}{flag}")
        time.sleep(args.pace)


if __name__ == "__main__":
    main()
