"""Experiment: which chunk REPRESENTATION best supports model-centric retrieval of
OpenReview review opinions?

We harvested structured reviews (scripts/reviews/harvest_openreview.py). The open
question for ingest is the chunk unit. This compares 5 strategies on BGE-M3 dense
cosine, over a fixed model-centric query set, with three lenses:
  (1) diversity@10  — distinct papers in top-10 (does it spread or over-concentrate?)
  (2) topic precision@10 — for topic queries with a keyword relevance predicate
  (3) qualitative top-k snippets — does the hit actually NAME/ground the model?

No production write; pure in-memory numpy cosine over the harvested sample.

Strategies
  S1 whole_review     : per review = summary+strengths+weaknesses+questions (no title)
  S2 weaknesses_only  : per review = weaknesses text only (no title)
  S3 title_weakness   : per review = "Paper: <title>\nWeakness: <weaknesses>"  (context-injected)
  S4 per_point        : per enumerated weakness point, title-prefixed (finest grain)
  S5 paper_digest     : per paper = title + all reviewers' weaknesses + meta-review
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path
from core.indexing.embedders import get_embedder

QUERIES = [
    ("eval_baselines",  "weaknesses in the experimental evaluation and missing baseline comparisons", None),
    ("theory_assump",   "concerns that the theoretical assumptions are too strong or unrealistic", None),
    ("novelty",         "criticism of limited technical novelty compared to prior work", None),
    ("scalability",     "scalability and computational cost concerns of the proposed model", None),
    ("reproducibility", "reproducibility concerns and missing code or implementation details", None),
    # topic queries carry a relevance predicate (keywords in title+abstract)
    ("diffusion",       "concerns about diffusion model sample quality and generation",
                        ("diffusion", "denoising", "score-based", "generative")),
    ("graph_nn",        "graph neural network oversmoothing and message passing limitations",
                        ("graph neural", "gnn", "graph convolution", "oversmooth", "message passing", "node classification")),
    ("llm_reasoning",   "large language model reasoning and benchmark evaluation concerns",
                        ("language model", "llm", "reasoning", "in-context", "chain-of-thought", "prompt")),
]


def _clean(s):
    return (s or "").strip()


def split_points(text: str) -> list[str]:
    text = _clean(text)
    if not text:
        return []
    parts = re.split(r"(?:\n\s*(?:\d+[\.\)]|[-*•])\s+)|(?:\n\n+)", text)
    pts = [p.strip() for p in parts if len(p.strip()) >= 40]
    return pts or ([text] if len(text) >= 40 else [])


def build_chunks(records: list[dict]) -> dict[str, list[dict]]:
    """Return {strategy: [{text, paper_id, title}]}."""
    S = {k: [] for k in ("S1_whole", "S2_weak", "S3_title_weak", "S4_per_point", "S5_digest")}
    for r in records:
        title = _clean(r.get("title"))
        pid = r["paper_id"]
        weaks = []
        for rv in r.get("reviews", []):
            summ, stre = _clean(rv.get("summary")), _clean(rv.get("strengths"))
            weak, ques = _clean(rv.get("weaknesses")), _clean(rv.get("questions"))
            if weak:
                weaks.append(weak)
            whole = "\n".join(x for x in (
                f"Summary: {summ}" if summ else "",
                f"Strengths: {stre}" if stre else "",
                f"Weaknesses: {weak}" if weak else "",
                f"Questions: {ques}" if ques else "") if x)
            if whole:
                S["S1_whole"].append({"text": whole, "paper_id": pid, "title": title})
            if weak:
                S["S2_weak"].append({"text": weak, "paper_id": pid, "title": title})
                S["S3_title_weak"].append({"text": f"Paper: {title}\nWeakness: {weak}", "paper_id": pid, "title": title})
                for pt in split_points(weak):
                    S["S4_per_point"].append({"text": f"Paper: {title}\nWeakness: {pt}", "paper_id": pid, "title": title})
        if weaks:
            meta = _clean(r.get("meta_review"))
            digest = f"Paper: {title}\nReviewer concerns:\n" + "\n".join(f"- {w}" for w in weaks)
            if meta:
                digest += f"\nMeta-review: {meta}"
            S["S5_digest"].append({"text": digest, "paper_id": pid, "title": title})
    return S


def embed_all(embedder, texts: list[str], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        res = embedder.embed_texts(texts[i:i + batch])
        out.append(np.asarray(res.dense, dtype=np.float32))
    m = np.concatenate(out, axis=0)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return m


def relevant(rec_by_pid: dict, pid: str, kws) -> bool:
    r = rec_by_pid.get(pid, {})
    hay = (_clean(r.get("title")) + " " + _clean(r.get("abstract"))).lower()
    return any(k in hay for k in kws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--snippets", type=int, default=3)
    args = ap.parse_args()

    records = []
    for p in args.inputs:
        records += [json.loads(l) for l in Path(p).open(encoding="utf-8")]
    records = [r for r in records if r.get("n_reviews", 0) > 0]
    rec_by_pid = {r["paper_id"]: r for r in records}
    print(f"loaded {len(records)} papers with reviews from {args.inputs}")

    chunks = build_chunks(records)
    for k, v in chunks.items():
        print(f"  {k:16} chunks={len(v)}")

    embedder = get_embedder()
    print(f"embedder ready: {type(embedder).__name__} dim={embedder.dimension}")

    qmat = embed_all(embedder, [q[1] for q in QUERIES])

    # per strategy: embed, score every query, collect metrics
    results = {}  # strat -> {qkey -> [(score, idx)]}
    strat_mat = {}
    for strat, items in chunks.items():
        print(f"embedding {strat} ({len(items)} chunks)...")
        strat_mat[strat] = embed_all(embedder, [it["text"] for it in items])

    print("\n" + "=" * 100)
    print("PER-STRATEGY SUMMARY  (diversity@%d = distinct papers in top-%d; topicP@%d on 3 topic queries)" % (args.topk, args.topk, args.topk))
    print("=" * 100)
    print("%-16s %8s %10s %12s %12s" % ("strategy", "nchunks", "mean_top1", "diversity@%d" % args.topk, "topicP@%d" % args.topk))
    for strat, items in chunks.items():
        mat = strat_mat[strat]
        top1s, divs, tprec = [], [], []
        for qi, (qkey, qtext, kws) in enumerate(QUERIES):
            sims = qmat[qi] @ mat.T
            order = np.argsort(-sims)[:args.topk]
            top1s.append(float(sims[order[0]]))
            pids = [items[j]["paper_id"] for j in order]
            divs.append(len(set(pids)))
            if kws:
                hits = sum(1 for pid in pids if relevant(rec_by_pid, pid, kws))
                tprec.append(hits / len(pids))
            results.setdefault(strat, {})[qkey] = order
        print("%-16s %8d %10.3f %12.1f %12.2f" % (
            strat, len(items), float(np.mean(top1s)), float(np.mean(divs)),
            float(np.mean(tprec)) if tprec else float("nan")))

    # qualitative side-by-side for a few queries
    show_q = ["eval_baselines", "graph_nn", "novelty"]
    for qkey in show_q:
        qtext = next(q[1] for q in QUERIES if q[0] == qkey)
        print("\n" + "#" * 100)
        print(f"QUERY [{qkey}]: {qtext}")
        for strat in ("S2_weak", "S3_title_weak", "S4_per_point", "S5_digest"):
            items, mat = chunks[strat], strat_mat[strat]
            qi = next(i for i, q in enumerate(QUERIES) if q[0] == qkey)
            sims = qmat[qi] @ mat.T
            order = np.argsort(-sims)[:args.snippets]
            print(f"\n  --- {strat} ---")
            for j in order:
                it = items[j]
                snip = " ".join(it["text"].split())[:160]
                print(f"   {sims[j]:.3f} [{it['title'][:38]}] {snip}")


if __name__ == "__main__":
    main()
