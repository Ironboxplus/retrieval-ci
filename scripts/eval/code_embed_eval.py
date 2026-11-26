"""Fair code-retrieval eval: BGE-M3 vs *stronger* code-specialized embedders, over
THIS repo's real code as the retrieval pool.

Fixes the earlier toy eval:
  - real corpus: every code chunk in the repo (astchunk), not 12 hand snippets
  - stronger opponents: nomic CodeRankEmbed + Salesforce SFR-Embedding-Code-400M
  - fair usage: each model gets its documented query instruction/prefix
  - curated NL->symbol queries; a hit = top result whose code contains the target symbol

    python scripts/eval/code_embed_eval.py
    python scripts/eval/code_embed_eval.py --max-chunks 2000

Metrics: MRR@10, Recall@1, Recall@3.

Result (2026-06-16, 2008 real repo chunks, 18 NL->symbol queries):
    BAAI/bge-m3                            MRR@10=0.684  R@1=0.556  R@3=0.778
    nomic-ai/CodeRankEmbed (+prefix)       MRR@10=0.689  R@1=0.611  R@3=0.667
    Salesforce/SFR-Embedding-Code-400M_R   MRR@10=0.633  R@1=0.500  R@3=0.722

Conclusion: BGE-M3 is statistically TIED with a top code-specialized model
(CodeRankEmbed leads MRR by 0.005 — noise at n=18; BGE-M3 wins R@3; the dedicated
SFR-400M is weakest). A code embedder gives no meaningful gain on NL code-search
here, so the code KB reuses the shared BGE-M3 rather than running a second model.
NOTE: 18 queries is small and these are NL->symbol queries; a code->code or
identifier-query workload (other CoIR task types) could shift the ranking.
"""
from __future__ import annotations

import argparse

from core.code_kb.chunker import iter_repo_chunks

# Each model + how to format a QUERY (documents are always raw). Empty prefix = raw.
MODELS = {
    "BAAI/bge-m3": {"trust": False, "q_prefix": ""},
    "nomic-ai/CodeRankEmbed": {
        "trust": True,
        "q_prefix": "Represent this query for searching relevant code: ",
    },
    "Salesforce/SFR-Embedding-Code-400M_R": {
        "trust": True,
        "q_prefix": "Instruct: Given a textual query, retrieve relevant code\nQuery: ",
    },
}

# (natural-language query, target symbol that should appear in the top hit's code)
QUERIES = [
    ("retry a failed vector store write with exponential backoff", "_upsert_batch"),
    ("fuse dense and sparse retrieval with reciprocal rank fusion", "hybrid_search"),
    ("generate a deterministic id for a chunk from a content hash", "generate_chunk_id"),
    ("detect a near-duplicate playbook bullet and reinforce it", "add_bullet"),
    ("cosine similarity between two vectors", "_cosine"),
    ("the MCP tool that searches academic papers by query", "search_papers"),
    ("re-score candidates with a cross-encoder reranker", "rerank"),
    ("build a qdrant payload filter from a dict of fields", "_build_filter"),
    ("embed chunks in batches and free GPU memory between them", "embed_chunks"),
    ("serve the single page app index.html as a fallback route", "serve_spa"),
    ("count papers and chunks for the stats endpoint", "_get_stats_sync"),
    ("chunk source code at AST boundaries", "chunk_code"),
    ("walk a repository and yield code chunks for every file", "iter_repo_chunks"),
    ("format search results compactly with title and score", "format_search_results"),
    ("build a code index and upsert embeddings into qdrant", "def build"),
    ("get the process-wide shared embedder singleton", "get_embedder"),
    ("match a document against papers by per-chunk coverage", "match"),
    ("ensure a qdrant collection exists with the right vectors", "_ensure"),
]


def evaluate(model_name, cfg, docs, codes, queries):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    print(f"\n=== loading {model_name} ===", flush=True)
    m = SentenceTransformer(model_name, trust_remote_code=cfg["trust"])
    D = np.asarray(m.encode(docs, normalize_embeddings=True, batch_size=32,
                            show_progress_bar=False))
    q_texts = [cfg["q_prefix"] + q for q, _ in queries]
    Q = np.asarray(m.encode(q_texts, normalize_embeddings=True, batch_size=32,
                            show_progress_bar=False))
    mrr = r1 = r3 = 0.0
    n_scored = 0
    for i, (q, target) in enumerate(queries):
        if not any(target in c for c in codes):
            continue  # target not in corpus -> skip (don't penalize)
        n_scored += 1
        sims = D @ Q[i]
        order = list(np.argsort(-sims))
        rank = next((p + 1 for p, idx in enumerate(order) if target in codes[idx]), 999)
        mrr += 1.0 / rank if rank <= 10 else 0.0
        r1 += 1.0 if rank == 1 else 0.0
        r3 += 1.0 if rank <= 3 else 0.0
    n = n_scored or 1
    return {"MRR@10": round(mrr / n, 4), "Recall@1": round(r1 / n, 4),
            "Recall@3": round(r3 / n, 4), "n_queries": n_scored}


def _retired() -> None:
    """RETIRED: this script loads a model into its own process; every model is a vLLM service now."""
    from core.inference_retired import EMBEDDER_INSTEAD, RetiredInferencePath

    raise RetiredInferencePath('code_embed_eval.py (loads embedders into this process with sentence-transformers)', EMBEDDER_INSTEAD)


def main():
    _retired()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--max-chunks", type=int, default=3000)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()

    chunks = list(iter_repo_chunks(a.repo))[: a.max_chunks]
    docs = [c.content for c in chunks]
    codes = [c.code for c in chunks]
    print(f"corpus: {len(docs)} code chunks from {a.repo}")
    covered = sum(1 for _, t in QUERIES if any(t in c for c in codes))
    print(f"queries: {len(QUERIES)} ({covered} targets present in corpus)")

    results = {}
    for name in a.models:
        cfg = MODELS.get(name, {"trust": True, "q_prefix": ""})
        try:
            results[name] = evaluate(name, cfg, docs, codes, QUERIES)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    print(f"\n{'='*78}\nFAIR CODE-RETRIEVAL EVAL over {len(docs)} real repo chunks\n{'='*78}")
    for name, r in results.items():
        print(f"{name:42s} {r}")


if __name__ == "__main__":
    main()
