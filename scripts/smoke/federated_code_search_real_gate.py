"""Real gate for POST /api/code/federated-search (docs/federated_code_search_plan.md A5).

Read-only. Talks to a real deployed compute API — real Qdrant, real BGE-M3, real
cross-encoder — and checks the federated route against the two single-plane routes
it is built from. Nothing here is a stand-in: if the fusion, scoping or exclusion is
wrong, the result sets diverge from what the existing routes return.

Run on the compute host (loopback internal port), acting as a real Code KB owner:

    python scripts/smoke/federated_code_search_real_gate.py \
        --user-id <owner> --project <scopeA> --project <scopeB> \
        --public-source pion-webrtc

Exit code 0 only when every check passes. Latency is printed, not asserted, except
for the p95 budget in check 6.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import httpx

QUERY = "retry with exponential backoff on a failed network request"


def _key_private(hit: dict) -> tuple:
    return (hit.get("project"), hit["path"], hit["start_line"], hit["end_line"])


def _key_public(hit: dict) -> tuple:
    return (hit["source_id"], hit["revision"], hit["path"], hit["start_line"], hit["end_line"])


class Gate:
    def __init__(self, base: str, user_id: str) -> None:
        self.client = httpx.Client(base_url=base, timeout=180.0, trust_env=False)
        self.user_headers = {"X-Retrival-User-Id": user_id, "X-Retrival-User-Tier": "admin"}
        self.failures: list[str] = []

    def post(self, path: str, body: dict) -> dict:
        response = self.client.post(path, json=body, headers=self.user_headers)
        if response.status_code != 200:
            raise RuntimeError(f"{path} -> HTTP {response.status_code}: {response.text[:500]}")
        return response.json()

    def get(self, path: str) -> object:
        response = self.client.get(path, headers=self.user_headers)
        response.raise_for_status()
        return response.json()

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--project", action="append", required=True,
                        help="a completed, indexed private scope of --user-id; pass two")
    parser.add_argument("--public-source", default="pion-webrtc")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--latency-runs", type=int, default=20)
    args = parser.parse_args()
    if len(args.project) < 2:
        parser.error("pass at least two --project scopes")

    gate = Gate(args.base_url, args.user_id)
    top_k = args.top_k
    scope_a, scope_b = args.project[0], args.project[1]

    # 1. Private-only, one scope == the existing single-scope route.
    fed = gate.post("/api/code/federated-search",
                    {"query": QUERY, "top_k": top_k, "projects": [scope_a]})
    old = gate.post("/api/code/search", {"query": QUERY, "top_k": top_k, "project": scope_a})
    fed_keys = {_key_private(h) for h in fed["hits"]}
    old_keys = {_key_private(h) for h in old["hits"]}
    gate.check("1 single scope matches /api/code/search", fed_keys == old_keys and bool(fed_keys),
               f"federated={len(fed_keys)} old={len(old_keys)} "
               f"only_fed={len(fed_keys - old_keys)} only_old={len(old_keys - fed_keys)}")
    gate.check("1b no public leg by default",
               all(h["origin"] == "private" for h in fed["hits"]) and not fed["public"]["included"])

    # 2. Two scopes: every hit belongs to one of them, and both contribute.
    fed2 = gate.post("/api/code/federated-search",
                     {"query": QUERY, "top_k": 50, "projects": [scope_a, scope_b]})
    projects_seen = {h.get("project") for h in fed2["hits"]}
    gate.check("2 multi-scope hits stay inside the requested scopes",
               bool(projects_seen) and projects_seen <= {scope_a, scope_b},
               f"seen={sorted(p or '' for p in projects_seen)}")
    gate.check("2b both requested scopes contribute", {scope_a, scope_b} <= projects_seen,
               f"seen={sorted(p or '' for p in projects_seen)}")

    # 3. Empty projects == owner-wide, but only compare when nothing is excluded.
    fed_all = gate.post("/api/code/federated-search", {"query": QUERY, "top_k": top_k})
    excluded = fed_all["private"].get("excluded_scopes") or []
    if excluded:
        print(f"[SKIP] 3 owner-wide equality — {len(excluded)} scope(s) currently excluded: {excluded}")
    else:
        old_all = gate.post("/api/code/search", {"query": QUERY, "top_k": top_k})
        a = {_key_private(h) for h in fed_all["hits"]}
        b = {_key_private(h) for h in old_all["hits"]}
        gate.check("3 empty projects matches owner-wide /api/code/search", a == b and bool(a),
                   f"federated={len(a)} old={len(b)}")

    # 4. Public leg == /api/public-code/search, and pinned to the active revision.
    fed_pub = gate.post("/api/code/federated-search", {
        "query": QUERY, "top_k": top_k, "projects": [scope_a],
        "include_public": True, "source_ids": [args.public_source],
    })
    pub = gate.post("/api/public-code/search",
                    {"query": QUERY, "top_k": top_k, "source_ids": [args.public_source]})
    fed_pub_hits = [h for h in fed_pub["hits"] if h["origin"] == "public"]
    origins = {h["origin"] for h in fed_pub["hits"]}
    gate.check("4 both origins present when include_public", origins == {"private", "public"},
               f"origins={sorted(origins)}")
    # Fusion truncates to top_k across both legs, so compare the public hits the
    # federated route returned against the public route's own pool for that size.
    pub_keys = {_key_public(h) for h in pub["hits"]}
    fed_pub_keys = {_key_public(h) for h in fed_pub_hits}
    gate.check("4b every federated public hit is a real public-route hit",
               bool(fed_pub_keys) and fed_pub_keys <= pub_keys,
               f"federated_public={len(fed_pub_keys)} not_in_public_route={len(fed_pub_keys - pub_keys)}")
    sources = gate.get("/api/public-code/sources")
    sources = sources.get("sources", sources) if isinstance(sources, dict) else sources
    active = {s["source_id"]: s.get("active_revision") for s in sources}
    stale = [h for h in fed_pub_hits if active.get(h["source_id"]) != h["revision"]]
    gate.check("4c public hits are on the active revision", not stale, f"stale={len(stale)}")

    # 5. Real cross-encoder.
    fed_rr = gate.post("/api/code/federated-search", {
        "query": QUERY, "top_k": top_k, "projects": [scope_a],
        "include_public": True, "source_ids": [args.public_source], "rerank": True,
    })
    scores = [h.get("reranker_score") for h in fed_rr["hits"]]
    gate.check("5 rerank applied with finite reranker scores",
               fed_rr["rerank_applied"] and fed_rr["fusion"] == "rerank"
               and all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores),
               f"rerank_applied={fed_rr['rerank_applied']} scores={len(scores)}")

    # 6. Latency: serial, warmed. Budget = 1.2 x (p95 private + p95 public).
    def timed(path: str, body: dict) -> list[float]:
        gate.post(path, body)  # warm
        samples = []
        for _ in range(args.latency_runs):
            started = time.perf_counter()
            gate.post(path, body)
            samples.append((time.perf_counter() - started) * 1000)
        return samples

    def p95(samples: list[float]) -> float:
        return statistics.quantiles(samples, n=20)[18]

    private_ms = timed("/api/code/search", {"query": QUERY, "top_k": top_k, "project": scope_a})
    public_ms = timed("/api/public-code/search",
                      {"query": QUERY, "top_k": top_k, "source_ids": [args.public_source]})
    fed_ms = timed("/api/code/federated-search", {
        "query": QUERY, "top_k": top_k, "projects": [scope_a],
        "include_public": True, "source_ids": [args.public_source],
    })
    for name, samples in (("private", private_ms), ("public", public_ms), ("federated", fed_ms)):
        print(f"  latency {name:<9} p50={statistics.median(samples):7.1f}ms p95={p95(samples):7.1f}ms")
    budget = 1.2 * (p95(private_ms) + p95(public_ms))
    gate.check("6 federated p95 within 1.2x of the two legs' p95 sum",
               p95(fed_ms) <= budget, f"p95={p95(fed_ms):.1f}ms budget={budget:.1f}ms")

    print()
    if gate.failures:
        print(f"GATE FAILED: {gate.failures}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
