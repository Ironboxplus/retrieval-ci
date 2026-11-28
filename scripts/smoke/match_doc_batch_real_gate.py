"""Opt-in real HTTP gate for an isolated match-doc staging backend.

No mocks: the supplied URL must expose real BGE-M3, Qdrant and reranker.
Never point this at production for the 10/40 chunk cases without approval.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--paper-id", default="acl-2026-long-1")
    parser.add_argument("--chunks", type=int, choices=(1, 3, 10, 40), required=True)
    parser.add_argument("--output-dir", default=".release-artifacts/match-doc-batch-gate")
    args = parser.parse_args()
    body = {"input_type": "paper_id", "paper_id": args.paper_id, "top_k": 3,
            "max_query_chunks": args.chunks}
    started = time.perf_counter()
    response = httpx.post(args.api_url.rstrip("/") + "/api/match-doc", json=body,
                          timeout=60.0)
    elapsed = time.perf_counter() - started
    result = {"status_code": response.status_code, "wall_seconds": elapsed,
              "request": body, "body": response.json()}
    path = Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"candidate-mqc{args.chunks}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    response.raise_for_status()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
